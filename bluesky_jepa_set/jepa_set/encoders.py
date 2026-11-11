"""Encoders: 1D CNN (port of the reference) and query-centric Transformer.

Both map the observation Dict to one embedding [B, out_dim].
No weight shape depends on the count of other aircraft.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .obs import split_obs, absent_mask, DIST_INDEX

N_ZONES = 5    # ObstacleObservation(n=5) in both competition environments
N_EDGES = 12   # SectorBoundaryObservation(n=12) in both competition environments


class EncoderBase(nn.Module):
    out_dim: int

    def forward(self, obs, masks=None):
        """obs: dict of tensors [B, n]. masks: optional dict ac/zone/edge of bool [B, n], True = absent.
        Without masks, an all-zero row is an absent object."""
        own, ac, zone, edge = split_obs(obs)
        m = {"ac": absent_mask(ac), "zone": absent_mask(zone), "edge": absent_mask(edge)}
        if masks:
            m.update(masks)
        return self.encode(own, ac, zone, edge, m)


class CnnEncoder(EncoderBase):
    """Port of the reference ObservationEncoder (Conv1d stack + state MLP + dense layer).

    Differences from the reference (it reads 120 lidar rays, 3 object types as channels, 4 stacked frames):
    - one Conv1d branch per object type, because each type has its own length and features;
    - kernel 3 with padding in place of kernel 4 with stride 2, because the axes are short (1 to 12);
    - the aircraft branch ends with a masked max over the axis, so the aircraft count is free;
    - no frame stack: the observation already holds the relative velocity.
    The Conv1d reads the object axis in its given order (aircraft nearest first, edge points in outline order).
    """

    def __init__(self, out_dim=64, ch=70, state_hidden=64):
        super().__init__()
        self.out_dim = out_dim

        def branch(n_feat, circular=False):
            mode = "circular" if circular else "zeros"
            return nn.Sequential(
                nn.Conv1d(n_feat, ch, 3, padding=1, padding_mode=mode), nn.ReLU(),
                nn.Conv1d(ch, ch // 2, 3, padding=1, padding_mode=mode), nn.ReLU())

        self.ac_cnn = branch(7)
        self.zone_cnn = branch(4)
        self.edge_cnn = branch(3, circular=True)   # the sector outline is a closed ring
        self.state_mlp = nn.Sequential(nn.Linear(5, state_hidden), nn.ReLU(),
                                       nn.Linear(state_hidden, state_hidden), nn.ReLU())
        n_flat = ch // 2 + (ch // 2) * N_ZONES + (ch // 2) * N_EDGES + state_hidden
        self.dense = nn.Sequential(nn.Linear(n_flat, out_dim), nn.ReLU(),
                                   nn.Linear(out_dim, out_dim), nn.ReLU())

    def encode(self, own, ac, zone, edge, m):
        b = own.shape[0]
        ac = ac.masked_fill(m["ac"].unsqueeze(-1), 0.0)
        zone = zone.masked_fill(m["zone"].unsqueeze(-1), 0.0)
        edge = edge.masked_fill(m["edge"].unsqueeze(-1), 0.0)
        if ac.shape[1] == 0:
            a = own.new_zeros(b, self.ac_cnn[2].out_channels)
        else:
            a = self.ac_cnn(ac.transpose(1, 2))                       # [B, C, Na]
            a = a.masked_fill(m["ac"].unsqueeze(1), float("-inf")).amax(dim=2)
            a = torch.where(torch.isinf(a), torch.zeros_like(a), a)   # no aircraft -> zeros
        z = self.zone_cnn(zone.transpose(1, 2)).flatten(1)
        e = self.edge_cnn(edge.transpose(1, 2)).flatten(1)
        s = self.state_mlp(own)
        return self.dense(torch.cat([a, z, e, s], dim=1))


class QueryBlock(nn.Module):
    """Cross-attention (the query reads the keys), then an MLP. Pre-norm, residual."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_m = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))

    def forward(self, q, keys, key_mask):
        a, _ = self.attn(self.norm_q(q), keys, keys, key_padding_mask=key_mask, need_weights=False)
        q = q + a
        return q + self.mlp(self.norm_m(q))


class SetTFEncoder(EncoderBase):
    """Query-centric Transformer (design option a).

    Query: one token from the own-aircraft and goal features.
    Keys: one token per other aircraft, no-fly zone and sector edge point, plus one learned null token.
    Type identity comes from one projection per type. There are no slot or position vectors.
    Each key gets a sine-cosine code of its distance (recovers the nearest-first order information).
    """

    def __init__(self, out_dim=64, d_model=64, n_heads=4, n_layers=2, n_freq=4):
        super().__init__()
        self.out_dim = out_dim
        self.register_buffer("freqs", math.pi * (2.0 ** torch.arange(n_freq)), persistent=False)
        self.own_proj = nn.Sequential(nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.proj = nn.ModuleDict({
            "ac": nn.Linear(7 + 2 * n_freq, d_model),
            "zone": nn.Linear(4 + 2 * n_freq, d_model),
            "edge": nn.Linear(3 + 2 * n_freq, d_model)})
        self.key_norm = nn.LayerNorm(d_model)
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.null_token, std=0.02)
        self.blocks = nn.ModuleList([QueryBlock(d_model, n_heads) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim), nn.GELU())

    def _tokens(self, name, x):
        d = x[..., DIST_INDEX[name]].unsqueeze(-1) * self.freqs
        return self.proj[name](torch.cat([x, d.sin(), d.cos()], dim=-1))

    def encode(self, own, ac, zone, edge, m):
        b = own.shape[0]
        keys = torch.cat([self._tokens("ac", ac), self._tokens("zone", zone), self._tokens("edge", edge)], dim=1)
        keys = torch.cat([self.key_norm(keys), self.null_token.expand(b, -1, -1)], dim=1)
        never = torch.zeros(b, 1, dtype=torch.bool, device=own.device)
        key_mask = torch.cat([m["ac"], m["zone"], m["edge"], never], dim=1)
        q = self.own_proj(own).unsqueeze(1)
        for blk in self.blocks:
            q = blk(q, keys, key_mask)
        return self.head(q.squeeze(1))


def build_encoder(kind, out_dim=64, **kw):
    if kind == "cnn":
        return CnnEncoder(out_dim=out_dim, **kw)
    if kind == "set_tf":
        return SetTFEncoder(out_dim=out_dim, **kw)
    raise ValueError(f"unknown encoder {kind!r} (use cnn or set_tf)")


def n_params(module):
    return sum(p.numel() for p in module.parameters())
