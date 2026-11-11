from __future__ import annotations

from typing import Mapping
import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# This schema is taken from docs/competition/GETTING_STARTED.md and
# core/observations.py supplied by the user. All observation values are
# numeric, normalized by the environment, and represented as entity tokens.
TOKEN_FEATURES = 8

def _as_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == 1 else x

def observation_to_tokens(obs: Mapping[str, torch.Tensor]):
    """Convert the competition Dict observation into [B, T, 8] tokens.

    Token order: ownship, waypoint, intruders (nearest first), obstacles
    (nearest first), sector perimeter points. A separate scalar sector flag
    is included in the ownship/global token.
    """
    airspeed = _as_batch(obs["airspeed"].float())
    inside = _as_batch(obs["inside_sector"].float())
    cos_d = _as_batch(obs["cos_drift"].float())
    sin_d = _as_batch(obs["sin_drift"].float())
    wdist = _as_batch(obs["waypoint_distance"].float())

    x = _as_batch(obs["x_r"].float())
    y = _as_batch(obs["y_r"].float())
    vx = _as_batch(obs["vx_r"].float())
    vy = _as_batch(obs["vy_r"].float())
    ct = _as_batch(obs["cos_track"].float())
    st = _as_batch(obs["sin_track"].float())
    idist = _as_batch(obs["intruder_distance"].float())

    od = _as_batch(obs["obstacle_distance"].float())
    oc = _as_batch(obs["obstacle_cos_bearing"].float())
    os = _as_batch(obs["obstacle_sin_bearing"].float())
    oran = _as_batch(obs["obstacle_radius"].float())

    sd = _as_batch(obs["sector_point_distance"].float())
    sc = _as_batch(obs["sector_point_cos_bearing"].float())
    ss = _as_batch(obs["sector_point_sin_bearing"].float())

    b = airspeed.shape[0]
    device = airspeed.device
    dtype = airspeed.dtype

    def zeros_like_batch(n):
        return torch.zeros((b, n), device=device, dtype=dtype)

    own = torch.stack([
        airspeed[:, 0], inside[:, 0], zeros_like_batch(1)[:, 0],
        zeros_like_batch(1)[:, 0], zeros_like_batch(1)[:, 0],
        zeros_like_batch(1)[:, 0], zeros_like_batch(1)[:, 0],
        zeros_like_batch(1)[:, 0],
    ], dim=-1).unsqueeze(1)

    waypoint = torch.stack([
        cos_d, sin_d, wdist,
        zeros_like_batch(cos_d.shape[1]), zeros_like_batch(cos_d.shape[1]),
        zeros_like_batch(cos_d.shape[1]), zeros_like_batch(cos_d.shape[1]),
        zeros_like_batch(cos_d.shape[1]),
    ], dim=-1)

    intruders = torch.stack([x, y, vx, vy, ct, st, idist,
                             zeros_like_batch(x.shape[1])], dim=-1)

    obstacles = torch.stack([od, oc, os, oran,
                             zeros_like_batch(od.shape[1]),
                             zeros_like_batch(od.shape[1]),
                             zeros_like_batch(od.shape[1]),
                             zeros_like_batch(od.shape[1])], dim=-1)

    sector = torch.stack([sd, sc, ss,
                          zeros_like_batch(sd.shape[1]),
                          zeros_like_batch(sd.shape[1]),
                          zeros_like_batch(sd.shape[1]),
                          zeros_like_batch(sd.shape[1]),
                          zeros_like_batch(sd.shape[1])], dim=-1)

    tokens = torch.cat([own, waypoint, intruders, obstacles, sector], dim=1)
    return tokens


def token_count_from_space(observation_space: spaces.Dict) -> int:
    return (
        1
        + observation_space["cos_drift"].shape[0]
        + observation_space["x_r"].shape[0]
        + observation_space["obstacle_distance"].shape[0]
        + observation_space["sector_point_distance"].shape[0]
    )


class StructuredViT(nn.Module):
    """Small Transformer over structured aircraft/entity tokens.

    This is not an image ViT: there are no image patches or ImageNet weights.
    It is a domain-specific Transformer for the competition's numeric Dict.
    """
    def __init__(self, n_tokens: int, features_dim: int = 128,
                 d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_tokens = n_tokens
        self.features_dim = features_dim
        self.input_proj = nn.Linear(TOKEN_FEATURES, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, features_dim), nn.GELU())
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, TOKEN_FEATURES)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor | None = None):
        b, t, _ = tokens.shape
        if t != self.n_tokens:
            raise ValueError(f"Expected {self.n_tokens} tokens, got {t}. Check observation schema.")
        z = self.input_proj(tokens)
        if mask is not None:
            z = torch.where(mask.unsqueeze(-1), self.mask_token.expand(b, t, -1), z)
        cls = self.cls_token.expand(b, -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos_embed
        return self.norm(self.transformer(z))

    def forward(self, obs):
        tokens = observation_to_tokens(obs)
        encoded = self.encode_tokens(tokens)
        return self.head(encoded[:, 0])

    def masked_reconstruction(self, tokens: torch.Tensor, mask_ratio: float = 0.35):
        b, t, _ = tokens.shape
        n_mask = max(1, int(t * mask_ratio))
        noise = torch.rand(b, t, device=tokens.device)
        mask = torch.zeros(b, t, dtype=torch.bool, device=tokens.device)
        mask.scatter_(1, noise.argsort(dim=1)[:, :n_mask], True)
        encoded = self.encode_tokens(tokens, mask=mask)
        prediction = self.reconstruction_head(encoded[:, 1:])
        return prediction, tokens, mask


class BlueSkyViTExtractor(BaseFeaturesExtractor):
    """SB3-compatible features extractor for the competition Dict space."""
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__(observation_space, features_dim)
        n_tokens = token_count_from_space(observation_space)
        self.encoder = StructuredViT(
            n_tokens=n_tokens, features_dim=features_dim,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, observations):
        return self.encoder(observations)
