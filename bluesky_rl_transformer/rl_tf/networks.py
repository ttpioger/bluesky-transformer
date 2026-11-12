"""Networks: Transformer encoder over lidar ray tokens, actor head, attention critic.

Layout of the reference method (avbiswas/navigation-mappo-rl): one observation encoder shared by the
actor and the critic; actor MLP (64, 64, tanh) with a tanh mean and a state-independent log_std;
centralized critic with one multi-head attention layer over the aircraft encodings.
The change: the Conv1d stack over the lidar scan is replaced by a Transformer over ray tokens.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Normal


class LidarTransformerEncoder(nn.Module):
    """Tokens: one per patch of `patch` adjacent rays (all frames and channels of these rays), plus one
    token from the own-aircraft state. Self-attention over all tokens. The output of the state token,
    joined with the state encoding, goes through the dense layer of the reference encoder.

    state [B, S] and lidar [B, C, R] -> features [B, features_dim].

    ac_tokens = K > 0 (2026-11-12): the state ends with K rows of AC_FEATURES values and K flags (Lidar.aircraft_tokens).
    Each row becomes one more token (a small MLP plus one learned "aircraft" vector, no position vector: the
    rows are a set). An empty row is masked: no token attends to it. K = 0 gives the encoder of before.

    zone_tokens = Z > 0 (2026-11-12): after the aircraft rows the state has Z restricted-area rows, Z flags and one
    goal row (Lidar.zone_tokens_vec). Each area row is one token (MLP plus one learned "area" vector, empty rows
    masked); the goal row is one token (MLP plus one learned "goal" vector).
    act_tokens = H > 0 (2026-11-12): the state then ends with the last H own actions (H rows of turn and speed, H
    flags; ObsBuilder). Each row is one token (MLP plus one learned vector per place in time, empty rows masked).
    edge_tokens = E > 0 (2026-11-12): the state then ends with E sector-edge rows and E flags (Lidar.edge_tokens_vec).
    Each row is one token (MLP plus one learned vector per direction bin, empty rows masked).
    ac_cpa = 1: an aircraft row has 2 more values (closest point of approach).
    fourier = F > 0: the input of the aircraft, area, goal and edge MLPs is the row plus sin and cos of each value
    at the frequencies pi * 2^j, j = 0..F-1 (the rows are small numbers; this separates near values).
    cls_token = 1: one learned read-out token at place 0; its output replaces the output of the own-state token.
    All options at 0 give the encoder of before, with the same start weights.
    """

    def __init__(self, state_dim, lidar_ch, n_rays, patch=4, d_model=64, n_heads=4, n_layers=2,
                 state_hidden=128, features_dim=256, ac_tokens=0, zone_tokens=0, act_tokens=0,
                 edge_tokens=0, ac_cpa=0, fourier=0, cls_token=0):
        super().__init__()
        from .lidar import (AC_FEATURES, ZONE_FEATURES, GOAL_FEATURES, ACT_FEATURES, EDGE_FEATURES, CPA_FEATURES,
                            extra_dim)
        self.k, self.ac_f = int(ac_tokens), AC_FEATURES + CPA_FEATURES * int(bool(ac_cpa))
        self.e, self.edge_f, self.four, self.cls = int(edge_tokens), EDGE_FEATURES, int(fourier), int(bool(cls_token))
        self.z, self.zone_f, self.goal_f = int(zone_tokens), ZONE_FEATURES, GOAL_FEATURES
        self.h, self.act_f = int(act_tokens), ACT_FEATURES
        self.extra = extra_dim(self.k, self.z, self.h, self.e, ac_cpa)
        w = 1 + 2 * self.four                                           # input values per row value
        state_dim = state_dim - self.extra                              # size of the own state alone
        if n_rays % patch:
            raise ValueError("n_rays must be a multiple of patch")
        self.patch, self.n_tok, self.out_dim = patch, n_rays // patch, features_dim
        self.patch_embed = nn.Linear(lidar_ch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok, d_model))       # ray direction of each token
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.state_net = nn.Sequential(nn.Linear(state_dim, state_hidden), nn.ReLU(),
                                       nn.Linear(state_hidden, state_hidden), nn.ReLU())
        self.state_token = nn.Linear(state_hidden, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.dense = nn.Sequential(nn.Linear(d_model + state_hidden, features_dim), nn.ReLU(),
                                   nn.Linear(features_dim, features_dim), nn.ReLU())
        if self.k:                                     # made last: the weights of the other layers start as before
            self.ac_embed = nn.Sequential(nn.Linear(self.ac_f * w, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.ac_type = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.ac_type, std=0.02)
        mlp = lambda n: nn.Sequential(nn.Linear(n, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        if self.z:
            self.zone_embed, self.goal_embed = mlp(ZONE_FEATURES * w), mlp(GOAL_FEATURES * w)
            self.zone_type, self.goal_type = (nn.Parameter(torch.zeros(1, 1, d_model)) for _ in range(2))
            nn.init.trunc_normal_(self.zone_type, std=0.02)
            nn.init.trunc_normal_(self.goal_type, std=0.02)
        if self.h:
            self.act_embed = mlp(ACT_FEATURES)
            self.act_pos = nn.Parameter(torch.zeros(1, self.h, d_model))    # place in time of each action
            nn.init.trunc_normal_(self.act_pos, std=0.02)
        if self.e:
            self.edge_embed = mlp(EDGE_FEATURES * w)
            self.edge_pos = nn.Parameter(torch.zeros(1, self.e, d_model))   # direction bin of each edge token
            nn.init.trunc_normal_(self.edge_pos, std=0.02)
        if self.cls:
            self.cls_tok = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_tok, std=0.02)
        if self.four:
            self.register_buffer("freqs", torch.pi * 2.0 ** torch.arange(self.four, dtype=torch.float32))

    def ff(self, x):
        """Row values [..., f] -> [..., f * (1 + 2F)]: the values, then sin and cos at each frequency."""
        if not self.four:
            return x
        a = (x.unsqueeze(-1) * self.freqs).flatten(-2)
        return torch.cat([x, a.sin(), a.cos()], dim=-1)

    def tokens(self, lidar):
        b, c, r = lidar.shape
        x = lidar.reshape(b, c, self.n_tok, self.patch).permute(0, 2, 1, 3).reshape(b, self.n_tok, c * self.patch)
        return self.patch_embed(x) + self.pos

    def forward(self, state, lidar):
        if not (self.extra or self.cls):
            s = self.state_net(state)
            x = torch.cat([self.state_token(s).unsqueeze(1), self.tokens(lidar)], dim=1)
            x = self.blocks(x)
            return self.dense(torch.cat([self.norm(x[:, 0]), s], dim=1))
        b, o = state.shape[0], state.shape[1] - self.extra
        s = self.state_net(state[:, :o])
        toks = [self.state_token(s).unsqueeze(1), self.tokens(lidar)]
        pads = [torch.zeros(b, 1 + self.n_tok + self.cls, dtype=torch.bool, device=state.device)]
        if self.cls:
            toks.insert(0, self.cls_tok.expand(b, -1, -1))             # place 0 = the read-out

        def rows(o, n, f):                             # n rows of f values, then n flags -> rows, empty mask, new offset
            return (state[:, o:o + n * f].reshape(b, n, f), state[:, o + n * f:o + n * (f + 1)] < 0.5, o + n * (f + 1))

        if self.k:
            ac, empty, o = rows(o, self.k, self.ac_f)                    # empty [B, K]: True = no aircraft
            toks.append(self.ac_embed(self.ff(ac)) + self.ac_type)
            pads.append(empty)
        if self.z:
            zone, empty, o = rows(o, self.z, self.zone_f)
            toks += [self.zone_embed(self.ff(zone)) + self.zone_type,
                     self.goal_embed(self.ff(state[:, o:o + self.goal_f])).unsqueeze(1) + self.goal_type]
            pads += [empty, torch.zeros(b, 1, dtype=torch.bool, device=state.device)]
            o += self.goal_f
        if self.h:
            act, empty, o = rows(o, self.h, self.act_f)
            toks.append(self.act_embed(act) + self.act_pos)
            pads.append(empty)
        if self.e:
            edge, empty, o = rows(o, self.e, self.edge_f)
            toks.append(self.edge_embed(self.ff(edge)) + self.edge_pos)
            pads.append(empty)
        x = self.blocks(torch.cat(toks, dim=1), src_key_padding_mask=torch.cat(pads, dim=1))
        return self.dense(torch.cat([self.norm(x[:, 0]), s], dim=1))


class AttentionCritic(nn.Module):
    """Centralized critic of the reference: query, key and value projections, one multi-head attention
    layer over the aircraft encodings, mean over the aircraft, then an MLP: one value per environment.
    No mask: as in the reference (black_death), a finished aircraft enters with its zero observation."""

    def __init__(self, features_dim, hidden=64, n_heads=4):
        super().__init__()
        self.query, self.key, self.value = (nn.Linear(features_dim, hidden) for _ in range(3))
        self.mha = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.layers = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, feats):
        """feats [B, N, F] -> values [B]."""
        a, _ = self.mha(self.query(feats), self.key(feats), self.value(feats), need_weights=False)
        return self.layers(a.mean(dim=1)).squeeze(-1)


class Policy(nn.Module):
    def __init__(self, state_dim, lidar_ch, n_rays, action_dim=2, patch=4, d_model=64, n_heads=4, n_layers=2,
                 features_dim=256, hidden=64, log_std_init=0.0, ac_tokens=0, zone_tokens=0, act_tokens=0,
                 edge_tokens=0, ac_cpa=0, fourier=0, cls_token=0):
        super().__init__()
        self.encoder = LidarTransformerEncoder(state_dim, lidar_ch, n_rays, patch, d_model, n_heads, n_layers,
                                               features_dim=features_dim, ac_tokens=ac_tokens,
                                               zone_tokens=zone_tokens, act_tokens=act_tokens,
                                               edge_tokens=edge_tokens, ac_cpa=ac_cpa, fourier=fourier,
                                               cls_token=cls_token)
        self.actor = nn.Sequential(nn.Linear(features_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, action_dim), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.critic = AttentionCritic(features_dim, hidden)

    def forward(self, state, lidar):
        """state [B, N, S], lidar [B, N, C, R] -> Normal over actions [B, N, A], value [B] (one per environment)."""
        b, n = state.shape[:2]
        f = self.encoder(state.reshape(b * n, -1), lidar.reshape(b * n, *lidar.shape[2:])).reshape(b, n, -1)
        mean = self.actor(f)
        std = self.log_std.clamp(-20.0, 2.0).exp().expand_as(mean)     # clamp of the reference
        return Normal(mean, std), self.critic(f)

    @torch.no_grad()
    def act(self, state, lidar, deterministic=False):
        """Returns raw action [B, N, A], log-prob [B, N], value [B, N] (the environment value, repeated per aircraft)."""
        d, v = self(state, lidar)
        raw = d.mean if deterministic else d.sample()
        return raw, d.log_prob(raw).sum(-1), v[:, None].expand(-1, state.shape[1])


def n_params(module):
    return sum(p.numel() for p in module.parameters())


NET_KEYS = ["patch", "d_model", "n_heads", "n_layers", "features_dim", "log_std_init"]


def build_policy(dims, args):
    return Policy(dims["state_dim"], dims["lidar_ch"], dims["n_rays"], ac_tokens=dims.get("ac_tokens", 0),
                  zone_tokens=dims.get("zone_tokens", 0), act_tokens=dims.get("act_tokens", 0),
                  edge_tokens=dims.get("edge_tokens", 0), ac_cpa=dims.get("ac_cpa", 0),
                  fourier=args.get("fourier", 0), cls_token=args.get("cls_token", 0),
                  **{k: args[k] for k in NET_KEYS})


def save_checkpoint(path, policy, args, dims, extra=None):
    torch.save({"policy": policy.state_dict(), "args": dict(args), "dims": dict(dims),
                "n_params": n_params(policy), **(extra or {})}, path)


def load_policy(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    policy = build_policy(ck["dims"], ck["args"]).to(device)
    policy.load_state_dict(ck["policy"])
    return policy, ck
