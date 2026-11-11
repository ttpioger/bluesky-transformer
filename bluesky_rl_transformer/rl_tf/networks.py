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
    """

    def __init__(self, state_dim, lidar_ch, n_rays, patch=4, d_model=64, n_heads=4, n_layers=2,
                 state_hidden=128, features_dim=256):
        super().__init__()
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

    def tokens(self, lidar):
        b, c, r = lidar.shape
        x = lidar.reshape(b, c, self.n_tok, self.patch).permute(0, 2, 1, 3).reshape(b, self.n_tok, c * self.patch)
        return self.patch_embed(x) + self.pos

    def forward(self, state, lidar):
        s = self.state_net(state)
        x = torch.cat([self.state_token(s).unsqueeze(1), self.tokens(lidar)], dim=1)
        x = self.blocks(x)
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
                 features_dim=256, hidden=64, log_std_init=0.0):
        super().__init__()
        self.encoder = LidarTransformerEncoder(state_dim, lidar_ch, n_rays, patch, d_model, n_heads, n_layers,
                                               features_dim=features_dim)
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
    return Policy(dims["state_dim"], dims["lidar_ch"], dims["n_rays"], **{k: args[k] for k in NET_KEYS})


def save_checkpoint(path, policy, args, dims, extra=None):
    torch.save({"policy": policy.state_dict(), "args": dict(args), "dims": dict(dims),
                "n_params": n_params(policy), **(extra or {})}, path)


def load_policy(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    policy = build_policy(ck["dims"], ck["args"]).to(device)
    policy.load_state_dict(ck["policy"])
    return policy, ck
