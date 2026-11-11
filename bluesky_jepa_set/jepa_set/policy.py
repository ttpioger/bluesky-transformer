"""Actor, set-attention critic, JEPA module and checkpoint helpers. Same heads for all cells."""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.distributions import Normal
from .encoders import build_encoder, n_params


class Actor(nn.Module):
    """Shared actor: encoder -> MLP -> tanh mean. Gaussian noise with one learned log_std per action.
    The environment gets the action clipped to [-1, 1] (same rule as the reference)."""

    def __init__(self, encoder_kind, emb_dim=64, hidden=64, action_dim=2, log_std_init=-1.0):
        super().__init__()
        self.encoder = build_encoder(encoder_kind, out_dim=emb_dim)
        self.head = nn.Sequential(nn.Linear(emb_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, action_dim))
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01)   # start mean near the zero action
        nn.init.zeros_(self.head[-1].bias)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def mean_from_emb(self, z):
        return torch.tanh(self.head(z))

    def dist(self, obs):
        z = self.encoder(obs)
        mean = self.mean_from_emb(z)
        return Normal(mean, self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)), z

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        d, _ = self.dist(obs)
        raw = d.mean if deterministic else d.sample()
        return raw, d.log_prob(raw).sum(-1)


class SetCritic(nn.Module):
    """Centralized critic. One token per aircraft (own encoder), one self-attention block
    across the live aircraft, one value per aircraft. Dead aircraft are masked.
    No aircraft-id vector: the aircraft count is free. With one aircraft it is a plain critic."""

    def __init__(self, encoder_kind, emb_dim=64, n_heads=4):
        super().__init__()
        self.encoder = build_encoder(encoder_kind, out_dim=emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.GELU(), nn.Linear(2 * emb_dim, emb_dim))
        self.value = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, 1))

    def forward(self, obs, alive):
        """obs: dict of tensors [B, N, n]. alive: bool [B, N]. Returns values [B, N] (0 for dead aircraft)."""
        b, n = alive.shape
        flat = {k: v.reshape(b * n, -1) for k, v in obs.items()}
        x = self.encoder(flat).reshape(b, n, -1)
        dead = ~alive
        dead = dead & ~dead.all(dim=1, keepdim=True)   # a row with no live aircraft must not be fully masked
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=dead, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return self.value(x).squeeze(-1) * alive.float()


class Jepa(nn.Module):
    """Action-conditioned temporal JEPA around an online encoder.

    z_i = f(frame_i) for k frames; h = GRU(z_1..z_k); prediction = g(h, action);
    target = f_ema(next frame) with stop-gradient. Loss = MSE + var_coef * variance hinge.
    """

    def __init__(self, encoder, hidden=64, action_dim=2, var_coef=1.0):
        super().__init__()
        d = encoder.out_dim
        self.online = encoder                       # shared with the actor
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.mixer = nn.GRU(d, hidden, batch_first=True)
        self.predictor = nn.Sequential(nn.Linear(hidden + action_dim, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.var_coef = var_coef

    def aux_parameters(self):
        return list(self.mixer.parameters()) + list(self.predictor.parameters())

    def loss(self, hist, action, nxt):
        """hist: dict of tensors [B, k, n]. action: [B, 2]. nxt: dict of tensors [B, n]."""
        b, k = action.shape[0], next(iter(hist.values())).shape[1]
        z = self.online({key: v.reshape(b * k, -1) for key, v in hist.items()}).reshape(b, k, -1)
        h = self.mixer(z)[0][:, -1]
        pred = self.predictor(torch.cat([h, action], dim=-1))
        with torch.no_grad():
            tgt = self.target(nxt)
        pred_loss = (pred - tgt).pow(2).mean()
        z_last = z[:, -1]
        std = torch.sqrt(z_last.var(dim=0, unbiased=False) + 1e-4)
        var_loss = torch.relu(1.0 - std).mean()
        stats = {"pred_loss": pred_loss.item(), "var_loss": var_loss.item(),
                 "spread": std.mean().item(), "target_spread": tgt.std(dim=0, unbiased=False).mean().item()}
        return pred_loss + self.var_coef * var_loss, stats

    @torch.no_grad()
    def ema_update(self, tau=0.99):
        for pt, po in zip(self.target.parameters(), self.online.parameters()):
            pt.mul_(tau).add_(po.detach(), alpha=1.0 - tau)


def save_checkpoint(path, actor, critic=None, jepa=None, args=None, extra=None):
    ck = {"actor": actor.state_dict(), "args": dict(args or {}), "n_params_actor": n_params(actor)}
    if critic is not None:
        ck["critic"] = critic.state_dict()
    if jepa is not None:
        ck["jepa_target"] = jepa.target.state_dict()
        ck["jepa_mixer"] = jepa.mixer.state_dict()
        ck["jepa_predictor"] = jepa.predictor.state_dict()
    ck.update(extra or {})
    torch.save(ck, path)


def load_actor(path, device="cpu"):
    """Returns (actor, checkpoint dict). Network sizes come from the saved args."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    actor = Actor(a["encoder"], emb_dim=a.get("emb_dim", 64), log_std_init=a.get("log_std_init", -1.0)).to(device)
    actor.load_state_dict(ck["actor"])
    return actor, ck
