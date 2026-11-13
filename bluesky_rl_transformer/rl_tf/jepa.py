"""Temporal, action-conditioned JEPA for the encoder of the policy (2026-11-13).

z_t = f(o_t) is the output of LidarTransformerEncoder (the features that the actor and the critic read).
A predictor g rolls the embedding forward with the actions: p_1 = z_t + g(z_t, a_t), p_k = p_{k-1} + g(p_{k-1}, a_{t+k-1}).
The target of step k is f_ema(o_{t+k}): a slow copy of f (moving average of its weights, no gradient).
Loss = temporal_coef * mean over k of MSE(p_k, target_k) + var_coef * variance hinge + cov_coef * covariance term
on z_t (VICReg terms against collapse) + masked_coef * masked loss. Only the encoder goes into PPO; the predictors
and the slow copy stay here.

Masked loss (2026-11-13, I-JEPA on the tokens of o_t, masked_coef > 0): per sample, one block of map patches
(map_block x map_block of the G x G patch grid) and one arc of lidar patches (lidar_arc adjacent patches, the arc
can go over ray 0) are the targets. The online encoder runs with the targets out of the context (no token attends
to them). A small Transformer predictor gets the context outputs, one learned "mask" vector at each target place
and one learned vector per token place, and predicts the outputs of the slow copy at the targets (the slow copy
sees the full observation). Loss: MSE. The map and lidar targets are hidden together, so the predictor must
use the other kinds (aircraft, own state) and the rest of both images.

References (identifiers from memory, not checked online): Assran et al. 2023, I-JEPA, arXiv 2301.08243;
Grill et al. 2020, BYOL (moving-average target), arXiv 2006.07733; Bardes et al. 2022, VICReg, arXiv 2105.04906.
DOI not verified for all three.
"""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch import nn


class TemporalJepa(nn.Module):
    def __init__(self, encoder, action_dim=2, hidden=512, act_embed=16, var_coef=1.0, cov_coef=0.04,
                 temporal_coef=1.0, masked_coef=0.0, map_block=2, lidar_arc=6, mpred_dim=128, mpred_layers=2,
                 mpred_heads=4):
        super().__init__()
        d = encoder.out_dim
        self.online = encoder
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.act_embed = nn.Linear(action_dim, act_embed)
        self.predictor = nn.Sequential(nn.Linear(d + act_embed, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                       nn.GELU(), nn.Linear(hidden, d))
        nn.init.zeros_(self.predictor[-1].weight)            # start: p_k = z_t (predict "no change")
        nn.init.zeros_(self.predictor[-1].bias)
        self.var_coef, self.cov_coef = float(var_coef), float(cov_coef)
        self.temporal_coef, self.masked_coef = float(temporal_coef), float(masked_coef)
        self.mask_generator = None                     # torch.Generator for the target places; None = global
        if self.masked_coef > 0:                       # made last: the weights above start as before
            self.map_block = min(int(map_block), encoder.map_g) if encoder.n_map else 0
            self.lidar_arc = min(int(lidar_arc), encoder.n_tok)
            if not (self.map_block or self.lidar_arc):
                raise ValueError("masked JEPA needs map patches (map_px > 0, map_block > 0) or lidar_arc > 0")
            d, dp, T = encoder.blocks.layers[0].linear1.in_features, int(mpred_dim), encoder.n_tokens
            self.mpred_in = nn.Linear(d, dp)
            self.mask_tok = nn.Parameter(torch.zeros(1, 1, dp))
            self.mpred_pos = nn.Parameter(torch.zeros(1, T, dp))
            nn.init.trunc_normal_(self.mask_tok, std=0.02)
            nn.init.trunc_normal_(self.mpred_pos, std=0.02)
            layer = nn.TransformerEncoderLayer(dp, mpred_heads, 2 * dp, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.mpred = nn.TransformerEncoder(layer, mpred_layers, enable_nested_tensor=False)
            self.mpred_out = nn.Sequential(nn.LayerNorm(dp), nn.Linear(dp, d))

    def predictor_params(self):
        """Number of weights outside the encoder and its slow copy."""
        skip = {id(q) for q in self.online.parameters()} | {id(q) for q in self.target.parameters()}
        return sum(q.numel() for q in self.parameters() if id(q) not in skip)

    def sample_targets(self, b, device):
        """Target token places [B, M] (M = map_block^2 + lidar_arc): one map block and one lidar arc per sample."""
        enc, idx = self.online, []
        if self.map_block:
            g, s = enc.map_g, self.map_block
            r = torch.randint(0, g - s + 1, (b, 1), device=device, generator=self.mask_generator)
            c = torch.randint(0, g - s + 1, (b, 1), device=device, generator=self.mask_generator)
            k = torch.arange(s, device=device)
            cell = ((r + k[None, :])[:, :, None] * g + (c + k[None, :])[:, None, :]).reshape(b, s * s)
            idx.append(enc.map_start + cell)
        if self.lidar_arc:
            n = enc.n_tok
            a = torch.randint(0, n, (b, 1), device=device, generator=self.mask_generator)
            idx.append(enc.lidar_start + (a + torch.arange(self.lidar_arc, device=device)[None, :]) % n)
        return torch.cat(idx, dim=1)

    def masked_loss(self, state, lidar):
        """state [B, S], lidar [B, C, R] (one observation each). Returns the MSE at the target tokens."""
        b, dev = state.shape[0], state.device
        idx = self.sample_targets(b, dev)
        hide = torch.zeros(b, self.online.n_tokens, dtype=torch.bool, device=dev)
        hide.scatter_(1, idx, True)
        ctx, pad = self.online.token_outputs(state, lidar, hide)
        with torch.no_grad():
            tgt, _ = self.target.token_outputs(state, lidar)
        pick = lambda x: torch.gather(x, 1, idx[:, :, None].expand(-1, -1, x.shape[-1]))
        x = torch.where(hide[:, :, None], self.mask_tok.expand(b, ctx.shape[1], -1), self.mpred_in(ctx))
        y = self.mpred(x + self.mpred_pos, src_key_padding_mask=pad)
        return (self.mpred_out(pick(y)) - pick(tgt)).pow(2).mean()

    def step(self, z, a):
        return z + self.predictor(torch.cat([z, self.act_embed(a)], dim=-1))

    def loss(self, state, lidar, actions):
        """state [B, H+1, S], lidar [B, H+1, C, R], actions [B, H, A] (actions a_t .. a_{t+H-1}).
        Returns (loss, dict of float diagnostics)."""
        b, h1 = state.shape[:2]
        h = h1 - 1
        z = self.online(state[:, 0], lidar[:, 0])
        p, pred = z, torch.zeros((), device=z.device)
        per_k = []
        if self.temporal_coef > 0:
            with torch.no_grad():
                tgt = self.target(state[:, 1:].reshape(b * h, -1), lidar[:, 1:].reshape(b * h, *lidar.shape[2:]))
                tgt = tgt.reshape(b, h, -1)
            for k in range(h):
                p = self.step(p, actions[:, k])
                e = (p - tgt[:, k]).pow(2).mean()
                per_k.append(e.item())
                pred = pred + e / h
        var, cov = vicreg_terms(z)
        loss = (pred if self.temporal_coef == 1.0 else self.temporal_coef * pred) + self.var_coef * var + self.cov_coef * cov
        info = {"pred": float(pred.item()), "var": float(var.item()), "cov": float(cov.item()),
                "spread": float(z.detach().std(dim=0).mean().item()), "pred_per_k": per_k}
        if self.masked_coef > 0:
            m = self.masked_loss(state[:, 0], lidar[:, 0])
            loss = loss + self.masked_coef * m
            info["masked"] = float(m.item())
        return loss, info

    @torch.no_grad()
    def update_target(self, tau):
        for po, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.mul_(tau).add_(po.detach(), alpha=1.0 - tau)
        for bo, bt in zip(self.online.buffers(), self.target.buffers()):
            bt.copy_(bo)


def vicreg_terms(z, eps=1e-4):
    """Variance hinge (std of each dimension >= 1) and mean squared off-diagonal covariance, over the batch."""
    std = torch.sqrt(z.var(dim=0) + eps)
    var = torch.relu(1.0 - std).mean()
    zc = z - z.mean(dim=0)
    c = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    off = c - torch.diag(torch.diag(c))
    return var, off.pow(2).sum() / z.shape[1]


def ema_tau(step, total, start=0.996):
    """Cosine schedule of the moving-average rate from `start` to 1 (BYOL)."""
    return 1.0 - (1.0 - start) * (math.cos(math.pi * min(step, total) / max(total, 1)) + 1.0) / 2.0


def window_starts(has_next, horizon):
    """Indices i such that rows i .. i+horizon are consecutive steps of one flight (has_next[i..i+horizon-1] = 1)."""
    ok = has_next.astype(bool)
    n = len(ok)
    if n <= horizon:
        return np.zeros(0, dtype=np.int64)
    run = np.ones(n - horizon, dtype=bool)
    for k in range(horizon):
        run &= ok[k:n - horizon + k]
    return np.nonzero(run)[0]


# ------------------------------------------------------------------ linear probes on frozen features
def ridge_fit(x, y, lam=1e-2):
    """Closed-form ridge regression with a bias. x [N, D], y [N] -> weight [D + 1]."""
    xb = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    a = xb.T @ xb + lam * len(x) * np.eye(xb.shape[1])
    return np.linalg.solve(a, xb.T @ y)


def ridge_predict(w, x):
    return np.concatenate([x, np.ones((len(x), 1))], axis=1) @ w


def auc(score, label):
    """Area under the ROC curve (rank formula, ties get the mean rank). NaN if one class is absent."""
    label = label.astype(bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):                                     # mean rank of each run of equal scores
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def future_labels(flag, has_next, horizon):
    """1 where `flag` is 1 in any of the next `horizon` steps of the same flight (steps t .. t+horizon-1), and the
    flag is 0 at the step before t (the event is not already going on). Also returns the mask of the rows where
    the label is defined (the flag was 0 before)."""
    n = len(flag)
    f, nxt = flag.astype(bool), has_next.astype(bool)
    label = f.copy()                                      # k = 0: the step that follows o_t
    reach = np.ones(n, dtype=bool)                        # reach[t]: row t + k is in the same flight as row t
    for k in range(1, horizon):
        reach[:n - k] &= nxt[k - 1:n - 1]
        reach[n - k:] = False
        label[:n - k] |= reach[:n - k] & f[k:]
    prev = np.zeros(n, dtype=bool)                        # flag of the step before o_t, same flight
    prev[1:] = f[:-1] & nxt[:-1]
    return label, ~prev
