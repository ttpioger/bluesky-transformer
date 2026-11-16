"""Latent world model and CEM planner on a frozen encoder (2026-11-13), adapted from V-JEPA 2-AC.

State of the model at step t (all from the FROZEN encoder of a policy or of a JEPA run):
  h [B, T, d]  token outputs (the same values as encoder.token_outputs),
  s [B, Hs]    own-state encoding (encoder.state_net),
  pad [B, T]   empty rows of the observation at t (True = no token); kept fixed during a rollout.
  z = encoder.dense(h[:, 0] (+) s) is exactly the feature vector that the policy actor and critic read.

Predictor P (Markov: it sees step t only; the 4 frames inside the observation carry the motion):
  Transformer over the T tokens + one own-state token + one action token (pre-norm layers, GELU, no dropout).
  Residual output, last layers start at zero (start: "no change"): h(t+1) = h(t) + dh, s(t+1) = s(t) + ds.
  Neighbour commands (multi agent, 2026-11-13 late): each aircraft token row j can get (turn, speed, known) of
  the aircraft in that row, added to its input token through a small MLP (last layer at zero). known = 0 (scripted
  traffic, or a command that is not known) gives the learned "unknown" input.

Heads: cross-attention from the own-state token to all tokens, then an MLP -> 6 numbers per predicted step:
  0-2 logits: intrusion / inside a restricted area / outside the sector during the step (labels f_intr, f_area, f_out),
  3   decrease of the goal distance in the step (km, divided by norms[2]),
  4   reward of the step (the reward weights of the data, normalised with norms[0], norms[1]),
  5   value of the state: discounted return to go of the recorded flight, in PPO units (reward_scale, gamma of the
      policy run), normalised with norms[3], norms[4]. The return of the recorded controllers, not of the planner.

Loss (WorldModel.loss): rollout of H steps from h(t) with the recorded actions against the encoder outputs at
t+k (squared error over the tokens that are real at t and at t+k, plus the own-state encoding), the one-step
error from the true state at t+k-1, and the head losses on the predicted and on the true states.

Known limit: tokens are matched over time by place. The aircraft rows are sorted by distance, so one row can
hold another aircraft at t+k; the predictor must learn the new order too. The neighbour commands follow the
aircraft that held row j at t.

References (identifiers from memory, not checked): Assran et al. 2025, V-JEPA 2, arXiv 2506.09985;
Hansen et al. 2022, TD-MPC, arXiv 2203.04955. DOI not verified.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

N_OUT = 6                                              # 3 event logits, goal step, reward, value
NB_F = 3                                               # neighbour command: turn, speed, known


class WorldModel(nn.Module):
    def __init__(self, encoder, action_dim=2, n_layers=4, n_heads=4, head_hidden=128):
        super().__init__()
        self.encoder = encoder
        encoder.requires_grad_(False)                  # frozen: the predicted vectors stay in its space
        d = encoder.blocks.layers[0].linear1.in_features
        hs = encoder.state_token.in_features
        self.d, self.hs, self.own_i = d, hs, int(encoder.cls)       # own-state token: place 0, or 1 after a read-out
        self.k, self.ac0 = int(encoder.k), int(encoder.lidar_start + encoder.n_tok)   # aircraft rows and their place
        self.n_layers, self.n_heads, self.head_hidden = int(n_layers), int(n_heads), int(head_hidden)
        self.act_in = nn.Sequential(nn.Linear(action_dim, d), nn.GELU(), nn.Linear(d, d))
        self.s_in = nn.Linear(hs, d)
        self.kind = nn.Parameter(torch.zeros(2, d))    # learned vector of the action token (0) and the own-state token (1)
        nn.init.trunc_normal_(self.kind, std=0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.dh, self.ds = nn.Linear(d, d), nn.Linear(d, hs)
        self.nb_in = nn.Sequential(nn.Linear(NB_F, d), nn.GELU(), nn.Linear(d, d))
        for m in (self.dh, self.ds, self.nb_in[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.head_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.head_mlp = nn.Sequential(nn.Linear(2 * d + hs, head_hidden), nn.GELU(), nn.Linear(head_hidden, head_hidden),
                                      nn.GELU(), nn.Linear(head_hidden, N_OUT))
        # reward mean, reward std, goal-step std (km), value mean, value std
        self.register_buffer("norms", torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0]))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def params(self):
        """Trained weights (the encoder is not trained here)."""
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ model
    @torch.no_grad()
    def encode(self, state, lidar):
        """state [B, S], lidar [B, C, R] -> h [B, T, d], s [B, Hs], pad [B, T] (encoder.token_outputs + state_net)."""
        enc = self.encoder
        x, pad, s = enc.embed(state, lidar)
        x = enc.blocks(x, src_key_padding_mask=pad)
        return enc.norm(x), s, pad

    def z(self, h, s):
        """Feature vector of the policy (LidarTransformerEncoder.forward) from h and s."""
        return self.encoder.dense(torch.cat([h[:, 0], s], dim=1))

    def step(self, h, s, pad, a, nb=None):
        """One 10 s step: (h, s) at t, own command a [B, A] and neighbour commands nb [B, K, 3] (None = all unknown)
        -> (h, s) at t+1."""
        b, t = h.shape[:2]
        x = torch.cat([h, (self.s_in(s) + self.kind[1]).unsqueeze(1), (self.act_in(a) + self.kind[0]).unsqueeze(1)], 1)
        if self.k:
            if nb is None:
                nb = h.new_zeros(b, self.k, NB_F)
            x = torch.cat([x[:, :self.ac0], x[:, self.ac0:self.ac0 + self.k] + self.nb_in(nb), x[:, self.ac0 + self.k:]], 1)
        m = torch.cat([pad, pad.new_zeros(b, 2)], dim=1)
        y = self.norm(self.blocks(x, src_key_padding_mask=m))
        return h + self.dh(y[:, :t]), s + self.ds(y[:, t])

    def heads(self, h, s, pad):
        """[B, 6]: event logits (3), goal step, reward, value (the last three normalised)."""
        q = h[:, self.own_i:self.own_i + 1]
        a, _ = self.head_attn(q, h, h, key_padding_mask=pad, need_weights=False)
        return self.head_mlp(torch.cat([a[:, 0], h[:, self.own_i], s], dim=1))

    def reward(self, out):
        return out[:, 4] * self.norms[1] + self.norms[0]

    def value(self, out):
        return out[:, 5] * self.norms[4] + self.norms[3]

    @staticmethod
    def probs(out):
        return torch.sigmoid(out[:, :3])

    # ------------------------------------------------------------------ training
    def loss(self, state, lidar, actions, labels, pos_weight, nb=None, tf_coef=1.0, s_coef=1.0, head_coef=1.0):
        """state [B, H+1, S], lidar [B, H+1, C, R], actions [B, H, A]; labels [B, H, 6]: for step k (from row t+k-1
        to row t+k) the 3 event flags, the decrease of the goal distance (km), the reward and the value target of
        row t+k. nb [B, H, K, 3] neighbour commands of step k (None = unknown). pos_weight [3]: weight of the
        positive rows of each event (rare events). Returns (loss, info)."""
        b, h1 = state.shape[:2]
        H = h1 - 1
        hb, sb, pb = self.encode(state.reshape(b * h1, -1), lidar.reshape(b * h1, *lidar.shape[2:]))
        hb, sb, pb = hb.reshape(b, h1, *hb.shape[1:]), sb.reshape(b, h1, -1), pb.reshape(b, h1, -1)
        y = torch.cat([labels[..., :3], labels[..., 3:4] / self.norms[2],
                       (labels[..., 4:5] - self.norms[0]) / self.norms[1],
                       (labels[..., 5:6] - self.norms[3]) / self.norms[4]], dim=-1)
        h, s, pad0 = hb[:, 0], sb[:, 0], pb[:, 0]
        roll = one = hd = torch.zeros((), device=state.device)
        per_k = []
        for k in range(1, h1):
            a = actions[:, k - 1]
            n_k = None if nb is None else nb[:, k - 1]
            h, s = self.step(h, s, pad0, a, n_k)
            e = latent_error(h, s, hb[:, k], sb[:, k], ~(pad0 | pb[:, k]), s_coef)
            per_k.append(float(e.item()))
            roll = roll + e / H
            if tf_coef > 0 and k > 1:                  # k = 1 of the rollout is already a one-step prediction
                ht, st = self.step(hb[:, k - 1], sb[:, k - 1], pb[:, k - 1], a, n_k)
                one = one + latent_error(ht, st, hb[:, k], sb[:, k], ~(pb[:, k - 1] | pb[:, k]), s_coef) / (H - 1)
            hd = hd + (head_error(self.heads(h, s, pad0), y[:, k - 1], pos_weight)
                       + head_error(self.heads(hb[:, k], sb[:, k], pb[:, k]), y[:, k - 1], pos_weight)) / (2 * H)
        loss = roll + tf_coef * one + head_coef * hd
        return loss, {"roll": float(roll.item()), "one": float(one.item()), "head": float(hd.item()), "roll_per_k": per_k}


def latent_error(h, s, ht, st, ok, s_coef=1.0):
    """Mean squared error over the real tokens (ok [B, T]) plus s_coef times the error of the own-state encoding."""
    e = (h - ht).pow(2).mean(-1)
    okf = ok.float()
    return (e * okf).sum() / okf.sum().clamp_min(1.0) + s_coef * (s - st).pow(2).mean()


def head_error(out, y, pos_weight):
    return (F.binary_cross_entropy_with_logits(out[:, :3], y[:, :3], pos_weight=pos_weight)
            + F.mse_loss(out[:, 3:], y[:, 3:]))


# ---------------------------------------------------------------------- data helpers
def returns_to_go(reward, has_next, gamma, scale):
    """Discounted return to go per row, PPO units: G[i] = scale * reward[i] + gamma * G[i+1] inside one flight;
    G = 0 at the last row of a flight (has_next 0: no action there; no bootstrap, as in the PPO of scripts.train)."""
    g = np.zeros(len(reward), dtype=np.float64)
    nxt = 0.0
    for i in range(len(reward) - 1, -1, -1):
        nxt = scale * float(reward[i]) + gamma * nxt if has_next[i] else 0.0
        g[i] = nxt
    return g.astype(np.float32)


def step_labels(data, starts, horizon):
    """Labels [N, H, 6] of the windows that start at the rows `starts`: for step k = 1..H (row t+k-1 to row t+k)
    f_intr, f_area, f_out of row t+k-1, goal_km[t+k-1] - goal_km[t+k], reward of row t+k-1, ret of row t+k."""
    rows = starts[:, None] + np.arange(horizon)[None, :]
    g = data["goal_km"].astype(np.float64)
    return np.stack([data["f_intr"][rows], data["f_area"][rows], data["f_out"][rows], g[rows] - g[rows + 1],
                     data["reward"][rows], data["ret"][rows + 1]], axis=-1).astype(np.float32)


def action_lookup(data):
    """Sorted keys (flight, aircraft, step) -> row, for neighbour_commands."""
    key = (data["episode"].astype(np.int64) * 256 + data["agent"].astype(np.int64)) * 65536 + data["step"].astype(np.int64)
    order = np.argsort(key, kind="stable")
    return key[order], order


def neighbour_commands(data, starts, horizon, lookup):
    """[N, H, K, 3]: for each window, the command (turn, speed, known) that the aircraft in aircraft row j of row t
    flew at step t+k-1. known = 0 where the row holds no learning aircraft (nb_ids -1) or that aircraft has no
    action at that step (it finished)."""
    keys, order = lookup
    nb = data["nb_ids"][starts].astype(np.int64)                       # [N, K]
    ep = data["episode"][starts].astype(np.int64)[:, None]
    st = data["step"][starts].astype(np.int64)[:, None]
    out = np.zeros((len(starts), horizon, nb.shape[1], NB_F), dtype=np.float32)
    for k in range(horizon):
        key = (ep * 256 + nb) * 65536 + st + k
        pos = np.clip(np.searchsorted(keys, key), 0, len(keys) - 1)
        rows = order[pos]
        found = (nb >= 0) & (keys[pos] == key) & (data["has_next"][rows] == 1)
        out[:, k, :, :2] = np.where(found[..., None], data["action"][rows], 0.0)
        out[:, k, :, 2] = found
    return out


def aircraft_row_ids(lidar, obs):
    """Callsigns of the aircraft in the aircraft-token rows of `obs` (the order of Lidar.aircraft_tokens)."""
    from rl_tf.lidar import GEOM_KEY
    g = obs[GEOM_KEY]
    ids = g.get("aircraft_ids")
    ac = np.asarray(g["aircraft"], dtype=np.float64).reshape(-1, 2)
    if not lidar.ac_tokens or ids is None or not len(ac):
        return []
    dist = np.linalg.norm(ac, axis=1)
    order = np.argsort(dist, kind="stable")
    order = order[dist[order] <= lidar.max_range + lidar.ac_radius][:lidar.ac_tokens]
    return [ids[i] for i in order]


# ---------------------------------------------------------------------- planner
class Planner:
    """Receding-horizon planner with the cross-entropy method (CEM) in the latent model.

    level 1 (safety filter): roll the policy mean out in the model for `horizon` steps (all aircraft together; each
      aircraft-token row gets the rollout commands of its aircraft). If the largest predicted probability of an
      intrusion or of being inside a restricted area is >= `threshold`, search; else fly the policy mean (then the
      action is the action of the policy alone).
    level 2: always search.
    Search: `rounds` rounds of `samples` command sequences of `horizon` steps around a mean (start: the policy
      rollout) with the spread max(policy std, init_std); keep the best `elites`, refit the mean and the spread
      (at least min_std). Sequence 0 of round 1 is the policy rollout itself, sequence 0 of a later round is the
      current mean. The first command of the best sequence of all rounds is flown.
    Score (units of the PPO return): sum_k gamma^(k-1) * reward_scale * predicted reward(k)
      - turn_cost * sum_k |turn(k) - turn(k-1)| + gamma^H * V(t+H).
      value "critic": V = the policy critic on z(t+H) (single agent: the critic is trained on one aircraft).
      value "head": V = the value head of the world model (multi agent: the central critic needs all aircraft).
    Neighbours (multi agent): the other aircraft follow their current plan (start: their policy rollout).
      order "independent": all aircraft search at once against the policy rollouts of the others.
      order "sequential": aircraft search one after another (the order of `order_idx`); a later aircraft sees the
      chosen plans of the earlier ones."""

    def __init__(self, wm, policy, level=1, samples=64, rounds=3, horizon=6, elites=8, threshold=0.5, turn_cost=0.0,
                 init_std=0.5, min_std=0.05, gamma=0.99, reward_scale=1.0, value="critic", order="independent", seed=0,
                 device="cpu"):
        if level not in (1, 2):
            raise ValueError("level must be 1 or 2")
        if not 1 <= elites <= samples:
            raise ValueError("need 1 <= elites <= samples")
        if value not in ("critic", "head") or order not in ("independent", "sequential"):
            raise ValueError("value: critic|head, order: independent|sequential")
        self.wm, self.policy, self.level = wm.eval(), policy.eval(), int(level)
        self.S, self.R, self.H, self.E = int(samples), int(rounds), int(horizon), int(elites)
        self.threshold, self.turn_cost, self.init_std, self.min_std = float(threshold), float(turn_cost), float(init_std), float(min_std)
        self.gamma, self.scale, self.value, self.order = float(gamma), float(reward_scale), value, order
        self.gen = torch.Generator(device=device).manual_seed(int(seed))

    def nb_from(self, plans, row_ids):
        """Neighbour commands [N, K, H, 3] from the plans [N, H, A] of all aircraft; row_ids [N, K] (-1 = unknown)."""
        if row_ids is None:
            return None
        known = row_ids >= 0
        g = plans[row_ids.clamp_min(0)] * known[..., None, None]
        return torch.cat([g, known[..., None, None].float().expand(-1, -1, self.H, 1)], dim=-1)

    @torch.no_grad()
    def policy_rollout(self, h, s, pad, row_ids):
        """Policy means of all aircraft rolled out together: sequences [N, H, A], risk [N] (largest intrusion or
        area probability over the rollout)."""
        seq, risk = [], torch.zeros(h.shape[0], device=h.device)
        for _ in range(self.H):
            a = self.policy.actor(self.wm.z(h, s)).clamp(-1.0, 1.0)
            seq.append(a)
            nb = None
            if row_ids is not None:
                known = row_ids >= 0
                nb = torch.cat([a[row_ids.clamp_min(0)] * known[..., None], known[..., None].float()], dim=-1)
            h, s = self.wm.step(h, s, pad, a, nb)
            risk = torch.maximum(risk, self.wm.probs(self.wm.heads(h, s, pad))[:, :2].max(dim=1).values)
        return torch.stack(seq, dim=1), risk

    @torch.no_grad()
    def score(self, h, s, pad, seqs, prev_turn, nbseq=None):
        """h [A, T, d], s [A, Hs], pad [A, T], seqs [A, S, H, act], prev_turn [A], nbseq [A, K, H, 3] -> [A, S]."""
        n, m = seqs.shape[:2]
        h, s, pad, prev = (x.repeat_interleave(m, dim=0) for x in (h, s, pad, prev_turn))
        nbs = None if nbseq is None else nbseq.repeat_interleave(m, dim=0)
        J, disc, out = torch.zeros(n * m, device=h.device), 1.0, None
        for k in range(self.H):
            a = seqs[:, :, k].reshape(n * m, -1)
            h, s = self.wm.step(h, s, pad, a, None if nbs is None else nbs[:, :, k])
            out = self.wm.heads(h, s, pad)
            J = J + disc * self.scale * self.wm.reward(out) - self.turn_cost * (a[:, 0] - prev).abs()
            prev, disc = a[:, 0], disc * self.gamma
        v = self.wm.value(out) if self.value == "head" else self.policy.critic(self.wm.z(h, s)[:, None, :])
        return (J + disc * v).reshape(n, m)

    @torch.no_grad()
    def search(self, h, s, pad, prior, prev_turn, nbseq):
        """CEM for the aircraft of the batch. Returns best sequences [A, H, act], best score, score of the prior."""
        mu = prior.clone()
        sd = self.policy.log_std.clamp(-20.0, 2.0).exp().clamp_min(self.init_std).expand_as(prior).clone()
        best, best_J, prior_J = prior.clone(), torch.full((prior.shape[0],), -float("inf"), device=h.device), None
        for r in range(self.R):
            eps = torch.randn(prior.shape[0], self.S, self.H, prior.shape[-1], generator=self.gen, device=h.device)
            seqs = (mu[:, None] + sd[:, None] * eps).clamp(-1.0, 1.0)
            seqs[:, 0] = prior if r == 0 else mu
            J = self.score(h, s, pad, seqs, prev_turn, nbseq)
            if r == 0:
                prior_J = J[:, 0].clone()
            top = J.topk(self.E, dim=1)
            el = torch.gather(seqs, 1, top.indices[:, :, None, None].expand(-1, -1, self.H, seqs.shape[-1]))
            better = top.values[:, 0] > best_J
            best[better] = el[better, 0]
            best_J = torch.where(better, top.values[:, 0], best_J)
            mu, sd = el.mean(dim=1), el.std(dim=1).clamp_min(self.min_std)
        return best, best_J, prior_J

    @torch.no_grad()
    def act(self, state, lidar, prev_turn, row_ids=None, order_idx=None):
        """state [N, S], lidar [N, C, R], prev_turn [N] (last flown turn command), row_ids [N, K] (batch index of the
        aircraft in each aircraft-token row, -1 = not a learning aircraft; None = single agent), order_idx: planning
        order for "sequential" (default 0..N-1) -> actions [N, A], info."""
        h, s, pad = self.wm.encode(state, lidar)
        out = self.policy.actor(self.wm.z(h, s)).clamp(-1.0, 1.0)
        plans, risk = self.policy_rollout(h, s, pad, row_ids)
        active = torch.ones_like(risk, dtype=torch.bool) if self.level == 2 else risk >= self.threshold
        info = {"risk": risk, "active": active}
        if not bool(active.any()):
            return out, info
        best_J = torch.full_like(risk, float("nan"))
        prior_J = torch.full_like(risk, float("nan"))
        if self.order == "independent" or row_ids is None:
            idx = active.nonzero().squeeze(1)
            nbseq = self.nb_from(plans, row_ids)
            seq, bj, pj = self.search(h[idx], s[idx], pad[idx], plans[idx], prev_turn[idx],
                                      None if nbseq is None else nbseq[idx])
            out[idx], best_J[idx], prior_J[idx] = seq[:, 0], bj, pj
        else:
            for i in (order_idx if order_idx is not None else range(state.shape[0])):
                if not bool(active[i]):
                    continue
                sl = slice(i, i + 1)
                seq, bj, pj = self.search(h[sl], s[sl], pad[sl], plans[sl], prev_turn[sl], self.nb_from(plans, row_ids)[sl])
                plans[i], out[i], best_J[i], prior_J[i] = seq[0], seq[0, 0], bj[0], pj[0]
        info.update(best_J=best_J, prior_J=prior_J)
        return out, info


class PlannerFn:
    """policy_fn for rollout_episode (rl_tf/envs.py) that flies the planner. Call reset() before each episode.
    lidar: the Lidar of the policy (needed in multi agent to know which aircraft sits in which aircraft row).
    Sequential planning order: callsign order."""

    def __init__(self, planner, device="cpu", lidar=None):
        self.p, self.device, self.lidar = planner, device, lidar
        self.reset()

    def reset(self):
        self.prev, self.decisions, self.active, self.seconds = {}, 0, 0, 0.0

    def __call__(self, raw_obs, built):
        t0 = time.perf_counter()
        ids = list(built)
        st = torch.as_tensor(np.stack([built[a][0] for a in ids]), dtype=torch.float32, device=self.device)
        li = torch.as_tensor(np.stack([built[a][1] for a in ids]), dtype=torch.float32, device=self.device)
        pt = torch.as_tensor([self.prev.get(a, 0.0) for a in ids], dtype=torch.float32, device=self.device)
        row_ids = None
        if self.lidar is not None and self.lidar.ac_tokens and len(ids) > 1:
            pos, K = {a: i for i, a in enumerate(ids)}, self.lidar.ac_tokens
            rows = []
            for a in ids:
                r = [pos.get(c, -1) for c in aircraft_row_ids(self.lidar, raw_obs[a])]
                rows.append(r + [-1] * (K - len(r)))
            row_ids = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        order_idx = sorted(range(len(ids)), key=lambda i: ids[i])
        act, info = self.p.act(st, li, pt, row_ids, order_idx)
        act = act.cpu().numpy().astype(np.float32)
        self.decisions += len(ids)
        self.active += int(info["active"].sum().item())
        self.seconds += time.perf_counter() - t0
        self.prev.update({a: float(act[i][0]) for i, a in enumerate(ids)})
        return {a: act[i] for i, a in enumerate(ids)}

    def stats(self):
        return {"plan_decisions": self.decisions, "plan_active": self.active, "plan_seconds": self.seconds}


# ---------------------------------------------------------------------- files
def save_world_model(path, wm, enc_args, enc_dims, extra=None):
    torch.save({"world_model": wm.state_dict(), "wm_args": {"n_layers": wm.n_layers, "n_heads": wm.n_heads,
                                                           "head_hidden": wm.head_hidden},
                "args": dict(enc_args), "dims": dict(enc_dims), **(extra or {})}, path)


def load_world_model(path, device="cpu"):
    from rl_tf.networks import build_policy
    ck = torch.load(path, map_location=device, weights_only=False)
    enc = build_policy(ck["dims"], ck["args"]).encoder
    wm = WorldModel(enc, **ck["wm_args"]).to(device)
    wm.load_state_dict(ck["world_model"])
    return wm.eval(), ck


def same_weights(a, b):
    """True if two modules have the same parameter names and exactly equal values."""
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k].cpu(), sb[k].cpu()) for k in sa)
