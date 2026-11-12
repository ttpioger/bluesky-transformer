"""Small helpers: seeds, GAE, pass criterion, JSON output."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch

EVENT_KEYS = ["intrusion_events", "restricted_area_events", "sector_exit_events"]
WINDOW = 20          # agent-episodes in the first and the last window
MIN_WAYPOINT = 0.95
MAX_EVENTS = 0.05


def setup_torch(seed, threads=1):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_gae(rew, val, done, last_val, gamma, lam):
    """GAE of the reference (stable-baselines3 rollout buffer). Arrays [T, M] (M = environments times
    aircraft). done[t] = the environment ended at step t (all aircraft finished, or the time limit:
    no bootstrap in both cases, as in the reference). last_val [M] = value after the last step.
    Returns (advantage, return)."""
    T = rew.shape[0]
    adv = np.zeros_like(rew, dtype=np.float32)
    last = np.zeros(rew.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nnt = 1.0 - done[t].astype(np.float32)
        delta = rew[t] + gamma * nv * nnt - val[t]
        last = delta + gamma * lam * nnt * last
        adv[t] = last
    return adv, adv + val


def window_stats(rows):
    if not rows:
        return None
    out = {"n_agent_episodes": len(rows)}
    for k in [k for k, v in rows[0].items() if k not in ("env_steps", "episode_seed") and isinstance(v, (int, float))]:
        out[k] = float(np.mean([r[k] for r in rows]))
    out["ep_return_std"] = float(np.std([r["ep_return"] for r in rows]))
    return out


def events_ok(w):
    return w["waypoint_reached"] >= MIN_WAYPOINT and all(w[k] <= MAX_EVENTS for k in EVENT_KEYS)


def pass_criterion(episodes):
    """Over the last 20 agent-episodes of training: waypoint_reached >= 0.95, each event mean <= 0.05,
    and ep_return above the first 20 by more than 2 standard errors."""
    first, last = window_stats(episodes[:WINDOW]), window_stats(episodes[-WINDOW:])
    if len(episodes) < 2 * WINDOW:
        return {"pass": False, "reason": f"fewer than {2 * WINDOW} agent-episodes", "first_window": first, "last_window": last}
    se = math.sqrt((first["ep_return_std"] ** 2 + last["ep_return_std"] ** 2) / WINDOW)
    ok_e, ok_r = events_ok(last), last["ep_return"] > first["ep_return"] + 2.0 * se
    return {"pass": bool(ok_e and ok_r), "events_ok": bool(ok_e), "return_clearly_above_first": bool(ok_r),
            "first_window": first, "last_window": last}


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))
    print(f"Wrote {p}")


def wandb_init(mode, project, name, config=None):
    """Weights & Biases run, or None if mode is "off". The run id is the run name, so that scripts.train and
    scripts.compare write to the same run."""
    if mode == "off":
        return None
    import wandb
    return wandb.init(project=project, name=name, id=name, resume="allow", mode=mode, config=config)


def wandb_upload(wb, path):
    """Copy one file (model or summary) to the Files tab of the W&B run. A later call replaces the file."""
    if wb is not None:
        wb.save(str(path), base_path=str(Path(path).parent.parent), policy="now")


SUMMARY_ROWS = [("waypoint_reached", "goal completed (share of flights)"), ("flight_time", "flight time (s)"),
                ("intrusion_events", "intrusion events"), ("intrusion_time", "intrusion time (s)"),
                ("restricted_area_events", "restricted area events"), ("time_in_restricted_area", "restricted area time (s)"),
                ("sector_exit_events", "sector exit events"), ("time_outside_sector", "time outside sector (s)")]


def summary_table(straight, policy, seed, wb=None):
    """Table of the mean metrics per flight: metric | fly straight | seed N (our Transformer), same scenarios.
    straight and policy: {metric key: mean}. Prints the table, sends it to W&B (key held_out/summary) if wb is
    given, and returns (columns, rows)."""
    cols = ["metric", "fly straight", f"seed {seed} (our Transformer)"]
    rows = [[label, round(float(straight[k]), 2), round(float(policy[k]), 2)] for k, label in SUMMARY_ROWS]
    print(f"\n{cols[0]:<36} {cols[1]:>14} {cols[2]:>28}")
    for r in rows:
        print(f"{r[0]:<36} {r[1]:>14.2f} {r[2]:>28.2f}")
    if wb is not None:
        import wandb
        wb.log({"held_out/summary": wandb.Table(columns=cols, data=rows)})
    return cols, rows


def add_wandb_args(p):
    p.add_argument("--wandb", choices=["off", "online", "offline"], default="off",
                   help="Send the main metrics (and in scripts.compare the figures and GIFs) to Weights & Biases.")
    p.add_argument("--wandb-project", default="bluesky-rl-transformer")


def policy_fn_from(policy, device="cpu", deterministic=True):
    """policy_fn for rollout_episode: joint forward pass over the live aircraft."""
    def fn(raw_obs, built):
        ids = list(built)
        state = torch.as_tensor(np.stack([built[a][0] for a in ids])[None], device=device)
        lidar = torch.as_tensor(np.stack([built[a][1] for a in ids])[None], device=device)
        raw, _, _ = policy.act(state, lidar, deterministic=deterministic)
        act = raw[0].clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
        return {a: act[i] for i, a in enumerate(ids)}
    return fn
