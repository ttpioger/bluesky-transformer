"""Small helpers shared by the scripts: cell names, pass criterion, JSON output."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from .obs import stack_obs, to_torch

CELLS = {"R": ("cnn", "off"), "A": ("set_tf", "off"), "B": ("cnn", "on"), "C": ("set_tf", "on")}
EVENT_KEYS = ["intrusion_events", "restricted_area_events", "sector_exit_events"]
WINDOW = 20          # agent-episodes in the first and the last window
MIN_WAYPOINT = 0.95  # "near 1"
MAX_EVENTS = 0.05    # "near 0"


def cell_name(encoder, jepa):
    return {v: k for k, v in CELLS.items()}[(encoder, jepa)]


def setup_torch(seed, threads=1):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)


def window_stats(rows):
    if not rows:
        return None
    keys = [k for k in rows[0] if k != "env_steps"]
    out = {"n_agent_episodes": len(rows)}
    for k in keys:
        out[k] = float(np.mean([r[k] for r in rows]))
    out["ep_return_std"] = float(np.std([r["ep_return"] for r in rows]))
    return out


def events_ok(w):
    return w["waypoint_reached"] >= MIN_WAYPOINT and all(w[k] <= MAX_EVENTS for k in EVENT_KEYS)


def _clearly_above(first, w):
    se = math.sqrt((first["ep_return_std"] ** 2 + w["ep_return_std"] ** 2) / WINDOW)
    return w["ep_return"] > first["ep_return"] + 2.0 * se


def pass_criterion(episodes):
    """Criterion fixed before the runs. Over the last 20 agent-episodes: waypoint_reached >= 0.95,
    each event mean <= 0.05, and ep_return above the first 20 by more than 2 standard errors.
    steps_to_criterion: env steps at the end of the first 20-episode window (no overlap with the
    first window) that meets the same rule."""
    if len(episodes) < 2 * WINDOW:
        return {"pass": False, "reason": f"fewer than {2 * WINDOW} agent-episodes",
                "first_window": window_stats(episodes[:WINDOW]), "last_window": window_stats(episodes[-WINDOW:]),
                "steps_to_criterion": None}
    first, last = window_stats(episodes[:WINDOW]), window_stats(episodes[-WINDOW:])
    ok_e, ok_r = events_ok(last), _clearly_above(first, last)
    steps = None
    for j in range(2 * WINDOW, len(episodes) + 1):
        w = window_stats(episodes[j - WINDOW:j])
        if events_ok(w) and _clearly_above(first, w):
            steps = int(episodes[j - 1]["env_steps"])
            break
    return {"pass": bool(ok_e and ok_r), "events_ok": bool(ok_e), "return_clearly_above_first": bool(ok_r),
            "first_window": first, "last_window": last, "steps_to_criterion": steps}


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))
    print(f"Wrote {p}")


def actor_policy(actor, device="cpu", deterministic=True):
    """policy_fn for rollout_episode: {id: obs} -> {id: clipped action}."""
    def fn(obs):
        ids = list(obs)
        raw, _ = actor.act(to_torch(stack_obs([obs[a] for a in ids]), device), deterministic=deterministic)
        act = raw.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
        return {a: act[i] for i, a in enumerate(ids)}
    return fn
