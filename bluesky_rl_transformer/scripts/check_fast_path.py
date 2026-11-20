"""Bit-identical check of the 2026-11-20 speed changes (multi agent, no training, 1 process):
  1. rl_tf/fast_metrics.py (FlightEnv fast_metrics=True) against the original CompetitionZooEnv metrics;
  2. Lidar.zone_tokens_vec (edges of all areas in one array) against a copy of the code before the change.
Two environments fly the same scenarios with the same actions in lockstep. After every decision the script
compares all observations, rewards, flags and infos (metrics) exactly, and the zone tokens of every aircraft
byte for byte. Actions, per episode in turn: zero (fly straight: many area entries and intrusions), uniform
random in [-1, 1] (many turns), constant (1, -1) (turn away from the goal: sector exits). It also times both
versions.

  python -m scripts.check_fast_path --episodes 6
Writes results/check_fast_path_<host>.json. Exit code 1 if any difference is found.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import platform
import socket
import sys
import time
from pathlib import Path
import numpy as np
import torch
from rl_tf.envs import FlightEnv, REWARD_DEFAULTS, v2_kwargs
from rl_tf.lidar import Lidar, token_cfg, ray_segments, GEOM_KEY, DIST_NORM_KM, ZONE_FEATURES


def zone_tokens_ref(self, obs):
    """Lidar.zone_tokens_vec before the 2026-11-20 change (verbatim copy, reference only)."""
    Z = self.zone_tokens
    feat, flag = np.zeros((Z, ZONE_FEATURES), dtype=np.float32), np.zeros(Z, dtype=np.float32)
    g = obs[GEOM_KEY]
    goal = np.asarray(g["goal"], dtype=np.float64).reshape(2)
    gd = float(np.linalg.norm(goal))
    rows = []
    for poly in g["zones"]:
        v = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        p, q = v, np.roll(v, -1, axis=0)
        d = q - p
        s = np.clip(-(p * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0.0, 1.0)
        c = p + s[:, None] * d
        dist = np.linalg.norm(c, axis=1)
        j = int(dist.argmin())
        inside = int(np.isfinite(ray_segments(np.array([[1.0, 0.0]]), p, q)).sum()) % 2 == 1
        cen = v.mean(axis=0)
        ang = np.arctan2(v[:, 1], v[:, 0])
        rel = (ang - np.arctan2(cen[1], cen[0]) + np.pi) % (2.0 * np.pi) - np.pi
        il, ir = int(rel.argmin()), int(rel.argmax())
        blocks = inside or (gd > 1e-9 and bool((ray_segments((goal / gd)[None], p, q)[0] <= gd).any()))
        n = DIST_NORM_KM
        rows.append((dist[j], [c[j, 0] / n, c[j, 1] / n, dist[j] / n,
                               np.cos(ang[il]), np.sin(ang[il]), np.linalg.norm(v[il]) / n,
                               np.cos(ang[ir]), np.sin(ang[ir]), np.linalg.norm(v[ir]) / n,
                               float(blocks), float(inside)]))
    order = np.argsort([r[0] for r in rows], kind="stable")[:Z] if rows else []
    for i, j in enumerate(order):
        feat[i], flag[i] = rows[j][1], 1.0
    bearing = goal / gd if gd > 1e-9 else np.array([1.0, 0.0])
    goal_row = np.array([goal[0] / DIST_NORM_KM, goal[1] / DIST_NORM_KM, gd / DIST_NORM_KM, bearing[0], bearing[1],
                         float(g.get("time_left", 1.0))], dtype=np.float32)
    return np.concatenate([feat.reshape(-1), flag, goal_row])


def same(a, b, path, diffs):
    """Exact equality of nested dicts, lists, arrays and scalars (NaN equals NaN). Appends differences."""
    if isinstance(a, dict):
        if set(a) != set(b):
            diffs.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
            return
        for k in a:
            same(a[k], b[k], f"{path}.{k}", diffs)
    elif isinstance(a, (list, tuple)) and not (len(a) and np.isscalar(a[0])):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]", diffs)
    else:
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
            if not (x.shape == y.shape and x.dtype.kind == "f" and np.array_equal(x, y, equal_nan=True)
                    and x.tobytes() == y.tobytes()):
                diffs.append(f"{path}: {x.ravel()[:4]} vs {y.ravel()[:4]}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/sa_v2m_s0_c2.pt", help="Only for the reward, v2 and token settings.")
    p.add_argument("--n-agents", type=int, default=10)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=7, help="First scenario seed.")
    p.add_argument("--max-steps", type=int, default=400)
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    a = torch.load(args.model, map_location="cpu", weights_only=False)["args"]
    rew = dict(REWARD_DEFAULTS)
    rew["goal_distance_penalty"] = 0.0
    rew.update({k: a[k] for k in rew if k in a})
    lidar = Lidar(n_rays=a["n_rays"], fov_deg=a["fov_deg"], max_range_km=a["max_range_km"],
                  zone_flag=bool(a.get("zone_flag", 0)), speed=bool(a.get("lidar_speed", 0)), **token_cfg(a))
    if not lidar.zone_tokens:
        lidar.zone_tokens = 5                          # check the tokens also for a checkpoint without them
    with contextlib.redirect_stdout(io.StringIO()):    # BlueSky start-up text
        ref = FlightEnv("multi", args.n_agents, v2=v2_kwargs(a), **rew)
        fast = FlightEnv("multi", args.n_agents, v2=v2_kwargs(a), fast_metrics=True, **rew)
    diffs, n_dec, n_tok = [], 0, 0
    t = dict(step_ref=0.0, step_fast=0.0, tok_ref=0.0, tok_fast=0.0)
    events = dict(intrusion_events=0, restricted_area_events=0, sector_exit_events=0, waypoint_reached=0)

    def fly(env, seed, mode, e, key, tokens):
        """One full episode (BlueSky has one global simulator, so the two versions fly one after the other).
        Returns the reset observation and the list of step outputs."""
        nonlocal n_tok
        rng = np.random.default_rng(1000 + e)
        obs = env.reset(seed=seed)
        rec, steps = [obs], 0
        while env.agents and steps < args.max_steps:
            if tokens:
                for ag in env.agents:
                    t0 = time.perf_counter()
                    tr = zone_tokens_ref(lidar, obs[ag])
                    t1 = time.perf_counter()
                    tf = lidar.zone_tokens_vec(obs[ag])
                    t2 = time.perf_counter()
                    t["tok_ref"] += t1 - t0
                    t["tok_fast"] += t2 - t1
                    n_tok += 1
                    if tr.dtype != tf.dtype or tr.tobytes() != tf.tobytes():
                        diffs.append(f"ep{e}.step{steps}.{ag}: zone tokens differ, max abs {np.abs(tr - tf).max()}")
            if mode == "zero":
                acts = {ag: np.zeros(2, dtype=np.float32) for ag in env.agents}
            elif mode == "side":                       # constant turn away from the goal: sector exits
                acts = {ag: np.array([1.0, -1.0], dtype=np.float32) for ag in env.agents}
            else:
                acts = {ag: rng.uniform(-1.0, 1.0, 2).astype(np.float32) for ag in env.agents}
            t0 = time.perf_counter()
            out = env.step(acts)
            t[key] += time.perf_counter() - t0
            rec.append(out)
            obs = out[0]
            steps += 1
        return rec

    for e in range(args.episodes):
        seed, mode = args.seed + e, ("zero", "random", "side")[e % 3]
        rec_r = fly(ref, seed, mode, e, "step_ref", tokens=True)
        rec_f = fly(fast, seed, mode, e, "step_fast", tokens=False)
        if len(rec_r) != len(rec_f):
            diffs.append(f"ep{e}: {len(rec_r) - 1} vs {len(rec_f) - 1} decisions")
        for s, (x, y) in enumerate(zip(rec_r, rec_f)):
            same(x, y, f"ep{e}.{'reset' if s == 0 else f'step{s - 1}'}", diffs)
        for out in rec_r[1:]:
            for ag, done in out[2].items():
                if done or out[3][ag]:
                    for k in events:
                        events[k] += int(out[4][ag][k])
        n_dec += len(rec_r) - 1
        print(f"episode {e} seed {seed} {mode}: {len(rec_r) - 1} decisions, differences so far {len(diffs)}",
              flush=True)
        if len(diffs) > 20:
            break
    ref.close()
    fast.close()
    res = dict(host=socket.gethostname(), cpu=platform.processor() or platform.machine(), numpy=np.__version__,
               episodes=args.episodes, first_seed=args.seed, n_agents=args.n_agents, decisions=n_dec,
               token_calls=n_tok, events_in_finished_flights=events, n_differences=len(diffs), differences=diffs[:20],
               ms_per_decision=dict(env_ref=1000 * t["step_ref"] / max(n_dec, 1),
                                    env_fast=1000 * t["step_fast"] / max(n_dec, 1)),
               ms_per_token_call=dict(ref=1000 * t["tok_ref"] / max(n_tok, 1), fast=1000 * t["tok_fast"] / max(n_tok, 1)),
               status="EXPLORATORY")
    out = Path("results") / f"check_fast_path_{res['host']}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("BIT-IDENTICAL" if not diffs else "DIFFERENCES FOUND", "->", out)
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
