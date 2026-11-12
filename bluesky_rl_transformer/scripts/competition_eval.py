"""Competition protocol for a trained model: the first 1000 episodes of seed 42 (repo root scripts/evaluate_competition.py).

Protocol of the official script: one environment, reset(seed=42) for episode 0, then reset() with no seed, so the
scenario sequence continues from the same random stream. The scenario of episode i does not depend on the actions
(multi agent; test: scripts.competition_eval --self-check --env ma). So a worker process gets the episodes [a, b): it does `a` resets
with no flight to move the stream forward, then it flies its episodes. The result is the same as one process.
Single agent: always one process (the same test failed there with 3 processes).

Output: results/NAME_competition_KIND.json (mean, std, one row per flight) and a table
metric | fly straight | seed N (our Transformer), printed and sent to W&B (key competition/summary).
The fly-straight values depend only on KIND, the episode count and the seed. They are kept in
results/competition_straight_KIND_N_seedS.json and read from there by later runs.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
from rl_tf.envs import FlightEnv, REWARD_DEFAULTS, METRIC_KEYS, rollout_episode
from rl_tf.lidar import Lidar, token_cfg
from rl_tf.common import setup_torch, write_json, wandb_init, add_wandb_args, summary_table, SUMMARY_ROWS

SEED, N_EPISODES, N_AGENTS_MA = 42, 1000, 10             # official values


def _job(job):
    """Worker: the episodes [a, b) of the stream of `seed`, with the model (or the zero action if model is None)."""
    import contextlib
    import io
    model, kind, a, b, seed, device = job
    setup_torch(0)
    rew, lidar, history = dict(REWARD_DEFAULTS), Lidar(), 1
    if model is None:
        fn = lambda raw, built: {k: np.zeros(2, dtype=np.float32) for k in raw}
    else:
        from rl_tf.networks import load_policy
        from rl_tf.common import policy_fn_from
        policy, ck = load_policy(model, device)
        c = ck["args"]
        rew = {**REWARD_DEFAULTS, "goal_distance_penalty": 0.0, **{k: c[k] for k in REWARD_DEFAULTS if k in c}}
        lidar = Lidar(n_rays=c["n_rays"], fov_deg=c["fov_deg"], max_range_km=c["max_range_km"],
                      zone_flag=bool(c.get("zone_flag", 0)), speed=bool(c.get("lidar_speed", 0)), **token_cfg(c))
        history = c["history"]
        fn = policy_fn_from(policy, device, deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        env = FlightEnv("single" if kind == "sa" else "multi", N_AGENTS_MA, **rew)
        for i in range(a):                                   # move the random stream to episode a, no flight
            env.env.reset(seed=seed if i == 0 else None)
        rows = []
        for i in range(a, b):
            ep, _, _ = rollout_episode(env, fn, seed if i == 0 else None, lidar, history)
            rows += [{"episode_index": i, "aircraft": ac, **r} for ac, r in ep.items()]
        env.close()
    return rows


def run(model, kind, episodes, seed, workers, device="cpu"):
    """All flights of the protocol, in episode order. One row per flight (per aircraft and episode)."""
    # Single agent: one process. The check with several processes failed there (2026-11-12, cause not found), and
    # one process follows the official loop step by step.
    w = 1 if kind == "sa" else max(1, min(workers, episodes))
    edges = np.linspace(0, episodes, w + 1).astype(int)
    jobs = [(model, kind, int(edges[i]), int(edges[i + 1]), seed, device) for i in range(w)]
    if w == 1:
        parts = [_job(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(w) as pool:
            parts = pool.map(_job, jobs)
    return [r for p in parts for r in p]


def stats(rows):
    keys = METRIC_KEYS + ["ep_return"]
    return ({k: float(np.mean([r[k] for r in rows])) for k in keys}, {k: float(np.std([r[k] for r in rows])) for k in keys})


def self_check(kind, episodes, workers):
    """The fly-straight metrics of this script (several workers) against the official loop (one process)."""
    import contextlib
    import io
    ours = run(None, kind, episodes, SEED, workers)
    with contextlib.redirect_stdout(io.StringIO()):
        env = FlightEnv("single" if kind == "sa" else "multi", N_AGENTS_MA).env      # the bare competition environment
        ref = []
        for ep in range(episodes):
            if kind == "sa":
                env.reset(seed=SEED if ep == 0 else None)
                te = tr = False
                while not (te or tr):
                    _, _, te, tr, info = env.step(np.zeros(2, dtype=np.float32))
                ref.append((ep, "own", info))
            else:
                obs, _ = env.reset(seed=SEED if ep == 0 else None)
                while env.agents:
                    obs, _, te, tr, infos = env.step({a: np.zeros(2, dtype=np.float32) for a in env.agents})
                    ref += [(ep, a, infos[a]) for a in infos if te[a] or tr[a]]
        env.close()
    # compare flight by flight (episode, aircraft): aircraft that end at the same step can be listed in another order
    a = {(r["episode_index"], r["aircraft"]): [float(r[k]) for k in METRIC_KEYS] for r in ours}
    b = {(ep, ac): [float(info[k]) for k in METRIC_KEYS] for ep, ac, info in ref}
    same = len(ours) == len(ref) and a == b
    print(f"self-check {kind}, {episodes} episodes, {workers} workers: {len(ours)} flights against {len(ref)} official flights; "
          f"all 8 metrics of all flights equal: {same}")
    return same


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None)
    p.add_argument("--env", choices=["sa", "ma"], default=None, help="Default: the setting of the model.")
    p.add_argument("--episodes", type=int, default=N_EPISODES, help=f"Official: {N_EPISODES}.")
    p.add_argument("--seed", type=int, default=SEED, help=f"Official: {SEED}.")
    p.add_argument("--workers", type=int, default=6, help="Processes. Each one uses one thread.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--name", default=None, help="Default: model file name (the figures go to the W&B run of the model).")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--self-check", action="store_true", help="Compare fly straight with the official loop, then stop.")
    add_wandb_args(p)
    args = p.parse_args(argv)
    if args.self_check:
        ok = self_check(args.env or "ma", args.episodes, args.workers)
        raise SystemExit(0 if ok else 1)
    if args.model is None:
        p.error("give --model (or --self-check)")
    import torch
    c = torch.load(args.model, map_location="cpu", weights_only=False)
    kind = args.env or ("sa" if c["args"]["setting"] == "single" else "ma")
    name, out = args.name or Path(args.model).stem, Path(args.out_dir)
    official = args.episodes == N_EPISODES and args.seed == SEED
    if not official:
        print(f"NOTE: {args.episodes} episodes of seed {args.seed}; the official protocol is {N_EPISODES} episodes of seed {SEED}.")

    t0 = time.time()
    cache = out / "results" / f"competition_straight_{kind}_{args.episodes}_seed{args.seed}.json"
    if cache.exists():
        straight = json.loads(cache.read_text())
        print(f"Read {cache}")
    else:
        s_rows = run(None, kind, args.episodes, args.seed, args.workers)
        s_mean, s_std = stats(s_rows)
        straight = {"policy": "fly straight (zero action)", "kind": kind, "episodes": args.episodes, "seed": args.seed,
                    "n_flights": len(s_rows), "mean": s_mean, "std": s_std,
                    "note": "ep_return uses the default reward weights of this folder"}
        write_json(cache, straight)
    rows = run(args.model, kind, args.episodes, args.seed, args.workers, args.device)
    mean, std = stats(rows)
    wall = time.time() - t0
    unit = "episodes" if kind == "sa" else "agent episodes"
    print(f"\n{name}: competition protocol {kind}, {args.episodes} episodes of seed {args.seed}, {len(rows)} {unit}, "
          f"{'OFFICIAL count' if official else 'NOT the official count'}, {wall:.0f} s")
    wb = wandb_init(args.wandb, args.wandb_project, name)
    cols, table = summary_table(straight["mean"], mean, c["args"].get("seed", "?"), wb, key="competition/summary")
    print("std over the flights: " + ", ".join(f"{label} {std[k]:.2f}" for k, label in SUMMARY_ROWS))
    write_json(out / "results" / f"{name}_competition_{kind}.json", {
        "status": "EXPLORATORY", "name": name, "model": args.model, "kind": kind, "episodes": args.episodes,
        "seed": args.seed, "official_protocol": official, "n_flights": len(rows), "n_params": c["n_params"],
        "deterministic": True, "mean": mean, "std": std, "std_note": "population std over the flights (one model seed)",
        "fly_straight": {k: straight[k] for k in ["mean", "std", "n_flights"]}, "table": {"columns": cols, "rows": table},
        "wall_seconds": wall, "rows": rows})
    if wb is not None:
        wb.summary.update({f"competition/{k}_{s}": v for s, d in [("mean", mean), ("std", std)] for k, v in d.items()})
        wb.summary.update({"competition/episodes": args.episodes, "competition/seed": args.seed})
        wb.finish()


if __name__ == "__main__":
    main()
