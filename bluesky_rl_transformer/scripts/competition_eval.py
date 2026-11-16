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

Planner (2026-11-13, rl_tf/world_model.py): --world-model models/wm_NAME.pt --plan-level 1|2 flies the model with
the CEM planner in the latent world model. Level 1 = safety filter (search only when the predicted risk of the
policy rollout is >= --plan-threshold), level 2 = search at every step with the policy as prior and its critic as
the value after the horizon. The world model must sit on the same encoder weights as --model (checked).
Check: --plan-level 1 --plan-threshold 2 never searches and must give the same metrics as no planner.
Output name: NAME_planL (results/NAME_planL_competition_KIND.json), with the planner settings and the share of
decisions where the planner searched.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
from rl_tf.envs import FlightEnv, REWARD_DEFAULTS, METRIC_KEYS, START_KEYS, rollout_episode
from rl_tf.lidar import Lidar, token_cfg
from rl_tf.common import setup_torch, write_json, wandb_init, add_wandb_args, summary_table, SUMMARY_ROWS

SEED, N_EPISODES, N_AGENTS_MA = 42, 1000, 10             # official values


def _job(job):
    """Worker: the episodes [a, b) of the stream of `seed`, with the model (or the zero action if model is None)."""
    import contextlib
    import io
    model, kind, a, b, seed, device = job[:6]
    plan = job[6] if len(job) > 6 else None
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
        if plan is not None:
            from rl_tf.world_model import load_world_model, same_weights, Planner, PlannerFn
            wm, _ = load_world_model(plan["world_model"], device)
            if not same_weights(wm.encoder, policy.encoder):
                raise ValueError(f"{plan['world_model']}: its encoder weights differ from those of {model}")
            if c.get("shaping", "off") != "off":
                raise ValueError("the planner score has no shaping term; the policy was trained with shaping")
            cfg = {k: v for k, v in plan.items() if k != "world_model"}
            fn = PlannerFn(Planner(wm, policy, gamma=c["gamma"], reward_scale=c["reward_scale"], device=device,
                                   **{**cfg, "seed": cfg["seed"] + a}), device, lidar)
    with contextlib.redirect_stdout(io.StringIO()):
        env = FlightEnv("single" if kind == "sa" else "multi", N_AGENTS_MA, **rew)
        for i in range(a):                                   # move the random stream to episode a, no flight
            env.env.reset(seed=seed if i == 0 else None)
        rows = []
        for i in range(a, b):
            if hasattr(fn, "reset"):
                fn.reset()
            ep, _, _ = rollout_episode(env, fn, seed if i == 0 else None, lidar, history)
            extra = fn.stats() if hasattr(fn, "stats") else {}
            rows += [{"episode_index": i, "aircraft": ac, **r, **extra} for ac, r in ep.items()]
        env.close()
    return rows


def run(model, kind, episodes, seed, workers, device="cpu", plan=None):
    """All flights of the protocol, in episode order. One row per flight (per aircraft and episode)."""
    # Single agent: one process. The check with several processes failed there (2026-11-12, cause not found), and
    # one process follows the official loop step by step.
    w = 1 if kind == "sa" else max(1, min(workers, episodes))
    edges = np.linspace(0, episodes, w + 1).astype(int)
    jobs = [(model, kind, int(edges[i]), int(edges[i + 1]), seed, device) + ((plan,) if plan is not None else ())
            for i in range(w)]
    if w == 1:
        parts = [_job(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(w) as pool:
            parts = pool.map(_job, jobs)
    return [r for p in parts for r in p]


def stats(rows):
    keys = METRIC_KEYS + [k for k in START_KEYS if rows and all(k in r for r in rows)] + ["ep_return"]
    return ({k: float(np.mean([r[k] for r in rows])) for k in keys}, {k: float(np.std([r[k] for r in rows])) for k in keys})


def plan_stats(rows):
    """Planner use over all episodes (each row of one episode carries the counts of that episode)."""
    ep = {}
    for r in rows:
        ep[r["episode_index"]] = (r.get("plan_decisions", 0), r.get("plan_active", 0), r.get("plan_seconds", 0.0))
    d, a, s = (sum(v[i] for v in ep.values()) for i in range(3))
    return {"decisions": int(d), "searched": int(a), "searched_share": a / max(d, 1),
            "seconds_per_decision": s / max(d, 1)}


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
    p.add_argument("--world-model", default=None, help="Latent world model (scripts.train_world_model) for the planner.")
    p.add_argument("--plan-level", type=int, choices=[1, 2], default=None,
                   help="1: safety filter (search only at high predicted risk); 2: search at every step.")
    p.add_argument("--plan-samples", type=int, default=64)
    p.add_argument("--plan-rounds", type=int, default=3)
    p.add_argument("--plan-horizon", type=int, default=6, help="Planned steps (10 s each).")
    p.add_argument("--plan-elites", type=int, default=8)
    p.add_argument("--plan-threshold", type=float, default=0.5, help="Level 1: predicted risk that starts a search.")
    p.add_argument("--plan-turn-cost", type=float, default=0.0,
                   help="Cost per unit of turn change in the plan score (PPO return units). 0 = no turn term.")
    p.add_argument("--plan-init-std", type=float, default=0.5, help="Least start spread of the search.")
    p.add_argument("--plan-seed", type=int, default=0)
    p.add_argument("--plan-value", choices=["auto", "critic", "head"], default="auto",
                   help="Value after the horizon: the policy critic, or the value head of the world model. auto: critic "
                        "for a single-agent policy (its critic sees one aircraft), head for a multi-agent policy.")
    p.add_argument("--plan-order", choices=["independent", "sequential"], default="independent",
                   help="Multi agent: all aircraft search at once against the policy rollouts of the others, or one "
                        "after another in callsign order (a later aircraft sees the chosen plans of the earlier ones).")
    add_wandb_args(p)
    args = p.parse_args(argv)
    if args.self_check:
        ok = self_check(args.env or "ma", args.episodes, args.workers)
        raise SystemExit(0 if ok else 1)
    if args.model is None:
        p.error("give --model (or --self-check)")
    if (args.world_model is None) != (args.plan_level is None):
        p.error("--world-model and --plan-level go together")
    plan = None
    if args.plan_level is not None:
        plan = {"world_model": args.world_model, "level": args.plan_level, "samples": args.plan_samples,
                "rounds": args.plan_rounds, "horizon": args.plan_horizon, "elites": args.plan_elites,
                "threshold": args.plan_threshold, "turn_cost": args.plan_turn_cost, "init_std": args.plan_init_std,
                "seed": args.plan_seed, "order": args.plan_order}
    import torch
    c = torch.load(args.model, map_location="cpu", weights_only=False)
    if plan is not None:
        plan["value"] = (args.plan_value if args.plan_value != "auto"
                         else ("critic" if c["args"]["setting"] == "single" else "head"))
    kind = args.env or ("sa" if c["args"]["setting"] == "single" else "ma")
    suffix = (f"_plan{args.plan_level}" + ("_seq" if args.plan_order == "sequential" else "")) if plan else ""
    name, out = args.name or Path(args.model).stem + suffix, Path(args.out_dir)
    official = args.episodes == N_EPISODES and args.seed == SEED
    if not official:
        print(f"NOTE: {args.episodes} episodes of seed {args.seed}; the official protocol is {N_EPISODES} episodes of seed {SEED}.")

    t0 = time.time()
    cache = out / "results" / f"competition_straight_{kind}_{args.episodes}_seed{args.seed}.json"
    straight = json.loads(cache.read_text()) if cache.exists() else None
    if straight is not None and kind == "sa" and not all(k in straight["mean"] for k in START_KEYS):
        straight = None                                # cache from before START_KEYS (2026-11-13): compute again
    if straight is not None:
        print(f"Read {cache}")
    else:
        s_rows = run(None, kind, args.episodes, args.seed, args.workers)
        s_mean, s_std = stats(s_rows)
        straight = {"policy": "fly straight (zero action)", "kind": kind, "episodes": args.episodes, "seed": args.seed,
                    "n_flights": len(s_rows), "mean": s_mean, "std": s_std,
                    "note": "ep_return uses the default reward weights of this folder"}
        if all("start_intrusions" in r for r in s_rows):
            straight["n_flights_start_in_intrusion"] = sum(r["start_intrusions"] > 0 for r in s_rows)
        write_json(cache, straight)
    rows = run(args.model, kind, args.episodes, args.seed, args.workers, args.device, plan)
    mean, std = stats(rows)
    wall = time.time() - t0
    unit = "episodes" if kind == "sa" else "agent episodes"
    print(f"\n{name}: competition protocol {kind}, {args.episodes} episodes of seed {args.seed}, {len(rows)} {unit}, "
          f"{'OFFICIAL count' if official else 'NOT the official count'}, {wall:.0f} s")
    wb = wandb_init(args.wandb, args.wandb_project, name)
    cols, table = summary_table(straight["mean"], mean, c["args"].get("seed", "?"), wb, key="competition/summary")
    print("std over the flights: " + ", ".join(f"{label} {std[k]:.2f}" for k, label in SUMMARY_ROWS if k in std))
    start = {}
    if all("start_intrusions" in r for r in rows):     # single agent: flights that start in intrusion
        clean = [r for r in rows if r["start_intrusions"] == 0]
        start = {"n_flights_start_in_intrusion": len(rows) - len(clean),
                 "mean_flights_without_start_intrusion": stats(clean)[0] if clean else {},
                 "note": "start_intrusions and *_no_start are not official metrics: they leave out intrusions with "
                         "aircraft that are within 5 NM at the reset sample, until that aircraft goes out of 5 NM"}
        print(f"flights that start in intrusion: {start['n_flights_start_in_intrusion']} of {len(rows)}")
    if plan:
        ps = plan_stats(rows)
        print(f"planner level {plan['level']}: searched in {ps['searched']} of {ps['decisions']} decisions "
              f"({100 * ps['searched_share']:.1f} %), {1000 * ps['seconds_per_decision']:.1f} ms per decision")
    write_json(out / "results" / f"{name}_competition_{kind}.json", {
        "status": "EXPLORATORY", "name": name, "model": args.model, "kind": kind, "episodes": args.episodes,
        "seed": args.seed, "official_protocol": official, "n_flights": len(rows), "n_params": c["n_params"],
        "deterministic": True, "mean": mean, "std": std, "std_note": "population std over the flights (one model seed)",
        "fly_straight": {k: straight[k] for k in ["mean", "std", "n_flights", "n_flights_start_in_intrusion"]
                         if k in straight}, "table": {"columns": cols, "rows": table},
        "start_intrusion": start, "wall_seconds": wall, "plan": plan,
        "plan_stats": plan_stats(rows) if plan else None, "rows": rows})
    if wb is not None:
        wb.summary.update({f"competition/{k}_{s}": v for s, d in [("mean", mean), ("std", std)] for k, v in d.items()})
        wb.summary.update({"competition/episodes": args.episodes, "competition/seed": args.seed})
        wb.finish()


if __name__ == "__main__":
    main()
