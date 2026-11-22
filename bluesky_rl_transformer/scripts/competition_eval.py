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

Test-time options (2026-11-19), no training, permitted by docs/competition/COMPETITION.md:
  --action-frequency N --d-heading D --d-speed S   decision rate and action scale of the environment
                                                   (same turn rate as trained: N 5, D 22.5, S 3.333);
  --zone-margin-km M    the policy sees each restricted area grown by M km (the metrics use the real areas);
  --ensemble PATH --ensemble-mode mean|zone [--switch-km K]   a second model (rl_tf/ensemble.py).
The output name gets the tags _afN _dhD _dsS _zmM _ensMODE_MODEL[K]. Without options the jobs are the same as before.
Check of the ensemble: --ensemble with the same model, mode mean, must give the same flights as the model alone.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
from rl_tf.envs import FlightEnv, REWARD_DEFAULTS, METRIC_KEYS, START_KEYS, rollout_episode, v2_kwargs
from rl_tf.lidar import Lidar, token_cfg
from rl_tf.common import setup_torch, write_json, wandb_init, add_wandb_args, summary_table, SUMMARY_ROWS
from rl_tf.heading_mask import add_mask_args, mask_overrides, mask_settings

SEED, N_EPISODES, N_AGENTS_MA = 42, 1000, 10             # official values


def _job(job):
    """Worker: the episodes [a, b) of the stream of `seed`, with the model (or the zero action if model is None)."""
    import contextlib
    import io
    model, kind, a, b, seed, device = job[:6]
    plan = job[6] if len(job) > 6 else None
    opts = (job[7] if len(job) > 7 else None) or {}           # test-time options (2026-11-19), {} = none
    setup_torch(0)
    rew, lidar, history, v2 = dict(REWARD_DEFAULTS), Lidar(), 1, None
    if model is None:
        fn = lambda raw, built: {k: np.zeros(2, dtype=np.float32) for k in raw}
    else:
        policy, policy_fn, rew, lidar, history, c = _load_member(model, device)
        fn = policy_fn
        v2 = v2_kwargs({**c, **(opts.get("mask") or {})})    # action mapping of the training (2026-11-19); heading
                                                             # mask: checkpoint values, flags override (2026-11-21)
        if opts.get("ensemble"):
            from rl_tf.ensemble import EnsembleFn
            second = _load_member(opts["ensemble"], device)
            v2b = v2_kwargs(second[5])
            if (v2b["action_map"], v2b["goal_offset_deg"]) != (v2["action_map"], v2["goal_offset_deg"]):
                raise ValueError(f"ensemble: {opts['ensemble']} has another action mapping than {model}")
            fn = EnsembleFn([(policy_fn, lidar, history), (second[1], second[3], second[4])],
                            mode=opts["ensemble_mode"], switch_km=opts["switch_km"])
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
    env_kw = opts.get("env_kw") or {}
    af = int(env_kw.get("action_frequency", 10))
    max_steps = max(400, int(np.ceil(3100 / af)) + 1)          # 3000 s budget at af seconds per decision
    with contextlib.redirect_stdout(io.StringIO()):
        env = FlightEnv("single" if kind == "sa" else "multi", N_AGENTS_MA, env_kw=env_kw,
                        zone_margin_km=opts.get("zone_margin_km", 0.0), v2=v2, **rew)
        env.extra_info = True                                # start geometry in the rows (2026-11-21), not scored
        for i in range(a):                                   # move the random stream to episode a, no flight
            env.env.reset(seed=seed if i == 0 else None)
        rows = []
        for i in range(a, b):
            if hasattr(fn, "reset"):
                fn.reset()
            ep, _, _ = rollout_episode(env, fn, seed if i == 0 else None, lidar, history, max_steps=max_steps)
            extra = fn.stats() if hasattr(fn, "stats") else {}
            rows += [{"episode_index": i, "aircraft": ac, **r, **extra} for ac, r in ep.items()]
        env.close()
    return rows


def _load_member(model, device):
    """A saved policy and its settings: (policy, deterministic policy_fn, reward weights, Lidar, history, args)."""
    from rl_tf.networks import load_policy
    from rl_tf.common import policy_fn_from
    policy, ck = load_policy(model, device)
    c = ck["args"]
    rew = {**REWARD_DEFAULTS, "goal_distance_penalty": 0.0, **{k: c[k] for k in REWARD_DEFAULTS if k in c}}
    lidar = Lidar(n_rays=c["n_rays"], fov_deg=c["fov_deg"], max_range_km=c["max_range_km"],
                  zone_flag=bool(c.get("zone_flag", 0)), speed=bool(c.get("lidar_speed", 0)), **token_cfg(c))
    return policy, policy_fn_from(policy, device, deterministic=True), rew, lidar, c["history"], c


def run(model, kind, episodes, seed, workers, device="cpu", plan=None, opts=None):
    """All flights of the protocol, in episode order. One row per flight (per aircraft and episode).
    opts: test-time options (env_kw, zone_margin_km, ensemble, ensemble_mode, switch_km); None = none."""
    # Single agent: one process. The check with several processes failed there (2026-11-12, cause not found), and
    # one process follows the official loop step by step.
    w = 1 if kind == "sa" else max(1, min(workers, episodes))
    edges = np.linspace(0, episodes, w + 1).astype(int)
    jobs = [(model, kind, int(edges[i]), int(edges[i + 1]), seed, device) + ((plan, opts) if (plan or opts) else ())
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


def mask_stats(rows):
    """Heading mask over all flights (rows with the mask keys): decisions, projected share, mean angle, fallbacks."""
    rows = [r for r in rows if "mask_decisions" in r]
    if not rows:
        return None
    dec, proj = sum(r["mask_decisions"] for r in rows), sum(r["mask_projected"] for r in rows)
    ang = sum(r["mask_mean_angle_deg"] * r["mask_projected"] for r in rows)
    return {"decisions": int(dec), "projected": int(proj), "projected_share": proj / max(dec, 1),
            "mean_projection_deg": ang / max(proj, 1), "fallbacks": int(sum(r["mask_fallbacks"] for r in rows)),
            "fallbacks_per_flight": float(np.mean([r["mask_fallbacks"] for r in rows]))}


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
    # test-time options (2026-11-19), no training; all permitted by docs/competition/COMPETITION.md
    p.add_argument("--action-frequency", type=int, default=None,
                   help="Seconds of simulation per decision (environment default 10).")
    p.add_argument("--d-heading", type=float, default=None, help="Degrees of turn per unit of action (default 45).")
    p.add_argument("--d-speed", type=float, default=None, help="Knots of speed change per unit of action (default 20/3).")
    p.add_argument("--zone-margin-km", type=float, default=0.0,
                   help="The policy sees each restricted area grown by this margin (km). Scored metrics: real areas.")
    p.add_argument("--ensemble", default=None, help="Second model (models/NAME.pt) for a test-time ensemble.")
    p.add_argument("--ensemble-mode", choices=["mean", "zone"], default="mean",
                   help="mean: mean action of the two models; zone: the second model near a restricted area.")
    p.add_argument("--switch-km", type=float, default=20.0,
                   help="Mode zone: the second model flies when an area edge is within this distance (km).")
    add_mask_args(p, from_checkpoint=True)               # heading mask (2026-11-21)
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
    env_kw = {k: v for k, v in [("action_frequency", args.action_frequency), ("d_heading", args.d_heading),
                                ("d_speed", args.d_speed)] if v is not None}
    mo = mask_overrides(args)
    opts = {"env_kw": env_kw, "zone_margin_km": args.zone_margin_km, "ensemble": args.ensemble,
            "ensemble_mode": args.ensemble_mode, "switch_km": args.switch_km, "mask": mo}
    if plan and (env_kw or args.zone_margin_km > 0 or args.ensemble or mo):
        p.error("the planner does not combine with the test-time options")
    tags = "".join(t for t in [
        f"_af{args.action_frequency}" if args.action_frequency is not None else "",
        f"_dh{args.d_heading:g}" if args.d_heading is not None else "",
        f"_ds{args.d_speed:g}" if args.d_speed is not None else "",
        f"_zm{args.zone_margin_km:g}" if args.zone_margin_km > 0 else "",
        f"_ens{args.ensemble_mode}_{Path(args.ensemble).stem}" if args.ensemble else "",
        f"{args.switch_km:g}" if args.ensemble and args.ensemble_mode == "zone" else "",
        f"_mask{mo['heading_mask']}" if "heading_mask" in mo else "",
        f"_md{mo['mask_dist_km']:g}" if "mask_dist_km" in mo else "",
        f"_ms{mo['mask_step_deg']:g}" if "mask_step_deg" in mo else "",
        f"_mm{mo['mask_margin_km']:g}" if "mask_margin_km" in mo else "",
        f"_msec{mo['mask_sector']}" if "mask_sector" in mo else "",
        f"_m{mo['mask_mode']}" if "mask_mode" in mo else "",
        f"_mi{mo['mask_intruders']}" if "mask_intruders" in mo else "",
        f"_min{mo['mask_intr_nm']:g}" if "mask_intr_nm" in mo else "",
        f"_mih{mo['mask_intr_horizon_s']:g}" if "mask_intr_horizon_s" in mo else "",
        f"_mr{mo['mask_route']}" if "mask_route" in mo else "",
        f"_mrk{mo['mask_route_km']:g}" if "mask_route_km" in mo else ""])
    if not tags:
        opts = None                                      # no option: the same jobs as before 2026-11-19
    import torch
    c = torch.load(args.model, map_location="cpu", weights_only=False)
    if plan is not None:
        plan["value"] = (args.plan_value if args.plan_value != "auto"
                         else ("critic" if c["args"]["setting"] == "single" else "head"))
    kind = args.env or ("sa" if c["args"]["setting"] == "single" else "ma")
    suffix = (f"_plan{args.plan_level}" + ("_seq" if args.plan_order == "sequential" else "")) if plan else ""
    name, out = args.name or Path(args.model).stem + suffix + tags, Path(args.out_dir)
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
    rows = run(args.model, kind, args.episodes, args.seed, args.workers, args.device, plan, opts)
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
    if opts:
        print(f"test-time options: {opts}")
    if args.ensemble and args.ensemble_mode == "zone":
        n2, nd = sum(r.get("ens_second_model", 0) for r in rows), sum(r.get("ens_decisions", 0) for r in rows)
        print(f"ensemble zone: second model in {n2} of {nd} decisions ({100 * n2 / max(nd, 1):.1f} %)")
    hm = mask_settings(v2_kwargs({**c["args"], **mo}))
    ms = mask_stats(rows)
    if ms:
        print(f"heading mask {hm}: projected {ms['projected']} of {ms['decisions']} decisions "
              f"({100 * ms['projected_share']:.1f} %), mean projection {ms['mean_projection_deg']:.1f} deg, "
              f"all-blocked fallbacks {ms['fallbacks']}")
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
        "start_intrusion": start, "wall_seconds": wall, "plan": plan, "test_time": opts,
        "plan_stats": plan_stats(rows) if plan else None, "heading_mask": hm, "mask_stats": ms, "rows": rows})
    if wb is not None:
        wb.summary.update({f"competition/{k}_{s}": v for s, d in [("mean", mean), ("std", std)] for k, v in d.items()})
        wb.summary.update({"competition/episodes": args.episodes, "competition/seed": args.seed})
        wb.finish()


if __name__ == "__main__":
    main()
