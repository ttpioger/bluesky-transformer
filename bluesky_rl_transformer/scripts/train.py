"""PPO (single agent) and MAPPO (multi agent) with a Transformer over lidar ray tokens.

One code path for both settings. --workers environment processes run in parallel (one thread each).
All PPO/MAPPO choices follow avbiswas/navigation-mappo-rl (commit 309ac1d); only the encoder and the
reward differ (see README).
--reset-seed N gives the same scenario in every episode (overfit test); -1 gives random scenarios.
This script is a training run: start it through ~/bin/notify_run.sh.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from rl_tf.envs import (VecEnv, FlightEnv, add_reward_args, reward_kwargs, obs_dims, rollout_episode, turn_stats,
                        lidar_cfg_from_args, METRIC_KEYS, START_KEYS, V2_DEFAULTS, v2_kwargs, cat_prob_at)
from rl_tf.lidar import Lidar
from rl_tf.heading_mask import add_mask_args
from rl_tf.networks import build_policy, save_checkpoint, n_params, load_encoder, AttentionCritic
from rl_tf.common import (setup_torch, compute_gae, pass_criterion, window_stats, write_json, policy_fn_from, wandb_upload,
                          wandb_init, add_wandb_args, parse_curriculum, curriculum_stage, parse_lagrangian,
                          lagrange_step, lagrangian_advantage, WINDOW)

EVAL_SEED0 = 900_000_000   # held-out scenario seeds for the final flights without noise


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="Run name: models/NAME.pt and results/NAME_summary.json.")
    p.add_argument("--setting", choices=["single", "multi"], default="single")
    p.add_argument("--n-agents", type=int, default=2, help="Aircraft count (multi agent only).")
    p.add_argument("--workers", type=int, default=8, help="Environment processes (reference: 8 environments).")
    p.add_argument("--steps", type=int, default=30000, help="Environment steps, summed over the workers.")
    p.add_argument("--rollout-steps", type=int, default=512, help="Steps per worker and update (reference: 256; 512 > 300-step max flight).")
    p.add_argument("--epochs", type=int, default=2, help="Reference: 2.")
    p.add_argument("--batch-size", type=int, default=128, help="Aircraft rows per minibatch (reference: 128).")
    p.add_argument("--lr", type=float, default=5e-4, help="Reference: 5e-4.")
    p.add_argument("--lr-final", type=float, default=None,
                   help="2026-11-21: the learning rate goes linearly from --lr to this value over --steps (env steps). "
                        "Default: constant --lr (as before).")
    p.add_argument("--gamma", type=float, default=0.99, help="Reference: 0.9; 0.99 gives a horizon of about 100 steps (1000 s).")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5, help="One clip over all weights (as in the reference).")
    # lidar and frame stack
    p.add_argument("--n-rays", type=int, default=120,
                   help="Lidar rays. 0 = no lidar (no lidar tokens; the aircraft tokens still use --max-range-km).")
    p.add_argument("--fov-deg", type=float, default=360.0)
    p.add_argument("--max-range-km", type=float, default=50.0,
                   help="Lidar radius around the aircraft, km. A ray sees nothing beyond it (100 before 2026-11-12).")
    p.add_argument("--ac-tokens", type=int, default=0,
                   help="K > 0: K more Transformer tokens, one per aircraft that the lidar can see (the K nearest). 0: baseline.")
    p.add_argument("--route-token", type=int, default=0,
                   help="K > 0 (2026-11-22, needs --zone-tokens): one more token after the goal token: the point 10 km "
                        "ahead on the shortest path around the restricted areas grown by K km (rl_tf/route.py). The "
                        "policy learns how to use it; the action reference stays the goal bearing. 0 = off.")
    p.add_argument("--zone-tokens", type=int, default=0,
                   help="Restricted-area tokens: the Z nearest areas (edge point, the two ends, blocks-the-goal flag) "
                        "plus one goal token with the time left. A scenario has 5 areas. 0 = off.")
    p.add_argument("--act-tokens", type=int, default=0,
                   help="Action-history tokens: the last H own actions, one token each. 0 = off.")
    p.add_argument("--edge-tokens", type=int, default=0,
                   help="Sector-edge tokens: E direction bins around the aircraft, each with the nearest point of the "
                        "sector edge in that direction (no range limit). 0 = off.")
    p.add_argument("--ac-cpa", type=int, choices=[0, 1], default=0,
                   help="1: each aircraft token also has the time to the closest point of approach and the distance there.")
    p.add_argument("--lidar-speed", type=int, choices=[0, 1], default=1,
                   help="1: 4th lidar channel with the closing speed of the aircraft that the ray hits.")
    p.add_argument("--map-px", type=int, default=0,
                   help="P > 0: map image of P x P pixels around the aircraft (restricted areas, outside the sector, "
                        "goal; newest frame only), one token per patch. 0 = off (baseline).")
    p.add_argument("--map-range-km", type=float, default=150.0,
                   help="Map image: half side of the square, km (the image covers 2 x this value).")
    p.add_argument("--map-patch", type=int, default=8, help="Map image: pixels per patch side (one token per patch).")
    p.add_argument("--history", type=int, default=4, help="Stacked frames (reference: 4).")
    p.add_argument("--zone-flag", type=int, choices=[0, 1], default=1,
                   help="1: the own state has a flag 'inside a no-fly zone'. 0: as before 2026-11-11.")
    # network
    p.add_argument("--patch", type=int, default=4, help="Adjacent rays per token.")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--features-dim", type=int, default=256)
    p.add_argument("--fourier", type=int, default=0,
                   help="F > 0: the object tokens (aircraft, area, goal, sector edge) get sin and cos of each value at F "
                        "frequencies (pi, 2 pi, 4 pi, ...) in addition to the value. 0 = off.")
    p.add_argument("--cls-token", type=int, choices=[0, 1], default=0,
                   help="1: a learned read-out token collects the result; the own-state token then only carries the state.")
    p.add_argument("--log-std-init", type=float, default=0.0, help="Start exploration noise (reference: 0.0).")
    p.add_argument("--log-std-set", type=float, default=None,
                   help="2026-11-21: set log_std to this value after --init-policy (the start model brings its own "
                        "log_std). Default: keep it.")
    # JEPA pretraining and curriculum (2026-11-13); the defaults give the training of before
    p.add_argument("--init-encoder", default="",
                   help="File of a pretrained encoder (scripts.pretrain_jepa) or of a policy; its encoder weights are the "
                        "start weights. The token and network flags must be the same. Empty = random start weights.")
    p.add_argument("--freeze-encoder-steps", type=int, default=0,
                   help="No encoder update before this many environment steps (the heads adapt first). 0 = off.")
    p.add_argument("--curriculum", default="",
                   help="Stages 'I:O@S,...': from environment step S on, training scenarios have I other aircraft and O "
                        "restricted areas (competition: 10:5). Example: '2:1@0,5:3@300000,10:5@800000'. The final "
                        "flights without noise always use 10:5. Empty = no curriculum. Multi agent: I must be 0 (the "
                        "environment has no scripted aircraft; change the number of learning aircraft with a chain of "
                        "runs, --n-agents and --init-policy).")
    p.add_argument("--init-policy", default="",
                   help="Policy file (scripts.train) whose encoder, actor and log_std are the start weights; the critic "
                        "starts new. The token and network flags must be the same. A single-agent policy can start a "
                        "multi-agent run. Empty = off.")
    # reward
    p.add_argument("--reward-scale", type=float, default=0.0002, help="Scale of the reward used for learning only.")
    p.add_argument("--shaping", choices=["potential", "off"], default="off",
                   help="off (default): the learning reward has only negative values. potential: add gamma*phi(s')-phi(s), "
                        "phi = time_penalty * estimated seconds to the waypoint; this term is positive during progress "
                        "to the waypoint, so it breaks the only-negative constraint.")
    p.add_argument("--shaping-speed", type=float, default=150.0, help="Nominal speed (m/s) in the potential.")
    # v2 (2026-11-19, single-agent rebuild). Defaults = baseline behaviour. See V2_DEFAULTS in rl_tf/envs.py.
    p.add_argument("--action-map", choices=["rel", "goal"], default=V2_DEFAULTS["action_map"],
                   help="goal: turn action = heading offset from the bearing to the goal (change clipped to +-d_heading).")
    p.add_argument("--goal-offset-deg", type=float, default=V2_DEFAULTS["goal_offset_deg"],
                   help="action_map goal: heading offset (deg) at turn action +-1.")
    p.add_argument("--cpa-margin-penalty", type=float, default=V2_DEFAULTS["cpa_margin_penalty"],
                   help="Per second, times max(0, 1 - d_cpa / margin) over the other aircraft. <= 0.")
    p.add_argument("--cpa-margin-nm", type=float, default=V2_DEFAULTS["cpa_margin_nm"])
    p.add_argument("--cpa-horizon-s", type=float, default=V2_DEFAULTS["cpa_horizon_s"])
    p.add_argument("--zone-margin-penalty", type=float, default=V2_DEFAULTS["zone_margin_penalty"],
                   help="Per second, times max(0, 1 - d_zone / buffer). <= 0.")
    p.add_argument("--zone-buffer-km", type=float, default=V2_DEFAULTS["zone_buffer_km"])
    p.add_argument("--timeout-penalty", type=float, default=V2_DEFAULTS["timeout_penalty"],
                   help="<= 0: added once when a flight ends at the time limit with the goal not reached. 0 = off.")
    p.add_argument("--skip-start-events", type=int, choices=[0, 1], default=V2_DEFAULTS["skip_start_events"],
                   help="Single agent. 1: no event penalty for what exists at the reset sample (an aircraft already "
                        "within 5 NM, a start inside a restricted area). The per-second penalties stay. 0 = as before.")
    add_mask_args(p)                                   # heading mask (2026-11-21), rl_tf/heading_mask.py
    p.add_argument("--mask-learn", choices=["exec", "policy"], default="exec",
                   help="2026-11-22, heading mask on: exec = PPO stores the executed (projected) turn and its log-prob "
                        "(before); policy = PPO stores the policy's own action, the mask is part of the environment "
                        "(no far-off actions at small noise: ma_v2m_c2_mask got NaN weights with exec).")
    p.add_argument("--cat-penalty", type=float, default=0.0,
                   help="Training only, single agent: < 0 ends the flight at the first avoidable violation "
                        "(intrusion without the start intrusions, area entry, sector exit) with this reward. 0 = off.")
    p.add_argument("--cat-prob-start", type=float, default=1.0,
                   help="Random CaT (Chane-Sane et al. 2024): probability that a violation ends the flight at step 0. "
                        "1.0 = always (as before).")
    p.add_argument("--cat-prob-end", type=float, default=1.0, help="Random CaT: probability after the ramp.")
    p.add_argument("--cat-prob-steps", type=int, default=0, help="Random CaT: env steps of the linear ramp.")
    p.add_argument("--replay-frac", type=float, default=0.0,
                   help="Single agent: share of resets that fly a seed of an earlier failed flight again. 0 = off.")
    p.add_argument("--replay-cap", type=int, default=256, help="Failed seeds kept per worker.")
    p.add_argument("--replay-max", type=int, default=3, help="Replays of one seed before it leaves the pool.")
    # Lagrangian (PPO-Lagrangian, 2026-11-13); empty = off, then the training is that of before
    p.add_argument("--lagrangian", default="",
                   help="Constraints 'metric:limit,...' on the mean flight metrics (competition units per flight), e.g. "
                        "'intrusion_events_no_start:0.3,time_in_restricted_area:5,time_outside_sector:6'. Each gets a "
                        "multiplier lam >= 0 and a cost critic; the advantage is (A_r - sum lam A_c) / (1 + sum lam). "
                        "The cost of a step is the increase of the metric times --reward-scale, so lam is in the units "
                        "of a reward weight (lam 3 on intrusion events = a penalty of -3 per event). The fixed reward "
                        "weights stay. Single agent can use the *_no_start metrics.")
    p.add_argument("--lambda-lr", type=float, default=0.05,
                   help="Step of the multiplier per update, times the relative violation (cost / limit - 1, clipped "
                        "to [-1, 1]).")
    p.add_argument("--lambda-init", type=float, default=0.0)
    p.add_argument("--lambda-max", type=float, default=100.0)
    add_reward_args(p)
    add_wandb_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reset-seed", type=int, default=-1, help="Scenario seed used at every reset. -1 = random scenarios.")
    p.add_argument("--eval-episodes", type=int, default=10, help="Final flights without noise (random scenarios: held-out seeds).")
    p.add_argument("--save-every", type=int, default=20, help="Save the model every N updates.")
    # long runs (2026-11-20): save points, validation on other seeds, best model, resume over several jobs.
    # All off by default: the training is then that of before.
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="Env steps between save points: the weights go to models/NAME_ckpt/step_N.pt and the full "
                        "training state (optimizer, counters, curve, worker seed counters) to models/NAME_state.pt. "
                        "0 = off.")
    p.add_argument("--val-every", type=int, default=0,
                   help="Env steps between validations: a weight snapshot (models/NAME_ckpt/step_N.pt) flies the "
                        "competition protocol on --val-seeds in background processes. The best snapshot (leaderboard "
                        "rule: clean %%, then goal %%, then events) goes to models/NAME_best.pt, all records to "
                        "results/NAME_val.json (rl_tf/validate.py). 0 = off.")
    p.add_argument("--val-seeds", default="7", help="Validation seeds, comma list. Never 42 (the leaderboard test seed).")
    p.add_argument("--val-episodes", type=int, default=100, help="Validation episodes per seed.")
    p.add_argument("--val-workers", type=int, default=2, help="Validation processes (one thread each).")
    p.add_argument("--resume", action="store_true",
                   help="Continue from models/NAME_state.pt if the file exists (else start new). --steps may be "
                        "larger than before. Not bit-identical to a run without a stop: the random generators of "
                        "the environment workers start again (their seed counters and replay pools continue).")
    p.add_argument("--max-wall-hours", type=float, default=0.0,
                   help="Save the state and stop with exit code 3 after this many hours of this process, so that a "
                        "SLURM job ends before its time limit. The next job continues with --resume. 0 = off.")
    p.add_argument("--keep-episodes", type=int, default=0,
                   help="Keep only the first 20 and the last K training flights in memory, in the state file and in "
                        "the summary (the counts stay exact). 0 = all (as before).")
    p.add_argument("--seed-layout", choices=["narrow", "wide"], default="narrow",
                   help="Training scenario seeds. narrow: 10 000 per worker (as before). wide: 1 000 000 per worker; "
                        "use it when a worker flies more than 10 000 flights (about 1.5 M steps per worker).")
    p.add_argument("--fast-metrics", type=int, default=0, choices=[0, 1],
                   help="Multi agent only (2026-11-20): the training workers compute the scored metrics once per "
                        "simulation step for all aircraft (rl_tf/fast_metrics.py). Bit-identical metrics, see "
                        "scripts.check_fast_path. Validation and evaluation keep the original code. 0 = off.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--learner-threads", type=int, default=6, help="Torch threads in the update phase (workers are idle then).")
    p.add_argument("--inference-threads", type=int, default=1,
                   help="Torch threads for policy.act in the rollout (workers wait then). 1 = as before (2026-11-19).")
    p.add_argument("--out-dir", default=".", help="Base folder for models/ and results/.")
    return p


# arguments that may change between the jobs of one run (--resume); others give a warning, the rest an error
RESUME_FREE = {"steps", "resume", "max_wall_hours", "ckpt_every", "val_every", "val_seeds", "val_episodes",
               "val_workers", "wandb", "wandb_project", "device", "learner_threads", "inference_threads", "out_dir",
               "eval_episodes", "save_every", "keep_episodes", "init_policy", "init_encoder"}
RESUME_HARD = {"setting", "n_agents", "workers", "seed", "seed_layout", "reset_seed"}


def check_resume(old, new):
    diff = sorted(k for k in set(old) | set(new) if k not in RESUME_FREE and old.get(k) != new.get(k))
    hard = [k for k in diff if k in RESUME_HARD]
    if hard:
        raise ValueError(f"--resume: these settings differ from the saved run: "
                         f"{ {k: (old.get(k), new.get(k)) for k in hard} }")
    if diff:
        print(f"WARNING --resume: settings differ from the saved run: { {k: (old.get(k), new.get(k)) for k in diff} }",
              flush=True)


def next_mark(total, every):
    """First multiple of `every` above `total` (0 = off: never)."""
    return (total // every + 1) * every if every > 0 else float("inf")


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_torch(args.seed, 1)
    dev = args.device
    reset_seed = None if args.reset_seed < 0 else args.reset_seed
    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    lidar_cfg = lidar_cfg_from_args(args)
    stages = parse_curriculum(args.curriculum)
    if args.setting == "multi" and any(i != 0 for _, i, _ in stages):
        raise ValueError("--curriculum with --setting multi: the environment has no scripted aircraft, so I must be 0 "
                         "(only the restricted areas change). Change the number of learning aircraft with a chain of "
                         "runs: --n-agents and --init-policy.")
    if args.init_policy and args.init_encoder:
        raise ValueError("give --init-policy or --init-encoder, not both")
    cons = parse_lagrangian(args.lagrangian)
    cost_keys, limits = [k for k, _ in cons], np.array([v for _, v in cons], dtype=np.float64)
    allowed = METRIC_KEYS + (START_KEYS if args.setting == "single" else [])
    if any(k not in allowed for k in cost_keys):
        raise ValueError(f"--lagrangian: metrics must be from {allowed}")
    v2 = v2_kwargs(vars(args))
    masked = bool(v2["heading_mask"]) and args.mask_learn == "exec"   # heading mask: PPO stores the executed actions
    if args.setting != "single" and (args.cat_penalty < 0.0 or args.replay_frac > 0.0 or args.skip_start_events):
        raise ValueError("--cat-penalty, --replay-frac and --skip-start-events: single agent only")
    if args.fast_metrics and args.setting == "single":
        raise ValueError("--fast-metrics: multi agent only")
    if args.cat_penalty > 0.0:
        raise ValueError("--cat-penalty must be <= 0 (the reward has only negative terms)")
    if not (0.0 < args.cat_prob_start <= 1.0 and 0.0 < args.cat_prob_end <= 1.0):
        raise ValueError("--cat-prob-start and --cat-prob-end must be in (0, 1]")
    random_cat = args.cat_penalty < 0.0 and min(args.cat_prob_start, args.cat_prob_end) < 1.0
    if min(args.cat_prob_start, args.cat_prob_end) < 1.0 and args.cat_penalty == 0.0:
        raise ValueError("--cat-prob-start/--cat-prob-end < 1 need --cat-penalty < 0")
    if not 0.0 <= args.replay_frac < 1.0:
        raise ValueError("--replay-frac must be in [0, 1)")
    if args.seed_layout == "wide" and not (0 <= args.seed <= 20 and args.workers <= 100):
        raise ValueError("--seed-layout wide: --seed must be in [0, 20] and --workers <= 100")
    if min(args.ckpt_every, args.val_every, args.keep_episodes) < 0:
        raise ValueError("--ckpt-every, --val-every and --keep-episodes must be >= 0")
    state_path = out / "models" / f"{args.name}_state.pt"
    ckpt_dir = out / "models" / f"{args.name}_ckpt"
    rs = None
    if args.resume and state_path.exists():
        rs = torch.load(state_path, map_location="cpu", weights_only=False)
        check_resume(rs["args"], vars(args))
        if rs.get("finished") and rs["total"] >= args.steps:
            print(f"{args.name}: already finished at env_steps={rs['total']} (>= --steps {args.steps}); nothing to do.",
                  flush=True)
            return 4
        print(f"Resume {args.name} from {state_path} at env_steps={rs['total']} (updates {rs['updates']})", flush=True)
    elif args.resume:
        print(f"--resume: {state_path} does not exist; new run.", flush=True)
    rw0 = reward_kwargs(args)
    if args.cat_penalty < 0.0 and abs(args.cat_penalty) <= abs(rw0["time_penalty"]) * 3000.0:
        print(f"WARNING --cat-penalty {args.cat_penalty}: not below the time cost of a full flight "
              f"({rw0['time_penalty'] * 3000.0:.1f}); ending a flight early can pay off.", flush=True)
    cfg = dict(setting=args.setting, n_agents=args.n_agents, reward=rw0, lidar=lidar_cfg,
               history=args.history, shaping=args.shaping == "potential", shaping_speed=args.shaping_speed,
               seed=args.seed, reset_seed=reset_seed, v2=v2, cat_penalty=args.cat_penalty,
               replay_frac=args.replay_frac, replay_cap=args.replay_cap, replay_max=args.replay_max,
               seed_layout=args.seed_layout, worker_state=rs["worker_state"] if rs else None,
               fast_metrics=bool(args.fast_metrics))
    dims = obs_dims(lidar_cfg, args.history)
    policy = build_policy(dims, vars(args)).to(dev)
    if args.init_encoder and rs is None:
        load_encoder(policy, args.init_encoder, dims, vars(args))
        print(f"Encoder start weights from {args.init_encoder}", flush=True)
    if args.init_policy and rs is None:
        from rl_tf.networks import load_actor
        load_actor(policy, args.init_policy, dims, vars(args))
        print(f"Encoder, actor and log_std start weights from {args.init_policy} (critic new)", flush=True)
        if args.log_std_set is not None:
            with torch.no_grad():
                policy.log_std.fill_(args.log_std_set)
            print(f"log_std set to {args.log_std_set}", flush=True)
    frozen = args.freeze_encoder_steps > 0 and (rs is None or rs["total"] < args.freeze_encoder_steps)
    if frozen:                                         # Adam skips a weight without gradient
        policy.encoder.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    n_par = {"policy": n_params(policy), "encoder": n_params(policy.encoder),
             "actor_head": n_params(policy.actor) + policy.log_std.numel(), "critic_head": n_params(policy.critic)}
    C = len(cons)
    if C:                                              # made after the policy: its start weights stay as before
        ccrit = torch.nn.ModuleList([AttentionCritic(args.features_dim, 64) for _ in range(C)]).to(dev)
        copt = torch.optim.Adam(ccrit.parameters(), lr=args.lr)
        lam = np.full(C, float(args.lambda_init))
        cfg["cost_keys"] = cost_keys
        n_par["cost_critics"] = n_params(ccrit)
        print(f"Lagrangian: limits {dict(cons)}, lambda_lr {args.lambda_lr}, lambda_init {args.lambda_init}", flush=True)

    def lag_extra():
        return {"lagrangian": {"constraints": cons, "lambda": lam.tolist(), "cost_critics": ccrit.state_dict()}} if C else None
    model_path = out / "models" / f"{args.name}.pt"
    if rs is not None:                                 # weights, optimizer and random state of the saved run
        policy.load_state_dict(rs["policy"])
        opt.load_state_dict(rs["opt"])
        if C:
            ccrit.load_state_dict(rs["ccrit"])
            copt.load_state_dict(rs["copt"])
            lam = np.asarray(rs["lam"], dtype=np.float64)
        torch.set_rng_state(rs["torch_rng"])
        np.random.set_state(rs["np_rng"])
        if str(dev).startswith("cuda") and rs.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(rs["cuda_rng"])
    wb = wandb_init(args.wandb, args.wandb_project, args.name, {**vars(args), "n_params": n_par})

    vec = VecEnv(cfg, args.workers)
    E, N = vec.n_envs, vec.n_agents
    print(f"{args.name}: setting {args.setting}, aircraft {N}, workers {E}, n_params {n_par}", flush=True)
    episodes, curve = [], []
    total, updates, n_cut_total, t_start = 0, 0, 0, time.time()
    n_nonfinite = 0
    n_skipped = 0                                      # minibatch steps skipped: non-finite loss or gradient
    episodes_first, n_eps_total, wall_before, n_resumes = None, 0, 0.0, 0
    if rs is not None:
        total, updates, n_cut_total, n_nonfinite = rs["total"], rs["updates"], rs["n_cut_total"], rs["n_nonfinite"]
        episodes, curve, episodes_first, n_eps_total = rs["episodes"], rs["curve"], rs["episodes_first"], rs["n_eps_total"]
        wall_before, n_resumes = rs["wall"], rs.get("n_resumes", 0) + 1
        n_skipped = rs.get("n_skipped", 0)
    next_ckpt, next_val = next_mark(total, args.ckpt_every), next_mark(total, args.val_every)
    validator = None
    if args.val_every > 0:
        from rl_tf.validate import Validator, parse_seeds
        validator = Validator(args.setting, parse_seeds(args.val_seeds), args.val_episodes, args.val_workers,
                        out / "results" / f"{args.name}_val.json", out / "models" / f"{args.name}_best.pt", wb)
        if ckpt_dir.exists():                          # snapshots whose validation was lost in a stopped job
            for f in sorted(ckpt_dir.glob("step_*.pt")):
                st_ = int(f.stem.split("_")[1])
                if st_ <= total and st_ not in validator.done_steps():
                    validator.submit(st_, f)

    def snapshot(step):
        """Weights at this step (same format as models/NAME.pt; scripts.competition_eval can load it)."""
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{step:010d}.pt"
        save_checkpoint(path, policy, vars(args), dims, {**(lag_extra() or {}), "env_steps": step})
        return path

    def save_state(finished=False):
        st_ = {"policy": policy.state_dict(), "opt": opt.state_dict(), "args": vars(args), "dims": dims,
               "total": total, "updates": updates, "n_cut_total": n_cut_total, "n_nonfinite": n_nonfinite, "n_skipped": n_skipped,
               "n_eps_total": n_eps_total, "episodes_first": episodes_first, "episodes": episodes, "curve": curve,
               "timing": timing, "wall": wall_before + time.time() - t_start, "n_resumes": n_resumes,
               "finished": finished, "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if str(dev).startswith("cuda") else None,
               "worker_state": None if finished else vec.snapshot()}
        if C:
            st_.update(ccrit=ccrit.state_dict(), copt=copt.state_dict(), lam=lam.tolist())
        tmp = state_path.with_suffix(".tmp")
        torch.save(st_, tmp)
        os.replace(tmp, state_path)
    tt = lambda x: torch.as_tensor(x, device=dev)
    # wall-clock timers (seconds): environment (reset + step), inference (policy.act), update (GAE + PPO epochs)
    timing = {"env": 0.0, "inference": 0.0, "update": 0.0}
    if rs is not None:
        timing = dict(rs["timing"])
    sync = (lambda: torch.cuda.synchronize()) if str(dev).startswith("cuda") else (lambda: None)

    density = None
    cat_prob = None
    while total < args.steps:
        if args.lr_final is not None:                  # linear decay of the learning rate over the run (by env steps)
            lr_now = args.lr + (args.lr_final - args.lr) * min(1.0, total / float(args.steps))
            for g_ in opt.param_groups:
                g_["lr"] = lr_now
        if random_cat:                                 # random CaT schedule, set at the start of each rollout
            p_now = cat_prob_at(total, args.cat_prob_start, args.cat_prob_end, args.cat_prob_steps)
            if p_now != cat_prob:
                cat_prob = p_now
                vec.set_cat_prob(cat_prob)
        T = max(1, min(args.rollout_steps, int(np.ceil((args.steps - total) / E))))
        if stages and curriculum_stage(stages, total) != density:
            density = curriculum_stage(stages, total)
            vec.set_density(*density)                  # effective from the reset below
            print(f"curriculum at env_steps={total}: {density[0]} other aircraft, {density[1]} restricted areas", flush=True)
        if frozen and total >= args.freeze_encoder_steps:
            frozen = False
            policy.encoder.requires_grad_(True)
            print(f"encoder updates start at env_steps={total}", flush=True)
        t0 = time.perf_counter()
        (state, lidar, _), n_cut = vec.reset()          # reference: env.reset() at the start of every rollout
        timing["env"] += time.perf_counter() - t0
        n_cut_total += n_cut
        B_state = np.zeros((T, E, N, dims["state_dim"]), dtype=np.float32)
        B_lidar = np.zeros((T, E, N, dims["lidar_ch"], dims["n_rays"]), dtype=np.float32)
        B_raw = np.zeros((T, E, N, 2), dtype=np.float32)
        B_logp, B_val, B_rew = (np.zeros((T, E, N), dtype=np.float32) for _ in range(3))
        B_done = np.zeros((T, E, N), dtype=bool)
        if C:
            B_cost, B_cval = np.zeros((T, E, N, C), np.float32), np.zeros((T, E, N, C), np.float32)
        n_before = len(episodes)

        torch.set_num_threads(args.inference_threads)
        for t in range(T):
            t0 = time.perf_counter()
            if C:
                raw, logp, val, feats, *dist = policy.act(tt(state), tt(lidar), return_feats=True, return_dist=masked)
                with torch.no_grad():
                    B_cval[t] = torch.stack([c(feats) for c in ccrit], -1)[:, None].expand(-1, N, -1).cpu().numpy()
            else:
                raw, logp, val, *dist = policy.act(tt(state), tt(lidar), return_dist=masked)
            raw = raw.cpu().numpy()                          # .cpu() waits for the GPU, so the timer is exact
            timing["inference"] += time.perf_counter() - t0
            bad = ~np.isfinite(raw).all(-1)                  # diagnostic: a NaN action gives a BlueSky HDG error
            if bad.any():
                n_nonfinite += int(bad.sum())
                if n_nonfinite == int(bad.sum()):
                    print(f"WARNING non-finite action at env_steps={total}: {int(bad.sum())} aircraft; "
                          f"state finite {bool(np.isfinite(state[bad]).all())}, lidar finite {bool(np.isfinite(lidar[bad]).all())}",
                          flush=True)
            B_state[t], B_lidar[t], B_raw[t] = state, lidar, raw
            B_logp[t], B_val[t] = logp.cpu().numpy(), val.cpu().numpy()
            t0 = time.perf_counter()
            res = vec.step(np.clip(raw, -1.0, 1.0))
            timing["env"] += time.perf_counter() - t0
            if masked and "proj" in res and res["proj"].any():           # heading mask: store the executed turn and its log-prob
                proj = res["proj"]
                done_raw = raw.copy()
                done_raw[..., 0] = np.where(proj, res["exec"][..., 0], raw[..., 0])
                with torch.no_grad():
                    lp = dist[0].log_prob(tt(done_raw)).sum(-1).cpu().numpy()
                B_raw[t] = done_raw
                B_logp[t] = np.where(proj, lp, B_logp[t])
            total += E
            B_rew[t] = (res["rew"] + args.gamma * res["phi_next"] - res["phi"]) * args.reward_scale
            B_done[t] = res["done"]
            if C:
                B_cost[t] = res["cost"] * args.reward_scale
            for e in res["episodes"]:
                episodes.append({"env_steps": total, **e})
            n_eps_total += len(res["episodes"])
            state, lidar, _ = res["cur"]
        t0 = time.perf_counter()
        if C:
            _, _, last_val, feats = policy.act(tt(state), tt(lidar), return_feats=True)
            with torch.no_grad():
                last_cval = torch.stack([c(feats) for c in ccrit], -1)[:, None].expand(-1, N, -1).cpu().numpy()
            last_val = last_val.cpu().numpy()
        else:
            last_val = policy.act(tt(state), tt(lidar))[2].cpu().numpy()
        timing["inference"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        M = E * N
        adv, ret = compute_gae(B_rew.reshape(T, M), B_val.reshape(T, M), B_done.reshape(T, M),
                               last_val.reshape(M), args.gamma, args.gae_lambda)   # no advantage normalization
        if C:                                          # one GAE per constraint, then the Lagrangian advantage
            gc = [compute_gae(B_cost[..., i].reshape(T, M), B_cval[..., i].reshape(T, M), B_done.reshape(T, M),
                              last_cval[..., i].reshape(M), args.gamma, args.gae_lambda) for i in range(C)]
            adv = lagrangian_advantage(adv, np.stack([g[0] for g in gc]), lam)
            f_cret = tt(np.stack([g[1].reshape(-1) for g in gc], axis=-1))

        # one sample = one aircraft row (as the reference); the critic sees all aircraft of its environment
        flat = lambda x: tt(x.reshape(T * E, *x.shape[2:]))
        f_state, f_lidar = flat(B_state), flat(B_lidar)
        f_raw, f_logp = tt(B_raw.reshape(T * E * N, 2)), tt(B_logp.reshape(-1))
        f_adv, f_ret = tt(adv.reshape(-1)), tt(ret.reshape(-1))
        torch.set_num_threads(args.learner_threads)
        for _ in range(args.epochs):
            perm = torch.randperm(T * E * N, device=dev)
            for s in range(0, T * E * N, args.batch_size):
                b = perm[s:s + args.batch_size]
                te, ag = b // N, b % N
                if C:
                    d_all, v, feats = policy.forward_feats(f_state[te], f_lidar[te])
                else:
                    d_all, v = policy(f_state[te], f_lidar[te])
                d = torch.distributions.Normal(d_all.mean[torch.arange(len(b)), ag], d_all.stddev[torch.arange(len(b)), ag])
                ratio = torch.exp((d.log_prob(f_raw[b]).sum(-1) - f_logp[b]).clamp(max=20.0))   # guard: no inf ratio
                a = f_adv[b]
                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                value_loss = (v - f_ret[b]).pow(2).mean()
                entropy = d.entropy().sum(-1).mean()
                loss = policy_loss + args.vf_coef * value_loss - args.entropy_coef * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                if not (torch.isfinite(loss) and torch.isfinite(gn)):   # guard: a NaN step would break the weights
                    n_skipped += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                if C:                                  # cost critics read the features, no gradient into the encoder
                    vc = torch.stack([c(feats.detach()) for c in ccrit], -1)
                    cost_value_loss = (vc - f_cret[b]).pow(2).mean()
                    copt.zero_grad(set_to_none=True)
                    (args.vf_coef * cost_value_loss).backward()
                    torch.nn.utils.clip_grad_norm_(ccrit.parameters(), args.max_grad_norm)
                    copt.step()
        sync()
        timing["update"] += time.perf_counter() - t0
        updates += 1

        new_eps = episodes[n_before:]
        row = {"env_steps": total, "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
               "log_std": policy.log_std.mean().item(), "n_episodes": len(new_eps), "n_cut_episodes": n_cut}
        if args.lr_final is not None:
            row["lr"] = lr_now
        row["n_skipped_steps"] = n_skipped
        if stages:
            row["n_intruders"], row["n_obstacles"] = density
        if C:                                          # dual step on the flights that ended in this rollout
            if new_eps:
                mean_c = np.array([np.mean([e[k] for e in new_eps]) for k in cost_keys])
                row.update({f"cost_{k}": float(v) for k, v in zip(cost_keys, mean_c)})
                lam = lagrange_step(lam, mean_c, limits, args.lambda_lr, args.lambda_max)
            row.update({f"lambda_{k}": float(v) for k, v in zip(cost_keys, lam)})
            row["cost_value_loss"] = cost_value_loss.item()
        txt = ""
        if new_eps:
            for k in ["ep_return", "waypoint_reached", "intrusion_events", "restricted_area_events",
                      "sector_exit_events", "flight_time"]:
                row[k] = float(np.mean([e[k] for e in new_eps]))
            for k in ["cat_end", "replay", "mask_proj_share", "mask_mean_angle_deg", "mask_fallbacks"]:
                if k in new_eps[0]:
                    row[k] = float(np.mean([e.get(k, 0.0) for e in new_eps]))
            txt = (f" | eps={len(new_eps)} ep_return={row['ep_return']:.1f} waypoint={row['waypoint_reached']:.2f} "
                   f"intrusions={row['intrusion_events']:.2f} zones={row['restricted_area_events']:.2f} "
                   f"sector_exit={row['sector_exit_events']:.2f} flight_time={row['flight_time']:.0f}")
            txt += "".join(f" {k}={row[k]:.2f}" for k in ["cat_end", "replay", "mask_proj_share",
                                                          "mask_mean_angle_deg", "mask_fallbacks"] if k in row)
        curve.append(row)
        if wb is not None:
            extra = {k: float(np.mean([e[k] for e in new_eps])) for k in
                     ["intrusion_time", "time_in_restricted_area", "time_outside_sector"]} if new_eps else {}
            wb.log({f"train/{k}": v for k, v in {**row, **extra}.items() if k != "env_steps"}, step=total)
        if C:
            txt += " | lambda " + " ".join(f"{k}={v:.3f}" for k, v in zip(cost_keys, lam))
        print(f"env_steps={total} policy_loss={row['policy_loss']:.4f} value_loss={row['value_loss']:.4f} "
              f"log_std={row['log_std']:.2f}{txt}", flush=True)
        if updates % args.save_every == 0:
            save_checkpoint(model_path, policy, vars(args), dims, lag_extra())
            wandb_upload(wb, model_path)
        if args.keep_episodes > 0 and len(episodes) > args.keep_episodes + WINDOW:
            if episodes_first is None:
                episodes_first = episodes[:WINDOW]
            episodes = episodes[-args.keep_episodes:]
        snap = None
        if validator is not None and total >= next_val:
            snap = snapshot(total)
            validator.submit(total, snap)
            wandb_upload(wb, snap)                     # W&B Files: NAME_ckpt/step_*.pt, for checks on another machine
            next_val = next_mark(total, args.val_every)
        if total >= next_ckpt:
            if snap is None:
                snapshot(total)
            save_state()
            next_ckpt = next_mark(total, args.ckpt_every)
            print(f"save point at env_steps={total}: {state_path}", flush=True)
        if validator is not None:
            validator.poll()
        if args.max_wall_hours > 0 and total < args.steps and time.time() - t_start > args.max_wall_hours * 3600:
            save_state()
            vec.close()
            print(f"wall limit {args.max_wall_hours} h: state saved at env_steps={total}; continue with --resume.",
                  flush=True)
            if validator is not None:
                validator.poll(block=True)
                validator.close()
            if wb is not None:
                wb.finish()
            return 3

    wall = wall_before + time.time() - t_start
    timing.pop("other", None)
    timing["other"] = wall - sum(timing.values())
    timing_share = {k: v / max(wall, 1e-9) for k, v in timing.items()}
    print("timing (s, share of wall): " + ", ".join(f"{k} {v:.1f} ({timing_share[k]:.0%})" for k, v in timing.items()))
    vec.close()
    save_checkpoint(model_path, policy, vars(args), dims, lag_extra())
    print(f"Saved {model_path}")
    wandb_upload(wb, model_path)
    if validator is not None:                                # the final weights are validated too; wait for all jobs
        if total not in validator.done_steps():
            validator.submit(total, snapshot(total))
        validator.poll(block=True)
        validator.close()
        if validator.best is not None:
            print(f"best snapshot: env_steps={validator.best['env_steps']}, clean {100 * validator.best['mean']['clean']:.1f} % "
                  f"on seeds {validator.seeds} -> {validator.best_path}", flush=True)

    # final flights without noise (this process has no BlueSky environment until here)
    torch.set_num_threads(1)
    seeds = [reset_seed] if reset_seed is not None else [EVAL_SEED0 + i for i in range(args.eval_episodes)]
    greedy_rows, turns = [], []
    if args.eval_episodes > 0:
        env = FlightEnv(args.setting, args.n_agents, v2=v2, **reward_kwargs(args))   # no cat_penalty: full flights
        fn = policy_fn_from(policy, dev, deterministic=True)
        for s in seeds:
            rows, _, tr = rollout_episode(env, fn, s, Lidar(**lidar_cfg), args.history)
            greedy_rows += [{"episode_seed": s, "aircraft": a, **r, **turn_stats(tr[a])} for a, r in rows.items()]
        env.close()
    if wb is not None:
        gm = window_stats(greedy_rows) or {}
        wb.summary.update({f"final_no_noise/{k}": v for k, v in gm.items()})
        wb.summary.update({"wall_seconds": wall, "n_agent_episodes": n_eps_total})
    summary_path = out / "results" / f"{args.name}_summary.json"
    write_json(summary_path, {
        "status": "EXPLORATORY",
        "task": "overfit on one fixed scenario" if reset_seed is not None else "random scenarios",
        "name": args.name, "setting": args.setting, "encoder": "lidar ray-token Transformer", "n_aircraft": N,
        "seed": args.seed, "reset_seed": reset_seed, "n_params": n_par, "env_steps": total,
        "n_agent_episodes": n_eps_total, "n_cut_agent_episodes": n_cut_total,
        "n_nonfinite_actions": n_nonfinite, "n_skipped_steps": n_skipped, **pass_criterion((episodes_first or []) + episodes),
        "greedy_eval": {"seeds": seeds, "mean": window_stats(greedy_rows), "rows": greedy_rows},
        "wall_seconds": wall, "seconds_per_step": wall / max(total, 1),
        "timing_seconds": timing, "timing_share": timing_share,
        "lagrangian": {"constraints": cons, "lambda_final": lam.tolist(), "lambda_lr": args.lambda_lr,
                       "lambda_init": args.lambda_init, "lambda_max": args.lambda_max} if C else None,
        "validation": validator.summary() if validator is not None else None, "n_resumes": n_resumes,
        "episodes_kept": "all" if args.keep_episodes == 0 else f"first {WINDOW} and last {args.keep_episodes}",
        "args": vars(args), "curve": curve, "episodes": (episodes_first or []) + episodes})
    if args.ckpt_every > 0 or args.resume:
        save_state(finished=True)
    if wb is not None:
        wandb_upload(wb, summary_path)
        wb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
