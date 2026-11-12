"""PPO (single agent) and MAPPO (multi agent) with a Transformer over lidar ray tokens.

One code path for both settings. --workers environment processes run in parallel (one thread each).
All PPO/MAPPO choices follow avbiswas/navigation-mappo-rl (commit 309ac1d); only the encoder and the
reward differ (see README).
--reset-seed N gives the same scenario in every episode (overfit test); -1 gives random scenarios.
This script is a training run: start it through ~/bin/notify_run.sh.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from rl_tf.envs import (VecEnv, FlightEnv, add_reward_args, reward_kwargs, obs_dims, rollout_episode, turn_stats)
from rl_tf.lidar import Lidar
from rl_tf.networks import build_policy, save_checkpoint, n_params
from rl_tf.common import (setup_torch, compute_gae, pass_criterion, window_stats, write_json, policy_fn_from, wandb_upload,
                          wandb_init, add_wandb_args)

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
    p.add_argument("--gamma", type=float, default=0.99, help="Reference: 0.9; 0.99 gives a horizon of about 100 steps (1000 s).")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5, help="One clip over all weights (as in the reference).")
    # lidar and frame stack
    p.add_argument("--n-rays", type=int, default=120)
    p.add_argument("--fov-deg", type=float, default=360.0)
    p.add_argument("--max-range-km", type=float, default=50.0,
                   help="Lidar radius around the aircraft, km. A ray sees nothing beyond it (100 before 2026-11-12).")
    p.add_argument("--ac-tokens", type=int, default=0,
                   help="K > 0: K more Transformer tokens, one per aircraft that the lidar can see (the K nearest). 0: baseline.")
    p.add_argument("--lidar-speed", type=int, choices=[0, 1], default=1,
                   help="1: 4th lidar channel with the closing speed of the aircraft that the ray hits.")
    p.add_argument("--history", type=int, default=4, help="Stacked frames (reference: 4).")
    p.add_argument("--zone-flag", type=int, choices=[0, 1], default=1,
                   help="1: the own state has a flag 'inside a no-fly zone'. 0: as before 2026-11-11.")
    # network
    p.add_argument("--patch", type=int, default=4, help="Adjacent rays per token.")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--features-dim", type=int, default=256)
    p.add_argument("--log-std-init", type=float, default=0.0, help="Start exploration noise (reference: 0.0).")
    # reward
    p.add_argument("--reward-scale", type=float, default=0.0002, help="Scale of the reward used for learning only.")
    p.add_argument("--shaping", choices=["potential", "off"], default="off",
                   help="off (default): the learning reward has only negative values. potential: add gamma*phi(s')-phi(s), "
                        "phi = time_penalty * estimated seconds to the waypoint; this term is positive during progress "
                        "to the waypoint, so it breaks the only-negative constraint.")
    p.add_argument("--shaping-speed", type=float, default=150.0, help="Nominal speed (m/s) in the potential.")
    add_reward_args(p)
    add_wandb_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reset-seed", type=int, default=-1, help="Scenario seed used at every reset. -1 = random scenarios.")
    p.add_argument("--eval-episodes", type=int, default=10, help="Final flights without noise (random scenarios: held-out seeds).")
    p.add_argument("--save-every", type=int, default=20, help="Save the model every N updates.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--learner-threads", type=int, default=6, help="Torch threads in the update phase (workers are idle then).")
    p.add_argument("--out-dir", default=".", help="Base folder for models/ and results/.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_torch(args.seed, 1)
    dev = args.device
    reset_seed = None if args.reset_seed < 0 else args.reset_seed
    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    lidar_cfg = dict(n_rays=args.n_rays, fov_deg=args.fov_deg, max_range_km=args.max_range_km,
                     zone_flag=bool(args.zone_flag), speed=bool(args.lidar_speed), ac_tokens=args.ac_tokens)
    cfg = dict(setting=args.setting, n_agents=args.n_agents, reward=reward_kwargs(args), lidar=lidar_cfg,
               history=args.history, shaping=args.shaping == "potential", shaping_speed=args.shaping_speed,
               seed=args.seed, reset_seed=reset_seed)
    dims = obs_dims(lidar_cfg, args.history)
    policy = build_policy(dims, vars(args)).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    n_par = {"policy": n_params(policy), "encoder": n_params(policy.encoder),
             "actor_head": n_params(policy.actor) + policy.log_std.numel(), "critic_head": n_params(policy.critic)}
    model_path = out / "models" / f"{args.name}.pt"
    wb = wandb_init(args.wandb, args.wandb_project, args.name, {**vars(args), "n_params": n_par})

    vec = VecEnv(cfg, args.workers)
    E, N = vec.n_envs, vec.n_agents
    print(f"{args.name}: setting {args.setting}, aircraft {N}, workers {E}, n_params {n_par}", flush=True)
    episodes, curve = [], []
    total, updates, n_cut_total, t_start = 0, 0, 0, time.time()
    n_nonfinite = 0
    tt = lambda x: torch.as_tensor(x, device=dev)
    # wall-clock timers (seconds): environment (reset + step), inference (policy.act), update (GAE + PPO epochs)
    timing = {"env": 0.0, "inference": 0.0, "update": 0.0}
    sync = (lambda: torch.cuda.synchronize()) if str(dev).startswith("cuda") else (lambda: None)

    while total < args.steps:
        T = max(1, min(args.rollout_steps, int(np.ceil((args.steps - total) / E))))
        t0 = time.perf_counter()
        (state, lidar, _), n_cut = vec.reset()          # reference: env.reset() at the start of every rollout
        timing["env"] += time.perf_counter() - t0
        n_cut_total += n_cut
        B_state = np.zeros((T, E, N, dims["state_dim"]), dtype=np.float32)
        B_lidar = np.zeros((T, E, N, dims["lidar_ch"], dims["n_rays"]), dtype=np.float32)
        B_raw = np.zeros((T, E, N, 2), dtype=np.float32)
        B_logp, B_val, B_rew = (np.zeros((T, E, N), dtype=np.float32) for _ in range(3))
        B_done = np.zeros((T, E, N), dtype=bool)
        n_before = len(episodes)

        torch.set_num_threads(1)
        for t in range(T):
            t0 = time.perf_counter()
            raw, logp, val = policy.act(tt(state), tt(lidar))
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
            total += E
            B_rew[t] = (res["rew"] + args.gamma * res["phi_next"] - res["phi"]) * args.reward_scale
            B_done[t] = res["done"]
            for e in res["episodes"]:
                episodes.append({"env_steps": total, **e})
            state, lidar, _ = res["cur"]
        t0 = time.perf_counter()
        last_val = policy.act(tt(state), tt(lidar))[2].cpu().numpy()
        timing["inference"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        M = E * N
        adv, ret = compute_gae(B_rew.reshape(T, M), B_val.reshape(T, M), B_done.reshape(T, M),
                               last_val.reshape(M), args.gamma, args.gae_lambda)   # no advantage normalization

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
                d_all, v = policy(f_state[te], f_lidar[te])
                d = torch.distributions.Normal(d_all.mean[torch.arange(len(b)), ag], d_all.stddev[torch.arange(len(b)), ag])
                ratio = torch.exp(d.log_prob(f_raw[b]).sum(-1) - f_logp[b])
                a = f_adv[b]
                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                value_loss = (v - f_ret[b]).pow(2).mean()
                entropy = d.entropy().sum(-1).mean()
                loss = policy_loss + args.vf_coef * value_loss - args.entropy_coef * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()
        sync()
        timing["update"] += time.perf_counter() - t0
        updates += 1

        new_eps = episodes[n_before:]
        row = {"env_steps": total, "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
               "log_std": policy.log_std.mean().item(), "n_episodes": len(new_eps), "n_cut_episodes": n_cut}
        txt = ""
        if new_eps:
            for k in ["ep_return", "waypoint_reached", "intrusion_events", "restricted_area_events",
                      "sector_exit_events", "flight_time"]:
                row[k] = float(np.mean([e[k] for e in new_eps]))
            txt = (f" | eps={len(new_eps)} ep_return={row['ep_return']:.1f} waypoint={row['waypoint_reached']:.2f} "
                   f"intrusions={row['intrusion_events']:.2f} zones={row['restricted_area_events']:.2f} "
                   f"sector_exit={row['sector_exit_events']:.2f} flight_time={row['flight_time']:.0f}")
        curve.append(row)
        if wb is not None:
            extra = {k: float(np.mean([e[k] for e in new_eps])) for k in
                     ["intrusion_time", "time_in_restricted_area", "time_outside_sector"]} if new_eps else {}
            wb.log({f"train/{k}": v for k, v in {**row, **extra}.items() if k != "env_steps"}, step=total)
        print(f"env_steps={total} policy_loss={row['policy_loss']:.4f} value_loss={row['value_loss']:.4f} "
              f"log_std={row['log_std']:.2f}{txt}", flush=True)
        if updates % args.save_every == 0:
            save_checkpoint(model_path, policy, vars(args), dims)
            wandb_upload(wb, model_path)

    wall = time.time() - t_start
    timing["other"] = wall - sum(timing.values())
    timing_share = {k: v / max(wall, 1e-9) for k, v in timing.items()}
    print("timing (s, share of wall): " + ", ".join(f"{k} {v:.1f} ({timing_share[k]:.0%})" for k, v in timing.items()))
    vec.close()
    save_checkpoint(model_path, policy, vars(args), dims)
    print(f"Saved {model_path}")
    wandb_upload(wb, model_path)

    # final flights without noise (this process has no BlueSky environment until here)
    torch.set_num_threads(1)
    seeds = [reset_seed] if reset_seed is not None else [EVAL_SEED0 + i for i in range(args.eval_episodes)]
    greedy_rows, turns = [], []
    if args.eval_episodes > 0:
        env = FlightEnv(args.setting, args.n_agents, **reward_kwargs(args))
        fn = policy_fn_from(policy, dev, deterministic=True)
        for s in seeds:
            rows, _, tr = rollout_episode(env, fn, s, Lidar(**lidar_cfg), args.history)
            greedy_rows += [{"episode_seed": s, "aircraft": a, **r, **turn_stats(tr[a])} for a, r in rows.items()]
        env.close()
    if wb is not None:
        gm = window_stats(greedy_rows) or {}
        wb.summary.update({f"final_no_noise/{k}": v for k, v in gm.items()})
        wb.summary.update({"wall_seconds": wall, "n_agent_episodes": len(episodes)})
    summary_path = out / "results" / f"{args.name}_summary.json"
    write_json(summary_path, {
        "status": "EXPLORATORY",
        "task": "overfit on one fixed scenario" if reset_seed is not None else "random scenarios",
        "name": args.name, "setting": args.setting, "encoder": "lidar ray-token Transformer", "n_aircraft": N,
        "seed": args.seed, "reset_seed": reset_seed, "n_params": n_par, "env_steps": total,
        "n_agent_episodes": len(episodes), "n_cut_agent_episodes": n_cut_total,
        "n_nonfinite_actions": n_nonfinite, **pass_criterion(episodes),
        "greedy_eval": {"seeds": seeds, "mean": window_stats(greedy_rows), "rows": greedy_rows},
        "wall_seconds": wall, "seconds_per_step": wall / max(total, 1),
        "timing_seconds": timing, "timing_share": timing_share,
        "args": vars(args), "curve": curve, "episodes": episodes})
    if wb is not None:
        wandb_upload(wb, summary_path)
        wb.finish()


if __name__ == "__main__":
    main()
