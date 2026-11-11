"""PPO (single agent) and MAPPO (multi agent) for cells R, A, B, C. One code path for both settings.

Cell = --encoder cnn|set_tf and --jepa off|on. Fast overfit mode: --reset-seed N gives the same
scenario in every episode. This script is a training run: start it through ~/bin/notify_run.sh.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from jepa_set.envs import FlightEnv, add_reward_args, reward_kwargs, seconds_to_go, metrics_row, rollout_episode
from jepa_set.obs import ALL_KEYS, stack_obs, zero_obs, to_torch
from jepa_set.encoders import n_params
from jepa_set.policy import Actor, SetCritic, Jepa, save_checkpoint
from jepa_set.common import cell_name, setup_torch, pass_criterion, write_json, actor_policy


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="Run name: models/NAME.pt and results/NAME_summary.json.")
    p.add_argument("--setting", choices=["single", "multi"], default="multi")
    p.add_argument("--n-agents", type=int, default=2, help="Aircraft count (multi agent only).")
    p.add_argument("--encoder", choices=["cnn", "set_tf"], default="set_tf")
    p.add_argument("--jepa", choices=["off", "on"], default="off")
    p.add_argument("--jepa-ckpt", default=None, help="Pretrained JEPA checkpoint (from scripts.pretrain_jepa).")
    p.add_argument("--jepa-aux-coef", type=float, default=0.1, help="Weight of the JEPA loss during RL. 0 = pretraining only.")
    p.add_argument("--jepa-k", type=int, default=3, help="History length in frames for the JEPA loss.")
    p.add_argument("--jepa-tau", type=float, default=0.99)
    p.add_argument("--init-actor", default=None, help="Checkpoint with an actor (for example from scripts.imitate).")
    p.add_argument("--steps", type=int, default=30000, help="Environment steps.")
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256, help="Time steps per minibatch.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5, help="Applied to the actor and to the critic separately.")
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--log-std-init", type=float, default=-1.0, help="Start exploration noise (std = exp of this).")
    p.add_argument("--reward-scale", type=float, default=0.001, help="Scale of the reward used for learning only.")
    p.add_argument("--shaping", choices=["potential", "off"], default="potential",
                   help="potential: add gamma*phi(s')-phi(s), phi = time_penalty * estimated seconds to the waypoint.")
    p.add_argument("--shaping-speed", type=float, default=150.0, help="Nominal speed (m/s) in the potential.")
    add_reward_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reset-seed", type=int, default=1, help="Scenario seed used at every reset. -1 = random scenarios.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out-dir", default=".", help="Base folder for models/ and results/.")
    return p


def compute_gae(rew, val, next_val, alive, term, trunc, gamma, lam):
    """Arrays [T, N]. next_val holds the bootstrap value where trunc is set and at the last row."""
    T, N = rew.shape
    adv = np.zeros((T, N), dtype=np.float32)
    for i in range(N):
        gae = 0.0
        for t in reversed(range(T)):
            if not alive[t, i]:
                gae = 0.0
                continue
            if term[t, i]:
                nv, carry = 0.0, 0.0
            elif trunc[t, i] or t == T - 1:
                nv, carry = next_val[t, i], 0.0
            else:
                nv, carry = val[t + 1, i], gae
            delta = rew[t, i] + gamma * nv - val[t, i]
            gae = delta + gamma * lam * carry
            adv[t, i] = gae
    return adv, adv + val


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_torch(args.seed, args.threads)
    dev = args.device
    reset_seed = None if args.reset_seed < 0 else args.reset_seed
    cell = cell_name(args.encoder, args.jepa)
    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)

    env = FlightEnv(args.setting, args.n_agents, **reward_kwargs(args))
    ids = env.ids
    N = len(ids)
    actor = Actor(args.encoder, emb_dim=args.emb_dim, log_std_init=args.log_std_init).to(dev)
    critic = SetCritic(args.encoder, emb_dim=args.emb_dim).to(dev)
    jepa = Jepa(actor.encoder).to(dev) if args.jepa == "on" else None
    if args.init_actor:
        actor.load_state_dict(torch.load(args.init_actor, map_location=dev, weights_only=False)["actor"])
        with torch.no_grad():
            actor.log_std.fill_(args.log_std_init)
        print(f"Loaded actor from {args.init_actor}")
    if jepa is not None and args.jepa_ckpt:
        ck = torch.load(args.jepa_ckpt, map_location=dev, weights_only=False)
        actor.encoder.load_state_dict(ck["encoder"])
        jepa.target.load_state_dict(ck["target"])
        jepa.mixer.load_state_dict(ck["mixer"])
        jepa.predictor.load_state_dict(ck["predictor"])
        print(f"Loaded JEPA encoder from {args.jepa_ckpt}")
    use_aux = jepa is not None and args.jepa_aux_coef > 0
    opt_actor = torch.optim.Adam(list(actor.parameters()) + (jepa.aux_parameters() if use_aux else []), lr=args.lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr)
    n_par = {"actor": n_params(actor), "encoder": n_params(actor.encoder), "critic": n_params(critic),
             "jepa_only": (n_params(jepa.mixer) + n_params(jepa.predictor)) if jepa is not None else 0}
    print(f"cell {cell} ({args.encoder}, jepa {args.jepa}) setting {args.setting} aircraft {N} n_params {n_par}", flush=True)

    def phi(o):
        return args.time_penalty * seconds_to_go(o, args.shaping_speed) if args.shaping == "potential" else 0.0

    def joint(obs, live, template):
        rows = [obs[a] if a in live else template for a in ids]
        return stack_obs(rows), np.array([a in live for a in ids])

    def critic_values(obs_np, alive_np):
        with torch.no_grad():
            o = {k: torch.as_tensor(v[None], device=dev) for k, v in obs_np.items()}
            return critic(o, torch.as_tensor(alive_np[None], device=dev))[0].cpu().numpy()

    obs = env.reset(seed=reset_seed if reset_seed is not None else args.seed)
    template = zero_obs(obs[ids[0]])
    ep_ret = {a: 0.0 for a in ids}
    episodes, curve = [], []
    total, t_start = 0, time.time()
    K = args.jepa_k

    while total < args.steps:
        T = min(args.rollout_steps, args.steps - total)
        B = {k: np.zeros((T, N, template[k].shape[0]), dtype=np.float32) for k in ALL_KEYS}
        NB = {k: np.zeros_like(v) for k, v in B.items()}
        alive = np.zeros((T, N), dtype=bool)
        term = np.zeros((T, N), dtype=bool)
        trunc = np.zeros((T, N), dtype=bool)
        raw = np.zeros((T, N, 2), dtype=np.float32)
        logp = np.zeros((T, N), dtype=np.float32)
        rew = np.zeros((T, N), dtype=np.float32)
        val = np.zeros((T, N), dtype=np.float32)
        next_val = np.zeros((T, N), dtype=np.float32)
        ep_start = np.zeros((T, N), dtype=np.int64)
        cur_start = np.zeros(N, dtype=np.int64)
        n_done_before = len(episodes)

        for t in range(T):
            live = list(env.agents)
            j_obs, j_alive = joint(obs, live, template)
            for k in ALL_KEYS:
                B[k][t] = j_obs[k]
            alive[t] = j_alive
            ep_start[t] = cur_start
            val[t] = critic_values(j_obs, j_alive)
            li = [ids.index(a) for a in live]
            r_t, lp_t = actor.act(to_torch({k: v[li] for k, v in j_obs.items()}, dev))
            raw[t, li] = r_t.cpu().numpy()
            logp[t, li] = lp_t.cpu().numpy()
            acts = {a: np.clip(raw[t, i], -1.0, 1.0) for a, i in zip(live, li)}

            nobs, r, te, tr, infos = env.step(acts)
            total += 1
            for a, i in zip(live, li):
                for k in ALL_KEYS:
                    NB[k][t, i] = nobs[a][k]
                term[t, i], trunc[t, i] = te[a], tr[a] and not te[a]
                shaped = args.gamma * (0.0 if te[a] else phi(nobs[a])) - phi(obs[a])
                rew[t, i] = (r[a] + shaped) * args.reward_scale
                ep_ret[a] += r[a]
                if te[a] or tr[a]:
                    episodes.append({"env_steps": total, **metrics_row(infos[a], ep_ret[a])})
                    ep_ret[a] = 0.0
            # Bootstrap value: at the time limit, and at the last row of the rollout.
            boot = [a for a in live if trunc[t, ids.index(a)] or (t == T - 1 and not te[a])]
            if boot:
                nj_obs, nj_alive = joint(nobs, boot, template)
                next_val[t] = critic_values(nj_obs, nj_alive)
            obs = nobs
            if not env.agents:
                obs = env.reset(seed=reset_seed)
                cur_start[:] = t + 1

        adv, ret = compute_gae(rew, val, next_val, alive, term, trunc, args.gamma, args.gae_lambda)
        a_live = adv[alive]
        adv = (adv - a_live.mean()) / (a_live.std() + 1e-8)

        obs_t, nobs_t = to_torch(B, dev), to_torch(NB, dev)
        alive_t = torch.as_tensor(alive, device=dev)
        raw_t, logp_t = torch.as_tensor(raw, device=dev), torch.as_tensor(logp, device=dev)
        adv_t, ret_t = torch.as_tensor(adv, device=dev), torch.as_tensor(ret, device=dev)
        # history rows for the JEPA loss: frames t-K+1..t of the same episode (first frame repeated)
        steps_idx = np.arange(T)[:, None, None] - np.arange(K - 1, -1, -1)[None, None, :]
        hist_idx = torch.as_tensor(np.maximum(steps_idx, ep_start[:, :, None]), device=dev)   # [T, N, K]
        stats = {}
        for _ in range(args.epochs):
            perm = torch.randperm(T, device=dev)
            for s in range(0, T, args.minibatch_size):
                tb = perm[s:s + args.minibatch_size]
                st, si = torch.nonzero(alive_t[tb], as_tuple=True)
                st = tb[st]                                    # time index and aircraft index of live samples
                d, _ = actor.dist({k: v[st, si] for k, v in obs_t.items()})
                new_logp = d.log_prob(raw_t[st, si]).sum(-1)
                ratio = torch.exp(new_logp - logp_t[st, si])
                a = adv_t[st, si]
                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                actor_loss = policy_loss - args.entropy_coef * d.entropy().sum(-1).mean()
                if use_aux:
                    hi = hist_idx[st, si]                      # [M, K]
                    hist = {k: v[hi, si[:, None]] for k, v in obs_t.items()}
                    j_loss, j_stats = jepa.loss(hist, raw_t[st, si].clamp(-1.0, 1.0), {k: v[st, si] for k, v in nobs_t.items()})
                    actor_loss = actor_loss + args.jepa_aux_coef * j_loss
                    stats.update({f"jepa_{k}": v for k, v in j_stats.items()})
                opt_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                opt_actor.step()
                if use_aux:
                    jepa.ema_update(args.jepa_tau)

                v_pred = critic({k: v[tb] for k, v in obs_t.items()}, alive_t[tb])
                m = alive_t[tb].float()
                value_loss = 0.5 * ((v_pred - ret_t[tb]).pow(2) * m).sum() / m.sum().clamp(min=1.0)
                opt_critic.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                opt_critic.step()

        new_eps = episodes[n_done_before:]
        row = {"env_steps": total, "policy_loss": float(policy_loss), "value_loss": float(value_loss),
               "log_std": float(actor.log_std.mean()), "n_episodes": len(new_eps), **stats}
        txt = ""
        if new_eps:
            for k in ["ep_return", "waypoint_reached", "intrusion_events", "restricted_area_events",
                      "sector_exit_events", "flight_time"]:
                row[k] = float(np.mean([e[k] for e in new_eps]))
            txt = (f" | eps={len(new_eps)} ep_return={row['ep_return']:.1f} waypoint={row['waypoint_reached']:.2f} "
                   f"intrusions={row['intrusion_events']:.2f} restricted={row['restricted_area_events']:.2f} "
                   f"sector_exit={row['sector_exit_events']:.2f} flight_time={row['flight_time']:.0f}")
        curve.append(row)
        print(f"env_steps={total} policy_loss={row['policy_loss']:.4f} value_loss={row['value_loss']:.4f} "
              f"log_std={row['log_std']:.2f}{txt}", flush=True)

    wall = time.time() - t_start
    greedy = None
    if reset_seed is not None:   # one flight without noise on the fixed scenario
        greedy, _ = rollout_episode(env, actor_policy(actor, dev, deterministic=True), reset_seed)
    env.close()
    model_path = out / "models" / f"{args.name}.pt"
    save_checkpoint(model_path, actor, critic, jepa, vars(args))
    print(f"Saved {model_path}")
    crit = pass_criterion(episodes)
    write_json(out / "results" / f"{args.name}_summary.json", {
        "status": "EXPLORATORY", "task": "overfit on one fixed scenario" if reset_seed is not None else "random scenarios",
        "name": args.name, "cell": cell, "setting": args.setting, "encoder": args.encoder, "jepa": args.jepa,
        "n_aircraft": N, "seed": args.seed, "reset_seed": reset_seed, "n_params": n_par,
        "env_steps": total, "n_agent_episodes": len(episodes), **crit,
        "greedy_eval": greedy, "wall_seconds": wall, "seconds_per_step": wall / max(total, 1),
        "args": vars(args), "curve": curve, "episodes": episodes})


if __name__ == "__main__":
    main()
