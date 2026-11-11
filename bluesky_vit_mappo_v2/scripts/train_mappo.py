from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.optim import Adam
import bluesky_gym
from bluesky_zoo.competition_v0 import CompetitionZooEnv
from gymnasium.spaces.utils import flatten
from bluesky_vit.structured_vit import observation_to_tokens
from bluesky_vit.mappo_networks import SharedViTActor, CentralizedViTCritic


def obs_batch(obs_list, device):
    return {k: torch.as_tensor(np.stack([o[k] for o in obs_list]), dtype=torch.float32, device=device)
            for k in obs_list[0]}


def main():
    p = argparse.ArgumentParser(description="Centralized-critic MAPPO for CompetitionZooEnv.")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--n-agents", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--pretrained", default=None)
    p.add_argument("--output", default="models/mappo_bluesky_vit.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--reset-seed", type=int, default=None,
                   help="Reset every episode with this seed (same scenario each time). Used for overfit tests.")
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="Multiply rewards used for learning (logged episode returns stay unscaled).")
    p.add_argument("--reach-reward", type=float, default=1.0, help="Env reward for reaching the waypoint.")
    p.add_argument("--drift-penalty", type=float, default=-0.01, help="Env reward per radian of heading drift, per step.")
    p.add_argument("--intrusion-penalty", type=float, default=-1.0, help="Env reward per second of intrusion.")
    p.add_argument("--restricted-penalty", type=float, default=-1.0, help="Env reward per second in a restricted area.")
    p.add_argument("--sector-exit-penalty", type=float, default=-1.0, help="Env reward per second outside the sector.")
    p.add_argument("--time-penalty", type=float, default=0.0,
                   help="Extra reward added at every step for every live agent (use a negative value).")
    p.add_argument("--init-actor", default=None,
                   help="Checkpoint with an 'actor' state dict (e.g. from imitate_controller) to start from.")
    p.add_argument("--log-std-init", type=float, default=None,
                   help="Overwrite the actor log_std (exploration noise) after loading.")
    p.add_argument("--summary", default=None, help="Write a JSON summary (first/last episode statistics) here.")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    bluesky_gym.register_envs()
    env = CompetitionZooEnv(
        n_agents=args.n_agents, render_mode=None,
        reach_reward=args.reach_reward, drift_penalty=args.drift_penalty,
        intrusion_penalty=args.intrusion_penalty,
        restricted_area_penalty=args.restricted_penalty,
        sector_exit_penalty=args.sector_exit_penalty,
    )
    possible = list(env.possible_agents)
    n_agents = len(possible)
    obs_space = env.observation_space(possible[0])
    flat_dim = flatten(obs_space, obs_space.sample()).shape[0]
    joint_dim = flat_dim * n_agents
    with torch.no_grad():
        sample = {k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in obs_space.sample().items()}
        n_tokens = observation_to_tokens(sample).shape[1]

    actor = SharedViTActor(n_tokens=n_tokens).to(device)
    critic = CentralizedViTCritic(joint_dim, n_agents).to(device)
    if args.pretrained:
        ckpt = torch.load(
            args.pretrained,
            map_location="cpu",
            weights_only=False,
        )

        if ckpt["n_tokens"] != n_tokens:
            raise ValueError(
                f"Pretrained encoder has {ckpt['n_tokens']} tokens, "
                f"but MAPPO expects {n_tokens}. "
                "Pretrain using the same CompetitionZooEnv configuration."
            )

        actor.encoder.load_state_dict(
            ckpt["encoder_state_dict"],
            strict=True,
        )

        print(f"Loaded pretrained encoder with {n_tokens} tokens.")
    if args.init_actor:
        ck = torch.load(args.init_actor, map_location="cpu", weights_only=False)
        if ck["n_tokens"] != n_tokens:
            raise ValueError(f"init actor has {ck['n_tokens']} tokens, expected {n_tokens}")
        actor.load_state_dict(ck["actor"], strict=True)
        print(f"Loaded actor from {args.init_actor}")
    if args.log_std_init is not None:
        with torch.no_grad():
            actor.log_std.fill_(args.log_std_init)
        print(f"log_std set to {args.log_std_init}")
    optimizer = Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)

    def joint_state(obs_dict, active):
        parts = []
        for agent in possible:
            if agent in active and agent in obs_dict:
                parts.append(flatten(obs_space, obs_dict[agent]).astype(np.float32))
            else:
                parts.append(np.zeros(flat_dim, dtype=np.float32))
        return np.concatenate(parts)

    obs, _ = env.reset(seed=args.reset_seed if args.reset_seed is not None else args.seed)
    total_env_steps = 0
    ep_ret = defaultdict(float)   # running return per agent
    all_eps = []                  # finished agent-episodes: (return, waypoint, intrusion, restricted, sector_exit)
    history = []
    while total_env_steps < args.steps:
        # Per-agent trajectories are contiguous while an agent remains alive.
        trajectories = defaultdict(list)
        finished = []
        for _ in range(args.rollout_steps):
            active = list(env.agents)
            if not active:
                obs, _ = env.reset(seed=args.reset_seed)
                active = list(env.agents)
            state_np = joint_state(obs, active)
            state_t = torch.as_tensor(state_np[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                values = critic(state_t)[0].cpu().numpy()
            active_obs = [obs[a] for a in active]
            ob_t = obs_batch(active_obs, device)
            with torch.no_grad():
                dist = actor.distribution(ob_t)
                raw = dist.sample()
                actions_t = torch.tanh(raw)
                logp = (dist.log_prob(raw) - torch.log(1 - actions_t.pow(2) + 1e-6)).sum(-1)
            action_dict = {a: actions_t[i].cpu().numpy().astype(np.float32) for i, a in enumerate(active)}
            raw_np = raw.cpu().numpy()
            logp_np = logp.cpu().numpy()

            next_obs, rewards, terms, truncs, infos = env.step(action_dict)
            next_active = list(env.agents)
            next_state_np = joint_state(next_obs, next_active)
            with torch.no_grad():
                next_values = critic(torch.as_tensor(next_state_np[None], dtype=torch.float32, device=device))[0].cpu().numpy()

            for i, agent in enumerate(active):
                idx = possible.index(agent)
                done = bool(terms.get(agent, False) or truncs.get(agent, False))
                trajectories[agent].append({
                    "obs": {k: np.asarray(v).copy() for k, v in obs[agent].items()},
                    "raw": raw_np[i].copy(),
                    "logp": float(logp_np[i]),
                    "reward": (float(rewards.get(agent, 0.0)) + args.time_penalty) * args.reward_scale,
                    "value": float(values[idx]),
                    "next_value": 0.0 if done else float(next_values[idx]),
                    "done": done,
                    "state": state_np.copy(),
                    "agent_idx": idx,
                })
                ep_ret[agent] += float(rewards.get(agent, 0.0)) + args.time_penalty
                if done:
                    inf = infos.get(agent, {})
                    finished.append((ep_ret.pop(agent, 0.0),
                                     float(inf.get("waypoint_reached", 0)),
                                     float(inf.get("intrusion_events", 0)),
                                     float(inf.get("restricted_area_events", 0)),
                                     float(inf.get("sector_exit_events", 0))))
            obs = next_obs
            total_env_steps += 1
            if not env.agents:
                obs, _ = env.reset(seed=args.reset_seed)
            if total_env_steps >= args.steps:
                break

        # GAE per agent; terminated/truncated agents do not bootstrap past their final step.
        records = []
        for agent, traj in trajectories.items():
            gae = 0.0
            for item in reversed(traj):
                nonterminal = 0.0 if item["done"] else 1.0
                delta = item["reward"] + args.gamma * item["next_value"] * nonterminal - item["value"]
                gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
                item["adv"] = gae
                item["return"] = gae + item["value"]
                records.append(item)

        if not records:
            continue
        obs_keys = list(records[0]["obs"])
        actor_obs = {k: torch.as_tensor(np.stack([r["obs"][k] for r in records]),
                                        dtype=torch.float32, device=device) for k in obs_keys}
        raw_actions = torch.as_tensor(np.stack([r["raw"] for r in records]), dtype=torch.float32, device=device)
        old_logp = torch.as_tensor([r["logp"] for r in records], dtype=torch.float32, device=device)
        advantages = torch.as_tensor([r["adv"] for r in records], dtype=torch.float32, device=device)
        returns = torch.as_tensor([r["return"] for r in records], dtype=torch.float32, device=device)
        states = torch.as_tensor(np.stack([r["state"] for r in records]), dtype=torch.float32, device=device)
        agent_idx = torch.as_tensor([r["agent_idx"] for r in records], dtype=torch.long, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n = len(records)
        for _epoch in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, args.minibatch_size):
                ids = perm[start:start + args.minibatch_size]
                new_logp, entropy = actor.evaluate_raw_actions(
                    {k: v[ids] for k, v in actor_obs.items()}, raw_actions[ids]
                )
                ratio = torch.exp(new_logp - old_logp[ids])
                surr1 = ratio * advantages[ids]
                surr2 = torch.clamp(ratio, 1-args.clip, 1+args.clip) * advantages[ids]
                policy_loss = -torch.min(surr1, surr2).mean()
                values_pred = critic(states[ids]).gather(1, agent_idx[ids, None]).squeeze(1)
                value_loss = 0.5 * (values_pred - returns[ids]).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Clip actor and critic separately: a large critic gradient must not shrink the actor gradient.
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optimizer.step()

        mean_return = np.mean([r["return"] for r in records])
        ep_txt = ""
        if finished:
            all_eps.extend(finished)
            m = np.mean(np.array(finished), axis=0)
            history.append({"env_steps": total_env_steps, "n_episodes": len(finished),
                            "ep_return": float(m[0]), "waypoint_reached": float(m[1]),
                            "intrusion_events": float(m[2]), "restricted_area_events": float(m[3]),
                            "sector_exit_events": float(m[4])})
            ep_txt = (f" | eps={len(finished)} ep_return={m[0]:.3f} waypoint={m[1]:.2f} "
                      f"intrusions={m[2]:.2f} restricted={m[3]:.2f} sector_exit={m[4]:.2f}")
        print(f"env_steps={total_env_steps:,} samples={len(records):,} "
              f"policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f} "
              f"mean_return_target={mean_return:.4f}{ep_txt}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": actor.state_dict(), "critic": critic.state_dict(),
        "n_agents": n_agents, "joint_state_dim": joint_dim,
        "n_tokens": n_tokens, "args": vars(args),
    }, out)
    env.close()
    print(f"Saved MAPPO checkpoint to {out}")
    if args.summary:
        import json
        names = ["ep_return", "waypoint_reached", "intrusion_events", "restricted_area_events", "sector_exit_events"]
        def window(rows):
            if not rows:
                return None
            m = np.mean(np.array(rows), axis=0)
            return {"n_agent_episodes": len(rows), **{k: float(v) for k, v in zip(names, m)}}
        w = 20
        summary = {
            "status": "EXPLORATORY",
            "task": "overfit on one fixed scenario" if args.reset_seed is not None else "random scenarios",
            "n_agents": n_agents, "reset_seed": args.reset_seed, "seed": args.seed,
            "env_steps": total_env_steps, "n_params_actor": sum(p.numel() for p in actor.parameters()),
            "n_params_critic": sum(p.numel() for p in critic.parameters()),
            "n_agent_episodes": len(all_eps),
            "first_window": window(all_eps[:w]), "last_window": window(all_eps[-w:]),
            "args": vars(args), "curve": history,
        }
        sp = Path(args.summary)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=1))
        print(f"Wrote summary to {sp}")


if __name__ == "__main__":
    main()
