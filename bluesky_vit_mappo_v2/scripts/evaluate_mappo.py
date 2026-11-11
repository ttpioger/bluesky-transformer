from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch
import bluesky_gym
from bluesky_zoo.competition_v0 import CompetitionZooEnv
from gymnasium.spaces.utils import flatten
from bluesky_vit.structured_vit import observation_to_tokens
from bluesky_vit.mappo_networks import SharedViTActor


METRICS = ["waypoint_reached", "intrusion_events", "intrusion_time",
           "restricted_area_events", "time_in_restricted_area",
           "sector_exit_events", "time_outside_sector", "flight_time", "total_reward"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/mappo_bluesky_vit.pt")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-agents", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--csv", default="mappo_eval.csv")
    p.add_argument("--reset-seed", type=int, default=None, help="Reset every episode with this seed (overfit test).")
    args = p.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    bluesky_gym.register_envs()
    env = CompetitionZooEnv(n_agents=args.n_agents, render_mode=None)
    possible = list(env.possible_agents)
    obs_space = env.observation_space(possible[0])
    sample = {k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in obs_space.sample().items()}
    n_tokens = observation_to_tokens(sample).shape[1]
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    actor = SharedViTActor(n_tokens=n_tokens).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    rows = []
    obs, _ = env.reset(seed=args.reset_seed if args.reset_seed is not None else args.seed)
    ep = 0
    current = {}
    while ep < args.episodes:
        active = list(env.agents)
        if not active:
            rows.append({"episode": ep, **current})
            ep += 1
            current = {}
            if ep >= args.episodes:
                break
            obs, _ = env.reset(seed=args.reset_seed)
            active = list(env.agents)
        batch = {k: torch.as_tensor(np.stack([obs[a][k] for a in active]),
                                    dtype=torch.float32, device=device) for k in obs[active[0]]}
        with torch.no_grad():
            actions, _, _, _ = actor.act(batch, deterministic=True)
        action_dict = {a: actions[i].cpu().numpy().astype(np.float32) for i, a in enumerate(active)}
        next_obs, rewards, terms, truncs, infos = env.step(action_dict)
        for agent in active:
            if terms.get(agent, False) or truncs.get(agent, False):
                info = infos.get(agent, {})
                for key in METRICS:
                    current[f"{agent}_{key}"] = info.get(key, np.nan)
        obs = next_obs
    env.close()
    if rows:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} episode records to {args.csv}")
        for metric in METRICS:
            vals = [r.get(f"{a}_{metric}") for r in rows for a in possible
                    if r.get(f"{a}_{metric}") is not None]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                print(f"{metric:28s} mean={np.mean(vals):.4f}")
    print("For official reported results, use scripts.evaluate_competition --env ma with its policy-loader edit point.")


if __name__ == "__main__":
    main()
