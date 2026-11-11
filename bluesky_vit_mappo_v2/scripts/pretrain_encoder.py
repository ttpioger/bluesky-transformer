from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import bluesky_gym
from bluesky_zoo.competition_v0 import CompetitionZooEnv
from bluesky_vit.structured_vit import StructuredViT, observation_to_tokens


def parse_args():
    p = argparse.ArgumentParser(description="Collect BlueSky observations and self-supervise the structured ViT.")
    p.add_argument("--transitions", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="checkpoints/bluesky_vit_encoder.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-ratio", type=float, default=0.35)
    p.add_argument("--n-agents", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    bluesky_gym.register_envs()
    bluesky_gym.register_envs()

    env = CompetitionZooEnv(
        n_agents=args.n_agents,
        render_mode=None,
    )

    obs, _ = env.reset(seed=args.seed)

    samples = {k: [] for k in obs[next(iter(obs))]}

    try:
        for step in range(args.transitions):
            active_agents = list(env.agents)

            if not active_agents:
                obs, _ = env.reset()
                active_agents = list(env.agents)

            # Collect observations using the same multi-agent schema as MAPPO.
            for agent in active_agents:
                for key, value in obs[agent].items():
                    samples[key].append(
                        np.asarray(value, dtype=np.float32).copy()
                    )

            actions = {
                agent: env.action_space(agent).sample()
                for agent in active_agents
            }

            obs, rewards, terminations, truncations, infos = env.step(actions)

            if not env.agents:
                obs, _ = env.reset()

            if (step + 1) % 5000 == 0:
                print(
                    f"Collected {step + 1:,}/{args.transitions:,} "
                    f"environment steps"
                )
    finally:
        env.close()

    tensors = {k: torch.as_tensor(np.stack(v), dtype=torch.float32) for k, v in samples.items()}
    dummy = {k: v[:1] for k, v in tensors.items()}
    with torch.no_grad():
        token_count = observation_to_tokens(dummy).shape[1]
    model = StructuredViT(token_count, features_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    keys = list(tensors)
    dataset = TensorDataset(*(tensors[k] for k in keys))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    for epoch in range(args.epochs):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            obs_batch = {k: v.to(device) for k, v in zip(keys, batch)}
            tokens = observation_to_tokens(obs_batch)
            prediction, target, mask = model.masked_reconstruction(tokens, args.mask_ratio)
            loss = ((prediction - target).pow(2).mean(-1) * mask.float()).sum() / mask.float().sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * tokens.shape[0]
            count += tokens.shape[0]
        print(f"epoch {epoch + 1:03d}/{args.epochs}  masked_token_mse={total / max(count, 1):.6f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state_dict": model.state_dict(),
        "n_tokens": token_count,
        "features_dim": 128,
        "observation_keys": keys,
        "seed": args.seed,
    }, out)
    print(f"Saved pretrained structured encoder to {out}")
    print(f"Device used: {device}")


if __name__ == "__main__":
    main()
