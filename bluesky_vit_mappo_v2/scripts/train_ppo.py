from __future__ import annotations
import argparse
from pathlib import Path
import torch
import gymnasium as gym
import bluesky_gym
from stable_baselines3 import PPO
from bluesky_vit.structured_vit import BlueSkyViTExtractor


def main():
    p = argparse.ArgumentParser(description="Single-agent PPO with structured BlueSky ViT.")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", default=None)
    p.add_argument("--output", default="models/ppo_bluesky_vit_sa")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    bluesky_gym.register_envs()
    env = gym.make("CompetitionEnv-v0")
    policy_kwargs = dict(
        features_extractor_class=BlueSkyViTExtractor,
        features_extractor_kwargs=dict(features_dim=128, d_model=128, n_heads=4, n_layers=3),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )
    model = PPO(
        "MultiInputPolicy", env, policy_kwargs=policy_kwargs,
        learning_rate=3e-4, n_steps=1024, batch_size=128,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        verbose=1, seed=args.seed, device=args.device,
        tensorboard_log="runs/ppo_vit_sa",
    )
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        model.policy.features_extractor.encoder.load_state_dict(
            checkpoint["encoder_state_dict"], strict=True
        )
        print(f"Loaded pretrained encoder: {args.pretrained}")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    env.close()
    print(f"Saved PPO model to {args.output}")


if __name__ == "__main__":
    main()
