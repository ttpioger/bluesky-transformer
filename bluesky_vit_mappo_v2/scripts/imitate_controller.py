"""Imitation sanity test: train the ViT actor to copy a scripted controller.

Controller: turn = clip(-gain * sin_drift, -1, 1), speed change = 0.
Data: roll out the env, execute controller + Gaussian noise, label with the clean controller.
Test afterwards with: python -m scripts.evaluate_mappo --model <output> ...
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import bluesky_gym
from bluesky_zoo.competition_v0 import CompetitionZooEnv
from bluesky_vit.structured_vit import observation_to_tokens
from bluesky_vit.mappo_networks import SharedViTActor


def controller(obs, gain):
    return np.array([np.clip(-gain * float(obs["sin_drift"][0]), -1.0, 1.0), 0.0], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--samples", type=int, default=6000, help="Agent-steps to collect.")
    p.add_argument("--noise", type=float, default=0.3, help="Std of noise added to executed actions.")
    p.add_argument("--gain", type=float, default=3.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reset-seed", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="models/imitation_2ag.pt")
    p.add_argument("--summary", default=None)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bluesky_gym.register_envs()
    env = CompetitionZooEnv(n_agents=args.n_agents, render_mode=None)
    possible = list(env.possible_agents)
    obs_space = env.observation_space(possible[0])
    sample = {k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in obs_space.sample().items()}
    n_tokens = observation_to_tokens(sample).shape[1]

    t0 = time.time()
    data, labels = [], []
    obs, _ = env.reset(seed=args.reset_seed if args.reset_seed is not None else args.seed)
    while len(labels) < args.samples:
        active = list(env.agents)
        if not active:
            obs, _ = env.reset(seed=args.reset_seed)
            active = list(env.agents)
        acts = {}
        for a in active:
            y = controller(obs[a], args.gain)
            data.append({k: np.asarray(v).copy() for k, v in obs[a].items()})
            labels.append(y)
            acts[a] = np.clip(y + rng.normal(0, args.noise, 2), -1, 1).astype(np.float32)
        obs, *_ = env.step(acts)
    env.close()
    t_collect = time.time() - t0

    device = args.device
    keys = list(data[0])
    X = {k: torch.as_tensor(np.stack([d[k] for d in data]), dtype=torch.float32, device=device) for k in keys}
    Y = torch.as_tensor(np.stack(labels), dtype=torch.float32, device=device)
    actor = SharedViTActor(n_tokens=n_tokens).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    n = len(Y)
    losses = []
    t1 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, args.minibatch_size):
            ids = perm[s:s + args.minibatch_size]
            pred = torch.tanh(actor.distribution({k: v[ids] for k, v in X.items()}).mean)
            loss = (pred - Y[ids]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(ids)
        losses.append(tot / n)
        print(f"epoch {ep + 1}/{args.epochs} mse={losses[-1]:.5f} elapsed={time.time() - t1:.0f}s", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "n_agents": len(possible), "n_tokens": n_tokens, "args": vars(args)}, out)
    print(f"Saved {out}")
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps({
            "status": "EXPLORATORY", "task": "imitation of scripted controller", "seed": args.seed,
            "n_params_actor": sum(q.numel() for q in actor.parameters()), "samples": n,
            "collect_seconds": t_collect, "train_seconds": time.time() - t1,
            "first_mse": losses[0], "last_mse": losses[-1], "args": vars(args)}, indent=1))


if __name__ == "__main__":
    main()
