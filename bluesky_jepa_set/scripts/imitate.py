"""Supervised sanity test: the encoder plus the actor head must fit the scripted controller.

Step 1: regression of the clean controller action on saved observations (MSE).
Step 2: one closed-loop flight without noise on the fixed scenario.
Pass (fixed before the runs): final MSE <= 0.005 and waypoint_reached = 1 for every aircraft.
Do this test before any RL claim.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import torch
from jepa_set.data import load_transitions
from jepa_set.envs import FlightEnv, rollout_episode
from jepa_set.encoders import n_params
from jepa_set.policy import Actor, save_checkpoint
from jepa_set.common import setup_torch, write_json, actor_policy

MSE_LIMIT = 0.005


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--setting", choices=["single", "multi"], default="multi")
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--encoder", choices=["cnn", "set_tf"], default="set_tf")
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reset-seed", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out-dir", default=".")
    args = p.parse_args(argv)
    setup_torch(args.seed, args.threads)
    dev = args.device
    obs, _, meta = load_transitions(args.data)
    X = {k: torch.as_tensor(v, device=dev) for k, v in obs.items()}
    Y = torch.as_tensor(meta["label"], dtype=torch.float32, device=dev)
    n = Y.shape[0]
    actor = Actor(args.encoder, emb_dim=args.emb_dim).to(dev)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    curve, t0 = [], time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for s in range(0, n, args.batch_size):
            ids = perm[s:s + args.batch_size]
            d, _ = actor.dist({k: v[ids] for k, v in X.items()})
            loss = (d.mean - Y[ids]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(ids)
        curve.append(tot / n)
        print(f"epoch {ep + 1}/{args.epochs} mse={curve[-1]:.5f}", flush=True)
    env = FlightEnv(args.setting, args.n_agents)
    rows, _ = rollout_episode(env, actor_policy(actor, dev, deterministic=True), args.reset_seed)
    env.close()
    reached = all(r["waypoint_reached"] >= 1.0 for r in rows.values()) and len(rows) == len(env.ids)
    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / "models" / f"{args.name}.pt", actor, args=vars(args))
    ok = bool(curve[-1] <= MSE_LIMIT and reached)
    print(f"imitation {'PASS' if ok else 'FAIL'}: final mse {curve[-1]:.5f} (limit {MSE_LIMIT}), closed loop {rows}")
    write_json(out / "results" / f"{args.name}_summary.json", {
        "status": "EXPLORATORY", "task": "imitation of the scripted controller", "name": args.name,
        "setting": args.setting, "encoder": args.encoder, "seed": args.seed, "n_samples": n,
        "n_params": {"actor": n_params(actor), "encoder": n_params(actor.encoder)},
        "first_mse": curve[0], "last_mse": curve[-1], "mse_limit": MSE_LIMIT, "closed_loop": rows,
        "pass": ok, "wall_seconds": time.time() - t0, "args": vars(args), "curve": curve})


if __name__ == "__main__":
    main()
