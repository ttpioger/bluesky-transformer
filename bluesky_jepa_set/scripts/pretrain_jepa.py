"""JEPA pretraining of an encoder on saved transitions. Logs the collapse monitor every epoch.

Collapse monitor: "spread" = mean over dimensions of the standard deviation of the embedding.
A spread below --collapse-limit at the end marks the run as collapsed.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from jepa_set.data import load_transitions, history_index
from jepa_set.encoders import build_encoder, n_params
from jepa_set.policy import Jepa
from jepa_set.common import setup_torch, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--encoder", choices=["cnn", "set_tf"], default="set_tf")
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--k", type=int, default=3, help="History length in frames.")
    p.add_argument("--tau", type=float, default=0.99, help="EMA rate of the target encoder.")
    p.add_argument("--var-coef", type=float, default=1.0)
    p.add_argument("--collapse-limit", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out-dir", default=".")
    args = p.parse_args(argv)
    setup_torch(args.seed, args.threads)
    dev = args.device
    obs, nxt, meta = load_transitions(args.data)
    hidx = torch.as_tensor(history_index(meta["agent"], meta["episode"], args.k), device=dev)
    obs_t = {k: torch.as_tensor(v, device=dev) for k, v in obs.items()}
    nxt_t = {k: torch.as_tensor(v, device=dev) for k, v in nxt.items()}
    act_t = torch.as_tensor(meta["action"], dtype=torch.float32, device=dev)
    n = act_t.shape[0]
    jepa = Jepa(build_encoder(args.encoder, out_dim=args.emb_dim), var_coef=args.var_coef).to(dev)
    opt = torch.optim.Adam([q for q in jepa.parameters() if q.requires_grad], lr=args.lr)
    curve, t0 = [], time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        agg = {}
        for s in range(0, n, args.batch_size):
            ids = perm[s:s + args.batch_size]
            if len(ids) < 2:
                continue
            loss, st = jepa.loss({k: v[hidx[ids]] for k, v in obs_t.items()}, act_t[ids],
                                 {k: v[ids] for k, v in nxt_t.items()})
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(jepa.parameters(), 1.0)
            opt.step()
            jepa.ema_update(args.tau)
            for k, v in st.items():
                agg.setdefault(k, []).append(v)
        row = {"epoch": ep + 1, **{k: float(np.mean(v)) for k, v in agg.items()}}
        curve.append(row)
        print(f"epoch {ep + 1}/{args.epochs} pred_loss={row['pred_loss']:.5f} var_loss={row['var_loss']:.4f} "
              f"spread={row['spread']:.4f} target_spread={row['target_spread']:.4f}", flush=True)
    out = Path(args.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    ck = out / "checkpoints" / f"{args.name}.pt"
    torch.save({"encoder": jepa.online.state_dict(), "target": jepa.target.state_dict(),
                "mixer": jepa.mixer.state_dict(), "predictor": jepa.predictor.state_dict(), "args": vars(args)}, ck)
    print(f"Saved {ck}")
    write_json(out / "results" / f"{args.name}_summary.json", {
        "status": "EXPLORATORY", "task": "JEPA pretraining on the fixed scenario", "name": args.name,
        "encoder": args.encoder, "seed": args.seed, "n_transitions": n,
        "n_params": {"encoder": n_params(jepa.online), "jepa_only": n_params(jepa.mixer) + n_params(jepa.predictor)},
        "final_spread": curve[-1]["spread"], "collapsed": bool(curve[-1]["spread"] < args.collapse_limit),
        "wall_seconds": time.time() - t0, "args": vars(args), "curve": curve})


if __name__ == "__main__":
    main()
