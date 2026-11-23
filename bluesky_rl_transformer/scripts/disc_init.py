"""Warm start of the discrete heading head from a Gaussian policy (2026-11-23, user OK "try this methodology").

The new policy (--heading-bins K) gets the encoder, the actor trunk (actor.0, actor.2), the critic and log_std of
the Gaussian model. The speed head copies row 1 of the old output layer (actor.4), so the speed action is the same.
The heading head (64 -> K logits) is fitted by cross-entropy to soft targets: p_k ~ exp(-(b_k - m)^2 / (2 s^2)),
b_k the bin values in [-1, 1], m the old policy's heading mean on the same state, s = --sigma. The states come from
the data files of scripts/distill_mask.py (DATA_DIR/data_*.npz; the old policy's own action is the "raw" field).
So the new policy starts by flying like the old one (argmax = nearest bin), with exploration over the bins.

Usage: python -m scripts.disc_init --model models/X.pt --data-dir DIR --out models/X_disc19.pt [--bins 19]
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from rl_tf.networks import build_policy, load_policy


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bins", type=int, default=19)
    p.add_argument("--sigma", type=float, default=0.15, help="soft-target width in action units (0.111 = one bin)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(5)
    old, ck = load_policy(a.model, "cpu")
    old.eval()
    args = dict(ck["args"], heading_bins=a.bins)
    new = build_policy(ck["dims"], args)
    sd = {k: v for k, v in ck["policy"].items() if not k.startswith("actor.4")}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("head_") for k in missing), (missing, unexpected)
    with torch.no_grad():
        new.head_s[0].weight.copy_(ck["policy"]["actor.4.weight"][1:2])
        new.head_s[0].bias.copy_(ck["policy"]["actor.4.bias"][1:2])
    from pathlib import Path
    d = [np.load(f) for f in sorted(Path(a.data_dir).glob("data_*.npz"))]
    S = torch.from_numpy(np.concatenate([x["state"] for x in d]))
    L = torch.from_numpy(np.concatenate([x["lidar"] for x in d]))
    with torch.no_grad():
        F = torch.cat([old.encoder(S[i:i + 2048], L[i:i + 2048]) for i in range(0, len(S), 2048)])
        Z = old.actor[:4](F)                                   # trunk output, the same in the new policy
        m = old.actor[4:](Z)[:, 0]                             # old heading mean
    bins = torch.linspace(-1.0, 1.0, a.bins)
    T = torch.softmax(-(bins[None] - m[:, None]) ** 2 / (2 * a.sigma ** 2), dim=1)
    opt = torch.optim.Adam(new.head_h.parameters(), lr=a.lr)
    n = len(Z)
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            loss = -(T[idx] * torch.log_softmax(new.head_h(Z[idx]), -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        lg = new.head_h(Z)
        near = (bins[None] - m[:, None]).abs().argmin(1)
        acc = float((lg.argmax(1) == near).float().mean())
        err = float((bins[lg.argmax(1)] - m).abs().mean() * 90.0)
        ent = float(torch.distributions.Categorical(logits=lg).entropy().mean())
        sp = float((new.head_s(Z)[:, 0] - old.actor[4:](Z)[:, 1]).abs().max())
    print(f"{n} states: argmax = nearest bin of the old mean for {100 * acc:.1f} %; mean |heading error| "
          f"{err:.1f} deg; mean entropy {ent:.2f} nats (uniform {np.log(a.bins):.2f}); speed max diff {sp:.2e}",
          flush=True)
    out = dict(ck, policy=new.state_dict(), args=args, disc_init=dict(model=a.model, data=a.data_dir, bins=a.bins,
               sigma=a.sigma, acc=acc, err_deg=err))
    out.pop("lagrangian", None)                                # optimizer state of the old run: not reused
    torch.save(out, a.out)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
