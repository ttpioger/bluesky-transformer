"""Rank-1 edit of the first actor layer (2026-11-23, user OK "try this").

Data: the decisions collected by scripts/distill_mask.py (policy input, own action, action after the shield).
Loss L = mean of w * |mean(s) - target|^2 (target = executed action, clipped to +-0.97; w = --w-corr on the
decisions the shield changed, 1 elsewhere). The gradient of L with respect to actor.0.weight [64, 256] is split by
SVD; only its top singular pair u1 v1^T is used: v1 = the encoder-feature direction most linked to the corrections,
u1 = the hidden-unit pattern that it drives. Edit: W0 <- W0 - alpha * ||W0||_F * u1 v1^T, for each alpha in
--alphas. One model per alpha; the loss on the data before / after is printed. Check the models with
scripts.competition_eval (tuning seeds 13 / 17 first).

Usage: python -m scripts.rank1_edit --model models/X.pt --data-dir DIR --out-dir DIR [--alphas 0.01 0.02 0.05 0.1]
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from rl_tf.networks import load_policy


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="step sizes; default (2026-11-23 fix): line search on the data loss over 1e-5 ... 1e-2 of ||W0||,\n                        then the best step and half of it (the first grid 0.01 ... 0.1 overshot: the loss rose)")
    p.add_argument("--w-corr", type=float, default=3.0)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = [np.load(f) for f in sorted(Path(a.data_dir).glob("data_*.npz"))]
    S = torch.from_numpy(np.concatenate([x["state"] for x in d]))
    L = torch.from_numpy(np.concatenate([x["lidar"] for x in d]))
    raw = np.concatenate([x["raw"] for x in d])
    exe = np.concatenate([x["exe"] for x in d])
    changed = np.abs(exe[:, 0] - raw[:, 0]) > 1e-4
    Y = torch.from_numpy(np.clip(exe, -0.97, 0.97).astype(np.float32))
    W = torch.from_numpy(np.where(changed, a.w_corr, 1.0).astype(np.float32))
    policy, ck = load_policy(a.model, "cpu")
    policy.eval()
    with torch.no_grad():                                   # encoder features once (the edit is after the encoder)
        F = torch.cat([policy.encoder(S[i:i + 2048], L[i:i + 2048]) for i in range(0, len(S), 2048)])
    W0 = policy.actor[0].weight

    def loss_of(w0):
        h = torch.tanh(F @ w0.T + policy.actor[0].bias)
        m = policy.actor[5](policy.actor[4](policy.actor[3](policy.actor[2](h))))     # rest of the actor
        return (W * ((m - Y) ** 2).sum(-1)).mean()

    w0 = W0.detach().clone().requires_grad_(True)
    base = loss_of(w0)
    base.backward()
    G = w0.grad.detach()
    U, Sv, Vh = torch.linalg.svd(G, full_matrices=False)
    R = torch.outer(U[:, 0], Vh[0])                          # unit Frobenius norm
    share = float(Sv[0] ** 2 / (Sv ** 2).sum())
    nW = float(W0.detach().norm())
    print(f"data {len(Y)} decisions, {int(changed.sum())} changed; loss {float(base):.5f}; top singular share of the "
          f"gradient {100 * share:.1f} %; ||W0|| {nW:.3f}", flush=True)
    log = dict(model=a.model, n=len(Y), n_changed=int(changed.sum()), loss0=float(base), share=share, edits={})
    if a.alphas is None:
        grid = np.concatenate([np.logspace(-5, -2, 19), -np.logspace(-5, -2, 19)])   # both signs, to be safe
        with torch.no_grad():
            ls = [float(loss_of(W0.detach() - g * nW * R)) for g in grid]
        k = int(np.argmin(ls))
        print("line search: " + ", ".join(f"{g:.1e}:{l:.4f}" for g, l in zip(grid, ls)), flush=True)
        a.alphas = [float(grid[k]), float(grid[k] / 2)] if ls[k] < float(base) else []
        print(f"best step {grid[k]:.2e} (loss {ls[k]:.5f} vs {float(base):.5f}); edits: {a.alphas}", flush=True)
        log["line_search"] = dict(grid=grid.tolist(), loss=ls)
    for al in a.alphas:
        w_new = W0.detach() - al * nW * R
        with torch.no_grad():
            l1 = float(loss_of(w_new))
        ck2 = copy.copy(ck)
        pol = {k: v.clone() for k, v in ck["policy"].items()}
        pol["actor.0.weight"] = w_new.clone()
        ck2["policy"] = pol
        ck2["rank1_edit"] = dict(model=a.model, alpha=al, data=a.data_dir, w_corr=a.w_corr)
        path = out / f"rank1_a{al:.2e}.pt"
        torch.save(ck2, path)
        log["edits"][str(al)] = dict(loss=l1, path=str(path))
        print(f"alpha {al:g}: loss {float(base):.5f} -> {l1:.5f} -> {path}", flush=True)
    json.dump(log, open(out / "rank1_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
