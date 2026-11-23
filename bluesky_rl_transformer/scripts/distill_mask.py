"""Distil the test-time safety filter into the policy weights (2026-11-23, user OK: "Try this for our SA and MA").

After RL training, the policy is fine-tuned by supervised learning (no RL) to output by itself the action that the
filter (mask, route, fb2) executed. This differs from shield training (mask in the training loop): here no filter
fixes the policy during the update; the policy must make the correction itself.

1. collect: fly the competition protocol with the filter on (stream seed --data-seed, never 7 / 11 / 13 / 17 / 42).
   At each decision, keep the policy input (state, lidar), the policy action (mean) and the executed action
   (env.last_exec, the action after the filter, in the same reference frame as the policy output). All decisions
   that the filter changed are kept; of the others, the share --keep-free (they anchor the unchanged behaviour).
2. train: loss = mean of w * |mean(s) - target|^2 over the kept decisions. target = executed action (clipped to
   +-0.97: the actor ends with tanh). w = --w-corr for the changed decisions, 1 for the others (target = own action).
   --train actor: only the action head changes; all: encoder and head. Adam, --lr, --epochs.
3. The fine-tuned policy is saved as OUT_DIR/distill_<tag>.pt; check it with scripts.competition_eval (tuning seeds
   13 / 17 first, then 7 / 11).

Usage (from bluesky_rl_transformer/, CPU node):
  python -m scripts.distill_mask --env ma --model models/X.pt --data-seed 23 --episodes 200 --procs 16 \
      --heading-mask 1 --mask-mode arc --mask-intruders 1 --mask-dist-km 9.6 --mask-route 1 --mask-route-km 5 \
      --mask-fb2 1 --out-dir /path/distill_ma [--configs actor:1:3e-5 all:3:1e-5]
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import io
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import torch
from rl_tf.heading_mask import add_mask_args, mask_overrides

FORBIDDEN = (7, 11, 13, 17, 42)          # model choice, tuning, leaderboard


def _collect(job):
    from scripts.competition_eval import _load_member, N_AGENTS_MA
    from rl_tf.envs import FlightEnv, rollout_episode, v2_kwargs
    from rl_tf.common import setup_torch
    model, kind, a, b, seed, opts, keep_free, out = job
    setup_torch(a)
    rng = np.random.default_rng(1000 + a)
    policy, policy_fn, rew, lidar, history, c = _load_member(model, "cpu")
    v2 = v2_kwargs({**c, **(opts.get("mask") or {})})
    pend, S, L, R, E = {}, [], [], [], []

    def fn(raw, built):
        acts = policy_fn(raw, built)
        pend.clear()
        for k in built:
            pend[k] = (built[k][0], built[k][1], np.asarray(acts[k], dtype=np.float32))
        return acts

    with contextlib.redirect_stdout(io.StringIO()):
        env = FlightEnv("single" if kind == "sa" else "multi", N_AGENTS_MA, env_kw={}, v2=v2, **rew)

        def on_step():
            ex = getattr(env, "last_exec", None)
            if not pend or not ex:
                return
            for k, (s, l, r) in pend.items():
                if k not in ex:
                    continue
                e = np.asarray(ex[k], dtype=np.float32)
                changed = abs(float(e[0]) - float(r[0])) > 1e-4
                if changed or rng.random() < keep_free:
                    S.append(s), L.append(l), R.append(r), E.append(e)
            pend.clear()

        for i in range(a):
            env.env.reset(seed=seed if i == 0 else None)
        for i in range(a, b):
            rollout_episode(env, fn, seed if i == 0 else None, lidar, history, max_steps=400, on_step=on_step)
        env.close()
    np.savez_compressed(out, state=np.stack(S).astype(np.float32), lidar=np.stack(L).astype(np.float32),
                        raw=np.stack(R), exe=np.stack(E))
    return out, len(S)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["sa", "ma"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data-seed", type=int, default=23)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--procs", type=int, default=16)
    p.add_argument("--keep-free", type=float, default=0.3, help="share of unchanged decisions kept (default 0.3)")
    p.add_argument("--configs", nargs="+", default=["actor:1:3e-5", "actor:3:3e-5", "all:1:1e-5", "all:3:1e-5"],
                   help="TRAIN:W_CORR:LR, one fine-tuned model each (TRAIN = actor or all)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--reuse-data", action="store_true", help="skip the collection if OUT_DIR/data_*.npz exist")
    p.add_argument("--out-dir", required=True)
    add_mask_args(p, from_checkpoint=True)
    args = p.parse_args(argv)
    if args.data_seed in FORBIDDEN:
        raise SystemExit(f"--data-seed {args.data_seed} is reserved (7/11 model choice, 13/17 tuning, 42 leaderboard)")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opts = {"mask": mask_overrides(args)}
    t0 = time.time()
    files = sorted(out.glob("data_*.npz"))
    if not (args.reuse_data and files):
        w = max(1, min(args.procs, args.episodes))
        edges = np.linspace(0, args.episodes, w + 1).astype(int)
        jobs = [(args.model, args.env, int(edges[i]), int(edges[i + 1]), args.data_seed, opts, args.keep_free,
                 str(out / f"data_{i:03d}.npz")) for i in range(w)]
        with mp.get_context("spawn").Pool(w, maxtasksperchild=1) as pool:
            res = pool.map(_collect, jobs)
        files = [Path(f) for f, _ in res]
        print(f"collected {sum(n for _, n in res)} decisions in {time.time() - t0:.0f} s", flush=True)
    d = [np.load(f) for f in files]
    S = torch.from_numpy(np.concatenate([x["state"] for x in d]))
    L = torch.from_numpy(np.concatenate([x["lidar"] for x in d]))
    raw = np.concatenate([x["raw"] for x in d])
    exe = np.concatenate([x["exe"] for x in d])
    changed = np.abs(exe[:, 0] - raw[:, 0]) > 1e-4
    Y = torch.from_numpy(np.clip(exe, -0.97, 0.97).astype(np.float32))
    print(f"data: {len(Y)} decisions, {int(changed.sum())} changed by the filter "
          f"({100 * changed.mean():.1f} % of the kept ones)", flush=True)
    torch.set_num_threads(max(1, args.procs))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(Y), generator=g)
    n_val = max(1, len(Y) // 20)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    from rl_tf.networks import load_policy
    log = dict(args=vars(args), n=len(Y), n_changed=int(changed.sum()), configs={})
    for cfg in args.configs:
        train, w_corr, lr = cfg.split(":")
        w_corr, lr = float(w_corr), float(lr)
        policy, ck = load_policy(args.model, "cpu")
        params = list(policy.actor.parameters()) if train == "actor" else \
            list(policy.encoder.parameters()) + list(policy.actor.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        W = torch.from_numpy(np.where(changed, w_corr, 1.0).astype(np.float32))
        C = torch.from_numpy(changed)

        def mean_of(idx):
            s, l = S[idx][:, None], L[idx][:, None]
            dist, _ = policy(s, l)
            return dist.mean[:, 0]

        def evaluate():
            policy.eval()
            with torch.no_grad():
                m = torch.cat([mean_of(val_idx[i:i + 2048]) for i in range(0, len(val_idx), 2048)])
            err = ((m - Y[val_idx]) ** 2).sum(-1)
            c = C[val_idx]
            policy.train()
            return float(err[c].mean()) if c.any() else 0.0, float(err[~c].mean()) if (~c).any() else 0.0

        hist = [dict(epoch=0, val_changed=evaluate()[0], val_free=evaluate()[1])]
        for ep in range(1, args.epochs + 1):
            order = tr_idx[torch.randperm(len(tr_idx), generator=g)]
            for i in range(0, len(order), args.batch):
                idx = order[i:i + args.batch]
                loss = (W[idx] * ((mean_of(idx) - Y[idx]) ** 2).sum(-1)).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()
            vc, vf = evaluate()
            hist.append(dict(epoch=ep, val_changed=vc, val_free=vf))
            print(f"[{cfg}] epoch {ep}: val error on changed {vc:.4f}, on unchanged {vf:.4f} "
                  f"(start {hist[0]['val_changed']:.4f} / {hist[0]['val_free']:.4f}) ({time.time() - t0:.0f} s)",
                  flush=True)
        tag = cfg.replace(":", "_")
        ck2 = copy.copy(ck)
        ck2["policy"] = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        ck2["distilled_from"] = dict(model=args.model, config=cfg, mask=opts["mask"], data_seed=args.data_seed,
                                     episodes=args.episodes)
        path = out / f"distill_{tag}.pt"
        torch.save(ck2, path)
        log["configs"][cfg] = dict(path=str(path), history=hist)
        with open(out / "distill_log.json", "w") as f:
            json.dump(log, f, indent=1)
    print(f"done in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
