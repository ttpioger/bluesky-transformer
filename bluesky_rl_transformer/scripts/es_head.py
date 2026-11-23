"""Evolution strategies on the last actor layers (2026-11-23, user OK: "Use this technic then for the ES").

ES (Salimans et al. 2017, "Evolution Strategies as a Scalable Alternative to Reinforcement Learning") fine-tunes the
policy with episode outcomes only, no gradient through the network. Here only the last layers of the action head change
(default actor.2 and actor.4, 4290 weights of about 790 000), so few generations are needed and the trained features
of the encoder stay as they are. The deterministic policy (action = mean) is scored, with the test-time stack (mask,
route, fb2) given by the flags, so ES tunes the policy for the setting it is used in.

Noise control (user request "make it less noisy for the score"):
  * antithetic pairs: each noise vector eps is tested as +eps and -eps on the SAME scenarios (common random numbers);
  * flights with a loss of separation at the start are left out (they fail for every policy);
  * cost per flight = (not clean) + 0.5 * (avoidable events: later intrusions + area entries + sector exits + goal not
    reached), a denser signal than clean yes / no;
  * centered ranks of the pair differences (robust to outliers).
Each generation flies new scenarios (stream seed --gen-seed0 + generation), so ES cannot fit one fixed set.
Every --val-every generations, the centre is scored on the tuning seeds (default 13 17, --val-episodes per seed). The
best centre on the tuning seeds is saved as OUT_DIR/es_best.pt. Seeds 7 / 11 (model choice) and 42 are not used here.

Usage (from bluesky_rl_transformer/, CPU node, single agent):
  python -m scripts.es_head --env sa --model models/X.pt --pairs 16 --episodes 48 --generations 60 --procs 32 \
      --heading-mask 1 --mask-mode arc --mask-intruders 1 --mask-dist-km 9.6 --mask-route 1 --mask-route-km 3 \
      --mask-fb2 1 --out-dir /path/es_sa
"""
from __future__ import annotations
import argparse
import copy
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import torch
from rl_tf.heading_mask import add_mask_args, mask_overrides


def _run(job):
    from scripts.competition_eval import _job
    tag, rest = job
    return tag, _job(rest)


def clean(r):
    return (r["waypoint_reached"] >= 1 and r["intrusion_events"] == 0 and r["restricted_area_events"] == 0
            and r["sector_exit_events"] == 0)


def events(r):
    later = r.get("intrusion_events_no_start", r["intrusion_events"])
    return later + r["restricted_area_events"] + r["sector_exit_events"] + (1 - min(1, r["waypoint_reached"]))


def cost(r):
    return float(not clean(r)) + 0.5 * events(r)


def keyed(rows):
    return {(r["episode_index"], r["aircraft"]): r for r in rows}


def centered_ranks(x):
    """Ranks scaled to [-0.5, 0.5]; equal values get their mean rank (all equal -> all 0, no step)."""
    x = np.asarray(x, dtype=np.float64)
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    for u in np.unique(x):
        sel = x == u
        r[sel] = r[sel].mean()
    return r / max(len(x) - 1, 1) - 0.5


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["sa", "ma"], default="sa")
    p.add_argument("--model", required=True)
    p.add_argument("--params", nargs="+", default=["actor.2.", "actor.4."], help="name prefixes of the tuned tensors")
    p.add_argument("--pairs", type=int, default=16, help="antithetic pairs per generation (default 16)")
    p.add_argument("--episodes", type=int, default=48, help="episodes per evaluation in a generation (default 48)")
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--sigma", type=float, default=0.05, help="noise size relative to the RMS of each tensor")
    p.add_argument("--lr", type=float, default=0.01, help="Adam step size relative to the RMS of each tensor")
    p.add_argument("--gen-seed0", type=int, default=1000)
    p.add_argument("--val-seeds", type=int, nargs="+", default=[13, 17])
    p.add_argument("--val-episodes", type=int, default=300)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--procs", type=int, default=32)
    p.add_argument("--seed", type=int, default=3, help="noise seed (never 42 for a choice)")
    p.add_argument("--out-dir", required=True)
    add_mask_args(p, from_checkpoint=True)
    args = p.parse_args(argv)
    for s in args.val_seeds:
        if s in (7, 11, 42):
            raise SystemExit(f"seed {s} is for model choice or the leaderboard; use other tuning seeds")
    out = Path(args.out_dir)
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    opts = {"mask": mask_overrides(args)}
    base = torch.load(args.model, map_location="cpu", weights_only=False)
    names = [k for k in base["policy"] if any(k.startswith(q) for q in args.params)]
    shapes = [tuple(base["policy"][k].shape) for k in names]
    sizes = [int(np.prod(s)) for s in shapes]
    theta = np.concatenate([base["policy"][k].float().numpy().ravel() for k in names]).astype(np.float64)
    rms = np.concatenate([np.full(n, float(np.sqrt(np.mean(base["policy"][k].float().numpy() ** 2))) + 1e-8)
                          for k, n in zip(names, sizes)])
    print(f"ES on {names}: {theta.size} weights; base {args.model}", flush=True)
    rng = np.random.default_rng(args.seed)
    m, v, b1, b2 = np.zeros_like(theta), np.zeros_like(theta), 0.9, 0.999

    def write(vec, path):
        ck = copy.copy(base)
        pol = dict(base["policy"])
        i = 0
        for k, s, n in zip(names, shapes, sizes):
            pol[k] = torch.from_numpy(vec[i:i + n].reshape(s)).to(base["policy"][k].dtype)
            i += n
        ck["policy"] = pol
        ck["es_from"] = args.model
        torch.save(ck, path)
        return path

    def val_jobs(path, tag):
        return [((tag, s), (path, args.env, 0, args.val_episodes, s, "cpu", None, opts)) for s in args.val_seeds]

    def val_score(res, tag):
        rows = [r for s in args.val_seeds for r in res[(tag, s)]]
        rows = [r for r in rows if r.get("start_intrusions", 0) == 0]
        return float(np.mean([clean(r) for r in rows])), float(np.mean([cost(r) for r in rows]))

    log = dict(args=vars(args), names=names, gens=[], vals=[])
    best = None
    t0 = time.time()
    pool = mp.get_context("spawn").Pool(args.procs, maxtasksperchild=1)
    try:
        for g in range(args.generations + 1):
            jobs, eps = [], None
            do_val = g % args.val_every == 0 or g == args.generations
            if do_val:
                cpath = write(theta, str(out / "tmp" / "centre.pt"))
                jobs += val_jobs(cpath, "val")
            if g < args.generations:
                eps = rng.standard_normal((args.pairs, theta.size))
                gs = args.gen_seed0 + g
                for i in range(args.pairs):
                    for sgn in (1, -1):
                        path = write(theta + sgn * args.sigma * rms * eps[i], str(out / "tmp" / f"p{i}_{sgn}.pt"))
                        jobs.append(((i, sgn), (path, args.env, 0, args.episodes, gs, "cpu", None, opts)))
            res = dict(pool.imap_unordered(_run, jobs))
            if do_val:
                vs = val_score(res, "val")
                log["vals"].append(dict(gen=g, clean=vs[0], cost=vs[1], seconds=time.time() - t0))
                flag = ""
                if best is None or vs[0] > best[0] + 1e-12 or (abs(vs[0] - best[0]) <= 1e-12 and vs[1] < best[1]):
                    best = vs
                    write(theta, str(out / "es_best.pt"))
                    flag = " NEW BEST -> es_best.pt"
                print(f"val gen {g}: clean {100 * vs[0]:.2f} % (start left out), cost {vs[1]:.4f}{flag} "
                      f"({time.time() - t0:.0f} s)", flush=True)
            if g == args.generations:
                break
            diffs = np.zeros(args.pairs)
            for i in range(args.pairs):
                a, b = keyed(res[(i, 1)]), keyed(res[(i, -1)])
                ks = [k for k in a if k in b and a[k].get("start_intrusions", 0) == 0]
                diffs[i] = np.mean([cost(b[k]) - cost(a[k]) for k in ks])    # > 0: +eps is better (lower cost)
            w = centered_ranks(diffs)
            grad = (w[:, None] * eps).sum(0) / (args.pairs * args.sigma)       # ascent direction in noise units
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad ** 2
            step = (m / (1 - b1 ** (g + 1))) / (np.sqrt(v / (1 - b2 ** (g + 1))) + 1e-8)
            theta = theta + args.lr * rms * step
            mean_cost = float(np.mean([cost(r) for (i, s), rows in res.items() if i != "val" for r in rows
                                       if r.get("start_intrusions", 0) == 0]))
            log["gens"].append(dict(gen=g, mean_cost=mean_cost, diff_mean=float(diffs.mean()),
                                    diff_abs=float(np.abs(diffs).mean()), seconds=time.time() - t0))
            print(f"gen {g}: mean cost {mean_cost:.4f}, |pair diff| {np.abs(diffs).mean():.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            with open(out / "es_log.json", "w") as f:
                json.dump(log, f, indent=1)
    finally:
        pool.close()
        pool.join()
    write(theta, str(out / "es_last.pt"))
    with open(out / "es_log.json", "w") as f:
        json.dump(log, f, indent=1)
    print(f"done: best tuning clean {100 * best[0]:.2f} % -> {out / 'es_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
