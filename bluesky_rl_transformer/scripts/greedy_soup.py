"""Greedy weight soup on tuning seeds (2026-11-23, user OK "Run option A").

Idea (Wortsman et al. 2022, "Model soups"): the weight average of save points of one run can be better than each
save point. Greedy: start with the best save point, then add a save point to the average only if the average gets
better on the tuning flights. Here a round tests the top --width remaining candidates in parallel and keeps the best
one (forward selection); it stops when no candidate improves the score.

Seeds: the soup is chosen on TUNING seeds (default 13 and 17), never on 7 / 11 (model choice) or 42 (leaderboard).
Confirm the result afterwards with scripts.competition_eval on seeds 7 and 11 (1000 episodes).

Score (less noisy than the clean rate of all flights): flights with a loss of separation at the start are left out
(they fail for every model, start_intrusions > 0). Score = clean share of the other flights; tie-break: fewer
avoidable events per flight (later intrusions + area entries + sector exits + goal not reached). All candidates fly
the same scenarios (same seeds, same episodes), so the comparison is paired.

Evaluation: the competition protocol of scripts.competition_eval (_job) with the given test-time options (mask, route,
fb2). Single agent: one process per (model, seed) (see competition_eval.run). Multi agent: --split processes per seed.

Usage (from bluesky_rl_transformer/, CPU node):
  python -m scripts.greedy_soup --env sa --models CKPT_DIR [more .pt or dirs] --episodes 300 --procs 32 \
      --heading-mask 1 --mask-mode arc --mask-intruders 1 --mask-dist-km 9.6 --mask-route 1 --mask-route-km 3 \
      --mask-fb2 1 --out-dir /path/soup_sa
Outputs: OUT_DIR/soup.pt (policy weights = mean of the chosen save points; args from the first one),
OUT_DIR/soup_log.json (all scores), OUT_DIR/tmp/*.pt (the averages that were tested).
"""
from __future__ import annotations
import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
import numpy as np
import torch
from rl_tf.heading_mask import add_mask_args, mask_overrides


def expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "step_*.pt")))
        else:
            out.append(p)
    return out


def label(path):
    p = Path(path)
    return f"{p.parent.name}/{p.stem}" if p.stem.startswith("step_") else p.stem


def average(paths, out):
    cks = [torch.load(f, map_location="cpu", weights_only=False) for f in paths]
    avg = {}
    for k, v in cks[0]["policy"].items():
        if torch.is_floating_point(v):
            avg[k] = torch.stack([c["policy"][k].float() for c in cks]).mean(0).to(v.dtype)
        else:
            avg[k] = cks[0]["policy"][k]
    res = {k: v for k, v in cks[0].items() if k != "policy"}
    res["policy"] = avg
    res["averaged_from"] = list(paths)
    torch.save(res, out)
    return out


def _run(job):
    from scripts.competition_eval import _job
    name, seed, rest = job[0], job[1], job[2]
    rows = _job(rest)
    return name, seed, rows


def evaluate(models, kind, seeds, episodes, opts, procs, split):
    """models: {name: path}. Returns {name: {(seed, episode, aircraft): row}}."""
    jobs = []
    n_split = 1 if kind == "sa" else max(1, split)
    edges = np.linspace(0, episodes, n_split + 1).astype(int)
    for name, path in models.items():
        for s in seeds:
            for i in range(n_split):
                jobs.append((name, s, (path, kind, int(edges[i]), int(edges[i + 1]), s, "cpu", None, opts)))
    out = {n: {} for n in models}
    with mp.get_context("spawn").Pool(min(procs, len(jobs)), maxtasksperchild=1) as pool:
        for name, s, rows in pool.imap_unordered(_run, jobs):
            for r in rows:
                out[name][(s, r["episode_index"], r["aircraft"])] = r
    return out


def clean(r):
    return (r["waypoint_reached"] >= 1 and r["intrusion_events"] == 0 and r["restricted_area_events"] == 0
            and r["sector_exit_events"] == 0)


def events(r):
    later = r.get("intrusion_events_no_start", r["intrusion_events"])
    return later + r["restricted_area_events"] + r["sector_exit_events"] + (1 - min(1, r["waypoint_reached"]))


def score(rows, keys):
    """(clean share of the flights in keys, mean avoidable events per flight)."""
    return (float(np.mean([clean(rows[k]) for k in keys])), float(np.mean([events(rows[k]) for k in keys])))


def better(a, b):
    return a[0] > b[0] + 1e-12 or (abs(a[0] - b[0]) <= 1e-12 and a[1] < b[1] - 1e-12)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["sa", "ma"], required=True)
    p.add_argument("--models", nargs="+", required=True, help="save points (.pt) or run checkpoint dirs")
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 17], help="tuning seeds (default 13 17)")
    p.add_argument("--episodes", type=int, default=300, help="episodes per seed (default 300)")
    p.add_argument("--procs", type=int, default=32)
    p.add_argument("--split", type=int, default=4, help="MA: processes per seed (default 4)")
    p.add_argument("--width", type=int, default=8, help="candidates tested per greedy round (default 8)")
    p.add_argument("--max-ingredients", type=int, default=8)
    p.add_argument("--out-dir", required=True)
    add_mask_args(p, from_checkpoint=True)
    args = p.parse_args(argv)
    for s in args.seeds:
        if s in (7, 11, 42):
            raise SystemExit(f"seed {s} is for model choice or the leaderboard; use other tuning seeds")
    out = Path(args.out_dir)
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    opts = {"mask": mask_overrides(args)}
    cands = {label(f): f for f in expand(args.models)}
    log = dict(env=args.env, seeds=args.seeds, episodes=args.episodes, mask=opts["mask"], candidates=cands, rounds=[])
    t0 = time.time()
    print(f"stage 1: {len(cands)} candidates, seeds {args.seeds}, {args.episodes} episodes each", flush=True)
    res = evaluate(cands, args.env, args.seeds, args.episodes, opts, args.procs, args.split)
    keys = sorted(set.intersection(*[set(r) for r in res.values()]))
    keys = [k for k in keys if all(res[n][k].get("start_intrusions", 0) == 0 for n in res)]
    sc = {n: score(res[n], keys) for n in cands}
    ranked = sorted(cands, key=lambda n: (-sc[n][0], sc[n][1]))
    for n in ranked:
        print(f"  {n:45s} clean {100 * sc[n][0]:6.2f} %  events/flight {sc[n][1]:.4f}", flush=True)
    log["single"] = {n: sc[n] for n in ranked}
    log["n_flights_scored"] = len(keys)
    soup, cur, cur_rows = [ranked[0]], sc[ranked[0]], res[ranked[0]]
    remaining = ranked[1:]
    print(f"start: {soup[0]} ({100 * cur[0]:.2f} %), {len(keys)} flights scored ({time.time() - t0:.0f} s)", flush=True)
    r = 0
    while remaining and len(soup) < args.max_ingredients:
        r += 1
        test = remaining[: args.width]
        paths = {}
        for c in test:
            paths[c] = average([cands[n] for n in soup + [c]], str(out / "tmp" / f"r{r}_{len(paths)}.pt"))
        rr = evaluate(paths, args.env, args.seeds, args.episodes, opts, args.procs, args.split)
        s2 = {c: score(rr[c], keys) for c in test}
        best = sorted(test, key=lambda c: (-s2[c][0], s2[c][1]))[0]
        win = sum(clean(rr[best][k]) and not clean(cur_rows[k]) for k in keys)
        loss = sum(clean(cur_rows[k]) and not clean(rr[best][k]) for k in keys)
        log["rounds"].append(dict(round=r, soup=list(soup), tested={c: s2[c] for c in test}, best=best,
                                  win=win, loss=loss))
        print(f"round {r}: soup {len(soup)} + best {best}: {100 * s2[best][0]:.2f} % (now {100 * cur[0]:.2f} %),"
              f" paired +{win}/-{loss} ({time.time() - t0:.0f} s)", flush=True)
        if not better(s2[best], cur):
            print("no candidate improves the soup: stop", flush=True)
            break
        soup.append(best)
        cur, cur_rows = s2[best], rr[best]
        remaining = [c for c in remaining if c != best]
    final = average([cands[n] for n in soup], str(out / "soup.pt"))
    log.update(soup=soup, soup_score=cur, final=final, seconds=time.time() - t0)
    with open(out / "soup_log.json", "w") as f:
        json.dump(log, f, indent=1)
    print(f"soup of {len(soup)}: {soup} -> {final}  score {100 * cur[0]:.2f} % (single best "
          f"{100 * sc[ranked[0]][0]:.2f} %)", flush=True)


if __name__ == "__main__":
    main()
