"""Checks of the heading mask (2026-11-21), no training.

  python -m scripts.check_heading_mask compare A.json B.json
      Per-flight rows of two scripts.competition_eval outputs: exact equality of all metric keys (check: the mask
      off gives the same flights as the code before the mask). Exit code 1 if a row differs.
  python -m scripts.check_heading_mask losses A_summary.json B_summary.json
      Training curves of two scripts.train smoke runs: exact equality of the losses. Exit code 1 if they differ.
  python -m scripts.check_heading_mask timing --model M --kind sa|ma --episodes N --seed S
      Time per decision with the mask off and on (same seeds, same policy, alternating rounds; the faster round
      counts). Writes results/check_heading_mask_timing_KIND.json.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
import numpy as np


def compare(a, b):
    ra, rb = json.loads(Path(a).read_text())["rows"], json.loads(Path(b).read_text())["rows"]
    key = lambda r: (r["episode_index"], r["aircraft"])
    da, db = {key(r): r for r in ra}, {key(r): r for r in rb}
    keys = sorted(set(ra[0]) & set(rb[0]) - {"episode_index", "aircraft"}) if ra and rb else []
    diff = [(k, m, da[k].get(m), db[k].get(m)) for k in da if k in db for m in keys if da[k].get(m) != db[k].get(m)]
    same = len(ra) == len(rb) and set(da) == set(db) and not diff
    print(f"compare: {len(ra)} and {len(rb)} flights, {len(keys)} keys per flight; identical: {same}")
    for d in diff[:10]:
        print("  differs:", d)
    return same


def losses(a, b):
    ca, cb = json.loads(Path(a).read_text())["curve"], json.loads(Path(b).read_text())["curve"]
    ks = ["policy_loss", "value_loss", "log_std"]
    va, vb = [[r[k] for k in ks] for r in ca], [[r[k] for k in ks] for r in cb]
    same = va == vb and len(va) > 0
    print(f"losses: {len(va)} and {len(vb)} updates; identical: {same}")
    for x, y in zip(va, vb):
        print("  ", x, y)
    return same


def timing(model, kind, episodes, seed, rounds=2):
    import torch
    from rl_tf.envs import FlightEnv, rollout_episode, v2_kwargs
    from scripts.competition_eval import _load_member
    torch.set_num_threads(1)
    policy, fn, rew, lidar, history, c = _load_member(model, "cpu")
    out = {}
    for r in range(rounds):
        for m in (0, 1):
            v2 = v2_kwargs({**c, "heading_mask": m})
            with contextlib.redirect_stdout(io.StringIO()):
                env = FlightEnv("single" if kind == "sa" else "multi", 10, v2=v2, **rew)
            t_step, n_dec = [0.0], [0]
            step0 = env.step

            def step(acts):
                t = time.perf_counter()
                res = step0(acts)
                t_step[0] += time.perf_counter() - t
                n_dec[0] += len(acts)
                return res
            env.step = step
            rows, t0 = [], time.perf_counter()
            for i in range(episodes):
                ep, _, _ = rollout_episode(env, fn, seed + i, lidar, history)
                rows += list(ep.values())
            wall = time.perf_counter() - t0
            env.close()
            rec = {"ms_per_decision": 1000 * wall / max(n_dec[0], 1),
                   "ms_env_step_per_decision": 1000 * t_step[0] / max(n_dec[0], 1), "decisions": n_dec[0]}
            if m:
                dec = sum(r["mask_decisions"] for r in rows)
                rec["projected_share"] = sum(r["mask_projected"] for r in rows) / max(dec, 1)
                rec["fallbacks"] = int(sum(r["mask_fallbacks"] for r in rows))
            best = out.get(m)
            if best is None or rec["ms_per_decision"] < best["ms_per_decision"]:
                out[m] = rec
            print(f"round {r} mask {m}: {rec}", flush=True)
    res = {"status": "EXPLORATORY", "model": model, "kind": kind, "episodes": episodes, "first_seed": seed,
           "mask_off": out[0], "mask_on": out[1],
           "overhead_decision": out[1]["ms_per_decision"] / out[0]["ms_per_decision"] - 1.0,
           "overhead_env_step": out[1]["ms_env_step_per_decision"] / out[0]["ms_env_step_per_decision"] - 1.0}
    print(json.dumps(res, indent=1))
    Path("results").mkdir(exist_ok=True)
    Path(f"results/check_heading_mask_timing_{kind}.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["compare", "losses", "timing"])
    p.add_argument("files", nargs="*")
    p.add_argument("--model")
    p.add_argument("--kind", choices=["sa", "ma"], default="sa")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    if a.what == "compare":
        sys.exit(0 if compare(*a.files) else 1)
    if a.what == "losses":
        sys.exit(0 if losses(*a.files) else 1)
    timing(a.model, a.kind, a.episodes, a.seed)


if __name__ == "__main__":
    main()
