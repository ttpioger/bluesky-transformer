"""Run cells x seeds for one setting in a pool of separate processes (BlueSky: one environment per process).

Stages: 0 collect transitions (serial, also warms the BlueSky cache); 1 imitation test per encoder;
2 per cell and seed: JEPA pretraining (cells B, C), then RL. Stop rules: no RL for an encoder that
fails the imitation test; no RL after a collapsed JEPA pretraining. This is a training run:
start it through ~/bin/notify_run.sh. Each job writes logs/JOB.log and results/JOB_summary.json.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from jepa_set.common import CELLS, write_json


def run(job, module, argv):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    Path("logs").mkdir(exist_ok=True)
    t0 = time.time()
    with open(f"logs/{job}.log", "w") as log:
        code = subprocess.run([sys.executable, "-m", module, *map(str, argv)], stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    print(f"[{time.strftime('%H:%M:%S')}] {job}: exit {code}, {time.time() - t0:.0f} s", flush=True)
    return {"job": job, "exit": code, "seconds": time.time() - t0}


def summary(job):
    f = Path(f"results/{job}_summary.json")
    return json.loads(f.read_text()) if f.exists() else {}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setting", choices=["single", "multi"], required=True)
    p.add_argument("--cells", nargs="+", default=["R", "A", "B", "C"], choices=list(CELLS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=6, help="Parallel processes, one torch thread each.")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--reset-seed", type=int, default=1)
    p.add_argument("--transitions", type=int, default=6000)
    p.add_argument("--jepa-epochs", type=int, default=30)
    p.add_argument("--imitate-epochs", type=int, default=60)
    p.add_argument("--jepa-aux-coef", type=float, default=0.1)
    p.add_argument("--skip-imitate", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default="", help="Suffix for all job names (for example _smoke).")
    p.add_argument("--train-args", nargs=argparse.REMAINDER, default=[], help="Extra arguments for scripts.train.")
    args = p.parse_args(argv)
    sa = "sa" if args.setting == "single" else "ma"
    base = ["--setting", args.setting, "--n-agents", args.n_agents, "--reset-seed", args.reset_seed]
    dev = ["--device", args.device]
    data = f"data/transitions_{args.setting}{args.tag}.npz"
    jobs = []
    if not Path(data).exists():
        jobs.append(run(f"collect_{sa}{args.tag}", "scripts.collect", [*base, "--transitions", args.transitions, "--out", data]))
        if jobs[-1]["exit"] != 0:
            sys.exit(f"collect failed: see logs/collect_{sa}{args.tag}.log")
    encoders = sorted({CELLS[c][0] for c in args.cells})
    ok_enc = {e: True for e in encoders}
    with ThreadPoolExecutor(args.workers) as pool:
        if not args.skip_imitate:
            names = {e: f"imit_{sa}_{e}{args.tag}" for e in encoders}
            jobs += list(pool.map(lambda e: run(names[e], "scripts.imitate", [
                "--name", names[e], "--data", data, "--encoder", e, "--epochs", args.imitate_epochs, *base, *dev]), encoders))
            for e in encoders:
                ok_enc[e] = bool(summary(names[e]).get("pass", False))
                if not ok_enc[e]:
                    print(f"STOP RULE: encoder {e} failed the imitation test. No RL run for this encoder.", flush=True)

        def chain(cs):
            cell, seed = cs
            enc, jepa = CELLS[cell]
            name = f"{sa}_{cell}_s{seed}{args.tag}"
            out = []
            targv = ["--name", name, "--encoder", enc, "--jepa", jepa, "--seed", seed, "--steps", args.steps, *base, *dev,
                     "--jepa-aux-coef", args.jepa_aux_coef, *args.train_args]
            if jepa == "on":
                jname = f"jepa_{name}"
                out.append(run(jname, "scripts.pretrain_jepa", ["--name", jname, "--data", data, "--encoder", enc,
                                                                 "--epochs", args.jepa_epochs, "--seed", seed, *dev]))
                if out[-1]["exit"] != 0 or summary(jname).get("collapsed", True):
                    print(f"STOP RULE: {jname} failed or collapsed. No RL run for {name}.", flush=True)
                    return out
                targv += ["--jepa-ckpt", f"checkpoints/{jname}.pt"]
            out.append(run(name, "scripts.train", targv))
            return out

        todo = [(c, s) for c in args.cells for s in args.seeds if ok_enc[CELLS[c][0]]]
        for res in pool.map(chain, todo):
            jobs += res
    failed = [j["job"] for j in jobs if j["exit"] != 0]
    write_json(f"results/grid_{args.setting}{args.tag}_summary.json", {
        "status": "EXPLORATORY", "setting": args.setting, "cells": args.cells, "seeds": args.seeds,
        "imitation_pass": ok_enc, "failed_jobs": failed, "jobs": jobs, "args": vars(args)})
    subprocess.run([sys.executable, "-m", "scripts.compare", "--tag", args.tag])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
