"""Run scripts.train for several settings and seeds, one job after the other (each job uses all workers).

A failed job does not stop the next jobs. Each job writes logs/NAME.log, logs/NAME.exit and
results/NAME_summary.json. After a successful job, scripts.evaluate writes gifs/NAME.gif (one held-out
scenario, lidar rays drawn) and results/NAME_eval.json; --gif-episodes 0 skips this step. Flags after --train-args go to scripts.train (last option).
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from rl_tf.common import write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings", nargs="+", default=["single", "multi"], choices=["single", "multi"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=400000)
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tag", default="", help="Suffix for all job names.")
    p.add_argument("--gif-episodes", type=int, default=1, help="Episodes in gifs/NAME.gif after each job (0 = no GIF).")
    p.add_argument("--gif-seed", type=int, default=900000000, help="Scenario seed of the GIF (held out).")
    p.add_argument("--train-args", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args(argv)
    Path("logs").mkdir(exist_ok=True)
    jobs = []
    for setting in args.settings:
        for seed in args.seeds:
            name = f"{'sa' if setting == 'single' else 'ma'}_tf_s{seed}{args.tag}"
            cmd = [sys.executable, "-W", "ignore", "-m", "scripts.train", "--name", name, "--setting", setting,
                   "--n-agents", str(args.n_agents), "--seed", str(seed), "--steps", str(args.steps),
                   "--workers", str(args.workers), *args.train_args]
            with open(f"logs/{name}.log", "w") as log:
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            Path(f"logs/{name}.exit").write_text(f"{code}\n")
            if code == 0 and args.gif_episodes > 0:
                ev = [sys.executable, "-W", "ignore", "-m", "scripts.evaluate", "--model", f"models/{name}.pt",
                      "--episodes", str(args.gif_episodes), "--reset-seed", str(args.gif_seed), "--gif", f"gifs/{name}.gif"]
                with open(f"logs/{name}_gif.log", "w") as log:
                    gif_code = subprocess.run(ev, stdout=log, stderr=subprocess.STDOUT).returncode
                print(f"{name}: gif exit {gif_code}", flush=True)
            jobs.append({"name": name, "exit_code": code, "summary": f"results/{name}_summary.json"})
            print(f"{name}: exit {code}", flush=True)
    write_json(f"results/run_seeds{args.tag}_summary.json", {"status": "EXPLORATORY", "jobs": jobs, "args": vars(args)})
    sys.exit(1 if any(j["exit_code"] for j in jobs) else 0)


if __name__ == "__main__":
    main()
