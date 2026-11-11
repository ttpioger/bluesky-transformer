"""Check: the same seed gives bit-identical output, alone and inside a process pool.

Runs scripts.train three times with one seed: run a alone, then runs b and c at the same time.
Compares the episode list, the training curve and every actor weight. Show this check before the first real run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import torch


def digest(name, out_dir):
    s = json.load(open(f"{out_dir}/results/{name}_summary.json"))
    ck = torch.load(f"{out_dir}/models/{name}.pt", map_location="cpu", weights_only=False)
    h = hashlib.sha256(json.dumps([s["episodes"], s["curve"], s["greedy_eval"]], sort_keys=True).encode())
    for k in sorted(ck["actor"]):
        h.update(ck["actor"][k].numpy().tobytes())
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setting", choices=["single", "multi"], default="multi")
    p.add_argument("--encoder", choices=["cnn", "set_tf"], default="set_tf")
    p.add_argument("--jepa", choices=["off", "on"], default="on")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="determinism")
    args = p.parse_args(argv)

    def run(tag):
        name = f"det_{tag}"
        subprocess.run([sys.executable, "-m", "scripts.train", "--name", name, "--setting", args.setting,
                        "--encoder", args.encoder, "--jepa", args.jepa, "--steps", str(args.steps),
                        "--rollout-steps", "200", "--seed", str(args.seed), "--out-dir", args.out_dir],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return digest(name, args.out_dir)

    a = run("a")
    with ThreadPoolExecutor(2) as pool:
        b, c = pool.map(run, ["b", "c"])
    print(f"alone      {a}\nin pool 1  {b}\nin pool 2  {c}")
    ok = a == b == c
    print("BIT-IDENTICAL: yes" if ok else "BIT-IDENTICAL: NO")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
