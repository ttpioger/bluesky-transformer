"""Weight average of training snapshots (2026-11-22). Averages every floating tensor of the policy state_dict;
args and dims come from the last snapshot. Snapshots of one run only (same start model, same basin).

Usage: python -m analysis.avg_checkpoints --out models/NAME.pt --min-step 23000000 --max-step 25000000 CKPT_DIR
"""
import argparse
import glob
import os

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir", help="models/<run>_ckpt (files step_<10 digits>.pt)")
    p.add_argument("--min-step", type=int, required=True)
    p.add_argument("--max-step", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    files = []
    for f in sorted(glob.glob(os.path.join(a.ckpt_dir, "step_*.pt"))):
        s = int(os.path.basename(f)[5:-3])
        if a.min_step <= s <= a.max_step:
            files.append((s, f))
    if len(files) < 2:
        raise SystemExit(f"need at least 2 snapshots in [{a.min_step}, {a.max_step}], found {len(files)}")
    cks = [torch.load(f, map_location="cpu", weights_only=False) for _, f in files]
    avg = {}
    for k, v in cks[0]["policy"].items():
        if torch.is_floating_point(v):
            avg[k] = torch.stack([c["policy"][k].float() for c in cks]).mean(0).to(v.dtype)
        else:
            avg[k] = cks[-1]["policy"][k]
    out = {k: v for k, v in cks[-1].items() if k != "policy"}
    out["policy"] = avg
    out["averaged_from"] = [f for _, f in files]
    out["env_steps"] = files[-1][0]
    torch.save(out, a.out)
    print(f"{a.out}: mean of {len(files)} snapshots, steps {[s for s, _ in files]}")


if __name__ == "__main__":
    main()
