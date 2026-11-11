"""One table over cells and seeds (mean and std over seeds) from results/*_summary.json.

All results are EXPLORATORY. An overfit pass shows only that the learning loop works.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np

MATCH = [("environment", "not matched", "reference: 2D lidar navigation; here: BlueSky competition environment"),
         ("aircraft count", "not matched", "reference: up to 8 agents in 8 parallel environments; here: 1 or 2 aircraft, 1 environment"),
         ("steps", "not matched", "reference: up to 1e7; here: the --steps value of the run"),
         ("reward and reward scale", "not matched", "reference: dense progress reward, no scale; here: time and event penalties, scale 0.001"),
         ("encoder input", "not matched", "reference: 120 lidar rays, 4 stacked frames; here: object lists, 1 frame")]


def ms(vals):
    vals = [v for v in vals if v is not None]
    return "n/a" if not vals else f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None, help="Default: results/compareTAG.md")
    p.add_argument("--tag", default="", help="Use only the runs with this name suffix (for example _smoke).")
    args = p.parse_args(argv)
    args.out = args.out or f"results/compare{args.tag}.md"
    pattern = re.compile(rf"^(sa|ma)_[RABC]_s\d+{re.escape(args.tag)}$")
    runs = []
    for f in sorted(Path(args.results).glob("*_summary.json")):
        d = json.loads(f.read_text())
        if pattern.match(d.get("name", "")) and d.get("last_window"):
            runs.append(d)
    lines = ["# Overfit comparison (status: EXPLORATORY)", "",
             "One fixed scenario per setting. Mean ± std over seeds. Windows: first and last 20 agent-episodes of training (with exploration noise).",
             "An overfit pass shows only that the learning loop works. This table does not show that a cell is better or worse in general.", ""]
    head = ["setting", "cell", "encoder", "jepa", "seeds", "n_params actor", "pass", "waypoint (last)", "intrusion ev.",
            "restricted ev.", "sector-exit ev.", "flight time s (last)", "return (first)", "return (last)",
            "steps to criterion", "greedy waypoint", "greedy flight time s", "s/step"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for setting in ["single", "multi"]:
        for cell in ["R", "A", "B", "C"]:
            g = [r for r in runs if r["setting"] == setting and r["cell"] == cell]
            if not g:
                continue
            lw = lambda k: ms([r["last_window"][k] for r in g])
            gre = lambda k: ms([float(np.mean([a[k] for a in r["greedy_eval"].values()])) for r in g if r.get("greedy_eval")])
            stc = [r["steps_to_criterion"] for r in g]
            row = [setting, cell, g[0]["encoder"], g[0]["jepa"], str(sorted(r["seed"] for r in g)),
                   str(g[0]["n_params"]["actor"]), f"{sum(r['pass'] for r in g)}/{len(g)}",
                   lw("waypoint_reached"), lw("intrusion_events"), lw("restricted_area_events"), lw("sector_exit_events"),
                   lw("flight_time"), ms([r["first_window"]["ep_return"] for r in g]), lw("ep_return"),
                   f"{ms(stc)} ({sum(s is not None for s in stc)}/{len(g)} reached)", gre("waypoint_reached"),
                   gre("flight_time"), ms([r["seconds_per_step"] for r in g])]
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Comparison with the reference method (avbiswas/navigation-mappo-rl)", "",
              "| item | mark | detail |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in MATCH]
    lines += ["", "No number in this table can be compared with a number of the reference method."]
    text = "\n".join(lines) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)
    print(f"Wrote {args.out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
