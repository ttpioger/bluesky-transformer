"""Validation curves of the long DAIC runs from Weights & Biases (read only; starts no run).

Question: does val/clean (seeds 7 and 11, 100 episodes each) still rise, or does it peak and drop (overfit)?
Reads val/* of each W&B run (entity lego_block, project bluesky-rl-transformer, run id = run name), prints a table
per run (env_steps of each snapshot, clean, goal, events, flight time) and the best snapshot by the leaderboard rule
(clean, then goal), and saves one figure: val/clean and val/goal against env steps.

    ../.venv/bin/python analysis/wandb_val_curves.py                       # the three long runs
    ../.venv/bin/python analysis/wandb_val_curves.py --runs ma_v2m_ft --out figures/val_ma.png
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["val/env_steps", "val/clean", "val/goal", "val/intrusion_events", "val/restricted_area_events",
        "val/sector_exit_events", "val/flight_time", "val/best_clean"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3 (light surface)


def fetch(api, path, name):
    try:
        run = api.run(f"{path}/{name}")
    except Exception as e:                            # not created yet (job not started) or wrong id
        return None, str(e)
    # history(samples=...) is fast; scan_history pages through every training step (minutes). Val rows are few
    # (one per 1 M steps), so a sample cap of 10,000 returns all of them.
    hist = run.history(keys=KEYS[:-1], samples=10_000, pandas=False)
    rows = [{k: h.get(k) for k in KEYS} for h in hist if h.get("val/env_steps") is not None]
    rows.sort(key=lambda r: r["val/env_steps"])
    return {"state": run.state, "rows": rows}, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["ma_v2m_ft", "sa_v2m_c2L", "sa_v2m_soft_c2L"])
    p.add_argument("--path", default="lego_block/bluesky-rl-transformer")
    p.add_argument("--out", default=str(ROOT / "figures" / "val_curves_long_runs.png"))
    p.add_argument("--json", default=None, help="Also write the fetched rows to this JSON file.")
    a = p.parse_args()
    import wandb
    api = wandb.Api(timeout=60)
    data = {}
    for name in a.runs:
        d, err = fetch(api, a.path, name)
        if d is None:
            print(f"{name}: no W&B run ({err.splitlines()[0][:120]})")
            continue
        data[name] = d
        print(f"\n{name} (W&B state {d['state']}): {len(d['rows'])} validation records")
        if not d["rows"]:
            continue
        print(f"{'env_steps':>12} {'clean %':>8} {'goal %':>7} {'intr/fl':>8} {'area/fl':>8} {'sect/fl':>8} {'time s':>7}")
        for r in d["rows"]:
            print(f"{int(r['val/env_steps']):>12,} {100 * r['val/clean']:>8.2f} {100 * r['val/goal']:>7.2f} "
                  f"{r['val/intrusion_events']:>8.4f} {r['val/restricted_area_events']:>8.4f} "
                  f"{r['val/sector_exit_events']:>8.4f} {r['val/flight_time']:>7.1f}")
        best = max(d["rows"], key=lambda r: (r["val/clean"], r["val/goal"]))
        print(f"best on seeds 7+11: env_steps {int(best['val/env_steps']):,}, clean {100 * best['val/clean']:.2f} %, "
              f"goal {100 * best['val/goal']:.2f} %")
    if a.json:
        Path(a.json).write_text(json.dumps(data, indent=1))
    plotted = {n: d for n, d in data.items() if d["rows"]}
    if not plotted:
        print("\nNo validation records yet: no figure.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for (name, d), c in zip(plotted.items(), COLORS):
        x = [r["val/env_steps"] / 1e6 for r in d["rows"]]
        for ax, key in zip(axes, ["val/clean", "val/goal"]):
            y = [100 * r[key] for r in d["rows"]]
            ax.plot(x, y, color=c, lw=2, marker="o", ms=4, label=name)
            ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), textcoords="offset points", xytext=(5, 0),
                        fontsize=8, color="#52514e", va="center")
    for ax, t in zip(axes, ["Clean flights (%)", "Goal reached (%)"]):
        ax.set_title(t, fontsize=11, loc="left")
        ax.set_xlabel("Training env steps (millions)")
        ax.grid(alpha=0.3, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("% of validation flights")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Validation on seeds 7 and 11 (100 episodes each) during the long runs", fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
