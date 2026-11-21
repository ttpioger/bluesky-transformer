"""Place our results against the unofficial BlueSky-Gym leaderboard (read only, 2026-11-21).

Source (public, self-reported, not the official judging):
  https://huggingface.co/spaces/yojul/bluesky-gym-leaderboard-unofficial
The leaderboard uses seed 42 (official protocol). Our seed-7 / seed-11 results are NOT matched to it; they only
give an approximate placement. Never use seed 42 for a choice. This script never submits anything.

Usage (from bluesky_rl_transformer/):
  python -m analysis.leaderboard_compare results/s7k_X_competition_sa.json [more files ...]
      [--val-json results/NAME_val.json] [--wandb-run NAME] [--snapshot results/leaderboard_*.json]
      [--offline] [--markdown]
Each run without --snapshot / --offline fetches the leaderboard once and saves results/leaderboard_<YYYYmmdd-HHMM>.json.
Ranking rule: clean % (desc), goal % (desc), intrusions, area, sector (asc), flight time (asc).
Clean flight = goal reached and 0 intrusion, 0 area, 0 sector events.
"""
from __future__ import annotations
import argparse
import datetime as dt
import glob
import json
import re
import urllib.request
from pathlib import Path

API = "https://yojul-bluesky-gym-leaderboard-unofficial.hf.space/gradio_api/call/refresh"
COLS = ["clean", "goal", "intr", "area", "sector", "ftime"]
# Highest possible clean share (unavoidable flights), from the laptop analysis of 2026-11-21. None = not known.
MAX_CLEAN = {("sa", 42): 90.4, ("sa", 7): 90.0, ("ma", 42): 99.56, ("ma", 7): 99.41}
# Snapshot read on 2026-11-21 (fallback when the API does not answer).
SNAPSHOT_20261121 = {
    "sa": [("TEAM ROBIN", 89.2, 100.0, 0.116, 0.003, 0.000, 912.6), ("somnathab3", 89.2, 100.0, 0.116, 0.003, 0.000, 997.2),
           ("Smart", 89.2, 100.0, 0.123, 0.003, 0.000, 1094.3), ("Natasumushi", 89.1, 99.8, 0.128, 0.002, 0.000, 1018.6),
           ("IASL", 88.9, 99.9, 0.127, 0.003, 0.000, 1025.7), ("Moin!", 88.1, 99.9, 0.135, 0.002, 0.001, 1015.0),
           ("solomon", 79.3, 96.1, 0.211, 0.044, 0.000, 1265.4), ("yojul", 48.4, 89.8, 0.551, 0.134, 0.043, 1762.2)],
    "ma": [("TEAM ROBIN", 99.55, 100.00, 0.0000, 0.0056, 0.0000, 887.9), ("somnathab3", 99.55, 100.00, 0.0000, 0.0065, 0.0000, 945.9),
           ("IASL", 99.34, 99.68, 0.0008, 0.0039, 0.0000, 980.4), ("Smart", 98.11, 98.86, 0.0060, 0.0051, 0.0001, 1107.3),
           ("solomon", 93.67, 98.04, 0.0178, 0.0527, 0.0001, 1024.6)]}


def fetch(timeout=60):
    """Both tables from the read API: {"sa": [...], "ma": [...]}, each row a dict with Team, COLS and Submitted."""
    req = urllib.request.Request(API, data=json.dumps({"data": ["Events per flight"]}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    eid = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["event_id"]
    text = urllib.request.urlopen(f"{API}/{eid}", timeout=timeout).read().decode()
    line = next(l for l in text.splitlines() if l.startswith("data:"))
    tables = json.loads(line[len("data:"):])
    out = {}
    for kind, t in zip(["sa", "ma"], tables):
        h = t["headers"]
        idx = {k: h.index(n) for k, n in [("clean", "Clean flights [%]"), ("goal", "Goal reached [%]"),
                                           ("intr", "Intrusions"), ("area", "Restricted area"),
                                           ("sector", "Outside sector"), ("ftime", "Flight time [s]")]}
        out[kind] = [{"team": re.sub(r"\*", "", r[h.index("Team")]), **{k: float(r[i]) for k, i in idx.items()},
                      "submitted": r[h.index("Submitted")] if "Submitted" in h else ""} for r in t["data"]]
    return out


def from_snapshot_const():
    return {k: [dict(zip(["team"] + COLS, r), submitted="") for r in v] for k, v in SNAPSHOT_20261121.items()}


def rank_key(r):
    return (-r["clean"], -r["goal"], r["intr"], r["area"], r["sector"], r["ftime"])


def ours_from_eval(path):
    d = json.loads(Path(path).read_text())
    rows = d["rows"]
    clean = sum(r["waypoint_reached"] >= 1 and r["intrusion_events"] == 0 and r["restricted_area_events"] == 0
                and r["sector_exit_events"] == 0 for r in rows)
    n = len(rows)
    mean = lambda k: sum(float(r[k]) for r in rows) / n
    official = bool(d.get("official_protocol")) and d.get("seed") == 42
    return d["kind"], {"team": f"OURS {d['name']}", "clean": 100.0 * clean / n, "goal": 100.0 * mean("waypoint_reached"),
                       "intr": mean("intrusion_events"), "area": mean("restricted_area_events"),
                       "sector": mean("sector_exit_events"), "ftime": mean("flight_time"),
                       "seed": d["seed"], "n": n, "matched": official}


def ours_from_val(rec, name, kind):
    m = rec["mean"]
    return {"team": f"OURS {name} val@{rec['env_steps']}", "clean": 100.0 * m["clean"], "goal": 100.0 * m["goal"],
            "intr": m["intrusion_events"], "area": m["restricted_area_events"], "sector": m["sector_exit_events"],
            "ftime": m["flight_time"], "seed": "+".join(str(s) for s in rec["seeds"]), "n": m["n_flights"],
            "matched": False}


def val_points(path):
    """Best and last validation records of results/NAME_val.json."""
    d = json.loads(Path(path).read_text())
    name = Path(path).name.replace("_val.json", "")
    h = d["history"]
    out = [d["best"]] + ([h[-1]] if h and h[-1]["env_steps"] != d["best"]["env_steps"] else [])
    return d["kind"], [ours_from_val(r, name, d["kind"]) for r in out]


def wandb_points(run, project="lego_block/bluesky-rl-transformer"):
    """Best and last validation points of a W&B run (val/* keys)."""
    import wandb
    r = wandb.Api().run(f"{project}/{run}")
    kind = "sa" if r.config.get("setting") == "single" else "ma"
    keys = ["val/env_steps", "val/clean", "val/goal", "val/intrusion_events", "val/restricted_area_events",
            "val/sector_exit_events", "val/flight_time", "val/n_flights"]
    hist = [h for h in r.scan_history(keys=keys) if h.get("val/clean") is not None]
    recs = [{"env_steps": int(h["val/env_steps"]), "seeds": [r.config.get("val_seeds", "?")],
             "mean": {k.split("/")[1]: h[k] for k in keys[1:]}} for h in hist]
    if not recs:
        return kind, []
    best = min(recs, key=lambda x: rank_key({"clean": x["mean"]["clean"], "goal": x["mean"]["goal"],
                                             "intr": x["mean"]["intrusion_events"],
                                             "area": x["mean"]["restricted_area_events"],
                                             "sector": x["mean"]["sector_exit_events"], "ftime": x["mean"]["flight_time"]}))
    sel = [best] + ([recs[-1]] if recs[-1]["env_steps"] != best["env_steps"] else [])
    return kind, [ours_from_val(x, run, kind) for x in sel]


def table(kind, board, ours, markdown=False):
    third = sorted(board, key=rank_key)[min(2, len(board) - 1)]["clean"] if board else None
    allr = sorted([dict(b, ours=False) for b in board] + [dict(o, ours=True) for o in ours], key=rank_key)
    hdr = ["#", "Team / run", "Clean %", "Goal %", "Intr", "Area", "Sector", "Time s", "Gap to 3rd", "Of max", "Note"]
    lines, places = [], []
    for i, r in enumerate(allr, 1):
        note, gap, ofmax = "", "", ""
        if r["ours"]:
            seed = r["seed"]
            mx = MAX_CLEAN.get((kind, seed)) if isinstance(seed, int) else None
            gap = f"{third - r['clean']:+.2f}" if third is not None else ""
            ofmax = f"{100.0 * r['clean'] / mx:.1f} %" if mx else "n/a"
            note = f"seed {seed}, n {r['n']}" + ("" if r["matched"] else ", NOT MATCHED (seed 7/11, not 42)")
            rank_board = 1 + sum(rank_key(b) < rank_key(r) for b in board)
            places.append(f"{r['team']}: would place {rank_board} of {len(board) + 1}"
                          + ("" if r["matched"] else ", not matched"))
        dec = 2 if kind == "ma" else 1
        lines.append([str(i), r["team"], f"{r['clean']:.{dec}f}", f"{r['goal']:.{dec}f}", f"{r['intr']:.4f}",
                      f"{r['area']:.4f}", f"{r['sector']:.4f}", f"{r['ftime']:.0f}", gap, ofmax, note])
    title = f"{'Single agent' if kind == 'sa' else 'Multi agent'} (leaderboard: seed 42, 1000 episodes)"
    if markdown:
        out = [f"**{title}**", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        out += ["| " + " | ".join(("**" + c + "**") if l[1].startswith("OURS") and j == 1 else c for j, c in enumerate(l))
                + " |" for l in lines]
    else:
        w = [max(len(h), *(len(l[j]) for l in lines)) for j, h in enumerate(hdr)]
        out = [title, "  ".join(h.ljust(w[j]) for j, h in enumerate(hdr))]
        out += ["  ".join(c.ljust(w[j]) for j, c in enumerate(l)) for l in lines]
    return "\n".join(out + [""] + places + [""])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("evals", nargs="*", help="results/*_competition_{sa,ma}.json files")
    p.add_argument("--val-json", action="append", default=[], help="results/NAME_val.json (best and last point)")
    p.add_argument("--wandb-run", action="append", default=[], help="W&B run name (best and last val point)")
    p.add_argument("--snapshot", default=None, help="Saved leaderboard JSON (no fetch). Default: fetch once.")
    p.add_argument("--offline", action="store_true", help="No fetch: the newest results/leaderboard_*.json, "
                                                          "else the built-in 2026-11-21 snapshot.")
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args(argv)
    if a.snapshot:
        board, src = json.loads(Path(a.snapshot).read_text())["tables"], a.snapshot
    elif a.offline:
        saved = sorted(glob.glob("results/leaderboard_*.json"))
        board, src = ((json.loads(Path(saved[-1]).read_text())["tables"], saved[-1]) if saved
                      else (from_snapshot_const(), "built-in snapshot 2026-11-21"))
    else:
        try:
            board = fetch()
            now = dt.datetime.now().strftime("%Y%m%d-%H%M")
            src = f"results/leaderboard_{now}.json"
            Path(src).write_text(json.dumps({"source": API, "read_at": now, "status": "UNOFFICIAL, self-reported",
                                             "tables": board}, indent=1))
        except Exception as e:                                     # API down: newest saved file or built-in
            print(f"fetch failed ({e}); using a saved snapshot")
            _offline(a)
            return
    _report(a, board, src)


def _offline(a):
    a.offline, a.snapshot = True, None
    saved = sorted(glob.glob("results/leaderboard_*.json"))
    board, src = ((json.loads(Path(saved[-1]).read_text())["tables"], saved[-1]) if saved
                  else (from_snapshot_const(), "built-in snapshot 2026-11-21"))
    _report(a, board, src)


def _report(a, board, src):
    ours = {"sa": [], "ma": []}
    for f in a.evals:
        k, r = ours_from_eval(f)
        ours[k].append(r)
    for f in a.val_json:
        k, rs = val_points(f)
        ours[k] += rs
    for run in a.wandb_run:
        k, rs = wandb_points(run)
        ours[k] += rs
    print(f"Leaderboard source: {src} (unofficial, self-reported; judges also weigh novelty and report quality)\n")
    for kind in ["sa", "ma"]:
        if ours[kind]:
            print(table(kind, board[kind], ours[kind], a.markdown))


if __name__ == "__main__":
    main()
