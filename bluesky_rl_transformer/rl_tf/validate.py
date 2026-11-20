"""Validation during training (2026-11-20): each weight snapshot flies the competition protocol on held-out seeds.

The jobs run in background processes (scripts.competition_eval._job), so the training does not wait for them.
Protocol per validation seed: reset(seed=S) for episode 0, then reset() with no seed (the official loop), the first
`episodes` episodes. Single agent: one process per seed (as in scripts.competition_eval). Multi agent: the episodes
of a seed are split over the processes.

Rank of two snapshots = the leaderboard rule: clean flights % (desc), goal % (desc), then intrusion events,
restricted-area events, sector exits, flight time (asc). Clean flight = goal reached and no intrusion, no
restricted-area entry, no sector exit (official metrics, start intrusions included).
The best snapshot is copied to models/NAME_best.pt (with its validation record under the key "val").
All records go to results/NAME_val.json. Never use seed 42 here: it is the test seed of the leaderboard.
"""
from __future__ import annotations
import json
import multiprocessing as mp
import os
from pathlib import Path
import numpy as np
import torch

RANK_KEYS = ["intrusion_events", "restricted_area_events", "sector_exit_events", "flight_time"]


def is_clean(r, start_left_out=False):
    intr = r.get("intrusion_events_no_start", r["intrusion_events"]) if start_left_out else r["intrusion_events"]
    return (float(r["waypoint_reached"]) >= 1.0 and float(intr) == 0.0 and float(r["restricted_area_events"]) == 0.0
            and float(r["sector_exit_events"]) == 0.0)


def val_metrics(rows):
    """Leaderboard metrics over the flights (rows of scripts.competition_eval._job)."""
    m = {"n_flights": len(rows), "clean": float(np.mean([is_clean(r) for r in rows])),
         "goal": float(np.mean([float(r["waypoint_reached"]) >= 1.0 for r in rows]))}
    m.update({k: float(np.mean([r[k] for r in rows])) for k in RANK_KEYS})
    if rows and all("intrusion_events_no_start" in r for r in rows):      # single agent, not official
        m["clean_start_left_out"] = float(np.mean([is_clean(r, True) for r in rows]))
    return m


def rank_key(m):
    """Smaller = better (leaderboard order)."""
    return (-m["clean"], -m["goal"]) + tuple(m[k] for k in RANK_KEYS)


def parse_seeds(text):
    seeds = [int(s) for s in str(text).split(",") if s.strip()]
    if 42 in seeds:
        raise ValueError("--val-seeds: 42 is the leaderboard test seed; choose the model on other seeds")
    if not seeds:
        raise ValueError("--val-seeds: give at least one seed")
    return seeds


class Validator:
    def __init__(self, setting, seeds, episodes, workers, val_json, best_path, wb=None):
        self.kind = "sa" if setting == "single" else "ma"
        self.seeds, self.episodes, self.workers = list(seeds), int(episodes), max(1, int(workers))
        self.val_json, self.best_path, self.wb = Path(val_json), Path(best_path), wb
        self.history = json.loads(self.val_json.read_text())["history"] if self.val_json.exists() else []
        self.best = min(self.history, key=lambda h: rank_key(h["mean"])) if self.history else None
        # one process per job: each job starts its own BlueSky (as scripts.competition_eval)
        self.pool = mp.get_context("spawn").Pool(self.workers, maxtasksperchild=1)
        self.pending = []                               # (step, snapshot path, [(seed, AsyncResult)])
        if wb is not None:
            wb.define_metric("val/env_steps")
            wb.define_metric("val/*", step_metric="val/env_steps")

    def done_steps(self):
        return {h["env_steps"] for h in self.history} | {p[0] for p in self.pending}

    def submit(self, step, path):
        from scripts.competition_eval import _job
        jobs = []
        for s in self.seeds:
            if self.kind == "sa":                      # single agent: one process per seed (official loop)
                edges = [0, self.episodes]
            else:
                edges = np.linspace(0, self.episodes, min(self.workers, self.episodes) + 1).astype(int).tolist()
            for a, b in zip(edges[:-1], edges[1:]):
                jobs.append((s, self.pool.apply_async(_job, ((str(path), self.kind, int(a), int(b), s, "cpu"),))))
        self.pending.append((int(step), str(path), jobs))
        print(f"validation queued: env_steps={step}, {len(jobs)} jobs, seeds {self.seeds}, "
              f"{self.episodes} episodes each", flush=True)

    def poll(self, block=False):
        """Handle the finished validations (all of them if block). Returns the new records."""
        new, keep = [], []
        for step, path, jobs in self.pending:
            if block or all(r.ready() for _, r in jobs):
                by_seed = {}
                for s, r in jobs:
                    by_seed.setdefault(s, []).extend(r.get())
                new.append(self._record(step, path, by_seed))
            else:
                keep.append((step, path, jobs))
        self.pending = keep
        return new

    def _record(self, step, path, by_seed):
        rows = [r for s in self.seeds for r in by_seed[s]]
        rec = {"env_steps": step, "snapshot": path, "seeds": self.seeds, "episodes_per_seed": self.episodes,
               "mean": val_metrics(rows), "per_seed": {str(s): val_metrics(by_seed[s]) for s in self.seeds}}
        better = self.best is None or rank_key(rec["mean"]) < rank_key(self.best["mean"])
        rec["new_best"] = bool(better)
        self.history.append(rec)
        self.history.sort(key=lambda h: h["env_steps"])
        if better:
            self.best = rec
            ck = torch.load(path, map_location="cpu", weights_only=False)
            ck["val"] = rec
            tmp = self.best_path.with_suffix(".tmp")
            torch.save(ck, tmp)
            os.replace(tmp, self.best_path)
        self._write()
        m = rec["mean"]
        print(f"validation env_steps={step}: clean {100 * m['clean']:.1f} % goal {100 * m['goal']:.1f} % "
              f"intrusions {m['intrusion_events']:.3f} area {m['restricted_area_events']:.3f} "
              f"sector {m['sector_exit_events']:.3f} flight {m['flight_time']:.0f} s"
              + (f" | clean (start left out) {100 * m['clean_start_left_out']:.1f} %" if "clean_start_left_out" in m else "")
              + (f" | NEW BEST -> {self.best_path}" if better else
                 f" | best stays env_steps={self.best['env_steps']} ({100 * self.best['mean']['clean']:.1f} %)"),
              flush=True)
        if self.wb is not None:
            self.wb.log({"val/env_steps": step, **{f"val/{k}": v for k, v in m.items()},
                         "val/best_clean": self.best["mean"]["clean"]})
        return rec

    def _write(self):
        tmp = self.val_json.with_suffix(".tmp")
        self.val_json.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"status": "EXPLORATORY", "kind": self.kind, "seeds": self.seeds,
                                   "episodes_per_seed": self.episodes, "rank_rule": "clean desc, goal desc, then "
                                   + ", ".join(RANK_KEYS) + " asc", "best": self.best, "history": self.history},
                                  indent=1))
        os.replace(tmp, self.val_json)

    def summary(self):
        return {"seeds": self.seeds, "episodes_per_seed": self.episodes, "best": self.best,
                "n_validations": len(self.history), "file": str(self.val_json), "best_model": str(self.best_path)}

    def close(self):
        self.pool.close()
        self.pool.join()
