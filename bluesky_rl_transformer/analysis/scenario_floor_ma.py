"""Unavoidable-failure floor of the multi-agent competition protocol (no simulator).

Replays the CompetitionZooEnv scenario sequence (first reset seed=SEED, then no seed; the environment draws
its random numbers only in ScenarioGenerator.generate) with the same generator settings as the environment
(bluesky_zoo/competition/competition.py: n_agents=10, n_obstacles=5, n_intruders=0, agent_speed=150,
center=(52, 4)) and tags each aircraft (one row per aircraft, ac_id KL001..KL0010 as in the eval rows):
  * start_in_area: the start lies inside a restricted area. The environment counts the area event at reset
    (_update_metrics(dt=0)), so the flight can never be clean. UNAVOIDABLE.
  * goal_in_area_deep: the goal lies inside an area and at least 5 km (the capture radius) from its edge, so
    the aircraft must enter the area to reach the goal (or miss the goal). UNAVOIDABLE.
  * goal_in_area: the goal lies inside an area (deep or not).
  * fallback: start or goal closer than 10 NM to an area. The normal placement keeps 10 NM clearance, so
    such an aircraft comes from the relaxed placement (core/scenario.py _generate_agent, no area check).
  * straight_crosses_area: the straight start->goal line crosses an area (a detour is needed; avoidable).
  * t_first_los_s: first loss of separation (< 5 NM) with another agent if all agents fly straight to their
    goals at constant speed (first-order estimate; agents leave at their goal).
Geometry: flat NM frame around the scenario center (same approximation as analysis/scenario_floor.py).

Check (--eval FILE, a results/*_competition_ma.json of the same seed and episode count): every start_in_area
aircraft must have restricted_area_events > 0 in the eval rows. The number of violations must be 0; else the
replay is not aligned with the evaluation.

    PYTHONPATH=.. ../.venv/bin/python -m analysis.scenario_floor_ma --seed 7 --episodes 1000 \
        --eval results/s7ma_sa_v2m_s0_c2_competition_ma.json --out results/scenario_floor_ma_seed7.json
"""

import argparse
import collections
import json
import math
import os

import numpy as np
from gymnasium.utils import seeding
from shapely.geometry import LineString, Point, Polygon

from core.scenario import ScenarioGenerator

INTRUSION_NM = 5.0
NM = 1852.0
CAPTURE_NM = 5.0 / 1.852          # waypoint capture radius 5 km (competition.py distance_margin)
CLEARANCE_NM = 10.0               # ScenarioGenerator obstacle_clearance_nm (normal placement)


def _xy_nm(center, latlon):
    """(lat, lon) -> flat (north, east) NM frame."""
    return ((latlon[0] - center[0]) * 60.0,
            (latlon[1] - center[1]) * 60.0 * math.cos(math.radians(center[0])))


def _first_los_s(p0, v0, p1, v1, t_max):
    """First time (s) at which |rel| < 5 NM, both on straight lines; None if never in [0, t_max]."""
    r = np.subtract(p1, p0) * NM
    v = np.subtract(v1, v0)
    a, b, c = v @ v, 2 * r @ v, r @ r - (INTRUSION_NM * NM) ** 2
    if c < 0:
        return 0.0
    if a < 1e-9:
        return None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / (2 * a)
    return t if 0 <= t <= t_max else None


def analyse(scn, ep):
    c = scn.center
    areas = [Polygon([_xy_nm(c, v) for v in o.vertices]) for o in scn.obstacles]
    ags = []
    for ag in scn.agents:
        s, g = np.array(_xy_nm(c, ag.start)), np.array(_xy_nm(c, ag.goal))
        d = float(np.hypot(*(g - s)))
        ags.append({"id": ag.ac_id, "s": s, "g": g, "v": (g - s) / d * ag.speed, "t_goal": d * NM / ag.speed})
    rows = []
    for i, a in enumerate(ags):
        ps, pg, line = Point(a["s"]), Point(a["g"]), LineString([a["s"], a["g"]])
        goal_depth = max((ar.exterior.distance(pg) for ar in areas if ar.contains(pg)), default=0.0)
        t_first = [t for j, b in enumerate(ags) if j != i
                   for t in [_first_los_s(a["s"], a["v"], b["s"], b["v"], min(a["t_goal"], b["t_goal"]))]
                   if t is not None]
        start_in = any(ar.contains(ps) for ar in areas)
        goal_in = goal_depth > 0.0
        rows.append({
            "episode_index": ep, "aircraft": a["id"],
            "start_in_area": start_in,
            "goal_in_area": goal_in,
            "goal_in_area_deep": goal_in and goal_depth >= CAPTURE_NM,
            "goal_depth_nm": round(goal_depth, 3),
            "d_start_area_nm": round(min((ar.exterior.distance(ps) for ar in areas), default=math.inf), 3),
            "d_goal_area_nm": round(min((ar.exterior.distance(pg) for ar in areas), default=math.inf), 3),
            "fallback": any(ar.distance(ps) < CLEARANCE_NM or ar.distance(pg) < CLEARANCE_NM for ar in areas),
            "straight_crosses_area": (not start_in and not goal_in) and any(line.intersects(ar) for ar in areas),
            "t_first_los_s": round(min(t_first), 1) if t_first else None,
        })
    return rows


def _cls(x):
    t = (("G" if x["waypoint_reached"] <= 0 else "") + ("I" if x["intrusion_events"] > 0 else "")
         + ("A" if x["restricted_area_events"] > 0 else "") + ("S" if x["sector_exit_events"] > 0 else ""))
    return t or "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--n-obstacles", type=int, default=5)
    ap.add_argument("--eval", default=None, help="results/*_competition_ma.json of the same seed and episodes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rows-out", default=None, help="JSON list of the per-aircraft rows")
    a = ap.parse_args()
    gen = ScenarioGenerator(n_obstacles=a.n_obstacles, n_intruders=0, n_agents=a.n_agents, agent_speed=150.0,
                            center=(52.0, 4.0))
    rng, _ = seeding.np_random(a.seed)
    rows = [r for ep in range(a.episodes) for r in analyse(gen.generate(rng), ep)]
    n = len(rows)

    def cnt(f, rs=rows):
        return int(sum(1 for r in rs if f(r)))

    unavoid = cnt(lambda r: r["start_in_area"] or r["goal_in_area_deep"])
    out = {
        "status": "EXPLORATORY", "seed": a.seed, "episodes": a.episodes, "n_flights": n,
        "start_in_area": cnt(lambda r: r["start_in_area"]),
        "goal_in_area": cnt(lambda r: r["goal_in_area"]),
        "goal_in_area_deep": cnt(lambda r: r["goal_in_area_deep"]),
        "unavoidable_flights": unavoid,
        "max_clean": n - unavoid, "max_clean_pct": round(100.0 * (n - unavoid) / n, 2),
        "fallback_placement": cnt(lambda r: r["fallback"]),
        "straight_crosses_area": cnt(lambda r: r["straight_crosses_area"]),
        "start_area_lt_5nm": cnt(lambda r: r["d_start_area_nm"] < 5),
        "goal_area_lt_5nm": cnt(lambda r: r["d_goal_area_nm"] < 5),
        "first_los_lt_s": {str(t): cnt(lambda r: r["t_first_los_s"] is not None and r["t_first_los_s"] < t)
                           for t in (30, 60, 120, 300)},
        "note": "unavoidable = start inside an area (event at reset) or goal >= 5 km inside an area; "
                "straight-line LoS: all agents fly straight to the goal at constant speed",
    }
    if a.eval:
        ev = json.load(open(a.eval))
        er = {(x["episode_index"], x["aircraft"]): x for x in ev["rows"]}
        miss = [k for k in ((r["episode_index"], r["aircraft"]) for r in rows) if k not in er]
        viol = cnt(lambda r: r["start_in_area"] and er[(r["episode_index"], r["aircraft"])]
                   ["restricted_area_events"] == 0, [r for r in rows if (r["episode_index"], r["aircraft"]) in er])
        fails = [r for r in rows if (r["episode_index"], r["aircraft"]) in er
                 and _cls(er[(r["episode_index"], r["aircraft"])]) != "clean"]

        def tag(r):
            if r["start_in_area"]:
                return "start_in_area"
            if r["goal_in_area_deep"]:
                return "goal_in_area_deep"
            if r["goal_in_area"]:
                return "goal_in_area_shallow"
            if r["straight_crosses_area"]:
                return "straight_crosses_area"
            return "none"
        tab = collections.defaultdict(collections.Counter)
        for r in fails:
            tab[_cls(er[(r["episode_index"], r["aircraft"])])][tag(r)] += 1
        clean = n - len(fails) - len(miss)
        out["eval"] = {
            "file": a.eval, "model": ev.get("model"), "rows_matched": n - len(miss), "rows_missing": len(miss),
            "check_start_in_area_without_area_event": viol,
            "check_ok": viol == 0 and not miss,
            "clean": clean, "clean_pct": round(100.0 * clean / n, 2),
            "avoidable_failures": len(fails) - cnt(lambda r: r["start_in_area"] or r["goal_in_area_deep"], fails),
            "failures_by_class_and_tag": {k: dict(v) for k, v in sorted(tab.items(), key=lambda kv: -sum(kv[1].values()))},
        }
    print(json.dumps(out, indent=1))
    if a.rows_out:
        with open(a.rows_out, "w") as f:
            json.dump(rows, f)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
