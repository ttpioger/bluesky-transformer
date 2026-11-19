"""Unavoidable-failure floor of the competition protocol (no simulator).

Replays the CompetitionEnv scenario sequence (first reset seed=SEED, then no seed) with the same
ScenarioGenerator and counts, per scenario:
  * spawn intrusion: an intruder spawns closer than 5 NM to the agent start (counted at reset);
  * start or goal inside a restricted area; distance of start / goal to the nearest area;
  * early straight-line conflicts: first loss of separation (both fly straight, first leg) before t s.

Check: the spawn-intrusion count must equal n_flights_start_in_intrusion of the fly-straight run
(94 on seed 42, 1000 flights).

    PYTHONPATH=.. ../.venv/bin/python -m analysis.scenario_floor --seed 42 --episodes 1000
"""

import argparse
import json
import math
import os

import numpy as np
from gymnasium.utils import seeding
from shapely.geometry import Point, Polygon

from core.scenario import ScenarioGenerator
from core.tools import kwikqdrdist

INTRUSION_NM = 5.0
NM = 1852.0


def _xy_nm(center, latlon):
    """(lat, lon) -> flat (north, east) NM frame of the generator."""
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


def analyse(scn):
    c = scn.center
    ag = scn.agents[0]
    s, g = _xy_nm(c, ag.start), _xy_nm(c, ag.goal)
    areas = [Polygon([_xy_nm(c, v) for v in o.vertices]) for o in scn.obstacles]
    sector = Polygon([_xy_nm(c, v) for v in scn.sector])
    dist_sg = math.hypot(g[0] - s[0], g[1] - s[1])
    v_ag = np.array([(g[0] - s[0]), (g[1] - s[1])]) / dist_sg * ag.speed
    t_goal = dist_sg * NM / ag.speed
    spawn, t_first = 0, []
    for rt in scn.intruder_routes:
        _, d = kwikqdrdist(ag.start[0], ag.start[1], rt.waypoints[0][0], rt.waypoints[0][1])
        if d < INTRUSION_NM:
            spawn += 1
            continue
        p0, p1 = _xy_nm(c, rt.waypoints[0]), _xy_nm(c, rt.waypoints[1])
        leg = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        v_in = np.array([(p1[0] - p0[0]), (p1[1] - p0[1])]) / leg * rt.speed
        t = _first_los_s(s, v_ag, p0, v_in, min(t_goal, leg * NM / rt.speed))
        if t is not None:
            t_first.append(t)
    d_start = min((a.exterior.distance(Point(s)) for a in areas), default=math.inf)
    d_goal = min((a.exterior.distance(Point(g)) for a in areas), default=math.inf)
    return {
        "spawn_intr": spawn,
        "start_in_area": any(a.contains(Point(s)) for a in areas),
        "goal_in_area": any(a.contains(Point(g)) for a in areas),
        "d_start_area_nm": d_start, "d_goal_area_nm": d_goal,
        "d_start_edge_nm": sector.exterior.distance(Point(s)),
        "d_goal_edge_nm": sector.exterior.distance(Point(g)),
        "t_first_los_s": min(t_first) if t_first else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--n-intruders", type=int, default=10)
    ap.add_argument("--n-obstacles", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rows-out", default=None, help="JSON list of the per-episode rows")
    a = ap.parse_args()
    gen = ScenarioGenerator(n_obstacles=a.n_obstacles, n_intruders=a.n_intruders, agent_speed=150.0,
                            center=(52.0, 4.0))
    rng, _ = seeding.np_random(a.seed)
    rows = [analyse(gen.generate(rng)) for _ in range(a.episodes)]

    def cnt(f):
        return int(sum(1 for r in rows if f(r)))

    tf = [r["t_first_los_s"] for r in rows if r["spawn_intr"] == 0]
    out = {
        "seed": a.seed, "episodes": a.episodes,
        "flights_spawn_intrusion": cnt(lambda r: r["spawn_intr"] > 0),
        "spawn_intrusion_events": int(sum(r["spawn_intr"] for r in rows)),
        "start_in_area": cnt(lambda r: r["start_in_area"]),
        "goal_in_area": cnt(lambda r: r["goal_in_area"]),
        "start_area_lt_5nm": cnt(lambda r: r["d_start_area_nm"] < 5),
        "goal_area_lt_5nm": cnt(lambda r: r["d_goal_area_nm"] < 5),
        "start_area_lt_10nm": cnt(lambda r: r["d_start_area_nm"] < 10),
        "goal_area_lt_10nm": cnt(lambda r: r["d_goal_area_nm"] < 10),
        "min_d_start_area_nm": round(min(r["d_start_area_nm"] for r in rows), 2),
        "min_d_goal_area_nm": round(min(r["d_goal_area_nm"] for r in rows), 2),
        "min_d_start_edge_nm": round(min(r["d_start_edge_nm"] for r in rows), 2),
        "min_d_goal_edge_nm": round(min(r["d_goal_edge_nm"] for r in rows), 2),
        "no_spawn_first_los_lt_s": {str(t): int(sum(1 for x in tf if x is not None and x < t))
                                    for t in (30, 60, 120, 300)},
        "note": "straight-line LoS uses the first intruder leg and the agent straight to the goal",
    }
    print(json.dumps(out, indent=1))
    if a.rows_out:
        with open(a.rows_out, "w") as f:
            json.dump([{"episode_index": i, **r} for i, r in enumerate(rows)], f, default=str)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
