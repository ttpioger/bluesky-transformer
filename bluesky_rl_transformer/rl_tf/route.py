"""Route around the restricted areas (test time, 2026-11-22).

Diagnosis (results of 2026-11-22, seeds 7 and 11): about 95 % of the area entries with the heading mask came after a
mask fallback. The aircraft flies toward the goal until an area is near, then no heading within +-goal_offset_deg of
the goal bearing is free, and the aircraft enters the area. The leaderboard leaders have almost no area entries.

Fix: the reference bearing of action_map "goal" is not the goal bearing, but the bearing to the next point of the
shortest area-free path to the goal (visibility graph). The nodes are the corners of each area grown by margin_km;
an edge is free if it does not cross any area grown by CHECK_FRAC * margin_km. So the aircraft turns early, toward
the tangent of the area, and the policy still turns away from other aircraft around that reference.
The direct line to the goal free: the reference is the goal bearing (the policy flies as in training).

Frame: own frame of rl_tf/lidar.py (own aircraft at the origin, x = nose, y = starboard, km).
"""
from __future__ import annotations
import numpy as np
from .lidar import points_in_polygon, inflate_polygon

CHECK_FRAC = 0.9          # edges are checked against the areas grown by this share of the margin
MARGINS = (1.0, 0.5, 0.25)
LOOKAHEAD_KM = 10.0       # the reference is the route point this far ahead along the route (pure pursuit)  # shares of margin_km tried in turn when the own aircraft or the goal is too near an area


def _cross(o, a, b):
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def segments_free(a, b, p, q):
    """For segments a[i]-b[i] ([N, 2] each): True if no polygon edge p[j]-q[j] ([M, 2]) crosses it. Returns [N] bool."""
    if len(p) == 0:
        return np.ones(len(a), dtype=bool)
    A, B = a[:, None, :], b[:, None, :]
    P, Q = p[None, :, :], q[None, :, :]
    d1, d2 = _cross(A, B, P), _cross(A, B, Q)
    d3, d4 = _cross(P, Q, A), _cross(P, Q, B)
    hit = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
    return ~hit.any(axis=1)


def _direct_free(polys_check, goal):
    """True if the segment origin -> goal crosses none of the polygons."""
    p = np.concatenate(polys_check)
    q = np.concatenate([np.roll(c, -1, axis=0) for c in polys_check])
    return bool(segments_free(np.zeros((1, 2)), np.asarray(goal, dtype=np.float64).reshape(1, 2), p, q)[0])


def _graph_route(polys_node, polys_check, goal, sector):
    """Shortest path origin -> goal. Returns the reference point [2] (goal if the line is free) or None (no path)."""
    origin = np.zeros(2)
    verts = [v for poly in polys_node for v in poly]
    nodes = [origin, np.asarray(goal, dtype=np.float64)]
    for v in verts:
        if any(points_in_polygon(v[None, :], c)[0] for c in polys_check):
            continue                                   # corner inside another area
        if sector is not None and not points_in_polygon(v[None, :], sector)[0]:
            continue                                   # corner outside the sector
        nodes.append(v)
    X = np.asarray(nodes)
    n = len(X)
    p = np.concatenate([c for c in polys_check]) if polys_check else np.zeros((0, 2))
    q = np.concatenate([np.roll(c, -1, axis=0) for c in polys_check]) if polys_check else np.zeros((0, 2))
    i, j = np.triu_indices(n, k=1)
    free = segments_free(X[i], X[j], p, q)
    if polys_check:                                    # a chord through a concave area: its midpoint is inside
        mid = 0.5 * (X[i] + X[j])
        for c in polys_check:
            free &= ~points_in_polygon(mid, c)
    W = np.full((n, n), np.inf)
    d = np.hypot(*(X[i] - X[j]).T)
    W[i[free], j[free]] = d[free]
    W[j[free], i[free]] = d[free]
    if np.isfinite(W[0, 1]):
        return X[1]
    dist = np.full(n, np.inf)                          # Dijkstra from the goal (node 1): then the next node is direct
    dist[1] = 0.0
    done = np.zeros(n, dtype=bool)
    for _ in range(n):
        k = int(np.argmin(np.where(done, np.inf, dist)))
        if not np.isfinite(dist[k]):
            break
        done[k] = True
        dist = np.minimum(dist, dist[k] + W[k])
    if not np.isfinite((W[0] + dist).min()):
        return None
    path, k = [X[0]], 0                                # path origin -> goal: each node goes to its best next node
    for _ in range(n):
        k = int(np.argmin(W[k] + dist))
        path.append(X[k])
        if k == 1:
            break
    return _lookahead(np.asarray(path), LOOKAHEAD_KM)


def _lookahead(path, look_km):
    """Point at path length look_km along the polyline path [P, 2] (pure pursuit), or its end. A point that moves
    ahead along the route: the aircraft cannot hit a corner exactly (turn radius), and a corner behind it would make it
    turn back and circle (seen in the smoke run of 2026-11-22: flights that circled until the time limit)."""
    seg = np.diff(path, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(ln)])
    if look_km >= cum[-1]:
        return path[-1]
    i = int(np.searchsorted(cum, look_km, side="right") - 1)
    f = (look_km - cum[i]) / max(ln[i], 1e-9)
    return path[i] + f * seg[i]


def route_point(zones, goal, margin_km, sector=None):
    """Next point (own frame, km) of the shortest area-free path to the goal, or the goal itself (direct line free,
    no path found, or the own aircraft inside an area). zones: list of polygons [V, 2]; goal [2]; sector [S, 2] or None."""
    goal = np.asarray(goal, dtype=np.float64).reshape(2)
    real = [np.asarray(z, dtype=np.float64).reshape(-1, 2) for z in zones]
    real = [v for v in real if len(v) >= 3]
    if not real:
        return goal
    origin = np.zeros((1, 2))
    if any(points_in_polygon(origin, v)[0] for v in real):
        return goal                                    # inside an area: fly out as the policy wants
    sec = None
    if sector is not None:
        s = np.asarray(sector, dtype=np.float64).reshape(-1, 2)
        if len(s) >= 3 and points_in_polygon(origin, s)[0]:
            sec = s
    for f in MARGINS:
        m = float(margin_km) * f
        check = [inflate_polygon(v, CHECK_FRAC * m) for v in real]
        if any(points_in_polygon(origin, c)[0] or points_in_polygon(goal[None, :], c)[0] for c in check):
            continue                                   # too near an area for this margin: try a smaller one
        if _direct_free(check, goal):                  # quick test: the straight line to the goal is free
            return goal
        node = [inflate_polygon(v, m) for v in real]
        nxt = _graph_route(node, check, goal, sec)
        if nxt is not None:
            return nxt
    return goal
