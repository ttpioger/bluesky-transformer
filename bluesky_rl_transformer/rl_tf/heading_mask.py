"""Heading mask for restricted areas and the sector edge (2026-11-21).

The policy action a[0] in [-1, 1] commands the heading goal bearing + a[0] * goal_offset_deg (action_map "goal").
The mask looks along that heading from the own position. If a restricted area (or, with sector=True, the sector
edge) is nearer than dist_km on that line, the heading is blocked. Then the mask moves the command to the nearest
allowed offset of the grid -G, -G + step, ..., +G (G = goal_offset_deg). Equal distance: the side nearer to the
current heading. All grid offsets blocked: the offset with the largest free distance (fallback).
Only static geometry (areas, sector); other aircraft never block a heading. Speed is not changed.
No mask when the own aircraft is inside a (mask) area. Outside the sector, the sector edge does not block.
Near the goal: the block distance is min(dist_km, goal distance), so that an edge behind the goal (an area or the
sector edge less than dist_km beyond the goal) does not block the direct path to the goal.

Frame: own aircraft at the origin, x = nose, y = starboard, km (rl_tf/lidar.py). Angles in degrees, positive =
starboard. Offsets are relative to the goal bearing; goal_rel_deg = goal bearing minus own heading.
"""
from __future__ import annotations
import numpy as np
from .lidar import ray_segments, polygon_edges, points_in_polygon, inflate_polygon

MASK_KEYS = ["heading_mask", "mask_dist_km", "mask_step_deg", "mask_margin_km", "mask_sector"]
MASK_DEFAULTS = dict(heading_mask=0, mask_dist_km=9.6, mask_step_deg=10.0, mask_margin_km=0.0, mask_sector=1)


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def mask_geometry(zones, sector, margin_km=0.0, use_sector=True):
    """Edges for the mask from the polygons of one observation (own frame, km).
    Returns None if the own aircraft is inside a mask area (no mask), else (p, q) of all blocking edges."""
    polys = [np.asarray(z, dtype=np.float64).reshape(-1, 2) for z in zones]
    polys = [inflate_polygon(v, margin_km) if margin_km > 0.0 else v for v in polys if len(v) >= 3]
    origin = np.zeros((1, 2))
    if any(points_in_polygon(origin, v)[0] for v in polys):
        return None
    if use_sector and sector is not None:
        s = np.asarray(sector, dtype=np.float64).reshape(-1, 2)
        if len(s) >= 3 and points_in_polygon(origin, s)[0]:   # outside the sector: the edge does not block
            polys = polys + [s]
    return polygon_edges(polys)


def free_distance(edges, rel_deg):
    """Distance (km) along each heading rel_deg (own frame, degrees) to the first blocking edge; inf = none."""
    rel = np.atleast_1d(np.asarray(rel_deg, dtype=np.float64))
    p, q = edges
    if len(p) == 0:
        return np.full(len(rel), np.inf)
    a = np.deg2rad(rel)
    return ray_segments(np.stack([np.cos(a), np.sin(a)], axis=1), p, q).min(axis=1)


def choose_offset(edges, goal_rel_deg, req_off, cur_off, goal_offset_deg, dist_km, step_deg, goal_km=np.inf):
    """Offset (degrees from the goal bearing) to fly. Returns (offset, projected, fallback).
    edges: mask_geometry output (None = no mask). req_off: requested offset. cur_off: current heading as an
    offset from the goal bearing (for the tie rule). goal_km: distance to the goal (see the module text)."""
    if edges is None:
        return float(req_off), False, False
    dist_km = min(float(dist_km), float(goal_km))
    if free_distance(edges, goal_rel_deg + req_off)[0] >= dist_km:
        return float(req_off), False, False
    G = float(goal_offset_deg)
    n = int(round(2.0 * G / float(step_deg)))
    cand = np.linspace(-G, G, n + 1)
    d = free_distance(edges, goal_rel_deg + cand)
    ok = d >= dist_km
    if not ok.any():
        return float(cand[int(d.argmax())]), True, True
    gap = np.abs(cand - req_off)
    best = gap[ok].min()
    tied = np.flatnonzero(ok & (np.abs(gap - best) <= 1e-9))
    k = tied[int(np.abs(cand[tied] - cur_off).argmin())]
    return float(cand[k]), True, False


def mask_settings(v2):
    """The mask part of a v2 dict (for JSON outputs)."""
    return {k: v2.get(k, MASK_DEFAULTS[k]) for k in MASK_KEYS}


def wrap_deg(x):
    return _wrap(x)


def add_mask_args(p, from_checkpoint=False):
    """The mask flags. from_checkpoint (evaluation): default None = the value saved in the model checkpoint."""
    d = (lambda k: None) if from_checkpoint else (lambda k: MASK_DEFAULTS[k])
    src = " Default: the value in the checkpoint." if from_checkpoint else ""
    p.add_argument("--heading-mask", type=int, choices=[0, 1], default=d("heading_mask"),
                   help="1: heading mask for restricted areas and the sector edge (action_map goal; rl_tf/heading_mask.py)."
                        + src)
    p.add_argument("--mask-dist-km", type=float, default=d("mask_dist_km"),
                   help=f"Blocked heading: an area or the sector edge nearer than this on the heading line "
                        f"(default {MASK_DEFAULTS['mask_dist_km']} km, about 2 A320 turn radii)." + src)
    p.add_argument("--mask-step-deg", type=float, default=d("mask_step_deg"),
                   help=f"Grid of candidate offsets (default {MASK_DEFAULTS['mask_step_deg']:g} deg)." + src)
    p.add_argument("--mask-margin-km", type=float, default=d("mask_margin_km"),
                   help="The mask sees each area grown by this margin (default 0 = the real polygons)." + src)
    p.add_argument("--mask-sector", type=int, choices=[0, 1], default=d("mask_sector"),
                   help="1: the sector edge also blocks a heading (default 1)." + src)


def mask_overrides(args):
    """Mask flags given on the command line (evaluation): {key: value}, without the None values."""
    return {k: getattr(args, k) for k in MASK_KEYS if getattr(args, k, None) is not None}
