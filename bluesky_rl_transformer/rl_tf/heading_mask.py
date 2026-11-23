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

MASK_KEYS = ["heading_mask", "mask_dist_km", "mask_step_deg", "mask_margin_km", "mask_sector", "mask_mode",
             "mask_intruders", "mask_intr_nm", "mask_intr_horizon_s", "mask_route", "mask_route_km",
             "mask_proj_penalty", "mask_fallback_penalty", "mask_fb2", "mask_fb2_dv_ms", "mask_route_cooldown",
             "mask_speed", "mask_route_escalate"]
MASK_DEFAULTS = dict(heading_mask=0, mask_dist_km=9.6, mask_step_deg=10.0, mask_margin_km=0.0, mask_sector=1,
                     mask_mode="ray", mask_intruders=0, mask_intr_nm=5.5, mask_intr_horizon_s=120.0,
                     mask_route=0, mask_route_km=5.0, mask_proj_penalty=0.0, mask_fallback_penalty=0.0,
                     mask_fb2=0, mask_fb2_dv_ms=10.0, mask_route_cooldown=0, mask_speed=0,
                     mask_route_escalate=0)
# mask_route_escalate (2026-11-23, test time): when the progress guard fires, re-plan with a wider margin
# (mask_route_km x 2, then x 3) before the route goes off. Video SA seed 73 (cooldown on): the route led into a narrow
# gap between areas that the mask blocks (block distance 9.6 km > route margin 3 km); the aircraft circled there.
# mask_route_cooldown (2026-11-23, test time): decisions the route stays off after the progress guard fires; then it
# comes back with a fresh progress window. 0 = off for the rest of the flight (before). Diagnosis (lrd_avg5 + shield,
# seeds 7 / 11): the route was off in 100 % of the flights that missed the goal (video SA seed 73: a traffic delay
# early on switched the route off; later the aircraft circled behind a thin area until the time limit).
# mask_speed (2026-11-23, test time, arc mode with intruders): when traffic (not the static geometry) blocks the
# requested heading, first test the own speed -dv / +dv (dv = mask_fb2_dv_ms); if one makes the request free, keep the
# heading and send that speed action for this decision (video SA seed 31: late conflict near the goal).
# mask_fallback_penalty (2026-11-22, training): <= 0, reward per decision where all candidate headings are blocked
# (mask fallback = the aircraft is trapped). 90 % of the area entries and later intrusions came after a fallback.
# mask_proj_penalty (2026-11-22, shield distillation, training): <= 0, reward per decision that the mask corrects,
# times |executed - requested offset| / goal_offset_deg (a 90 degree correction costs the full value). With
# --mask-learn policy the policy learns to request safe headings itself, so that the mask is needed less.
# mask_fb2 (2026-11-23, test time, arc mode): better trap fallback. Diagnosis (SA, lrd_avg5 + arcI + route3, seeds 7
# and 11): 91-95 % of the flights with a later intrusion had a mask fallback. In a trap (no candidate within +-G free),
# fb2 searches the full circle, tests the own speed - dv, + 0 and + dv (speed action -1, 0, +1 for this decision) and
# also counts aircraft already nearer than mask_intr_nm, from FB2_NEAR_T0_S on (the heading that opens the distance).
# mask_route (2026-11-22, test time, rl_tf/route.py): the reference bearing of action_map "goal" is the bearing to
# the next point of the shortest path around the areas grown by mask_route_km (not the goal bearing). Works with or
# without heading_mask; with the mask, the mask uses the same reference.
# mask_intruders (2026-11-21, arc mode only): a candidate is also blocked if its predicted path comes nearer than
# mask_intr_nm to the constant-velocity track of another aircraft within mask_intr_horizon_s. Aircraft already
# nearer than mask_intr_nm at the decision do not block (that conflict exists already). If no candidate is free of
# both: the statically free candidate with the largest separation, else the longest static free path (fallback).
# mask_mode (2026-11-21): "ray" = the straight line along the target heading (first version).
# "arc" = the predicted flight path: start on the current heading, turn at the BlueSky turn rate
# (g * tan(bank) / TAS) toward the target heading, then straight. A candidate is blocked if this path enters a mask
# area or leaves the sector within the block distance. Fallback: the longest free path. Sampling: ARC_DT_S seconds.
ARC_DT_S = 2.0
ARC_FALLBACK_FACTOR = 3.0                              # fallback ranks the free path up to this times the distance


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
    p.add_argument("--mask-intruders", type=int, choices=[0, 1], default=d("mask_intruders"),
                   help="Arc mode: 1 = other aircraft (constant velocity) also block a heading." + src)
    p.add_argument("--mask-intr-nm", type=float, default=d("mask_intr_nm"),
                   help=f"Least predicted separation (default {MASK_DEFAULTS['mask_intr_nm']} NM)." + src)
    p.add_argument("--mask-intr-horizon-s", type=float, default=d("mask_intr_horizon_s"),
                   help=f"Prediction horizon (default {MASK_DEFAULTS['mask_intr_horizon_s']:g} s)." + src)
    p.add_argument("--mask-route", type=int, choices=[0, 1], default=d("mask_route"),
                   help="1: fly around the restricted areas: reference bearing = next point of the shortest "
                        "area-free path (rl_tf/route.py; action_map goal)." + src)
    p.add_argument("--mask-route-km", type=float, default=d("mask_route_km"),
                   help=f"Margin of the route around each area (default {MASK_DEFAULTS['mask_route_km']:g} km)." + src)
    p.add_argument("--mask-proj-penalty", type=float, default=d("mask_proj_penalty"),
                   help="<= 0 (training, shield distillation): reward per mask correction, times the corrected angle / "
                        "goal_offset_deg. 0 = off." + src)
    p.add_argument("--mask-fallback-penalty", type=float, default=d("mask_fallback_penalty"),
                   help="<= 0 (training): reward per decision where the mask finds no free heading (trapped). 0 = off."
                        + src)
    p.add_argument("--mask-fb2", type=int, choices=[0, 1], default=d("mask_fb2"),
                   help="Arc mode: 1 = trap fallback over the full circle with a speed change (test time)." + src)
    p.add_argument("--mask-fb2-dv-ms", type=float, default=d("mask_fb2_dv_ms"),
                   help=f"Speed change tested by fb2 (default {MASK_DEFAULTS['mask_fb2_dv_ms']:g} m/s)." + src)
    p.add_argument("--mask-route-cooldown", type=int, default=d("mask_route_cooldown"),
                   help="Decisions the route stays off after the progress guard (0 = rest of the flight)." + src)
    p.add_argument("--mask-route-escalate", type=int, choices=[0, 1], default=d("mask_route_escalate"),
                   help="1 = on a progress-guard event, re-plan the route with margin x2, then x3, before it goes off." + src)
    p.add_argument("--mask-speed", type=int, choices=[0, 1], default=d("mask_speed"),
                   help="Arc mode, intruders: 1 = try a speed change before turning away from traffic." + src)
    p.add_argument("--mask-mode", choices=["ray", "arc"], default=d("mask_mode"),
                   help="ray: straight line along the target heading (default). arc: predicted flight path with "
                        "the BlueSky turn rate from the current heading." + src)


def mask_overrides(args):
    """Mask flags given on the command line (evaluation): {key: value}, without the None values."""
    return {k: getattr(args, k) for k in MASK_KEYS if getattr(args, k, None) is not None}


def mask_polys(zones, sector, margin_km=0.0, use_sector=True):
    """Arc mode: (areas, sector or None) from the polygons of one observation, or None if the own aircraft is inside
    a mask area (no mask). The sector is None when it does not block (off, or the own aircraft is outside it)."""
    polys = [np.asarray(z, dtype=np.float64).reshape(-1, 2) for z in zones]
    polys = [inflate_polygon(v, margin_km) if margin_km > 0.0 else v for v in polys if len(v) >= 3]
    origin = np.zeros((1, 2))
    if any(points_in_polygon(origin, v)[0] for v in polys):
        return None
    sec = None
    if use_sector and sector is not None:
        s = np.asarray(sector, dtype=np.float64).reshape(-1, 2)
        if len(s) >= 3 and points_in_polygon(origin, s)[0]:
            sec = s
    return polys, sec


def arc_paths(target_rel_deg, speed_ms, turn_rate_dps, length_km, dt=ARC_DT_S):
    """Predicted positions (own frame, km) for each target heading [C]: turn from heading 0 (the nose) toward the
    target at turn_rate_dps (shortest way), then straight. Returns points [C, P, 2] at the path lengths [P] (km)."""
    tgt = _wrap(np.atleast_1d(np.asarray(target_rel_deg, dtype=np.float64)))
    step_km = speed_ms * dt / 1000.0
    n = max(1, int(np.ceil(length_km / max(step_km, 1e-6))))
    t = dt * np.arange(n)                              # heading during each sample interval (start of the interval)
    h = np.sign(tgt)[:, None] * np.minimum(turn_rate_dps * t[None, :], np.abs(tgt)[:, None])
    a = np.deg2rad(h)
    pts = np.cumsum(np.stack([np.cos(a), np.sin(a)], axis=-1) * step_km, axis=1)
    return pts, step_km * np.arange(1, n + 1)


def free_path(geo, target_rel_deg, speed_ms, turn_rate_dps, length_km):
    """Path length (km) to the first point inside a mask area or outside the sector, per target; inf = none."""
    pts, s = arc_paths(target_rel_deg, speed_ms, turn_rate_dps, length_km)
    C, P = pts.shape[:2]
    flat = pts.reshape(-1, 2)
    bad = np.zeros(C * P, dtype=bool)
    areas, sec = geo
    for v in areas:
        bad |= points_in_polygon(flat, v)
    if sec is not None:
        bad |= ~points_in_polygon(flat, sec)
    bad = bad.reshape(C, P)
    first = np.where(bad.any(axis=1), bad.argmax(axis=1), -1)
    return np.where(first >= 0, s[np.maximum(first, 0)], np.inf)


def min_separation(target_rel_deg, speed_ms, turn_rate_dps, intr):
    """Least distance (km) over the horizon between the predicted own path (per target) and the constant-velocity
    tracks of the other aircraft. intr: dict(pos [M, 2] km, vel [M, 2] m/s, own frame, absolute velocity;
    sep_km; horizon_s). Aircraft already nearer than sep_km are left out. inf = no aircraft."""
    pos = np.asarray(intr["pos"], dtype=np.float64).reshape(-1, 2)
    vel = np.asarray(intr["vel"], dtype=np.float64).reshape(-1, 2)
    keep = np.hypot(pos[:, 0], pos[:, 1]) >= intr["sep_km"]
    pos, vel = pos[keep], vel[keep]
    n_t = len(np.atleast_1d(target_rel_deg))
    if len(pos) == 0:
        return np.full(n_t, np.inf)
    pts, _ = arc_paths(target_rel_deg, speed_ms, turn_rate_dps, speed_ms * intr["horizon_s"] / 1000.0)
    t = ARC_DT_S * np.arange(1, pts.shape[1] + 1)                     # time of each path point (s)
    other = pos[None, :, :] + vel[None, :, :] * t[:, None, None] / 1000.0   # [P, M, 2]
    d = np.linalg.norm(pts[:, :, None, :] - other[None], axis=-1)          # [C, P, M]
    return d.min(axis=(1, 2))


def choose_offset_arc(geo, goal_rel_deg, req_off, cur_off, goal_offset_deg, dist_km, step_deg, speed_ms,
                      turn_rate_dps, goal_km=np.inf, intr=None):
    """Arc mode of choose_offset (same rules; the free distance is along the predicted path). intr: see
    min_separation (None = static geometry only)."""
    if geo is None and intr is None:
        return float(req_off), False, False
    if geo is None:                                    # inside an area: only the intruder part
        geo = ([], None)
    dist_km = min(float(dist_km), float(goal_km))
    req_ok = free_path(geo, goal_rel_deg + req_off, speed_ms, turn_rate_dps, dist_km)[0] >= dist_km
    if req_ok and intr is not None:
        req_ok = min_separation(goal_rel_deg + req_off, speed_ms, turn_rate_dps, intr)[0] >= intr["sep_km"]
    if req_ok:
        return float(req_off), False, False
    G = float(goal_offset_deg)
    n = int(round(2.0 * G / float(step_deg)))
    cand = np.linspace(-G, G, n + 1)
    d = free_path(geo, goal_rel_deg + cand, speed_ms, turn_rate_dps, ARC_FALLBACK_FACTOR * dist_km)
    ok_s = d >= dist_km
    ok = ok_s
    if intr is not None:
        sep = min_separation(goal_rel_deg + cand, speed_ms, turn_rate_dps, intr)
        ok = ok_s & (sep >= intr["sep_km"])
    if not ok.any():
        if intr is not None and ok_s.any():
            k = np.flatnonzero(ok_s)[int(sep[ok_s].argmax())]
            return float(cand[k]), True, True
        return float(cand[int(d.argmax())]), True, True
    gap = np.abs(cand - req_off)
    best = gap[ok].min()
    tied = np.flatnonzero(ok & (np.abs(gap - best) <= 1e-9))
    k = tied[int(np.abs(cand[tied] - cur_off).argmin())]
    return float(cand[k]), True, False


FB2_NEAR_T0_S = 30.0                                   # near aircraft: separation counted from this time on


def separation_all(target_rel_deg, speed_ms, turn_rate_dps, intr, t0_near=FB2_NEAR_T0_S):
    """As min_separation, but aircraft already nearer than sep_km are kept: for them only the path points from
    t0_near seconds on count (so the candidate that opens the distance scores higher). Returns [C] km."""
    pos = np.asarray(intr["pos"], dtype=np.float64).reshape(-1, 2)
    vel = np.asarray(intr["vel"], dtype=np.float64).reshape(-1, 2)
    n_t = len(np.atleast_1d(target_rel_deg))
    if len(pos) == 0:
        return np.full(n_t, np.inf)
    pts, _ = arc_paths(target_rel_deg, speed_ms, turn_rate_dps, speed_ms * intr["horizon_s"] / 1000.0)
    t = ARC_DT_S * np.arange(1, pts.shape[1] + 1)
    other = pos[None, :, :] + vel[None, :, :] * t[:, None, None] / 1000.0
    d = np.linalg.norm(pts[:, :, None, :] - other[None], axis=-1)          # [C, P, M]
    near = np.hypot(pos[:, 0], pos[:, 1]) < intr["sep_km"]
    if near.any():
        d[:, t < t0_near, :] = np.where(near[None, None, :], np.inf, d[:, t < t0_near, :])
    return d.min(axis=(1, 2))


def choose_trap_fb2(geo, goal_rel_deg, req_off, cur_off, dist_km, step_deg, speed_ms, turn_rate_dps, goal_km,
                    intr, dv_ms):
    """Trap fallback of fb2 (call it only when choose_offset_arc returned fallback=True).
    Candidates: offsets over the full circle x own speed (-dv, 0, +dv). Rules, in order:
    1. statically free and separation >= sep_km: the offset nearest to req_off, then speed 0 first, then the side
       nearer to the current heading; 2. statically free: the largest separation; 3. the longest free path (speed 0).
    Returns (offset deg, speed action -1/0/+1)."""
    if geo is None:
        geo = ([], None)
    dkm = min(float(dist_km), float(goal_km))
    cand = np.arange(-180.0, 180.0, float(step_deg))
    d = free_path(geo, goal_rel_deg + cand, speed_ms, turn_rate_dps, ARC_FALLBACK_FACTOR * dkm)
    ok_s = d >= dkm
    if not ok_s.any():
        return float(cand[int(d.argmax())]), 0
    if intr is None:                                   # static trap only: the free offset nearest to the request
        gap = np.abs(_wrap(cand - req_off))
        gap = np.where(ok_s, gap, np.inf)
        best = gap.min()
        tied = np.flatnonzero(np.abs(gap - best) <= 1e-9)
        k = tied[int(np.abs(_wrap(cand[tied] - cur_off)).argmin())]
        return float(cand[k]), 0
    best_key, best_sep = None, None
    for spd, dv in ((0, 0.0), (-1, -float(dv_ms)), (1, float(dv_ms))):
        v = max(speed_ms + dv, 50.0)
        rate = turn_rate_dps * speed_ms / v            # same bank angle: rate ~ 1 / speed
        sep = separation_all(goal_rel_deg + cand, v, rate, intr)
        for k in np.flatnonzero(ok_s):
            key = (0 if sep[k] >= intr["sep_km"] else 1, abs(_wrap(cand[k] - req_off)) if sep[k] >= intr["sep_km"]
                   else -sep[k], abs(spd), abs(_wrap(cand[k] - cur_off)))
            if best_key is None or key < best_key:
                best_key, best_sep = key, (float(cand[k]), spd)
    return best_sep


def speed_option(geo, goal_rel_deg, req_off, speed_ms, turn_rate_dps, dist_km, goal_km, intr, dv_ms):
    """mask_speed (2026-11-23): the request is blocked; test the own speed -dv and +dv on the requested heading.
    Returns the speed action (-1 / +1) whose path is statically free and keeps separation >= sep_km (the larger
    separation first), or 0 if neither does."""
    if intr is None:
        return 0
    if geo is None:
        geo = ([], None)
    dkm = min(float(dist_km), float(goal_km))
    tgt = np.array([goal_rel_deg + req_off])
    best, best_sep = 0, -np.inf
    for spd, dv in ((-1, -float(dv_ms)), (1, float(dv_ms))):
        v = max(speed_ms + dv, 50.0)
        rate = turn_rate_dps * speed_ms / v
        if free_path(geo, tgt, v, rate, dkm)[0] < dkm:
            continue
        sep = float(min_separation(tgt, v, rate, intr)[0])
        if sep >= intr["sep_km"] and sep > best_sep:
            best, best_sep = spd, sep
    return best
