"""Training-only fast path for the scored metrics of the multi-agent competition environment (2026-11-20).

CompetitionZooEnv._update_metrics runs once per aircraft after each 1 s simulation step. For each aircraft it calls
kwikqdrdist once per other aircraft and BlueSky checkInside once per area (one Python call per point). A cProfile on
the laptop (10 aircraft, 400 decisions) gave 28 % of the loop time in this code.

install(env) replaces _update_metrics on one environment instance. The new method computes, once per simulation
step and for all live aircraft together: the pair distances, the area and sector membership and the goal distance.
It then updates the metrics of the aircraft as the original code does (same rules, same edge detection).
The element-wise float operations are the same as in core.tools.kwikqdrdist and BlueSky checkInside, so the
metrics are bit-identical (check: scripts.check_fast_path). The original class and the evaluation and scoring
code are not changed.
"""
from __future__ import annotations
import types
import numpy as np


def dist_nm(lata, lona, latb, lonb):
    """Distance part of core.tools.kwikqdrdist: the same operations in the same order, element-wise."""
    nm = 1852.
    re = 6371000.
    dlat = np.radians(latb - lata)
    dlon = np.radians(((lonb - lona) + 180) % 360 - 180)
    cavelat = np.cos(np.radians(lata + latb) * 0.5)
    dangle = np.sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    return re * dangle / nm


def _inside(bs, name, lat, lon, alt):
    """BlueSky checkInside for many points; one point is duplicated (checkInside needs more than a 0-d input)."""
    if len(lat) == 1:
        return np.asarray(bs.tools.areafilter.checkInside(name, np.repeat(lat, 2), np.repeat(lon, 2),
                                                          np.repeat(alt, 2)))[:1].astype(bool)
    return np.asarray(bs.tools.areafilter.checkInside(name, lat, lon, alt)).astype(bool)


def _snapshot(env, bs, nm2km):
    """Metric inputs of all live aircraft at this simulation step. Cached until the positions, the live list or
    the episode change."""
    agents = list(env.agents)
    key = (env._fm_gen, tuple(agents), bs.traf.lat.tobytes(), bs.traf.lon.tobytes(), bs.traf.alt.tobytes())
    if env._fm_key == key:
        return env._fm_snap
    idx = np.array([bs.traf.id2idx(a) for a in agents], dtype=int)
    ok = idx >= 0
    live = [a for a, o in zip(agents, ok) if o]
    li = idx[ok]
    lat, lon, alt = bs.traf.lat[li], bs.traf.lon[li], bs.traf.alt[li]
    snap = {"row": {a: k for k, a in enumerate(live)}}
    if len(live):
        d = dist_nm(lat[:, None], lon[:, None], lat[None, :], lon[None, :])      # [own, other]
        close = d < env.intrusion_distance
        np.fill_diagonal(close, False)
        snap["intr"] = [{live[j] for j in np.flatnonzero(close[i])} for i in range(len(live))]
        areas = {name: _inside(bs, name, lat, lon, alt) for name in env._obstacle_area_names}
        snap["obs"] = [{name for name in env._obstacle_area_names if areas[name][i]} for i in range(len(live))]
        snap["outside"] = ~_inside(bs, "sector", lat, lon, alt)
        glat = np.array([env._goal[a][0] for a in live], dtype=np.float64)
        glon = np.array([env._goal[a][1] for a in live], dtype=np.float64)
        snap["goal"] = dist_nm(lat, lon, glat, glon) * nm2km < env.distance_margin
    env._fm_key, env._fm_snap = key, snap
    return snap


def install(env):
    """Replace _update_metrics of this CompetitionZooEnv instance by the batched version. Returns env."""
    import bluesky as bs
    import sys
    nm2km = sys.modules[type(env).__module__].NM2KM
    orig_update, orig_reset = env._update_metrics, env.reset
    env._fm_gen, env._fm_key, env._fm_snap = 0, None, None

    def reset(*args, **kwargs):
        env._fm_gen += 1                                # new scenario: new areas and goals
        env._fm_key = None
        return orig_reset(*args, **kwargs)

    def update_metrics(self, ac_id, dt=None):
        if dt is None:
            dt = self.sim_dt
        snap = _snapshot(self, bs, nm2km)
        k = snap["row"].get(ac_id)
        if k is None:                                   # aircraft not in the traffic arrays: original code path
            return orig_update(ac_id, dt)
        m = self.metrics[ac_id]
        m["flight_time"] += dt
        now_intr = set(snap["intr"][k])
        m["intrusion_events"] += len(now_intr - self._in_intrusion_with[ac_id])
        m["intrusion_time"] += len(now_intr) * dt
        self._in_intrusion_with[ac_id] = now_intr
        now_obs = set(snap["obs"][k])
        m["restricted_area_events"] += len(now_obs - self._obstacles_occupied[ac_id])
        if now_obs:
            m["time_in_restricted_area"] += dt
        self._obstacles_occupied[ac_id] = now_obs
        outside = bool(snap["outside"][k])
        if outside and not self._outside_sector[ac_id]:
            m["sector_exit_events"] += 1
        if outside:
            m["time_outside_sector"] += dt
        self._outside_sector[ac_id] = outside
        if bool(snap["goal"][k]) and not self._waypoint_reached[ac_id]:
            self._waypoint_reached[ac_id] = True
            m["waypoint_reached"] = 1

    env.reset = reset
    env._update_metrics = types.MethodType(update_metrics, env)
    return env
