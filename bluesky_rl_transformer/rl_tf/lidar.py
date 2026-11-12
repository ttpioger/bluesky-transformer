"""Lidar for the BlueSky competition task, cast on the exact simulator geometry (as the reference).

The reference method (avbiswas/navigation-mappo-rl, nav/ray_intersection.py and nav/environment.py)
casts its rays on the exact shapes of its simulator. Each ray keeps only its nearest hit of any type.
The hit goes into one of 3 channels (obstacle, boundary, agent); the other channels stay 0.
A ray whose nearest hit is the own goal gives 0 in all channels. Value: max_range - t.

This module does the same on BlueSky. FlightEnv (envs.py) adds the key GEOM_KEY to each observation:
- "zones":    list of [V, 2] arrays, the no-fly zone polygons (scenario.obstacles vertices);
- "sector":   [S, 2] array, the sector polygon (scenario.sector, all points);
- "aircraft": [M, 2] array, the other aircraft, each a disc with the intrusion radius (5 NM);
- "goal":     [2] array, the own waypoint, a disc with the capture radius ("goal_radius", km).
Frame: own aircraft at the origin, x = nose, y = starboard, unit km.

Ray value: closeness = 1 - t / max_range, i.e. (max_range - t) / max_range, the reference value divided
by the constant max_range. Inside a disc, the ray gives the distance to the exit point (reference rule).
Inside a zone polygon, the ray gives the distance to the polygon edge where it leaves the zone.
"""
from __future__ import annotations
import numpy as np

NM2KM = 1.852
DEG2KM = 111.32                                        # flat-earth factor of core/rendering.py
DIST_NORM_KM = 170.0                                   # waypoint distance unit of the observation (potential shaping)
GEOM_KEY = "lidar_geometry"
N_CHANNELS = 3                                         # zone, boundary, aircraft (as the reference)
AC_FEATURES = 6                                        # aircraft token: x, y, vx, vy, distance, closing speed
SPEED_NORM_MS = 350.0                                  # closing-speed unit: largest value measured was 314 m/s (2026-11-12)
N_TYPES = 4                                            # the 3 channels + the own goal (no channel)
TYPE_NAMES = ["zone", "boundary", "aircraft", "goal"]
OWN_KEYS = ["airspeed", "inside_sector", "cos_drift", "sin_drift", "waypoint_distance"]
STATE_DIM = len(OWN_KEYS) + 2                          # own features + previous action (turn, speed)


def state_dim(zone_flag=True):
    """Size of the own state of one frame. zone_flag adds the inside-zone flag (2026-11-11)."""
    return STATE_DIM + int(bool(zone_flag))


def ray_circles(u, c, r):
    """Distance along each ray to each circle, reference rule (nav/ray_intersection.py):
    the first crossing with t >= 0; inside the circle this is the exit point.
    u: [R, 2] unit vectors, c: [M, 2], r: [M]. Returns [R, M]; inf = no hit."""
    if len(r) == 0:
        return np.full((u.shape[0], 0), np.inf)
    b = u @ c.T                                        # [R, M] projection of the centre on the ray
    cc = (c * c).sum(1) - r * r                        # [M]; negative = origin inside
    disc = b * b - cc[None, :]
    sq = np.sqrt(np.maximum(disc, 0.0))
    t1, t2 = b - sq, b + sq
    ok = disc >= 0.0
    return np.where(ok & (t1 >= 0.0), t1, np.where(ok & (t2 >= 0.0), t2, np.inf))


def ray_segments(u, p, q):
    """Distance along each ray to each segment p->q. u: [R, 2], p and q: [M, 2]. Returns [R, M]."""
    if len(p) == 0:
        return np.full((u.shape[0], 0), np.inf)
    d = q - p                                          # [M, 2]
    denom = u[:, 0:1] * d[None, :, 1] - u[:, 1:2] * d[None, :, 0]         # cross(u, d)  [R, M]
    p_x_d = (p[:, 0] * d[:, 1] - p[:, 1] * d[:, 0])[None, :]              # cross(p, d)  [1, M]
    p_x_u = p[None, :, 0] * u[:, 1:2] - p[None, :, 1] * u[:, 0:1]         # cross(p, u)  [R, M]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = p_x_d / denom
        s = p_x_u / denom
    ok = (np.abs(denom) > 1e-12) & (t >= 0.0) & (s >= 0.0) & (s <= 1.0)
    return np.where(ok, t, np.inf)


def polygon_edges(polys):
    """List of closed polygons [V, 2] -> start and end points of all edges, [M, 2] each."""
    if not polys:
        return np.zeros((0, 2)), np.zeros((0, 2))
    p = np.concatenate([np.asarray(v, dtype=np.float64) for v in polys])
    q = np.concatenate([np.roll(np.asarray(v, dtype=np.float64), -1, axis=0) for v in polys])
    return p, q


def to_own_frame(lat, lon, lat0, lon0, hdg_deg):
    """Flat-earth lat/lon (degrees) -> own frame km (x = nose, y = starboard). Heading: 0 = north, clockwise."""
    east = (np.asarray(lon, dtype=np.float64) - lon0) * np.cos(np.deg2rad(lat0)) * DEG2KM
    north = (np.asarray(lat, dtype=np.float64) - lat0) * DEG2KM
    h = np.deg2rad(hdg_deg)
    return np.stack([east * np.sin(h) + north * np.cos(h), east * np.cos(h) - north * np.sin(h)], axis=-1)


class Lidar:
    def __init__(self, n_rays=120, fov_deg=360.0, max_range_km=100.0, ac_radius_km=5.0 * NM2KM, zone_flag=True,
                 speed=False, ac_tokens=0):
        self.n_rays, self.fov_deg, self.max_range, self.ac_radius = n_rays, fov_deg, max_range_km, ac_radius_km
        self.zone_flag = bool(zone_flag)               # the own state has the inside-zone flag
        self.speed = bool(speed)                       # 4th channel: closing speed of the aircraft that the ray hits
        self.n_channels = N_CHANNELS + int(self.speed)
        self.ac_tokens = int(ac_tokens)                # aircraft tokens: the K nearest aircraft that the lidar can see
        if fov_deg >= 360.0:
            ang = -180.0 + 360.0 * np.arange(n_rays) / n_rays          # ring, no doubled ray
        else:
            ang = np.linspace(-fov_deg / 2.0, fov_deg / 2.0, n_rays)
        self.angles_deg = ang                                           # 0 = nose, positive = starboard
        a = np.deg2rad(ang)
        self.u = np.stack([np.cos(a), np.sin(a)], axis=1)

    def distances(self, obs):
        """Nearest hit distance per object type and ray, km. [4, R] (zone, boundary, aircraft, goal); inf = no hit."""
        g = obs[GEOM_KEY]
        out = np.full((N_TYPES, self.n_rays), np.inf)
        p, q = polygon_edges(g["zones"])
        if len(p):
            out[0] = ray_segments(self.u, p, q).min(axis=1)
        p, q = polygon_edges([g["sector"]])
        if len(p):
            out[1] = ray_segments(self.u, p, q).min(axis=1)
        ac = np.asarray(g["aircraft"], dtype=np.float64).reshape(-1, 2)
        if len(ac):
            out[2] = ray_circles(self.u, ac, np.full(len(ac), self.ac_radius)).min(axis=1)
        out[3] = ray_circles(self.u, np.asarray(g["goal"], dtype=np.float64).reshape(1, 2),
                             np.array([float(g["goal_radius"])]))[:, 0]
        return out

    def nearest(self, obs):
        """Reference rule: per ray the nearest hit within range. Returns (t km [R], type index [R]; -1 = no hit)."""
        d = self.distances(obs)
        kind = d.argmin(axis=0)
        t = d[kind, np.arange(self.n_rays)]
        kind = np.where(t <= self.max_range, kind, -1)
        return t, kind

    def scan(self, obs):
        """Closeness in [0, 1], one non-zero channel per ray at most. [3, R] float32.
        With speed: [4, R]; channel 3 = closing speed of the aircraft that the ray hits, in [-1, 1]."""
        t, kind = self.nearest(obs)
        out = np.zeros((self.n_channels, self.n_rays), dtype=np.float32)
        r = np.flatnonzero((kind >= 0) & (kind < N_CHANNELS))          # goal hits and misses stay 0
        out[kind[r], r] = (1.0 - t[r] / self.max_range).astype(np.float32)
        if self.speed:
            out[N_CHANNELS] = self.closing_speed(obs, kind)
        return out

    def aircraft_tokens(self, obs):
        """The K = ac_tokens nearest other aircraft whose 5 NM disc is within the lidar radius, nearest first.
        Returns [K * AC_FEATURES + K] float32: K rows of (x, y, vx, vy, distance, closing speed), then K flags
        (1 = a real aircraft, 0 = empty row). Own frame (x = nose, y = starboard). Positions and the distance are
        divided by (lidar radius + disc radius); velocities (other minus own) and the closing speed by
        SPEED_NORM_MS. Empty rows are 0."""
        K = self.ac_tokens
        feat, flag = np.zeros((K, AC_FEATURES), dtype=np.float32), np.zeros(K, dtype=np.float32)
        g = obs[GEOM_KEY]
        ac = np.asarray(g["aircraft"], dtype=np.float64).reshape(-1, 2)
        if K and len(ac):
            vel = g.get("aircraft_vel")
            vel = np.zeros_like(ac) if vel is None else np.asarray(vel, dtype=np.float64).reshape(-1, 2)
            reach = self.max_range + self.ac_radius
            dist = np.linalg.norm(ac, axis=1)
            order = np.argsort(dist, kind="stable")
            order = order[dist[order] <= reach][:K]
            n = len(order)
            if n:
                c, v, d = ac[order], vel[order], np.maximum(dist[order], 1e-9)
                closing = -(c * v).sum(axis=1) / d
                feat[:n] = np.concatenate([c / reach, np.clip(v / SPEED_NORM_MS, -2.0, 2.0), (d / reach)[:, None],
                                           np.clip(closing / SPEED_NORM_MS, -1.0, 1.0)[:, None]], axis=1)
                flag[:n] = 1.0
        return np.concatenate([feat.reshape(-1), flag])

    def closing_speed(self, obs, kind):
        """Per ray whose nearest hit is an aircraft: the rate at which the distance between the own aircraft and
        that aircraft decreases, divided by SPEED_NORM_MS and clipped to [-1, 1]. Positive = it comes nearer.
        Other rays: 0. Needs "aircraft_vel" ([M, 2], velocity of each aircraft minus the own velocity, own
        frame, m/s) in the geometry; without it the channel is 0."""
        out = np.zeros(self.n_rays, dtype=np.float32)
        g = obs[GEOM_KEY]
        r = np.flatnonzero(kind == 2)
        vel = g.get("aircraft_vel")
        if len(r) == 0 or vel is None:
            return out
        ac = np.asarray(g["aircraft"], dtype=np.float64).reshape(-1, 2)
        vel = np.asarray(vel, dtype=np.float64).reshape(-1, 2)
        m = ray_circles(self.u[r], ac, np.full(len(ac), self.ac_radius)).argmin(axis=1)     # the aircraft of each ray
        dist = np.maximum(np.linalg.norm(ac[m], axis=1), 1e-9)
        closing = -(ac[m] * vel[m]).sum(axis=1) / dist                                      # m/s
        out[r] = np.clip(closing / SPEED_NORM_MS, -1.0, 1.0).astype(np.float32)
        return out


def inside_zone(obs):
    """1.0 if the own aircraft (origin of the own frame) is inside a no-fly zone polygon, else 0.0.
    Even-odd rule: a ray from the origin crosses the polygon edges an odd number of times."""
    u = np.array([[1.0, 0.0]])
    for z in obs[GEOM_KEY]["zones"]:
        p, q = polygon_edges([z])
        if int(np.isfinite(ray_segments(u, p, q)).sum()) % 2 == 1:
            return 1.0
    return 0.0


def own_state(obs, prev_action, zone_flag=False):
    """Own-aircraft state of one frame: 5 own features, the inside-zone flag (if zone_flag), the previous
    action. [7] or [8] float32."""
    own = [float(np.asarray(obs[k]).reshape(-1)[0]) for k in OWN_KEYS]
    if zone_flag:
        own.append(inside_zone(obs))
    return np.asarray(own + [float(prev_action[0]), float(prev_action[1])], dtype=np.float32)
