"""Environment side: reward, frame stack, and a pool of environment processes.

The BlueSky environment code and its metrics are not changed. Reward weights go in through the
environment constructor. The time term, the event terms and the turn-change term are added here.
BlueSky allows one environment per process, so the pool has one process per environment.
"""
from __future__ import annotations
import os
from collections import deque
import multiprocessing as mp
import numpy as np
from .lidar import Lidar, own_state, state_dim, N_CHANNELS, DIST_NORM_KM, GEOM_KEY, NM2KM, to_own_frame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # no window for rgb_array rendering

METRIC_KEYS = ["waypoint_reached", "flight_time", "intrusion_events", "intrusion_time",
               "restricted_area_events", "time_in_restricted_area",
               "sector_exit_events", "time_outside_sector"]

# Reward. Unit of the time terms: per simulated second. Event terms: once per entry or exit.
# Hard constraint (user, 2026-11-11): the reward has only negative terms. No waypoint reward. check_reward
# rejects a positive weight.
# 2026-11-11 (user): the no-fly zone terms were -50 per second and -500 per entry. With these values the
# policy of sa_tf_neg_s0 flew a circle at its start point. The zone terms are now -10 and -100, and the
# goal-distance term gives a cost to each second far from the goal.
REWARD_DEFAULTS = dict(drift_penalty=-0.01, time_penalty=-5.0, intrusion_penalty=-10.0,
                       restricted_penalty=-10.0, sector_exit_penalty=-50.0,
                       zone_entry_penalty=-100.0, sector_exit_event_penalty=-500.0, turn_change_penalty=0.0,
                       goal_distance_penalty=-0.05, heading_penalty=0.0,
                       intrusion_event_penalty=0.0)

# Reward presets (--reward-preset). "friend": the reward that a friend of the user uses (user, 2026-11-11):
# r_t = -sum over the simulated seconds k of the step (100 x1_k + 10 x2_k + 10 x3_k + 1 (1 - cos psi_t) + 0.1),
# x1 = aircraft within 5 NM, x2 = in a restricted area, x3 = outside the sector, psi = angle between the
# heading and the bearing to the waypoint at the end of the step. All other terms are 0.
REWARD_PRESETS = {
    "friend": dict(drift_penalty=0.0, time_penalty=-0.1, intrusion_penalty=-100.0, restricted_penalty=-10.0,
                   sector_exit_penalty=-10.0, zone_entry_penalty=0.0, sector_exit_event_penalty=0.0,
                   turn_change_penalty=0.0, goal_distance_penalty=0.0, heading_penalty=-1.0,
                   intrusion_event_penalty=0.0),
    # "los" (user, 2026-11-11): loss of separation (LoS = another aircraft within 5 NM) -0.5 per second and
    # -3 per event, no-fly zone -0.3 per second and -2 per entry, outside the sector -0.2 per second,
    # time -0.015 per second. The user gave no other term, so all other terms are 0.
    "los": dict(drift_penalty=0.0, time_penalty=-0.015, intrusion_penalty=-0.5, restricted_penalty=-0.3,
                sector_exit_penalty=-0.2, zone_entry_penalty=-2.0, sector_exit_event_penalty=0.0,
                turn_change_penalty=0.0, goal_distance_penalty=0.0, heading_penalty=0.0,
                intrusion_event_penalty=-3.0)}
assert all(set(v) == set(REWARD_DEFAULTS) for v in REWARD_PRESETS.values())
REWARD_HELP = dict(
    drift_penalty="Per radian of drift, per step.",
    time_penalty="Per second flown (delta of the flight_time metric).",
    intrusion_penalty="Per second in intrusion.", restricted_penalty="Per second in a no-fly zone.",
    sector_exit_penalty="Per second outside the sector.",
    zone_entry_penalty="Once per entry into a no-fly zone.",
    sector_exit_event_penalty="Once per exit from the sector.",
    turn_change_penalty="Per unit of change of the turn action between two steps (give a negative value).",
    goal_distance_penalty="Per second flown and per km of distance to the goal (distance after the step).",
    heading_penalty="Per second flown, times (1 - cos psi); psi = angle between the heading and the bearing "
                    "to the goal after the step.",
    intrusion_event_penalty="Once per start of an intrusion (another aircraft comes within 5 NM).")


def add_reward_args(p):
    for k, v in REWARD_DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=float, default=None,
                       help=f"{REWARD_HELP[k]} Default: {v}, or the preset value.")
    p.add_argument("--reward-preset", choices=["none"] + list(REWARD_PRESETS), default="none",
                   help="A preset sets all reward weights. A weight flag that you give overrides the preset.")


def check_reward(rw):
    """Hard constraint: every reward weight is zero or negative, so the reward of each step is <= 0."""
    bad = {k: v for k, v in rw.items() if v > 0.0}
    if bad:
        raise ValueError(f"the reward has only negative terms; positive weights are not allowed: {bad}")
    return rw


def reward_kwargs(args):
    base = REWARD_PRESETS.get(getattr(args, "reward_preset", "none"), REWARD_DEFAULTS)
    for k, v in base.items():                    # order: given flag, preset, default. The result goes into
        if getattr(args, k) is None:             # args, so that the checkpoint holds the weights in use.
            setattr(args, k, v)
    return {k: getattr(args, k) for k in REWARD_DEFAULTS}


class FlightEnv:
    """One interface for the single-agent and the multi-agent competition environment.

    reset(seed) -> {id: obs}. step({id: action}) -> obs, reward, terminated, truncated, info (dicts by id).
    `agents` is the list of live aircraft ids after the call. `ids` is the fixed list of all ids.
    """

    def __init__(self, setting="multi", n_agents=2, render_mode=None, **reward):
        import bluesky_gym
        reward.pop("reach_reward", None)                 # key of older checkpoints; the waypoint reward is removed
        rw = check_reward({**REWARD_DEFAULTS, **reward})
        self.rw = rw
        kw = dict(reach_reward=0.0, drift_penalty=rw["drift_penalty"],
                  intrusion_penalty=rw["intrusion_penalty"], restricted_area_penalty=rw["restricted_penalty"],
                  sector_exit_penalty=rw["sector_exit_penalty"])
        self.setting = setting
        if setting == "single":
            import gymnasium as gym
            bluesky_gym.register_envs()
            self.env = gym.make("CompetitionEnv-v0", render_mode=render_mode, **kw)
            self.ids = ["own"]
        elif setting == "multi":
            from bluesky_zoo.competition_v0 import CompetitionZooEnv
            self.env = CompetitionZooEnv(n_agents=n_agents, render_mode=render_mode, **kw)
            self.ids = list(self.env.possible_agents)
        else:
            raise ValueError("setting must be single or multi")
        self.agents = []
        self._prev = {}

    def reset(self, seed=None):
        if self.setting == "single":
            o, _ = self.env.reset(seed=seed)
            obs = {"own": o}
        else:
            obs, _ = self.env.reset(seed=seed)
        self.agents = list(obs)
        self._prev = {a: dict(t=0.0, zone=0.0, exit=0.0, turn=0.0, intr=0.0) for a in self.agents}
        self._geom = {}
        self._add_geometry(obs)
        return obs

    def step(self, actions):
        if self.setting == "single":
            o, r, te, tr, info = self.env.step(np.asarray(actions["own"], dtype=np.float64))
            obs, rew, term, trunc, infos = {"own": o}, {"own": r}, {"own": te}, {"own": tr}, {"own": info}
            self.agents = [] if (te or tr) else ["own"]
        else:
            obs, rew, term, trunc, infos = self.env.step(actions)
            self.agents = list(self.env.agents)
        self._add_geometry(obs)
        rw, out_rew = self.rw, {}
        for a in obs:
            i, p = infos[a], self._prev[a]
            t, zone, ext = float(i["flight_time"]), float(i["restricted_area_events"]), float(i["sector_exit_events"])
            turn, intr = float(actions[a][0]), float(i["intrusion_events"])
            gx, gy = obs[a][GEOM_KEY]["goal"]                          # own frame: origin = own aircraft, x = nose
            goal_km = float(np.hypot(gx, gy))
            cos_psi = float(gx) / goal_km if goal_km > 1e-9 else 1.0
            out_rew[a] = (float(rew[a]) + rw["time_penalty"] * (t - p["t"])
                          + rw["goal_distance_penalty"] * goal_km * (t - p["t"])
                          + rw["heading_penalty"] * (1.0 - cos_psi) * (t - p["t"])
                          + rw["zone_entry_penalty"] * (zone - p["zone"])
                          + rw["sector_exit_event_penalty"] * (ext - p["exit"])
                          + rw["intrusion_event_penalty"] * (intr - p["intr"])
                          + rw["turn_change_penalty"] * abs(turn - p["turn"]))
            p.update(t=t, zone=zone, exit=ext, turn=turn, intr=intr)
        return obs, out_rew, {a: bool(term[a]) for a in obs}, {a: bool(trunc[a]) for a in obs}, infos

    def _ac_id(self, a):
        return self.env.unwrapped.agent if self.setting == "single" else a

    def _add_geometry(self, obs):
        """Add the exact simulator geometry for the lidar (own frame, km) to each observation (key GEOM_KEY).
        An aircraft that BlueSky has deleted (waypoint reached) keeps its last geometry."""
        import bluesky as bs
        inner = self.env.unwrapped
        sc = inner.scenario
        for a, o in obs.items():
            idx = bs.traf.id2idx(self._ac_id(a))
            if idx < 0:
                o[GEOM_KEY] = self._geom[a]
                continue
            lat0, lon0, hdg = float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), float(bs.traf.hdg[idx])
            frame = lambda ll: to_own_frame(np.asarray(ll)[:, 0], np.asarray(ll)[:, 1], lat0, lon0, hdg)
            others = np.arange(bs.traf.ntraf) != idx
            goal = (inner.goal_lat, inner.goal_lon) if self.setting == "single" else inner._goal[a]
            g = dict(zones=[frame(z.vertices) for z in sc.obstacles], sector=frame(sc.sector),
                     aircraft=to_own_frame(bs.traf.lat[others], bs.traf.lon[others], lat0, lon0, hdg).reshape(-1, 2),
                     goal=frame([goal])[0], goal_radius=float(inner.distance_margin),
                     own_lat=lat0, own_lon=lon0, own_hdg=hdg)
            self._geom[a] = o[GEOM_KEY] = g

    def render(self):
        return self.env.render()

    # Ray colours per hit type (no-fly zone, sector boundary, other aircraft, own goal) and for "no hit in range".
    RAY_COLORS = [(220, 40, 40), (30, 150, 30), (255, 140, 0), (255, 255, 255)]
    RAY_MISS = (150, 150, 150)

    def render_rays(self, obs, lidar, ray_step=2):
        """Frame with the lidar rays of each live aircraft drawn on the scene (as in navigation-mappo-rl).
        A ray ends at its nearest hit (colour of the hit type) or at the lidar range (grey).
        ray_step: draw every n-th ray, so that the GIF stays readable."""
        from PIL import Image, ImageDraw
        frame = self.render()
        inner = self.env.unwrapped
        img = Image.fromarray(frame)
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for a in self.agents:
            if a not in obs:
                continue
            g = obs[a][GEOM_KEY]
            x0, y0 = inner.projection.project(g["own_lat"], g["own_lon"])
            t, kind = lidar.nearest(obs[a])
            for r in range(0, lidar.n_rays, ray_step):
                hit = kind[r] >= 0
                length = t[r] if hit else lidar.max_range
                # ray angle: 0 = nose, positive = starboard = clockwise, as the aviation heading
                dx, dy = inner.projection.direction(g["own_hdg"] + lidar.angles_deg[r], length)
                col = self.RAY_COLORS[kind[r]] if hit else self.RAY_MISS
                draw.line([(x0, y0), (x0 + dx, y0 + dy)], fill=col + ((200,) if hit else (110,)), width=1)
                if hit:
                    draw.ellipse([x0 + dx - 2, y0 + dy - 2, x0 + dx + 2, y0 + dy + 2], fill=col + (255,))
        return np.asarray(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"))

    def close(self):
        self.env.close()


class ObsBuilder:
    """Frame stack of one aircraft: K frames of (own state, lidar scan), oldest first.
    At reset the older frames are zeros and the first observation is the newest frame
    (supersuit frame_stack_v1, as in the reference)."""

    def __init__(self, lidar, history=4):
        self.lidar, self.k = lidar, history
        self.frames = deque(maxlen=history)

    def reset(self, obs):
        f = (own_state(obs, (0.0, 0.0), self.lidar.zone_flag), self.lidar.scan(obs))
        z = (np.zeros_like(f[0]), np.zeros_like(f[1]))
        self.frames.clear()
        self.frames.extend([z] * (self.k - 1) + [f])

    def push(self, obs, action):
        self.frames.append((own_state(obs, action, self.lidar.zone_flag), self.lidar.scan(obs)))

    def get(self):
        """state [K * 7] (K * 8 with the inside-zone flag), lidar [K * 3, n_rays]."""
        return (np.concatenate([f[0] for f in self.frames]), np.concatenate([f[1] for f in self.frames], axis=0))


def obs_dims(lidar_cfg, history):
    return dict(state_dim=state_dim(lidar_cfg.get("zone_flag", True)) * history, lidar_ch=N_CHANNELS * history, n_rays=lidar_cfg["n_rays"])


def potential(obs, time_penalty, speed):
    """Shaping potential: time penalty times the estimated seconds to the waypoint at `speed` m/s."""
    return time_penalty * float(np.asarray(obs["waypoint_distance"]).reshape(-1)[0]) * DIST_NORM_KM * 1000.0 / speed


def metrics_row(info, ep_return):
    row = {k: float(info.get(k, 0.0)) for k in METRIC_KEYS}
    row["ep_return"] = float(ep_return)
    return row


def controller(obs, gain=1.0):
    """Scripted baseline: turn against the drift, no speed change. Gain 1 gives no oscillation."""
    return np.array([np.clip(-gain * float(obs["sin_drift"][0]), -1.0, 1.0), 0.0], dtype=np.float32)


def rollout_episode(env, policy_fn, seed, lidar, history, render=False, max_steps=400, rays=True):
    """One episode. policy_fn(raw obs by id, (state, lidar) by id) -> action by id.
    rays: draw the lidar rays on the frames (render only).
    Returns (metric rows by id, frames, turn actions by id)."""
    obs = env.reset(seed=seed)
    builders = {a: ObsBuilder(lidar, history) for a in obs}
    for a in obs:
        builders[a].reset(obs[a])
    grab = (lambda o: env.render_rays(o, lidar)) if rays else (lambda o: env.render())
    frames = [grab(obs)] if render else []
    ret, turns, rows, steps = {a: 0.0 for a in obs}, {a: [] for a in obs}, {}, 0
    while env.agents and steps < max_steps:
        live = list(env.agents)
        acts = policy_fn({a: obs[a] for a in live}, {a: builders[a].get() for a in live})
        obs, rew, term, trunc, infos = env.step(acts)
        steps += 1
        if render:
            frames.append(grab(obs))
        for a in live:
            ret[a] += rew[a]
            turns[a].append(float(acts[a][0]))
            builders[a].push(obs[a], acts[a])
            if term[a] or trunc[a]:
                rows[a] = metrics_row(infos[a], ret[a])
    return rows, frames, turns


def turn_stats(turns):
    """Mean absolute turn (degrees per step at 45 degrees per unit) and the share of sign flips."""
    t = np.asarray(turns, dtype=np.float64)
    s = np.sign(t[np.abs(t) > 1e-3])
    flips = float((s[1:] != s[:-1]).mean()) if len(s) > 1 else 0.0
    return {"mean_abs_turn_deg": float(np.abs(t).mean() * 45.0) if len(t) else 0.0, "turn_sign_flip_rate": flips}


# ------------------------------------------------------------------ pool of environment processes
def _episode_seed(cfg, worker, episode):
    if cfg["reset_seed"] is not None:
        return cfg["reset_seed"]
    return 1000 + cfg["seed"] * 1_000_000 + worker * 10_000 + episode


def _worker(conn, cfg, worker):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):          # BlueSky start-up text
        env = FlightEnv(cfg["setting"], cfg["n_agents"], **cfg["reward"])
    lidar, K = Lidar(**cfg["lidar"]), cfg["history"]
    ids = env.ids
    N, dims = len(ids), obs_dims(cfg["lidar"], cfg["history"])
    st = {"obs": None, "builders": None, "ret": None, "episode": 0, "started": False}

    def phi(o):
        return potential(o, cfg["reward"]["time_penalty"], cfg["shaping_speed"]) if cfg["shaping"] else 0.0

    def do_reset():
        st["obs"] = env.reset(seed=_episode_seed(cfg, worker, st["episode"]))
        st["episode"] += 1
        st["builders"] = {a: ObsBuilder(lidar, K) for a in st["obs"]}
        for a in st["obs"]:
            st["builders"][a].reset(st["obs"][a])
        st["ret"] = {a: 0.0 for a in ids}

    def pack(agents):
        state = np.zeros((N, dims["state_dim"]), dtype=np.float32)
        lid = np.zeros((N, dims["lidar_ch"], dims["n_rays"]), dtype=np.float32)
        alive = np.zeros(N, dtype=bool)
        for a in agents:
            i = ids.index(a)
            state[i], lid[i] = st["builders"][a].get()
            alive[i] = True
        return state, lid, alive

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "ids":
                conn.send(ids)
            elif cmd == "reset":
                # The reference resets at the start of every rollout. Aircraft still flying are cut (not logged).
                cut = len(env.agents) if st["started"] else 0
                do_reset()
                st["started"] = True
                conn.send((pack(env.agents), cut))
            elif cmd == "step":
                # black_death semantics of the reference: a finished aircraft gets its terminal observation
                # once, then a zero observation and reward 0. The done flag is set only when the whole
                # environment ends (all aircraft finished or time limit), for every aircraft.
                live = list(env.agents)
                acts = {a: np.asarray(data[ids.index(a)], dtype=np.float32) for a in live}
                nobs, rew, te, tr, infos = env.step(acts)
                r, p, pn = np.zeros(N, np.float32), np.zeros(N, np.float32), np.zeros(N, np.float32)
                episodes = []
                for a in live:
                    i = ids.index(a)
                    r[i], p[i] = rew[a], phi(st["obs"][a])
                    pn[i] = 0.0 if te[a] else phi(nobs[a])
                    st["ret"][a] += rew[a]
                    st["builders"][a].push(nobs[a], acts[a])
                    if te[a] or tr[a]:
                        episodes.append(metrics_row(infos[a], st["ret"][a]))
                st["obs"] = nobs
                done = not env.agents
                if done:
                    do_reset()
                    cur = pack(env.agents)
                else:
                    cur = pack(live)          # terminal observation of an aircraft that finished at this step
                conn.send(dict(cur=cur, rew=r, phi=p, phi_next=pn, done=np.full(N, done), episodes=episodes))
            elif cmd == "close":
                env.close()
                conn.close()
                return
    except (EOFError, KeyboardInterrupt):
        pass


class VecEnv:
    """E environment processes, stepped in lock-step. Arrays have the shape [E, N, ...]."""

    def __init__(self, cfg, n_envs):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(n_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, cfg, w), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.conns[0].send(("ids", None))
        self.ids = self.conns[0].recv()
        self.n_envs, self.n_agents = n_envs, len(self.ids)

    @staticmethod
    def _stack(packs):
        return tuple(np.stack(x) for x in zip(*packs))

    def reset(self):
        """Returns (state, lidar, alive) and the count of aircraft-episodes cut by this reset."""
        for c in self.conns:
            c.send(("reset", None))
        res = [c.recv() for c in self.conns]
        return self._stack([r[0] for r in res]), sum(r[1] for r in res)

    def step(self, actions):
        """actions [E, N, 2] -> dict of arrays [E, N, ...] and the list of finished episodes."""
        for c, a in zip(self.conns, actions):
            c.send(("step", a))
        res = [c.recv() for c in self.conns]
        out = {k: np.stack([r[k] for r in res]) for k in ["rew", "phi", "phi_next", "done"]}
        out["cur"] = self._stack([r["cur"] for r in res])
        out["episodes"] = [e for r in res for e in r["episodes"]]
        return out

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
