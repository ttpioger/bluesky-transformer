"""One interface for the single-agent and the multi-agent competition environment.

The environment code and its metrics are not changed. The reward weights go in through the
environment constructor. The time term is added here from the `flight_time` metric.
BlueSky allows one environment per process.
"""
from __future__ import annotations
import os
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # no window for rgb_array rendering

METRIC_KEYS = ["waypoint_reached", "flight_time", "intrusion_events", "intrusion_time",
               "restricted_area_events", "time_in_restricted_area",
               "sector_exit_events", "time_outside_sector"]

# Reward (user decision 2026-11-10). Unit of the four time terms: per simulated second.
REWARD_DEFAULTS = dict(reach_reward=1.0, drift_penalty=-0.01, intrusion_penalty=-10.0,
                       restricted_penalty=-0.1, sector_exit_penalty=-0.1, time_penalty=-5.0)
WAYPOINT_NORM_M = 170.0 * 1000.0   # waypoint_distance is in units of 170 km


def add_reward_args(p):
    p.add_argument("--reach-reward", type=float, default=REWARD_DEFAULTS["reach_reward"])
    p.add_argument("--drift-penalty", type=float, default=REWARD_DEFAULTS["drift_penalty"],
                   help="Per radian of drift, per step.")
    p.add_argument("--intrusion-penalty", type=float, default=REWARD_DEFAULTS["intrusion_penalty"],
                   help="Per second in intrusion.")
    p.add_argument("--restricted-penalty", type=float, default=REWARD_DEFAULTS["restricted_penalty"],
                   help="Per second in a no-fly zone.")
    p.add_argument("--sector-exit-penalty", type=float, default=REWARD_DEFAULTS["sector_exit_penalty"],
                   help="Per second outside the sector.")
    p.add_argument("--time-penalty", type=float, default=REWARD_DEFAULTS["time_penalty"],
                   help="Per second flown (delta of the flight_time metric).")


def reward_kwargs(args):
    return {k: getattr(args, k) for k in REWARD_DEFAULTS}


class FlightEnv:
    """reset(seed) -> {id: obs}. step({id: action}) -> obs, reward, done, truncated, info (dicts by id).

    `agents` is the list of live aircraft ids after the call. `ids` is the fixed list of all ids.
    reward = environment reward + time_penalty * (seconds flown in this step).
    """

    def __init__(self, setting="multi", n_agents=2, render_mode=None, reach_reward=1.0, drift_penalty=-0.01,
                 intrusion_penalty=-10.0, restricted_penalty=-0.1, sector_exit_penalty=-0.1, time_penalty=-5.0):
        import bluesky_gym
        kw = dict(reach_reward=reach_reward, drift_penalty=drift_penalty, intrusion_penalty=intrusion_penalty,
                  restricted_area_penalty=restricted_penalty, sector_exit_penalty=sector_exit_penalty)
        self.setting = setting
        self.time_penalty = time_penalty
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
        self._t_prev = {}

    def reset(self, seed=None):
        if self.setting == "single":
            o, _ = self.env.reset(seed=seed)
            obs = {"own": o}
        else:
            obs, _ = self.env.reset(seed=seed)
        self.agents = list(obs)
        self._t_prev = {a: 0.0 for a in self.agents}
        return obs

    def step(self, actions):
        if self.setting == "single":
            o, r, te, tr, info = self.env.step(np.asarray(actions["own"], dtype=np.float64))
            obs, rew, term, trunc, infos = {"own": o}, {"own": r}, {"own": te}, {"own": tr}, {"own": info}
            self.agents = [] if (te or tr) else ["own"]
        else:
            obs, rew, term, trunc, infos = self.env.step(actions)
            self.agents = list(self.env.agents)
        out_rew = {}
        for a in obs:
            ft = float(infos[a]["flight_time"])
            out_rew[a] = float(rew[a]) + self.time_penalty * (ft - self._t_prev[a])
            self._t_prev[a] = ft
        return obs, out_rew, {a: bool(term[a]) for a in obs}, {a: bool(trunc[a]) for a in obs}, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def controller(obs, gain=3.0):
    """Scripted controller: turn against the drift, no speed change."""
    return np.array([np.clip(-gain * float(obs["sin_drift"][0]), -1.0, 1.0), 0.0], dtype=np.float32)


def seconds_to_go(obs, speed=150.0):
    """Estimated seconds to the waypoint at the nominal speed (used by the shaping potential)."""
    return float(obs["waypoint_distance"][0]) * WAYPOINT_NORM_M / speed


def metrics_row(info, ep_return):
    row = {k: float(info.get(k, 0.0)) for k in METRIC_KEYS}
    row["ep_return"] = float(ep_return)
    return row


def rollout_episode(env, policy_fn, seed, render=False, max_steps=400):
    """One episode. policy_fn({id: obs}) -> {id: action}. Returns (metric rows by id, frames)."""
    obs = env.reset(seed=seed)
    frames = [env.render()] if render else []
    ret = {a: 0.0 for a in obs}
    rows = {}
    steps = 0
    while env.agents and steps < max_steps:
        live = list(env.agents)
        acts = policy_fn({a: obs[a] for a in live})
        obs, rew, term, trunc, infos = env.step(acts)
        steps += 1
        if render:
            frames.append(env.render())
        for a in live:
            ret[a] += rew[a]
            if term[a] or trunc[a]:
                rows[a] = metrics_row(infos[a], ret[a])
    return rows, frames
