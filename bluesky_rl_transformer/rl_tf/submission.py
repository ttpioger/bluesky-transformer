"""Adapter for the official harness scripts/evaluate_competition.py (repo root), 2026-11-21.

The harness calls load_policy(kind, model_path) -> act(obs), then make_env(kind) and runs its own fixed loop:
  sa: obs, info = env.reset(seed=...); obs, _, te, tr, info = env.step(act(obs)) until te or tr; record = info
  ma: obs, infos = env.reset(seed=...); actions = {a: act(obs[a]) for a in env.agents}; env.step(actions) ...
SubmissionEnv wraps the base competition environment with FlightEnv: the lidar geometry, the observation
history (ObsBuilder), the action mapping of the training (action_map "goal") and the heading mask (settings from
the checkpoint; environment variable BLUESKY_TF_HEADING_MASK=0|1 overrides heading_mask). The infos are those of
the base environment, so the harness scores the same metrics; the scoring code, the scenarios and the fixed
parameters are not changed. The env runs the policy once for all live aircraft together (the same joint pass as
scripts.competition_eval, so the same floats) and puts each action into the observation under ACTION_KEY;
act() returns that action.
"""
from __future__ import annotations
import contextlib
import io
import os
import numpy as np
import torch
from .envs import FlightEnv, ObsBuilder, REWARD_DEFAULTS, v2_kwargs
from .lidar import Lidar, token_cfg
from .networks import load_policy as _load
from .common import policy_fn_from

ACTION_KEY = "tf_action"


def load_member(path, device="cpu"):
    """(policy, deterministic policy_fn, reward weights, Lidar, history, v2 settings) of a checkpoint."""
    policy, ck = _load(path, device)                     # no policy.eval(): the same module mode as competition_eval
                                                         # (eval mode switches torch to another Transformer kernel)
    c = ck["args"]
    rew = {**REWARD_DEFAULTS, "goal_distance_penalty": 0.0, **{k: c[k] for k in REWARD_DEFAULTS if k in c}}
    lidar = Lidar(n_rays=c["n_rays"], fov_deg=c["fov_deg"], max_range_km=c["max_range_km"],
                  zone_flag=bool(c.get("zone_flag", 0)), speed=bool(c.get("lidar_speed", 0)), **token_cfg(c))
    over = os.environ.get("BLUESKY_TF_HEADING_MASK")
    v2 = v2_kwargs({**c, **({"heading_mask": int(over)} if over not in (None, "") else {})})
    return policy, policy_fn_from(policy, device, deterministic=True), rew, lidar, c["history"], v2


def make_act(member):
    """act(obs) -> action for one aircraft; the action is in the observation (SubmissionEnv)."""
    def act(obs):
        return obs[ACTION_KEY]
    return act


class SubmissionEnv:
    """The base competition environment with the model inputs, the action mapping and the heading mask."""

    def __init__(self, kind, member, n_agents=10):
        self.kind = kind
        _, self.fn, rew, self.lidar, self.history, v2 = member
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.fe = FlightEnv("single" if kind == "sa" else "multi", n_agents, v2=v2, **rew)
        self.builders = {}

    @property
    def agents(self):
        return list(self.fe.agents)

    def _with_input(self, obs, ids):
        live = [a for a in self.fe.agents if a in obs]    # aircraft that act next (joint pass, as rollout_episode)
        if live:
            acts = self.fn({a: obs[a] for a in live}, {a: self.builders[a].get() for a in live})
            for a in live:
                obs[a][ACTION_KEY] = acts[a]
        return obs

    def reset(self, seed=None, options=None):
        obs = self.fe.reset(seed=seed)
        self.builders = {a: ObsBuilder(self.lidar, self.history) for a in obs}
        for a in obs:
            self.builders[a].reset(obs[a])
        obs = self._with_input(obs, list(obs))
        return (obs["own"], {}) if self.kind == "sa" else (obs, {a: {} for a in obs})

    def step(self, action):
        acts = {"own": np.asarray(action, dtype=np.float32)} if self.kind == "sa" else dict(action)
        obs, rew, te, tr, infos = self.fe.step(acts)
        done = self.fe.last_exec or acts                   # heading mask on: the executed actions
        for a in obs:
            self.builders[a].push(obs[a], done[a])
        obs = self._with_input(obs, list(obs))
        if self.kind == "sa":
            return obs["own"], rew["own"], te["own"], tr["own"], infos["own"]
        return obs, rew, te, tr, infos

    def close(self):
        self.fe.close()
