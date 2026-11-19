"""
Community leaderboard evaluation script (unofficial).

Runs the competition protocol -- the first 1000 episodes of seed 42 -- with
YOUR environment and policy, and writes a CSV you can upload to the
community leaderboard Space.

Setup: copy this file into the `scripts/` folder of the bluesky-gym
competition branch, edit the two functions in the EDIT block, then run from
the repo root:

    python -m scripts.evaluate_leaderboard --env sa     # -> leaderboard_sa.csv
    python -m scripts.evaluate_leaderboard --env ma     # -> leaderboard_ma.csv

Expected run time on a laptop CPU: ~25-55 min (sa), ~40 min-2.5 h (ma).
Use --episodes 10 for a quick check (such a file is rejected by the Space).

Do not change anything outside the EDIT block: the CSV contains a fingerprint
of every scenario, and the Space checks it against the official sequence.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import os

import numpy as np

SEED = 42
N_EPISODES = 1000
N_AGENTS_MA = 10
METRIC_KEYS = [
    "waypoint_reached", "flight_time",
    "intrusion_events", "intrusion_time",
    "restricted_area_events", "time_in_restricted_area",
    "sector_exit_events", "time_outside_sector",
    "total_reward",
]


# ==================== EDIT THESE TWO FOR YOUR SUBMISSION ======================
# bluesky_rl_transformer (2026-11-19): lidar ray-token Transformer policy, single agent only.
# The same code path as bluesky_rl_transformer/scripts/competition_eval.py (FlightEnv geometry, ObsBuilder frame
# stack, policy_fn_from, deterministic), so the CSV must equal results/<model>_competition_sa.json flight by flight.
# Model: environment variable TF_MODEL, default the big model.
_TF_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bluesky_rl_transformer"))
_TF_MODEL = os.environ.get("TF_MODEL", os.path.join(_TF_ROOT, "models", "sa_tf_info_ff_cls_big_s0.pt"))
_TF = {}


def _tf_setup():
    """Load the checkpoint once: policy, reward weights, lidar and history settings of the run."""
    if not _TF:
        import sys
        sys.path.insert(0, _TF_ROOT)
        from rl_tf.common import setup_torch
        from rl_tf.networks import load_policy as tf_load_policy
        from rl_tf.envs import REWARD_DEFAULTS, v2_kwargs
        from rl_tf.lidar import Lidar, token_cfg
        setup_torch(0)
        policy, ck = tf_load_policy(_TF_MODEL, "cpu")
        c = ck["args"]
        if c["setting"] != "single":
            raise ValueError(f"{_TF_MODEL}: setting {c['setting']}, this entry is for the single-agent track")
        _TF.update(policy=policy, history=c["history"], v2=v2_kwargs(c),
                   rew={**REWARD_DEFAULTS, "goal_distance_penalty": 0.0, **{k: c[k] for k in REWARD_DEFAULTS if k in c}},
                   lidar=Lidar(n_rays=c["n_rays"], fov_deg=c["fov_deg"], max_range_km=c["max_range_km"],
                               zone_flag=bool(c.get("zone_flag", 0)), speed=bool(c.get("lidar_speed", 0)),
                               **token_cfg(c)))
        print(f"policy {_TF_MODEL}", flush=True)
    return _TF


def make_env(kind):
    """Return your environment (subclass / wrapper stack of the official one).

    Keep the default constructor parameters (n_intruders, n_agents, n_obstacles,
    scoring params). You may change observation, reward, action mapping and
    action_frequency.
    """
    if kind != "sa":
        raise NotImplementedError("bluesky_rl_transformer entry: single agent (--env sa) only")
    import contextlib
    import io
    import gymnasium as gym
    tf = _tf_setup()
    from rl_tf.envs import FlightEnv, ObsBuilder
    with contextlib.redirect_stdout(io.StringIO()):          # BlueSky start-up text
        fe = FlightEnv("single", v2=tf["v2"], **tf["rew"])    # CompetitionEnv-v0, default scenario and scoring
        # v2 action_map "goal" (2026-11-19): heading offset from the goal bearing, clipped to +-45 deg (action mapping only)

    class TransformerObs(gym.Wrapper):
        """Observation only: adds the simulator geometry (own frame) and the frame stack (state, lidar) of the
        policy under the key "_tf_built". Scenario, dynamics and metrics are those of the wrapped environment."""

        def reset(self, **kw):
            obs, info = self.env.reset(**kw)
            fe._geom = {}
            fe._add_geometry({"own": obs})
            self.builder = ObsBuilder(tf["lidar"], tf["history"])
            self.builder.reset(obs)
            obs["_tf_built"] = self.builder.get()
            return obs, info

        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            env_action = fe.env_action({"own": action})["own"]
            obs, r, te, tr, info = self.env.step(np.asarray(env_action, dtype=np.float64))
            fe._add_geometry({"own": obs})
            self.builder.push(obs, action)
            obs["_tf_built"] = self.builder.get()
            return obs, r, te, tr, info

    return TransformerObs(fe.env)


def load_policy(kind):
    """Return ``act(obs, agent_id) -> action``.

    Example (SB3):
        from stable_baselines3 import PPO
        model = PPO.load("my_model.zip", device="cpu")
        return lambda obs, agent_id: model.predict(obs, deterministic=True)[0]
    """
    if kind != "sa":
        raise NotImplementedError("bluesky_rl_transformer entry: single agent (--env sa) only")
    tf = _tf_setup()
    from rl_tf.common import policy_fn_from
    fn = policy_fn_from(tf["policy"], "cpu", deterministic=True)
    return lambda obs, agent_id: fn({agent_id: obs}, {agent_id: obs["_tf_built"]})[agent_id]
# =============================================================================


def _round(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    return x


def scenario_hash(env):
    """Fingerprint of the scenario of the current episode (rounded so that it
    is stable across operating systems)."""
    scenario = env.unwrapped.scenario if hasattr(env, "unwrapped") else env.scenario
    payload = json.dumps(_round(dataclasses.asdict(scenario)), sort_keys=True, default=float)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_single_agent(n_episodes, act):
    env = make_env("sa")
    rows = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=SEED if ep == 0 else None)
        h = scenario_hash(env)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(act(obs, "KL001"))
            done = terminated or truncated
        rows.append({"episode": ep, "agent_id": "KL001", "scenario_hash": h, **info})
        _progress(ep, n_episodes)
    env.close()
    return rows


def run_multi_agent(n_episodes, act):
    env = make_env("ma")
    rows = []
    for ep in range(n_episodes):
        observations, infos = env.reset(seed=SEED if ep == 0 else None)
        h = scenario_hash(env)
        while env.agents:
            actions = {a: act(observations[a], a) for a in env.agents}
            observations, _, terms, truncs, infos = env.step(actions)
            for a in infos:
                if terms[a] or truncs[a]:
                    rows.append({"episode": ep, "agent_id": a, "scenario_hash": h, **infos[a]})
        _progress(ep, n_episodes)
    env.close()
    return rows


def _progress(ep, n):
    if (ep + 1) % 50 == 0 or ep + 1 == n:
        print(f"  {ep + 1}/{n} episodes", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["sa", "ma"], default="sa")
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    act = load_policy(args.env)
    run = run_single_agent if args.env == "sa" else run_multi_agent
    rows = run(args.episodes, act)

    out = args.out or f"leaderboard_{args.env}.csv"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "agent_id", "scenario_hash"] + METRIC_KEYS,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    goal = 100 * np.mean([r["waypoint_reached"] for r in rows])
    intr = np.mean([r["intrusion_events"] for r in rows])
    print(f"\nwrote {len(rows)} rows to {out}  (goal completion {goal:.1f}%, "
          f"intrusion events {intr:.2f}/flight)")
    if args.episodes != N_EPISODES:
        print(f"NOTE: {args.episodes} episodes only -- the leaderboard requires {N_EPISODES}.")


if __name__ == "__main__":
    main()
