# BlueSky structured ViT + PPO / MAPPO (v2)

This revision uses the observation schema from the supplied competition
`GETTING_STARTED.md` and `core/observations.py`. It does **not** assume
`P_x, P_y, V_x, V_y` ownship fields: the actual base observation is a Dict:
waypoint drift/distance, own airspeed, relative intruder state, obstacle
geometry, and sector-boundary points.

The encoder converts the Dict into typed numeric tokens:
- ownship token: normalized airspeed and `inside_sector`
- waypoint tokens: `cos_drift`, `sin_drift`, `waypoint_distance`
- intruder tokens: `x_r`, `y_r`, `vx_r`, `vy_r`, `cos_track`, `sin_track`, `intruder_distance`
- obstacle tokens: distance, bearing cosine/sine, radius
- sector tokens: distance and bearing cosine/sine

It is a **structured-observation Transformer**, not an ImageNet image ViT.
The observation values are already normalized by the competition components.

## Install

Run from the root of the cloned `AI4REAL-NET-Competition` repository:

```bash
uv sync
uv add torch
```

The repository's existing dependencies should include Gymnasium, Stable-Baselines3,
PettingZoo and SuperSuit if you use the corresponding paths. If PyTorch is already
in the lockfile, `uv add torch` is unnecessary.

Copy the `bluesky_vit/` and `scripts/` directories from this bundle into the
repository root (merge with existing `scripts/`; do not overwrite the official
`evaluate_competition.py`).

## 1. Optional self-supervised pretraining

```bash
uv run python -m scripts.pretrain_encoder \
  --transitions 50000 --epochs 30 --batch-size 256
```

This collects observations from `CompetitionEnv-v0` with random actions and
trains masked-token reconstruction. It does not optimize the competition
reward. On a GTX 1060 6 GB, start with `--batch-size 64` or `128` if memory
is tight. Simulator data collection is CPU/BlueSky-bound; GPU use occurs in
the encoder training phase.

## 2. Single-agent PPO (required baseline track)

```bash
uv run python -m scripts.train_ppo \
  --timesteps 1000000 \
  --pretrained checkpoints/bluesky_vit_encoder.pt
```

Omit `--pretrained` to train from scratch.

## 3. Multi-agent MAPPO (centralized critic)

```bash
uv run python -m scripts.train_mappo \
  --n-agents 10 --steps 100000 \
  --pretrained checkpoints/bluesky_vit_encoder.pt
```

This script implements a shared structured-ViT actor and a centralized critic
that consumes the joint observation of all agents. It is an experimental
research trainer; validate it with short runs before committing to long runs.
BlueSky is a process-global singleton, so keep one BlueSky-backed environment
per process.

## 4. Quick MAPPO evaluation

```bash
uv run python -m scripts.evaluate_mappo \
  --model models/mappo_bluesky_vit.pt --episodes 20
```

This is a convenience evaluator. For competition reporting, use the official
`uv run python -m scripts.evaluate_competition --env ma` script, and connect
the saved actor to its documented policy-loader edit point. The official
evaluator is the authority for the 1000 episodes / seed 42 protocol.

## Reward and objective metrics

The supplied multi-agent environment's default training reward is:

```python
reach_reward * waypoint_reached_delta
+ drift_penalty * drift_radians
+ intrusion_penalty * intrusion_time_delta
+ restricted_area_penalty * restricted_time_delta
+ sector_exit_penalty * outside_sector_time_delta
```

Defaults in the supplied `CompetitionZooEnv` are reach `+1.0`, drift `-0.01`,
intrusion-time `-1.0`, restricted-area-time `-1.0`, and outside-sector-time
`-1.0`. These are training rewards, not the competition score. The fixed
scoring metrics are reported in each agent's `info`:
`waypoint_reached`, `intrusion_events`, `intrusion_time`,
`restricted_area_events`, `time_in_restricted_area`, `sector_exit_events`,
`time_outside_sector`, and `flight_time`. Do not change `_update_metrics`
or `_get_info`.

The single-agent `CompetitionEnv` may have its own constructor defaults;
this bundle leaves the environment reward untouched and learns from whatever
`env.step()` returns. Verify the single-agent reward implementation in
`bluesky_gym/envs/competition_env.py` before assuming its coefficients match
the multi-agent defaults.

## Important validation

These scripts were generated from the docs and observation component code
provided in the conversation. They have been syntax-checked, but not executed
against your local BlueSky install. Run a smoke test first:

```bash
uv run python -m scripts.pretrain_encoder --transitions 100 --epochs 1 --batch-size 16
uv run python -m scripts.train_mappo --n-agents 3 --steps 30 --rollout-steps 8 --epochs 1 --minibatch-size 16
```

If a smoke test fails, share the complete traceback plus the exact current
`train_zoo.py`, `bluesky_zoo/competition_v0.py` import path, and
`bluesky_gym/envs/competition_env.py`. BlueSky APIs can differ by branch/version.
