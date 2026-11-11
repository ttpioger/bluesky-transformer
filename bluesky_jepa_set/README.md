# bluesky_jepa_set: Transformer and JEPA cells against the 1D CNN, overfit test

Status of all results from this folder: EXPLORATORY.

## Question

Can RL with a Transformer encoder learn one fixed BlueSky competition scenario on a CPU laptop, where the 1D CNN encoder of the reference method is the control? Does a temporal JEPA change the result?

## Cells

| Cell | `--encoder` | `--jepa` | Content |
|---|---|---|---|
| R | cnn | off | 1D CNN (port of the reference encoder), RL only |
| A | set_tf | off | Query-centric Transformer, RL only |
| B | cnn | on | 1D CNN, JEPA pretraining plus JEPA auxiliary loss |
| C | set_tf | on | Transformer, JEPA pretraining plus JEPA auxiliary loss |

All cells use the same scenario, seeds, step budget, reward, reward scale, PPO code, actor head and critic head. Only the encoder and the JEPA flag differ. Single agent (PPO, `CompetitionEnv-v0`) and multi agent (MAPPO, `CompetitionZooEnv`, 2 aircraft) use one code path.

## Parameter counts (n_params)

| Part | cnn | set_tf |
|---|---|---|
| Actor (encoder + head + log_std) | 86 941 | 87 044 |
| Encoder only | 78 489 | 78 592 |
| Critic (own encoder + set attention) | 112 154 | 112 257 |
| JEPA-only parts (GRU + predictor, cells B and C) | 33 408 | 33 408 |

The actor counts differ by 0.1 percent. The JEPA-only parts are not used to select actions.

## Reward (user decision, 2026-11-10)

Per simulated second: -5 flown (delta of the `flight_time` metric), -10 in intrusion, -0.1 in a no-fly zone, -0.1 outside the sector. Also +1 for the waypoint and -0.01 per radian of drift per step. Each value is a flag. The environment code and its metrics are not changed: four weights go through the environment constructor, and `jepa_set/envs.py` adds the time term.

Learning uses the reward times 0.001 (`--reward-scale`) plus a potential-based shaping term (`--shaping potential`). The potential is the time penalty times the estimated seconds to the waypoint at 150 m/s. This term does not change the best policy. `ep_return` in all outputs is the unscaled reward without the shaping term.

Caveat: the time term is 50 times the no-fly-zone term. A flight through a no-fly zone can be the best policy for this reward.

## Pass criterion (fixed before the runs)

Over the last 20 agent-episodes of training: mean `waypoint_reached` >= 0.95; mean `intrusion_events`, `restricted_area_events` and `sector_exit_events` each <= 0.05; mean `ep_return` above the mean of the first 20 agent-episodes by more than 2 standard errors. `steps_to_criterion` is the environment step count at the end of the first 20-episode window that meets this rule and does not overlap the first window. Training episodes include exploration noise. `greedy_eval` in the summary is one flight without noise.

Supervised sanity test (before any RL claim): each encoder plus the actor head must fit the scripted controller (turn = -3 times sin_drift, clipped). Pass: final MSE <= 0.005 and the waypoint reached in one closed-loop flight.

## What a pass shows

An overfit pass shows only that the learning loop works on one scenario. It does not show that a cell is better or worse in general. Three seeds and one scenario give no general claim. No number can be compared with the reference method (environment, aircraft count, steps, reward: all not matched; see `results/compare.md`).

## Known limits

- The JEPA cells use 6000 extra transitions and extra compute before RL. The RL step budget is equal across cells.
- The CNN port is not the reference encoder: one Conv1d branch per object type, kernel 3, no frame stack (see `jepa_set/encoders.py`).
- The Transformer loses the outline order of the sector edge points.
- The actor does not use the frame history. Only the JEPA loss uses it.
- Not run by the author of the code: any training run, any timing run of 3000 steps or more.

## Files

- `design_review.md`: design choices, references, risks, stop rules.
- `RUN.md`: setup, smoke commands, run commands, output files.
- `jepa_set/`: `obs.py`, `encoders.py`, `policy.py` (actor, critic, JEPA), `envs.py`, `data.py`, `common.py`.
- `scripts/`: `collect`, `pretrain_jepa`, `imitate`, `train`, `evaluate` (metrics and GIF), `compare`, `run_grid`, `check_determinism`.
- `tests/`: unit tests (CPU, no BlueSky, about 2 seconds).
