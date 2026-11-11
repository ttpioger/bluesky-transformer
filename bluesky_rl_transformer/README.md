# bluesky_rl_transformer: the reference MAPPO method with a Transformer over lidar rays

Created 2026-11-10 23:24. Updated 2026-11-10 23:45, 2026-11-11 00:00, 2026-11-11 00:15, 2026-11-11 00:23, 2026-11-11 00:30, 2026-11-11 00:53. Status of all results from this folder: EXPLORATORY. No training run is done yet.

## Question

Does the method of avbiswas/navigation-mappo-rl (commit 309ac1d) learn the BlueSky competition task when a Transformer over lidar ray tokens replaces its 1D CNN, with a reward that makes no-fly zones and sector exits expensive?

## Method

| Part | Reference | This folder | Mark |
|---|---|---|---|
| Input | lidar, 3 channels (obstacle, boundary, agent), 4 stacked frames, plus agent state | the same layout: 120 rays, 3 channels (no-fly zone, sector boundary, aircraft), 4 frames, plus own state | matched (layout) |
| Lidar source | ray casting on the exact shapes of its 2D simulator; per ray only the nearest hit of any type, in one channel; own goal hit gives 0 in all channels; value max_range - t | ray casting on the exact BlueSky geometry: zone polygons, full sector polygon, other aircraft (5 NM discs), own goal (capture disc); per ray only the nearest hit, in one channel; goal hit gives 0; value (max_range - t) / max_range (`rl_tf/lidar.py`) | matched (rule); the objects differ because the task differs |
| Encoder | Conv1d(12 to 64, k 4, s 2), Conv1d(64 to 32, k 4, s 2), state MLP (128, 128), dense 256 | **Transformer**: 30 ray tokens (4 adjacent rays each, all frames and channels) plus 1 state token, 2 self-attention layers, d_model 64, 4 heads; state MLP (128, 128), dense 256 | changed on purpose |
| Encoder use | shared by actor and critic, one Adam, one gradient clip 0.5 | the same | matched |
| Actor | MLP (64, 64, tanh), tanh mean, default init, state-independent log_std (init 0, clamp [-20, 2]), actions clipped to [-1, 1] | the same | matched |
| Critic | one multi-head attention layer (64, 4 heads) over the agent encodings, no mask, mean over agents, one value per environment | the same | matched |
| Finished aircraft | black_death: terminal observation once, then zero observation, reward 0, rows stay in the batch and the loss | the same | matched |
| Frame stack | 4 frames; older frames are zeros at reset (supersuit frame_stack_v1) | the same | matched |
| Episode end | done only when the environment ends (all agents finished or time limit); no bootstrap at the time limit | the same | matched |
| Rollout | env.reset() at the start of every rollout of 256 steps; flights still running are cut | env.reset() at the start of every rollout of **512** steps (longer than the 300-step maximum flight) | not matched (changed on purpose) |
| PPO | lr 5e-4, 2 epochs, minibatch 128 agent rows, 256 steps times 8 environments, gamma 0.9, lambda 0.95, clip 0.2, vf 0.5, entropy 0, no advantage normalization | the same, except **512 steps** times 8 environments and **gamma 0.99** | not matched (rollout length and gamma changed on purpose) |
| Reward | dense progress, -0.05 per step, +10 goal, -10 collision | see below; only negative terms, no waypoint reward, no shaping term by default | not matched (changed on purpose) |

The only intended differences are the encoder and the reward. The lidar objects and the observation differ because the task differs.

No number from this folder can be compared with a number of the reference method.

Lidar details. Frame: own aircraft at the origin, x = nose, y = starboard (flat-earth, 111.32 km per degree as `core/rendering.py`). Range 100 km. Ray value = 1 - distance / 100 km (0 = no hit or goal hit, 1 = contact). Inside a disc or a zone polygon, a ray gives the distance to its exit point (reference rule for discs). `FlightEnv` adds the geometry to each observation (key `lidar_geometry`). The policy thus needs the zone polygons, the sector polygon, the other aircraft positions and its goal from the simulator, not only the observation Dict. The own state per frame is: airspeed, inside_sector, cos_drift, sin_drift, waypoint_distance, previous turn action, previous speed action.

n_params: policy 306 757 (encoder 215 808, actor head 20 740, critic head 70 209).

Inside-zone flag (2026-11-11, `--zone-flag 1`, default): the own state of each frame also has a flag "inside a no-fly zone" (even-odd rule on the zone polygons). With the flag, n_params: policy 307 269 (encoder 216 320). `--zone-flag 0` gives the state of before. A checkpoint without the key `zone_flag` is read without the flag. Check: the flag is equal to the environment state on 1800 of 1800 steps (6 scenarios, random actions, 63 steps inside a zone).

Waypoint input: the first observation holds the waypoint (cos_drift, sin_drift, waypoint_distance in units of 170 km). Check on 6 scenarios: the values agree with the simulator geometry.

Weights & Biases (2026-11-11, package `wandb` 0.31.0 in `../.venv`): `--wandb online` (or `offline`) and `--wandb-project` on `scripts.train` and `scripts.compare`. Training sends the metrics of each update (`train/...`) and the final flights without noise. `scripts.compare` sends the track figure, the training curves, both GIFs, the per-scenario table and the held-out means to the same run (run id = run name). Default: off.

## Reward

| Term | Value | Unit |
|---|---|---|
| time | -5 | per second flown |
| intrusion | -10 | per second |
| no-fly zone | -10 (was -50 until 2026-11-11) | per second |
| outside the sector | -50 | per second |
| zone entry | -100 (was -500 until 2026-11-11) | once per entry |
| sector exit | -500 | once per exit |
| drift | -0.01 | per radian, per step |
| turn change | 0 (flag `--turn-change-penalty`, give a negative value) | per unit of change of the turn action |
| goal distance | -0.05 (new, 2026-11-11) | per second flown and per km to the goal |
| heading | 0 (flag `--heading-penalty`) | per second flown, times (1 - cos psi); psi = angle between the heading and the bearing to the goal |

Each value is a flag. Ten seconds in a zone cost -200 (entry included); a detour of 100 seconds costs -500 from the time term. At 100 km from the goal, the goal-distance term costs -5 per second, the same as the time term.

Reason for the 2026-11-11 change: with -50 per second and -500 per entry, the policy of `sa_tf_neg_s0` (2 000 000 steps) flew a circle at its start point (50 held-out scenarios: goal reached 0.0, mean turn 43.3 degrees per step, return -20 905; fly straight: -9 405). The new values are not tested in a training run. The episode does not end at a zone entry or a sector exit ("shot down"), for two reasons. The multi-agent environment cannot remove an aircraft without a code change. An early end also stops the time penalty, which can reward a crash. A flight to the time limit (3000 s) costs -15 000 from the time term alone.

Preset `--reward-preset friend` (user, 2026-11-11, the reward of a friend of the user): per second, -100 per aircraft within 5 NM, -10 in a no-fly zone, -10 outside the sector, -(1 - cos psi), -0.1 time; all other terms 0. A weight flag that you give overrides the preset value. Not matched with the friend's setup: the decision step here is 10 simulated seconds (the friend: 5); the per-second rates are the same. The friend's method, network and learning settings are not known here.

Preset `--reward-preset los` (user, 2026-11-11): loss of separation (another aircraft within 5 NM) -0.5 per second and -3 per event (new term `--intrusion-event-penalty`), no-fly zone -0.3 per second and -2 per entry, outside the sector -0.2 per second, time -0.015 per second; all other terms 0.

Hard constraint (user, 2026-11-11): the reward has only negative terms. The waypoint reward (+1) is removed. `check_reward` in `rl_tf/envs.py` stops the run if a weight is positive. Check on 2026-11-11 (8 flights, random actions, 2175 steps): the largest step reward is -25.0.

Learning uses the reward times 0.0002 (`--reward-scale`). The potential-based term is off by default (`--shaping off`). `--shaping potential` adds it again (potential = time penalty times the estimated seconds to the waypoint at 150 m/s). That term is positive during progress to the waypoint, so it breaks the constraint. `ep_return` in all outputs is the unscaled reward without the potential term.

Without the potential term, the drift term is the only dense signal for the direction to the waypoint. At -0.01 per radian it is 5000 times smaller than the time term (-50 per step). `--drift-penalty -10` is the first value to try (not tested).

## Setup

- Environments: `CompetitionEnv-v0` (single agent, 10 scripted aircraft) and `CompetitionZooEnv` (multi agent, default 2 aircraft). The environment code is not changed.
- Scenarios: random (`--reset-seed -1`, default). Final flights without noise use 10 held-out scenario seeds (900000000 and up). `--reset-seed N` gives one fixed scenario (overfit test).
- Run all commands from this folder with `PYTHONPATH=..` and the interpreter `../.venv/bin/python`.

Tests (2 seconds, no BlueSky):

    PYTHONPATH=.. ../.venv/bin/python -m tests.run_tests

One training run:

    PYTHONPATH=.. ~/bin/notify_run.sh sa_tf_s0 ../.venv/bin/python -m scripts.train --name sa_tf_s0 --setting single --seed 0 --steps 400000
    PYTHONPATH=.. ~/bin/notify_run.sh ma_tf_s0 ../.venv/bin/python -m scripts.train --name ma_tf_s0 --setting multi --n-agents 2 --seed 0 --steps 400000

Overnight plan (2 seeds, single agent then multi agent with 4 aircraft; extra training flags go after `--train-args`, as the last option). After each run, `run_seeds` writes `gifs/NAME.gif` (held-out scenario 900000000, lidar rays drawn) and `results/NAME_eval.json`; `--gif-episodes 0` skips this step:

    PYTHONPATH=.. ~/bin/notify_run.sh rl_tf_s01_ma4 ../.venv/bin/python -m scripts.run_seeds --settings single multi --seeds 0 1 --n-agents 4 --steps 400000

Evaluation with metrics, turn statistics and a GIF (no training). The GIF shows the lidar rays as in navigation-mappo-rl: a ray ends at its nearest hit (red: no-fly zone, green: sector boundary, orange: other aircraft, white: own goal) or at the 100 km range (grey). `--no-rays` gives the plain frame:

    PYTHONPATH=.. ../.venv/bin/python -m scripts.evaluate --model models/sa_tf_s0.pt --episodes 3 --reset-seed 900000000 --gif gifs/sa_tf_s0.gif
    PYTHONPATH=.. ../.venv/bin/python -m scripts.evaluate --policy controller --setting single --gif gifs/controller_sa.gif

Comparison with the fly-straight baseline (single agent, no training). It writes `figures/NAME_tracks.png` (tracks on the scenario seeds 2 to 5, fly straight and the policy), `figures/NAME_curves.png` (training curves from the summary file), `gifs/NAME_sa.gif`, `gifs/NAME_vs_straight.gif` and `results/NAME_compare.json` (50 held-out scenarios, mean and std). "s in violation" = intrusion_time + time_in_restricted_area + time_outside_sector:

    PYTHONPATH=.. ../.venv/bin/python -m scripts.compare --model models/sa_tf_s0.pt

Outputs: `results/NAME_summary.json` (pass flags, first and last window, final flights without noise with `mean_abs_turn_deg` and `turn_sign_flip_rate`, curve, all episodes), `models/NAME.pt` (saved every 20 updates and at the end), `logs/NAME.log`.

## Checks done (2026-11-10)

- Unit tests: 9 passed (lidar geometry, frame stack, masks, GAE, one learning step), before the polygon lidar.
- Lidar against the environment metrics (single agent, seed 1, zero action, old circle lidar): the aircraft channel shows contact (0 to 0.31 km) on the 4 steps with intrusion time, and 0.40 km or more on all other steps.
- Same seed twice (multi agent, 256 steps, random scenarios): bit-identical weights.
- Speed with 6 workers, 2304 steps: 0.0141 s/step single agent, 0.0083 s/step multi agent (before the 2026-11-10 alignment).
- 2026-11-10, after the alignment to the reference: 9 unit tests passed. `compute_gae` gives the same advantages and returns as the reference `RolloutBuffer` (max difference 0.0, random data, T 50, 6 rows). Smoke run, 8 workers, 4096 steps: 0.0051 s/step single agent, 0.0093 s/step multi agent.

- 2026-11-11, 10 aircraft (multi agent), 8 workers, 8192 steps, CPU: 0.0635 s/step (2 aircraft: 0.0072 s/step). Environment step alone: 0.042 s with 10 aircraft, 0.018 s with 2. The rest is the update: each minibatch row also encodes all N aircraft of its environment for the critic, so the update cost grows with N squared.
- 2026-11-11, 4 aircraft, 8 workers, 8192 steps, CPU: 0.0160 s/step, 0 non-finite actions, 0 `txt2hdg` errors.
- 2026-11-11, GPU: torch 2.12.0+cu130 stops with `CUBLAS_STATUS_ARCH_MISMATCH` on the GTX 1060. Training uses the CPU.
- 2026-11-11, polygon lidar (one hit per ray): 10 unit tests passed. Against the metrics (single agent, zero action, seeds 1 to 3): on the 4 intrusion steps of seed 1, the nearest aircraft ray is 0.0 to 0.3 km; on the first zone step of seed 1 and seed 2, the nearest zone ray is 0.0 and 0.8 km. In the GIF frames, the zone hits lie on the drawn zone polygon edges.
- 2026-11-11, polygon lidar, 8 workers, 8192 steps, CPU: 0.00513 s/step single agent, 0.01709 s/step with 4 aircraft. 0 non-finite actions, 0 `txt2hdg` errors.

- 2026-11-11, `scripts.compare`, fly straight on the scenario seeds 2, 3, 4, 5: 426, 10, 180, 145 s in violation. These are the values of the example figure that the user gave (2026-11-11).
- 2026-11-11, new reward defaults (no waypoint reward, no shaping), `--drift-penalty -10 --turn-change-penalty -5`, 8 workers, 16 384 steps, CPU: 0.00497 s/step single agent. 11 unit tests passed.

## Limits

- No training run is done. It is not known if the policy learns.
- Inside a zone, the zone rays give the distance to the zone edge, as outside. The lidar alone does not tell inside from outside (as in the reference, where an agent cannot enter an obstacle). The observation has no inside-zone flag; only the reward tells the policy.
- A turn command near heading 0 can fail: `core/actions.py` writes the heading as text, a value such as `3.5e-06` is rejected by BlueSky (`txt2hdg needs either a float...`), and the aircraft keeps its last heading command for that step. This is in the environment code, which this folder does not change. `n_nonfinite_actions` in the summary counts NaN actions from the policy, a different cause.
- Reset at every rollout (reference rule): a BlueSky flight lasts up to 300 steps (3000 s, 10 s per action). With the reference rollout of 256 steps, all 8 flights were cut in a single-agent smoke run. The rollout is now 512 steps, so the first flight of each environment always finishes. A second flight that starts late in a rollout can still be cut. Cut flights count in `n_cut_agent_episodes`, not in the training metrics.
- gamma is 0.99 (reference 0.9). The horizon is about 100 steps (1000 s), so a detour around a zone is inside the horizon. The potential shaping uses the same gamma. With 512-step rollouts, a run of the same length has half as many updates, but the same number of gradient steps.
- The reward scale 0.0002 is a choice without a test. Early flights that leave the sector give returns near -100 000; a large value loss can slow the shared encoder.
- A higher drift penalty is not the direct fix for a turn command that changes sign at each step. The drift term gives the same cost to a steady detour and to an oscillation with the same mean heading error. It also gives a cost to each detour around a zone or an aircraft. The turn-change term is the direct fix.
- The turn-change term is off by default. The policies of `bluesky_jepa_set` changed the turn sign at almost every step; the same can occur here. `--turn-change-penalty -5` is the first value to try.
- The lidar needs the simulator geometry. A harness that gives only the observation Dict cannot run this policy.

## Next step

Run one single-agent training run of 2 000 000 steps. Then run `scripts.compare` on the model and read `results/NAME_compare.json`.

Source files read on 2026-11-10: `bluesky_gym/envs/competition_env.py`, `bluesky_zoo/competition/competition.py`, `core/observations.py`, `core/actions.py`; reference repo `networks/actor_critic_network.py`, `rl/mappo.py`, `train_mappo.py`, `nav/environment.py`.
