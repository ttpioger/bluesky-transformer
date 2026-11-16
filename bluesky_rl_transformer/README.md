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

Lidar radius and speed channel (2026-11-12). `--max-range-km` is the lidar radius around the own aircraft; a ray sees nothing beyond it. The default is now 50 km (100 km before); the ray value is 1 - distance / radius. `--lidar-speed 1` (default) adds a 4th lidar channel: for a ray whose nearest hit is an aircraft, the closing speed (rate at which the distance between the two aircraft decreases) divided by 350 m/s and clipped to [-1, 1]; positive = the aircraft comes nearer; 0 on all other rays. `FlightEnv` adds the key `aircraft_vel` (velocity of each aircraft minus the own velocity, own frame, m/s) to the geometry. With 4 frames, a patch has 4 rays x 16 values = 64 numbers. n_params: policy 308 293 (encoder 217 344). A checkpoint without the key `lidar_speed` is read without the channel. Check (6 held-out scenarios, straight flight, 3600 aircraft pairs): the closing speed differs from the measured distance change over the 10 s step by 1.06 m/s (median), 7.71 m/s (95 %), 40.73 m/s (max); range of the closing speed -308 to 314 m/s; an aircraft is in the 50 km lidar on 332 of 360 steps. `--max-range-km 100 --lidar-speed 0` gives the lidar of before.

Aircraft tokens (2026-11-12, `--ac-tokens K`, default 0 = baseline). With K > 0 the Transformer gets K more tokens, one per other aircraft whose 5 NM disc is within the lidar radius (the K nearest, nearest first). One token has 6 values in the own frame: x, y, vx, vy (velocity of the aircraft minus the own velocity), distance, closing speed; positions divided by (lidar radius + disc radius), speeds by 350 m/s. A small MLP (6 -> 64 -> 64) plus one learned "aircraft" vector makes the token; there is no position vector, because the aircraft are a set. An empty row is masked in the attention. The tokens are from the newest frame only and go at the end of the state array (K x 6 values, then K flags), so the rest of the training code is the same. With K = 10: n_params policy 312 965 (encoder 222 016), 41 tokens. Checks: with `--ac-tokens 0`, a smoke run (8192 steps, seed 0) gives bit-identical weights to the code of before; values in an empty row do not change the action; a value in a real row does. Not tested in a full training run.

Restricted-area, goal and action-history tokens (2026-11-12, default 0 = off). `--zone-tokens Z`: one token per restricted area (the Z nearest of all areas, no range limit; a scenario has 5) with 11 values in the own frame: nearest edge point (x, y, distance), the port-side end and the starboard-side end of the area as seen from the own aircraft (cos, sin of the bearing, distance to the corner), a flag "the straight line to the goal crosses the area", a flag "inside the area". Lengths are divided by 170 km. The flag also adds one goal token: x, y, distance, cos and sin of the bearing, share of the 3000 s that is left. `--act-tokens H`: one token for each of the last H own actions (turn, speed), with a learned vector for the place in time; rows before the first action are masked. Each token type has its own MLP (values -> 64 -> 64). n_params: 317 957 with Z = 5; 312 901 with H = 4; 327 237 with K = 10, Z = 5, H = 4. Checks: 15 unit tests pass; with all flags at 0, and with `--ac-tokens 10` alone, a 1024-step run gives bit-identical weights to the code before.

Sector-edge tokens, closest point of approach, Fourier features, read-out token (2026-11-12 evening, default 0 = off). `--edge-tokens E`: the circle around the own aircraft is cut into E direction bins (bin 0 on the nose, then clockwise); each bin gives one token with the nearest point of the sector edge in that direction (x, y, distance divided by 170 km, cos and sin of the bearing, flag "outside the sector"), no range limit; a bin without a point is masked. `--ac-cpa 1`: each aircraft token gets 2 more values for straight flight at the velocities of now: time to the closest point (divided by 300 s, at most 2, 0 if the aircraft goes away) and the distance there. `--fourier F`: the aircraft, area, goal and edge MLPs also get sin and cos of each value at the frequencies pi * 2^j, j = 0..F-1. `--cls-token 1`: a learned read-out token at place 0 collects the result. More capacity uses the options that were there: `--d-model 128 --n-layers 3`. Status: one new unit test (`test_edge_cpa_fourier_and_readout_tokens`); NOT run yet, and the bit-identical check at options 0 is NOT done yet (no code runs on the laptop).

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

## JEPA pretraining and curriculum (2026-11-13)

Temporal, action-conditioned JEPA (`rl_tf/jepa.py`). The encoder f gives z_t = f(o_t). A predictor rolls z forward with the actions, and the target of step k is f_ema(o_{t+k}), a slow copy of f with no gradient. Loss: MSE over k = 1..H plus VICReg variance and covariance terms. Only the encoder goes into PPO.

1. `scripts.collect_jepa`: scripted flights only (fly straight, goal controller, goal controller with noise, random turns), scenario seeds 3 000 000 + i, densities 10:5, 10:5, 5:3, 2:1 (other aircraft : restricted areas). Shards `data/jepa_NAME_wK.npz` (not in git).
2. `scripts.pretrain_jepa`: JEPA on the shards, then linear probes on validation flights (intrusion within 60 s, area entry within 60 s: AUC; goal distance: R²) for the raw input, the start weights, the JEPA encoder and each `--probe-model`. Writes `models/jepa_NAME.pt` and `results/jepa_NAME_summary.json`. Stop rule: spread < 0.05 = collapse, exit code 2.
3. `scripts.train --init-encoder models/jepa_NAME.pt` (same token and network flags), optional `--freeze-encoder-steps N`.

### Map image and masked-token JEPA (2026-11-13, default off)

Each input kind gets one role. The map image shows the fixed map; the lidar gives exact distances near the aircraft; the aircraft tokens give the moving objects with velocity and CPA.

- `--map-px P` (0 = off): an image of P x P pixels, own frame (row 0 = ahead, column 0 = port), square of side 2 x `--map-range-km` (default 150). Channels: share of the pixel inside a restricted area, share outside the sector (2 x 2 samples per pixel), goal (point split over the 4 nearest pixel centres; 0 if far outside). Newest frame only. The image goes at the end of the state, after the time-left share. The time-left share goes into the own-state network (it was in the goal token before).
- `--map-patch p` (default 8): one token per p x p patch, (P / p)^2 tokens, one linear layer plus one learned vector per patch place (as ViT).
- Option 1 token set: `--map-px 32 --map-patch 8 --ac-tokens 10 --ac-cpa 1 --zone-tokens 0 --edge-tokens 0 --act-tokens 0 --cls-token 1`: read-out 1 + own state 1 + lidar 30 + aircraft 10 + map 16 = 58 tokens. The past actions stay in the own state of each of the 4 frames.
- Masked-token JEPA (`scripts.pretrain_jepa --masked-coef 1`): per sample, one block of `--map-block` x `--map-block` map patches and one arc of `--lidar-arc` lidar patches are hidden from the online encoder. A predictor Transformer (`--mpred-dim`, `--mpred-layers`) predicts the outputs of the slow copy at those tokens. `--temporal-coef` (default 1) keeps the temporal target; `--temporal-coef 0` gives the masked target alone.
- Status: 2 new unit tests (`test_map_image_and_tokens`, `test_masked_jepa`). NOT run yet. Bit-identical check with the map off NOT done yet. Pixel-drawing time per step NOT measured.

`scripts.train --curriculum '2:1@0,5:3@300000,10:5@800000'`: from step S on, training scenarios have I other aircraft and O areas. The final flights without noise and all evaluation scripts use 10:5. All new flags at their defaults give the training of before.

No lidar (2026-11-13): `--n-rays 0` removes the lidar tokens and their weights. With the map image and the aircraft tokens, the policy reads own state 1 + aircraft 10 + map 16 (+ read-out 1) = 28 tokens. The aircraft tokens still use `--max-range-km` as their radius. The masked JEPA then hides map blocks only. Limit: near a restricted area or the sector edge, the policy sees distances only at the map resolution (9.4 km per pixel at 32 px over +-150 km). Unit test: `test_no_lidar_map_jepa`. NOT run yet.

### Latent world model and CEM planner (2026-11-13 evening, after V-JEPA 2-AC)

Design page: "JEPA World-Model Planner". Code: `rl_tf/world_model.py`, `scripts/train_world_model.py`, planner flags of `scripts.competition_eval`.

- World model on a **frozen** encoder (the encoder of the policy that the planner flies with). State: token outputs h [T, d], own-state encoding s, empty-row mask. Predictor: Transformer over the T tokens + one own-state token + one action token, residual, last layer at zero (start = "no change"), Markov (sees step t only). Cost heads: cross-attention from the own-state token to all tokens, then an MLP: intrusion / restricted area / outside sector (logits), goal-distance decrease, reward of the step. Loss: rollout of H = 6 steps against the encoder outputs at t+k, one-step loss from the true state, head losses on predicted and true states.
- `scripts.collect_jepa` now records the reward of each step (with the reward flags of the command) and can fly a trained policy: `--controllers policy,... --policy-model FILE` (sampled actions). Without these flags the flights are the same as before; the data has one more column.
- `scripts.train_world_model --encoder-from models/POLICY.pt --data A B`: checks that the data reward equals the reward of the policy run. Validation: per step k the rollout error against "no change", AUC of the predicted intrusion and area probability, R^2 of reward and goal step. Writes `models/wm_NAME.pt`, `results/wm_NAME_summary.json`.
- `scripts.competition_eval --model models/POLICY.pt --world-model models/wm_NAME.pt --plan-level 1|2`: level 1 = safety filter (search only when the predicted risk of the policy rollout over 60 s is >= `--plan-threshold`, default 0.5), level 2 = search at every step. Search: CEM, 64 sequences x 3 rounds x 6 steps around the policy rollout; score = predicted reward (PPO units: reward scale, gamma) + gamma^6 x policy critic at the end; first command flown, plan again each step. Check: `--plan-level 1 --plan-threshold 2` never searches and must give the metrics of no planner. Output `results/POLICY_planL_competition_sa.json` with `plan_stats` (share of decisions with a search, ms per decision).
- Known limits: tokens are matched over time by place (aircraft rows are sorted by distance; no aircraft id in the data). Multi-agent: each aircraft plans alone, the critic sees one aircraft.
- Status: 4 new unit tests (`test_world_model_start_z_and_loss`, `test_world_model_step_labels`, `test_planner_filter_keeps_policy_and_search_beats_prior`, `test_world_model_save_and_load`). NOT run yet (no runs on the laptop). Only a syntax check ran.

#### Multi agent (2026-11-13 late)

- Recording: `scripts.collect_jepa --setting multi --n-agents N` records all N aircraft (aircraft j of flight i: controller (i * N + j) % len). Rows of one aircraft are contiguous. New columns in both settings: `agent`, `nb_ids [K]` (aircraft index in each aircraft-token row, -1 = empty or scripted). The geometry of each observation now has `aircraft_ids` (callsigns, same order as `aircraft`). `--densities` in multi agent: first number 0 (default 0:5,0:5,0:3,0:1).
- World model: each aircraft-token row can get the command (turn, speed, known) of its aircraft (MLP, last layer at 0). `--nb-drop 0.3` hides them in 30 % of the training windows (the model also learns "unknown"). New head 5: value = discounted return to go of the recorded flights in PPO units (gamma and reward scale of the policy file).
- Planner: all aircraft roll out together; each row follows the plan of its aircraft (start: policy rollout). `--plan-order independent` (default: all search at once) or `sequential` (callsign order; a later aircraft sees the chosen plans of the earlier ones). `--plan-value auto|critic|head` (auto: critic for a single-agent policy, value head for a multi-agent policy, whose central critic needs all aircraft).
- Training chain: `scripts.train --init-policy FILE` loads encoder, actor and log_std (critic new; a single-agent policy can start a multi-agent run). `--curriculum` with `--setting multi` now stops with an error unless I = 0 (that environment has no scripted aircraft); change the number of learning aircraft with a chain of runs (`--n-agents 2 -> 4 -> 10`, each with `--init-policy` of the run before).
- Limits: the value head learns the return of the recorded controllers, not of the planner. Neighbour commands follow the aircraft that held row j at t (rows are re-sorted by distance at each step).
- Status: 4 more unit tests (`test_returns_and_neighbour_commands`, `test_aircraft_row_ids_follow_token_order`, `test_planner_multi_agent_neighbours`, `test_load_actor_keeps_new_critic`). NOT run yet. Only a syntax check ran.

## PPO-Lagrangian (2026-11-13 late, default off)

`scripts.train --lagrangian 'intrusion_events_no_start:0.3,time_in_restricted_area:5,time_outside_sector:6'`: each flight metric gets a limit (per flight, competition units), a multiplier lam >= 0 and a cost critic (one attention critic per constraint on the encoder features, no gradient into the encoder, own Adam). The cost of a step is the increase of the metric (computed in the environment workers) times `--reward-scale`, so lam has the units of a reward weight. Advantage: (A_r - sum lam A_c) / (1 + sum lam); lam = 0 gives the advantage of before. Dual step after each update on the flights that ended in the rollout: lam <- clip(lam + `--lambda-lr` x clip(cost / limit - 1, -1, 1), 0, `--lambda-max`). The fixed reward weights stay (the Lagrangian adds to them). lam and the mean costs go into the curve, the log line and W&B; the checkpoint holds lam and the cost critics (`lagrangian` key; `load_policy` ignores it). Single agent can use the `*_no_start` metrics (start intrusions cannot be avoided). Unit tests: `test_lagrangian_parse_step_and_advantage`, `test_policy_act_features_do_not_change_actions`. NOT run yet; bit-identical check with the flag off NOT done yet.

## Start intrusions (single agent, 2026-11-13; not official metrics)

The official metrics count an intrusion at reset when another aircraft is already within 5 NM. No policy can avoid it. `FlightEnv` (single agent) adds three keys to `info` and to the flight rows; the official metrics do not change:
- `start_intrusions`: aircraft within 5 NM at the reset sample;
- `intrusion_events_no_start`, `intrusion_time_no_start`: the intrusion metrics without these start intrusions. A start intrusion stays left out until that aircraft goes out of 5 NM; a later new intrusion with it counts. So `intrusion_events_no_start = intrusion_events - start_intrusions`.

`scripts.competition_eval` shows the three rows in its table, writes `start_intrusion` (flights that start in intrusion, and the means over the other flights) to `results/NAME_competition_sa.json`, and computes the fly-straight cache again if it has no such keys. Unit test: `test_start_intrusions_left_out` (fake environment). NOT run yet.

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
