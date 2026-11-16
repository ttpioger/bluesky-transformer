# Time vs intrusion — sa_tf_tok3_s0

Created 2026-11-14 10:30. Status: **EXPLORATORY** (one training seed, seed 0). n_params 327 237.

**Provisional verdict: (3) Neither.** The verdict is provisional. The script
`time_vs_intrusion.py` has not run yet. Its confidence intervals and the paired test can change the verdict (see "What would change the verdict").

## 0. Scope and corrections to the task

Facts:

- Target model: `sa_tf_tok3_s0` (the user's message: "the tok3 previous flight model, not the jepa").
- The numbers in the task ("96 s intrusion in the first 20 flights, 20.9 s in the last 20") come from `sa_tf_tok3_s0`. Its first window has 95.75 s and its last window has 20.9 s (`results/sa_tf_tok3_s0_summary.json`). They do not come from `ma10_tf_los_speed_s0` (116.0 → 62.0 s) or from `sa_tf_los_speed_s0` (133.8 → 119.8 s).
- In these windows, intrusion time and flight time both decrease (flight time 2681 → 1038 s). The windows do not show "intrusion up while flight time down".
- **Saved training checkpoints do not exist.** `scripts/train.py:336-338` writes the same file `models/NAME.pt` every 20 updates and overwrites it. The W&B upload (`wandb_upload`, `policy="now"`) also replaces the file. I did not check the W&B version history in the UI.
- `models/sa_tf_tok3_s0.pt` is not on the laptop. It is on the remote desktop (the competition file names it).
- All three runs use the same reward: preset `los` plus `heading_penalty -0.05`, `reward_scale 0.05`, `gamma 0.99`.

## 1. Where the files are

| What | Where |
|---|---|
| Reward weights | `rl_tf/envs.py:61-82` (`REWARD_DEFAULTS`, `REWARD_PRESETS["los"]`); the weights of a run are in `args` of its summary |
| Reward terms added by this repo | `rl_tf/envs.py:183-189` (`FlightEnv.step`) |
| Reward terms of the environment | `../bluesky_gym/envs/competition_env.py:281-301` (`_get_reward`) |
| Metrics (intrusion, area, sector) | `../bluesky_gym/envs/competition_env.py:321-367` (`_update_metrics`, 1 s sampling) |
| Learning reward (scale, shaping) | `scripts/train.py:243`; GAE `rl_tf/common.py:21-35` |
| Per-flight training log | `results/sa_tf_tok3_s0_summary.json`, key `episodes` (16 398 flights; metrics, `ep_return`, `env_steps`) |
| Per-flight end evaluation | `results/sa_tf_tok3_s0_competition_sa.json`, key `rows` (1000 flights, seed-42 stream, no action noise) |
| Text log | `sa_tf_tok3_s0.log` (per update) |
| Checkpoint | `models/sa_tf_tok3_s0.pt` (final only, remote desktop) |
| Other policy on the same stream | `results/sa_tf_info_ff_cls_big_s0_competition_sa.json` (same reward weights) |

## 2. Question

Does the policy buy flight time with intrusions? Three answers are possible:
(1) mispriced trade-off, (2) loophole in the measurement or the penalty, (3) neither.

## 3. Method

- Reward audit and exchange rate: from the code only (sections 4 and 5).
- Trade-off evidence, decomposition and counterfactual: the script `time_vs_intrusion.py`. It reads the existing result files. It runs no simulation.
- Decomposition: each reward term is weight × final flight metric. The rest, `r_other = ep_return − Σ terms`, is the heading term. This is exact because the drift, goal-distance and turn-change weights are 0. The script checks that `r_other ≤ 0` for each flight.
- Counterfactual: `sa_tf_info_ff_cls_big_s0` flew the same 1000 scenarios. The scenario generator uses `env.np_random` only at reset (`competition_env.py:174`). Thus the scenario of episode *i* does not depend on the policy. Pairing is by `episode_index`. This is a code-reading fact; no run has checked it.

## 4. Reward audit (facts from code)

| Term | Weight | When it applies |
|---|---|---|
| time | −0.015 | per simulated second flown |
| intrusion time | −0.5 | per second **and per other aircraft** within 5 NM |
| intrusion event | −3.0 | once per start of an intrusion with one aircraft |
| restricted-area time | −0.3 | per second inside ≥ 1 area |
| restricted-area entry | −2.0 | once per entry |
| outside-sector time | −0.2 | per second |
| sector-exit event | 0 | — |
| heading | −0.05 | per second × (1 − cos ψ), ψ = heading vs bearing to goal |
| drift, goal distance, turn change | 0 | — |
| waypoint reached | 0 (removed) | — |

Checks asked in the task:

- **Per event instead of per second?** No. Intrusion has both a per-second term and a per-event term. The per-second term counts each intruding aircraft (`intrusion_time += len(now_intr) * dt`, `competition_env.py:342`).
- **Caps?** None. No term has a maximum.
- **Distance thresholds?** The intrusion threshold is 5 NM, a hard step. Nothing is paid at 5.01 NM. The competition metric uses the same 5 NM threshold, so a flight just outside 5 NM is legal under the metric too.
- **Metric sampling gaps?** None. One policy step is 10 s; the metric is sampled each 1 s (`competition_env.py:248-251`).
- **Reward clipping or normalisation?** None. The learning reward is `ep reward × 0.05` (`train.py:243`). There is no reward normalisation, no advantage normalisation (`train.py:263`) and no value clipping. The scale multiplies every term, so it does not change the exchange rate.
- **PPO clips:** ratio clip 0.2 and gradient-norm clip 0.5. They limit the size of one update. They do not change the optimum.
- **Intrusions present at reset** count as one event and are paid at the first step (`envs.py:160`, `competition_env.py:233`). The agent cannot avoid them. This adds cost, not a loophole. The tok3 files have no `start_intrusions` column (that column exists since 2026-11-13).
- **Time limit:** a flight ends at 3000 s with no bootstrap (`common.py:22-24`). The goal is reached in 99 % of the end-evaluation flights, so this has no effect on the intrusion question.
- **Training log bias:** a flight still in progress at the end of a 512-step rollout is reset and not logged (`envs.py:438-440`). 3904 flights were cut and 16 398 were logged. Long flights are under-represented in the training log.

## 5. Exchange rate (arithmetic from the weights)

- 1 s of intrusion = 0.5 / 0.015 = **33.3 s** of flight time.
- 1 intrusion event = 3 / 0.015 = **200 s** of flight time.
- One mean intrusion in the competition flights is about 36 s long (15.99 s / 0.44 events, ratio of means). Its cost is 0.5 × 36 + 3 ≈ 21 reward units ≈ **1400 s** of flight time. The mean flight is 1010 s.
- Heading term during a detour: the cost per flown second rises from 0.015 (ψ = 0°) to 0.022 (30°), 0.040 (60°) and 0.065 (90°).
- Discounting (γ = 0.99 per 10 s step): a detour pays its extra seconds at the end of the flight, k steps later. The discount makes these seconds cheaper, so the rate rises to 33.3 / 0.99^k (55 at k = 50 steps). Discounting makes intrusions relatively more expensive.
- Effective rate after scaling, clipping and normalisation: the same as above (nothing changes it).

Interpretation: a detour must cost more than about 1400 s to make one mean intrusion the cheaper choice. The reward does not make time worth more than safety.

## 6. Results already in the files (facts, before the script)

Training flights of `sa_tf_tok3_s0`, 20 bins of 100 000 environment steps (means, no CI yet):

| env steps (M) | flights | flight time (s) | intrusion time (s) | intrusion events |
|---|---|---|---|---|
| 0.0-0.1 | 363 | 2014 | 102.8 | 1.39 |
| 0.1-0.2 | 671 | 1257 | 55.4 | 1.00 |
| 0.3-0.4 | 821 | 1066 | 31.4 | 0.64 |
| 0.6-0.7 | 865 | 1016 | 23.9 | 0.51 |
| 0.9-1.0 | 862 | 1021 | 21.6 | 0.49 |
| 1.2-1.3 | 855 | 1025 | 16.2 | 0.37 |
| 1.5-1.6 | 839 | 1055 | 15.8 | 0.38 |
| 1.8-1.9 | 888 | 1012 | 13.6 | 0.35 |
| 1.9-2.0 | 868 | 1017 | 17.9 | 0.46 |

(Selected rows. The script writes all 20 bins with CIs to `numbers.md`.)

- 0 to 0.4 M steps: flight time and intrusion time both fall.
- After 0.4 M steps: flight time stays at about 1000-1060 s. Intrusion time continues to fall, from about 31 s to 14-18 s.
- 3 of 19 neighbour-bin pairs show "flight time down, intrusion time up": 0.4→0.5 M (−42 s, +0.8 s), 0.7→0.8 M (−3 s, +5.1 s), 1.3→1.4 M (−26 s, +3.2 s). The script tests whether their CIs separate.

End evaluation, 1000 flights of the seed-42 stream, no action noise:

| policy | goal | flight time (s) | intrusion time (s) | intrusion events | restricted-area time (s) | outside-sector time (s) | return (same reward) |
|---|---|---|---|---|---|---|---|
| sa_tf_tok3_s0 | 0.99 | 1009.5 | 15.99 | 0.44 | 7.73 | 11.58 | −40.83 |
| sa_tf_info_ff_cls_big_s0 | 1.00 | 1000.6 | 13.83 | 0.29 | 19.35 | 6.92 | −41.39 |
| fly straight (means only) | 1.00 | 852.0 | 96.32 | 1.13 | 129.93 | 0.00 | ≤ −104.9 * |

\* Known terms under the tok3 reward. The heading term is unknown (no rows), and it is ≤ 0. So −104.9 is an upper bound.

- tok3, within its 1000 flights: flights with an intrusion are longer (mean 1066 s) than flights without one (982 s). Spearman(flight time, intrusion time) = +0.13 (point estimate; the script adds the CI).
- The big model has fewer intrusions **and** shorter flights than tok3. Its return is 0.56 lower. By hand arithmetic, most of this loss comes from more restricted-area time (+11.6 s × 0.3 ≈ −3.5), not from flight time. The script gives the exact split.

Where the observation "intrusion up while flight time down" occurs:

- Not in the first/last training windows.
- Not as a trend in the training bins after 0.4 M steps.
- Not within the end-evaluation flights (the correlation is positive).
- Only between fly-straight and the trained policies (a different policy, the zero action), and in 3 small neighbour-bin steps of training.
- Interpretation: the per-update W&B curves (`train/flight_time`, `train/intrusion_time`) are means of about 10 flights per update. Their noise can show an apparent opposite movement. I did not check the W&B curves.

## 7. Verdict (interpretation)

**Provisional: (3) Neither.**

Evidence for (3):

1. The reward prices one mean intrusion at about 1400 s of flight time. This is more than one full flight (section 5).
2. The code has no loophole of the listed kinds (section 4).
3. In the end evaluation, flights with intrusions take longer, not shorter.
4. A policy with fewer intrusions (big model) is also faster on the same scenarios.
5. Intrusion still decreases late in training (0.35-0.46 events per flight at 1.8-2.0 M steps). This agrees with incomplete learning. Traffic density can cause the remaining intrusions; the positive correlation (longer flights in denser traffic) agrees with this, but it does not prove it.

What the data does not show:

- It does not show the cause of the remaining 0.44 events per flight. Start intrusions (unavoidable) are one possible part. The tok3 files cannot separate them.
- It does not give a paired, same-policy-family "more cautious" counterfactual. The big model differs in size and inputs.
- One seed only. No result here is REPORTABLE.

## 8. What would change the verdict

- To (1): the paired test gives a negative Spearman(Δ flight time, Δ intrusion time) with its CI below 0, **and** the policy with fewer intrusions loses return mainly through `r_time` or `r_other`.
- To (1): a checkpoint trend (same scenarios, no noise) shows intrusion rising while flight time falls. This needs saved checkpoints (they do not exist; see section 10).
- To (2): `r_other > 0` for some flights (the decomposition would then miss a term), or flights where intrusion time > 0 but the return shows no intrusion cost.
- To "inconclusive": the 3 neighbour-bin pairs have separate CIs and the paired test has a CI that includes 0.

## 9. Smallest fix (proposal only; not applied)

The provisional verdict is (3), so the reward needs no reweighting. For the remaining intrusions:

- Train with a constraint: the PPO-Lagrangian code exists (`--lagrangian 'intrusion_events_no_start:0.3,...'`, run `sa_tf_tok3_lag_s0`, not run).
- Use `intrusion_events_no_start` as the constraint metric. Then the agent is not charged for start intrusions it cannot avoid.
- If (1) is confirmed: the same Lagrangian run is the fix. A change of weights is not needed first, because the rate is already high.

## 10. Counterfactual policy (step 4)

Facts: no cautious version of tok3 exists. The big model is the only other policy on the same stream with the same reward. The script uses it (`--alt sa_tf_info_ff_cls_big_s0`).

Cheap ways to get a cautious tok3 policy (each needs your approval):

| option | compute | over 30 min? |
|---|---|---|
| A. `sa_tf_tok3_lag_s0` (Lagrangian, code exists) + competition eval | 1 h 58 min training (tok3 wall time on cuda) + 41 min eval (2470 s for tok3) | yes |
| B. tok3 + a safety filter at evaluation only (new eval code needed) on 200 scenarios | about 8 min (scaled from 2470 s for 1000 flights; not measured) | no, but it needs new code |
| C. Re-train tok3 with saved checkpoints every 20 updates (needs a change in `train.py`) | 1 h 58 min + 41 min per checkpoint eval | yes |

## 11. W&B

- `--wandb online` (or `offline`) on the script writes one W&B run `sa_tf_tok3_s0_time_vs_intrusion` (group `sa_tf_tok3_s0`, job type `analysis`, same project `bluesky-rl-transformer`). It does not write into the training run.
- Content: trend curves per training bin (`trend/*` against `trend/env_steps`, means with CI, Spearman with CI), per-flight tables and scatter plots (`flights/*`, `scatter/*`), summary values (`rate/*`, `corr/*`, `paired/*`, `decomp/*`), the figures, and an artifact with `flights.csv`, `stats.json` and `numbers.md`.
- With `--checkpoint-evals STEP=FILE ...` it also writes `ckpt/*` against `ckpt/env_steps`.
- Live metrics during a future training run need a change in `scripts/train.py` (not done, training code is frozen). Proposal: in the `wb.log` block, add the mean of weight × metric per reward term of `new_eps` and `r_other`; add a flag that keeps a copy of each saved checkpoint (`models/NAME_stepXXXXXXX.pt`).

## 12. Next step

Run the script on the remote desktop (command in the terminal answer). Then I read `numbers.md` and `stats.json` and update sections 6-8 of this report.

## Limits

- One training seed. Training flights have action noise; end-evaluation flights have none.
- The training log leaves out 3904 cut flights (long flights are under-represented).
- The scenario pairing is from code reading, not from a run check.
- The fly-straight file has means only; its heading term is unknown.

## Sources (read 2026-11-14)

`results/sa_tf_tok3_s0_summary.json`, `results/sa_tf_tok3_s0_competition_sa.json`,
`results/sa_tf_info_ff_cls_big_s0_summary.json`, `results/sa_tf_info_ff_cls_big_s0_competition_sa.json`,
`results/sa_tf_los_speed_s0_summary.json`, `results/ma10_tf_los_speed_s0_summary.json`,
`results/competition_straight_sa_1000_seed42.json`, `rl_tf/envs.py`, `rl_tf/common.py`, `scripts/train.py`,
`scripts/competition_eval.py`, `../bluesky_gym/envs/competition_env.py`.
