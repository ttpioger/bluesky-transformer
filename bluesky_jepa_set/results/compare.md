# Overfit comparison (status: EXPLORATORY)

One fixed scenario per setting. Mean ± std over seeds. Windows: first and last 20 agent-episodes of training (with exploration noise).
An overfit pass shows only that the learning loop works. This table does not show that a cell is better or worse in general.

| setting | cell | encoder | jepa | seeds | n_params actor | pass | waypoint (last) | intrusion ev. | restricted ev. | sector-exit ev. | flight time s (last) | return (first) | return (last) | steps to criterion | greedy waypoint | greedy flight time s | s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | R | cnn | off | [0, 1, 2] | 86941 | 0/3 | 1.00 ± 0.00 | 0.28 ± 0.14 | 1.05 ± 0.74 | 0.00 ± 0.00 | 995.42 ± 38.12 | -15401.27 ± 606.39 | -5009.65 ± 189.13 | n/a (0/3 reached) | 1.00 ± 0.00 | 966.67 ± 15.43 | 0.02 ± 0.00 |
| single | A | set_tf | off | [0, 1, 2] | 87044 | 0/3 | 1.00 ± 0.00 | 0.40 ± 0.29 | 1.52 ± 0.31 | 0.00 ± 0.00 | 973.45 ± 1.41 | -12818.69 ± 236.46 | -4903.18 ± 26.29 | n/a (0/3 reached) | 1.00 ± 0.00 | 955.33 ± 4.92 | 0.03 ± 0.00 |
| single | B | cnn | on | [0, 1, 2] | 86941 | 0/3 | 0.97 ± 0.02 | 1.32 ± 0.54 | 0.02 ± 0.02 | 0.33 ± 0.47 | 1574.02 ± 446.81 | -16439.86 ± 207.28 | -9025.82 ± 2395.17 | n/a (0/3 reached) | 1.00 ± 0.00 | 1299.33 ± 398.38 | 0.03 ± 0.00 |
| single | C | set_tf | on | [0, 1, 2] | 87044 | 0/3 | 1.00 ± 0.00 | 0.37 ± 0.34 | 0.82 ± 0.62 | 0.00 ± 0.00 | 994.70 ± 22.17 | -15404.12 ± 260.00 | -5042.25 ± 135.62 | 22081.00 ± 0.00 (1/3 reached) | 1.00 ± 0.00 | 970.67 ± 15.41 | 0.03 ± 0.00 |
| multi | R | cnn | off | [0, 1, 2] | 86941 | 0/3 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.43 ± 0.37 | 0.00 ± 0.00 | 921.22 ± 2.33 | -14783.56 ± 246.95 | -4615.66 ± 12.12 | n/a (0/3 reached) | 1.00 ± 0.00 | 914.83 ± 2.49 | 0.03 ± 0.00 |
| multi | A | set_tf | off | [0, 1, 2] | 87044 | 0/3 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.23 ± 0.05 | 0.00 ± 0.00 | 919.25 ± 0.51 | -13572.57 ± 710.72 | -4606.95 ± 2.00 | n/a (0/3 reached) | 1.00 ± 0.00 | 913.83 ± 1.93 | 0.04 ± 0.00 |
| multi | B | cnn | on | [0, 1, 2] | 86941 | 0/3 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.03 ± 0.45 | 0.00 ± 0.00 | 930.30 ± 3.10 | -14610.81 ± 107.38 | -4662.34 ± 16.82 | n/a (0/3 reached) | 1.00 ± 0.00 | 916.67 ± 1.84 | 0.03 ± 0.00 |
| multi | C | set_tf | on | [0, 1, 2] | 87044 | 0/3 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.10 ± 0.25 | 0.00 ± 0.00 | 924.15 ± 3.17 | -15010.22 ± 168.56 | -4631.95 ± 16.36 | n/a (0/3 reached) | 1.00 ± 0.00 | 914.83 ± 0.94 | 0.03 ± 0.00 |

## Comparison with the reference method (avbiswas/navigation-mappo-rl)

| item | mark | detail |
|---|---|---|
| environment | not matched | reference: 2D lidar navigation; here: BlueSky competition environment |
| aircraft count | not matched | reference: up to 8 agents in 8 parallel environments; here: 1 or 2 aircraft, 1 environment |
| steps | not matched | reference: up to 1e7; here: the --steps value of the run |
| reward and reward scale | not matched | reference: dense progress reward, no scale; here: time and event penalties, scale 0.001 |
| encoder input | not matched | reference: 120 lidar rays, 4 stacked frames; here: object lists, 1 frame |

No number in this table can be compared with a number of the reference method.
