# Overfit comparison (status: EXPLORATORY)

One fixed scenario per setting. Mean ± std over seeds. Windows: first and last 20 agent-episodes of training (with exploration noise).
An overfit pass shows only that the learning loop works. This table does not show that a cell is better or worse in general.

| setting | cell | encoder | jepa | seeds | n_params actor | pass | waypoint (last) | intrusion ev. | restricted ev. | sector-exit ev. | flight time s (last) | return (first) | return (last) | steps to criterion | greedy waypoint | greedy flight time s | s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | R | cnn | off | [0, 1] | 86941 | 0/2 | 0.05 ± 0.05 | 2.00 ± 0.20 | 1.60 ± 0.40 | 0.95 ± 0.05 | 2986.40 ± 13.60 | -16174.26 ± 61.50 | -16174.26 ± 61.50 | n/a (0/2 reached) | 0.00 ± 0.00 | 3000.00 ± 0.00 | 0.02 ± 0.00 |
| single | A | set_tf | off | [0, 1] | 87044 | 0/2 | 0.10 ± 0.00 | 1.65 ± 0.05 | 1.30 ± 0.30 | 0.80 ± 0.00 | 2890.10 ± 72.30 | -16028.39 ± 323.31 | -16028.39 ± 323.31 | n/a (0/2 reached) | 1.00 ± 0.00 | 1189.00 ± 72.00 | 0.03 ± 0.00 |
| single | B | cnn | on | [0, 1] | 86941 | 0/2 | 0.00 ± 0.00 | 1.30 ± 0.10 | 1.20 ± 0.40 | 1.00 ± 0.10 | 3000.00 ± 0.00 | -16282.74 ± 1.71 | -16282.74 ± 1.71 | n/a (0/2 reached) | 0.00 ± 0.00 | 3000.00 ± 0.00 | 0.03 ± 0.00 |
| single | C | set_tf | on | [0, 1] | 87044 | 0/2 | 0.00 ± 0.00 | 1.60 ± 0.40 | 1.05 ± 0.65 | 1.00 ± 0.10 | 3000.00 ± 0.00 | -16215.22 ± 366.74 | -16215.22 ± 366.74 | n/a (0/2 reached) | 0.00 ± 0.00 | 3000.00 ± 0.00 | 0.02 ± 0.00 |
| multi | R | cnn | off | [0, 1] | 86941 | 0/2 | 0.05 ± 0.00 | 0.10 ± 0.00 | 1.65 ± 0.10 | 1.00 ± 0.10 | 2944.07 ± 24.48 | -14941.38 ± 129.43 | -14941.38 ± 129.43 | n/a (0/2 reached) | 0.00 ± 0.00 | 3000.00 ± 0.00 | 0.04 ± 0.00 |
| multi | A | set_tf | off | [0, 1] | 87044 | 0/2 | 0.28 ± 0.07 | 0.00 ± 0.00 | 2.20 ± 0.05 | 0.75 ± 0.05 | 2686.50 ± 172.45 | -13554.20 ± 869.87 | -13554.20 ± 869.87 | n/a (0/2 reached) | 1.00 ± 0.00 | 1013.00 ± 9.50 | 0.04 ± 0.00 |
| multi | B | cnn | on | [0, 1] | 86941 | 0/2 | 0.05 ± 0.00 | 0.00 ± 0.00 | 2.17 ± 0.07 | 1.02 ± 0.12 | 2904.65 ± 4.50 | -14685.68 ± 21.94 | -14685.68 ± 21.94 | n/a (0/2 reached) | 0.25 ± 0.25 | 2485.25 ± 514.75 | 0.03 ± 0.00 |
| multi | C | set_tf | on | [0, 1] | 87044 | 0/2 | 0.03 ± 0.03 | 0.05 ± 0.05 | 2.95 ± 0.10 | 0.78 ± 0.12 | 2994.47 ± 5.53 | -15126.98 ± 41.48 | -15126.98 ± 41.48 | n/a (0/2 reached) | 0.50 ± 0.00 | 2511.50 ± 455.00 | 0.02 ± 0.00 |

## Comparison with the reference method (avbiswas/navigation-mappo-rl)

| item | mark | detail |
|---|---|---|
| environment | not matched | reference: 2D lidar navigation; here: BlueSky competition environment |
| aircraft count | not matched | reference: up to 8 agents in 8 parallel environments; here: 1 or 2 aircraft, 1 environment |
| steps | not matched | reference: up to 1e7; here: the --steps value of the run |
| reward and reward scale | not matched | reference: dense progress reward, no scale; here: time and event penalties, scale 0.001 |
| encoder input | not matched | reference: 120 lidar rays, 4 stacked frames; here: object lists, 1 frame |

No number in this table can be compared with a number of the reference method.
