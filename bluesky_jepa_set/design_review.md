# Design review: Transformer and JEPA cells for the BlueSky competition task

Status of all statements: EXPLORATORY. Novelty not verified for every design choice below.

## Facts found (read, not assumed)

- Reference method (github.com/avbiswas/navigation-mappo-rl, commit 309ac1d). Encoder: Conv1d(12 -> 64, kernel 4, stride 2), Conv1d(64 -> 32, kernel 4, stride 2) on 120 lidar rays; the 12 channels are 4 stacked frames times 3 object types. An MLP (128, 128) reads the agent state. A dense layer gives 256 features. Actor and critic share this encoder. Actor: MLP (64, 64, tanh), tanh on the mean, state-independent log_std with start value 0, actions clipped to [-1, 1]. Critic: one multi-head attention layer over the agent encodings, then a mean. Training: one Adam optimizer, lr 5e-4, gamma 0.9, GAE lambda 0.95, 2 epochs, batch 128, 256 steps times 8 environments, entropy coefficient 0, one gradient clip (0.5) over all weights. Reward: dense progress to the goal, -0.05 per step, +10 goal, -10 collision.
- The reference has a dense progress reward. The BlueSky reward has none. This is a possible cause of the first failed overfit run (not verified).
- Single-agent and multi-agent observations have the same 19 keys. Only the count of other aircraft differs (10 scripted aircraft against N-1 agents). `reset(seed=1)` gives identical observations (checked). Absent aircraft are all-zero rows.
- The turn action is relative: each step adds `action * 45` degrees to the heading. Zero-mean action noise makes the heading a random walk. A policy without feedback cannot reach the 5 km capture radius by chance.
- Single agent, seed 1, zero action: waypoint reached in 102 steps, 1 intrusion (25 s), 1 restricted-area entry (3 s).

## 1. Transformer (cells A and C): option (a), query-centric cross-attention

| Option | Cost per layer | Decision |
|---|---|---|
| (a) own-aircraft token is the only query | O(M) | **Selected** |
| (b) Set Transformer (SAB, PMA) | O(M^2) | Not selected: object-to-object attention costs more and the observation is already own-aircraft-centred |
| (c) pairwise bias from geometry | O(M) with (a) | Not built: with one query the bias is a function of one key, and the key content already holds distance and relative speed |
| (d) Perceiver latents | O(L M) | Not selected: made for large inputs; here M is 18 to 27 |
| (e) graph attention | O(M k) | Not selected: changes what the actor sees |
| (f) GTrXL gates | - | Fallback only if training is unstable |

Design. One query token: MLP of (airspeed, inside_sector, cos_drift, sin_drift, waypoint_distance). Key tokens: one linear layer per object type (other aircraft, no-fly zone, sector edge point); the layer is the type vector. No slot vectors and no position vectors, so no weight depends on the object count. A key mask removes absent objects. One learned null token is always present, so a row with no objects gives no NaN. Each of the 2 layers: cross-attention (query reads keys), then an MLP, both with residual connections. Order information: each key gets a sine-cosine code of its distance (4 frequencies). The outline order of the sector edge points is lost; each point keeps distance and bearing. Sizes: d_model 64, 4 heads, 2 layers.

## 2. JEPA (cells B and C)

Action-conditioned temporal JEPA. The encoder f (1D CNN in cell B, Transformer in cell C) embeds each of k = 3 frames. A GRU mixes the k embeddings into h. A predictor g(h, action) gives the predicted embedding of the next frame. The target is the embedding of the next frame from an EMA copy of f with stop-gradient (tau 0.99). Loss: mean squared error plus a variance term (hinge on the per-dimension standard deviation, target 1) against collapse. The collapse monitor is the mean standard deviation of the embedding ("spread"), logged every epoch.

Use with RL: pretraining on saved transitions, then an auxiliary loss during RL (`--jepa-aux-coef`, default 0.1; 0 gives pretraining only). The actor reads only the embedding of the current frame. The GRU and the predictor are JEPA-only parts. Thus the actor has the same parameter count with JEPA on and off.

## 3. Changes against the first attempt

- Reward (user decision): -5 per second flown, -10 per second in intrusion, -0.1 per second in a no-fly zone, -0.1 per second outside the sector, +1 waypoint, -0.01 per radian of drift. The time term uses the `flight_time` metric. The environment code is not changed.
- Potential-based shaping (flag `--shaping potential`, default): potential = time penalty times the estimated seconds to the waypoint at 150 m/s. It does not change the best policy (Ng et al. 1999) and gives a dense signal. Logged returns do not contain it.
- Lower start noise (log_std -1.0), small start weights in the action layer (start mean near zero), value bootstrap at the time limit.

## 4. References

All identifiers are from memory. None was checked online in this work.

- Vaswani et al. 2017, Attention Is All You Need (arXiv 1706.03762). DOI not verified.
- Lee et al. 2019, Set Transformer (arXiv 1810.00825). DOI not verified.
- Jaegle et al. 2021, Perceiver (arXiv 2103.03206). DOI not verified.
- Velickovic et al. 2018, Graph Attention Networks (arXiv 1710.10903). DOI not verified.
- Parisotto et al. 2020, Stabilizing Transformers for RL, GTrXL (arXiv 1910.06764). DOI not verified.
- Assran et al. 2023, I-JEPA (arXiv 2301.08243). DOI not verified.
- Grill et al. 2020, BYOL, EMA target (arXiv 2006.07733). DOI not verified.
- Bardes et al. 2022, VICReg, variance term (arXiv 2105.04906). DOI not verified.
- Schulman et al. 2017, PPO (arXiv 1707.06347). DOI not verified.
- Yu et al. 2022, MAPPO (arXiv 2103.01955). DOI not verified.
- Ng, Harada, Russell 1999, Policy invariance under reward transformations (ICML). DOI not verified.

## 5. Risks and stop conditions

- Risk: the time term (-5 per second) is 50 times the zone term (-0.1 per second). A flight through a no-fly zone can be the best policy for this reward. Then the pass criterion "restricted-area events near 0" fails for a reason that is not a learning fault.
- Risk: JEPA cells use extra transitions and extra compute before RL. The RL step budget is equal, the total budget is not.
- Risk: 3 seeds and one scenario. An overfit pass shows only that the learning loop works.
- Stop 1: an encoder does not fit the scripted controller by imitation (final MSE above 0.005 or no waypoint in the closed-loop flight). Then do no RL run for this encoder.
- Stop 2: the JEPA spread falls below 0.05 (collapse). Then do no RL run for the JEPA cells.
- Stop 3: the smoke run gives more than 1 hour per run. Then cut `--steps` first.
- Stop 4: cell R fails the pass criterion on all seeds. Then the fault is in the loop or the reward, and no encoder comparison is valid.
