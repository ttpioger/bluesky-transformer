"""Collect transitions (observation, action, next observation, aircraft id) from the fixed scenario.

Behaviour policy: scripted controller plus Gaussian noise. The noise level changes per episode.
The file also holds the clean controller action (label) for the imitation test.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from jepa_set.envs import FlightEnv, controller
from jepa_set.obs import stack_obs
from jepa_set.data import save_transitions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setting", choices=["single", "multi"], default="multi")
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--transitions", type=int, default=6000, help="Aircraft-steps to collect.")
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.1, 0.3, 0.6])
    p.add_argument("--reset-seed", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Default: data/transitions_SETTING.npz")
    args = p.parse_args(argv)
    out = Path(args.out or f"data/transitions_{args.setting}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = FlightEnv(args.setting, args.n_agents)
    O, NX, A, L, AG, EP, TT, DN = [], [], [], [], [], [], [], []
    ep = -1
    while len(A) < args.transitions:
        obs = env.reset(seed=args.reset_seed)
        ep += 1
        noise = args.noise_levels[ep % len(args.noise_levels)]
        t = 0
        while env.agents:
            live = list(env.agents)
            labels = {a: controller(obs[a]) for a in live}
            acts = {a: np.clip(labels[a] + rng.normal(0.0, noise, 2), -1.0, 1.0).astype(np.float32) for a in live}
            nobs, _, te, tr, _ = env.step(acts)
            for a in live:
                O.append(obs[a]); NX.append(nobs[a]); A.append(acts[a]); L.append(labels[a])
                AG.append(env.ids.index(a)); EP.append(ep); TT.append(t); DN.append(te[a] or tr[a])
            obs = nobs
            t += 1
    env.close()
    save_transitions(out, stack_obs(O), stack_obs(NX), np.stack(A), np.stack(L), np.array(AG), np.array(EP),
                     np.array(TT), np.array(DN), {"args": vars(args), "episodes": ep + 1})
    print(f"Wrote {out}: {len(A)} transitions, {ep + 1} episodes")


if __name__ == "__main__":
    main()
