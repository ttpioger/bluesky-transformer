"""Test-time ensemble of trained policies (2026-11-19, scripts/competition_eval.py --ensemble). No training.

Each member keeps its own lidar settings and frame stack, so models with other token settings can be combined.
Modes:
  mean  the mean of the member actions (each action already clipped to [-1, 1]);
  zone  member 1 (the second model) when a restricted area is within switch_km of the own aircraft or the own
        aircraft is inside one, else member 0 (the first model).
The frame stack of every member is pushed with the action that was flown (the ensemble action).
Check: two copies of one model in mode mean give the same flights as the model alone.
"""
from __future__ import annotations
import numpy as np
from .envs import ObsBuilder
from .lidar import GEOM_KEY, zone_distance


class EnsembleFn:
    def __init__(self, members, mode="mean", switch_km=20.0):
        """members: list of (policy_fn, lidar, history); policy_fn(raw obs by id, (state, lidar) by id) -> action by id."""
        if mode not in ("mean", "zone"):
            raise ValueError(f"mode must be mean or zone, got {mode}")
        if mode == "zone" and len(members) != 2:
            raise ValueError("mode zone needs exactly two members")
        self.members, self.mode, self.switch_km = members, mode, float(switch_km)
        self.n_dec = self.n_second = 0
        self.reset()

    def reset(self):
        """Call before each episode."""
        self.builders, self.last = None, None
        self.ep_dec = self.ep_second = 0

    def __call__(self, raw, built):
        ids = list(raw)
        if self.builders is None:
            self.builders = [{} for _ in self.members]
        for b, (_, lidar, hist) in zip(self.builders, self.members):
            for a in ids:
                if a not in b:                        # first decision of this aircraft: start its frame stack
                    b[a] = ObsBuilder(lidar, hist)
                    b[a].reset(raw[a])
                else:                                 # the observation after the action flown at the last decision
                    b[a].push(raw[a], self.last[a])
        acts = [fn(raw, {a: b[a].get() for a in ids}) for (fn, _, _), b in zip(self.members, self.builders)]
        out = {}
        for a in ids:
            if self.mode == "mean":
                out[a] = np.mean([m[a] for m in acts], axis=0).astype(np.float32)
            else:
                near = zone_distance(raw[a][GEOM_KEY]["zones"]) <= self.switch_km
                out[a] = acts[1 if near else 0][a]
                self.ep_second += int(near)
            self.ep_dec += 1
        self.last = {**(self.last or {}), **out}
        return out

    def stats(self):
        """Counts of the last episode (competition_eval puts them in each flight row)."""
        return {"ens_decisions": self.ep_dec, "ens_second_model": self.ep_second}
