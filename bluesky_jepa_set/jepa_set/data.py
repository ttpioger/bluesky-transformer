"""Transition file (npz): obs_<key>, next_<key>, action, label, agent, episode, t, done."""
from __future__ import annotations
import numpy as np
from .obs import ALL_KEYS


def save_transitions(path, obs, nxt, action, label, agent, episode, t, done, meta):
    arrays = {f"obs_{k}": obs[k] for k in ALL_KEYS}
    arrays.update({f"next_{k}": nxt[k] for k in ALL_KEYS})
    np.savez_compressed(path, action=action, label=label, agent=agent, episode=episode, t=t, done=done,
                        meta=np.array(str(meta)), **arrays)


def load_transitions(path):
    d = np.load(path, allow_pickle=False)
    obs = {k: d[f"obs_{k}"] for k in ALL_KEYS}
    nxt = {k: d[f"next_{k}"] for k in ALL_KEYS}
    return obs, nxt, {k: d[k] for k in ["action", "label", "agent", "episode", "t", "done"]}


def history_index(agent, episode, k):
    """Row indices [n, k] of the frames t-k+1..t of the same aircraft and episode.
    Rows are in time order. The first frame of an episode is repeated when the history is short."""
    n = len(agent)
    prev = np.arange(n)
    last = {}
    for i in range(n):
        key = (int(episode[i]), int(agent[i]))
        if key in last:
            prev[i] = last[key]
        last[key] = i
    cols = [np.arange(n)]
    for _ in range(k - 1):
        cols.append(prev[cols[-1]])
    return np.stack(cols[::-1], axis=1)
