"""Observation Dict -> tensors. One row per object; an all-zero row is an absent object."""
from __future__ import annotations
import numpy as np
import torch

OWN_KEYS = ["airspeed", "inside_sector", "cos_drift", "sin_drift", "waypoint_distance"]
AC_KEYS = ["x_r", "y_r", "vx_r", "vy_r", "cos_track", "sin_track", "intruder_distance"]
ZONE_KEYS = ["obstacle_distance", "obstacle_cos_bearing", "obstacle_sin_bearing", "obstacle_radius"]
EDGE_KEYS = ["sector_point_distance", "sector_point_cos_bearing", "sector_point_sin_bearing"]
ALL_KEYS = OWN_KEYS + AC_KEYS + ZONE_KEYS + EDGE_KEYS
# Index of the distance feature in each object type (used by the distance code).
DIST_INDEX = {"ac": 6, "zone": 0, "edge": 0}


def stack_obs(obs_list):
    """List of observation dicts -> dict of float32 arrays [len, n]."""
    return {k: np.stack([np.asarray(o[k], dtype=np.float32) for o in obs_list]) for k in ALL_KEYS}


def zero_obs(like):
    return {k: np.zeros_like(np.asarray(like[k], dtype=np.float32)) for k in ALL_KEYS}


def to_torch(obs, device="cpu"):
    return {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device) for k, v in obs.items()}


def split_obs(obs):
    """Dict of tensors [B, n] -> own [B, 5], ac [B, Na, 7], zone [B, Nz, 4], edge [B, Ne, 3]."""
    own = torch.cat([obs[k] for k in OWN_KEYS], dim=-1)
    ac = torch.stack([obs[k] for k in AC_KEYS], dim=-1)
    zone = torch.stack([obs[k] for k in ZONE_KEYS], dim=-1)
    edge = torch.stack([obs[k] for k in EDGE_KEYS], dim=-1)
    return own, ac, zone, edge


def absent_mask(x):
    """True where the object row is all zero (absent object)."""
    return x.abs().sum(-1) == 0
