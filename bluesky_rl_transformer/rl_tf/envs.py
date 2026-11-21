"""Environment side: reward, frame stack, and a pool of environment processes.

The BlueSky environment code and its metrics are not changed. Reward weights go in through the
environment constructor. The time term, the event terms and the turn-change term are added here.
BlueSky allows one environment per process, so the pool has one process per environment.
"""
from __future__ import annotations
import os
from collections import deque
import multiprocessing as mp
import numpy as np
from .lidar import (Lidar, own_state, state_dim, extra_dim, N_CHANNELS, ACT_FEATURES, DIST_NORM_KM, GEOM_KEY, NM2KM,
                    to_own_frame, inflate_polygon, zone_distance)
from .heading_mask import MASK_DEFAULTS, mask_geometry, choose_offset

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # no window for rgb_array rendering

METRIC_KEYS = ["waypoint_reached", "flight_time", "intrusion_events", "intrusion_time",
               "restricted_area_events", "time_in_restricted_area",
               "sector_exit_events", "time_outside_sector"]
# Single agent only (2026-11-13), NOT official metrics: the official metrics count an intrusion at reset when another
# aircraft is already within 5 NM. These keys leave such start intrusions out (see track_start_intrusions).
START_KEYS = ["start_intrusions", "intrusion_events_no_start", "intrusion_time_no_start"]


def track_start_intrusions(inner):
    """Wrap the metric methods of the single-agent competition environment `inner` (no change to its metrics).
    A start intrusion: another aircraft within the intrusion distance at the reset sample (t = 0). It stays left out
    until that aircraft leaves the intrusion distance; a later new intrusion with the same aircraft counts.
    Returns the live dict {start_intrusions, intrusion_events_no_start, intrusion_time_no_start} of the flight."""
    init0, upd0 = inner._init_metrics, inner._update_metrics
    out = {k: 0 for k in START_KEYS}
    st = {"first": True, "start": set(), "counted": set()}

    def init(ac_id):
        init0(ac_id)
        out.update(start_intrusions=0, intrusion_events_no_start=0, intrusion_time_no_start=0.0)
        st.update(first=True, start=set(), counted=set())

    def update(ac_id, dt=None):
        upd0(ac_id, dt)
        now = set(inner._intruders_in_intrusion)
        if st["first"]:                                # the reset sample
            st["first"], st["start"] = False, now
            out["start_intrusions"] = len(now)
            return
        st["start"] &= now                             # a start intrusion ends when that aircraft goes out
        counted = now - st["start"]
        out["intrusion_events_no_start"] += len(counted - st["counted"])
        out["intrusion_time_no_start"] += len(counted) * (inner.sim_dt if dt is None else dt)
        st["counted"] = counted

    inner._init_metrics, inner._update_metrics = init, update
    return out

# Reward. Unit of the time terms: per simulated second. Event terms: once per entry or exit.
# Hard constraint (user, 2026-11-11): the reward has only negative terms. No waypoint reward. check_reward
# rejects a positive weight.
# 2026-11-11 (user): the no-fly zone terms were -50 per second and -500 per entry. With these values the
# policy of sa_tf_neg_s0 flew a circle at its start point. The zone terms are now -10 and -100, and the
# goal-distance term gives a cost to each second far from the goal.
REWARD_DEFAULTS = dict(drift_penalty=-0.01, time_penalty=-5.0, intrusion_penalty=-10.0,
                       restricted_penalty=-10.0, sector_exit_penalty=-50.0,
                       zone_entry_penalty=-100.0, sector_exit_event_penalty=-500.0, turn_change_penalty=0.0,
                       goal_distance_penalty=-0.05, heading_penalty=0.0,
                       intrusion_event_penalty=0.0)

# Reward presets (--reward-preset). "friend": the reward that a friend of the user uses (user, 2026-11-11):
# r_t = -sum over the simulated seconds k of the step (100 x1_k + 10 x2_k + 10 x3_k + 1 (1 - cos psi_t) + 0.1),
# x1 = aircraft within 5 NM, x2 = in a restricted area, x3 = outside the sector, psi = angle between the
# heading and the bearing to the waypoint at the end of the step. All other terms are 0.
REWARD_PRESETS = {
    "friend": dict(drift_penalty=0.0, time_penalty=-0.1, intrusion_penalty=-100.0, restricted_penalty=-10.0,
                   sector_exit_penalty=-10.0, zone_entry_penalty=0.0, sector_exit_event_penalty=0.0,
                   turn_change_penalty=0.0, goal_distance_penalty=0.0, heading_penalty=-1.0,
                   intrusion_event_penalty=0.0),
    # "los" (user, 2026-11-11): loss of separation (LoS = another aircraft within 5 NM) -0.5 per second and
    # -3 per event, no-fly zone -0.3 per second and -2 per entry, outside the sector -0.2 per second,
    # time -0.015 per second. The user gave no other term, so all other terms are 0.
    "los": dict(drift_penalty=0.0, time_penalty=-0.015, intrusion_penalty=-0.5, restricted_penalty=-0.3,
                sector_exit_penalty=-0.2, zone_entry_penalty=-2.0, sector_exit_event_penalty=0.0,
                turn_change_penalty=0.0, goal_distance_penalty=0.0, heading_penalty=0.0,
                intrusion_event_penalty=-3.0)}
assert all(set(v) == set(REWARD_DEFAULTS) for v in REWARD_PRESETS.values())
REWARD_HELP = dict(
    drift_penalty="Per radian of drift, per step.",
    time_penalty="Per second flown (delta of the flight_time metric).",
    intrusion_penalty="Per second in intrusion.", restricted_penalty="Per second in a no-fly zone.",
    sector_exit_penalty="Per second outside the sector.",
    zone_entry_penalty="Once per entry into a no-fly zone.",
    sector_exit_event_penalty="Once per exit from the sector.",
    turn_change_penalty="Per unit of change of the turn action between two steps (give a negative value).",
    goal_distance_penalty="Per second flown and per km of distance to the goal (distance after the step).",
    heading_penalty="Per second flown, times (1 - cos psi); psi = angle between the heading and the bearing "
                    "to the goal after the step.",
    intrusion_event_penalty="Once per start of an intrusion (another aircraft comes within 5 NM).")


def add_reward_args(p):
    for k, v in REWARD_DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=float, default=None,
                       help=f"{REWARD_HELP[k]} Default: {v}, or the preset value.")
    p.add_argument("--reward-preset", choices=["none"] + list(REWARD_PRESETS), default="none",
                   help="A preset sets all reward weights. A weight flag that you give overrides the preset.")


def check_reward(rw):
    """Hard constraint: every reward weight is zero or negative, so the reward of each step is <= 0."""
    bad = {k: v for k, v in rw.items() if v > 0.0}
    if bad:
        raise ValueError(f"the reward has only negative terms; positive weights are not allowed: {bad}")
    return rw


# v2 options (2026-11-19, single agent rebuild). Defaults = the baseline behaviour (no change).
# action_map "goal": the turn action a in [-1, 1] commands the heading (bearing to the goal + a * goal_offset_deg).
#   The wrapper sends the heading change to that target, clipped to +-d_heading of the environment (default 45 deg,
#   constructor unchanged; the aircraft turns less than 45 deg in a 10 s decision). "rel": the action of the env.
# cpa_margin_penalty: per second, times max over the other aircraft of (1 - d_cpa / margin) (0 when d_cpa >= margin);
#   d_cpa = closest distance on straight lines within cpa_horizon_s. Margin = cpa_margin_nm (5 NM = separation).
# zone_margin_penalty: per second, times (1 - d_zone / zone_buffer_km) (0 when d_zone >= buffer; 1 inside a zone).
# Both margin weights are <= 0 (the reward has only negative terms).
# timeout_penalty (2026-11-20): <= 0, added once when a flight ends at the time limit with the goal not reached
#   (a clean flight needs the goal; before, the time limit had no cost of its own). 0 = off.
# skip_start_events (2026-11-20, single agent): 1 = the event penalties leave out what exists at the reset sample and
#   cannot be avoided: intrusions with aircraft already within 5 NM (until that aircraft goes out of 5 NM) and a
#   start inside a restricted area or outside the sector. The per-second penalties stay (the policy can shorten that
#   time). The scored metrics do not change. 0 = as before.
# heading_mask (2026-11-21, action_map "goal" only): see rl_tf/heading_mask.py. 0 = off (behaviour as before).
V2_DEFAULTS = dict(action_map="rel", goal_offset_deg=90.0, cpa_margin_penalty=0.0, cpa_margin_nm=7.5,
                   cpa_horizon_s=120.0, zone_margin_penalty=0.0, zone_buffer_km=10.0, timeout_penalty=0.0,
                   skip_start_events=0, **MASK_DEFAULTS)


def cat_prob_at(env_steps, start, end, ramp_steps):
    """Random CaT schedule (2026-11-19): linear from start to end over ramp_steps env steps, then end."""
    if ramp_steps <= 0:
        return float(end)
    f = min(1.0, max(0.0, env_steps / float(ramp_steps)))
    return float(start + (end - start) * f)


def v2_kwargs(c):
    """v2 options from a dict of run arguments (old checkpoints: the defaults)."""
    c = c if isinstance(c, dict) else vars(c)
    out = {k: (c.get(k) if c.get(k) is not None else v) for k, v in V2_DEFAULTS.items()}
    if out["action_map"] not in ("rel", "goal"):
        raise ValueError(f"action_map must be rel or goal, got {out['action_map']}")
    if out["cpa_margin_penalty"] > 0.0 or out["zone_margin_penalty"] > 0.0:
        raise ValueError("the margin penalties must be <= 0 (the reward has only negative terms)")
    if out["timeout_penalty"] > 0.0:
        raise ValueError("timeout_penalty must be <= 0 (the reward has only negative terms)")
    if out["heading_mask"] and out["action_map"] != "goal":
        raise ValueError("heading_mask needs action_map goal")
    if out["heading_mask"] and (out["mask_step_deg"] <= 0.0 or out["mask_dist_km"] < 0.0 or out["mask_margin_km"] < 0.0):
        raise ValueError("mask_step_deg must be > 0; mask_dist_km and mask_margin_km must be >= 0")
    return out


def cpa_margin_cost(g, margin_km, horizon_s):
    """max over the other aircraft of (1 - d_cpa / margin_km), clipped to [0, 1]; d_cpa = closest distance (km)
    in [0, horizon_s] if both fly straight. g: geometry of one observation (own frame, km; aircraft_vel in m/s)."""
    p = np.asarray(g["aircraft"], dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        return 0.0
    v = np.asarray(g["aircraft_vel"], dtype=np.float64).reshape(-1, 2) / 1000.0      # km/s
    vv = (v * v).sum(-1)
    t = np.where(vv > 1e-12, -(p * v).sum(-1) / np.maximum(vv, 1e-12), 0.0)
    t = np.clip(t, 0.0, horizon_s)
    d = np.hypot(*(p + v * t[:, None]).T)
    return float(np.clip(1.0 - d / margin_km, 0.0, 1.0).max())


def zone_margin_cost(g, buffer_km):
    """(1 - d / buffer_km) clipped to [0, 1]; d = distance (km) from the own aircraft to the nearest zone."""
    d = zone_distance(g["zones"])
    return float(np.clip(1.0 - d / buffer_km, 0.0, 1.0)) if np.isfinite(d) else 0.0


def reward_kwargs(args):
    base = REWARD_PRESETS.get(getattr(args, "reward_preset", "none"), REWARD_DEFAULTS)
    for k, v in base.items():                    # order: given flag, preset, default. The result goes into
        if getattr(args, k) is None:             # args, so that the checkpoint holds the weights in use.
            setattr(args, k, v)
    return {k: getattr(args, k) for k in REWARD_DEFAULTS}


class FlightEnv:
    """One interface for the single-agent and the multi-agent competition environment.

    reset(seed) -> {id: obs}. step({id: action}) -> obs, reward, terminated, truncated, info (dicts by id).
    `agents` is the list of live aircraft ids after the call. `ids` is the fixed list of all ids.
    """

    def __init__(self, setting="multi", n_agents=2, render_mode=None, env_kw=None, zone_margin_km=0.0, v2=None,
                 cat_penalty=0.0, cat_seed=None, fast_metrics=False, **reward):
        """env_kw (2026-11-19): extra constructor arguments of the competition environment that the rules permit
        (action_frequency, d_heading, d_speed). zone_margin_km: the restricted areas in the geometry of the
        observation are grown by this margin (test-time option; the scored metrics use the real areas).
        v2 (2026-11-19): see V2_DEFAULTS; None = the defaults (baseline behaviour).
        cat_penalty (training only, single agent): < 0 ends the flight at the first avoidable violation (a new
        intrusion that is not a start intrusion, a restricted-area entry, a sector exit) with this reward added.
        0 = off. The scored metrics are not changed.
        cat_prob (2026-11-19, set with set_cat_prob; default 1.0): probability that a violation ends the flight
        (random CaT, after Chane-Sane et al. 2024, arXiv:2403.18765). 1.0 = always (no random draw, as before).
        A violation that does not end the flight gets only the usual event penalties. cat_seed: seed of that draw."""
        import bluesky_gym
        reward.pop("reach_reward", None)                 # key of older checkpoints; the waypoint reward is removed
        rw = check_reward({**REWARD_DEFAULTS, **reward})
        self.rw = rw
        kw = dict(reach_reward=0.0, drift_penalty=rw["drift_penalty"],
                  intrusion_penalty=rw["intrusion_penalty"], restricted_area_penalty=rw["restricted_penalty"],
                  sector_exit_penalty=rw["sector_exit_penalty"])
        allowed = {"action_frequency", "d_heading", "d_speed"}
        if env_kw and set(env_kw) - allowed:
            raise ValueError(f"env_kw: only {sorted(allowed)} may change, got {sorted(env_kw)}")
        kw.update(env_kw or {})
        self.v2 = v2_kwargs(v2 or {})
        if self.v2["skip_start_events"] and setting != "single":
            raise ValueError("skip_start_events: single agent only (the multi-agent environment has no start tracker)")
        if cat_penalty > 0.0:
            raise ValueError("cat_penalty must be <= 0 (the reward has only negative terms)")
        if cat_penalty < 0.0 and setting != "single":
            raise ValueError("cat_penalty: single agent only")
        self.cat_penalty = float(cat_penalty)
        self.cat_prob = 1.0
        self._cat_rng = np.random.default_rng(cat_seed)
        self.zone_margin_km = float(zone_margin_km)
        self.setting = setting
        if setting == "single":
            import gymnasium as gym
            bluesky_gym.register_envs()
            self.env = gym.make("CompetitionEnv-v0", render_mode=render_mode, **kw)
            self.ids = ["own"]
            self._start_metrics = track_start_intrusions(self.env.unwrapped)
        elif setting == "multi":
            from bluesky_zoo.competition_v0 import CompetitionZooEnv
            self.env = CompetitionZooEnv(n_agents=n_agents, render_mode=render_mode, **kw)
            if fast_metrics:                           # training only (2026-11-20), see rl_tf/fast_metrics.py
                from .fast_metrics import install
                install(self.env)
            self.ids = list(self.env.possible_agents)
        else:
            raise ValueError("setting must be single or multi")
        self.agents = []
        self._prev = {}
        self.last_exec = None                          # heading mask: executed policy actions of the last step
        self._mask_stats = {}

    def reset(self, seed=None):
        if self.setting == "single":
            o, info0 = self.env.reset(seed=seed)
            obs = {"own": o}
            # violation counts at the reset sample (start inside an area is not avoidable)
            self._cat_prev = dict(intr=0.0, zone=float(info0["restricted_area_events"]),
                                  exit=float(info0["sector_exit_events"]))
        else:
            obs, _ = self.env.reset(seed=seed)
        self.agents = list(obs)
        self._prev = {a: dict(t=0.0, zone=0.0, exit=0.0, turn=0.0, intr=0.0) for a in self.agents}
        if self.v2["skip_start_events"]:               # events that exist at the reset sample get no event penalty
            self._prev["own"].update(zone=self._cat_prev["zone"], exit=self._cat_prev["exit"])
        self._geom = {}
        self._add_geometry(obs)
        self.last_exec = None
        if self.v2["heading_mask"]:                    # per flight: decisions, projected, sum of |angle|, fallbacks
            self._mask_stats = {a: dict(dec=0, proj=0, angle=0.0, fb=0) for a in self.ids}
        return obs

    def env_action(self, actions):
        """Policy actions by id -> actions of the competition environment (action_map). "rel": no change."""
        if self.v2["action_map"] == "rel":
            return actions
        import bluesky as bs
        from core.tools import kwikqdrdist
        inner, out = self.env.unwrapped, {}
        mask = bool(self.v2["heading_mask"])
        if mask:
            self.last_exec = {}
        for a, act in actions.items():
            act = np.asarray(act, dtype=np.float64).copy()
            idx = bs.traf.id2idx(self._ac_id(a))
            if mask:
                ex = np.clip(act, -1.0, 1.0)
            if idx >= 0:
                goal = (inner.goal_lat, inner.goal_lon) if self.setting == "single" else inner._goal[a]
                qdr, dist_nm = kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], goal[0], goal[1])
                if mask:                               # heading mask (2026-11-21), rl_tf/heading_mask.py
                    ex[0] = self._mask_turn(a, float(qdr), float(dist_nm) * NM2KM, float(bs.traf.hdg[idx]), ex[0])
                    target = float(qdr) + float(ex[0]) * self.v2["goal_offset_deg"]
                else:
                    target = float(qdr) + float(np.clip(act[0], -1.0, 1.0)) * self.v2["goal_offset_deg"]
                delta = (target - float(bs.traf.hdg[idx]) + 180.0) % 360.0 - 180.0
                act[0] = float(np.clip(delta / float(inner.d_heading), -1.0, 1.0))
            out[a] = act
            if mask:
                self.last_exec[a] = ex.astype(np.float32)
        return out

    def _mask_turn(self, a, qdr, goal_km, hdg, turn):
        """Heading mask for one aircraft: policy turn action (clipped) -> executed turn action. Static geometry of
        the last observation (areas and sector, own frame). Counts the decision in the flight statistics."""
        v2, g = self.v2, self._geom[a]
        G = float(v2["goal_offset_deg"])
        zones = g["zones"] if self.zone_margin_km <= 0.0 else self._true_zones(a)
        edges = mask_geometry(zones, g["sector"], v2["mask_margin_km"], bool(v2["mask_sector"]))
        goal_rel = (qdr - hdg + 180.0) % 360.0 - 180.0
        cur_off = (hdg - qdr + 180.0) % 360.0 - 180.0
        req = float(turn) * G
        off, proj, fb = choose_offset(edges, goal_rel, req, cur_off, G, v2["mask_dist_km"], v2["mask_step_deg"],
                                      goal_km)
        st = self._mask_stats.setdefault(a, dict(dec=0, proj=0, angle=0.0, fb=0))
        st["dec"] += 1
        if proj:
            st["proj"] += 1
            st["angle"] += abs(off - req)
            st["fb"] += int(fb)
        return off / G if proj else float(turn)

    def _true_zones(self, a):
        """The real restricted areas in the own frame (the observation may show them grown by zone_margin_km)."""
        import bluesky as bs
        idx = bs.traf.id2idx(self._ac_id(a))
        lat0, lon0, hdg = float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), float(bs.traf.hdg[idx])
        return [to_own_frame(np.asarray(z.vertices)[:, 0], np.asarray(z.vertices)[:, 1], lat0, lon0, hdg)
                for z in self.env.unwrapped.scenario.obstacles]

    def step(self, actions):
        env_acts = self.env_action(actions)
        if self.setting == "single":
            o, r, te, tr, info = self.env.step(np.asarray(env_acts["own"], dtype=np.float64))
            info = {**info, **self._start_metrics}     # official metrics unchanged; START_KEYS added
            obs, rew, term, trunc, infos = {"own": o}, {"own": r}, {"own": te}, {"own": tr}, {"own": info}
            self.agents = [] if (te or tr) else ["own"]
        else:
            obs, rew, term, trunc, infos = self.env.step(env_acts)
            self.agents = list(self.env.agents)
        self._add_geometry(obs)
        v2 = self.v2
        if self.cat_penalty < 0.0:                     # constraints as terminations (training only)
            i, cp = infos["own"], self._cat_prev
            now = dict(intr=float(i["intrusion_events_no_start"]), zone=float(i["restricted_area_events"]),
                       exit=float(i["sector_exit_events"]))
            hit = any(now[k] > cp[k] for k in now)
            self._cat_prev = now
            if hit and self.cat_prob < 1.0:            # random CaT: no draw at cat_prob 1.0 (behaviour as before)
                hit = bool(self._cat_rng.random() < self.cat_prob)
            if hit and not (term["own"] or trunc["own"]):
                term = {"own": True}
                self.agents = []
            infos["own"] = {**i, "cat_end": 1.0 if hit else 0.0}
        rw, out_rew = self.rw, {}
        for a in obs:
            i, p = infos[a], self._prev[a]
            t, zone, ext = float(i["flight_time"]), float(i["restricted_area_events"]), float(i["sector_exit_events"])
            ex = self.last_exec                       # heading mask on: the executed turn action
            turn = float(ex[a][0]) if ex and a in ex else float(actions[a][0])
            intr = float(i["intrusion_events_no_start"] if v2["skip_start_events"] else i["intrusion_events"])
            gx, gy = obs[a][GEOM_KEY]["goal"]                          # own frame: origin = own aircraft, x = nose
            goal_km = float(np.hypot(gx, gy))
            cos_psi = float(gx) / goal_km if goal_km > 1e-9 else 1.0
            out_rew[a] = (float(rew[a]) + rw["time_penalty"] * (t - p["t"])
                          + rw["goal_distance_penalty"] * goal_km * (t - p["t"])
                          + rw["heading_penalty"] * (1.0 - cos_psi) * (t - p["t"])
                          + rw["zone_entry_penalty"] * (zone - p["zone"])
                          + rw["sector_exit_event_penalty"] * (ext - p["exit"])
                          + rw["intrusion_event_penalty"] * (intr - p["intr"])
                          + rw["turn_change_penalty"] * abs(turn - p["turn"]))
            if v2["cpa_margin_penalty"] < 0.0:
                out_rew[a] += v2["cpa_margin_penalty"] * (t - p["t"]) * cpa_margin_cost(
                    obs[a][GEOM_KEY], v2["cpa_margin_nm"] * NM2KM, v2["cpa_horizon_s"])
            if v2["zone_margin_penalty"] < 0.0:
                out_rew[a] += v2["zone_margin_penalty"] * (t - p["t"]) * zone_margin_cost(
                    obs[a][GEOM_KEY], v2["zone_buffer_km"])
            if self.cat_penalty < 0.0 and infos[a].get("cat_end"):
                out_rew[a] += self.cat_penalty
            if v2["timeout_penalty"] < 0.0 and trunc[a] and not term[a] and float(i["waypoint_reached"]) < 1.0:
                out_rew[a] += v2["timeout_penalty"]
            p.update(t=t, zone=zone, exit=ext, turn=turn, intr=intr)
        if self.v2["heading_mask"]:                    # mask statistics of the flight so far (not scored)
            for a in obs:
                st = self._mask_stats.get(a, dict(dec=0, proj=0, angle=0.0, fb=0))
                infos[a] = {**infos[a], "mask_decisions": float(st["dec"]), "mask_projected": float(st["proj"]),
                            "mask_angle_sum_deg": float(st["angle"]), "mask_fallbacks": float(st["fb"])}
        return obs, out_rew, {a: bool(term[a]) for a in obs}, {a: bool(trunc[a]) for a in obs}, infos

    def set_cat_prob(self, p):
        """Random CaT (2026-11-19): probability in (0, 1] that a violation ends the flight, from the next step on."""
        if not 0.0 < float(p) <= 1.0:
            raise ValueError("cat_prob must be in (0, 1]")
        self.cat_prob = float(p)

    def set_density(self, n_intruders, n_obstacles):
        """Curriculum (2026-11-13): the scenario generator makes n_intruders other aircraft and n_obstacles restricted
        areas from the next reset on. The generator reads both counts at each reset (core/scenario.py). The size of
        the observation of the environment does not change: absent objects are empty rows. The competition values
        are 10 and 5 (the constructor defaults)."""
        gen = getattr(self.env.unwrapped, "scenario_generator", None)
        if gen is None:
            raise ValueError("this environment has no scenario_generator; curriculum is not possible")
        gen.n_intruders, gen.n_obstacles = int(n_intruders), int(n_obstacles)

    def _ac_id(self, a):
        return self.env.unwrapped.agent if self.setting == "single" else a

    def _add_geometry(self, obs):
        """Add the exact simulator geometry for the lidar (own frame, km) to each observation (key GEOM_KEY).
        An aircraft that BlueSky has deleted (waypoint reached) keeps its last geometry."""
        import bluesky as bs
        inner = self.env.unwrapped
        sc = inner.scenario
        for a, o in obs.items():
            idx = bs.traf.id2idx(self._ac_id(a))
            if idx < 0:
                o[GEOM_KEY] = self._geom[a]
                continue
            lat0, lon0, hdg = float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), float(bs.traf.hdg[idx])
            frame = lambda ll: to_own_frame(np.asarray(ll)[:, 0], np.asarray(ll)[:, 1], lat0, lon0, hdg)
            others = np.arange(bs.traf.ntraf) != idx
            h = np.deg2rad(hdg)                        # velocity of the others minus the own velocity, own frame, m/s
            ve, vn = bs.traf.gseast[others] - bs.traf.gseast[idx], bs.traf.gsnorth[others] - bs.traf.gsnorth[idx]
            vel = np.stack([ve * np.sin(h) + vn * np.cos(h), ve * np.cos(h) - vn * np.sin(h)], axis=-1).reshape(-1, 2)
            goal = (inner.goal_lat, inner.goal_lon) if self.setting == "single" else inner._goal[a]
            t_now = float(inner.metrics[inner.agent]["flight_time"]) if self.setting == "single" else float(inner.sim_time)
            zones = [frame(z.vertices) for z in sc.obstacles]
            if self.zone_margin_km > 0.0:
                zones = [inflate_polygon(z, self.zone_margin_km) for z in zones]
            g = dict(zones=zones, sector=frame(sc.sector),
                     aircraft=to_own_frame(bs.traf.lat[others], bs.traf.lon[others], lat0, lon0, hdg).reshape(-1, 2),
                     aircraft_vel=vel, goal=frame([goal])[0], goal_radius=float(inner.distance_margin),
                     own_lat=lat0, own_lon=lon0, own_hdg=hdg,
                     time_left=float(np.clip(1.0 - t_now / float(inner.episode_time_limit), 0.0, 1.0)),
                     aircraft_ids=[str(c) for c in np.asarray(bs.traf.id)[others]])   # callsigns, same order as aircraft
            self._geom[a] = o[GEOM_KEY] = g

    def render(self):
        return self.env.render()

    # Ray colours per hit type (no-fly zone, sector boundary, other aircraft, own goal) and for "no hit in range".
    RAY_COLORS = [(220, 40, 40), (30, 150, 30), (255, 140, 0), (255, 255, 255)]
    RAY_MISS = (150, 150, 150)

    def render_rays(self, obs, lidar, ray_step=2):
        """Frame with the lidar rays of each live aircraft drawn on the scene (as in navigation-mappo-rl).
        A ray ends at its nearest hit (colour of the hit type) or at the lidar range (grey).
        ray_step: draw every n-th ray, so that the GIF stays readable."""
        from PIL import Image, ImageDraw
        frame = self.render()
        inner = self.env.unwrapped
        img = Image.fromarray(frame)
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for a in self.agents:
            if a not in obs:
                continue
            g = obs[a][GEOM_KEY]
            x0, y0 = inner.projection.project(g["own_lat"], g["own_lon"])
            t, kind = lidar.nearest(obs[a])
            for r in range(0, lidar.n_rays, ray_step):
                hit = kind[r] >= 0
                length = t[r] if hit else lidar.max_range
                # ray angle: 0 = nose, positive = starboard = clockwise, as the aviation heading
                dx, dy = inner.projection.direction(g["own_hdg"] + lidar.angles_deg[r], length)
                col = self.RAY_COLORS[kind[r]] if hit else self.RAY_MISS
                draw.line([(x0, y0), (x0 + dx, y0 + dy)], fill=col + ((200,) if hit else (110,)), width=1)
                if hit:
                    draw.ellipse([x0 + dx - 2, y0 + dy - 2, x0 + dx + 2, y0 + dy + 2], fill=col + (255,))
        return np.asarray(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"))

    def close(self):
        self.env.close()


class ObsBuilder:
    """Frame stack of one aircraft: K frames of (own state, lidar scan), oldest first.
    At reset the older frames are zeros and the first observation is the newest frame
    (supersuit frame_stack_v1, as in the reference)."""

    def __init__(self, lidar, history=4):
        self.lidar, self.k = lidar, history
        self.frames = deque(maxlen=history)

    def reset(self, obs):
        f = (own_state(obs, (0.0, 0.0), self.lidar.zone_flag), self.lidar.scan(obs))
        z = (np.zeros_like(f[0]), np.zeros_like(f[1]))
        self.frames.clear()
        self.frames.extend([z] * (self.k - 1) + [f])
        self.ac = self.lidar.aircraft_tokens(obs) if self.lidar.ac_tokens else None
        self.zone = self.lidar.zone_tokens_vec(obs) if self.lidar.zone_tokens else None
        h = self.lidar.act_tokens                      # last H actions, oldest first; an empty row has flag 0
        self.acts = deque([(0.0, 0.0, 0.0)] * h, maxlen=h) if h else None
        self.edge = self.lidar.edge_tokens_vec(obs) if self.lidar.edge_tokens else None
        self.map = self.lidar.map_vec(obs) if self.lidar.map_px else None

    def push(self, obs, action):
        self.frames.append((own_state(obs, action, self.lidar.zone_flag), self.lidar.scan(obs)))
        self.ac = self.lidar.aircraft_tokens(obs) if self.lidar.ac_tokens else None
        self.zone = self.lidar.zone_tokens_vec(obs) if self.lidar.zone_tokens else None
        self.edge = self.lidar.edge_tokens_vec(obs) if self.lidar.edge_tokens else None
        self.map = self.lidar.map_vec(obs) if self.lidar.map_px else None
        if self.acts is not None:
            self.acts.append((float(action[0]), float(action[1]), 1.0))

    def get(self):
        """state [K * 7] (K * 8 with the inside-zone flag), lidar [K * 3, n_rays] (K * 4 with the speed channel)."""
        state = [f[0] for f in self.frames]
        if self.ac is not None:                        # aircraft tokens of the newest frame, at the end of the state
            state.append(self.ac)
        if self.zone is not None:                      # restricted-area tokens and the goal token of the newest frame
            state.append(self.zone)
        if self.acts is not None:                      # H rows of (turn, speed), then H flags
            a = np.asarray(self.acts, dtype=np.float32)
            state.append(np.concatenate([a[:, :ACT_FEATURES].reshape(-1), a[:, ACT_FEATURES]]))
        if self.edge is not None:                      # sector-edge tokens of the newest frame
            state.append(self.edge)
        if self.map is not None:                       # map block of the newest frame: time left, then the image
            state.append(self.map)
        return (np.concatenate(state), np.concatenate([f[1] for f in self.frames], axis=0))


def lidar_cfg_from_args(args):
    """Lidar and token settings from the flags of scripts.train (also used by the JEPA scripts)."""
    return dict(n_rays=args.n_rays, fov_deg=args.fov_deg, max_range_km=args.max_range_km,
                zone_flag=bool(args.zone_flag), speed=bool(args.lidar_speed), ac_tokens=args.ac_tokens,
                zone_tokens=args.zone_tokens, act_tokens=args.act_tokens,
                edge_tokens=args.edge_tokens, ac_cpa=args.ac_cpa, map_px=args.map_px, map_range_km=args.map_range_km)


def obs_dims(lidar_cfg, history):
    tok = {k: int(lidar_cfg.get(k, 0)) for k in ["ac_tokens", "zone_tokens", "act_tokens", "edge_tokens", "ac_cpa",
                                                 "map_px"]}   # values at the end of the state
    return dict(state_dim=state_dim(lidar_cfg.get("zone_flag", True)) * history + extra_dim(**tok), **tok,
                lidar_ch=(N_CHANNELS + int(bool(lidar_cfg.get("speed", False)))) * history, n_rays=lidar_cfg["n_rays"])


def potential(obs, time_penalty, speed):
    """Shaping potential: time penalty times the estimated seconds to the waypoint at `speed` m/s."""
    return time_penalty * float(np.asarray(obs["waypoint_distance"]).reshape(-1)[0]) * DIST_NORM_KM * 1000.0 / speed


def metrics_row(info, ep_return):
    row = {k: float(info.get(k, 0.0)) for k in METRIC_KEYS}
    row.update({k: float(info[k]) for k in START_KEYS if k in info})     # single agent only
    row["ep_return"] = float(ep_return)
    if "mask_decisions" in info:                       # heading mask on (2026-11-21), not scored
        dec, proj = float(info["mask_decisions"]), float(info["mask_projected"])
        row.update(mask_decisions=dec, mask_projected=proj, mask_fallbacks=float(info["mask_fallbacks"]),
                   mask_proj_share=proj / max(dec, 1.0),
                   mask_mean_angle_deg=float(info["mask_angle_sum_deg"]) / max(proj, 1.0))
    return row


def controller(obs, gain=1.0):
    """Scripted baseline: turn against the drift, no speed change. Gain 1 gives no oscillation."""
    return np.array([np.clip(-gain * float(obs["sin_drift"][0]), -1.0, 1.0), 0.0], dtype=np.float32)


def rollout_episode(env, policy_fn, seed, lidar, history, render=False, max_steps=400, rays=True, on_step=None):
    """One episode. policy_fn(raw obs by id, (state, lidar) by id) -> action by id.
    rays: draw the lidar rays on the frames (render only).
    on_step: called with no argument after the reset and after every step (for example to record tracks).
    Returns (metric rows by id, frames, turn actions by id)."""
    obs = env.reset(seed=seed)
    builders = {a: ObsBuilder(lidar, history) for a in obs}
    for a in obs:
        builders[a].reset(obs[a])
    grab = (lambda o: env.render_rays(o, lidar)) if rays else (lambda o: env.render())
    frames = [grab(obs)] if render else []
    if on_step is not None:
        on_step()
    ret, turns, rows, steps = {a: 0.0 for a in obs}, {a: [] for a in obs}, {}, 0
    while env.agents and steps < max_steps:
        live = list(env.agents)
        acts = policy_fn({a: obs[a] for a in live}, {a: builders[a].get() for a in live})
        obs, rew, term, trunc, infos = env.step(acts)
        steps += 1
        if render:
            frames.append(grab(obs))
        if on_step is not None:
            on_step()
        done_acts = getattr(env, "last_exec", None) or acts     # heading mask on: the executed actions
        for a in live:
            ret[a] += rew[a]
            turns[a].append(float(done_acts[a][0]))
            builders[a].push(obs[a], done_acts[a])
            if term[a] or trunc[a]:
                rows[a] = metrics_row(infos[a], ret[a])
    return rows, frames, turns


def turn_stats(turns):
    """Mean absolute turn (degrees per step at 45 degrees per unit) and the share of sign flips."""
    t = np.asarray(turns, dtype=np.float64)
    s = np.sign(t[np.abs(t) > 1e-3])
    flips = float((s[1:] != s[:-1]).mean()) if len(s) > 1 else 0.0
    return {"mean_abs_turn_deg": float(np.abs(t).mean() * 45.0) if len(t) else 0.0, "turn_sign_flip_rate": flips}


# ------------------------------------------------------------------ pool of environment processes
def _episode_seed(cfg, worker, episode):
    """Scenario seed of a new training flight. narrow (default, as before): 10 000 seeds per worker; a long run
    (more than 10 000 flights per worker) then flies the seeds of the next worker again. wide (2026-11-20, long
    runs): 1 000 000 seeds per worker and 100 000 000 per run seed (run seed <= 20, workers <= 100)."""
    if cfg["reset_seed"] is not None:
        return cfg["reset_seed"]
    if cfg.get("seed_layout", "narrow") == "wide":
        return 1000 + cfg["seed"] * 100_000_000 + worker * 1_000_000 + episode
    return 1000 + cfg["seed"] * 1_000_000 + worker * 10_000 + episode


def clean_flight(info):
    """Clean as in the leaderboard: goal reached, no intrusion, no restricted-area entry, no sector exit."""
    return (float(info["waypoint_reached"]) >= 1.0 and float(info["intrusion_events"]) == 0.0
            and float(info["restricted_area_events"]) == 0.0 and float(info["sector_exit_events"]) == 0.0
            and not info.get("cat_end"))


def _replay_update(pool, seed, info, was_replay, cfg):
    """Pool update after a flight. A start intrusion alone is not a failure to replay (not avoidable)."""
    i = dict(info)
    i["intrusion_events"] = i.get("intrusion_events_no_start", i["intrusion_events"])
    if clean_flight(i):
        pool.pop(seed, None)
        return
    if was_replay:
        if pool.get(seed, 0) >= int(cfg.get("replay_max", 3)):
            pool.pop(seed, None)
        return
    pool[seed] = 0
    while len(pool) > int(cfg.get("replay_cap", 256)):
        pool.pop(next(iter(pool)))


def _worker(conn, cfg, worker):
    import contextlib
    import io
    from .common import step_costs
    with contextlib.redirect_stdout(io.StringIO()):          # BlueSky start-up text
        env = FlightEnv(cfg["setting"], cfg["n_agents"], v2=cfg.get("v2"), cat_penalty=cfg.get("cat_penalty", 0.0),
                        cat_seed=cfg["seed"] * 1000 + worker + 13, fast_metrics=cfg.get("fast_metrics", False),
                        **cfg["reward"])
    lidar, K = Lidar(**cfg["lidar"]), cfg["history"]
    ids = env.ids
    N, dims = len(ids), obs_dims(cfg["lidar"], cfg["history"])
    st = {"obs": None, "builders": None, "ret": None, "episode": 0, "started": False}
    cost_keys = list(cfg.get("cost_keys", []))     # Lagrangian constraints (2026-11-13); empty = off
    C = len(cost_keys)

    def phi(o):
        return potential(o, cfg["reward"]["time_penalty"], cfg["shaping_speed"]) if cfg["shaping"] else 0.0

    # Failure replay (2026-11-19, single agent): the seeds of failed flights go into a pool of this worker; a share
    # replay_frac of the resets flies a pool seed again. A seed leaves the pool after a clean flight or replay_max
    # replays (unavoidable failures). New seeds keep the baseline sequence (the counter moves on new seeds only).
    replay_frac = float(cfg.get("replay_frac", 0.0))
    if replay_frac > 0.0:
        rrng = np.random.default_rng(cfg["seed"] * 1000 + worker + 7)
        pool = {}                                   # seed -> replays so far (insertion order = age)
        st.update(seed=None, replay=False)
    ws = (cfg.get("worker_state") or {}).get(worker)    # resume (2026-11-20): seed counter and replay pool
    if ws:
        st["episode"] = int(ws["episode"])
        if replay_frac > 0.0 and ws.get("pool"):
            pool.update({int(k): int(v) for k, v in ws["pool"].items()})

    def do_reset():
        if replay_frac > 0.0 and pool and rrng.random() < replay_frac:
            seed = list(pool)[int(rrng.integers(len(pool)))]
            pool[seed] += 1
            st.update(seed=seed, replay=True)
        else:
            seed = _episode_seed(cfg, worker, st["episode"])
            st["episode"] += 1
            if replay_frac > 0.0:
                st.update(seed=seed, replay=False)
        st["obs"] = env.reset(seed=seed)
        st["builders"] = {a: ObsBuilder(lidar, K) for a in st["obs"]}
        for a in st["obs"]:
            st["builders"][a].reset(st["obs"][a])
        st["ret"] = {a: 0.0 for a in ids}
        st["cprev"] = {a: np.zeros(C) for a in ids}   # cumulative constraint metrics of the flight so far

    def pack(agents):
        state = np.zeros((N, dims["state_dim"]), dtype=np.float32)
        lid = np.zeros((N, dims["lidar_ch"], dims["n_rays"]), dtype=np.float32)
        alive = np.zeros(N, dtype=bool)
        for a in agents:
            i = ids.index(a)
            state[i], lid[i] = st["builders"][a].get()
            alive[i] = True
        return state, lid, alive

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "ids":
                conn.send(ids)
            elif cmd == "reset":
                # The reference resets at the start of every rollout. Aircraft still flying are cut (not logged).
                cut = len(env.agents) if st["started"] else 0
                do_reset()
                st["started"] = True
                conn.send((pack(env.agents), cut))
            elif cmd == "step":
                # black_death semantics of the reference: a finished aircraft gets its terminal observation
                # once, then a zero observation and reward 0. The done flag is set only when the whole
                # environment ends (all aircraft finished or time limit), for every aircraft.
                live = list(env.agents)
                acts = {a: np.asarray(data[ids.index(a)], dtype=np.float32) for a in live}
                nobs, rew, te, tr, infos = env.step(acts)
                has_exec = env.last_exec is not None    # heading mask on (env.reset below clears last_exec)
                if has_exec:                      # history and learner get the executed actions
                    ex_all = np.asarray(data, dtype=np.float32).copy()
                    proj = np.zeros(N, dtype=bool)
                    for a in live:
                        i = ids.index(a)
                        proj[i] = float(env.last_exec[a][0]) != float(acts[a][0])
                        ex_all[i] = env.last_exec[a]
                        acts[a] = env.last_exec[a]
                r, p, pn = np.zeros(N, np.float32), np.zeros(N, np.float32), np.zeros(N, np.float32)
                episodes = []
                cost = np.zeros((N, C), np.float32)
                for a in live:
                    i = ids.index(a)
                    if C:
                        cost[i], st["cprev"][a] = step_costs(infos[a], st["cprev"][a], cost_keys)
                    r[i], p[i] = rew[a], phi(st["obs"][a])
                    pn[i] = 0.0 if te[a] else phi(nobs[a])
                    st["ret"][a] += rew[a]
                    st["builders"][a].push(nobs[a], acts[a])
                    if te[a] or tr[a]:
                        row = metrics_row(infos[a], st["ret"][a])
                        if "cat_end" in infos[a]:
                            row["cat_end"] = float(infos[a]["cat_end"])
                        if replay_frac > 0.0:
                            row["replay"] = float(st["replay"])
                            _replay_update(pool, st["seed"], infos[a], st["replay"], cfg)
                        episodes.append(row)
                st["obs"] = nobs
                done = not env.agents
                if done:
                    do_reset()
                    cur = pack(env.agents)
                else:
                    cur = pack(live)          # terminal observation of an aircraft that finished at this step
                msg = dict(cur=cur, rew=r, phi=p, phi_next=pn, done=np.full(N, done), episodes=episodes)
                if has_exec:
                    msg["exec"], msg["proj"] = ex_all, proj
                if C:
                    msg["cost"] = cost
                conn.send(msg)
            elif cmd == "cat_prob":               # random CaT: effective from the next step
                env.set_cat_prob(data)
                conn.send(True)
            elif cmd == "density":                # curriculum: effective from the next reset
                env.set_density(*data)
                conn.send(True)
            elif cmd == "snapshot":               # resume state of this worker (2026-11-20)
                conn.send({"episode": st["episode"], "pool": dict(pool) if replay_frac > 0.0 else None})
            elif cmd == "close":
                env.close()
                conn.close()
                return
    except (EOFError, KeyboardInterrupt):
        pass


class VecEnv:
    """E environment processes, stepped in lock-step. Arrays have the shape [E, N, ...]."""

    def __init__(self, cfg, n_envs):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(n_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, cfg, w), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.conns[0].send(("ids", None))
        self.ids = self.conns[0].recv()
        self.n_envs, self.n_agents = n_envs, len(self.ids)

    @staticmethod
    def _stack(packs):
        return tuple(np.stack(x) for x in zip(*packs))

    def reset(self):
        """Returns (state, lidar, alive) and the count of aircraft-episodes cut by this reset."""
        for c in self.conns:
            c.send(("reset", None))
        res = [c.recv() for c in self.conns]
        return self._stack([r[0] for r in res]), sum(r[1] for r in res)

    def step(self, actions):
        """actions [E, N, 2] -> dict of arrays [E, N, ...] and the list of finished episodes."""
        for c, a in zip(self.conns, actions):
            c.send(("step", a))
        res = [c.recv() for c in self.conns]
        out = {k: np.stack([r[k] for r in res]) for k in ["rew", "phi", "phi_next", "done"]}
        out["cur"] = self._stack([r["cur"] for r in res])
        out["episodes"] = [e for r in res for e in r["episodes"]]
        if "cost" in res[0]:                           # Lagrangian: [E, N, C] cost of this step
            out["cost"] = np.stack([r["cost"] for r in res])
        if "exec" in res[0]:                           # heading mask: executed actions [E, N, 2], projected [E, N]
            out["exec"], out["proj"] = np.stack([r["exec"] for r in res]), np.stack([r["proj"] for r in res])
        return out

    def set_cat_prob(self, p):
        """Random CaT: all workers use the new probability from their next step on."""
        for c in self.conns:
            c.send(("cat_prob", float(p)))
        for c in self.conns:
            c.recv()

    def set_density(self, n_intruders, n_obstacles):
        """Curriculum: all workers use the new counts from their next reset on."""
        for c in self.conns:
            c.send(("density", (n_intruders, n_obstacles)))
        for c in self.conns:
            c.recv()

    def snapshot(self):
        """Resume state of the workers {worker: {"episode", "pool"}} (2026-11-20). The random generators of the
        workers are not saved, so a resumed run is not bit-identical to a run without a stop."""
        for c in self.conns:
            c.send(("snapshot", None))
        return {w: c.recv() for w, c in enumerate(self.conns)}

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
