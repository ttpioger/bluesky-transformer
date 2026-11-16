"""Time vs intrusion: does the policy buy flight time with intrusions?

This script reads result files that exist already. It runs no simulation and no training.
It does not import or change the training code.

Inputs (folder --results-dir):
  RUN_summary.json            training flights (key "episodes"), greedy flights, reward weights (key "args")
  RUN_competition_sa.json     1000 flights of the seed-42 stream (scripts.competition_eval), one row per flight
  ALT_competition_sa.json     optional (--alt): other policies on the SAME seed-42 stream, paired by episode_index
  ALT_summary.json            optional: reward weights of ALT (to score ALT with the reward of RUN)
  competition_straight_sa_1000_seed42.json   optional: fly-straight means (no rows)
  --checkpoint-evals STEP=FILE  optional: competition files of saved checkpoints, for the trend across checkpoints

Outputs (folder --out-dir):
  flights.csv            one row per flight: metrics, return, return per reward term (reward of RUN)
  stats.json             all numbers below
  numbers.md             tables of the numbers (auto-generated; report.md is written by hand)
  fig_scatter.png, fig_trend.png, fig_decomposition.png, fig_checkpoints.png (only with --checkpoint-evals)
  W&B (--wandb online|offline): the same numbers, tables, figures and files, in a run named RUN_time_vs_intrusion
  (group RUN, job_type analysis).

Reward of this repo (rl_tf/envs.py FlightEnv.step + bluesky_gym CompetitionEnv._get_reward): each term is
weight x increase of one flight metric, except the heading, drift, goal-distance and turn-change terms. So the
return of a flight splits exactly into weight x final metric per term, plus one rest term
r_other = ep_return - sum(terms). With drift, goal-distance and turn-change weights at 0 (all los runs),
r_other is the heading term alone.

Run from bluesky_rl_transformer/:
  ../.venv/bin/python analysis/time_vs_intrusion/time_vs_intrusion.py --run sa_tf_tok3_s0 \
      --alt sa_tf_info_ff_cls_big_s0 --wandb online
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

# (column, weight key in args, metric key, when it applies). Source: rl_tf/envs.py REWARD_HELP and FlightEnv.step,
# bluesky_gym/envs/competition_env.py _get_reward and _update_metrics.
TERMS = [
    ("r_time", "time_penalty", "flight_time", "per simulated second flown"),
    ("r_intrusion_time", "intrusion_penalty", "intrusion_time",
     "per second and per other aircraft within 5 NM (metric sampled every 1 s)"),
    ("r_intrusion_event", "intrusion_event_penalty", "intrusion_events",
     "once per start of an intrusion with one aircraft (also an intrusion already present at reset)"),
    ("r_restricted_time", "restricted_penalty", "time_in_restricted_area",
     "per second inside >= 1 restricted area (one count also inside 2 areas)"),
    ("r_zone_entry", "zone_entry_penalty", "restricted_area_events", "once per entry into a restricted area"),
    ("r_outside_time", "sector_exit_penalty", "time_outside_sector", "per second outside the sector"),
    ("r_sector_exit_event", "sector_exit_event_penalty", "sector_exit_events", "once per exit from the sector"),
]
REST_KEYS = ["heading_penalty", "drift_penalty", "goal_distance_penalty", "turn_change_penalty"]
REST_HELP = {"heading_penalty": "per second flown x (1 - cos psi), psi = heading vs bearing to goal (after the step)",
             "drift_penalty": "per radian of drift, per 10 s step (env term)",
             "goal_distance_penalty": "per second flown x km to the goal",
             "turn_change_penalty": "per unit of change of the turn action between two steps"}
METRICS = ["waypoint_reached", "flight_time", "intrusion_time", "intrusion_events", "time_in_restricted_area",
           "restricted_area_events", "time_outside_sector", "sector_exit_events"]
STEP_SECONDS = 10          # action_frequency of CompetitionEnv (one policy step = 10 simulated seconds)
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"   # dataviz reference palette, categorical slots 1 and 2


# ------------------------------------------------------------------ statistics
def rankdata(a):
    """Average ranks (ties share the mean rank), 1-based."""
    u, inv, cnt = np.unique(np.asarray(a, dtype=np.float64), return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]


def spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def boot_ci(stat, arrays, n_boot, rng, level=0.95):
    """Percentile bootstrap over flights (rows resampled together). Returns (estimate, low, high)."""
    n = len(arrays[0])
    est = stat(*arrays)
    if n < 3 or not np.isfinite(est):
        return est, float("nan"), float("nan")
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = stat(*(a[i] for a in arrays))
    vals = vals[np.isfinite(vals)]
    if len(vals) < n_boot // 2:
        return est, float("nan"), float("nan")
    q = (1 - level) / 2
    return float(est), float(np.quantile(vals, q)), float(np.quantile(vals, 1 - q))


def mean_ci(x, n_boot, rng):
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return dict(n=0, mean=float("nan"), lo=float("nan"), hi=float("nan"))
    i = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[i].mean(1)
    return dict(n=int(len(x)), mean=float(x.mean()), lo=float(np.quantile(m, 0.025)), hi=float(np.quantile(m, 0.975)))


def corr_block(rows, n_boot, rng, x="flight_time", y="intrusion_time"):
    xs = np.array([r[x] for r in rows], np.float64)
    ys = np.array([r[y] for r in rows], np.float64)
    rho, lo, hi = boot_ci(spearman, [xs, ys], n_boot, rng)
    return dict(n=len(rows), spearman=rho, ci_lo=lo, ci_hi=hi,
                status="REPORTABLE-size" if len(rows) >= 200 else "EXPLORATORY (n < 200)")


# ------------------------------------------------------------------ reward terms
def weights_of(args):
    keys = [w for _, w, _, _ in TERMS] + REST_KEYS
    missing = [k for k in keys if args.get(k) is None]
    if missing:
        raise ValueError(f"reward weights missing in args: {missing}")
    return {k: float(args[k]) for k in keys}


def split_terms(row, w_run, w_src):
    """Return per term under the reward of RUN (w_run). The flight was produced under w_src (ep_return in w_src).
    r_other: the rest (heading, drift, goal distance, turn change) rescaled to w_run when the rest of w_src is the
    heading term alone; NaN when it cannot be rescaled."""
    out = {col: w_run[wk] * float(row.get(mk, 0.0)) for col, wk, mk, _ in TERMS}
    known_src = sum(w_src[wk] * float(row.get(mk, 0.0)) for _, wk, mk, _ in TERMS)
    rest_src = float(row["ep_return"]) - known_src
    same_rest = all(w_run[k] == w_src[k] for k in REST_KEYS)
    only_heading = all(w_src[k] == 0.0 and w_run[k] == 0.0 for k in REST_KEYS if k != "heading_penalty")
    if same_rest:
        out["r_other"] = rest_src
    elif only_heading and w_src["heading_penalty"] != 0.0:
        out["r_other"] = rest_src * w_run["heading_penalty"] / w_src["heading_penalty"]
    elif only_heading and w_run["heading_penalty"] == 0.0:
        out["r_other"] = 0.0
    else:
        out["r_other"] = float("nan")
    out["rest_src"] = rest_src
    out["ep_return_original"] = sum(out[c] for c, _, _, _ in TERMS) + out["r_other"]
    return out


TERM_COLS = [c for c, _, _, _ in TERMS] + ["r_other"]


# ------------------------------------------------------------------ loading
def load_json(path):
    return json.loads(Path(path).read_text())


def flight_rows(run, res, w_run, early_until, late_from):
    """All flights of RUN: training, greedy, competition. Each row gets the per-term split."""
    s = load_json(res / f"{run}_summary.json")
    rows = []
    for e in s["episodes"]:
        st = int(e["env_steps"])
        phase = "early" if st < early_until else ("late" if st >= late_from else "middle")
        rows.append(dict(source="train", policy=run, phase=phase, env_steps=st, episode_index="", episode_seed="",
                         **{k: float(e[k]) for k in METRICS}, ep_return=float(e["ep_return"])))
    for e in s.get("greedy_eval", {}).get("rows", []):
        rows.append(dict(source="greedy_eval", policy=run, phase="end", env_steps=int(s["env_steps"]), episode_index="",
                         episode_seed=e.get("episode_seed", ""), **{k: float(e[k]) for k in METRICS},
                         ep_return=float(e["ep_return"])))
    comp = res / f"{run}_competition_sa.json"
    c = load_json(comp) if comp.exists() else None
    if c is not None:
        for e in c["rows"]:
            rows.append(dict(source="competition", policy=run, phase="end", env_steps=int(s["env_steps"]),
                             episode_index=int(e["episode_index"]), episode_seed=c["seed"],
                             **{k: float(e[k]) for k in METRICS}, ep_return=float(e["ep_return"])))
    for r in rows:
        r.update(split_terms(r, w_run, w_run))
        r["has_intrusion"] = int(r["intrusion_time"] > 0 or r["intrusion_events"] > 0)
    return s, c, rows


def alt_rows(name, res, w_run, run_comp, source="competition_alt", path=None):
    p = Path(path) if path else res / f"{name}_competition_sa.json"
    c = load_json(p)
    if run_comp is not None and (c["seed"] != run_comp["seed"] or c["episodes"] != run_comp["episodes"]):
        raise ValueError(f"{p}: seed/episodes {c['seed']}/{c['episodes']} differ from RUN "
                         f"{run_comp['seed']}/{run_comp['episodes']}; the flights are not paired")
    sp = res / f"{name}_summary.json"
    w_src, w_note = w_run, "weights of RUN assumed (no summary file)"
    if sp.exists():
        try:
            w_src, w_note = weights_of(load_json(sp)["args"]), "weights from its summary"
        except ValueError as err:
            w_note = f"weights of RUN assumed ({err})"
    rows = []
    for e in c["rows"]:
        r = dict(source=source, policy=name, phase="end", env_steps="", episode_index=int(e["episode_index"]),
                 episode_seed=c["seed"], **{k: float(e[k]) for k in METRICS}, ep_return=float(e["ep_return"]))
        r.update(split_terms(r, w_run, w_src))
        r["has_intrusion"] = int(r["intrusion_time"] > 0 or r["intrusion_events"] > 0)
        rows.append(r)
    return c, rows, w_src, w_note


# ------------------------------------------------------------------ analyses
def audit(w, args, comp_rows):
    """Step 1 and 2: reward terms and exchange rates, from the weights of RUN only (plus one data value)."""
    terms = [dict(term=col, weight=w[wk], metric=mk, applies=when, kind="per event" if "event" in mk else "per second")
             for col, wk, mk, when in TERMS]
    terms += [dict(term=k, weight=w[k], metric="(not a flight metric)", applies=REST_HELP[k], kind="per step")
              for k in REST_KEYS]
    wt, wi, we, wh = w["time_penalty"], w["intrusion_penalty"], w["intrusion_event_penalty"], w["heading_penalty"]
    rate = dict(
        flight_s_per_intrusion_s=wi / wt if wt else float("inf"),
        flight_s_per_intrusion_event=we / wt if wt else float("inf"),
        note="flight seconds that cost the same reward as one intrusion second / one intrusion event")
    intr = [r for r in comp_rows if r["intrusion_events"] > 0]
    if intr:                                   # mean length of one intrusion in the competition flights of RUN
        L = sum(r["intrusion_time"] for r in intr) / sum(r["intrusion_events"] for r in intr)
        rate["mean_intrusion_s_per_event_competition"] = L
        rate["flight_s_per_mean_intrusion"] = (wi * L + we) / wt if wt else float("inf")
    # heading term during a detour: extra cost per flown second at angle psi to the goal bearing
    rate["detour_cost_per_s_at_psi"] = {str(p): wt + wh * (1 - math.cos(math.radians(p))) for p in (0, 30, 60, 90)}
    gamma = float(args.get("gamma", 0.99))
    # Avoid an intrusion now -> pay the extra flight seconds k steps later (end of the flight): discounted by gamma^k.
    rate["discounted_flight_s_per_intrusion_s"] = {str(k): (wi / wt) / gamma ** k for k in (0, 25, 50, 100)}
    scale = float(args.get("reward_scale", 1.0))
    eff = dict(reward_scale=scale, gamma=gamma, step_seconds=STEP_SECONDS,
               reward_clipping="none (train.py: B_rew = (rew + gamma*phi' - phi) * reward_scale)",
               reward_normalisation="none", advantage_normalisation="none (train.py: 'no advantage normalization')",
               shaping=args.get("shaping"), value_clipping="none", ppo_ratio_clip=args.get("clip"),
               max_grad_norm=args.get("max_grad_norm"),
               note="reward_scale multiplies every term, so it does not change the exchange rate. The PPO ratio "
                    "clip and the gradient-norm clip limit the size of one update; they do not change the rate "
                    "at the optimum, but they slow the learning from rare, large penalties.")
    return dict(terms=terms, exchange_rate=rate, effective=eff)


def locate(rows_train, rows_greedy, comp_by_policy, bins, n_boot, rng, s):
    """Where is 'intrusion up while flight time down'? Training bins, windows, end evaluations."""
    steps = np.array([r["env_steps"] for r in rows_train])
    edges = np.linspace(0, max(steps.max(), 1), bins + 1)
    out_bins = []
    for i in range(bins):
        m = (steps > edges[i]) & (steps <= edges[i + 1]) if i else (steps <= edges[1])
        sel = [r for r, k in zip(rows_train, m) if k]
        if not sel:
            continue
        b = dict(env_steps_lo=float(edges[i]), env_steps_hi=float(edges[i + 1]), n=len(sel))
        for k in ["flight_time", "intrusion_time", "intrusion_events", "waypoint_reached", "ep_return"]:
            b[k] = mean_ci([r[k] for r in sel], n_boot, rng)
        b["corr"] = corr_block(sel, n_boot, rng)
        out_bins.append(b)
    flags = []
    for a, b in zip(out_bins[:-1], out_bins[1:]):
        dt = b["flight_time"]["mean"] - a["flight_time"]["mean"]
        di = b["intrusion_time"]["mean"] - a["intrusion_time"]["mean"]
        if dt < 0 and di > 0:
            sig = b["intrusion_time"]["lo"] > a["intrusion_time"]["hi"] or b["flight_time"]["hi"] < a["flight_time"]["lo"]
            flags.append(dict(from_hi=a["env_steps_hi"], to_hi=b["env_steps_hi"], d_flight_time=dt, d_intrusion_time=di,
                              cis_separate=bool(sig)))
    win = {k: s.get(k) for k in ["first_window", "last_window"]}
    ends = {p: {k: float(np.mean([r[k] for r in rr])) for k in ["flight_time", "intrusion_time", "intrusion_events",
                                                                  "waypoint_reached", "ep_return_original"]}
            for p, rr in comp_by_policy.items()}
    greedy = {k: float(np.mean([r[k] for r in rows_greedy])) for k in ["flight_time", "intrusion_time"]} if rows_greedy else {}
    return dict(train_bins=out_bins, pattern_between_bins=flags, windows=win, greedy_eval=greedy,
                competition_means=ends)


def paired(run_rows, alt_rows_, n_boot, rng):
    """Same scenario (episode_index), RUN vs ALT. Delta = ALT - RUN."""
    a = {r["episode_index"]: r for r in alt_rows_}
    pairs = [(r, a[r["episode_index"]]) for r in run_rows if r["episode_index"] in a]
    d = {k: np.array([q[k] - p[k] for p, q in pairs], np.float64)
         for k in ["flight_time", "intrusion_time", "intrusion_events", "ep_return_original"] + TERM_COLS}
    rho = boot_ci(spearman, [d["flight_time"], d["intrusion_time"]], n_boot, rng)
    both = (d["intrusion_time"] < 0) & (d["flight_time"] > 0)
    out = dict(n_pairs=len(pairs), spearman_dtime_dintrusion=dict(zip(["rho", "lo", "hi"], rho)),
               share_alt_safer_and_slower=float(both.mean()) if len(pairs) else float("nan"),
               share_alt_less_intrusion=float((d["intrusion_time"] < 0).mean()) if len(pairs) else float("nan"),
               share_alt_more_intrusion=float((d["intrusion_time"] > 0).mean()) if len(pairs) else float("nan"),
               delta_mean={k: mean_ci(v[np.isfinite(v)], n_boot, rng) for k, v in d.items()},
               note="Delta = ALT - RUN on the same scenario. Score: reward of RUN (ep_return_original). "
                    "A trade-off (1) predicts: where ALT has less intrusion it flies longer (negative rho), and its "
                    "score loss comes from r_time / r_other.")
    return out, d


def decomposition(rows, n_boot, rng):
    out = {}
    for flag, name in [(1, "with_intrusion"), (0, "without_intrusion")]:
        sel = [r for r in rows if r["has_intrusion"] == flag]
        out[name] = {c: mean_ci([r[c] for r in sel if np.isfinite(r[c])], n_boot, rng)
                     for c in TERM_COLS + ["ep_return_original", "flight_time", "intrusion_time"]}
    return out


# ------------------------------------------------------------------ figures
def _style(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="#e6e6e3", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_scatter(path, run, comp_run, late, pair_d, alt_name, st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 3 if pair_d is not None else 2
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 4.6))
    panels = [(comp_run, "competition", f"(a) {run}: 1000 flights, seed-42 stream, no noise"),
              (late, "train_late", f"(b) {run}: training flights, late phase (with action noise)")]
    for ax, (rows, key, title) in zip(axs, panels):
        if rows:
            ax.scatter([r["flight_time"] for r in rows], [r["intrusion_time"] for r in rows], s=8, alpha=0.35,
                       color=BLUE, edgecolors="none", label="one flight")
            c = st.get(key)
            if c:
                title += f"\nSpearman {c['spearman']:.2f} [95% CI {c['ci_lo']:.2f}, {c['ci_hi']:.2f}], n={c['n']}"
        ax.set_xlabel("flight time (s)")
        ax.set_ylabel("intrusion time (s, summed over aircraft)")
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        _style(ax)
    if pair_d is not None:
        ax = axs[2]
        ax.axhline(0, color=GREY, linewidth=0.8)
        ax.axvline(0, color=GREY, linewidth=0.8)
        ax.scatter(pair_d["flight_time"], pair_d["intrusion_time"], s=8, alpha=0.35, color=ORANGE, edgecolors="none",
                   label="one scenario")
        p = st.get("paired")
        t = f"(c) same scenario: {alt_name} minus {run}"
        if p:
            r = p["spearman_dtime_dintrusion"]
            t += f"\nSpearman {r['rho']:.2f} [95% CI {r['lo']:.2f}, {r['hi']:.2f}], n={p['n_pairs']}"
        ax.set_title(t, fontsize=9, loc="left")
        ax.set_xlabel("difference in flight time (s)")
        ax.set_ylabel("difference in intrusion time (s)")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        _style(ax)
    cap = ("Figure. Flight time against intrusion time, one point per flight. A time-for-safety trade-off shows as a "
           "negative correlation (shorter flights with more intrusion). In (a) and (b) the scenarios differ between "
           "points, so traffic density can make both values rise together; (c) removes this: each point is one scenario "
           "flown by two policies. One training seed; exploratory.")
    fig.text(0.01, 0.005, cap, fontsize=7.5, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_trend(path, run, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array([(b["env_steps_lo"] + b["env_steps_hi"]) / 2e6 for b in bins])
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 8.2), sharex=True)
    for ax, k, lab in [(axs[0], "flight_time", "mean flight time (s)"),
                       (axs[1], "intrusion_time", "mean intrusion time (s)")]:
        m = np.array([b[k]["mean"] for b in bins])
        lo, hi = np.array([b[k]["lo"] for b in bins]), np.array([b[k]["hi"] for b in bins])
        ax.fill_between(x, lo, hi, color=BLUE, alpha=0.18, linewidth=0, label="95% bootstrap CI")
        ax.plot(x, m, color=BLUE, linewidth=2, marker="o", markersize=4, label="mean per bin")
        ax.set_ylabel(lab)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        _style(ax)
    ax = axs[2]
    r = np.array([b["corr"]["spearman"] for b in bins])
    lo, hi = np.array([b["corr"]["ci_lo"] for b in bins]), np.array([b["corr"]["ci_hi"] for b in bins])
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.fill_between(x, lo, hi, color=ORANGE, alpha=0.18, linewidth=0, label="95% bootstrap CI")
    ax.plot(x, r, color=ORANGE, linewidth=2, marker="o", markersize=4, label="Spearman per bin")
    ax.set_ylabel("Spearman (flight time,\nintrusion time)")
    ax.set_xlabel("environment steps (millions)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _style(ax)
    axs[0].set_title(f"{run}: training flights in {len(bins)} equal bins of environment steps", fontsize=9, loc="left")
    cap = ("Figure. Training trend. Top, middle: mean flight time and mean intrusion time per bin of training flights "
           "(stochastic policy). Bottom: rank correlation inside each bin; below 0 = shorter flights have more "
           "intrusion. Flights cut at a rollout end are not in the log, so long flights are under-represented. "
           "One seed; exploratory.")
    fig.text(0.01, 0.005, cap, fontsize=7.5, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_decomposition(path, run, dec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = TERM_COLS
    y = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for off, key, col, lab in [(-0.2, "with_intrusion", ORANGE, "flights with an intrusion"),
                               (0.2, "without_intrusion", BLUE, "flights without an intrusion")]:
        m = np.array([dec[key][c]["mean"] for c in cols])
        lo = np.array([dec[key][c]["lo"] for c in cols])
        hi = np.array([dec[key][c]["hi"] for c in cols])
        n = dec[key]["flight_time"]["n"]
        ax.barh(y + off, m, height=0.38, color=col, label=f"{lab} (n={n})")
        ax.errorbar(m, y + off, xerr=[m - lo, hi - m], fmt="none", ecolor="#3b3b38", elinewidth=0.8, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(cols)
    ax.axvline(0, color=GREY, linewidth=0.8)
    ax.set_xlabel("mean return per flight (reward units, reward of the run, unscaled)")
    ax.set_title(f"{run}: return per reward term, competition flights (seed-42 stream)", fontsize=9, loc="left")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    _style(ax)
    cap = ("Figure. Mean return per reward term (all terms are <= 0), with 95% bootstrap CI. r_other is the heading "
           "term (the rest of the return after the metric terms). If intrusions buy time, r_time is less negative in "
           "flights with an intrusion.")
    fig.text(0.01, 0.005, cap, fontsize=7.5, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_checkpoints(path, run, ck):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array([c["env_steps"] / 1e6 for c in ck])
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    for ax, k, lab in [(axs[0], "flight_time", "mean flight time (s)"), (axs[1], "intrusion_time", "mean intrusion time (s)")]:
        m = np.array([c[k]["mean"] for c in ck])
        ax.fill_between(x, [c[k]["lo"] for c in ck], [c[k]["hi"] for c in ck], color=BLUE, alpha=0.18, linewidth=0,
                        label="95% bootstrap CI")
        ax.plot(x, m, color=BLUE, linewidth=2, marker="o", markersize=5, label="checkpoint, same scenarios, no noise")
        ax.set_ylabel(lab)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        _style(ax)
    axs[1].set_xlabel("environment steps of the checkpoint (millions)")
    axs[0].set_title(f"{run}: saved checkpoints on the same scenario stream", fontsize=9, loc="left")
    fig.text(0.01, 0.005, "Figure. Each point: one checkpoint flown without action noise on the same scenarios. "
             "Intrusion rising while flight time falls supports a trade-off. One seed; exploratory.", fontsize=7.5,
             wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ output
CSV_COLS = (["source", "policy", "phase", "env_steps", "episode_index", "episode_seed"] + METRICS +
            ["ep_return", "ep_return_original"] + TERM_COLS + ["has_intrusion"])


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def fmt_ci(d, nd=1):
    return f"{d['mean']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"


def numbers_md(st, run):
    L = [f"# Numbers: time vs intrusion, {run}", "",
         f"Auto-generated by time_vs_intrusion.py on {st['created']}. Status: EXPLORATORY (one training seed). "
         "Values: mean [95% bootstrap CI].", "", "## Reward terms (reward of the run)", "",
         "| term | weight | metric | applies |", "|---|---|---|---|"]
    L += [f"| {t['term']} | {t['weight']} | {t['metric']} | {t['applies']} |" for t in st["audit"]["terms"]]
    r = st["audit"]["exchange_rate"]
    L += ["", "## Exchange rate", "", f"- 1 s of intrusion = {r['flight_s_per_intrusion_s']:.1f} s of flight time.",
          f"- 1 intrusion event = {r['flight_s_per_intrusion_event']:.1f} s of flight time."]
    if "flight_s_per_mean_intrusion" in r:
        L.append(f"- 1 mean intrusion ({r['mean_intrusion_s_per_event_competition']:.1f} s long) = "
                 f"{r['flight_s_per_mean_intrusion']:.0f} s of flight time.")
    L.append("- Discounted (extra flight seconds paid k steps later): " +
             ", ".join(f"k={k}: {v:.1f}" for k, v in r["discounted_flight_s_per_intrusion_s"].items()))
    L += ["", "## Where: training bins", "", "| env steps (M) | n | flight time (s) | intrusion time (s) | Spearman |",
          "|---|---|---|---|---|"]
    for b in st["locate"]["train_bins"]:
        c = b["corr"]
        L.append(f"| {b['env_steps_lo'] / 1e6:.2f}-{b['env_steps_hi'] / 1e6:.2f} | {b['n']} | {fmt_ci(b['flight_time'], 0)} | "
                 f"{fmt_ci(b['intrusion_time'])} | {c['spearman']:.2f} [{c['ci_lo']:.2f}, {c['ci_hi']:.2f}] |")
    fl = st["locate"]["pattern_between_bins"]
    L += ["", f"Bin pairs with flight time down and intrusion time up: {len(fl)}; with separate CIs: "
          f"{sum(f['cis_separate'] for f in fl)}."]
    L += ["", "## Correlation per source", "", "| source | n | Spearman [95% CI] | status |", "|---|---|---|---|"]
    for k, c in st["correlations"].items():
        L.append(f"| {k} | {c['n']} | {c['spearman']:.3f} [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}] | {c['status']} |")
    for name, p in st.get("paired", {}).items():
        r = p["spearman_dtime_dintrusion"]
        L += ["", f"## Paired: {name} minus {run} (same scenarios)", "",
              f"- n pairs {p['n_pairs']}; Spearman(d time, d intrusion) {r['rho']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]",
              f"- share where {name} has less intrusion: {p['share_alt_less_intrusion']:.3f}; more: "
              f"{p['share_alt_more_intrusion']:.3f}; less intrusion AND longer flight: {p['share_alt_safer_and_slower']:.3f}",
              "", "| delta (ALT - RUN) | mean [95% CI] |", "|---|---|"]
        L += [f"| {k} | {fmt_ci(v, 2)} |" for k, v in p["delta_mean"].items()]
    L += ["", "## Decomposition (competition flights of the run)", "", "| term | with intrusion | without intrusion |",
          "|---|---|---|"]
    d = st["decomposition"]
    for k in TERM_COLS + ["ep_return_original", "flight_time", "intrusion_time"]:
        L.append(f"| {k} | {fmt_ci(d['with_intrusion'][k], 2)} | {fmt_ci(d['without_intrusion'][k], 2)} |")
    L += ["", f"Reconstruction check (RUN flights): max rest term {st['checks']['max_rest_term']:.2e} "
          f"(must be <= 0 up to float error); flights with rest > 1e-6: {st['checks']['n_rest_positive']}."]
    return "\n".join(L) + "\n"


def to_wandb(args, st, rows, figs, files):
    import wandb
    run = args.run
    wb = wandb.init(project=args.wandb_project, name=f"{run}_time_vs_intrusion", id=f"{run}_time_vs_intrusion",
                    group=run, job_type="analysis", resume="allow", mode=args.wandb,
                    config=dict(run=run, alt=args.alt, n_boot=args.n_boot, bins=args.bins, early_until=args.early_until,
                                late_from=args.late_from, weights=st["weights"]))
    # trend: its own x axis, so that it never clashes with the step of other runs
    wb.define_metric("trend/env_steps")
    wb.define_metric("trend/*", step_metric="trend/env_steps")
    for b in st["locate"]["train_bins"]:
        d = {"trend/env_steps": (b["env_steps_lo"] + b["env_steps_hi"]) / 2}
        for k in ["flight_time", "intrusion_time", "intrusion_events", "waypoint_reached", "ep_return"]:
            d.update({f"trend/{k}_mean": b[k]["mean"], f"trend/{k}_lo": b[k]["lo"], f"trend/{k}_hi": b[k]["hi"]})
        d.update({"trend/spearman": b["corr"]["spearman"], "trend/spearman_lo": b["corr"]["ci_lo"],
                  "trend/spearman_hi": b["corr"]["ci_hi"], "trend/n": b["n"]})
        wb.log(d)
    if st.get("checkpoints"):
        wb.define_metric("ckpt/env_steps")
        wb.define_metric("ckpt/*", step_metric="ckpt/env_steps")
        for c in st["checkpoints"]:
            wb.log({"ckpt/env_steps": c["env_steps"], **{f"ckpt/{k}_mean": c[k]["mean"] for k in
                                                         ["flight_time", "intrusion_time", "ep_return_original"]}})
    # per-flight tables: full competition tables; training flights subsampled for the plot only (full CSV = artifact)
    rng = np.random.default_rng(0)
    for src in sorted({r["source"] for r in rows}):
        sel = [r for r in rows if r["source"] == src]
        if len(sel) > args.wandb_max_points:
            sel = [sel[i] for i in sorted(rng.choice(len(sel), args.wandb_max_points, replace=False))]
        t = wandb.Table(columns=CSV_COLS, data=[[r.get(k, "") for k in CSV_COLS] for r in sel])
        wb.log({f"flights/{src}": t,
                f"scatter/{src}": wandb.plot.scatter(t, "flight_time", "intrusion_time",
                                                     title=f"{run} {src}: flight time vs intrusion time (s)")})
    s = {}
    for k, v in st["audit"]["exchange_rate"].items():
        if isinstance(v, dict):
            s.update({f"rate/{k}/{kk}": vv for kk, vv in v.items()})
        elif isinstance(v, (int, float)):
            s[f"rate/{k}"] = v
    for k, c in st["correlations"].items():
        s.update({f"corr/{k}/spearman": c["spearman"], f"corr/{k}/lo": c["ci_lo"], f"corr/{k}/hi": c["ci_hi"],
                  f"corr/{k}/n": c["n"]})
    for name, p in st.get("paired", {}).items():
        r = p["spearman_dtime_dintrusion"]
        s.update({f"paired/{name}/spearman": r["rho"], f"paired/{name}/lo": r["lo"], f"paired/{name}/hi": r["hi"],
                  f"paired/{name}/share_safer_and_slower": p["share_alt_safer_and_slower"]})
        s.update({f"paired/{name}/delta_{k}": v["mean"] for k, v in p["delta_mean"].items()})
    for g, d in st["decomposition"].items():
        s.update({f"decomp/{g}/{k}": v["mean"] for k, v in d.items()})
    s["pattern_between_bins"] = len(st["locate"]["pattern_between_bins"])
    wb.summary.update(s)
    wb.log({f"figures/{Path(f).stem}": wandb.Image(str(f)) for f in figs})
    art = wandb.Artifact(f"{run}_time_vs_intrusion", type="analysis")
    for f in files:
        art.add_file(str(f))
    wb.log_artifact(art)
    wb.finish()


# ------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="sa_tf_tok3_s0")
    p.add_argument("--alt", nargs="*", default=[],
                   help="Other policies with NAME_competition_sa.json on the same stream (paired counterfactual).")
    p.add_argument("--checkpoint-evals", nargs="*", default=[],
                   help="STEP=FILE: competition file of a checkpoint saved at STEP environment steps.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default=str(HERE))
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--early-until", type=int, default=400_000, help="Training flights before this step: phase early.")
    p.add_argument("--late-from", type=int, default=1_000_000, help="Training flights from this step on: phase late.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--wandb", choices=["off", "online", "offline"], default="off")
    p.add_argument("--wandb-project", default="bluesky-rl-transformer")
    p.add_argument("--wandb-max-points", type=int, default=5000, help="Rows per W&B table/scatter (CSV keeps all).")
    args = p.parse_args(argv)
    import datetime
    res, out = Path(args.results_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.boot_seed)

    s0 = load_json(res / f"{args.run}_summary.json")
    w = weights_of(s0["args"])
    s, comp, rows = flight_rows(args.run, res, w, args.early_until, args.late_from)
    train = [r for r in rows if r["source"] == "train"]
    greedy = [r for r in rows if r["source"] == "greedy_eval"]
    comp_run = [r for r in rows if r["source"] == "competition"]
    rest = np.array([r["rest_src"] for r in rows])
    st = dict(created=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), run=args.run, status="EXPLORATORY",
              weights=w, checks=dict(max_rest_term=float(rest.max()), n_rest_positive=int((rest > 1e-6).sum()),
                                     n_cut_training_flights=s.get("n_cut_agent_episodes"),
                                     n_logged_training_flights=s.get("n_agent_episodes")))
    st["audit"] = audit(w, s["args"], comp_run)

    comp_by_policy = {args.run: comp_run} if comp_run else {}
    all_rows = list(rows)
    st["paired"], pair_d, alt_meta = {}, None, {}
    for name in args.alt:
        c, ar, w_src, note = alt_rows(name, res, w, comp)
        alt_meta[name] = dict(weights_note=note, weights_same_as_run=w_src == w)
        comp_by_policy[name] = ar
        all_rows += ar
        if comp_run:
            st["paired"][name], d = paired(comp_run, ar, args.n_boot, rng)
            pair_d = pair_d if pair_d is not None else d
    st["alt"] = alt_meta
    straight_p = res / f"competition_straight_sa_{comp['episodes'] if comp else 1000}_seed{comp['seed'] if comp else 42}.json"
    if straight_p.exists():                            # means only: known terms under the reward of RUN (upper bound)
        m = load_json(straight_p)["mean"]
        known = sum(w[wk] * float(m.get(mk, 0.0)) for _, wk, mk, _ in TERMS)
        st["fly_straight"] = dict(means={k: m.get(k) for k in METRICS}, score_known_terms_upper_bound=known,
                                  note="heading term unknown (no rows); every term is <= 0, so the full score is "
                                       "<= this value")

    st["locate"] = locate(train, greedy, comp_by_policy, args.bins, args.n_boot, rng, s)
    st["correlations"] = {}
    if comp_run:
        st["correlations"]["competition"] = corr_block(comp_run, args.n_boot, rng)
        st["correlations"]["competition_goal_reached"] = corr_block(
            [r for r in comp_run if r["waypoint_reached"] == 1], args.n_boot, rng)
        st["correlations"]["competition_with_intrusion"] = corr_block(
            [r for r in comp_run if r["has_intrusion"]], args.n_boot, rng)
    for ph in ["early", "middle", "late"]:
        st["correlations"][f"train_{ph}"] = corr_block([r for r in train if r["phase"] == ph], args.n_boot, rng)
    if greedy:
        st["correlations"]["greedy_eval"] = corr_block(greedy, args.n_boot, rng)
    for name, ar in comp_by_policy.items():
        if name != args.run:
            st["correlations"][f"competition_{name}"] = corr_block(ar, args.n_boot, rng)
    st["decomposition"] = decomposition(comp_run if comp_run else [r for r in train if r["phase"] == "late"],
                                        args.n_boot, rng)
    st["decomposition_train_late"] = decomposition([r for r in train if r["phase"] == "late"], args.n_boot, rng)

    st["checkpoints"] = []
    for item in args.checkpoint_evals:
        step, f = item.split("=", 1)
        c, ar, _, _ = alt_rows(f"ckpt_{step}", res, w, comp, source=f"checkpoint_{step}", path=f)
        all_rows += ar
        st["checkpoints"].append(dict(env_steps=int(step), file=f, **{k: mean_ci([r[k] for r in ar], args.n_boot, rng)
                                       for k in ["flight_time", "intrusion_time", "ep_return_original"]},
                                      corr=corr_block(ar, args.n_boot, rng)))
    st["checkpoints"].sort(key=lambda c: c["env_steps"])

    figs = [out / "fig_scatter.png", out / "fig_trend.png", out / "fig_decomposition.png"]
    scat_st = {"competition": st["correlations"].get("competition"), "train_late": st["correlations"]["train_late"],
               "paired": st["paired"].get(args.alt[0]) if args.alt else None}
    fig_scatter(figs[0], args.run, comp_run, [r for r in train if r["phase"] == "late"], pair_d,
                args.alt[0] if args.alt else "", scat_st)
    fig_trend(figs[1], args.run, st["locate"]["train_bins"])
    fig_decomposition(figs[2], args.run, st["decomposition"])
    if st["checkpoints"]:
        figs.append(out / "fig_checkpoints.png")
        fig_checkpoints(figs[-1], args.run, st["checkpoints"])

    for r in all_rows:
        r.pop("rest_src", None)
    files = [out / "flights.csv", out / "stats.json", out / "numbers.md"]
    write_csv(files[0], all_rows)
    files[1].write_text(json.dumps(st, indent=1, default=float))
    files[2].write_text(numbers_md(st, args.run))
    for f in files + figs:
        print(f"Wrote {f}")
    if args.wandb != "off":
        to_wandb(args, st, all_rows, figs, files)


if __name__ == "__main__":
    main()
