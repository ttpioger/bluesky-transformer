"""Weight-space edit of the turn output of a single-agent policy (2026-11-18). No training, no gradient.

Idea: weight-space programming (blog rikiyat.github.io/blog/weight-space-programming, no DOI): read the weights,
find the cause of one failure, write a small edit, keep it only if it wins on practice flights. Target here: the
zig-zag of the turn (sa_tf_info_ff_cls_big_s0: 37.3 deg per step, sign-flip rate 0.87 on 50 held-out flights).
The original model file is not changed. Each candidate is a new file.

Actor (rl_tf/networks.py Policy): Linear 256->64, tanh, Linear 64->64, tanh, Linear 64->2 (= actor.4), tanh.
Output 0 = turn (45 deg per unit), output 1 = speed. z = pre-activation of the turn, h = input of actor.4 (64 units).
The deterministic turn is tanh(z).

Steps (one call does 1 to 4; step 5 is scripts.competition_eval on the chosen file):
1. Read: the norm of each weight tensor; the turn row and the speed row of actor.4 (norm, bias, 8 largest weights).
2. Diagnose: the original policy flies --n-flights practice scenarios, seeds 800000000 + i. These are not training
   seeds (1000 + ...), not the held-out seeds (900000000 + i) and not the competition stream (seed 42).
   Per step: z, h, turn, previous turn (0 at the first step). Measures: share of steps with |tanh z| > 0.95,
   quantiles of |z|, sign-flip rate, lag-1 correlation of the turn (negative = the turn alternates), and a linear
   probe "previous turn ~ p . h + q" (least squares, fit on the first half of the flights, R2 on the second half).
3. Candidates. Each changes only row 0 of actor.4 and its bias (65 weights of 790 853 in the big model):
   E1 (gain): z' = c z, c in --gains.
   E2 (inertia): z' = z + k (p . h + q), k = kappa * median |z| of step 2, kappa in --kappas. The turn is pulled
      toward the previous turn, so an alternating turn becomes smaller (a low-pass in z). Note: the plan text said
      "-k" ("against the previous turn"); that sign makes an alternating turn larger, so the sign here is "+".
      k is set from the |z| scale of the model, so the same kappa means the same relative push for any model.
4. Rule (fixed before any number; user plan 2026-11-18): on the practice flights a candidate passes when
   (a) its mean return (the reward of the checkpoint, unscaled) is higher than the original's,
   (b) its goal rate is not lower, and (c) its mean intrusion events are not higher.
   The chosen edit is the passing candidate with the highest mean return. Paired bootstrap 95 % CIs
   (candidate - original, same scenarios) are written beside; they decide nothing.
5. (separate command) Test once: scripts.competition_eval --model models/NAME_ws_TAG.pt (1000 flights, seed 42).

Output: results/ws_NAME_report.json; models/NAME_ws_TAG.pt (one per candidate; models/ is git-ignored).

    PYTHONPATH=.. ../.venv/bin/python -m scripts.weight_edit --model models/sa_tf_info_ff_cls_big_s0.pt
    smoke: add --n-flights 6 --name smoke   (writes results/ws_smoke_report.json, models/smoke_ws_*.pt)
"""
from __future__ import annotations
import argparse
import copy
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import torch
from rl_tf.envs import FlightEnv, REWARD_DEFAULTS, rollout_episode, turn_stats, v2_kwargs
from rl_tf.lidar import Lidar, token_cfg
from rl_tf.common import setup_torch, write_json

PRACTICE_SEED0 = 800_000_000
TURN_W, TURN_B = "actor.4.weight", "actor.4.bias"
SAT = 0.95
KEYS = ["ep_return", "waypoint_reached", "flight_time", "intrusion_events", "intrusion_time",
        "time_in_restricted_area", "time_outside_sector", "mean_abs_turn_deg", "turn_sign_flip_rate"]


def _job(job):
    """Worker: the flights of one seed list with one model file (one BlueSky environment per process)."""
    import contextlib
    import io
    model, seeds, record = job
    setup_torch(0)
    from rl_tf.networks import load_policy
    from rl_tf.common import policy_fn_from
    policy, ck = load_policy(model, "cpu")
    policy.eval()
    a = ck["args"]
    rew = {**REWARD_DEFAULTS, "goal_distance_penalty": 0.0, **{k: a[k] for k in REWARD_DEFAULTS if k in a}}
    lidar = Lidar(n_rays=a["n_rays"], fov_deg=a["fov_deg"], max_range_km=a["max_range_km"],
                  zone_flag=bool(a.get("zone_flag", 0)), speed=bool(a.get("lidar_speed", 0)), **token_cfg(a))
    buf = {"h": [], "z": []}
    if record:
        def hook(mod, inp, out):                     # one forward pass per step, batch [1, 1, .]
            buf["h"].append(inp[0].reshape(-1, inp[0].shape[-1])[0].double().numpy().copy())
            buf["z"].append(float(out.reshape(-1, out.shape[-1])[0, 0]))
        policy.actor[4].register_forward_hook(hook)
    fn = policy_fn_from(policy, "cpu", deterministic=True)
    with contextlib.redirect_stdout(io.StringIO()):
        env = FlightEnv("single", v2=v2_kwargs(a), **rew)
    out = []
    for s in seeds:
        buf["h"].clear()
        buf["z"].clear()
        rows, _, turns = rollout_episode(env, fn, s, lidar, a["history"])
        t = np.asarray(turns["own"], dtype=np.float64)
        r = {"seed": s, "metrics": {**rows["own"], **turn_stats(t)}}
        if record:
            if len(buf["z"]) != len(t):
                raise RuntimeError(f"seed {s}: {len(buf['z'])} forward passes, {len(t)} steps")
            r.update(h=np.asarray(buf["h"]), z=np.asarray(buf["z"]), turn=t)
        out.append(r)
    env.close()
    return out


def run(model, seeds, workers, record=False):
    workers = max(1, min(workers, len(seeds)))
    chunks = [seeds[i::workers] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_job, [(str(model), c, record) for c in chunks])
    by_seed = {r["seed"]: r for chunk in res for r in chunk}
    return [by_seed[s] for s in seeds]


def read_weights(state):
    norms = {k: {"shape": list(v.shape), "norm": float(v.double().norm())} for k, v in state.items()}
    W, b = state[TURN_W].double(), state[TURN_B].double()
    rows = {}
    for i, name in enumerate(["turn", "speed"]):
        top = torch.argsort(W[i].abs(), descending=True)[:8]
        rows[name] = {"norm": float(W[i].norm()), "bias": float(b[i]),
                      "top_units": [[int(j), float(W[i, j])] for j in top]}
    return {"n_weights": int(sum(v.numel() for v in state.values())), "norms": norms, "rows": rows}


def _stack(flights):
    H, P, Z, T = [], [], [], []
    for f in flights:
        t = f["turn"]
        H.append(f["h"])
        P.append(np.concatenate([[0.0], t[:-1]]))
        Z.append(f["z"])
        T.append(t)
    return np.concatenate(H), np.concatenate(P), np.concatenate(Z), np.concatenate(T)


def diagnose(flights):
    half = max(1, len(flights) // 2)
    Hf, Pf, _, _ = _stack(flights[:half])
    Ht, Pt, _, _ = _stack(flights[half:] or flights[:half])
    _, _, Z, T = _stack(flights)
    coef, *_ = np.linalg.lstsq(np.c_[Hf, np.ones(len(Hf))], Pf, rcond=None)
    pred = np.c_[Ht, np.ones(len(Ht))] @ coef
    r2 = 1.0 - float(((Pt - pred) ** 2).sum()) / max(float(((Pt - Pt.mean()) ** 2).sum()), 1e-12)
    lag = [float(np.corrcoef(f["turn"][1:], f["turn"][:-1])[0, 1]) for f in flights
           if len(f["turn"]) > 2 and f["turn"][1:].std() > 0 and f["turn"][:-1].std() > 0]
    az = np.abs(Z)
    return {"steps": int(len(Z)),
            "share_saturated": float((np.abs(np.tanh(Z)) > SAT).mean()),
            "abs_z_quantiles": {q: float(np.quantile(az, float(q))) for q in ["0.1", "0.25", "0.5", "0.75", "0.9"]},
            "mean_abs_turn_deg": float(np.abs(T).mean() * 45.0),
            "sign_flip_rate_mean": float(np.mean([f["metrics"]["turn_sign_flip_rate"] for f in flights])),
            "lag1_corr_mean": float(np.mean(lag)) if lag else None,
            "probe": {"fit_flights": half, "test_flights": len(flights) - half, "r2_test": r2,
                      "p": coef[:-1].tolist(), "q": float(coef[-1])}}


def make_edit(ck, kind, value, probe):
    new = copy.deepcopy(ck)
    st = new["policy"]
    W, b = st[TURN_W], st[TURN_B]
    if kind == "gain":                             # z' = c z
        W[0] = W[0] * value
        b[0] = b[0] * value
    else:                                          # z' = z + k (p . h + q)
        W[0] = W[0] + value * torch.tensor(probe["p"], dtype=W.dtype)
        b[0] = b[0] + value * probe["q"]
    changed = {k: int((st[k] != ck["policy"][k]).sum()) for k in st}
    bad = {k: n for k, n in changed.items() if n and k not in (TURN_W, TURN_B)}
    if bad or changed[TURN_W] > W.shape[1] or changed[TURN_B] > 1 or not torch.equal(W[1], ck["policy"][TURN_W][1]):
        raise RuntimeError(f"the edit changed more than the turn row: {changed}")
    new["weight_edit"] = {"kind": kind, "value": float(value), "changed_weights": sum(changed.values())}
    return new, sum(changed.values())


def mean_metrics(flights):
    return {k: float(np.mean([f["metrics"][k] for f in flights])) for k in KEYS}


def paired_ci(a, b, key, n_boot=10000, seed=0):
    d = np.array([y["metrics"][key] - x["metrics"][key] for x, y in zip(a, b)])
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--name", default=None, help="Default: model file name.")
    p.add_argument("--n-flights", type=int, default=200)
    p.add_argument("--seed0", type=int, default=PRACTICE_SEED0)
    p.add_argument("--workers", type=int, default=6, help="Processes. Each one uses one thread.")
    p.add_argument("--gains", type=float, nargs="*", default=[0.5, 0.7], help="E1: z' = c z.")
    p.add_argument("--kappas", type=float, nargs="*", default=[0.5, 1.0],
                   help="E2: z' = z + kappa * median|z| * (p . h + q).")
    args = p.parse_args(argv)
    t0 = time.time()
    model = Path(args.model)
    name = args.name or model.stem
    ck = torch.load(model, map_location="cpu", weights_only=False)
    if ck["args"].get("setting") != "single":
        raise ValueError("single-agent models only")
    seeds = [args.seed0 + i for i in range(args.n_flights)]
    report = {"status": "EXPLORATORY", "name": name, "model": str(model), "n_flights": args.n_flights,
              "practice_seeds": [seeds[0], seeds[-1]], "deterministic": True,
              "rule": "pass: mean return higher AND goal rate not lower AND mean intrusion events not higher "
                      "(practice flights, against the original); chosen = passing candidate with the highest return",
              "read": read_weights(ck["policy"])}
    print(f"{name}: {report['read']['n_weights']} weights; turn row norm {report['read']['rows']['turn']['norm']:.3f}, "
          f"bias {report['read']['rows']['turn']['bias']:+.3f}", flush=True)

    orig = run(model, seeds, args.workers, record=True)
    diag = diagnose(orig)
    report["diagnosis"] = diag
    report["original"] = mean_metrics(orig)
    t_orig = time.time() - t0
    print(f"original: {args.n_flights} flights in {t_orig:.0f} s; saturated {diag['share_saturated']:.3f}, "
          f"flip rate {diag['sign_flip_rate_mean']:.3f}, lag-1 corr {diag['lag1_corr_mean']}, "
          f"median |z| {diag['abs_z_quantiles']['0.5']:.3f}, probe R2 (test half) {diag['probe']['r2_test']:.3f}",
          flush=True)

    z_med = diag["abs_z_quantiles"]["0.5"]
    cands = [("gain", c, f"E1c{round(c * 100):03d}") for c in args.gains] + \
            [("inertia", kp * z_med, f"E2k{round(kp * 100):03d}") for kp in args.kappas]
    report["candidates"] = {}
    for kind, value, tag in cands:
        new, n_changed = make_edit(ck, kind, value, diag["probe"])
        path = model.parent / f"{name}_ws_{tag}.pt"
        torch.save(new, path)
        fl = run(path, seeds, args.workers)
        m = mean_metrics(fl)
        o = report["original"]
        ok = {"return_higher": m["ep_return"] > o["ep_return"],
              "goal_not_lower": m["waypoint_reached"] >= o["waypoint_reached"],
              "intrusions_not_higher": m["intrusion_events"] <= o["intrusion_events"]}
        report["candidates"][tag] = {"kind": kind, "value": float(value), "file": str(path),
                                     "changed_weights": n_changed, "mean": m, "checks": ok, "pass": all(ok.values()),
                                     "paired": {k: paired_ci(orig, fl, k) for k in KEYS}}
        print(f"{tag} ({kind} {value:.3f}, {n_changed} weights): return {m['ep_return']:.2f} (orig {o['ep_return']:.2f}), "
              f"goal {m['waypoint_reached']:.3f} ({o['waypoint_reached']:.3f}), intrusion events "
              f"{m['intrusion_events']:.3f} ({o['intrusion_events']:.3f}), turn {m['mean_abs_turn_deg']:.1f} deg "
              f"({o['mean_abs_turn_deg']:.1f}), flips {m['turn_sign_flip_rate']:.2f} ({o['turn_sign_flip_rate']:.2f}) "
              f"-> {'PASS' if all(ok.values()) else 'fail'}", flush=True)

    passing = [t for t, c in report["candidates"].items() if c["pass"]]
    chosen = max(passing, key=lambda t: report["candidates"][t]["mean"]["ep_return"]) if passing else None
    report["chosen"] = chosen
    report["wall_seconds"] = time.time() - t0
    write_json(f"results/ws_{name}_report.json", report)
    print(f"chosen: {chosen or 'none (no candidate passes; keep the original)'}; {report['wall_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
