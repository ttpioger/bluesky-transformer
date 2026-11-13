"""Record scripted flights for JEPA pretraining (2026-11-13). No RL policy flies here.

Single agent, random scenarios. Flight i uses scenario seed SEED0 + i (far from the training seeds 1000 + ...,
the held-out seeds 900 000 000 + ... and the competition protocol), the controller POLICIES[i % 4] and the
density DENSITIES[(i // 4) % len] (other aircraft : restricted areas). Each worker process has one BlueSky
environment and one thread, and writes one shard: data/jepa_NAME_wK.npz.

Row j of a shard is one observation o_t (state, lidar as the policy sees them, float16), the action a_t that
followed it (0 for the last observation of a flight), has_next (1 = row j+1 is o_{t+1} of the same flight) and the
flags of the step after o_t: intrusion, restricted area, outside the sector (1 = some seconds of the step).

Use the same token and lidar flags as the training run, for example:
    PYTHONPATH=.. ../.venv/bin/python -m scripts.collect_jepa --name scripted1 --flights 1200 --workers 6 \
        --ac-tokens 10 --zone-tokens 5 --act-tokens 4 --edge-tokens 8 --ac-cpa 1 --lidar-speed 1
With the map image (--map-px 32) one row has about 10 KB (state 3 195 values, lidar 1 920 values, float16):
1200 flights of at most 300 steps give at most about 3.7 GB of shards.
"""
from __future__ import annotations
import contextlib
import io
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np

SEED0 = 3_000_000
POLICIES = ["straight", "goal", "goal_noise", "random"]


def make_controller(kind, rng):
    """Scripted controller: obs -> action (turn, speed) in [-1, 1]."""
    if kind == "straight":
        return lambda o: np.zeros(2, np.float32)
    gain = float(rng.uniform(0.5, 1.5))
    if kind == "goal":
        return lambda o: np.array([np.clip(-gain * float(o["sin_drift"][0]), -1, 1), 0.0], np.float32)
    if kind == "goal_noise":
        def f(o):
            a = np.array([-gain * float(o["sin_drift"][0]), 0.0]) + rng.normal(0.0, 0.3, 2)
            return np.clip(a, -1, 1).astype(np.float32)
        return f
    if kind == "random":                                   # Ornstein-Uhlenbeck turn and speed
        x = np.zeros(2)

        def f(o):
            x[:] = x + 0.3 * (0.0 - x) + rng.normal(0.0, 0.5, 2)
            return np.clip(x, -1, 1).astype(np.float32)
        return f
    raise ValueError(kind)


def run_worker(job):
    w, flights, lidar_cfg, history, densities, out_path = job
    import torch
    torch.set_num_threads(1)
    from rl_tf.envs import FlightEnv, ObsBuilder, REWARD_PRESETS, DIST_NORM_KM
    from rl_tf.lidar import Lidar
    with contextlib.redirect_stdout(io.StringIO()):          # BlueSky start-up text
        env = FlightEnv("single", **REWARD_PRESETS["los"])     # the reward is not used here
    lidar = Lidar(**lidar_cfg)
    cols = {k: [] for k in ["state", "lidar", "action", "has_next", "f_intr", "f_area", "f_out", "goal_km",
                            "episode", "step", "policy", "n_intr", "n_obs"]}
    t0 = time.time()
    for i in flights:
        rng = np.random.default_rng(SEED0 + i)
        kind = POLICIES[i % len(POLICIES)]
        n_intr, n_obs = densities[(i // len(POLICIES)) % len(densities)]
        env.set_density(n_intr, n_obs)
        ctrl = make_controller(kind, rng)
        obs = env.reset(seed=SEED0 + i)["own"]
        b = ObsBuilder(lidar, history)
        b.reset(obs)
        prev = {"intrusion_time": 0.0, "time_in_restricted_area": 0.0, "time_outside_sector": 0.0}
        t, done = 0, False
        while not done:
            s, l = b.get()
            a = ctrl(obs)
            nobs, _, te, tr, info = env.step({"own": a})
            info = info["own"]
            d = {k: float(info[k]) - prev[k] for k in prev}
            prev = {k: float(info[k]) for k in prev}
            done = bool(te["own"] or tr["own"])
            for k, v in [("state", s), ("lidar", l), ("action", a), ("has_next", 1), ("f_intr", d["intrusion_time"] > 0),
                         ("f_area", d["time_in_restricted_area"] > 0), ("f_out", d["time_outside_sector"] > 0),
                         ("goal_km", float(obs["waypoint_distance"][0]) * DIST_NORM_KM), ("episode", i), ("step", t),
                         ("policy", i % len(POLICIES)), ("n_intr", n_intr), ("n_obs", n_obs)]:
                cols[k].append(v)
            obs = nobs["own"]
            b.push(obs, a)
            t += 1
        s, l = b.get()                                     # last observation of the flight: no action, no next row
        for k, v in [("state", s), ("lidar", l), ("action", np.zeros(2, np.float32)), ("has_next", 0), ("f_intr", 0),
                     ("f_area", 0), ("f_out", 0), ("goal_km", float(obs["waypoint_distance"][0]) * DIST_NORM_KM),
                     ("episode", i), ("step", t), ("policy", i % len(POLICIES)), ("n_intr", n_intr), ("n_obs", n_obs)]:
            cols[k].append(v)
    env.close()
    types = dict(state=np.float16, lidar=np.float16, action=np.float32, has_next=np.uint8, f_intr=np.uint8,
                 f_area=np.uint8, f_out=np.uint8, goal_km=np.float32, episode=np.int32, step=np.int16,
                 policy=np.int8, n_intr=np.int8, n_obs=np.int8)
    arr = {k: np.asarray(v, dtype=types[k]) for k, v in cols.items()}
    np.savez(out_path, **arr)
    return {"worker": w, "rows": len(arr["has_next"]), "flights": len(flights), "seconds": time.time() - t0,
            "intrusion_steps": int(arr["f_intr"].sum()), "area_steps": int(arr["f_area"].sum()),
            "outside_steps": int(arr["f_out"].sum())}


def main(argv=None):
    from scripts.train import build_parser
    from rl_tf.envs import lidar_cfg_from_args, obs_dims
    from rl_tf.common import write_json
    p = build_parser()
    p.description = __doc__
    p.add_argument("--flights", type=int, default=1200, help="Number of recorded flights.")
    p.add_argument("--densities", default="10:5,10:5,5:3,2:1",
                   help="Other aircraft : restricted areas, one value per flight in turn (competition: 10:5).")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args(argv)
    lidar_cfg = lidar_cfg_from_args(args)
    dims = obs_dims(lidar_cfg, args.history)
    dens = [tuple(int(x) for x in d.split(":")) for d in args.densities.split(",")]
    out = Path(args.out_dir) / args.data_dir
    out.mkdir(parents=True, exist_ok=True)
    W = max(1, min(args.workers, args.flights))
    jobs = [(w, list(range(w, args.flights, W)), lidar_cfg, args.history, dens, str(out / f"jepa_{args.name}_w{w}.npz"))
            for w in range(W)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(W, maxtasksperchild=1) as pool:   # one BlueSky environment per process
        stats = pool.map(run_worker, jobs, chunksize=1)
    wall = time.time() - t0
    rows = sum(s["rows"] for s in stats)
    meta = {"status": "EXPLORATORY", "name": args.name, "flights": args.flights, "rows": rows,
            "rows_per_second": rows / max(wall, 1e-9), "wall_seconds": wall, "seed0": SEED0, "policies": POLICIES,
            "densities": dens, "dims": dims, "lidar": lidar_cfg, "history": args.history,
            "share_intrusion_steps": sum(s["intrusion_steps"] for s in stats) / max(rows, 1),
            "share_area_steps": sum(s["area_steps"] for s in stats) / max(rows, 1),
            "share_outside_steps": sum(s["outside_steps"] for s in stats) / max(rows, 1),
            "shards": [j[-1] for j in jobs], "workers": stats}
    (out / f"jepa_{args.name}_meta.json").write_text(json.dumps(meta, indent=1))
    write_json(Path(args.out_dir) / "results" / f"jepa_data_{args.name}_summary.json", meta)
    print(f"{rows} rows from {args.flights} flights in {wall:.0f} s ({meta['rows_per_second']:.0f} rows/s)")


if __name__ == "__main__":
    main()
