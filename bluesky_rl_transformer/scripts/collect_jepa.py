"""Record scripted flights for JEPA pretraining (2026-11-13). No RL policy flies here.

Single agent, random scenarios. Flight i uses scenario seed SEED0 + i (far from the training seeds 1000 + ...,
the held-out seeds 900 000 000 + ... and the competition protocol), the controller POLICIES[i % 4] and the
density DENSITIES[(i // 4) % len] (other aircraft : restricted areas). Each worker process has one BlueSky
environment and one thread, and writes one shard: data/jepa_NAME_wK.npz.

Row j of a shard is one observation o_t (state, lidar as the policy sees them, float16), the action a_t that
followed it (0 for the last observation of a flight), has_next (1 = row j+1 is o_{t+1} of the same flight) and the
flags of the step after o_t: intrusion, restricted area, outside the sector (1 = some seconds of the step).

Each row also has the reward of the step after o_t, with the reward flags of this command (2026-11-13 evening;
the controllers do not read it). Give the reward flags of the policy run: scripts.train_world_model checks them.
--controllers chooses the controllers (default: the 4 scripted ones). "policy" flies a trained policy
(--policy-model, sampled actions, CPU) and needs the same token and lidar flags as that policy.

Multi agent (--setting multi --n-agents N, 2026-11-13 late): all N aircraft are recorded; aircraft j of flight i
uses controller (i * N + j) % len(controllers). The rows of one aircraft are written one after another (so a
window of rows is one aircraft over time). Columns for both settings: agent (index of the aircraft in the flight)
and nb_ids [K] (agent index of the aircraft in each aircraft-token row; -1 = empty row or scripted traffic).
scripts.train_world_model uses them to give each token row the command of its aircraft.

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
POLICIES = ["straight", "goal", "goal_noise", "random"]     # default controllers, in turn
ALL_KINDS = POLICIES + ["policy"]                            # column "policy" = index in this list


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
    w, flights, lidar_cfg, history, densities, out_path, reward, controllers, policy_model, setting, n_agents = job
    import torch
    torch.set_num_threads(1)
    from rl_tf.envs import FlightEnv, ObsBuilder, DIST_NORM_KM
    from rl_tf.lidar import Lidar
    from rl_tf.world_model import aircraft_row_ids
    with contextlib.redirect_stdout(io.StringIO()):          # BlueSky start-up text
        env = FlightEnv(setting, n_agents, **reward)         # the reward is recorded, the controllers do not read it
    lidar = Lidar(**lidar_cfg)
    K = lidar.ac_tokens
    pol = None
    if policy_model:
        from rl_tf.networks import load_policy
        pol, _ = load_policy(policy_model, "cpu")
        torch.manual_seed(SEED0 + w)                         # sampled policy actions: one stream per worker
    cols = {k: [] for k in ["state", "lidar", "action", "has_next", "f_intr", "f_area", "f_out", "goal_km", "reward",
                            "episode", "step", "policy", "n_intr", "n_obs", "agent", "nb_ids"]}
    keys = ["intrusion_time", "time_in_restricted_area", "time_outside_sector"]
    t0 = time.time()
    for i in flights:
        n_intr, n_obs = densities[(i // len(controllers)) % len(densities)]
        env.set_density(n_intr, n_obs)
        obs = env.reset(seed=SEED0 + i)
        ids = list(env.ids)                                  # fixed order: "agent" column = index in this list
        n_other = n_intr if setting == "single" else len(ids) - 1
        kind, ctrl = {}, {}
        for j, a in enumerate(ids):                          # single agent: the controller and rng of before
            kind[a] = controllers[(i * len(ids) + j) % len(controllers)]
            rng = np.random.default_rng(SEED0 + i) if setting == "single" else np.random.default_rng([SEED0 + i, j])
            ctrl[a] = None if kind[a] == "policy" else make_controller(kind[a], rng)
        b = {a: ObsBuilder(lidar, history) for a in ids}
        for a in ids:
            b[a].reset(obs[a])
        prev = {a: {k: 0.0 for k in keys} for a in ids}
        rows = {a: [] for a in ids}                          # rows of each aircraft, written after the flight
        t = 0

        def nb_ids(o):
            r = [ids.index(c) if c in ids else -1 for c in aircraft_row_ids(lidar, o)]
            return np.asarray(r + [-1] * (K - len(r)), dtype=np.int8)

        while env.agents:
            live = list(env.agents)
            built = {a: b[a].get() for a in live}
            acts = {a: ctrl[a](obs[a]) for a in live if ctrl[a] is not None}
            pl = [a for a in live if ctrl[a] is None]
            if pl:
                raw, _, _ = pol.act(torch.as_tensor(np.stack([built[a][0] for a in pl]), dtype=torch.float32)[None],
                                    torch.as_tensor(np.stack([built[a][1] for a in pl]), dtype=torch.float32)[None],
                                    deterministic=False)
                acts.update({a: raw[0, n].clamp(-1.0, 1.0).numpy().astype(np.float32) for n, a in enumerate(pl)})
            nobs, rew, te, tr, info = env.step(acts)
            for a in live:
                d = {k: float(info[a][k]) - prev[a][k] for k in keys}
                prev[a] = {k: float(info[a][k]) for k in keys}
                rows[a].append(dict(state=built[a][0], lidar=built[a][1], action=acts[a], has_next=1,
                                    f_intr=d["intrusion_time"] > 0, f_area=d["time_in_restricted_area"] > 0,
                                    f_out=d["time_outside_sector"] > 0,
                                    goal_km=float(obs[a]["waypoint_distance"][0]) * DIST_NORM_KM,
                                    reward=float(rew[a]), step=t, nb_ids=nb_ids(obs[a])))
                obs[a] = nobs[a]
                b[a].push(obs[a], acts[a])
                if te[a] or tr[a]:                           # last observation of the flight: no action, no next row
                    s, l = b[a].get()
                    rows[a].append(dict(state=s, lidar=l, action=np.zeros(2, np.float32), has_next=0, f_intr=0,
                                        f_area=0, f_out=0, goal_km=float(obs[a]["waypoint_distance"][0]) * DIST_NORM_KM,
                                        reward=0.0, step=t + 1, nb_ids=nb_ids(obs[a])))
            t += 1
        for j, a in enumerate(ids):
            for r in rows[a]:
                for k, v in r.items():
                    cols[k].append(v)
                for k, v in [("episode", i), ("policy", ALL_KINDS.index(kind[a])), ("n_intr", n_other),
                             ("n_obs", n_obs), ("agent", j)]:
                    cols[k].append(v)
    env.close()
    types = dict(state=np.float16, lidar=np.float16, action=np.float32, has_next=np.uint8, f_intr=np.uint8,
                 f_area=np.uint8, f_out=np.uint8, goal_km=np.float32, reward=np.float32, episode=np.int32, step=np.int16,
                 policy=np.int8, n_intr=np.int8, n_obs=np.int8, agent=np.int8, nb_ids=np.int8)
    if not cols["nb_ids"]:
        cols["nb_ids"] = np.zeros((0, K), np.int8)
    arr = {k: np.asarray(v, dtype=types[k]) for k, v in cols.items()}
    np.savez(out_path, **arr)
    return {"worker": w, "rows": len(arr["has_next"]), "flights": len(flights), "seconds": time.time() - t0,
            "intrusion_steps": int(arr["f_intr"].sum()), "area_steps": int(arr["f_area"].sum()),
            "outside_steps": int(arr["f_out"].sum())}


def main(argv=None):
    from scripts.train import build_parser
    from rl_tf.envs import lidar_cfg_from_args, obs_dims, reward_kwargs
    from rl_tf.common import write_json
    p = build_parser()
    p.description = __doc__
    p.add_argument("--flights", type=int, default=1200, help="Number of recorded flights.")
    p.add_argument("--densities", default=None,
                   help="Other aircraft : restricted areas, one value per flight in turn (competition: 10:5). "
                        "Default: 10:5,10:5,5:3,2:1 (single agent); 0:5,0:5,0:3,0:1 (multi agent: the other aircraft "
                        "are the --n-agents learning aircraft, so the first number must be 0).")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--controllers", default=",".join(POLICIES),
                   help=f"Controllers in turn, from {ALL_KINDS}. 'policy' needs --policy-model.")
    p.add_argument("--policy-model", default="", help="Trained policy for the controller 'policy' (same token flags).")
    args = p.parse_args(argv)
    lidar_cfg = lidar_cfg_from_args(args)
    dims = obs_dims(lidar_cfg, args.history)
    reward = reward_kwargs(args)
    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    if not controllers or any(c not in ALL_KINDS for c in controllers):
        p.error(f"--controllers: use names from {ALL_KINDS}")
    if ("policy" in controllers) != bool(args.policy_model):
        p.error("the controller 'policy' and --policy-model go together")
    if args.policy_model:
        import torch
        from rl_tf.networks import encoder_mismatch
        ck = torch.load(args.policy_model, map_location="cpu", weights_only=False)
        bad = encoder_mismatch(ck["args"], ck["dims"], vars(args), dims)
        if bad:
            p.error(f"--policy-model: token or network settings differ from the flags of this command: {bad}")
    if args.densities is None:
        args.densities = "10:5,10:5,5:3,2:1" if args.setting == "single" else "0:5,0:5,0:3,0:1"
    dens = [tuple(int(x) for x in d.split(":")) for d in args.densities.split(",")]
    if args.setting == "multi" and any(i != 0 for i, _ in dens):
        p.error("--densities with --setting multi: the first number must be 0 (no scripted aircraft there)")
    out = Path(args.out_dir) / args.data_dir
    out.mkdir(parents=True, exist_ok=True)
    W = max(1, min(args.workers, args.flights))
    jobs = [(w, list(range(w, args.flights, W)), lidar_cfg, args.history, dens, str(out / f"jepa_{args.name}_w{w}.npz"),
             reward, controllers, args.policy_model, args.setting, args.n_agents) for w in range(W)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(W, maxtasksperchild=1) as pool:   # one BlueSky environment per process
        stats = pool.map(run_worker, jobs, chunksize=1)
    wall = time.time() - t0
    rows = sum(s["rows"] for s in stats)
    meta = {"status": "EXPLORATORY", "name": args.name, "flights": args.flights, "rows": rows,
            "rows_per_second": rows / max(wall, 1e-9), "wall_seconds": wall, "seed0": SEED0, "policies": controllers,
            "policy_kinds": ALL_KINDS, "policy_model": args.policy_model, "reward": reward, "setting": args.setting,
            "n_agents": args.n_agents if args.setting == "multi" else 1,
            "densities": dens, "dims": dims, "lidar": lidar_cfg, "history": args.history,
            "share_intrusion_steps": sum(s["intrusion_steps"] for s in stats) / max(rows, 1),
            "share_area_steps": sum(s["area_steps"] for s in stats) / max(rows, 1),
            "share_outside_steps": sum(s["outside_steps"] for s in stats) / max(rows, 1),
            "shards": [j[5] for j in jobs], "workers": stats}
    (out / f"jepa_{args.name}_meta.json").write_text(json.dumps(meta, indent=1))
    write_json(Path(args.out_dir) / "results" / f"jepa_data_{args.name}_summary.json", meta)
    print(f"{rows} rows from {args.flights} flights in {wall:.0f} s ({meta['rows_per_second']:.0f} rows/s)")


if __name__ == "__main__":
    main()
