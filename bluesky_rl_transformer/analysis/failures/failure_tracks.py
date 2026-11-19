"""Flown tracks of chosen episodes of the official single-agent protocol (seed 42, 1000 episodes), one figure per model.

Question: what does the policy do in the episodes where it does not reach its goal?
Protocol: the same as scripts/competition_eval.py (one process, reset with seed 42 at episode 0 only, the later
episodes continue the same random stream, deterministic policy, reward of the checkpoint). Only episode 0 has a
seed, so an episode i is rebuilt by flying the stream from episode 0 up to i (--mode replay, the default).
--mode skip only resets the environment for the episodes before i (no flight): faster, but a check of this kind
failed for single agent on 2026-11-12 (cause not found). Each mode compares the replayed metrics with the stored
row of results/NAME_competition_sa.json and writes the result to the JSON ("match"). Use the figures only if
all rows match.

Outputs (in --out-dir): NAME_tracks.png (one panel per episode), NAME_failures.json (replayed metrics, stored
metrics, match), optional NAME_epI.gif, and with --wandb the figure and a table in the W&B run NAME_failures.
New analysis code only: it imports from scripts/ and rl_tf/ and changes nothing there.

    PYTHONPATH=.. ../.venv/bin/python analysis/failures/failure_tracks.py --model models/sa_tf_tok3_s0.pt \
        --episodes 325 403 473 798 845 924 940 962
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                 # bluesky_rl_transformer/
sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402
from rl_tf.envs import FlightEnv, REWARD_DEFAULTS, METRIC_KEYS, rollout_episode, v2_kwargs   # noqa: E402
from rl_tf.lidar import Lidar, token_cfg                    # noqa: E402
from rl_tf.common import setup_torch, write_json, policy_fn_from, wandb_init, add_wandb_args   # noqa: E402

SEED = 42
TOL = 1e-6


def load(model, device):
    """Policy function, reward, lidar and history of a checkpoint, as in scripts/competition_eval.py _job."""
    from rl_tf.networks import load_policy
    policy, ck = load_policy(model, device)
    c = ck["args"]
    rew = {**REWARD_DEFAULTS, "goal_distance_penalty": 0.0, **{k: c[k] for k in REWARD_DEFAULTS if k in c}}
    lidar = Lidar(n_rays=c["n_rays"], fov_deg=c["fov_deg"], max_range_km=c["max_range_km"],
                  zone_flag=bool(c.get("zone_flag", 0)), speed=bool(c.get("lidar_speed", 0)), **token_cfg(c))
    return policy_fn_from(policy, device, deterministic=True), rew, lidar, c["history"], v2_kwargs(c)


def stored_rows(name, results_dir):
    p = Path(results_dir) / f"{name}_competition_sa.json"
    if not p.exists():
        return None, p
    return {r["episode_index"]: r for r in json.loads(p.read_text())["rows"]}, p


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="models/NAME.pt")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="Episode indexes of the seed-42 stream. Default: the episodes where this model did not "
                        "reach its goal (from results/NAME_competition_sa.json).")
    p.add_argument("--mode", choices=["replay", "skip"], default="replay")
    p.add_argument("--gif", action="store_true", help="Also write one GIF per chosen episode (slower).")
    p.add_argument("--gif-size", type=int, default=512)
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    p.add_argument("--device", default="cpu")
    add_wandb_args(p)
    p.add_argument("--wandb-name", default=None,
                   help="W&B run name and id. Default NAME_failures. The id is the name (resume=allow), so give a "
                        "new name if an earlier run of that name was cancelled or deleted.")
    args = p.parse_args(argv)

    from scripts.evaluate import TrackRecorder, tracks_figure, save_gif
    name = Path(args.model).stem
    stored, spath = stored_rows(name, args.results_dir)
    if args.episodes is None:
        if stored is None:
            p.error(f"{spath} not found: give --episodes")
        args.episodes = sorted(i for i, r in stored.items() if r["waypoint_reached"] < 1)
    wanted = sorted(set(args.episodes))
    if not wanted or wanted[0] < 0:
        p.error("give episode indexes >= 0")
    print(f"{name}: episodes {wanted}, mode {args.mode}, stream up to episode {wanted[-1]}")

    setup_torch(0)
    fn, rew, lidar, history, v2 = load(args.model, args.device)
    t0 = time.time()
    tracks, out_rows = {}, []
    with contextlib.redirect_stdout(io.StringIO()):
        env = FlightEnv("single", 2, render_mode="rgb_array" if args.gif else None, v2=v2, **rew)
        rec = TrackRecorder(env)
        for i in range(wanted[-1] + 1):
            keep = i in wanted
            if args.mode == "skip" and not keep:
                env.env.reset(seed=SEED if i == 0 else None)      # move the random stream, no flight
                continue
            if keep:
                rec.start(f"42, episode {i}")
            ep, frames, _ = rollout_episode(env, fn, SEED if i == 0 else None, lidar, history,
                                            render=args.gif and keep, on_step=rec if keep else None)
            if not keep:
                continue
            tracks[i] = rec.result(ep)
            row = {"episode_index": i, **ep["own"]}
            ref = stored.get(i) if stored else None
            row["stored"] = {k: ref[k] for k in METRIC_KEYS} if ref else None
            row["match"] = None if ref is None else all(abs(row[k] - ref[k]) <= TOL for k in METRIC_KEYS)
            out_rows.append(row)
            if frames:
                save_gif(frames, Path(args.out_dir) / f"{name}_ep{i}.gif", size=args.gif_size, max_frames=400)
        env.close()
    wall = time.time() - t0

    for r in out_rows:
        print(f"episode {r['episode_index']:>4}: goal {r['waypoint_reached']:.0f}, time {r['flight_time']:.0f} s, "
              f"intrusion events {r['intrusion_events']:.0f}, sector exits {r['sector_exit_events']:.0f}, "
              f"match stored {r['match']}")
    n_match = sum(bool(r["match"]) for r in out_rows)
    print(f"{n_match}/{len(out_rows)} episodes match the stored rows; wall {wall:.0f} s")
    if n_match < len(out_rows):
        print("WARNING: a replayed episode differs from the stored row; the figure does not show the evaluated flight.")

    fig = Path(args.out_dir) / f"{name}_tracks.png"
    tracks_figure([tracks[i] for i in wanted], fig, f"{name}, official protocol (seed 42)")
    write_json(Path(args.out_dir) / f"{name}_failures.json",
               {"status": "EXPLORATORY", "name": name, "model": args.model, "seed": SEED, "mode": args.mode,
                "episodes": wanted, "stored_file": str(spath), "n_match": n_match, "wall_seconds": wall,
                "rows": out_rows})

    wb = wandb_init(args.wandb, args.wandb_project, args.wandb_name or f"{name}_failures",
                    config={"model": args.model, "episodes": wanted, "mode": args.mode, "seed": SEED})
    if wb is not None:
        import wandb
        cols = ["episode_index", "match"] + METRIC_KEYS
        wb.log({"failures/tracks": wandb.Image(str(fig)),
                "failures/table": wandb.Table(columns=cols, data=[[r["episode_index"], r["match"]] +
                                                                  [r[k] for k in METRIC_KEYS] for r in out_rows])})
        wb.summary["failures/n_match"] = n_match
        wb.finish()


if __name__ == "__main__":
    main()
