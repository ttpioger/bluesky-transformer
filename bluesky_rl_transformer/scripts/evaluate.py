"""Evaluate a policy: competition metrics table, turn statistics, JSON file, optional GIF.

Policy: --model models/NAME.pt (trained), or --policy zero | controller (baselines without a model).
The metrics come from the unchanged environment (`info`). ep_return uses the reward of this folder.
The official score is scripts.evaluate_competition in the repo root.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from rl_tf.envs import FlightEnv, REWARD_DEFAULTS, METRIC_KEYS, controller, rollout_episode, turn_stats
from rl_tf.lidar import Lidar
from rl_tf.common import setup_torch, write_json, window_stats, policy_fn_from, wandb_init, add_wandb_args


def save_gif(frames, path, size=384, max_frames=150, duration=80):
    from PIL import Image
    if len(frames) > max_frames:
        frames = frames[::int(np.ceil(len(frames) / max_frames))]
    images = [Image.fromarray(f).resize((size, size), Image.BILINEAR) for f in frames]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], loop=0, duration=duration, optimize=True)
    print(f"Wrote {path} ({len(images)} frames)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument("--policy", choices=["zero", "controller"], default=None)
    p.add_argument("--setting", choices=["single", "multi"], default=None, help="Default: from the model.")
    p.add_argument("--n-agents", type=int, default=None, help="Default: from the model.")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--reset-seed", type=int, default=1, help="Scenario seed. Episode i uses reset-seed + i.")
    p.add_argument("--stochastic", action="store_true", help="Sample actions (default: mean action).")
    p.add_argument("--gif", default=None, help="Write a GIF of all episodes to this path.")
    p.add_argument("--no-rays", action="store_true", help="GIF without the lidar rays.")
    p.add_argument("--name", default=None, help="Default: model file name or policy name.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=".")
    add_wandb_args(p)
    args = p.parse_args(argv)
    if (args.model is None) == (args.policy is None):
        p.error("give exactly one of --model and --policy")
    setup_torch(0)
    rew, lidar_cfg, history, n_par = dict(REWARD_DEFAULTS), {}, 4, 0
    if args.model:
        from rl_tf.networks import load_policy
        policy, ck = load_policy(args.model, args.device)
        a = ck["args"]
        setting, n_agents = args.setting or a["setting"], args.n_agents or a["n_agents"]
        rew["goal_distance_penalty"] = 0.0             # a checkpoint without this key had no such term
        rew.update({k: a[k] for k in rew if k in a})
        lidar_cfg = dict(n_rays=a["n_rays"], fov_deg=a["fov_deg"], max_range_km=a["max_range_km"],
                         zone_flag=bool(a.get("zone_flag", 0)), speed=bool(a.get("lidar_speed", 0)), ac_tokens=int(a.get("ac_tokens", 0)))
        history, n_par = a["history"], ck["n_params"]
        fn = policy_fn_from(policy, args.device, deterministic=not args.stochastic)
        name = args.name or Path(args.model).stem
    else:
        setting, n_agents = args.setting or "single", args.n_agents or 2
        base = controller if args.policy == "controller" else (lambda o: np.zeros(2, dtype=np.float32))
        fn = lambda raw, built: {k: base(o) for k, o in raw.items()}
        name = args.name or f"{args.policy}_{setting}"
    env = FlightEnv(setting, n_agents, render_mode="rgb_array" if args.gif else None, **rew)
    lidar = Lidar(**lidar_cfg)
    rows, frames = [], []
    for i in range(args.episodes):
        ep_rows, fr, turns = rollout_episode(env, fn, args.reset_seed + i, lidar, history, render=bool(args.gif),
                                            rays=not args.no_rays)
        frames += fr
        rows += [{"episode_seed": args.reset_seed + i, "aircraft": a, **r, **turn_stats(turns[a])} for a, r in ep_rows.items()]
    env.close()
    cols = METRIC_KEYS + ["ep_return", "mean_abs_turn_deg", "turn_sign_flip_rate"]
    print(f"\n{name}: setting {setting}, aircraft {len(env.ids)}, status EXPLORATORY")
    print("seed aircraft " + " ".join(f"{c[:20]:>20}" for c in cols))
    for r in rows:
        print(f"{r['episode_seed']:>4} {r['aircraft']:>8} " + " ".join(f"{r[c]:>20.2f}" for c in cols))
    mean = window_stats([{c: r[c] for c in cols} for r in rows])
    print(f"{'mean':>13} " + " ".join(f"{mean[c]:>20.2f}" for c in cols))
    if args.gif:
        save_gif(frames, args.gif)
    write_json(Path(args.out_dir) / "results" / f"{name}_eval.json", {
        "status": "EXPLORATORY", "name": name, "setting": setting, "n_aircraft": len(env.ids),
        "n_params": n_par, "deterministic": not args.stochastic, "reward": rew, "mean": mean, "rows": rows, "gif": args.gif})

    # Training curves (if results/NAME_summary.json exists) and upload. The W&B run id is the name: with the
    # default name (model file name) the figures go to the run of scripts.train.
    import json
    out = Path(args.out_dir)
    summary_path, curves_path = out / "results" / f"{name}_summary.json", out / "figures" / f"{name}_curves.png"
    if summary_path.exists():
        from scripts.compare import curves_figure
        curves_figure(json.loads(summary_path.read_text()), curves_path)
    wb = wandb_init(args.wandb, args.wandb_project, name)
    if wb is not None:
        import wandb
        media = {}
        if summary_path.exists():
            media["figures/training_curves"] = wandb.Image(str(curves_path))
        if args.gif:
            media["videos/policy"] = wandb.Video(args.gif, format="gif")
        table = wandb.Table(columns=["episode_seed", "aircraft"] + cols)
        for r in rows:
            table.add_data(r["episode_seed"], r["aircraft"], *[r[c] for c in cols])
        media["held_out/per_aircraft"] = table
        wb.log(media)
        wb.summary.update({f"held_out/{k}": v for k, v in mean.items()})
        wb.finish()


if __name__ == "__main__":
    main()
