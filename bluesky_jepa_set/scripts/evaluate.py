"""Evaluate a policy on the fixed scenario: competition metrics table, JSON file, optional GIF.

Policy: --model models/NAME.pt (trained actor), or --policy zero | controller (baselines without a model).
The metrics come from the unchanged environment (`info`). ep_return uses the reward of this folder.
This is a quick check. The official score is scripts.evaluate_competition in the repo root.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from jepa_set.envs import FlightEnv, REWARD_DEFAULTS, METRIC_KEYS, controller, rollout_episode
from jepa_set.common import setup_torch, write_json, actor_policy, window_stats


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
    p.add_argument("--n-agents", type=int, default=None, help="Default: from the model. Any count works with one model.")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--reset-seed", type=int, default=1, help="Scenario seed. Episode i uses reset-seed + i.")
    p.add_argument("--stochastic", action="store_true", help="Sample actions (default: mean action).")
    p.add_argument("--gif", default=None, help="Write a GIF of all episodes to this path.")
    p.add_argument("--name", default=None, help="Default: model file name or policy name.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args(argv)
    if (args.model is None) == (args.policy is None):
        p.error("give exactly one of --model and --policy")
    setup_torch(args.seed)
    rew = dict(REWARD_DEFAULTS)
    if args.model:
        from jepa_set.policy import load_actor
        actor, ck = load_actor(args.model, args.device)
        actor.eval()
        a = ck["args"]
        setting, n_agents = args.setting or a.get("setting", "multi"), args.n_agents or a.get("n_agents", 2)
        rew.update({k: a[k] for k in rew if k in a})
        policy = actor_policy(actor, args.device, deterministic=not args.stochastic)
        name = args.name or Path(args.model).stem
        n_par = ck.get("n_params_actor")
    else:
        setting, n_agents = args.setting or "multi", args.n_agents or 2
        fn = controller if args.policy == "controller" else (lambda o: np.zeros(2, dtype=np.float32))
        policy = lambda obs: {k: fn(o) for k, o in obs.items()}
        name = args.name or f"{args.policy}_{setting}"
        n_par = 0
    env = FlightEnv(setting, n_agents, render_mode="rgb_array" if args.gif else None, **rew)
    rows, frames = [], []
    for i in range(args.episodes):
        ep_rows, fr = rollout_episode(env, policy, args.reset_seed + i, render=bool(args.gif))
        frames += fr
        for a_id, r in ep_rows.items():
            rows.append({"episode_seed": args.reset_seed + i, "aircraft": a_id, **r})
    env.close()
    cols = METRIC_KEYS + ["ep_return"]
    print(f"\n{name}: setting {setting}, aircraft {len(env.ids)}, status EXPLORATORY")
    print("seed aircraft " + " ".join(f"{c[:22]:>22}" for c in cols))
    for r in rows:
        print(f"{r['episode_seed']:>4} {r['aircraft']:>8} " + " ".join(f"{r[c]:>22.2f}" for c in cols))
    mean = window_stats([{c: r[c] for c in cols} for r in rows])
    print(f"{'mean':>13} " + " ".join(f"{mean[c]:>22.2f}" for c in cols))
    if args.gif:
        save_gif(frames, args.gif)
    write_json(Path(args.out_dir) / "results" / f"{name}_eval.json", {
        "status": "EXPLORATORY", "name": name, "setting": setting, "n_aircraft": len(env.ids),
        "n_params_actor": n_par, "deterministic": not args.stochastic, "reward": rew,
        "mean": mean, "rows": rows, "gif": args.gif})


if __name__ == "__main__":
    main()
