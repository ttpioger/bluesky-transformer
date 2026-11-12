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
from rl_tf.lidar import Lidar, NM2KM, DEG2KM
from rl_tf.common import setup_torch, write_json, window_stats, policy_fn_from, wandb_init, add_wandb_args


class TrackRecorder:
    """Records the position of all aircraft at every step of one episode (controlled and scripted)."""

    def __init__(self, env):
        self.env, self.inner = env, getattr(env.env, "unwrapped", env.env)
        self.single = env.ids == ["own"]

    def start(self, seed):
        self.seed, self.ctrl, self.close, self.others, self.gone = seed, {}, {}, {}, set()

    def __call__(self):
        import bluesky as bs
        inner = self.inner
        live = {inner.agent} if self.single else set(self.env.agents)
        goals = {inner.agent: (inner.goal_lat, inner.goal_lon)} if self.single else inner._goal
        pos = {ac: (float(bs.traf.lat[j]), float(bs.traf.lon[j])) for j, ac in enumerate(bs.traf.id)}
        for ac, (lat, lon) in pos.items():
            if ac not in goals:
                self.others.setdefault(ac, []).append((lat, lon))
                continue
            if ac in self.gone:
                continue
            d = [np.hypot((q[0] - lat) * DEG2KM, (q[1] - lon) * DEG2KM * np.cos(np.deg2rad(lat)))
                 for o, q in pos.items() if o != ac and not (o in goals and o in self.gone)]
            self.ctrl.setdefault(ac, []).append((lat, lon))
            self.close.setdefault(ac, []).append(bool(d and min(d) < inner.intrusion_distance * NM2KM))
            if ac not in live and len(self.ctrl[ac]) > 1:      # the flight has ended: this is its last point
                self.gone.add(ac)

    def result(self, rows):
        sc, inner = self.inner.scenario, self.inner
        goals = {inner.agent: (inner.goal_lat, inner.goal_lon)} if self.single else inner._goal
        key = (lambda ac: "own") if self.single else (lambda ac: ac)
        return dict(seed=self.seed, ctrl=self.ctrl, close=self.close, others=list(self.others.values()),
                    goals={ac: (float(g[0]), float(g[1])) for ac, g in goals.items()},
                    reached={ac: bool(rows[key(ac)]["waypoint_reached"] >= 0.5) for ac in self.ctrl if key(ac) in rows},
                    zones=[np.asarray(z.vertices).tolist() for z in sc.obstacles], sector=np.asarray(sc.sector).tolist(),
                    intrusion_time=float(np.sum([r["intrusion_time"] for r in rows.values()])),
                    restricted_time=float(np.sum([r["time_in_restricted_area"] for r in rows.values()])))


def tracks_figure(episodes, path, label):
    """One panel per episode: tracks of all controlled aircraft (one colour each), scripted aircraft in grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scripts.compare import _xy, RED, GREY, FILL, EDGE
    n = len(episodes)
    ncol = min(n, 5)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.6 * nrow + 1.2), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, f in zip(axes.ravel(), episodes):
        sector = np.asarray(f["sector"])
        lat0, lon0 = sector[:, 0].mean(), sector[:, 1].mean()
        for z in f["zones"]:
            ax.fill(*_xy(z, lat0, lon0), color=FILL, zorder=1)
        sx, sy = _xy(sector, lat0, lon0)
        ax.plot(np.append(sx, sx[0]), np.append(sy, sy[0]), color=EDGE, lw=4, solid_joinstyle="round", zorder=2)
        for tr in f["others"]:
            ox, oy = _xy(tr, lat0, lon0)
            ax.plot(ox, oy, color=GREY, lw=0.6, zorder=3)
            ax.plot(ox[0], oy[0], "o", color=GREY, ms=3, zorder=3)
        n_fail = 0
        for k, (ac, tr) in enumerate(sorted(f["ctrl"].items())):
            col = cmap(k % 10)
            x, y = _xy(tr, lat0, lon0)
            ax.plot(x, y, color=col, lw=1.4, zorder=4)
            c = f["close"][ac]
            for i in range(len(x) - 1):
                if c[i] or c[i + 1]:
                    ax.plot(x[i:i + 2], y[i:i + 2], color=RED, lw=3.0, zorder=5, solid_capstyle="round")
            ax.plot(x[0], y[0], "o", color=col, ms=6, zorder=6)
            gx, gy = _xy([f["goals"][ac]], lat0, lon0)
            ax.plot(gx, gy, "*", color=col, ms=11, mec="black", mew=0.5, zorder=6)
            if not f["reached"].get(ac, True):
                n_fail += 1
                ax.plot(x[-1], y[-1], "x", color="black", ms=8, mew=2, zorder=7)
        ax.set_title(f"seed {f['seed']}: {len(f['ctrl']) - n_fail}/{len(f['ctrl'])} goals\n"
                     f"intrusion {f['intrusion_time']:.0f} s, restricted area {f['restricted_time']:.0f} s (sum over aircraft)",
                     loc="left", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
        mx, my = 0.08 * (sx.max() - sx.min()), 0.08 * (sy.max() - sy.min())
        ax.set_xlim(sx.min() - mx, sx.max() + mx)
        ax.set_ylim(sy.min() - my, sy.max() + my)
        x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
        ax.plot([x0, x0 + 20.0], [y0, y0], color="black", lw=2, clip_on=False)
        ax.text(x0, y0, "20 NM", va="bottom", fontsize=8)
    handles = [Line2D([], [], color=cmap(0), marker="o", lw=1.4, label="Controlled aircraft (one colour each): start and flown track"),
               Line2D([], [], color=cmap(0), marker="*", lw=0, ms=11, mec="black", label="Goal of the aircraft with the same colour"),
               Line2D([], [], color=RED, lw=3.0, label="Closer than 5 NM to another aircraft"),
               Line2D([], [], color="black", marker="x", lw=0, ms=8, mew=2, label="End of a flight that did not reach its goal"),
               Patch(color=FILL, label="Restricted area"), Line2D([], [], color=EDGE, lw=4, label="Sector boundary")]
    if any(f["others"] for f in episodes):                 # the multi-agent setting has no scripted aircraft
        handles.insert(4, Line2D([], [], color=GREY, marker="o", ms=3, lw=0.6, label="Scripted aircraft: start and track"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{label}: flown tracks, policy without action noise, one panel per scenario", fontsize=11)
    fig.tight_layout(rect=(0, 0.9 / (4.6 * nrow + 1.2), 1, 0.97), h_pad=4.0)   # room for the scale bar above each title
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


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
    p.add_argument("--gif", default=None, help="GIF path. One file per episode: PATH without .gif + _epI.gif.")
    p.add_argument("--gif-episodes", default=None,
                   help="Episode indexes with a GIF, for example 3,4. Default: all episodes.")
    p.add_argument("--gif-ms", type=int, default=100, help="Display time of one frame (ms). Larger = slower.")
    p.add_argument("--gif-max-frames", type=int, default=400,
                   help="Frames per episode GIF. 400 keeps every decision step (an episode has 300 steps at most).")
    p.add_argument("--gif-size", type=int, default=512, help="GIF side length (pixels).")
    p.add_argument("--no-tracks", action="store_true", help="Do not write figures/NAME_tracks.png.")
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
    gif_eps = set(range(args.episodes)) if args.gif_episodes is None else {int(x) for x in args.gif_episodes.split(",")}
    rec = None if args.no_tracks else TrackRecorder(env)
    rows, gifs, tracks = [], [], []
    for i in range(args.episodes):
        if rec is not None:
            rec.start(args.reset_seed + i)
        ep_rows, fr, turns = rollout_episode(env, fn, args.reset_seed + i, lidar, history,
                                            render=bool(args.gif) and i in gif_eps, rays=not args.no_rays, on_step=rec)
        if rec is not None:
            tracks.append(rec.result(ep_rows))
        if fr:
            gifs.append(str(Path(args.gif).with_suffix("")) + f"_ep{i}.gif")
            save_gif(fr, gifs[-1], size=args.gif_size, max_frames=args.gif_max_frames, duration=args.gif_ms)
        rows += [{"episode_seed": args.reset_seed + i, "aircraft": a, **r, **turn_stats(turns[a])} for a, r in ep_rows.items()]
    env.close()
    cols = METRIC_KEYS + ["ep_return", "mean_abs_turn_deg", "turn_sign_flip_rate"]
    print(f"\n{name}: setting {setting}, aircraft {len(env.ids)}, status EXPLORATORY")
    print("seed aircraft " + " ".join(f"{c[:20]:>20}" for c in cols))
    for r in rows:
        print(f"{r['episode_seed']:>4} {r['aircraft']:>8} " + " ".join(f"{r[c]:>20.2f}" for c in cols))
    mean = window_stats([{c: r[c] for c in cols} for r in rows])
    print(f"{'mean':>13} " + " ".join(f"{mean[c]:>20.2f}" for c in cols))
    tracks_path = Path(args.out_dir) / "figures" / f"{name}_tracks.png"
    if tracks:
        tracks_figure(tracks, tracks_path, name)
    write_json(Path(args.out_dir) / "results" / f"{name}_eval.json", {
        "status": "EXPLORATORY", "name": name, "setting": setting, "n_aircraft": len(env.ids),
        "n_params": n_par, "deterministic": not args.stochastic, "reward": rew, "mean": mean, "rows": rows, "gifs": gifs})

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
        if tracks:
            media["figures/tracks"] = wandb.Image(str(tracks_path))
        for g in gifs:
            media[f"videos/policy_{Path(g).stem.split('_')[-1]}"] = wandb.Video(g, format="gif")
        table = wandb.Table(columns=["episode_seed", "aircraft"] + cols)
        for r in rows:
            table.add_data(r["episode_seed"], r["aircraft"], *[r[c] for c in cols])
        media["held_out/per_aircraft"] = table
        wb.log(media)
        wb.summary.update({f"held_out/{k}": v for k, v in mean.items()})
        wb.finish()


if __name__ == "__main__":
    main()
