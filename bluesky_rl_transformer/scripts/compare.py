"""Compare the trained policy with the fly-straight baseline (single agent). No training.

Outputs (NAME = model file name):
- figures/NAME_tracks.png     flown tracks, row 1 = fly straight (zero action), row 2 = the policy, one column per seed;
- figures/NAME_curves.png     training curves from results/NAME_summary.json (if the file exists);
- gifs/NAME_sa.gif            the policy on the track seeds, with the lidar rays (--no-rays gives plain frames);
- gifs/NAME_vs_straight.gif   fly straight (left) and the policy (right), same scenarios;
- results/NAME_compare.json   per-seed metrics and mean, std over the evaluation seeds, for both policies.

Both policies fly the same scenario seeds. The metrics come from the unchanged environment (`info`).
--workers processes run the flights in parallel (BlueSky allows one environment per process).

    PYTHONPATH=.. ../.venv/bin/python -m scripts.compare --model models/sa_tf_s0.pt
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from rl_tf.envs import FlightEnv, REWARD_DEFAULTS, METRIC_KEYS, ObsBuilder, metrics_row, turn_stats
from rl_tf.lidar import Lidar, NM2KM, DEG2KM
from rl_tf.common import setup_torch, write_json, wandb_init, add_wandb_args

EVAL_SEED0 = 900_000_000
CURVE_KEYS = ["ep_return", "waypoint_reached", "intrusion_time", "sector_exit_events"]
BLUE, RED, GREY, FILL, EDGE = "#2a78d6", "#d62728", "#7a7a7a", "#e6e4de", "#c2c2b0"


def fly(env, policy_fn, seed, lidar, history, render=False, max_steps=400, rays=True):
    """One single-agent flight. Returns the metrics and the tracks (lat, lon) of all aircraft."""
    import bluesky as bs
    obs = env.reset(seed=seed)
    inner = env.env.unwrapped
    own_id = inner.agent
    builder = ObsBuilder(lidar, history)
    builder.reset(obs["own"])
    own, close, others, turns, frames = [], [], {}, [], []

    def record():
        i = bs.traf.id2idx(own_id)
        if i < 0:
            return
        lat, lon = float(bs.traf.lat[i]), float(bs.traf.lon[i])
        own.append((lat, lon))
        d_min = np.inf
        for j, ac in enumerate(bs.traf.id):
            if j == i:
                continue
            p = (float(bs.traf.lat[j]), float(bs.traf.lon[j]))
            others.setdefault(ac, []).append(p)
            d = np.hypot((p[0] - lat) * DEG2KM, (p[1] - lon) * DEG2KM * np.cos(np.deg2rad(lat)))
            d_min = min(d_min, d)
        close.append(bool(d_min < inner.intrusion_distance * NM2KM))

    grab = (lambda o: env.render_rays(o, lidar)) if rays else (lambda o: env.render())
    record()
    if render:
        frames.append(grab(obs))
    ret, steps, row = 0.0, 0, None
    while env.agents and steps < max_steps:
        act = policy_fn({"own": obs["own"]}, {"own": builder.get()})
        obs, rew, term, trunc, infos = env.step(act)
        steps += 1
        record()
        if render:
            frames.append(grab(obs))
        ret += rew["own"]
        turns.append(float(act["own"][0]))
        builder.push(obs["own"], act["own"])
        if term["own"] or trunc["own"]:
            row = metrics_row(infos["own"], ret)
    sc = inner.scenario
    return dict(seed=seed, metrics={**row, **turn_stats(turns)}, own=own, close=close,
                others=list(others.values()), zones=[np.asarray(z.vertices).tolist() for z in sc.obstacles],
                sector=np.asarray(sc.sector).tolist(), goal=(float(inner.goal_lat), float(inner.goal_lon)),
                frames=frames)


def _job(job):
    """Worker: all flights of one seed list for both policies (one BlueSky environment per process)."""
    import contextlib
    import io
    model, seeds, render_seeds, device, rays = job
    setup_torch(0)
    from rl_tf.networks import load_policy
    from rl_tf.common import policy_fn_from
    policy, ck = load_policy(model, device)
    a = ck["args"]
    # A checkpoint from before 2026-11-11 has no goal-distance term: 0 keeps the reward of its training.
    rew = {**REWARD_DEFAULTS, "goal_distance_penalty": 0.0, **{k: a[k] for k in REWARD_DEFAULTS if k in a}}
    lidar = Lidar(n_rays=a["n_rays"], fov_deg=a["fov_deg"], max_range_km=a["max_range_km"],
                  zone_flag=bool(a.get("zone_flag", 0)), speed=bool(a.get("lidar_speed", 0)))          # no key: checkpoint from before the flag
    fns = {"straight": lambda raw, built: {k: np.zeros(2, dtype=np.float32) for k in raw},
           "policy": policy_fn_from(policy, device, deterministic=True)}
    with contextlib.redirect_stdout(io.StringIO()):
        env = FlightEnv("single", render_mode="rgb_array" if render_seeds else None, **rew)
    out = []
    for s in seeds:
        out.append({k: fly(env, fn, s, lidar, a["history"], render=s in render_seeds, rays=rays) for k, fn in fns.items()})
    env.close()
    return out


def run_flights(model, seeds, render_seeds, workers, device="cpu", rays=True):
    workers = max(1, min(workers, len(seeds)))
    chunks = [seeds[i::workers] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_job, [(model, c, set(render_seeds), device, rays) for c in chunks])
    by_seed = {r["straight"]["seed"]: r for chunk in res for r in chunk}
    return [by_seed[s] for s in seeds]


def violation_time(m):
    """Seconds in violation: closer than 5 NM to another aircraft, in a restricted area, or outside the sector."""
    return m["intrusion_time"] + m["time_in_restricted_area"] + m["time_outside_sector"]


# ------------------------------------------------------------------ figures
def _xy(latlon, lat0, lon0):
    p = np.asarray(latlon, dtype=np.float64).reshape(-1, 2)
    return (p[:, 1] - lon0) * np.cos(np.deg2rad(lat0)) * DEG2KM / NM2KM, (p[:, 0] - lat0) * DEG2KM / NM2KM


def draw_tracks(ax, f, title):
    sector = np.asarray(f["sector"])
    lat0, lon0 = sector[:, 0].mean(), sector[:, 1].mean()
    for z in f["zones"]:
        ax.fill(*_xy(z, lat0, lon0), color=FILL, zorder=1)
    x, y = _xy(sector, lat0, lon0)
    ax.plot(np.append(x, x[0]), np.append(y, y[0]), color=EDGE, lw=4, solid_joinstyle="round", zorder=2)
    for tr in f["others"]:
        ox, oy = _xy(tr, lat0, lon0)
        ax.plot(ox, oy, color=GREY, lw=0.6, zorder=3)
        ax.plot(ox[0], oy[0], "o", color=GREY, ms=3, zorder=3)
    x, y = _xy(f["own"], lat0, lon0)
    ax.plot(x, y, color=BLUE, lw=1.8, zorder=4)
    c = np.asarray(f["close"])
    for i in range(len(x) - 1):
        if c[i] or c[i + 1]:
            ax.plot(x[i:i + 2], y[i:i + 2], color=RED, lw=2.4, zorder=5, solid_capstyle="round")
    ax.plot(x[0], y[0], "o", color=BLUE, ms=6, zorder=6)
    gx, gy = _xy([f["goal"]], lat0, lon0)
    ax.plot(gx, gy, "*", color="black", ms=9, zorder=6)
    m = f["metrics"]
    extra = "" if m["waypoint_reached"] >= 0.5 else ", goal not reached"
    ax.set_title(f"{title}, seed {f['seed']}\n{violation_time(m):.0f} s in violation{extra}", loc="left", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    sx, sy = _xy(sector, lat0, lon0)                      # view = the sector plus a margin; tracks outside are cut
    mx, my = 0.08 * (sx.max() - sx.min()), 0.08 * (sy.max() - sy.min())
    ax.set_xlim(sx.min() - mx, sx.max() + mx)
    ax.set_ylim(sy.min() - my, sy.max() + my)
    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    ax.plot([x0, x0 + 20.0], [y0, y0], color="black", lw=2, clip_on=False)
    ax.text(x0, y0, "20 NM", va="bottom", fontsize=8)


def tracks_figure(flights, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    n = len(flights)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 8.6), squeeze=False)
    for j, f in enumerate(flights):
        draw_tracks(axes[0, j], f["straight"], "fly straight")
        draw_tracks(axes[1, j], f["policy"], label)
    handles = [Line2D([], [], color=BLUE, marker="o", lw=1.8, label="Controlled aircraft: start and flown track"),
               Line2D([], [], color=RED, lw=2.4, label="Closer than 5 NM to another aircraft"),
               Line2D([], [], color="black", marker="*", lw=0, ms=9, label="Goal"),
               Line2D([], [], color=GREY, marker="o", ms=3, lw=0.6, label="Scripted aircraft: start and track"),
               Patch(color=FILL, label="Restricted area"), Line2D([], [], color=EDGE, lw=4, label="Sector boundary")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Single agent: fly straight (top) and the policy without action noise (bottom), same scenarios.\n"
                 "Violation = seconds closer than 5 NM to another aircraft + in a restricted area + outside the sector.", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def curves_figure(summary, path, bins=120):
    """Training curves: mean over the episodes that end in each bin of environment steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eps = summary["episodes"]
    steps = np.asarray([e["env_steps"] for e in eps], dtype=np.float64)
    edges = np.linspace(0.0, steps.max(), min(bins, max(2, len(eps) // 5)) + 1)
    idx = np.clip(np.digitize(steps, edges) - 1, 0, len(edges) - 2)
    units = {"ep_return": "reward of this folder, unscaled, no shaping", "waypoint_reached": "share of flights",
             "intrusion_time": "s per flight", "sector_exit_events": "events per flight"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, k in zip(axes.ravel(), CURVE_KEYS):
        v = np.asarray([e[k] for e in eps], dtype=np.float64)
        keep = [b for b in range(len(edges) - 1) if (idx == b).any()]
        ax.plot([edges[b + 1] for b in keep], [v[idx == b].mean() for b in keep], color="#2e8b57", lw=1.2)
        ax.set_title(k, fontsize=10)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(units[k])
        ax.grid(alpha=0.3)
    fig.suptitle(f"{summary['name']}: training flights (with action noise), seed {summary['seed']}", fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def save_gif(frames, path, size=256, max_frames=120, duration=80):
    from PIL import Image
    if len(frames) > max_frames:
        frames = frames[::int(np.ceil(len(frames) / max_frames))]
    w = int(round(size * frames[0].shape[1] / frames[0].shape[0]))
    images = [Image.fromarray(f).resize((w, size), Image.BILINEAR) for f in frames]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], loop=0, duration=duration, optimize=True)
    print(f"Wrote {path} ({len(images)} frames)")


def side_by_side(a, b):
    n = max(len(a), len(b))
    a, b = a + [a[-1]] * (n - len(a)), b + [b[-1]] * (n - len(b))      # the shorter flight holds its last frame
    return [np.concatenate([x, y], axis=1) for x, y in zip(a, b)]


def aggregate(flights):
    keys = METRIC_KEYS + ["violation_time", "ep_return", "mean_abs_turn_deg", "turn_sign_flip_rate"]
    out = {}
    for pol in ["straight", "policy"]:
        v = {k: np.asarray([f[pol]["metrics"][k] for f in flights]) for k in keys}
        out[pol] = {k: {"mean": float(v[k].mean()), "std": float(v[k].std())} for k in keys}
    d = np.asarray([f["policy"]["metrics"]["violation_time"] - f["straight"]["metrics"]["violation_time"] for f in flights])
    out["violation_time_policy_minus_straight"] = {
        "mean": float(d.mean()), "std": float(d.std()), "n_policy_better": int((d < 0).sum()),
        "n_equal": int((d == 0).sum()), "n_policy_worse": int((d > 0).sum())}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4, 5], help="Scenario seeds of the track figure and the GIFs.")
    p.add_argument("--eval-episodes", type=int, default=50, help="Held-out scenarios (seed 900000000 and up) for the table.")
    p.add_argument("--label", default="lidar Transformer", help="Name of the policy in the figure.")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-rays", action="store_true", help="GIF frames without the lidar rays.")
    p.add_argument("--name", default=None, help="Default: model file name.")
    p.add_argument("--out-dir", default=".")
    add_wandb_args(p)
    args = p.parse_args(argv)
    name, out = args.name or Path(args.model).stem, Path(args.out_dir)
    eval_seeds = [EVAL_SEED0 + i for i in range(args.eval_episodes)]
    render = [] if args.no_gif else args.seeds
    flights = run_flights(args.model, args.seeds + eval_seeds, render, args.workers, rays=not args.no_rays)
    for f in flights:
        for pol in ["straight", "policy"]:
            f[pol]["metrics"]["violation_time"] = violation_time(f[pol]["metrics"])
    shown, held = flights[:len(args.seeds)], flights[len(args.seeds):]

    tracks_figure(shown, out / "figures" / f"{name}_tracks.png", args.label)
    summary_path = out / "results" / f"{name}_summary.json"
    if summary_path.exists():
        curves_figure(json.loads(summary_path.read_text()), out / "figures" / f"{name}_curves.png")
    else:
        print(f"No {summary_path}: no training curves")
    if not args.no_gif:
        save_gif([fr for f in shown for fr in f["policy"]["frames"]], out / "gifs" / f"{name}_sa.gif", max_frames=240)
        save_gif([fr for f in shown for fr in side_by_side(f["straight"]["frames"], f["policy"]["frames"])],
                 out / "gifs" / f"{name}_vs_straight.gif", max_frames=240)

    rows = [{"episode_seed": f["straight"]["seed"], "straight": f["straight"]["metrics"], "policy": f["policy"]["metrics"]}
            for f in flights]
    agg = aggregate(held) if held else None
    cols = ["waypoint_reached", "flight_time", "intrusion_time", "time_in_restricted_area", "time_outside_sector", "violation_time"]
    print(f"\n{name}: status EXPLORATORY, single agent, policy without action noise")
    print(f"{'seed':>10} {'policy':>9} " + " ".join(f"{c[:18]:>18}" for c in cols))
    for r in rows[:len(args.seeds)]:
        for pol in ["straight", "policy"]:
            print(f"{r['episode_seed']:>10} {pol:>9} " + " ".join(f"{r[pol][c]:>18.1f}" for c in cols))
    if agg:
        for pol in ["straight", "policy"]:
            print(f"{'held-out':>10} {pol:>9} " + " ".join(f"{agg[pol][c]['mean']:>9.1f} ±{agg[pol][c]['std']:>7.1f}" for c in cols))
        print(f"violation_time, policy minus straight, {len(held)} held-out scenarios: {agg['violation_time_policy_minus_straight']}")
    write_json(out / "results" / f"{name}_compare.json", {
        "status": "EXPLORATORY", "name": name, "model": args.model, "setting": "single", "deterministic": True,
        "track_seeds": args.seeds, "eval_seeds": eval_seeds, "std": "population std over the held-out scenarios (one model seed)",
        "held_out": agg, "rows": rows})

    wb = wandb_init(args.wandb, args.wandb_project, name)
    if wb is not None:
        import wandb
        media = {"figures/tracks": wandb.Image(str(out / "figures" / f"{name}_tracks.png"))}
        if summary_path.exists():
            media["figures/training_curves"] = wandb.Image(str(out / "figures" / f"{name}_curves.png"))
        if not args.no_gif:
            media["videos/policy"] = wandb.Video(str(out / "gifs" / f"{name}_sa.gif"), format="gif")
            media["videos/straight_vs_policy"] = wandb.Video(str(out / "gifs" / f"{name}_vs_straight.gif"), format="gif")
        table = wandb.Table(columns=["episode_seed", "flight"] + cols)
        for r in rows:
            for pol in ["straight", "policy"]:
                table.add_data(r["episode_seed"], pol, *[r[pol][c] for c in cols])
        media["held_out/per_scenario"] = table
        wb.log(media)
        if agg:
            for pol in ["straight", "policy"]:
                wb.summary.update({f"held_out/{pol}/{k}_{s}": v[s] for k, v in agg[pol].items() for s in ["mean", "std"]})
            wb.summary.update({f"held_out/violation_time_policy_minus_straight/{k}": v
                               for k, v in agg["violation_time_policy_minus_straight"].items()})
        wb.finish()


if __name__ == "__main__":
    main()
