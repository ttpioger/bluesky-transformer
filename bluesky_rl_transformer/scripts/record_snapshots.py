"""Video of several snapshots of a run on the same scenarios, side by side (2026-11-23, for the report).

Each panel flies one model (for example the save points 0.5M, 1.5M, 2.5M of one run) on the same scenario seed with
the same test-time options (heading mask, route, fb2). The frames show:
  grey   = sector edge; dark red = restricted areas; star = goal (one per controlled aircraft, MA: small stars)
  blue   = controlled aircraft with its track; black = other aircraft (SA intruders)
  circle = 2.5 NM around each controlled aircraft (two circles touch = loss of separation), red during a loss
  dashed circle around a goal = capture radius 5 km: the flight reaches its goal when it enters this circle, and
           its track ends there. Then the star turns green with the word "reached".
  arrows = heading the policy requested (light grey) and heading executed after the mask:
           green = no change, orange = the mask moved the heading, red = trap fallback (no free heading)
  magenta dot = route point (reference of the action mapping, --mask-route)
Panel title: label, time, running counts of new losses of separation and area entries (own geometry, for the video
only; the scored metrics are in the line under the panel at the end of the flight).

Outputs: OUT.gif (animation) and OUT_final.png (full tracks, for the report).
Usage (from bluesky_rl_transformer/, CPU, about 1 min per model and scenario):
  python -m scripts.record_snapshots --env sa --seeds 3 --models A.pt B.pt C.pt --labels 0.5M 2.0M 5.0M \
      --heading-mask 1 --mask-mode arc --mask-intruders 1 --mask-dist-km 9.6 --mask-route 1 --mask-route-km 3 \
      --out videos/s1_seed3
The model files are not changed. The environment code is not changed: the recorder wraps two methods of FlightEnv
(_mask_turn, _route_ref) on its own instance to read the mask decisions.
"""
from __future__ import annotations
import argparse
import contextlib
import io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.path import Path as MPath                        # noqa: E402
from PIL import Image                                            # noqa: E402
from rl_tf.envs import FlightEnv, rollout_episode, v2_kwargs     # noqa: E402
from rl_tf.heading_mask import add_mask_args, mask_overrides     # noqa: E402

NM2KM = 1.852
CAPTURE_KM = 5.0                                                 # waypoint capture radius of the competition
LOS_KM = 5.0 * NM2KM
ARROW_KM = 9.0                                                   # arrow length = ARROW_SHARE of the panel width
ARROW_SHARE = 0.12


def _xy(lat, lon, c):
    """Local km from the scenario centre (equirectangular, good enough for a sector of this size)."""
    return (np.asarray(lon) - c[1]) * 111.32 * np.cos(np.deg2rad(c[0])), (np.asarray(lat) - c[0]) * 110.57


def fly(model, kind, seed, mask, max_steps=400, n_agents=10):
    """One flight of one model. Returns (scenario, frames, metric rows by agent, controlled ids)."""
    from scripts.competition_eval import _load_member
    import bluesky as bs
    _, fn, rew, lidar, history, c = _load_member(model, "cpu")
    v2 = v2_kwargs({**c, **mask})
    with contextlib.redirect_stdout(io.StringIO()):
        env = FlightEnv("single" if kind == "sa" else "multi", n_agents, env_kw={}, v2=v2, **rew)
    cur = {"mask": {}, "route": {}}
    G = float(env.v2["goal_offset_deg"])
    orig_mt, orig_rr = env._mask_turn, env._route_ref

    def mask_turn(a, qdr, goal_km, hdg, turn, idx=None):
        fb0 = env._mask_stats.get(a, {}).get("fb", 0)
        out = orig_mt(a, qdr, goal_km, hdg, turn, idx)
        fb = env._mask_stats[a]["fb"] > fb0
        cur["mask"][a] = dict(req=qdr + float(turn) * G, exe=qdr + float(out) * G, proj=abs(out - turn) > 1e-9,
                              fb=bool(fb), spd=int(getattr(env, "_fb2_spd", {}).get(a, 0)))
        return out

    def route_ref(a, qdr, dist_nm, hdg):
        q, d = orig_rr(a, qdr, dist_nm, hdg)
        cur["route"][a] = (q, d * NM2KM) if (abs(q - qdr) > 1e-9 or abs(d - dist_nm) > 1e-9) else None
        return q, d

    env._mask_turn, env._route_ref = mask_turn, route_ref
    frames = []

    def on_step():
        ac = [(str(i), float(bs.traf.lat[k]), float(bs.traf.lon[k]), float(bs.traf.hdg[k]))
              for k, i in enumerate(bs.traf.id)]
        ctrl = {env._ac_id(a): a for a in env.agents}
        frames.append(dict(t=float(bs.sim.simt), ac=ac, ctrl=ctrl, mask=dict(cur["mask"]), route=dict(cur["route"])))
        cur["mask"].clear()
        cur["route"].clear()

    with contextlib.redirect_stdout(io.StringIO()):
        rows, _, _ = rollout_episode(env, fn, seed, lidar, history, max_steps=max_steps, on_step=on_step)
    t0 = frames[0]["t"] if frames else 0.0                       # BlueSky keeps one clock per process
    for f in frames:
        f["t"] -= t0
        scen = env.env.unwrapped.scenario
        ids = {a: env._ac_id(a) for a in env.ids}
        env.close()
    return scen, frames, rows, ids


def _events(frames, scen, ids):
    """Running counts per frame: new losses of separation and new area entries of the controlled aircraft."""
    c = scen.center
    areas = [MPath(np.stack(_xy(np.asarray(o.vertices)[:, 0], np.asarray(o.vertices)[:, 1], c), axis=1))
             for o in scen.obstacles]
    ctrl_ids = set(ids.values())
    los_prev, area_prev, n_los, n_area, out = set(), set(), 0, 0, []
    for f in frames:
        pos = {i: np.array(_xy(la, lo, c)) for i, la, lo, _ in f["ac"]}
        los, inarea = set(), set()
        for i in ctrl_ids & set(pos):
            for j, p in pos.items():
                if j != i and np.hypot(*(pos[i] - p)) < LOS_KM:
                    los.add((i, j))
            if any(a.contains_point(pos[i]) for a in areas):
                inarea.add(i)
        n_los += len(los - los_prev)
        n_area += len(inarea - area_prev)
        los_prev, area_prev = los, inarea
        out.append((n_los, n_area, {i for i, _ in los}, inarea))
    return out


def draw(ax, scen, ids, frames, k, ev, label, rows, final=False, extent=None, zoom_km=None):
    c = scen.center
    ax.clear()
    sec = np.asarray(scen.sector)
    sx, sy = _xy(sec[:, 0], sec[:, 1], c)
    ax.fill(np.r_[sx, sx[0]], np.r_[sy, sy[0]], facecolor="#f4f4f4", edgecolor="#888888", lw=1.0)
    for o in scen.obstacles:
        v = np.asarray(o.vertices)
        ox, oy = _xy(v[:, 0], v[:, 1], c)
        ax.fill(ox, oy, facecolor="#c0392b", alpha=0.35, edgecolor="#922b21", lw=0.8)
    f = frames[min(k, len(frames) - 1)]
    kk = min(k, len(frames) - 1)
    agent_of = {ac: a for a, ac in ids.items()}
    for spec in scen.agents:                                     # goals, capture circle, reached mark
        gx, gy = _xy(spec.goal[0], spec.goal[1], c)
        a = agent_of.get(spec.ac_id)
        last = max((j for j, fr in enumerate(frames) if any(i == spec.ac_id for i, *_ in fr["ac"])), default=-1)
        reached = (a in rows and rows[a]["waypoint_reached"] >= 1 and (final or kk >= last))
        col = "#2ca02c" if reached else "#1f6feb"
        ax.add_patch(plt.Circle((gx, gy), CAPTURE_KM, fill=False, ls="--", lw=0.8, color=col))
        ax.plot(gx, gy, marker="*", color=col, ms=12 if len(scen.agents) == 1 else 7, mec="k", mew=0.4)
        if reached:
            ax.text(gx + CAPTURE_KM * 1.1, gy + CAPTURE_KM * 0.3, "reached", fontsize=7 if len(scen.agents) > 1 else 9,
                    color="#2ca02c", weight="bold")
    if zoom_km and not final:                                    # camera on the (first) controlled aircraft
        own = [(la, lo) for i, la, lo, _ in f["ac"] if i in set(ids.values())]
        if own:
            ox, oy = _xy(own[0][0], own[0][1], c)
            extent = (ox - zoom_km, ox + zoom_km, oy - zoom_km, oy + zoom_km)
    arrow_km = ARROW_SHARE * ((extent[1] - extent[0]) if extent is not None else 150.0)
    n_los, n_area, los_ids, in_area = ev[min(k, len(ev) - 1)]
    ctrl_ids = set(ids.values())
    upto = frames[: min(k, len(frames) - 1) + 1]
    for i in {a for fr in upto for a, *_ in fr["ac"]}:          # tracks
        pts = [(la, lo) for fr in upto for a, la, lo, _ in fr["ac"] if a == i]
        if len(pts) > 1:
            px, py = _xy(np.array(pts)[:, 0], np.array(pts)[:, 1], c)
            ax.plot(px, py, color="#1f6feb" if i in ctrl_ids else "#555555", lw=1.0 if i in ctrl_ids else 0.6,
                    alpha=0.8 if i in ctrl_ids else 0.5)
    if not final:
        for i, la, lo, hdg in f["ac"]:
            x, y = _xy(la, lo, c)
            if i in ctrl_ids:
                bad = i in los_ids or i in in_area
                ax.add_patch(plt.Circle((x, y), LOS_KM / 2, fill=False, color="#d62728" if bad else "#1f6feb",
                                        lw=1.2 if bad else 0.7))
                ax.plot(x, y, "o", color="#1f6feb", ms=5)
                a = f["ctrl"].get(i)
                m = f["mask"].get(a) if a is not None else None
                if m is not None:
                    for h, col, w in ((m["req"], "#aaaaaa", 1.4),
                                      (m["exe"], "#d62728" if m["fb"] else ("#ff8c00" if m["proj"] else "#2ca02c"), 2.0)):
                        r = np.deg2rad(h)
                        ax.annotate("", xy=(x + arrow_km * np.sin(r), y + arrow_km * np.cos(r)), xytext=(x, y),
                                    arrowprops=dict(arrowstyle="-|>", color=col, lw=w + 0.6, mutation_scale=12))
                    if m["spd"]:
                        ax.text(x + 0.05 * arrow_km, y - 0.35 * arrow_km, "slow" if m["spd"] < 0 else "fast",
                                fontsize=8, color="#d62728", weight="bold")
                rp = f["route"].get(a) if a is not None else None
                if rp is not None:
                    r = np.deg2rad(rp[0])
                    ax.plot(x + rp[1] * np.sin(r), y + rp[1] * np.cos(r), "o", color="#c71585", ms=6)
            else:
                ax.plot(x, y, "^", color="#222222", ms=5)
    ax.set_aspect("equal")
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    title = f"{label}   t = {f['t']:.0f} s\nLoS {n_los}   area entries {n_area}"
    if final or k >= len(frames) - 1:
        clean = sum(1 for r in rows.values() if r["waypoint_reached"] >= 1 and r["intrusion_events"] == 0
                    and r["restricted_area_events"] == 0 and r["sector_exit_events"] == 0)
        title += f"\nscored: clean {clean}/{len(rows)}, goal {sum(r['waypoint_reached'] >= 1 for r in rows.values())}"
    ax.set_title(title, fontsize=8)


def _legend(fig, n_panels=3):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#aaaaaa", lw=1.4, label="policy request"),
         Line2D([], [], color="#2ca02c", lw=2, label="executed (no change)"),
         Line2D([], [], color="#ff8c00", lw=2, label="executed (mask moved it)"),
         Line2D([], [], color="#d62728", lw=2, label="trap fallback"),
         Line2D([], [], marker="o", color="#c71585", lw=0, label="route point"),
         Line2D([], [], marker="^", color="#222222", lw=0, label="other aircraft"),
         Line2D([], [], marker="o", mfc="none", ls="none", color="#1f6feb", ms=9, label="goal capture radius 5 km"),
         Line2D([], [], marker="*", color="#2ca02c", lw=0, ms=10, mec="k", mew=0.4, label="goal reached")]
    fig.legend(handles=h, loc="lower center", ncol=8 if n_panels >= 3 else 4, fontsize=7, frameon=False)


def _score(frames, rows):
    clean = sum(1 for r in rows.values() if r["waypoint_reached"] >= 1 and r["intrusion_events"] == 0
                and r["restricted_area_events"] == 0 and r["sector_exit_events"] == 0)
    proj = sum(1 for f in frames for m in f["mask"].values() if m["proj"])
    fb = sum(1 for f in frames for m in f["mask"].values() if m["fb"])
    return clean, len(rows), proj, fb


def scan(args, mask, panel_masks=None):
    """Seeds 1..N with the first and the last model (or panel): clean flights, mask corrections, trap fallbacks."""
    first, last = args.models[0], args.models[-1]
    m_first, m_last = (panel_masks[0], panel_masks[-1]) if panel_masks else (mask, mask)
    out = []
    print("seed | first: clean/n proj fb | last: clean/n proj fb", flush=True)
    for seed in range(1, args.scan + 1):
        a = _score(*fly(first, args.env, seed, m_first)[1:3])
        b = _score(*fly(last, args.env, seed, m_last)[1:3]) if (last != first or panel_masks) else a
        out.append((seed, a, b))
        print(f"{seed:4d} | {a[0]}/{a[1]} {a[2]:4d} {a[3]:3d} | {b[0]}/{b[1]} {b[2]:4d} {b[3]:3d}"
              + ("   <- differs" if a[0] != b[0] else ""), flush=True)
    good = sorted(out, key=lambda x: (x[1][0] == x[2][0], -(x[1][3] + x[2][3]), -(x[1][2] + x[2][2])))
    print("suggested seeds (result differs first, then most traps, then most corrections):",
          [s for s, _, _ in good[:8]], flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["sa", "ma"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="scenario seeds (one video per seed; not with --scan)")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--every", type=int, default=2, help="one frame every N decisions (default 2)")
    p.add_argument("--duration-ms", type=int, default=120)
    p.add_argument("--panel-in", type=float, default=4.2, help="panel size in inches (default 4.2)")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--zoom-km", type=float, default=None,
                   help="camera on the controlled aircraft, half width in km (SA; default: whole sector)")
    p.add_argument("--panels", nargs="+", default=None,
                   help="ablation mode: one model (--models X.pt), one panel per preset of the test-time stack: "
                        "m0 (policy only), mask (arcI), route (arcI + route), full (arcI + route + fb2). The route "
                        "margin is --panel-route-km. The mask flags on the command line are not used in this mode.")
    p.add_argument("--panel-route-km", type=float, default=5.0)
    p.add_argument("--scan", type=int, default=0,
                   help="N > 0: no video; fly the first and the last model on seeds 1..N and list the seeds where the "
                        "mask acts or the results differ (to choose scenarios)")
    p.add_argument("--out", required=True, help="output path without extension; _seedS is added")
    add_mask_args(p, from_checkpoint=True)
    args = p.parse_args(argv)
    mask = mask_overrides(args)
    panel_masks = None
    if args.panels:
        arc = dict(heading_mask=1, mask_mode="arc", mask_intruders=1, mask_dist_km=9.6)
        rt = dict(mask_route=1, mask_route_km=args.panel_route_km)
        presets = {"m0": dict(heading_mask=0), "mask": arc, "route": {**arc, **rt}, "full": {**arc, **rt, "mask_fb2": 1}}
        names = {"m0": "policy only (RL)", "mask": "+ mask (arcI)", "route": f"+ route {args.panel_route_km:g} km",
                 "full": "+ fb2 trap fallback"}
        panel_masks = [presets[k] for k in args.panels]
        args.models = [args.models[0]] * len(args.panels)
        args.labels = args.labels or [names[k] for k in args.panels]
    labels = args.labels or [Path(m).stem for m in args.models]
    assert len(labels) == len(args.models), "one label per model"
    if args.scan:
        return scan(args, mask, panel_masks)
    if not args.seeds:
        p.error("--seeds is needed (or --scan N)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        runs = []
        for j, (m, lab) in enumerate(zip(args.models, labels)):
            scen, frames, rows, ids = fly(m, args.env, seed, panel_masks[j] if panel_masks else mask)
            runs.append((scen, frames, rows, ids, _events(frames, scen, ids), lab))
            clean = sum(1 for r in rows.values() if r["waypoint_reached"] >= 1 and r["intrusion_events"] == 0
                        and r["restricted_area_events"] == 0 and r["sector_exit_events"] == 0)
            print(f"seed {seed} {lab}: {len(frames)} frames, clean {clean}/{len(rows)}, "
                  f"intrusions {sum(r['intrusion_events'] for r in rows.values()):g}, "
                  f"area {sum(r['restricted_area_events'] for r in rows.values()):g}", flush=True)
        sec = np.asarray(runs[0][0].sector)
        sx, sy = _xy(sec[:, 0], sec[:, 1], runs[0][0].center)
        pad = 5.0
        extent = (sx.min() - pad, sx.max() + pad, sy.min() - pad, sy.max() + pad)
        n = len(runs)
        fig, axes = plt.subplots(1, n, figsize=(args.panel_in * n, args.panel_in + 0.9), dpi=args.dpi)
        axes = np.atleast_1d(axes)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.08 if n >= 3 else 0.12, wspace=0.03)
        _legend(fig, n)
        n_frames = max(len(r[1]) for r in runs)
        images = []
        for k in list(range(0, n_frames, args.every)) + [n_frames - 1]:
            for ax, (scen, frames, rows, ids, ev, lab) in zip(axes, runs):
                draw(ax, scen, ids, frames, k, ev, lab, rows, extent=extent, zoom_km=args.zoom_km)
            fig.canvas.draw()
            images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        for ax, (scen, frames, rows, ids, ev, lab) in zip(axes, runs):
            draw(ax, scen, ids, frames, len(frames) - 1, ev, lab, rows, final=True, extent=extent)
        out = f"{args.out}_seed{seed}"
        fig.savefig(out + "_final.png", dpi=150)
        plt.close(fig)
        images += [images[-1]] * 15                               # hold the last frame
        pal = [im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in images]
        pal[0].save(out + ".gif", save_all=True, append_images=pal[1:], duration=args.duration_ms, loop=0,
                    optimize=True)
        print(f"wrote {out}.gif ({len(images)} frames) and {out}_final.png", flush=True)


if __name__ == "__main__":
    main()
