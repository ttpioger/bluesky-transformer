"""Pretrain the policy encoder with a temporal, action-conditioned JEPA on recorded flights (2026-11-13).
--masked-coef > 0 adds the masked-token target (I-JEPA style; map block and lidar arc, see rl_tf/jepa.py).
--temporal-coef 0 --masked-coef 1 gives the masked target alone.

Input: the shards of scripts.collect_jepa (--data NAME). The encoder is built from the same token and network
flags as scripts.train, so `scripts.train --init-encoder models/jepa_NAME.pt` with the same flags loads it.
Flights with episode % 10 == 0 are the validation split (never used for the gradient).

After training, linear probes (ridge regression on frozen features, validation flights) measure what the features
hold. Targets: an intrusion starts within the next 6 steps (60 s), an entry into a restricted area within the next
6 steps (AUC each, rows where the event is not already going on), the distance to the goal (R^2). The same probes
run on: the raw input, the encoder at its start weights (same seed), the JEPA encoder, and each --probe-model.

Stop rule: a validation spread (mean standard deviation of the features over the batch) below 0.05 = collapse;
the script then writes the summary with "collapsed": true and exits with code 2.
"""
from __future__ import annotations
import glob
import sys
import time
from pathlib import Path
import numpy as np
import torch

PROBE_STEPS = 6


def load_shards(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no shard matches {pattern}")
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, files


def features(encoder, data, rows, device, batch=1024):
    out = []
    encoder.eval()
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            r = rows[s:s + batch]
            st = torch.as_tensor(data["state"][r].astype(np.float32), device=device)
            li = torch.as_tensor(data["lidar"][r].astype(np.float32), device=device)
            out.append(encoder(st, li).float().cpu().numpy())
    encoder.train()
    return np.concatenate(out) if out else np.zeros((0, encoder.out_dim), np.float32)


def probe_table(feat_fn, data, tr_rows, va_rows, targets):
    """feat_fn(rows) -> [N, D]. Returns {target: score} (AUC or R^2) on the validation rows."""
    from rl_tf.jepa import ridge_fit, ridge_predict, auc
    ftr, fva = feat_fn(tr_rows), feat_fn(va_rows)
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fva = (ftr - mu) / sd, (fva - mu) / sd
    res = {}
    for name, (y, ok, kind) in targets.items():
        mt, mv = ok[tr_rows], ok[va_rows]
        w = ridge_fit(ftr[mt], y[tr_rows][mt].astype(np.float64))
        pv = ridge_predict(w, fva[mv])
        yv = y[va_rows][mv].astype(np.float64)
        if kind == "auc":
            res[name] = auc(pv, yv)
            res[name + "_positives"] = int(yv.sum())
        else:
            res[name] = float(1.0 - ((pv - yv) ** 2).sum() / max(((yv - yv.mean()) ** 2).sum(), 1e-12))
    return res


def main(argv=None):
    from scripts.train import build_parser
    from rl_tf.envs import lidar_cfg_from_args, obs_dims
    from rl_tf.networks import build_policy, save_encoder, n_params, load_policy, encoder_mismatch
    from rl_tf.jepa import TemporalJepa, ema_tau, window_starts, future_labels
    from rl_tf.common import setup_torch, write_json, wandb_init, wandb_upload
    p = build_parser()
    p.description = __doc__
    p.add_argument("--data", required=True, help="Data name of scripts.collect_jepa (files data/jepa_DATA_w*.npz).")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--horizon", type=int, default=3, help="Predicted steps (10 s each).")
    p.add_argument("--jepa-epochs", type=int, default=10)
    p.add_argument("--jepa-batch", type=int, default=256)
    p.add_argument("--jepa-lr", type=float, default=3e-4)
    p.add_argument("--ema-start", type=float, default=0.996, help="Start of the moving-average rate (cosine to 1).")
    p.add_argument("--var-coef", type=float, default=1.0)
    p.add_argument("--cov-coef", type=float, default=0.04)
    p.add_argument("--pred-hidden", type=int, default=512)
    p.add_argument("--temporal-coef", type=float, default=1.0, help="Weight of the temporal target. 0 = off.")
    p.add_argument("--masked-coef", type=float, default=0.0,
                   help="Weight of the masked-token target (map block + lidar arc). 0 = off (as before 2026-11-13).")
    p.add_argument("--map-block", type=int, default=2, help="Masked target: side of the block of map patches.")
    p.add_argument("--lidar-arc", type=int, default=6, help="Masked target: adjacent lidar patches in the arc.")
    p.add_argument("--mpred-dim", type=int, default=128, help="Width of the masked-target predictor.")
    p.add_argument("--mpred-layers", type=int, default=2, help="Layers of the masked-target predictor.")
    p.add_argument("--probe-rows", type=int, default=20000, help="Rows per split for the linear probes.")
    p.add_argument("--probe-model", nargs="*", default=[], help="Policy files whose encoders also get the probes.")
    p.add_argument("--log-every", type=int, default=200, help="Gradient steps between two validation checks.")
    args = p.parse_args(argv)
    setup_torch(args.seed, args.learner_threads)
    dev = args.device
    out = Path(args.out_dir)
    lidar_cfg = lidar_cfg_from_args(args)
    dims = obs_dims(lidar_cfg, args.history)
    data, files = load_shards(str(out / args.data_dir / f"jepa_{args.data}_w*.npz"))
    if data["state"].shape[1] != dims["state_dim"] or data["lidar"].shape[1:] != (dims["lidar_ch"], dims["n_rays"]):
        raise ValueError(f"data shapes {data['state'].shape}, {data['lidar'].shape} do not match the flags: {dims}")
    val_ep = data["episode"] % 10 == 0
    starts = window_starts(data["has_next"], args.horizon)
    tr_starts, va_starts = starts[~val_ep[starts]], starts[val_ep[starts]]
    print(f"{len(data['has_next'])} rows from {len(files)} shards; windows: train {len(tr_starts)}, validation {len(va_starts)}",
          flush=True)

    policy = build_policy(dims, vars(args)).to(dev)            # same start weights as scripts.train at this seed
    start_state = {k: v.clone() for k, v in policy.encoder.state_dict().items()}
    jepa = TemporalJepa(policy.encoder, hidden=args.pred_hidden, var_coef=args.var_coef, cov_coef=args.cov_coef,
                        temporal_coef=args.temporal_coef, masked_coef=args.masked_coef, map_block=args.map_block,
                        lidar_arc=args.lidar_arc, mpred_dim=args.mpred_dim, mpred_layers=args.mpred_layers).to(dev)
    if args.temporal_coef <= 0 and args.masked_coef <= 0:
        raise ValueError("--temporal-coef and --masked-coef are both 0: no target")
    params = [q for q in jepa.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=args.jepa_lr)
    n_par = {"encoder": n_params(policy.encoder), "predictors": jepa.predictor_params(),
             "tokens": policy.encoder.n_tokens}
    wb = wandb_init(args.wandb, args.wandb_project, f"jepa_{args.name}", {**vars(args), "n_params": n_par})
    H, B = args.horizon, args.jepa_batch
    ks = np.arange(H + 1)

    def batch(idx):
        rows = idx[:, None] + ks[None, :]
        st = torch.as_tensor(data["state"][rows].astype(np.float32), device=dev)
        li = torch.as_tensor(data["lidar"][rows].astype(np.float32), device=dev)
        ac = torch.as_tensor(data["action"][rows[:, :H]], device=dev)
        return st, li, ac

    rng = np.random.default_rng(args.seed)
    va_fixed = rng.choice(va_starts, size=min(len(va_starts), 4096), replace=False)

    def validate():
        """Validation: temporal error, masked error (own mask generator, fixed seed), spread."""
        tot, tot_m, n, info = 0.0, 0.0, 0, {}
        jepa.mask_generator = torch.Generator(device=dev).manual_seed(12345)    # the same masks at each check
        with torch.no_grad():
            for s in range(0, len(va_fixed), 512):
                l, info = jepa.loss(*batch(va_fixed[s:s + 512]))
                m = len(va_fixed[s:s + 512])
                tot, tot_m, n = tot + info["pred"] * m, tot_m + info.get("masked", 0.0) * m, n + m
        jepa.mask_generator = None
        return tot / max(n, 1), info["spread"], tot_m / max(n, 1)

    steps_per_epoch = max(1, len(tr_starts) // B)
    total_steps = steps_per_epoch * args.jepa_epochs
    curve, step, collapsed, t0 = [], 0, False, time.time()
    v0, s0, m0 = validate()
    curve.append({"step": 0, "val_pred": v0, "val_spread": s0, "val_masked": m0})
    print(f"start: val_pred={v0:.4f} spread={s0:.3f} n_params={n_par}", flush=True)
    for ep in range(args.jepa_epochs):
        perm = rng.permutation(tr_starts)
        for s in range(steps_per_epoch):
            loss, info = jepa.loss(*batch(perm[s * B:(s + 1) * B]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            jepa.update_target(ema_tau(step, total_steps, args.ema_start))
            if step % args.log_every == 0 or step == total_steps:
                vp, vs, vm = validate()
                row = {"step": step, "epoch": ep, "loss": float(loss.item()), "train_pred": info["pred"],
                       "train_masked": info.get("masked", 0.0), "var": info["var"], "cov": info["cov"],
                       "val_pred": vp, "val_masked": vm, "val_spread": vs, "seconds": time.time() - t0}
                curve.append(row)
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
                if wb is not None:
                    wb.log({f"jepa/{k}": v for k, v in row.items() if k != "step"}, step=step)
                if vs < 0.05:
                    collapsed = True
                    break
        if collapsed:
            print("STOP: collapse (validation spread < 0.05)", flush=True)
            break
    wall = time.time() - t0
    model_path = out / "models" / f"jepa_{args.name}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder(model_path, policy.encoder, vars(args), dims,
                 extra={"jepa": {"horizon": H, "steps": step, "collapsed": collapsed, "data": args.data,
                                 "temporal_coef": args.temporal_coef, "masked_coef": args.masked_coef}})
    print(f"Saved {model_path}", flush=True)

    # linear probes on frozen features, validation flights only for the scores
    rows_all = np.arange(len(data["has_next"]))
    tr_rows = rows_all[~val_ep]
    va_rows = rows_all[val_ep]
    tr_rows = rng.choice(tr_rows, size=min(len(tr_rows), args.probe_rows), replace=False)
    va_rows = rng.choice(va_rows, size=min(len(va_rows), args.probe_rows), replace=False)
    li, ok_i = future_labels(data["f_intr"], data["has_next"], PROBE_STEPS)
    la, ok_a = future_labels(data["f_area"], data["has_next"], PROBE_STEPS)
    targets = {"intrusion_within_60s_auc": (li, ok_i, "auc"), "area_entry_within_60s_auc": (la, ok_a, "auc"),
               "goal_distance_r2": (data["goal_km"], np.ones(len(li), bool), "r2")}
    probes = {}
    raw = lambda r: np.concatenate([data["state"][r].astype(np.float32),
                                    data["lidar"][r].astype(np.float32).reshape(len(r), -1)], axis=1)
    probes["raw input"] = probe_table(raw, data, tr_rows, va_rows, targets)
    fresh = build_policy(dims, vars(args)).to(dev)
    fresh.encoder.load_state_dict(start_state)
    probes["start weights"] = probe_table(lambda r: features(fresh.encoder, data, r, dev), data, tr_rows, va_rows, targets)
    probes["JEPA"] = probe_table(lambda r: features(policy.encoder, data, r, dev), data, tr_rows, va_rows, targets)
    for path in args.probe_model:
        try:
            pol, ck = load_policy(path, dev)
            if encoder_mismatch(ck["args"], ck["dims"], vars(args), dims):
                probes[Path(path).stem] = {"skipped": "token or network settings differ"}
                continue
            probes[Path(path).stem] = probe_table(lambda r: features(pol.encoder, data, r, dev), data, tr_rows, va_rows,
                                                  targets)
        except (OSError, KeyError, RuntimeError) as e:
            probes[Path(path).stem] = {"skipped": str(e)[:200]}
    print("\nlinear probes (validation flights):")
    for k, v in probes.items():
        print(f"  {k:<28} " + "  ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
    summary_path = out / "results" / f"jepa_{args.name}_summary.json"
    write_json(summary_path, {
        "status": "EXPLORATORY", "name": f"jepa_{args.name}", "data": args.data, "shards": files,
        "rows": int(len(data["has_next"])), "windows_train": int(len(tr_starts)), "windows_val": int(len(va_starts)),
        "n_params": n_par, "horizon": H, "temporal_coef": args.temporal_coef, "masked_coef": args.masked_coef, "gradient_steps": step, "collapsed": collapsed, "wall_seconds": wall,
        "seconds_per_step": wall / max(step, 1), "probe_steps": PROBE_STEPS, "probes": probes,
        "probe_rows": {"train": int(len(tr_rows)), "val": int(len(va_rows))}, "args": vars(args), "curve": curve})
    if wb is not None:
        wb.summary.update({f"probe/{k}/{a}": b for k, v in probes.items() for a, b in v.items() if isinstance(b, float)})
        wandb_upload(wb, model_path)
        wandb_upload(wb, summary_path)
        wb.finish()
    sys.exit(2 if collapsed else 0)


if __name__ == "__main__":
    main()
