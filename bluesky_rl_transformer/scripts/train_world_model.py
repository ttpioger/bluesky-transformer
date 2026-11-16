"""Train the latent world model (predictor + cost heads) on a frozen encoder (2026-11-13). See rl_tf/world_model.py.

Input:
  --encoder-from FILE  the encoder that the planner will use: a policy checkpoint (models/NAME.pt; its encoder part)
                       or a JEPA encoder (models/jepa_NAME.pt). For the planner levels 1 and 2 this must be the
                       policy that the planner flies with (scripts.competition_eval checks that the weights are equal).
                       The token and network settings come from this file; no token flags are needed here.
  --data A [B ...]     data names of scripts.collect_jepa (files data/jepa_A_w*.npz). The data must have the
                       reward column (collect_jepa of 2026-11-13 evening or later) and the reward weights of the
                       policy run (checked against data/jepa_A_meta.json).
Flights with episode % 10 == 0 are the validation split.

Multi-agent data (collect_jepa --setting multi) also gives each aircraft-token row the command of its aircraft
(neighbour commands; --nb-drop of the windows hide them). The value head learns the discounted return to go of the
recorded flights in PPO units (gamma and reward scale of the policy file, or --gamma / --reward-scale).

Validation (fixed windows): for k = 1..H the rollout error of the token vectors, the error of "no change"
(h(t) as the guess for h(t+k)) and their ratio; AUC of the predicted intrusion and restricted-area probability of
step k; R^2 of the predicted reward, goal step and value. The AUCs at k = 3 (30 s) and k = 6 (60 s) are the main numbers.

Output: models/wm_NAME.pt, results/wm_NAME_summary.json.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def build_parser():
    from rl_tf.common import add_wandb_args
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--encoder-from", required=True, help="Policy checkpoint or JEPA encoder file.")
    p.add_argument("--data", nargs="+", required=True, help="Data names of scripts.collect_jepa.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--horizon", type=int, default=6, help="Predicted steps (10 s each).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128, help="Windows per gradient step.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--layers", type=int, default=4, help="Predictor layers.")
    p.add_argument("--heads", type=int, default=4, help="Attention heads of the predictor and the cost heads.")
    p.add_argument("--tf-coef", type=float, default=1.0, help="Weight of the one-step loss from true states.")
    p.add_argument("--s-coef", type=float, default=1.0, help="Weight of the own-state encoding in the latent error.")
    p.add_argument("--head-coef", type=float, default=1.0, help="Weight of the cost-head losses.")
    p.add_argument("--max-pos-weight", type=float, default=100.0, help="Upper limit of the weight of rare positive rows.")
    p.add_argument("--nb-drop", type=float, default=0.3,
                   help="Share of training windows whose neighbour commands are hidden (the model also learns 'unknown').")
    p.add_argument("--gamma", type=float, default=None, help="Discount of the value head. Default: that of the policy.")
    p.add_argument("--reward-scale", type=float, default=None,
                   help="Reward scale of the value head (PPO units). Default: that of the policy.")
    p.add_argument("--val-windows", type=int, default=4096)
    p.add_argument("--val-every", type=int, default=500, help="Gradient steps between two validation checks.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    add_wandb_args(p)
    return p


def main(argv=None):
    from scripts.pretrain_jepa import load_shards
    from rl_tf.networks import build_policy, load_encoder, n_params
    from rl_tf.envs import reward_kwargs
    from rl_tf.jepa import window_starts, auc
    from rl_tf.world_model import (WorldModel, step_labels, save_world_model, returns_to_go, action_lookup,
                                   neighbour_commands)
    from rl_tf.common import setup_torch, write_json, wandb_init, wandb_upload
    args = build_parser().parse_args(argv)
    setup_torch(args.seed, args.threads)
    dev, out, H = args.device, Path(args.out_dir), args.horizon

    ck = torch.load(args.encoder_from, map_location="cpu", weights_only=False)
    enc_args, enc_dims = dict(ck["args"]), dict(ck["dims"])
    pol = build_policy(enc_dims, enc_args)
    load_encoder(pol, args.encoder_from, enc_dims, enc_args)
    run_reward = reward_kwargs(SimpleNamespace(**{**enc_args}))
    if "policy" not in ck and (args.gamma is None or args.reward_scale is None):
        raise ValueError(f"{args.encoder_from} is not a policy file: give --gamma and --reward-scale of the policy run")
    gamma = float(enc_args["gamma"] if args.gamma is None else args.gamma)
    scale = float(enc_args["reward_scale"] if args.reward_scale is None else args.reward_scale)
    K = int(enc_dims.get("ac_tokens", 0))

    parts, files = [], []
    for name in args.data:
        d, f = load_shards(str(out / args.data_dir / f"jepa_{name}_w*.npz"))
        if "reward" not in d:
            raise ValueError(f"data {name} has no reward column: record it again with this version of scripts.collect_jepa")
        meta_path = out / args.data_dir / f"jepa_{name}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("reward") != run_reward:
            raise ValueError(f"data {name}: reward weights {meta.get('reward')} differ from those of {args.encoder_from}: "
                             f"{run_reward}. Record the data with the reward flags of the policy run.")
        d["episode"] = d["episode"].astype(np.int64) + 10_000_000 * len(parts)   # flights of two data sets stay apart
        n = len(d["has_next"])
        if "agent" not in d:                           # data from before the multi-agent columns: all unknown
            d["agent"] = np.zeros(n, np.int8)
            d["nb_ids"] = np.full((n, K), -1, np.int8)
        d["ret"] = returns_to_go(d["reward"], d["has_next"], gamma, scale)
        parts.append(d)
        files += f
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if data["state"].shape[1] != enc_dims["state_dim"] or data["lidar"].shape[1:] != (enc_dims["lidar_ch"], enc_dims["n_rays"]):
        raise ValueError(f"data shapes {data['state'].shape}, {data['lidar'].shape} do not match the encoder: {enc_dims}")

    val_ep = data["episode"] % 10 == 0
    starts = window_starts(data["has_next"], H)
    tr_starts, va_starts = starts[~val_ep[starts]], starts[val_ep[starts]]
    tr_rows = np.unique((tr_starts[:, None] + np.arange(H)[None, :]).ravel())
    flags = np.stack([data["f_intr"], data["f_area"], data["f_out"]], 1)[tr_rows].astype(np.float64)
    pos = flags.mean(0)
    pos_weight = np.clip((1.0 - pos) / np.maximum(pos, 1e-6), 1.0, args.max_pos_weight)
    g = data["goal_km"].astype(np.float64)
    gstep = (g[tr_rows] - g[tr_rows + 1])
    rew = data["reward"][tr_rows].astype(np.float64)
    ret = data["ret"][tr_rows].astype(np.float64)
    norms = [float(rew.mean()), float(rew.std() + 1e-6), float(gstep.std() + 1e-6), float(ret.mean()),
             float(ret.std() + 1e-6)]
    lookup = action_lookup(data)
    multi = bool((data["nb_ids"] >= 0).any())
    print(f"{len(data['has_next'])} rows from {len(files)} shards; windows: train {len(tr_starts)}, validation "
          f"{len(va_starts)}; positive share {dict(zip(['intr', 'area', 'out'], pos.round(4)))}, pos_weight "
          f"{pos_weight.round(1)}, norms {np.round(norms, 4)}, neighbour commands {multi}", flush=True)

    wm = WorldModel(pol.encoder, n_layers=args.layers, n_heads=args.heads).to(dev)
    wm.norms.copy_(torch.tensor(norms))
    pw = torch.tensor(pos_weight, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(wm.params(), lr=args.lr)
    n_par = {"encoder_frozen": n_params(wm.encoder), "trained": sum(p.numel() for p in wm.params())}
    wb = wandb_init(args.wandb, args.wandb_project, f"wm_{args.name}", {**vars(args), "n_params": n_par})
    ks = np.arange(H + 1)

    def batch(idx, drop=0.0):
        rows = idx[:, None] + ks[None, :]
        st = torch.as_tensor(data["state"][rows].astype(np.float32), device=dev)
        li = torch.as_tensor(data["lidar"][rows].astype(np.float32), device=dev)
        ac = torch.as_tensor(data["action"][rows[:, :H]].astype(np.float32), device=dev)
        lab = torch.as_tensor(step_labels(data, idx, H), device=dev)
        nb = None
        if multi and K:
            n = neighbour_commands(data, idx, H, lookup)
            if drop > 0:
                n[rng.random(len(idx)) < drop] = 0.0       # whole window "unknown"
            nb = torch.as_tensor(n, device=dev)
        return st, li, ac, lab, nb

    rng = np.random.default_rng(args.seed)
    va_fixed = np.sort(rng.choice(va_starts, size=min(len(va_starts), args.val_windows), replace=False))

    @torch.no_grad()
    def validate():
        wm.eval()
        lab_all, pr, rw, gs, vl, err, base = [], [], [], [], [], np.zeros(H), np.zeros(H)
        n = 0
        for s in range(0, len(va_fixed), 256):
            st, li, ac, lab, nbt = batch(va_fixed[s:s + 256])
            b = st.shape[0]
            hb, sb, pb = wm.encode(st.reshape(b * (H + 1), -1), li.reshape(b * (H + 1), *li.shape[2:]))
            hb, sb, pb = hb.reshape(b, H + 1, *hb.shape[1:]), sb.reshape(b, H + 1, -1), pb.reshape(b, H + 1, -1)
            h, sv, pad0 = hb[:, 0], sb[:, 0], pb[:, 0]
            p_k, r_k, g_k, v_k = [], [], [], []
            for k in range(1, H + 1):
                h, sv = wm.step(h, sv, pad0, ac[:, k - 1], None if nbt is None else nbt[:, k - 1])
                ok = (~(pad0 | pb[:, k])).float()
                e = ((h - hb[:, k]).pow(2).mean(-1) * ok).sum(1) / ok.sum(1).clamp_min(1.0)
                e0 = ((hb[:, 0] - hb[:, k]).pow(2).mean(-1) * ok).sum(1) / ok.sum(1).clamp_min(1.0)
                err[k - 1] += float(e.sum())
                base[k - 1] += float(e0.sum())
                o = wm.heads(h, sv, pad0)
                p_k.append(wm.probs(o).cpu().numpy())
                r_k.append(wm.reward(o).cpu().numpy())
                g_k.append((o[:, 3] * wm.norms[2]).cpu().numpy())
                v_k.append(wm.value(o).cpu().numpy())
            pr.append(np.stack(p_k, 1))
            rw.append(np.stack(r_k, 1))
            gs.append(np.stack(g_k, 1))
            vl.append(np.stack(v_k, 1))
            lab_all.append(lab.cpu().numpy())
            n += b
        wm.train()
        pr, rw, gs, vl, lab = (np.concatenate(pr), np.concatenate(rw), np.concatenate(gs), np.concatenate(vl),
                               np.concatenate(lab_all))
        r2 = lambda p, y: float(1.0 - ((p - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        res = {"latent_error_per_k": (err / n).tolist(), "no_change_error_per_k": (base / n).tolist(),
               "ratio_per_k": (err / np.maximum(base, 1e-12)).tolist()}
        for j, ev in enumerate(["intrusion", "area", "outside"]):
            res[f"{ev}_auc_per_k"] = [auc(pr[:, k, j], lab[:, k, j]) for k in range(H)]
            res[f"{ev}_positives_per_k"] = [int(lab[:, k, j].sum()) for k in range(H)]
        res["reward_r2_per_k"] = [r2(rw[:, k], lab[:, k, 4]) for k in range(H)]
        res["goal_step_r2_per_k"] = [r2(gs[:, k], lab[:, k, 3]) for k in range(H)]
        res["value_r2_per_k"] = [r2(vl[:, k], lab[:, k, 5]) for k in range(H)]
        return res

    steps_per_epoch = max(1, len(tr_starts) // args.batch)
    total = steps_per_epoch * args.epochs
    curve, step, t0 = [], 0, time.time()
    v = validate()
    curve.append({"step": 0, **v})
    print(f"start: ratio_per_k={np.round(v['ratio_per_k'], 3)} n_params={n_par}", flush=True)
    wm.train()
    for ep in range(args.epochs):
        perm = rng.permutation(tr_starts)
        for s in range(steps_per_epoch):
            st, li, ac, lab, nbt = batch(perm[s * args.batch:(s + 1) * args.batch], args.nb_drop)
            loss, info = wm.loss(st, li, ac, lab, pw, nbt, tf_coef=args.tf_coef, s_coef=args.s_coef,
                                 head_coef=args.head_coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.params(), 1.0)
            opt.step()
            step += 1
            if step % args.val_every == 0 or step == total:
                v = validate()
                row = {"step": step, "epoch": ep, "loss": float(loss.item()), "train_roll": info["roll"],
                       "train_one": info["one"], "train_head": info["head"], "seconds": time.time() - t0, **v}
                curve.append(row)
                k3, k6 = min(2, H - 1), H - 1
                print(f"step={step} epoch={ep} loss={row['loss']:.4f} ratio@k1={v['ratio_per_k'][0]:.3f} "
                      f"ratio@k{H}={v['ratio_per_k'][k6]:.3f} intr_auc@30s={v['intrusion_auc_per_k'][k3]:.3f} "
                      f"intr_auc@{10 * H}s={v['intrusion_auc_per_k'][k6]:.3f} area_auc@{10 * H}s={v['area_auc_per_k'][k6]:.3f} "
                      f"reward_r2@k1={v['reward_r2_per_k'][0]:.3f} value_r2@k{H}={v['value_r2_per_k'][k6]:.3f} "
                      f"sec={row['seconds']:.0f}", flush=True)
                if wb is not None:
                    wb.log({"wm/loss": row["loss"], "wm/ratio_k1": v["ratio_per_k"][0], "wm/ratio_kH": v["ratio_per_k"][k6],
                            "wm/intr_auc_30s": v["intrusion_auc_per_k"][k3], "wm/intr_auc_kH": v["intrusion_auc_per_k"][k6],
                            "wm/area_auc_kH": v["area_auc_per_k"][k6], "wm/reward_r2_k1": v["reward_r2_per_k"][0]}, step=step)
    wall = time.time() - t0
    model_path = out / "models" / f"wm_{args.name}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_world_model(model_path, wm, enc_args, enc_dims, extra={
        "encoder_from": args.encoder_from, "horizon": H, "data": args.data, "pos_weight": pos_weight.tolist(),
        "reward": run_reward, "reward_scale": scale, "gamma": gamma, "neighbour_commands": multi})
    print(f"Saved {model_path}", flush=True)
    summary_path = out / "results" / f"wm_{args.name}_summary.json"
    write_json(summary_path, {
        "status": "EXPLORATORY", "name": f"wm_{args.name}", "encoder_from": args.encoder_from, "data": args.data,
        "shards": files, "rows": int(len(data["has_next"])), "windows_train": int(len(tr_starts)),
        "windows_val": int(len(va_starts)), "val_windows_used": int(len(va_fixed)), "n_params": n_par, "horizon": H,
        "gamma": gamma, "reward_scale": scale, "neighbour_commands": multi, "positive_share": pos.tolist(), "pos_weight": pos_weight.tolist(), "norms": norms, "gradient_steps": step,
        "wall_seconds": wall, "seconds_per_step": wall / max(step, 1), "final_validation": curve[-1],
        "args": vars(args), "curve": curve})
    if wb is not None:
        wandb_upload(wb, model_path)
        wandb_upload(wb, summary_path)
        wb.finish()


if __name__ == "__main__":
    main()
