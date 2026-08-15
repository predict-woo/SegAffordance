"""Per-sample breakdown of the pred-pred articulation consistency loss (L_pp).

Loads a split-arm checkpoint, forwards chosen VAL samples (same split/seed
as training and as tools/sf3d_vis_predictions.py, so indices match the
`valNNNN` tags in viz panel filenames), and prints every internal term of
PredPredArticulationLoss — plus a reference distribution over a random val
subset so a single sample's value has context.

Example (dev pod):
  /opt/venv/bin/python tools/diag_lpp_samples.py \
    --config config/sf3d_train_runpod_g9_closeup010.yaml \
    --ckpt experiments/20260815_sf3d_g9_closeup010/checkpoints/best-epoch23-valloss0.9191.ckpt \
    --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl \
    --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 \
    --indices 93 1684
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.scenefun3d import (  # noqa: E402
    SF3DDataset,
    get_default_transforms,
    split_dataset_by_scene,
)
from model.losses.geometric import (  # noqa: E402
    PredPredArticulationLoss,
    normalized_intrinsics,
)
from tools.sf3d_vis_predictions import load_model  # noqa: E402


def lpp_breakdown(out, trajectory_is_absolute, radius_floor=0.10):
    """Replicates PredPredArticulationLoss.forward for ONE sample (B=1),
    returning every intermediate the batch-mean loss hides."""
    traj = out.trajectory_pred.float()                      # (1, N, 3)
    d = traj - traj[:, :1] if trajectory_is_absolute else traj
    dhat = F.normalize(out.motion_pred.float(), p=2, dim=1, eps=1e-8)
    dhat_n = dhat[:, None, :]

    l_line = torch.cross(d, dhat_n.expand_as(d), dim=-1).pow(2).sum(-1).mean(-1)

    if trajectory_is_absolute:
        c = (out.origin_pred.float() - traj[:, 0])[:, None, :]
    else:
        c = (out.origin_pred - out.point_3d_pred).float()[:, None, :]

    def _line_dist(x):
        rel = x - c
        along = (rel * dhat_n).sum(-1, keepdim=True)
        return (rel - along * dhat_n).norm(dim=-1)

    r_hat = _line_dist(torch.zeros_like(d[:, :1]))          # (1, 1)
    radial = (_line_dist(d) - r_hat).pow(2).mean(-1)        # (1,)
    axial = (d * dhat_n).sum(-1).pow(2).mean(-1)            # (1,)
    l_circle = radial + axial

    p_rev = out.motion_type_logits.float().softmax(dim=-1)[:, 1]
    per_sample = (1.0 - p_rev) * l_line + p_rev * l_circle
    energy = d.pow(2).sum(dim=(-1, -2))

    # Gen-10 candidate normalization (2026-08-16 spec): dimensionless
    # branches. Line: fraction of mean squared displacement perpendicular
    # to the axis (intrinsically <= 1). Circle: residual relative to the
    # predicted orbit radius, r_hat floored at `radius_floor` (keep it
    # equal to the loss's radius_floor so probe and loss match).
    mean_sq_disp = d.pow(2).sum(-1).mean(-1)                # (1,)
    l_line_norm = l_line / mean_sq_disp.clamp(min=1e-8)
    r2 = r_hat.squeeze(1).clamp(min=radius_floor).pow(2)
    l_circle_norm = (radial + axial) / r2
    per_sample_norm = (1.0 - p_rev) * l_line_norm + p_rev * l_circle_norm

    net = d[0, -1]
    net_n = net.norm().clamp(min=1e-8)
    cos_axis = float((net / net_n * dhat[0]).sum())
    return {
        "p_rev": float(p_rev),
        "l_line_m2": float(l_line),
        "l_circle_m2": float(l_circle),
        "circle_radial_m2": float(radial),
        "circle_axial_m2": float(axial),
        "radial_rms_m": math.sqrt(max(float(radial), 0.0)),
        "axial_rms_m": math.sqrt(max(float(axial), 0.0)),
        "r_hat_m": float(r_hat),
        "L_pp": float(per_sample),
        "L_pp_norm": float(per_sample_norm),
        "l_line_norm": float(l_line_norm),
        "l_circle_norm": float(l_circle_norm),
        "energy_m2": float(energy),
        "net_extent_m": float(net_n),
        "cos(net_disp, axis)": cos_axis,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v2")
    ap.add_argument("--key-cache", required=True)
    ap.add_argument("--min-revolute-radius", type=float, default=0.0)
    ap.add_argument("--min-mask-area-frac", type=float, default=0.0)
    ap.add_argument("--edge-margin-frac", type=float, default=0.0)
    ap.add_argument("--indices", type=int, nargs="+", required=True,
                    help="val indices (the NNNN in viz 'valNNNN' filenames)")
    ap.add_argument("--ref-samples", type=int, default=512,
                    help="random val subset for the reference distribution "
                         "(0 disables)")
    ap.add_argument("--radius-floor", type=float, default=0.10,
                    help="r_hat floor for the normalized circle branch "
                         "(keep = the loss's radius_floor)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-npz", default=None,
                    help="directory to write per-index npz dumps of the "
                         "predicted/GT geometry (for offline 3D plotting; "
                         "matplotlib is not on the pod venv)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mp = load_model(args.config, args.ckpt, device)
    traj_abs = bool(getattr(mp, "trajectory_absolute", False))
    loss_mod = PredPredArticulationLoss(weight=1.0, trajectory_is_absolute=traj_abs)

    r, m, d = get_default_transforms((256, 256))
    ds = SF3DDataset(
        lmdb_data_root=args.data_root,
        key_cache_path=args.key_cache,
        frame_cache_path=os.path.join(args.data_root, "frames.lmdb"),
        rgb_transform=r, mask_transform=m, depth_transform=d,
        image_size_for_mask_reconstruction=(256, 256),
        point_source="element", return_trajectory_2d=True,
        min_revolute_radius=args.min_revolute_radius,
        min_mask_area_frac=args.min_mask_area_frac,
        edge_margin_frac=args.edge_margin_frac,
    )
    _, val = split_dataset_by_scene(ds, val_split_ratio=0.1, manual_seed=42)
    print(f"val samples: {len(val)}   trajectory_absolute={traj_abs}")

    def forward_index(vi):
        s = val[int(vi)]
        (img_t, depth_t, desc, _mask, _bbox, _pt, motion_gt, type_gt,
         img_size, _name, origin3d, K, t3d, _t2d, _t2dv) = s
        K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
        with torch.no_grad():
            word = model.tokenize([desc], 77).to(device)
            out = model(img_t[None].to(device), depth_t[None].to(device),
                        word, None, None, None, None,
                        K_norm.to(device).float())
        return out, desc, int(type_gt), motion_gt, origin3d, t3d

    for vi in args.indices:
        out, desc, type_gt, motion_gt, origin3d, t3d = forward_index(vi)
        total, _ = loss_mod(out, None)
        bd = lpp_breakdown(out, traj_abs, radius_floor=args.radius_floor)
        print(f"\n=== val{vi}  [GT {'rot' if type_gt else 'trans'}]  {desc[:60]}")
        for k, v in bd.items():
            print(f"  {k:22s} {v: .5f}")
        print(f"  {'loss_module L_pp':22s} {float(total): .5f}   (x config weight "
              f"0.5 -> {0.5 * float(total):.5f} in train logs)")
        if args.dump_npz:
            os.makedirs(args.dump_npz, exist_ok=True)
            path = os.path.join(args.dump_npz, f"val{vi}.npz")
            np.savez(
                path,
                desc=np.asarray(desc),
                type_gt=np.asarray(type_gt),
                trajectory_pred=out.trajectory_pred[0].float().cpu().numpy(),
                trajectory_absolute=np.asarray(traj_abs),
                point_3d_pred=out.point_3d_pred[0].float().cpu().numpy(),
                origin_pred=out.origin_pred[0].float().cpu().numpy(),
                motion_pred=out.motion_pred[0].float().cpu().numpy(),
                p_rev=np.asarray(bd["p_rev"]),
                r_hat=np.asarray(bd["r_hat_m"]),
                L_pp=np.asarray(bd["L_pp"]),
                traj_gt=t3d.float().numpy(),
                origin_gt=origin3d.float().numpy(),
                motion_gt=motion_gt.float().numpy(),
            )
            print(f"  dumped -> {path}")

    if args.ref_samples > 0:
        rng = np.random.default_rng(args.seed)
        ref = rng.choice(len(val), size=min(args.ref_samples, len(val)),
                         replace=False)
        vals, p_revs, nvals = [], [], []
        for j, vi in enumerate(ref):
            out = forward_index(vi)[0]
            bd = lpp_breakdown(out, traj_abs, radius_floor=args.radius_floor)
            if bd["energy_m2"] > 1e-6:
                vals.append(bd["L_pp"])
                nvals.append(bd["L_pp_norm"])
                p_revs.append(bd["p_rev"])
            if (j + 1) % 128 == 0:
                print(f"ref {j + 1}/{len(ref)}", flush=True)
        v = np.asarray(vals)
        print(f"\n=== reference over {len(v)} non-degenerate val samples")
        print(f"  L_pp  mean {v.mean():.5f}   p50 {np.percentile(v, 50):.5f}   "
              f"p90 {np.percentile(v, 90):.5f}   p99 {np.percentile(v, 99):.5f}   "
              f"max {v.max():.5f}")
        print(f"  mean p_rev {np.mean(p_revs):.3f}")
        n = np.asarray(nvals)
        print(f"  L_pp_NORM  mean {n.mean():.5f}   p50 {np.percentile(n, 50):.5f}   "
              f"p90 {np.percentile(n, 90):.5f}   p99 {np.percentile(n, 99):.5f}   "
              f"max {n.max():.5f}")


if __name__ == "__main__":
    main()
