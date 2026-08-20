"""Decompose the 2D reprojection error of the predicted 3D trajectory.

THE metric of the 2D-only arm: GT 2D track vs the predicted 3D curve
projected into the image (the exact math of TrajectoryProjectionLoss).
Splits the error into what the aggregate hides:

  e_raw    — mean uv distance, absolute (what the loss optimizes)
  e_anchor — uv distance between the projected ANCHOR (point_uv lifted
             with input depth) and the GT track's first point
  e_shape  — mean uv distance after aligning first points (trajectory
             SHAPE error, placement removed)
  gt_scale — the GT track's own motion scale (RMS about its first point),
             the natural yardstick for the other three

If e_anchor >> e_shape, the loss was dominated by point placement, not
trajectory shape — the trajectory head may be fine while the anchor drags
the metric.

  /opt/venv/bin/python -u tools/diag_proj2d.py \
    --config config/sf3d_train_runpod_g17_2donly.yaml \
    --ckpt experiments/20260818_sf3d_g17_2donly/checkpoints/best-epoch27-valloss2.3453.ckpt \
    --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl \
    --data-root /workspace/datasets/sf3d_processed_v3 \
    --input-size 512 --frame-cache-path /workspace/datasets/sf3d_frames_512.lmdb \
    --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05
"""
import argparse
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
    backproject_points,
    normalized_intrinsics,
    project_points,
)
from tools.sf3d_vis_predictions import load_model  # noqa: E402


def summarize(name, arr):
    a = np.asarray([x for x in arr if np.isfinite(x)])
    if len(a) == 0:
        print(f"  {name}: (empty)")
        return
    print(f"  {name:22s} n={len(a):5d}  mean {a.mean():.4f}  p50 "
          f"{np.percentile(a, 50):.4f}  p90 {np.percentile(a, 90):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v3")
    ap.add_argument("--key-cache", required=True)
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--frame-cache-path", default=None)
    ap.add_argument("--min-revolute-radius", type=float, default=0.0)
    ap.add_argument("--min-mask-area-frac", type=float, default=0.0)
    ap.add_argument("--edge-margin-frac", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.config, args.ckpt, device)
    sz = (args.input_size, args.input_size)
    fcache = args.frame_cache_path or os.path.join(args.data_root, "frames.lmdb")
    r, m, d = get_default_transforms(sz)
    ds = SF3DDataset(
        lmdb_data_root=args.data_root, key_cache_path=args.key_cache,
        frame_cache_path=fcache, rgb_transform=r, mask_transform=m,
        depth_transform=d, image_size_for_mask_reconstruction=sz,
        point_source="element", return_trajectory_2d=True,
        min_revolute_radius=args.min_revolute_radius,
        min_mask_area_frac=args.min_mask_area_frac,
        edge_margin_frac=args.edge_margin_frac,
    )
    _, val = split_dataset_by_scene(ds, val_split_ratio=0.1, manual_seed=42)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(val), size=min(args.samples, len(val)), replace=False)
    print(f"probing {len(idx)} val rows with {args.ckpt}", flush=True)

    rows = {"rot": {k: [] for k in ("e_raw", "e_anchor", "e_shape", "gt_scale")},
            "trans": {k: [] for k in ("e_raw", "e_anchor", "e_shape", "gt_scale")}}
    skipped = 0
    for j, vi in enumerate(idx):
        s = val[int(vi)]
        (img_t, depth_t, desc, _mask, _bbox, _pt, motion_gt, type_gt,
         img_size, _name, _origin3d, K, _t3d, t2d_px, t2d_valid) = s
        K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
        with torch.no_grad():
            word = model.tokenize([desc], 77).to(device)
            out = model(img_t[None].to(device), depth_t[None].to(device),
                        word, None, None, None, None, K_norm.to(device).float())
        # replicate the loss's projection chain exactly
        coords = out.point_uv.float()
        grid = (coords * 2.0 - 1.0).view(-1, 1, 1, 2)
        z = F.grid_sample(depth_t[None].to(device).float(), grid,
                          align_corners=False).view(-1)
        if float(z) <= 1e-3:
            skipped += 1
            continue
        anchor = backproject_points(K_norm.to(device).float(), coords, z)
        curve = anchor.unsqueeze(1) + out.trajectory_pred.float()
        in_front = curve[..., 2] > 0.05
        proj = project_points(K_norm.to(device).float(), curve)[0]      # (N,2)
        track = (t2d_px.float() / img_size.float().clamp(min=1.0)).to(device)
        valid = (t2d_valid.bool().to(device) if t2d_valid is not None
                 else torch.ones(track.shape[0], dtype=torch.bool, device=device))
        vmask = valid & in_front[0]
        if int(vmask.sum()) < 2:
            skipped += 1
            continue
        p, t = proj[vmask], track[vmask]
        e_raw = (p - t).norm(dim=-1).mean()
        e_anchor = (proj[0] - track[valid][0]).norm()
        e_shape = ((p - p[0]) - (t - t[0])).norm(dim=-1).mean()
        gt_scale = (t - t[0]).norm(dim=-1).pow(2).mean().sqrt()
        b = rows["rot" if int(type_gt) else "trans"]
        b["e_raw"].append(float(e_raw)); b["e_anchor"].append(float(e_anchor))
        b["e_shape"].append(float(e_shape)); b["gt_scale"].append(float(gt_scale))
        if (j + 1) % 100 == 0:
            print(f"  {j + 1}/{len(idx)}", flush=True)

    print(f"\nskipped (depth hole / behind camera): {skipped}")
    for t in ("rot", "trans"):
        print(f"\n=== {t} rows (uv units, image = 1.0) ===")
        for k in ("e_raw", "e_anchor", "e_shape", "gt_scale"):
            summarize(k, rows[t][k])


if __name__ == "__main__":
    main()
