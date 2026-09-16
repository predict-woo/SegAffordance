#!/usr/bin/env python3
"""Dump one SceneFun3D validation record for the method figure: the raw 512 px RGB frame, the depth
map (inferno-coloured), the GT part mask, and the GT geometry projected to normalised image coordinates (interaction point, hinge
axis line, sweep arc). Same dataset / split / key cache as tools/sf3d_vis_val.py, so `--idx` is the
`val N` number printed in that tool's panels.

Run on the dev pod:
    python viz/20260914_arthur_method_figure/dump_sample.py --idx 1696 \
        --out viz/20260914_arthur_method_figure/sample
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from datasets.scenefun3d import SF3DDataset, get_default_transforms, split_dataset_by_scene  # noqa: E402
from model.losses.geometric import normalized_intrinsics  # noqa: E402


def project(K_norm, X):
    """(3,) camera-frame point -> normalised uv in [0, 1]."""
    x = K_norm @ X
    return (x[:2] / max(float(x[2]), 1e-6)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v3")
    ap.add_argument("--frame-cache-path", default="/workspace/datasets/sf3d_frames_512.lmdb")
    ap.add_argument("--key-cache", default="/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ray-len", type=float, default=0.6)
    a = ap.parse_args()
    r, m, d = get_default_transforms(image_size=(512, 512))
    ds = SF3DDataset(
        lmdb_data_root=a.data_root, lmdb_path=f"{a.data_root}/data.lmdb", rgb_transform=r, mask_transform=m,
        depth_transform=d, image_size_for_mask_reconstruction=(512, 512), point_source="element",
        key_cache_path=a.key_cache, return_trajectory_2d=True, frame_cache_path=a.frame_cache_path,
        fast_pipeline=True, load_depth=True, min_revolute_radius=0.10, min_mask_area_frac=0.001, edge_margin_frac=0.05,
    )
    _, va = split_dataset_by_scene(ds, 0.1, 42)
    (img_t, depth_t, desc, mask_t, _bbox, point_gt, motion_gt, type_gt, img_size, fname,
     origin_3d, K, traj3d, traj2d_px, valid2d) = va[a.idx]
    os.makedirs(a.out, exist_ok=True)
    frame = img_t.permute(1, 2, 0).numpy()
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{a.out}/frame.png", frame[:, :, ::-1])
    # depth: metres -> inferno colormap over the valid range (invalid = 0 -> black)
    dep = depth_t[0].float().numpy() if depth_t is not None else None
    if dep is not None and np.isfinite(dep).any() and (dep > 0).any():
        valid = dep > 0
        lo, hi = np.percentile(dep[valid], 2), np.percentile(dep[valid], 98)
        dn = np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1)
        dn8 = (dn * 255).astype(np.uint8)
        dcol = cv2.applyColorMap(dn8, cv2.COLORMAP_INFERNO)
        dcol[~valid] = 0
        cv2.imwrite(f"{a.out}/depth.png", cv2.resize(dcol, (512, 512), interpolation=cv2.INTER_NEAREST))
    mask = (mask_t[0].numpy() > 0.5).astype(np.uint8) * 255
    cv2.imwrite(f"{a.out}/mask.png", cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST))
    W, H = float(img_size[0]), float(img_size[1])
    K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())[0].numpy()
    gt_dir = motion_gt.float().numpy()
    gt_dir = gt_dir / max(float(np.linalg.norm(gt_dir)), 1e-8)
    p0 = traj3d[0].float().numpy()
    o3 = origin_3d.float().numpy()
    rot = int(type_gt) == 1
    anchor = o3 if rot else p0
    t0 = -a.ray_len if rot else 0.0
    axis_uv = [project(K_norm, anchor + t * gt_dir) for t in np.linspace(t0, a.ray_len, 9)]
    tuv = (traj2d_px / torch.tensor([W, H])).numpy().tolist()
    if dep is not None:
        np.save(f"{a.out}/depth.npy", dep.astype(np.float32))          # metres, 0 = invalid
    meta3d = {
        "K_norm": K_norm.tolist(), "type": "rot" if rot else "trans",
        "origin_3d": o3.tolist(), "axis_dir": gt_dir.tolist(), "point_3d": p0.tolist(),
        "traj3d": traj3d.float().numpy().tolist(),
    }
    json.dump(meta3d, open(f"{a.out}/meta3d.json", "w"), indent=1)
    meta = {
        "val_index": a.idx, "file": str(fname), "description": desc,
        "type": "rot" if rot else "trans", "point_uv": point_gt.tolist(),
        "origin_uv": project(K_norm, o3) if rot else None,
        "axis_uv": axis_uv, "traj_uv": tuv, "traj_valid": [bool(v) for v in valid2d.numpy().tolist()],
        "image_size": [W, H],
    }
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    print("wrote", a.out, "|", desc)


if __name__ == "__main__":
    main()
