"""GT-side twist sanity panels: does the GT twist sweep the GT trajectory?

For N stratified val samples, takes the batch EXACTLY as the model/loss see
it (SF3DDataset with the training config's settings), builds the GT twist
with model.losses.twist.twist_from_gt(motion, type, origin_3d), anchors it
at the GT interaction point lifted with the input depth map (the same
conversion the prediction-side losses use), and draws one panel per sample:

    yellow points — the orbit for t in [0, t_max]: the sweep the STORED SIGN
                    implies (t_max = pi/2 revolute / 0.1 m prismatic, the
                    preprocessor's own constants)
    gray points   — the orbit for t in [-t_max, 0]: the opposite sign
    cyan points   — the GT trajectory (projected), start ringed white

If the sign contract holds (tools/sf3d_process.py derives the trajectory
FROM the signed axis), cyan lies on YELLOW everywhere. The header prints the
velocity-field agreement score sign_check=OK|FLIPPED|degenerate per sample,
so a broken sample is unmissable.

Run from the repo root on a pod:
    python tools/sf3d_vis_gt_twist.py --out viz/YYYYMMDD_sf3d_gt_twist_check --num 12
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.scenefun3d import SF3DDataset, get_default_transforms, split_dataset_by_scene
from model.losses.geometric import backproject_points, normalized_intrinsics, project_points
from model.losses.twist import screw_orbit, twist_from_gt
from sf3d_vis_predictions import draw_points_norm, put_lines
from viz_manifest import write_manifest

PANEL_W = 900


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v2")
    ap.add_argument("--key-cache", default="/workspace/cache/sf3d_v2_keys_cutoff05.pkl")
    ap.add_argument("--out", required=True,
                    help="dated batch dir under viz/, e.g. viz/YYYYMMDD_sf3d_gt_twist_check")
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    r, m, d = get_default_transforms((256, 256))
    ds = SF3DDataset(
        lmdb_data_root=args.data_root,
        key_cache_path=args.key_cache,
        frame_cache_path=os.path.join(args.data_root, "frames.lmdb"),
        rgb_transform=r, mask_transform=m, depth_transform=d,
        image_size_for_mask_reconstruction=(256, 256),
        point_source="element", return_trajectory_2d=True,
        fast_pipeline=True,
    )
    _, val = split_dataset_by_scene(ds, val_split_ratio=0.1, manual_seed=42)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(val))
    picks, want = [], {0: args.num // 2, 1: args.num - args.num // 2}
    for i in order:
        s = val[int(i)]
        t = int(s[7])
        if want[t] > 0:
            picks.append((int(i), s))
            want[t] -= 1
        if not want[0] and not want[1]:
            break

    os.makedirs(args.out, exist_ok=True)
    for rank, (vi, s) in enumerate(picks):
        (_img, depth_t, desc, _mask, _bbox, point_gt, motion_gt, type_gt,
         img_size, rgb_name, origin_3d, K, traj3d, _t2d, _t2dv) = s
        frame = cv2.imread(os.path.join(args.data_root, "images", rgb_name))
        if frame is None:
            continue
        scale = PANEL_W / frame.shape[1]
        frame = cv2.resize(frame, (PANEL_W, int(frame.shape[0] * scale)))

        # GT twist exactly as TwistLoss builds it
        twist = twist_from_gt(motion_gt[None], type_gt[None], origin_3d[None])

        # anchor: GT interaction point lifted with the input depth map — the
        # same conversion the prediction-side losses apply to coords_hat
        K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
        grid = (point_gt[None] * 2.0 - 1.0).view(1, 1, 1, 2)
        z = F.grid_sample(depth_t[None].float(), grid, align_corners=False).view(1)
        anchor = backproject_points(K_norm, point_gt[None].float(), z)

        t_max = math.pi / 2.0 if int(type_gt) == 1 else 0.1
        n_pts = 24
        fwd = screw_orbit(twist, anchor, torch.linspace(0.0, t_max, n_pts)[None])[0]
        bwd = screw_orbit(twist, anchor, torch.linspace(-t_max, 0.0, n_pts)[None])[0]

        # sign check: does the twist's velocity field push the GT trajectory
        # points along the GT sweep?
        omega, v = twist[0, :3], twist[0, 3:]
        pts, seg = traj3d[:-1], traj3d[1:] - traj3d[:-1]
        field = torch.linalg.cross(omega.expand_as(pts), pts, dim=-1) + v
        score = (field * seg).sum().item()
        sweep = seg.norm(dim=-1).sum().item()
        verdict = ("degenerate" if sweep < 1e-3 or abs(score) < 1e-6
                   else "OK" if score > 0 else "FLIPPED")

        def project(pts3d):
            vis = (pts3d[:, 2] > 0.05).numpy()
            uv = project_points(K_norm, pts3d[None])[0].clamp(-2, 3).numpy()
            return uv, vis

        p = frame
        uv, vis = project(bwd)
        p = draw_points_norm(p, uv[::-1], vis[::-1], (150, 150, 150), radius=3)
        uv, vis = project(fwd)
        p = draw_points_norm(p, uv, vis, (0, 230, 230), radius=3)
        uv, vis = project(traj3d)
        p = draw_points_norm(p, uv, vis, (255, 220, 0), radius=4)

        t_lbl = "rot" if int(type_gt) else "trans"
        p = put_lines(p, [
            f"GT [{t_lbl}]  sign_check={verdict}  z_anchor={z.item():.2f}m",
            desc[:70],
            "cyan=GT traj  yellow=+t orbit (stored sign)  gray=-t orbit",
        ], color=(255, 255, 255) if verdict == "OK" else (0, 0, 255))
        cv2.imwrite(os.path.join(args.out, f"{rank:02d}_{t_lbl}_{verdict}_val{vi}.jpg"),
                    p, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"wrote {rank:02d}_{t_lbl}_{verdict}_val{vi}.jpg  ({desc[:44]})")

    write_manifest(args.out, num=args.num, seed=args.seed, data_root=args.data_root)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
