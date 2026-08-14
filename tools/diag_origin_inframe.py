"""How often does the revolute origin project inside the camera frame?

Informs the gen-7 origin-head design (heatmap + depth needs an in-frame
2D target): for revolute val samples, project (a) the ANNOTATED origin
and (b) q* — the axis point perpendicular to the GT element point, the
actual supervision target since gen-6 — and report in-frame rates.

Run on a pod from the repo root:
  python tools/diag_origin_inframe.py \
      --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010.pkl \
      --min-revolute-radius 0.10 --num 1500
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.scenefun3d import (  # noqa: E402
    SF3DDataset, get_default_transforms, split_dataset_by_scene,
)


def project(K, p):
    """K (3,3), p (3,) camera coords -> (u, v, z)."""
    z = float(p[2])
    u = float(K[0, 0] * p[0] / p[2] + K[0, 2])
    v = float(K[1, 1] * p[1] / p[2] + K[1, 2])
    return u, v, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v2")
    ap.add_argument("--key-cache", default="/workspace/cache/sf3d_v2_keys_cutoff05_minrad010.pkl")
    ap.add_argument("--min-revolute-radius", type=float, default=0.10)
    ap.add_argument("--num", type=int, default=1500)
    ap.add_argument("--min-mask-area-frac", type=float, default=0.0,
                    help="must match the key cache / training config")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    r, m, d = get_default_transforms((256, 256))
    ds = SF3DDataset(
        lmdb_data_root=args.data_root, key_cache_path=args.key_cache,
        frame_cache_path=os.path.join(args.data_root, "frames.lmdb"),
        rgb_transform=r, mask_transform=m, depth_transform=d,
        image_size_for_mask_reconstruction=(256, 256),
        point_source="element", return_trajectory_2d=True,
        min_revolute_radius=args.min_revolute_radius,
        min_mask_area_frac=args.min_mask_area_frac,
    )
    _, val = split_dataset_by_scene(ds, val_split_ratio=0.1, manual_seed=42)
    rng = np.random.default_rng(args.seed)

    n = 0
    stats = {"origin_in": 0, "qstar_in": 0, "both_in": 0}
    margins_q = []
    for i in rng.permutation(len(val)):
        if n >= args.num:
            break
        s = val[int(i)]
        (_, _, _, _, _, _, motion, ty, img_size, _, origin, K, traj3d,
         _, _) = s
        if int(ty) != 1:
            continue
        n += 1
        w, h = float(img_size[0]), float(img_size[1])
        dhat = F.normalize(motion.float(), dim=0)
        p0 = traj3d[0].float()
        o = origin.float()
        q_star = o + ((p0 - o) @ dhat) * dhat

        def in_frame(pt):
            u, v, z = project(K.float(), pt)
            return (z > 0.05 and 0 <= u < w and 0 <= v < h), u, v

        oin, _, _ = in_frame(o)
        qin, qu, qv = in_frame(q_star)
        stats["origin_in"] += int(oin)
        stats["qstar_in"] += int(qin)
        stats["both_in"] += int(oin and qin)
        if qin:
            margins_q.append(min(qu, w - qu, qv, h - qv) / max(w, h))
        if n % 300 == 0:
            print(f"  {n} revolute samples...", flush=True)

    print(f"\nn revolute = {n} (split already radius-filtered)")
    print(f"annotated origin in frame: {100.0 * stats['origin_in'] / n:.1f}%")
    print(f"q* in frame:               {100.0 * stats['qstar_in'] / n:.1f}%")
    print(f"both in frame:             {100.0 * stats['both_in'] / n:.1f}%")
    if margins_q:
        mq = np.array(margins_q)
        print(f"q* border margin (frac of long side): p10 {np.percentile(mq, 10):.3f}"
              f"  median {np.percentile(mq, 50):.3f}")


if __name__ == "__main__":
    main()
