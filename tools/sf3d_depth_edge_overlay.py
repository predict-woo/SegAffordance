"""Overlay laser-scan depth discontinuities on the RGB frame.

hires_depth is rendered from the laser scan using the frame's camera pose
(depth roughness measures 0.00 mm - see tools/sf3d_depth_rgb_align.py), so
its silhouettes are where the SCAN thinks objects are.  Drawing them on the
photo shows directly whether the pose used to project annotations matches
what the camera actually saw.

Aligned  -> red depth edges trace the real object boundaries.
Misaligned -> red edges float over flat surfaces / miss obvious objects,
which is exactly the condition that puts masks on the wrong thing.  The
occlusion test in sf3d_process cannot detect this, because it compares laser
points against this same laser-rendered depth using the same pose: both are
wrong together and therefore agree.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import lmdb
import numpy as np


def overlay(rec, lmdb_root, out_path, max_w=1400):
    rgb = cv2.imread(str(lmdb_root / "images" / rec["rgb_image_path"]))
    dname = rec.get("depth_image_path")
    raw = cv2.imread(str(lmdb_root / "depth" / dname), cv2.IMREAD_UNCHANGED) if dname else None
    if rgb is None or raw is None:
        print(f"  missing rgb/depth for {out_path.name}")
        return
    H, W = rgb.shape[:2]
    depth = cv2.resize(raw.astype(np.float32) / 1000.0, (W, H), interpolation=cv2.INTER_NEAREST)

    # depth discontinuities: relative jump larger than 4% of local depth
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ref = np.maximum(depth, 0.3)
    edges = ((mag / ref) > 0.04).astype(np.uint8) * 255
    edges[depth <= 0] = 0
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    vis = rgb.copy()
    vis[edges > 0] = (0, 0, 255)

    # mark the stored mask so you can see what it landed on
    coords = np.asarray(rec.get("mask_coordinates_yx", []), dtype=np.int32)
    if coords.size:
        m = np.zeros((H, W), np.uint8)
        m[np.clip(coords[:, 0], 0, H - 1), np.clip(coords[:, 1], 0, W - 1)] = 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 3)

    s = max_w / W
    pair = np.hstack([
        cv2.resize(rgb, (max_w, int(H * s))),
        np.full((int(H * s), 6, 3), 22, np.uint8),
        cv2.resize(vis, (max_w, int(H * s))),
    ])
    cv2.putText(pair, "RGB", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(pair, "+ laser-depth edges (red), stored mask (green)",
                (max_w + 18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), pair, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--lmdb-root", type=Path, default=Path("/workspace/datasets/sf3d_processed"))
    ap.add_argument("--lmdb-path", type=Path,
                    default=Path("/workspace/datasets/sf3d_processed/data.lmdb"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(args.lmdb_path), readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        for k in args.keys:
            raw = txn.get(k.encode())
            if raw is None:
                print(f"  key not found: {k}")
                continue
            rec = pickle.loads(raw)
            name = k.split("/")[1] + "_" + k.split("/")[2] + ".jpg"
            overlay(rec, args.lmdb_root, args.out_dir / name)
    env.close()


if __name__ == "__main__":
    main()
