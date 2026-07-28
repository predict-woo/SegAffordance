"""Measure lateral alignment between the depth map and the RGB frame.

Why this matters: every geometric check in tools/sf3d_process.py happens in
laser-scan space.  The occlusion test in PointCloudToImageMapper compares
projected laser points against the DEPTH map, so if depth is itself derived
from the laser scan, that test is close to tautological and cannot notice
that the RGB photo shows something else.  A lateral (perpendicular to the
viewing ray) misregistration between the scan and the colour camera is
invisible to a depth-vs-depth probe, but it is exactly what shifts masks off
the object you see in the image.

Method: estimate the translation between the gradient-magnitude image of the
RGB frame and that of the depth frame by phase correlation.  Object
boundaries appear in both, so a consistent non-zero shift across many frames
means the two are misregistered; a shift scattered around zero means they
are aligned and mask offsets must come from somewhere else.

Also reports depth smoothness: laser-rendered depth is far smoother on flat
surfaces than ARKit sensor depth, which tells us which one hires_depth is.
"""

import argparse
import pickle
import random
from pathlib import Path

import cv2
import lmdb
import numpy as np


def grad_mag(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    p99 = np.percentile(m, 99)
    return np.clip(m / max(1e-6, p99), 0, 1).astype(np.float32)


def roughness(depth_m):
    """Median |depth - 5x5 median| on valid pixels, in mm. Rendered depth is
    near-zero; sensor depth is several mm or more."""
    valid = depth_m > 0
    if valid.sum() < 1000:
        return float("nan")
    med = cv2.medianBlur(depth_m, 5)
    resid = np.abs(depth_m - med)[valid]
    return float(np.median(resid) * 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb-root", type=Path, default=Path("/workspace/datasets/sf3d_processed"))
    ap.add_argument("--lmdb-path", type=Path, default=Path("/dev/shm/data.lmdb"))
    ap.add_argument("--key-cache", type=Path, default=Path("/workspace/cache/sf3d_lmdb_keys.pkl"))
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--work-width", type=int, default=960)
    args = ap.parse_args()

    keys = pickle.loads(args.key_cache.read_bytes())
    chosen = random.Random(args.seed).sample(keys, args.num_samples)

    env = lmdb.open(str(args.lmdb_path), readonly=True, lock=False,
                    readahead=False, meminit=False)
    print(f"{'frame':>5}  {'dx_px':>7} {'dy_px':>7} {'resp':>6}  {'depth_z':>7} "
          f"{'lat_err_cm':>10}  {'rough_mm':>8}")
    print("-" * 62)
    rows = []
    with env.begin() as txn:
        for i, k in enumerate(chosen, 1):
            rec = pickle.loads(txn.get(k))
            rgb = cv2.imread(str(args.lmdb_root / "images" / rec["rgb_image_path"]),
                             cv2.IMREAD_GRAYSCALE)
            dname = rec.get("depth_image_path")
            if rgb is None or not dname:
                continue
            dpath = args.lmdb_root / "depth" / dname
            if not dpath.is_file():
                continue
            raw = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue

            H, W = rgb.shape[:2]
            depth = cv2.resize(raw.astype(np.float32) / 1000.0, (W, H),
                               interpolation=cv2.INTER_NEAREST)
            rough = roughness(depth)

            s = args.work_width / W
            rgb_s = cv2.resize(rgb, (args.work_width, int(H * s)))
            dep_s = cv2.resize(depth, (args.work_width, int(H * s)),
                               interpolation=cv2.INTER_NEAREST)
            # zero (invalid) depth would create fake edges: inpaint-ish fill
            invalid = dep_s <= 0
            if invalid.any():
                dep_s = dep_s.copy()
                dep_s[invalid] = np.median(dep_s[~invalid]) if (~invalid).any() else 0.0

            a = grad_mag(rgb_s.astype(np.float32))
            b = grad_mag(dep_s)
            win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(b * win, a * win)
            dx, dy = dx / s, dy / s      # back to full-res pixels

            K = np.asarray(rec["camera_intrinsics"], dtype=np.float64)
            z = float(np.median(depth[depth > 0])) if (depth > 0).any() else float("nan")
            lat = np.hypot(dx, dy) * z / K[0, 0] * 100.0   # cm at median depth

            rows.append((dx, dy, resp, z, lat, rough))
            print(f"{i:>5}  {dx:>+7.1f} {dy:>+7.1f} {resp:>6.3f}  {z:>7.2f} "
                  f"{lat:>10.1f}  {rough:>8.2f}")
    env.close()

    if rows:
        a = np.array([(r[0], r[1], r[2], r[4], r[5]) for r in rows], dtype=float)
        print("\n--- summary ---")
        print(f"  dx: mean {a[:, 0].mean():+.1f} px, sd {a[:, 0].std():.1f}")
        print(f"  dy: mean {a[:, 1].mean():+.1f} px, sd {a[:, 1].std():.1f}")
        print(f"  |shift| as lateral error at median depth: mean {a[:, 3].mean():.1f} cm")
        print(f"  phase-correlation response: mean {a[:, 2].mean():.3f} "
              f"(higher = more trustworthy)")
        print(f"  depth roughness: median {np.nanmedian(a[:, 4]):.2f} mm "
              f"(<1 mm suggests depth RENDERED from the laser scan; "
              f"several mm suggests sensor depth)")


if __name__ == "__main__":
    main()
