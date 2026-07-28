"""Diagnose one SF3D annotation across all frames it appears in.

Answers two questions the sample panels raise:

1. Where does trajectory_3d_camera_coords[0] actually sit relative to the
   annotated element?  By construction in tools/sf3d_process.py the arc lies
   in the plane through the MOTION ORIGIN perpendicular to the rotation axis,
   so traj[0] is the element centroid slid ALONG the axis until it is level
   with the origin.  This reports that slide in metres: the element's own
   points (back-projected from the depth map inside the mask) have a non-zero
   axis coordinate, while every trajectory point has axis coordinate 0.

2. Is the laser-scan -> image projection actually aligned?  The motion origin
   is a real laser-scan point on the object, so if the camera pose is right
   the depth map sampled at the projected origin pixel should agree with
   motion_origin_3d_camera_coords[2].  A consistent signed gap across frames
   indicates a pose / coordinate-frame problem rather than occlusion.

Usage:
    python tools/sf3d_inspect_annot.py --annot-id 7d0386a2-... \
        --lmdb-path /dev/shm/data.lmdb --out-dir /tmp/inspect --max-frames 8
"""

import argparse
import pickle
from pathlib import Path

import cv2
import lmdb
import numpy as np

from sf3d_vis_samples import build_mask, render_sample


def backproject(mask, depth_m, K):
    """3D camera-frame points for mask pixels that have valid depth."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros((0, 3))
    z = depth_m[ys, xs]
    ok = z > 0
    ys, xs, z = ys[ok], xs[ok], z[ok]
    if not len(z):
        return np.zeros((0, 3))
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def analyse(rec, lmdb_root):
    out = {}
    rgb = cv2.imread(str(lmdb_root / "images" / rec["rgb_image_path"]))
    if rgb is None:
        return None
    h, w = rgb.shape[:2]
    K = np.asarray(rec["camera_intrinsics"], dtype=np.float64)
    mask = build_mask(rec, h, w)

    depth_m = None
    dname = rec.get("depth_image_path")
    if dname and (lmdb_root / "depth" / dname).is_file():
        raw = cv2.imread(str(lmdb_root / "depth" / dname), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            depth_m = cv2.resize(raw.astype(np.float32) / 1000.0, (w, h),
                                 interpolation=cv2.INTER_NEAREST)

    fsm = (rec.get("motion_info") or {}).get("frame_specific_motion_data") or {}
    omd = (rec.get("motion_info") or {}).get("original_motion_data") or {}
    out["motion_type"] = omd.get("motion_type", "?")
    o3 = np.asarray(fsm.get("motion_origin_3d_camera_coords", [np.nan] * 3), dtype=np.float64)
    d3 = np.asarray(fsm.get("motion_dir_3d_camera_coords", [np.nan] * 3), dtype=np.float64)
    o2 = fsm.get("motion_origin_2d_image_coords")
    out["origin_z_m"] = float(o3[2])
    out["mask_px"] = int((mask > 0).sum())

    # --- alignment probe: depth map vs the origin's own laser-scan depth ---
    if o2 and depth_m is not None:
        u, v = int(round(o2[0])), int(round(o2[1]))
        if 0 <= u < w and 0 <= v < h:
            patch = depth_m[max(0, v - 3): v + 4, max(0, u - 3): u + 4]
            valid = patch[patch > 0]
            if valid.size:
                out["depth_at_origin_m"] = float(np.median(valid))
                out["origin_depth_gap_m"] = float(np.median(valid) - o3[2])

    # --- where the element really is, relative to the arc plane ---
    if depth_m is not None and np.isfinite(d3).all() and np.linalg.norm(d3) > 1e-8:
        d = d3 / np.linalg.norm(d3)
        pts = backproject(mask, depth_m, K)
        if len(pts):
            rel = pts - o3[None, :]
            axis_coord = rel @ d                      # along the rotation axis
            perp = rel - np.outer(axis_coord, d)
            out["elem_axis_offset_m"] = float(np.median(axis_coord))
            out["elem_radius_m"] = float(np.median(np.linalg.norm(perp, axis=1)))
            out["elem_pts"] = int(len(pts))
            # How far the element extends ALONG the axis: if the element is a
            # bar parallel to the hinge, removing the along-axis component
            # slides traj[0] along the bar rather than leaving it at the
            # centroid.
            out["elem_axis_extent_m"] = float(np.percentile(axis_coord, 95)
                                              - np.percentile(axis_coord, 5))
            if len(pts) >= 3:
                c = pts - pts.mean(axis=0)
                principal = np.linalg.svd(c, full_matrices=False)[2][0]
                cosang = abs(float(principal @ d))
                out["elem_axis_angle_deg"] = float(np.degrees(np.arccos(np.clip(cosang, 0, 1))))

    traj = np.asarray(rec.get("trajectory_3d_camera_coords") or [],
                      dtype=np.float64).reshape(-1, 3)
    if len(traj) and np.isfinite(d3).all() and np.linalg.norm(d3) > 1e-8:
        d = d3 / np.linalg.norm(d3)
        rel0 = traj[0] - o3
        out["traj0_axis_offset_m"] = float(rel0 @ d)   # 0 by construction
        out["traj0_radius_m"] = float(np.linalg.norm(rel0 - (rel0 @ d) * d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot-id", help="inspect every frame containing this annot_id")
    ap.add_argument("--key", help="inspect one exact LMDB key "
                                  "(visit/video/timestamp/annot_id)")
    ap.add_argument("--contains", help="further restrict --annot-id frames to keys "
                                       "containing this substring (e.g. a video_id)")
    ap.add_argument("--lmdb-root", type=Path, default=Path("/workspace/datasets/sf3d_processed"))
    ap.add_argument("--lmdb-path", type=Path, default=Path("/dev/shm/data.lmdb"))
    ap.add_argument("--key-cache", type=Path, default=Path("/workspace/cache/sf3d_lmdb_keys.pkl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/inspect"))
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--panel-width", type=int, default=620)
    args = ap.parse_args()

    if args.key:
        picked = [args.key.encode()]
        print(f"inspecting single key {args.key}")
    else:
        keys = pickle.loads(args.key_cache.read_bytes())
        hits = [k for k in keys if k.decode().endswith(args.annot_id)
                and (not args.contains or args.contains in k.decode())]
        print(f"{len(hits)} frames contain annot_id {args.annot_id}")
        if not hits:
            return
        step = max(1, len(hits) // args.max_frames)
        picked = hits[::step][: args.max_frames]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(args.lmdb_path), readonly=True, lock=False,
                    readahead=False, meminit=False)
    hdr = ("frame  mask_px  origin_z  depth@origin   gap    elem_axis_off  "
           "elem_radius  traj0_axis_off  traj0_radius")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    with env.begin() as txn:
        for i, k in enumerate(picked, 1):
            rec = pickle.loads(txn.get(k))
            a = analyse(rec, args.lmdb_root)
            if a is None:
                print(f"{i:>5}  <rgb missing>")
                continue
            rows.append(a)
            print(f"{i:>5}  {a['mask_px']:>7}  {a['origin_z_m']:>8.3f}  "
                  f"{a.get('depth_at_origin_m', float('nan')):>11.3f}  "
                  f"{a.get('origin_depth_gap_m', float('nan')):>+6.3f}  "
                  f"{a.get('elem_axis_offset_m', float('nan')):>+13.3f}  "
                  f"{a.get('elem_radius_m', float('nan')):>11.3f}  "
                  f"{a.get('traj0_axis_offset_m', float('nan')):>+14.3f}  "
                  f"{a.get('traj0_radius_m', float('nan')):>12.3f}")
            panel, _, _ = render_sample(k.decode(), rec, args.lmdb_root, args.panel_width)
            if panel is not None:
                cv2.imwrite(str(args.out_dir / f"frame{i:02d}_{k.decode().split('/')[2]}.jpg"),
                            panel, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    env.close()

    if rows:
        g = [r["origin_depth_gap_m"] for r in rows if "origin_depth_gap_m" in r]
        ax = [r["elem_axis_offset_m"] for r in rows if "elem_axis_offset_m" in r]
        er = [r["elem_radius_m"] for r in rows if "elem_radius_m" in r]
        tr = [r["traj0_radius_m"] for r in rows if "traj0_radius_m" in r]
        print("\n--- across frames ---")
        if g:
            print(f"  depth-at-origin minus origin_z : mean {np.mean(g):+.3f} m, "
                  f"sd {np.std(g):.3f} m   (0 = pose consistent; large negative = "
                  f"origin projects onto something nearer)")
        if ax:
            print(f"  element axis offset from arc plane: mean {np.mean(ax):+.3f} m, "
                  f"sd {np.std(ax):.3f} m   (how far the arc sits from the element)")
        if er and tr:
            print(f"  element true radius {np.mean(er):.3f} m  vs  "
                  f"traj0 radius {np.mean(tr):.3f} m")
        ext = [r["elem_axis_extent_m"] for r in rows if "elem_axis_extent_m" in r]
        ang = [r["elem_axis_angle_deg"] for r in rows if "elem_axis_angle_deg" in r]
        if ext:
            print(f"  element extent ALONG the axis  : mean {np.mean(ext):.3f} m "
                  f"(traj[0] is slid to axis-coord 0, i.e. somewhere in this span)")
        if ang:
            print(f"  angle(element principal axis, rotation axis): "
                  f"mean {np.mean(ang):.1f} deg  (near 0 = element is a bar "
                  f"PARALLEL to the hinge -> the slide runs along the bar)")


if __name__ == "__main__":
    main()
