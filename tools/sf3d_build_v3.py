"""Derive sf3d_processed_v3 from v2: prismatic sweeps 0.1 m -> 0.7 m.

v2's synthetic trans trajectories travel a fixed, arbitrary 0.1 m while the
90-deg revolute arcs average 0.73 m (median 0.699 m over the gen-9 split,
measured 2026-08-16 by tools/diag_arc_length.py). v3 regenerates the trans
trajectories at TRANS_LENGTH_M = 0.7 (user decision: the revolute median)
so per-point supervision energy is comparable across motion types.

Derivation needs NO raw SceneFun3D data (the scratch volume may be gone):
the ray is recomputed from each record's stored frame_specific_motion_data
origin/direction — the same inputs the v2 preprocessor used — and the 2D
track is reprojected with the record's stored intrinsics. Everything else
copies as RAW BYTES (guaranteed identical, no pickle round-trip).

Per-record policy (trans rows only; rot rows always raw-copied):
  * stored polyline length in (0.05, 0.15) m — the standard 0.1 m synthesis:
      - frame origin/dir present, |dir| > 1e-8, traj[0] == origin (1e-6):
        exact recompute, linspace(0, 0.7) ray, same point count.
      - otherwise: uniform rescale about traj[0] (keeps traj[0], the 3D
        point target, and its projection, the 2D point target).
      - 2D track: reprojected from the new 3D points when the record has
        camera_intrinsics + image_dimensions_wh; kept as stored otherwise.
  * any other length (the 0.01 m degenerate fallbacks): kept as stored.

Run (dev pod; ~15 min):
  /opt/venv/bin/python -u tools/sf3d_build_v3.py \
    --src /workspace/datasets/sf3d_processed_v2 \
    --dst /workspace/datasets/sf3d_processed_v3
Smoke first with --limit 2000 --dst /root/v3_smoke.
"""
import argparse
import os
import pickle

import lmdb
import numpy as np

TRANS_LENGTH_M = 0.7
STD_LEN_LO, STD_LEN_HI = 0.05, 0.15


def project_trajectory_to_2d(trajectory_cam, K, width, height):
    """Mirror of tools/sf3d_process.py:299 (not imported: that module pulls
    the SceneFun3D SDK at import time). Keep the two in sync."""
    pts = np.asarray(trajectory_cam, dtype=np.float64).reshape(-1, 3)
    coords = np.zeros((len(pts), 2), dtype=np.float64)
    if len(pts) == 0:
        return coords.tolist(), []
    z = pts[:, 2]
    in_front = z > 1e-6
    if in_front.any():
        homo = (np.asarray(K, dtype=np.float64) @ pts[in_front].T).T
        coords[in_front] = homo[:, :2] / homo[:, 2:3]
    valid = (
        in_front
        & (coords[:, 0] >= 0)
        & (coords[:, 0] < float(width))
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < float(height))
    )
    return coords.tolist(), valid.tolist()


def rebuild_trans_record(rec):
    """Returns (new_trajectory_3d, mode) or (None, reason) if kept as-is."""
    traj = rec.get("trajectory_3d_camera_coords")
    if not traj:
        return None, "no_traj"
    t = np.asarray(traj, dtype=np.float64)
    length = float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())
    if not (STD_LEN_LO < length < STD_LEN_HI):
        return None, "degenerate_kept"

    frame = (rec.get("motion_info") or {}).get("frame_specific_motion_data") or {}
    origin = frame.get("motion_origin_3d_camera_coords")
    direction = frame.get("motion_dir_3d_camera_coords")
    n = len(t)
    if origin is not None and direction is not None:
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        dn = float(np.linalg.norm(d))
        if dn > 1e-8 and float(np.linalg.norm(t[0] - o)) < 1e-6:
            ts = np.linspace(0.0, TRANS_LENGTH_M, n)
            ray = o[None, :] + ts[:, None] * (d / dn)[None, :]
            return ray.tolist(), "recomputed"
    # Fallback: pure rescale about the stored start point.
    s = TRANS_LENGTH_M / length
    ray = t[0][None, :] + s * (t - t[0][None, :])
    return ray.tolist(), "rescaled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/datasets/sf3d_processed_v2")
    ap.add_argument("--dst", default="/workspace/datasets/sf3d_processed_v3")
    ap.add_argument("--limit", type=int, default=0, help="smoke: stop after N records")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    dst_lmdb = os.path.join(args.dst, "data.lmdb")
    if os.path.exists(os.path.join(dst_lmdb, "data.mdb")):
        raise SystemExit(f"{dst_lmdb} already exists — refusing to overwrite")

    src_env = lmdb.open(
        os.path.join(args.src, "data.lmdb"), readonly=True, lock=False
    )
    dst_env = lmdb.open(dst_lmdb, map_size=64 * 2**30)

    stats = {
        "total": 0, "rot_or_other_raw": 0, "trans_recomputed": 0,
        "trans_rescaled": 0, "trans_degenerate_kept": 0, "trans_no_traj": 0,
        "trans_2d_reprojected": 0, "trans_2d_kept_no_K": 0,
    }
    new_lengths = []

    with src_env.begin() as stxn:
        cursor = stxn.cursor()
        wtxn = dst_env.begin(write=True)
        pending = 0
        for key, val in cursor:
            stats["total"] += 1
            if stats["total"] % 25000 == 0:
                print(f"progress {stats['total']}", flush=True)
            if args.limit and stats["total"] > args.limit:
                break

            if key == b"__metadata__":
                meta = pickle.loads(val)
                meta["version"] = 3
                meta["derived_from"] = "sf3d_processed_v2"
                meta["trans_traj_length_m"] = TRANS_LENGTH_M
                meta["derivation"] = "tools/sf3d_build_v3.py (2026-08-16)"
                wtxn.put(key, pickle.dumps(meta))
                pending += 1
                continue

            rec = pickle.loads(val)
            original = (rec.get("motion_info") or {}).get("original_motion_data") or {}
            if original.get("motion_type", "trans") in ("rot", "rotation"):
                wtxn.put(key, val)  # raw copy, byte-identical
                stats["rot_or_other_raw"] += 1
            else:
                new_traj, mode = rebuild_trans_record(rec)
                if new_traj is None:
                    wtxn.put(key, val)
                    stats["trans_no_traj" if mode == "no_traj"
                          else "trans_degenerate_kept"] += 1
                else:
                    rec["trajectory_3d_camera_coords"] = new_traj
                    K = rec.get("camera_intrinsics")
                    wh = rec.get("image_dimensions_wh")
                    if K is not None and wh and wh[0] > 0 and wh[1] > 0:
                        c2d, v2d = project_trajectory_to_2d(
                            new_traj, K, wh[0], wh[1]
                        )
                        rec["trajectory_2d_image_coords"] = c2d
                        rec["trajectory_2d_valid"] = v2d
                        stats["trans_2d_reprojected"] += 1
                    else:
                        stats["trans_2d_kept_no_K"] += 1
                    wtxn.put(key, pickle.dumps(rec))
                    stats[f"trans_{mode}"] += 1
                    t = np.asarray(new_traj)
                    new_lengths.append(
                        float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())
                    )
            pending += 1
            if pending >= 5000:
                wtxn.commit()
                wtxn = dst_env.begin(write=True)
                pending = 0
        wtxn.commit()

    dst_env.sync()
    dst_env.close()

    # Layout: frames + full-res images/depth are shared, not duplicated.
    for link, target in (
        ("frames.lmdb", "../sf3d_processed_v2/frames.lmdb"),
        ("images", "../sf3d_processed/images"),
        ("depth", "../sf3d_processed/depth"),
    ):
        path = os.path.join(args.dst, link)
        if not os.path.lexists(path):
            os.symlink(target, path)

    print("\n=== sf3d_build_v3 done ===")
    for k, v in stats.items():
        print(f"  {k:24s} {v}")
    if new_lengths:
        a = np.asarray(new_lengths)
        print(f"  new trans lengths: mean {a.mean():.4f}  min {a.min():.4f}  "
              f"max {a.max():.4f}  (target {TRANS_LENGTH_M})")


if __name__ == "__main__":
    main()
