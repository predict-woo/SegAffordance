"""Overlay the EPIC-Fields COLMAP reconstruction on the EPIC test-record depth clouds (CPU, runs on the pod).

For every record cloud written by tools/epic_cloud_precompute.py (<clouds>/<key>.npz) this looks up the record's
video frame (`epic.onset_frame` of the LMDB record — the EPIC-KITCHENS rgb_frames index, i.e. the frame grid
EPIC-Fields registers), takes the pose of that frame (or of the nearest registered frame within +-max-offset
frames) from /workspace/datasets/epic_fields/<video>.json (see README.md there), transforms the video's sparse
points into the record's camera frame (COLMAP world-to-camera: X_cam = R(q) X + t, OpenCV axes) and estimates
the metric scale that puts them on the monocular depth cloud: project with K_render, read the depth cloud's z
from a z-buffer at those pixels and take median(z_depth / z_fields) over points in front of the camera and
inside the image. The scaled cloud is written back into the SAME npz as xyz_fields (float32 M x 3, metres),
rgb_fields (uint8 M x 3) plus fields_scale, fields_frame (name of the registered frame used), fields_offset
(frame delta to the record's frame), fields_n, fields_inside_frac (fraction of points in front + inside the
image), fields_agree_frac (fraction of those within 25 % of the depth cloud after scaling). Records whose video
has no EPIC-Fields JSON or no pose near the frame keep their npz unchanged (fields_n = 0 in the summary).

  /opt/venv/bin/python tools/epic_fields_overlay.py --clouds /workspace/datasets/epic_gt_annot/clouds \\
      --fields /workspace/datasets/epic_fields [--dry-run] [--limit N]
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

ROOT = "/workspace/datasets/epic_processed_2d"
MAX_OFFSET = 15


def quat_to_rot(qw, qx, qy, qz):
    """COLMAP (w, x, y, z) unit quaternion -> 3x3 rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]], np.float64)


def frame_name(idx):
    return f"frame_{int(idx):010d}.jpg"


def frame_index(name):
    return int(name[len("frame_"):-len(".jpg")])


def nearest_frame(images, frame, max_offset=MAX_OFFSET):
    """(name, offset) of the registered frame closest to `frame` within +-max_offset, else (None, None)."""
    if frame_name(frame) in images:
        return frame_name(frame), 0
    best = None
    for d in range(1, max_offset + 1):
        for cand in (frame + d, frame - d):  # prefer later frames on ties (the hand has arrived by then)
            if cand >= 1 and frame_name(cand) in images:
                return frame_name(cand), cand - frame
    return best, None


def world_to_cam(points_xyz, pose):
    qw, qx, qy, qz, tx, ty, tz = pose
    R = quat_to_rot(qw, qx, qy, qz)
    return points_xyz @ R.T + np.array([tx, ty, tz])


def zbuffer(xyz, K, size, down=2):
    """Min-z buffer of a camera-frame cloud at size/down resolution (NaN where empty)."""
    W, H = int(size[0]) // down, int(size[1]) // down
    z = xyz[:, 2]
    u = np.floor((K[0, 0] * xyz[:, 0] / z + K[0, 2]) / down).astype(np.int64)
    v = np.floor((K[1, 1] * xyz[:, 1] / z + K[1, 2]) / down).astype(np.int64)
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zb = np.full(H * W, np.inf, np.float32)
    np.minimum.at(zb, v[ok] * W + u[ok], z[ok].astype(np.float32))
    zb[~np.isfinite(zb)] = np.nan
    return zb.reshape(H, W), down


def estimate_scale(xyz_f, xyz_d, K, size):
    """(scale, inside_frac, agree_frac): median z_depth/z_fields over fields points in front + inside the image."""
    zb, down = zbuffer(xyz_d, K, size)
    H, W = zb.shape
    z = xyz_f[:, 2]
    front = z > 1e-6
    u = np.full(len(z), -1, np.int64)
    v = np.full(len(z), -1, np.int64)
    u[front] = np.floor((K[0, 0] * xyz_f[front, 0] / z[front] + K[0, 2]) / down)
    v[front] = np.floor((K[1, 1] * xyz_f[front, 1] / z[front] + K[1, 2]) / down)
    inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zd = np.full(len(z), np.nan, np.float32)
    zd[inside] = zb[v[inside], u[inside]]
    valid = inside & np.isfinite(zd)
    if valid.sum() < 20:
        return None, float(inside.mean()), 0.0
    ratio = zd[valid] / z[valid]
    scale = float(np.median(ratio))
    agree = float(np.mean(np.abs(np.log(ratio / scale)) < np.log(1.25)))
    return scale, float(inside.mean()), agree


def load_record_frames(root, keys):
    """{key: (video_id, onset_frame)} straight from the LMDB records (no dataset class needed)."""
    import lmdb
    out = {}
    env = lmdb.open(f"{root}/data.lmdb", readonly=True, lock=False, readahead=False)
    with env.begin() as t:
        for k in keys:
            raw = t.get(k.encode())
            if raw is None:
                continue
            e = pickle.loads(raw)["epic"]
            out[k] = (e["video_id"], int(e["onset_frame"]))
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", default="/workspace/datasets/epic_gt_annot/clouds")
    ap.add_argument("--fields", default="/workspace/datasets/epic_fields")
    ap.add_argument("--root", default=ROOT, help="EPIC LMDB root (for epic.onset_frame / video_id)")
    ap.add_argument("--max-offset", type=int, default=MAX_OFFSET)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="estimate and print, do not rewrite the npz files")
    a = ap.parse_args()
    t0 = time.time()
    records = json.load(open(os.path.join(a.clouds, "index.json")))["records"]
    if a.limit:
        records = records[: a.limit]
    frames = load_record_frames(a.root, [r["key"] for r in records])
    fields_cache, summary = {}, []
    for r in records:
        key, fn = r["key"], os.path.join(a.clouds, r["file"])
        vid, onset = frames.get(key, (key.split("/")[0], None))
        jf = os.path.join(a.fields, f"{vid}.json")
        row = dict(key=key, video=vid, frame=onset, fields_n=0)
        if onset is None or not os.path.exists(jf):
            summary.append(dict(row, why="no LMDB record" if onset is None else "no EPIC-Fields json"))
            print(f"{key:24s} SKIP {summary[-1]['why']}", flush=True)
            continue
        if vid not in fields_cache:
            d = json.load(open(jf))
            pts = np.asarray(d["points"], np.float64)
            fields_cache[vid] = (d["images"], pts[:, :3], np.clip(pts[:, 3:6], 0, 255).astype(np.uint8))
        images, pw, prgb = fields_cache[vid]
        name, off = nearest_frame(images, onset, a.max_offset)
        if name is None:
            idx = np.array([frame_index(k) for k in images])
            summary.append(dict(row, why=f"no registered frame within +-{a.max_offset} of {onset} (nearest is {int(np.abs(idx - onset).min())} away)"))
            print(f"{key:24s} SKIP {summary[-1]['why']}", flush=True)
            continue
        z = dict(np.load(fn, allow_pickle=False))
        K, size = z["K_render"].astype(np.float64), z["size"]
        xc = world_to_cam(pw, images[name])
        scale, inside, agree = estimate_scale(xc, z["xyz"].astype(np.float64), K, size)
        if scale is None:
            summary.append(dict(row, why="too few fields points on the depth cloud", inside_frac=inside))
            print(f"{key:24s} SKIP {summary[-1]['why']} (inside {inside:.2f})", flush=True)
            continue
        xf = (xc * scale).astype(np.float32)
        row.update(fields_n=int(len(xf)), scale=scale, inside_frac=inside, agree_frac=agree, fields_frame=name, offset=int(off))
        summary.append(row)
        print(f"{key:24s} M={len(xf):7d} scale={scale:7.4f} inside={inside:.2f} agree={agree:.2f} frame={name} off={off:+d}", flush=True)
        if a.dry_run:
            continue
        z.update(xyz_fields=xf, rgb_fields=prgb, fields_scale=np.float32(scale), fields_frame=np.str_(name),
                 fields_offset=np.int32(off), fields_n=np.int32(len(xf)), fields_inside_frac=np.float32(inside),
                 fields_agree_frac=np.float32(agree))
        tmp = fn + ".tmp.npz"
        np.savez_compressed(tmp, **z)
        os.replace(tmp, fn)
    ok = [s for s in summary if s["fields_n"] > 0]
    if ok:
        sc, ins, ag, offs = (np.array([s[k] for s in ok]) for k in ("scale", "inside_frac", "agree_frac", "offset"))
        print(f"\n{len(ok)}/{len(summary)} records got a fields cloud: scale median {np.median(sc):.3f} [{sc.min():.3f}, {sc.max():.3f}], "
              f"inside median {np.median(ins):.2f}, agree median {np.median(ag):.2f}, |offset| max {np.abs(offs).max()} "
              f"(exact frame for {(offs == 0).sum()})")
    if not a.dry_run:
        with open(os.path.join(a.clouds, "fields_summary.json"), "w") as f:
            json.dump(dict(fields_dir=a.fields, max_offset=a.max_offset, records=summary), f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
