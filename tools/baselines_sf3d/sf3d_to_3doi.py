"""SF3D LMDB -> 3DOI (monoarti) dataset layout.

Layout (``--out`` becomes the pod's ``/home/ubuntu/monoarti_data`` via a symlink, so
monoarti/dataset.py needs no path edits)::

  images/taskonomy_<visit>_point_<F>_view_0_domain_rgb.jpg      1024x768 RGB
  3doi_sf3d/data_{train,val,test}.pt                             torch-saved entry lists
  omnidata_filtered/depth_zbuffer/taskonomy/<visit>/point_<F>_view_0_domain_depth_zbuffer.png
                                                                 uint16 = metres * 512
  omnidata_filtered/mask_valid/taskonomy/<visit>/point_<F>_view_0_domain_depth_zbuffer.png
                                                                 255 where depth > 0
  omnidata_filtered/point_info/taskonomy/<visit>/point_<F>_view_0_domain_point_info.json
                                                                 {"field_of_view_rads": horizontal fov}
  frames_{split}.json                                            img_name -> {key, rotated, wh, K_out}

The ``taskonomy_`` prefix makes their loader read our input depth as the depth
supervision target (dataset.py:388-418; ``field_of_view_rads`` is the horizontal
FOV, test.py:463 ``focal = W/2 / tan(fov/2)``).  Portrait frames are rolled 90 deg
clockwise (sf3d_to_opd.roll_record) so every image is 1024x768 = their fixed
``image_size``.  One entry per frame, one instance per element:

  keypoint / affordance  element point (mask centroid snapped into the mask), normalised x,y
  movable one_hand, rigid yes, pull_or_push n/a (ignored in their action loss)
  kinematic rotation | translation
  bbox                   mask bbox, normalised xyxy
  mask                   ONE polygon (largest contour of the 3 px-dilated splat mask), normalised
  axis                   GT 3D axis line projected and cut by the [0.01, 0.99]^2 rectangle,
                         [x1,y1,x2,y2] with x1 < x2 (their validity test is axis[:, 0] > 0);
                         [-1,-1,-1,-1] when the projection degenerates.

Usage:
  python tools/baselines_sf3d/run.py tools/baselines_sf3d/sf3d_to_3doi.py \
      --out /workspace/datasets/baselines/stage/3doi --workers 16 [--limit-frames N]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baselines_sf3d import common as C  # noqa: E402
from tools.baselines_sf3d.sf3d_to_opd import needs_roll, roll_record, scale_intrinsics  # noqa: E402

OUT_W, OUT_H = 1024, 768
DEPTH_SCALE = 512.0
DILATE_PX = 3
AXIS_MARGIN = 0.01
SPLIT_NAMES = {"train": "train", "bvalid": "val", "test": "test"}
KINEMATIC = {"rot": "rotation", "trans": "translation"}


def img_name(visit, f):
    return f"taskonomy_{visit}_point_{f}_view_0_domain_rgb.jpg"


def depth_name(f):
    return f"point_{f}_view_0_domain_depth_zbuffer.png"


def info_name(f):
    return f"point_{f}_view_0_domain_point_info.json"


def hfov_rads(K_out, w=OUT_W):
    return float(2.0 * np.arctan(w / (2.0 * K_out[0][0])))


def resize_mask(mask_full, out_hw=(OUT_H, OUT_W)):
    return C.nearest_resize(mask_full, out_hw).astype(np.uint8)


def element_point(mask):
    """Mask centroid (x, y) snapped to the nearest mask pixel; normalised to [0,1]."""
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    if not mask[int(round(cy)), int(round(cx))]:
        i = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
        cx, cy = xs[i], ys[i]
    h, w = mask.shape
    return [float(cx) / w, float(cy) / h]


def mask_polygon(mask, dilate_px=DILATE_PX):
    """Largest external contour of the dilated mask -> [[x, y], ...] normalised; [] if none."""
    h, w = mask.shape
    m = mask.astype(np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    if len(c) < 3:
        return []
    return [[float(x) / w, float(y) / h] for x, y in c]


def bbox_norm(mask):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return [float(xs.min()) / w, float(ys.min()) / h, float(xs.max() + 1) / w, float(ys.max() + 1) / h]


def clip_line_to_rect(p, d, lo=AXIS_MARGIN, hi=1.0 - AXIS_MARGIN):
    """Segment of the 2D line p + t d inside [lo, hi]^2, as [x1, y1, x2, y2] with x1 <= x2, or None."""
    p, d = np.asarray(p, dtype=np.float64), np.asarray(d, dtype=np.float64)
    tmin, tmax = -np.inf, np.inf
    for i in range(2):
        if abs(d[i]) < 1e-12:
            if p[i] < lo or p[i] > hi:
                return None
            continue
        t0, t1 = (lo - p[i]) / d[i], (hi - p[i]) / d[i]
        tmin, tmax = max(tmin, min(t0, t1)), min(tmax, max(t0, t1))
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax - tmin < 1e-9:
        return None
    a, b = p + tmin * d, p + tmax * d
    if a[0] > b[0]:
        a, b = b, a
    return [float(a[0]), float(a[1]), float(b[0]), float(b[1])]


def axis_line_2d(origin_cam, dir_cam, K_out, w=OUT_W, h=OUT_H):
    """GT 3D axis -> normalised 2D line segment in the output image, or [-1]*4."""
    o = np.asarray(origin_cam, dtype=np.float64)
    n = np.asarray(dir_cam, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    K = np.asarray(K_out, dtype=np.float64)
    pts = np.stack([o - 0.5 * n, o + 0.5 * n])
    if (pts[:, 2] <= 0.05).any():
        pts = np.stack([o, o + 0.2 * n]) if o[2] > 0.05 else None
        if pts is None or (pts[:, 2] <= 0.05).any():
            return [-1.0, -1.0, -1.0, -1.0]
    uv = np.stack([(K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2]) / w, (K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]) / h], 1)
    d = uv[1] - uv[0]
    if np.linalg.norm(d) * max(w, h) < 1.0:  # axis along the viewing ray: no 2D line
        return [-1.0, -1.0, -1.0, -1.0]
    seg = clip_line_to_rect(uv[0], d / np.linalg.norm(d))
    return seg if seg is not None else [-1.0, -1.0, -1.0, -1.0]


def instance_from_record(rec, K_out):
    """One (rolled) record -> one 3DOI instance dict; None when the mask is empty."""
    w, h = rec["image_dimensions_wh"]
    m = resize_mask(C.mask_from_coords(rec["mask_coordinates_yx"], h, w))
    if m.sum() == 0:
        return None
    fs = rec["motion_info"]["frame_specific_motion_data"]
    kp = element_point(m)
    return {
        "keypoint": kp,
        "movable": "one_hand",
        "rigid": "yes",
        "kinematic": KINEMATIC[rec["motion_info"]["original_motion_data"]["motion_type"]],
        "pull_or_push": "n/a",
        "affordance": kp,
        "bbox": bbox_norm(m),
        "mask": mask_polygon(m),
        "axis": axis_line_2d(fs["motion_origin_3d_camera_coords"], fs["motion_dir_3d_camera_coords"], K_out),
    }


# ---------------------------------------------------------------- per frame
_ENV = None


def _env():
    global _ENV
    if _ENV is None:
        _ENV = C.open_lmdb()
    return _ENV


def process_frame(args):
    frame_key, keys, visit, f, out = args
    out = Path(out)
    with _env().begin() as txn:
        recs = [C.read_record(txn, k) for k in keys]
    rec0 = recs[0]
    root = Path(C.LMDB_ROOT)
    bgr = cv2.imread(str(root / "images" / rec0["rgb_image_path"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / "depth" / rec0["depth_image_path"]), cv2.IMREAD_UNCHANGED)
    assert bgr is not None and depth is not None and depth.dtype == np.uint16, frame_key
    w, h = rec0["image_dimensions_wh"]
    assert bgr.shape[:2] == (h, w) and depth.shape[:2] == (h, w), (frame_key, bgr.shape, depth.shape)
    rotated = False
    if needs_roll(rec0):
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        recs = [roll_record(r) for r in recs]
        rec0 = recs[0]
        rotated = True
    w, h = rec0["image_dimensions_wh"]
    K_out = scale_intrinsics(rec0["camera_intrinsics"], (w, h), (OUT_W, OUT_H)).tolist()
    name = img_name(visit, f)
    dst = out / "images" / name
    if not dst.exists():
        cv2.imwrite(str(dst), cv2.resize(bgr, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
    d_dir = out / "omnidata_filtered" / "depth_zbuffer" / "taskonomy" / visit
    m_dir = out / "omnidata_filtered" / "mask_valid" / "taskonomy" / visit
    i_dir = out / "omnidata_filtered" / "point_info" / "taskonomy" / visit
    for d in (d_dir, m_dir, i_dir):
        d.mkdir(parents=True, exist_ok=True)
    if not (i_dir / info_name(f)).exists():
        depth_s = cv2.resize(depth, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
        depth_512 = np.clip(depth_s.astype(np.float64) / 1000.0 * DEPTH_SCALE, 0, 65535).round().astype(np.uint16)
        cv2.imwrite(str(d_dir / depth_name(f)), depth_512)
        cv2.imwrite(str(m_dir / depth_name(f)), ((depth_s > 0) * 255).astype(np.uint8))
        json.dump({"field_of_view_rads": hfov_rads(K_out)}, open(i_dir / info_name(f), "w"))
    instances, inst_keys = [], []
    for rec, key in zip(recs, keys):
        inst = instance_from_record(rec, K_out)
        if inst is None:
            continue
        instances.append(inst)
        inst_keys.append(key)
    entry = {"img_name": name, "instances": instances}
    info = {"key": frame_key, "rotated": rotated, "wh": [int(w), int(h)], "K_out": K_out, "keys": inst_keys}
    return name, entry, info


def plan_jobs(keys, limit_frames=None):
    by_frame = defaultdict(list)
    for k in keys:
        by_frame[C.frame_of(k)].append(k)
    frames = sorted(by_frame)
    if limit_frames:
        frames = frames[:limit_frames]
    counter = defaultdict(int)
    jobs = []
    for fk in frames:
        visit = C.scene_of(fk)
        jobs.append((fk, by_frame[fk], visit, counter[visit]))
        counter[visit] += 1
    return jobs


def convert_split(split, keys, out, workers, limit_frames):
    import torch

    jobs = [(fk, ks, v, f, str(out)) for fk, ks, v, f in plan_jobs(keys, limit_frames)]
    t0 = time.time()
    entries, infos = [], {}
    n_inst = n_axis = 0
    with Pool(workers) as pool:
        for i, (name, entry, info) in enumerate(pool.imap(process_frame, jobs, chunksize=4)):
            entries.append(entry)
            infos[name] = info
            n_inst += len(entry["instances"])
            n_axis += sum(inst["axis"][0] > 0 for inst in entry["instances"])
            if (i + 1) % 2000 == 0:
                print(f"  {split}: {i + 1}/{len(jobs)} frames, {time.time() - t0:.0f}s", flush=True)
    (out / "3doi_sf3d").mkdir(exist_ok=True)
    torch.save(entries, out / "3doi_sf3d" / f"data_{SPLIT_NAMES[split]}.pt")
    json.dump(infos, open(out / f"frames_{SPLIT_NAMES[split]}.json", "w"))
    print(f"{split}->{SPLIT_NAMES[split]}: {len(entries)} frames, {n_inst} instances, {n_axis} with a 2D axis, "
          f"{time.time() - t0:.0f}s", flush=True)
    return {"frames": len(entries), "instances": n_inst, "with_axis": n_axis}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=str(Path(__file__).resolve().parents[2] / "experiments/baselines_sf3d/splits.json"))
    ap.add_argument("--key-cache", default=C.KEY_CACHE)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--only", default="test,bvalid,train")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    keys = C.load_keys(a.key_cache)
    per = C.keys_by_split(keys, json.load(open(a.splits)))
    stats = {}
    for split in a.only.split(","):
        stats[split] = convert_split(split, per[split], out, a.workers, a.limit_frames)
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
