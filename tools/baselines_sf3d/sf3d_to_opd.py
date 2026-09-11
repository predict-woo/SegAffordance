"""SF3D LMDB -> OPDMulti ``MotionDataset_h5`` layout.

Contract: docs/superpowers/plans/2026-09-12-sf3d-external-baselines.md, Task 3.

Output tree (``--out``)::

    MotionDataset_h5/{train,valid,test}.h5      {split}_images uint8 (N,192,256,3), {split}_filenames str
    MotionDataset_h5/depth.h5                   depth_images float32 (N,192,256,1) mm, depth_filenames str
    MotionDataset_h5/depth_{split}.h5           per-split parts merged into depth.h5
    MotionDataset_h5/annotations/MotionNet_{split}.json
    obj_info.json                               object_key -> {object_pose, diagonal, min_bound, max_bound}
    stats.json                                  pixel_mean / pixel_std [r, g, b, depth_mm] over train, categories
    frames_{split}.json                         file_name -> {"key": frame key, "rotated": bool, "wh": [w, h]}
    annots_{split}.json                         str(annotation id) -> our LMDB key

The upstream mapper (``opdformer/mask2former/data/motion_dataset_mapper.py``)
reads ``{dir}_images`` / ``{dir}_filenames`` from ``{dir}.h5`` and
``depth_images`` / ``depth_filenames`` from ``depth.h5``, and derives the
integer "model name" from the part of ``file_name`` before ``-``; hence the
``{visit}-{frame_idx:06d}.png`` naming. Geometry is written in OPDMulti's
camera convention (y up, -z forward) via ``common.cam_to_opd`` / ``F_YZ``.

SF3D frames are mixed orientation (about half are portrait, w < h). The h5
needs one fixed image shape, so portrait frames are rolled 90 degrees
clockwise (an exact camera roll, no aspect distortion) before resizing:
``roll_record`` rotates the mask coordinates, intrinsics, cam-to-world pose
and camera-frame motion geometry consistently (``p_new = R_ROLL @ p_old``),
and ``frames_{split}.json`` records ``rotated`` so ``opd_preds_to_jsonl.py``
can undo it. The OPD y/z flip is applied after the roll.

Idempotent per split through ``.done_{split}`` markers.
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

OUT_W, OUT_H = 256, 192
CATS = ["foot_push", "hook_pull", "hook_turn", "key_press", "pinch_pull", "plug_in", "tip_push", "unplug"]
CAT_IDS = {c: i + 1 for i, c in enumerate(CATS)}
SPLIT_NAMES = {"train": "train", "bvalid": "valid", "test": "test"}  # ours -> OPDMulti
RANGE_MAX_ROT = 1.5707963267948966
RANGE_MAX_TRANS = 0.7
MIN_DIAGONAL = 0.05
# Camera roll that accompanies cv2.ROTATE_90_CLOCKWISE of the image:
# p_new = R_ROLL @ p_old, i.e. (x, y, z) -> (-y, x, z); p_old = R_ROLL.T @ p_new.
R_ROLL = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
R_ROLL4 = np.eye(4)
R_ROLL4[:3, :3] = R_ROLL
R_ROLL_INV4 = R_ROLL4.T


def needs_roll(rec):
    w, h = rec["image_dimensions_wh"]
    return w < h


def roll_record(rec):
    """Portrait record -> the same frame after a 90 degree clockwise image
    rotation: pixel (y, x) -> (x, h - 1 - y); camera point p -> R_ROLL @ p;
    fx' = fy, fy' = fx, cx' = (h - 1) - cy, cy' = cx; c2w' = c2w @ R_ROLL^-1.
    Only the fields the converter reads are rolled (2-D motion origin and
    trajectories are left untouched)."""
    w, h = rec["image_dimensions_wh"]
    out = dict(rec)
    coords = rec["mask_coordinates_yx"]
    if coords is not None and len(coords):
        c = np.asarray(coords, dtype=np.int64).reshape(-1, 2)
        out["mask_coordinates_yx"] = np.stack([c[:, 1], (h - 1) - c[:, 0]], 1).tolist()
    K = np.asarray(rec["camera_intrinsics"], dtype=np.float64)
    out["camera_intrinsics"] = [[K[1, 1], 0.0, (h - 1) - K[1, 2]], [0.0, K[0, 0], K[0, 2]], [0.0, 0.0, 1.0]]
    c2w = np.asarray(rec["camera_extrinsics_cam_to_world"], dtype=np.float64)
    out["camera_extrinsics_cam_to_world"] = (c2w @ R_ROLL_INV4).tolist()
    if rec.get("camera_extrinsics_world_to_cam") is not None:
        w2c = np.asarray(rec["camera_extrinsics_world_to_cam"], dtype=np.float64)
        out["camera_extrinsics_world_to_cam"] = (R_ROLL4 @ w2c).tolist()
    mi = rec["motion_info"]
    fs = dict(mi["frame_specific_motion_data"])
    for k in ("motion_dir_3d_camera_coords", "motion_origin_3d_camera_coords"):
        fs[k] = (R_ROLL @ np.asarray(fs[k], dtype=np.float64)).tolist()
    out["motion_info"] = {**mi, "frame_specific_motion_data": fs}
    out["image_dimensions_wh"] = (h, w)
    out["rolled"] = True
    return out


def scale_intrinsics(K, in_wh, out_wh):
    """3x3 intrinsics for an image resized from ``in_wh`` to ``out_wh``."""
    S = np.diag([out_wh[0] / in_wh[0], out_wh[1] / in_wh[1], 1.0])
    return S @ np.asarray(K, dtype=np.float64)


def colmajor16(M):
    """4x4 -> 16 floats, column-major (OPDMulti's pose/extrinsic layout)."""
    M = np.asarray(M, dtype=np.float64)
    assert M.shape == (4, 4), M.shape
    return M.flatten(order="F").tolist()


def colmajor9(K):
    K = np.asarray(K, dtype=np.float64)
    assert K.shape == (3, 3), K.shape
    return K.flatten(order="F").tolist()


def file_names(visit, frame_idx):
    stem = f"{visit}-{frame_idx:06d}"
    return f"{stem}.png", f"{stem}_d.png"


def object_key(visit, frame_idx, ann_id):
    return f"{visit}_{frame_idx:06d}_{ann_id}"


def convert_record(rec, key, ann_id, image_id, categories):
    """One LMDB record -> one COCO annotation (``object_key`` filled by the caller)."""
    w, h = rec["image_dimensions_wh"]
    m_full = C.mask_from_coords(rec["mask_coordinates_yx"], h, w)
    m = C.nearest_resize(m_full, (OUT_H, OUT_W)).astype(bool)
    ys, xs = np.where(m)
    if xs.size:
        bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
    else:
        bbox = [0.0, 0.0, 0.0, 0.0]
    n_px = int(m.sum())
    mi = rec["motion_info"]
    fs = mi["frame_specific_motion_data"]
    rot = mi["original_motion_data"]["motion_type"] == "rot"
    axis = C.cam_to_opd(fs["motion_dir_3d_camera_coords"])
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    origin = C.cam_to_opd(fs["motion_origin_3d_camera_coords"])
    label = rec["label_info"]["label"]
    return {
        "id": int(ann_id),
        "image_id": int(image_id),
        "category_id": int(categories[label]),
        "bbox": bbox,
        "area": n_px,
        "iscrowd": 0,
        "height": OUT_H,
        "width": OUT_W,
        "segmentation": C.rle_encode(m),
        "object_key": None,
        "motion": {
            "type": "rotation" if rot else "translation",
            "axis": axis.tolist(),
            "origin": origin.tolist(),
            "partId": int(ann_id),
            "part_label": label,
            "isClosed": True,
            "rangeMin": 0,
            "rangeMax": RANGE_MAX_ROT if rot else RANGE_MAX_TRANS,
            "state": 0,
            "pixel_num": float(n_px),
            "bbox": bbox,
            "object_key": None,
        },
    }


def element_diagonal(rec, depth_mm):
    """(diagonal_m, min_bound, max_bound) of the element's camera-frame points
    (native-res mask pixels back-projected with the frame depth and K);
    diagonal clamped to >= MIN_DIAGONAL. Always a 3-tuple."""
    fallback = (MIN_DIAGONAL, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    coords = rec["mask_coordinates_yx"]
    if coords is None or len(coords) == 0:
        return fallback
    K = np.asarray(rec["camera_intrinsics"], dtype=np.float64)
    c = np.asarray(coords, dtype=np.int64)
    H, W = depth_mm.shape[:2]
    inside = (c[:, 0] >= 0) & (c[:, 0] < H) & (c[:, 1] >= 0) & (c[:, 1] < W)
    c = c[inside]
    if c.shape[0] == 0:
        return fallback
    z = depth_mm[c[:, 0], c[:, 1]].astype(np.float64) / 1000.0
    ok = z > 1e-3
    if ok.sum() < 3:
        return fallback
    x = (c[ok, 1] - K[0, 2]) * z[ok] / K[0, 0]
    y = (c[ok, 0] - K[1, 2]) * z[ok] / K[1, 1]
    P = np.stack([x, y, z[ok]], 1)
    mn, mx = P.min(0), P.max(0)
    d = float(np.linalg.norm(mx - mn))
    return max(d, MIN_DIAGONAL), mn.tolist(), mx.tolist()


def build_image_entry(rec, image_id, visit, frame_idx):
    fn, dfn = file_names(visit, frame_idx)
    c2w_opd = np.asarray(rec["camera_extrinsics_cam_to_world"], dtype=np.float64) @ C.F_YZ
    K_s = scale_intrinsics(rec["camera_intrinsics"], rec["image_dimensions_wh"], (OUT_W, OUT_H))
    return {
        "id": int(image_id),
        "file_name": fn,
        "depth_file_name": dfn,
        "height": OUT_H,
        "width": OUT_W,
        "license": 1,
        "coco_url": "",
        "flickr_url": "",
        "date_captured": "",
        "camera": {"intrinsic": colmajor9(K_s), "extrinsic": colmajor16(c2w_opd)},
    }


# ---------------------------------------------------------------- workers
_ENV = None


def _env():
    global _ENV
    if _ENV is None:
        _ENV = C.open_lmdb()
    return _ENV


def process_frame(args):
    """One frame -> (file_name, rgb (192,256,3) u8, depth (192,256,1) f32 mm,
    image entry, [annotations], {object_key: obj_info}, frame_info, keys)
    with frame_info = {"key", "rotated", "wh": [w, h] of the native frame}."""
    frame_key, keys, visit, frame_idx, image_id, ann_ids = args
    with _env().begin() as txn:
        recs = [C.read_record(txn, k) for k in keys]
    rec0 = recs[0]
    root = Path(C.LMDB_ROOT)
    bgr = cv2.imread(str(root / "images" / rec0["rgb_image_path"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / "depth" / rec0["depth_image_path"]), cv2.IMREAD_UNCHANGED)
    assert bgr is not None, rec0["rgb_image_path"]
    assert depth is not None and depth.dtype == np.uint16, rec0["depth_image_path"]
    w, h = rec0["image_dimensions_wh"]
    assert bgr.shape[:2] == (h, w) and depth.shape[:2] == (h, w), (frame_key, bgr.shape, depth.shape)
    assert all(tuple(r["image_dimensions_wh"]) == (w, h) for r in recs), frame_key
    frame_info = {"key": frame_key, "rotated": False, "wh": [int(w), int(h)]}
    if needs_roll(rec0):
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        recs = [roll_record(r) for r in recs]
        rec0 = recs[0]
        frame_info["rotated"] = True
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_s = cv2.resize(rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    depth_s = cv2.resize(depth, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST).astype(np.float32)[..., None]
    image = build_image_entry(rec0, image_id, visit, frame_idx)
    c2w_opd_flat = image["camera"]["extrinsic"]
    anns, obj = [], {}
    for rec, key, ann_id in zip(recs, keys, ann_ids):
        a = convert_record(rec, key, ann_id, image_id, CAT_IDS)
        ok = object_key(visit, frame_idx, ann_id)
        a["object_key"] = ok
        a["motion"]["object_key"] = ok
        d, mn, mx = element_diagonal(rec, depth)
        obj[ok] = {"object_pose": c2w_opd_flat, "diagonal": d, "min_bound": mn, "max_bound": mx}
        anns.append(a)
    return image["file_name"], rgb_s, depth_s, image, anns, obj, frame_info, list(keys)


# ---------------------------------------------------------------- driver
def plan_jobs(keys_in_split, limit_frames=None):
    """Group keys by frame, index frames per visit (sorted), assign ids from 1."""
    frames = defaultdict(list)
    for k in keys_in_split:
        frames[C.frame_of(k)].append(k)
    per_visit = defaultdict(list)
    for fk in frames:
        per_visit[C.scene_of(fk)].append(fk)
    jobs, image_id, ann_id = [], 1, 1
    for visit in sorted(per_visit):
        for idx, fk in enumerate(sorted(per_visit[visit])):
            ks = sorted(frames[fk])
            jobs.append((fk, ks, visit, idx, image_id, list(range(ann_id, ann_id + len(ks)))))
            image_id += 1
            ann_id += len(ks)
    if limit_frames:
        jobs = jobs[:limit_frames]
    return jobs


class RunningStats:
    """Per-channel mean/std accumulator (sums in float64)."""

    def __init__(self, n_ch):
        self.s = np.zeros(n_ch)
        self.s2 = np.zeros(n_ch)
        self.n = np.zeros(n_ch)

    def add(self, x, ch):
        x = np.asarray(x, dtype=np.float64)
        self.s[ch] += x.sum()
        self.s2[ch] += (x * x).sum()
        self.n[ch] += x.size

    def mean_std(self):
        n = np.maximum(self.n, 1)
        mean = self.s / n
        var = np.maximum(self.s2 / n - mean**2, 0.0)
        return mean.tolist(), np.sqrt(var).tolist()


def convert_split(split, opd_split, keys_in_split, out, h5dir, workers, limit_frames, stats):
    import h5py

    jobs = plan_jobs(keys_in_split, limit_frames)
    n = len(jobs)
    n_ann = sum(len(j[1]) for j in jobs)
    print(f"[{opd_split}] {n} frames / {n_ann} annotations from {len(keys_in_split)} keys, {workers} workers", flush=True)
    images, annotations, fmap, amap, obj_info, labels = [], [], {}, {}, {}, set()
    n_rot = 0
    t0 = time.time()
    with h5py.File(h5dir / f"{opd_split}.h5", "w") as hf, h5py.File(h5dir / f"depth_{opd_split}.h5", "w") as hd:
        di = hf.create_dataset(f"{opd_split}_images", (n, OUT_H, OUT_W, 3), np.uint8, chunks=(1, OUT_H, OUT_W, 3))
        dn = hf.create_dataset(f"{opd_split}_filenames", (n,), h5py.string_dtype("utf-8"))
        dd = hd.create_dataset("depth_images", (n, OUT_H, OUT_W, 1), np.float32, chunks=(1, OUT_H, OUT_W, 1))
        ddn = hd.create_dataset("depth_filenames", (n,), h5py.string_dtype("utf-8"))
        with Pool(workers) as pool:
            for i, (fn, rgb, dep, image, anns, obj, finfo, ks) in enumerate(pool.imap(process_frame, jobs, chunksize=8)):
                di[i] = rgb
                dn[i] = fn
                dd[i] = dep
                ddn[i] = image["depth_file_name"]
                images.append(image)
                annotations.extend(anns)
                obj_info.update(obj)
                fmap[fn] = finfo
                n_rot += int(finfo["rotated"])
                for ann, k in zip(anns, ks):
                    amap[str(ann["id"])] = k
                    labels.add(ann["motion"]["part_label"])
                if stats is not None and i % 50 == 0:
                    px = rgb.reshape(-1, 3)
                    for ch in range(3):
                        stats.add(px[:, ch], ch)
                    stats.add(dep[dep > 0], 3)
                if i % 500 == 0 or i == n - 1:
                    el = time.time() - t0
                    rate = (i + 1) / max(el, 1e-6)
                    print(f"[{opd_split}] {i + 1}/{n} frames  {el:6.0f}s  eta {(n - i - 1) / rate:6.0f}s", flush=True)
    unknown = labels - set(CATS)
    assert not unknown, f"labels not in CATS: {sorted(unknown)}"
    if split == "train" and not limit_frames:
        assert labels == set(CATS), f"train labels {sorted(labels)} != {CATS}"
    coco = {
        "images": images,
        "annotations": annotations,
        "info": {"description": f"SF3D {opd_split} in OPDMulti format", "source_split": split},
        "licenses": [{"id": 1, "name": "sf3d"}],
        "categories": [{"id": CAT_IDS[c], "name": c, "supercategory": "sf3d"} for c in CATS],
    }
    with open(h5dir / "annotations" / f"MotionNet_{opd_split}.json", "w") as f:
        json.dump(coco, f)
    with open(out / f"frames_{opd_split}.json", "w") as f:
        json.dump(fmap, f)
    with open(out / f"annots_{opd_split}.json", "w") as f:
        json.dump(amap, f)
    with open(out / f"obj_info_{opd_split}.json", "w") as f:
        json.dump(obj_info, f)
    print(
        f"[{opd_split}] wrote {len(images)} images ({n_rot} portrait frames rolled), "
        f"{len(annotations)} annotations, labels={sorted(labels)}",
        flush=True,
    )


def merge_depth(h5dir, block=512):
    import h5py

    parts = [h5dir / f"depth_{s}.h5" for s in SPLIT_NAMES.values() if (h5dir / f"depth_{s}.h5").exists()]
    with h5py.File(h5dir / "depth.h5", "w") as hd:
        handles = [h5py.File(p, "r") for p in parts]
        n = sum(int(p["depth_images"].shape[0]) for p in handles)
        dd = hd.create_dataset("depth_images", (n, OUT_H, OUT_W, 1), np.float32, chunks=(1, OUT_H, OUT_W, 1))
        dn = hd.create_dataset("depth_filenames", (n,), h5py.string_dtype("utf-8"))
        i = 0
        for p in handles:
            m = int(p["depth_images"].shape[0])
            for s in range(0, m, block):
                e = min(s + block, m)
                dd[i + s : i + e] = p["depth_images"][s:e]
                dn[i + s : i + e] = p["depth_filenames"][s:e]
            i += m
            p.close()
    print(f"[depth] merged {n} frames from {[p.name for p in parts]} into depth.h5", flush=True)
    return n


def merge_obj_info(out):
    obj_info = {}
    for s in SPLIT_NAMES.values():
        p = out / f"obj_info_{s}.json"
        if p.exists():
            with open(p) as f:
                obj_info.update(json.load(f))
    with open(out / "obj_info.json", "w") as f:
        json.dump(obj_info, f)
    print(f"[obj_info] {len(obj_info)} object keys", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output root, e.g. /workspace/datasets/baselines/data/opd_sf3d")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit-frames", type=int, default=None, help="convert at most N frames per split (smoke)")
    ap.add_argument("--key-cache", default=C.KEY_CACHE)
    a = ap.parse_args(argv)

    out = Path(a.out)
    h5dir = out / "MotionDataset_h5"
    (h5dir / "annotations").mkdir(parents=True, exist_ok=True)
    keys = C.load_keys(a.key_cache)
    splits = C.split_scenes(keys)
    kb = C.keys_by_split(keys, splits)
    print(f"keys={len(keys)} scenes: train={len(splits['train'])} bvalid={len(splits['bvalid'])} test={len(splits['test'])}", flush=True)

    for split, opd_split in SPLIT_NAMES.items():
        marker = out / f".done_{opd_split}"
        if marker.exists():
            print(f"[{opd_split}] already done ({marker}), skipping", flush=True)
            continue
        stats = RunningStats(4) if split == "train" else None
        convert_split(split, opd_split, kb[split], out, h5dir, a.workers, a.limit_frames, stats)
        if stats is not None:
            mean, std = stats.mean_std()
            with open(out / "stats.json", "w") as f:
                json.dump({"pixel_mean": mean, "pixel_std": std, "categories": CATS}, f, indent=1)
            print(f"[stats] mean={np.round(mean, 3).tolist()} std={np.round(std, 3).tolist()}", flush=True)
        marker.touch()

    merge_depth(h5dir)
    merge_obj_info(out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
