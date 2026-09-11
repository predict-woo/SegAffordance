"""USDNet scene-level predictions (``preds.pkl``) -> per-frame shared JSONL.

Input ``preds.pkl`` is what USDNet's trainer dumps with ``general.debug=true``
(``self.preds`` in ``trainer/trainer.py``):

    {visit: {"pred_masks": (N, K) float/bool over the points of <usd-dir>/test/<visit>.npy,
             "pred_scores": (K,), "pred_classes": (K,) 1 = rotation / 2 = translation,
             "pred_origins": (K, 3), "pred_axises": (K, 3)}}          # z-up scene frame

For every key of our SF3D test split (``common.load_keys`` order, scenes from
``splits.json``) the predicted instances are projected into that frame with the
LMDB record's ``camera_extrinsics_world_to_cam`` (laser frame -> OpenCV camera)
and ``camera_intrinsics``:

    p_laser = inv(T_up) p_up,   p_cam = w2c p_laser,   keep z > 0.05,   u = K p_cam / z,

splatting a disc of ``radius_px`` per point.  Candidates are instances with
``score >= 0.05`` and >= 20 points at ``mask > 0.5``; the candidate with the
highest IoU against the GT mask (native 1440x1920) is the match, ``matched=false``
if no candidate overlaps.  ``type = 1 if class == 1 else 0``,
``axis_cam = R_w2c @ R_up^-1 @ axis``, ``origin_cam = w2c @ inv(T_up) @ origin``,
``score`` = the instance score, ``mask_rle`` = the projected mask.

CLI:
    python tools/baselines_sf3d/usdnet_preds_to_jsonl.py --preds preds.pkl \
        --usd-dir /workspace/datasets/baselines/data/usdnet_sf3d --out preds.jsonl

Plan: docs/superpowers/plans/2026-09-12-sf3d-external-baselines.md (Task 9).
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from tools.baselines_sf3d import common as C

SCORE_THR = 0.05
MIN_POINTS = 20
MASK_THR = 0.5
MIN_DEPTH = 0.05
RADIUS_PX = 4


def load_T_up(usd_dir, visit, mode="test"):
    with open(Path(usd_dir) / mode / f"{visit}_T_up.json") as f:
        return np.asarray(json.load(f)["T_up"], dtype=np.float64).reshape(4, 4)


def load_points(usd_dir, visit, mode="test"):
    return np.load(Path(usd_dir) / mode / f"{visit}.npy", mmap_mode="r")[:, :3].astype(np.float64)


def project_instance(points_up, T_up, w2c, K, hw=(C.FRAME_H, C.FRAME_W), radius_px=RADIUS_PX):
    """Project z-up scene points into one frame -> bool mask (H, W)."""
    h, w = int(hw[0]), int(hw[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    p = np.asarray(points_up, dtype=np.float64).reshape(-1, 3)
    if len(p) == 0:
        return mask.astype(bool)
    M = np.asarray(w2c, dtype=np.float64) @ np.linalg.inv(np.asarray(T_up, dtype=np.float64))
    p_cam = (M[:3, :3] @ p.T).T + M[:3, 3]
    z = p_cam[:, 2]
    ok = z > MIN_DEPTH
    if not ok.any():
        return mask.astype(bool)
    K = np.asarray(K, dtype=np.float64)
    u = K[0, 0] * p_cam[ok, 0] / z[ok] + K[0, 2]
    v = K[1, 1] * p_cam[ok, 1] / z[ok] + K[1, 2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    inside = (ui >= -radius_px) & (ui < w + radius_px) & (vi >= -radius_px) & (vi < h + radius_px)
    if not inside.any():
        return mask.astype(bool)
    px = np.unique(np.stack([ui[inside], vi[inside]], 1), axis=0)
    if radius_px <= 0:
        keep = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        mask[px[keep, 1], px[keep, 0]] = 1
    else:
        for x, y in px:
            cv2.circle(mask, (int(x), int(y)), int(radius_px), 1, -1)
    return mask.astype(bool)


def to_camera(axis_up, origin_up, T_up, w2c):
    """z-up axis/origin -> OpenCV camera frame of the record (unit axis)."""
    T_up = np.asarray(T_up, dtype=np.float64)
    w2c = np.asarray(w2c, dtype=np.float64)
    R = w2c[:3, :3] @ np.linalg.inv(T_up[:3, :3])
    axis = R @ np.asarray(axis_up, dtype=np.float64).reshape(3)
    n = np.linalg.norm(axis)
    if n > 0:
        axis = axis / n
    M = w2c @ np.linalg.inv(T_up)
    origin = M[:3, :3] @ np.asarray(origin_up, dtype=np.float64).reshape(3) + M[:3, 3]
    return axis, origin


def candidate_instances(pred, score_thr=SCORE_THR, min_points=MIN_POINTS, mask_thr=MASK_THR):
    """-> list of (k, point_indices) for instances passing the score / size gates."""
    masks = np.asarray(pred["pred_masks"])
    scores = np.asarray(pred["pred_scores"], dtype=np.float64).reshape(-1)
    if masks.ndim == 1:
        masks = masks[:, None]
    out = []
    for k in range(masks.shape[1]):
        if k >= len(scores) or scores[k] < score_thr:
            continue
        idx = np.flatnonzero(np.asarray(masks[:, k]).astype(np.float64) > mask_thr)
        if len(idx) >= min_points:
            out.append((k, idx))
    return out


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def unmatched_line(key):
    return {"key": key, "matched": False, "score": None, "mask_rle": None,
            "type": None, "axis_cam": None, "origin_cam": None}


def convert_predictions(keys, get_record, preds, usd_dir, mode="test", radius_px=RADIUS_PX,
                        score_thr=SCORE_THR, min_points=MIN_POINTS, verbose=True):
    """Yield one JSONL line (dict) per key, in ``keys`` order.

    get_record(key) -> LMDB record dict with camera_extrinsics_world_to_cam,
    camera_intrinsics, mask_coordinates_yx and (optionally) image_dimensions_wh.
    """
    usd_dir = Path(usd_dir)
    scenes = {}  # visit -> (points_up, T_up, candidates, pred) or None
    remaining = defaultdict(int)
    for k in keys:
        remaining[C.frame_of(k)] += 1
    frame_cache = {}  # frame -> {k: mask}
    missing = set()

    for key in keys:
        visit = C.scene_of(key)
        if visit not in scenes:
            pred = preds.get(visit) or preds.get(str(visit))
            if pred is None:
                missing.add(visit)
                scenes[visit] = None
            else:
                scenes[visit] = (load_points(usd_dir, visit, mode), load_T_up(usd_dir, visit, mode),
                                 candidate_instances(pred, score_thr, min_points), pred)
        frame = C.frame_of(key)
        remaining[frame] -= 1
        scene = scenes[visit]
        if scene is None:
            yield unmatched_line(key)
            continue
        points_up, T_up, cands, pred = scene
        rec = get_record(key)
        w2c = np.asarray(rec["camera_extrinsics_world_to_cam"], dtype=np.float64).reshape(4, 4)
        K = np.asarray(rec["camera_intrinsics"], dtype=np.float64).reshape(3, 3)
        wh = rec.get("image_dimensions_wh") or (C.FRAME_W, C.FRAME_H)
        hw = (int(wh[1]), int(wh[0]))
        gt = C.mask_from_coords(rec.get("mask_coordinates_yx"), hw[0], hw[1]).astype(bool)

        if frame not in frame_cache:
            frame_cache[frame] = {k: project_instance(points_up[idx], T_up, w2c, K, hw, radius_px)
                                  for k, idx in cands}
        masks = frame_cache[frame]
        if remaining[frame] == 0:
            del frame_cache[frame]

        best_k, best_iou = None, 0.0
        for k, m in masks.items():
            iou = _iou(m, gt)
            if iou > best_iou:
                best_k, best_iou = k, iou
        if best_k is None:
            yield unmatched_line(key)
            continue
        axis, origin = to_camera(pred["pred_axises"][best_k], pred["pred_origins"][best_k], T_up, w2c)
        cls = int(np.asarray(pred["pred_classes"]).reshape(-1)[best_k])
        yield {
            "key": key,
            "matched": True,
            "score": float(np.asarray(pred["pred_scores"]).reshape(-1)[best_k]),
            "mask_rle": C.rle_encode(masks[best_k]),
            "type": 1 if cls == 1 else 0,
            "axis_cam": [float(v) for v in axis],
            "origin_cam": [float(v) for v in origin],
        }
    if verbose and missing:
        print(f"WARNING: {len(missing)} test scenes absent from preds.pkl: {sorted(missing)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", required=True, help="USDNet debug/val_preds/preds.pkl")
    ap.add_argument("--usd-dir", required=True, help="converter output dir (<out>)")
    ap.add_argument("--out", required=True, help="preds.jsonl")
    ap.add_argument("--splits", default="experiments/baselines_sf3d/splits.json")
    ap.add_argument("--keys", default=C.KEY_CACHE)
    ap.add_argument("--lmdb", default=C.LMDB_ROOT)
    ap.add_argument("--mode", default="test")
    ap.add_argument("--radius-px", type=int, default=RADIUS_PX)
    a = ap.parse_args(argv)

    with open(a.splits) as f:
        test_scenes = set(json.load(f)["test"])
    keys = [k for k in C.load_keys(a.keys) if C.scene_of(k) in test_scenes]
    with open(a.preds, "rb") as f:
        preds = pickle.load(f)
    preds = {str(k): v for k, v in preds.items()}
    print(f"{len(keys)} test keys over {len(test_scenes)} scenes; preds for {len(preds)} scenes", flush=True)

    env = C.open_lmdb(a.lmdb)
    n_matched = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with env.begin() as txn, open(a.out, "w") as f:
        for i, line in enumerate(convert_predictions(keys, lambda k: C.read_record(txn, k), preds,
                                                     a.usd_dir, a.mode, a.radius_px)):
            n_matched += bool(line["matched"])
            f.write(json.dumps(line) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(keys)} keys, {n_matched} matched", flush=True)
    print(f"wrote {a.out}: {len(keys)} lines, {n_matched} matched ({n_matched / max(len(keys), 1):.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
