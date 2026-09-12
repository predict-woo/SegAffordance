"""OPDFormer ``instances_predictions.pth`` -> shared prediction JSONL.

Contract: docs/superpowers/plans/2026-09-12-sf3d-external-baselines.md, Task 4.

Input (detectron2 / OPDMulti ``MotionEvaluator``): a list of
``{"image_id", "instances": [{"image_id", "category_id", "bbox", "score",
"segmentation" (RLE 192x256), "mtype" (0 rotation / 1 translation, THEIR
convention), "morigin", "maxis"}]}``, geometry in OPDMulti's camera frame.

Output: one JSON line per GT annotation of ``MotionNet_test.json`` (in
``common`` test-key order) with ``key, matched, score, mask_rle (native
(h, w) of the frame), type (0 trans / 1 rot, OUR convention), axis_cam,
origin_cam`` (OpenCV frame of the native, un-rolled frame).
Matching = max mask IoU at native resolution (both sides upsampled from
192x256 with the same nearest resample); IoU 0 -> unmatched.

Portrait frames were rolled 90 degrees clockwise by ``sf3d_to_opd.py``
(``frames_test.json`` says which); here the predicted mask is rotated back
counter-clockwise before upsampling to the native (h, w), and camera-frame
geometry is mapped back with ``p_old = (y', -x', z')`` after ``opd_to_cam``.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as _rle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baselines_sf3d import common as C  # noqa: E402

UNMATCHED = {"matched": False, "score": None, "mask_rle": None, "type": None, "axis_cam": None, "origin_cam": None}
# Inverse of sf3d_to_opd.R_ROLL: p_old = R_ROLL_INV @ p_new = (y', -x', z').
R_ROLL_INV = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
DEFAULT_FRAME = {"rotated": False, "wh": [C.FRAME_W, C.FRAME_H]}


def unroll_point(p):
    return (R_ROLL_INV @ np.asarray(p, dtype=np.float64)).tolist()


def unroll_mask(mask):
    """Undo cv2.ROTATE_90_CLOCKWISE on a dataset-frame mask (192x256 -> 256x192)."""
    return np.ascontiguousarray(np.rot90(np.asarray(mask), 1))


def _decode_rle(seg):
    counts = seg["counts"]
    if isinstance(counts, str):
        counts = counts.encode()
    return _rle.decode({"size": list(seg["size"]), "counts": counts}).astype(bool)


def decode_gt_segmentation(seg, h, w):
    """GT segmentation as written by sf3d_to_opd (RLE dict); polygons and
    uncompressed RLE are accepted too, for robustness."""
    if isinstance(seg, dict):
        if isinstance(seg["counts"], list):
            seg = _rle.frPyObjects(seg, h, w)
        return _decode_rle(seg)
    rles = _rle.frPyObjects(seg, h, w)
    return _rle.decode(_rle.merge(rles)).astype(bool)


def _to_native(mask, rotated, hw):
    """Dataset-frame mask (192x256) -> native frame (h, w) of the record."""
    m = np.asarray(mask, dtype=np.uint8)
    if rotated:
        m = unroll_mask(m)
    return C.nearest_resize(m, hw).astype(bool)


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def _mtype_to_ours(mtype):
    """Their 0 = rotation, 1 = translation -> our 1 = rotation, 0 = translation."""
    if isinstance(mtype, (list, tuple, np.ndarray)):
        theirs = int(np.argmax(np.asarray(mtype, dtype=np.float64)))
    else:
        theirs = int(round(float(mtype)))
    return 1 if theirs == 0 else 0


def match_and_convert(gt_ann, instances, frame_info=None):
    """Pick the predicted instance with the highest native-res mask IoU
    against ``gt_ann``; convert it to the shared schema (without ``key``).
    ``frame_info`` = ``frames_test.json`` entry ``{"key", "rotated", "wh"}``
    (default: landscape 1920x1440, not rotated)."""
    if not instances:
        return dict(UNMATCHED)
    fi = frame_info or DEFAULT_FRAME
    rotated = bool(fi.get("rotated", False))
    w, h = fi.get("wh", DEFAULT_FRAME["wh"])
    hw = (int(h), int(w))
    gh, gw = gt_ann.get("height", 192), gt_ann.get("width", 256)
    gt_full = _to_native(decode_gt_segmentation(gt_ann["segmentation"], gh, gw), rotated, hw)
    best, best_iou, best_mask = None, 0.0, None
    for inst in instances:
        m = _to_native(_decode_rle(inst["segmentation"]), rotated, hw)
        iou = _iou(gt_full, m)
        if iou > best_iou:
            best, best_iou, best_mask = inst, iou, m
    if best is None:
        return dict(UNMATCHED)
    axis = C.opd_to_cam(best["maxis"]).tolist()
    origin = C.opd_to_cam(best["morigin"]).tolist()
    if rotated:
        axis, origin = unroll_point(axis), unroll_point(origin)
    return {
        "matched": True,
        "score": float(best["score"]),
        "mask_rle": C.rle_encode(best_mask),
        "type": _mtype_to_ours(best["mtype"]),
        "axis_cam": axis,
        "origin_cam": origin,
    }


def load_frame_infos(path):
    """frames_test.json -> file_name -> {"key", "rotated", "wh"} (accepts the
    older plain ``file_name -> key`` form as landscape / not rotated)."""
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for fn, v in raw.items():
        if isinstance(v, str):
            out[fn] = {"key": v, **DEFAULT_FRAME}
        else:
            out[fn] = {"key": v["key"], "rotated": bool(v.get("rotated", False)), "wh": v.get("wh", DEFAULT_FRAME["wh"])}
    return out


def load_predictions(path):
    import torch

    try:
        preds = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        preds = torch.load(path, map_location="cpu")
    by_image = defaultdict(list)
    for entry in preds:
        by_image[int(entry["image_id"])].extend(entry.get("instances", []))
    return by_image


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="<output-dir>/inference/instances_predictions.pth")
    ap.add_argument("--data-dir", required=True, help="sf3d_to_opd.py --out directory")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--key-cache", default=C.KEY_CACHE)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)

    data = Path(a.data_dir)
    with open(data / "MotionDataset_h5" / "annotations" / "MotionNet_test.json") as f:
        coco = json.load(f)
    with open(data / "annots_test.json") as f:
        ann_to_key = json.load(f)
    gt_by_id = {int(ann["id"]): ann for ann in coco["annotations"]}
    key_to_ann = {k: gt_by_id[int(i)] for i, k in ann_to_key.items()}
    frame_infos = load_frame_infos(data / "frames_test.json")
    image_frame = {int(img["id"]): frame_infos[img["file_name"]] for img in coco["images"]}
    preds_by_image = load_predictions(a.pred)

    keys = C.load_keys(a.key_cache)
    test_keys = C.keys_by_split(keys, C.split_scenes(keys))["test"]
    present = [k for k in test_keys if k in key_to_ann]
    missing = len(test_keys) - len(present)
    if missing:
        print(f"warning: {missing}/{len(test_keys)} test keys not in annots_test.json (partial conversion?)", flush=True)

    n_matched = 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Full-resolution IoU against up to 100 instances per frame is ~1 s per key single-threaded
    # (85 min for the test split); fork a pool that inherits the loaded tables as globals.
    global _G_KEY_TO_ANN, _G_PREDS, _G_IMAGE_FRAME
    _G_KEY_TO_ANN, _G_PREDS, _G_IMAGE_FRAME = key_to_ann, preds_by_image, image_frame
    workers = max(1, a.workers)
    with open(out, "w") as f:
        if workers == 1:
            it = map(_convert_key, present)
        else:
            from multiprocessing import Pool
            pool = Pool(workers)
            it = pool.imap(_convert_key, present, chunksize=16)
        for i, (k, res) in enumerate(it):
            n_matched += bool(res["matched"])
            f.write(json.dumps({"key": k, **res}) + "\n")
            if i % 1000 == 0:
                print(f"{i}/{len(present)}", flush=True)
        if workers > 1:
            pool.close(); pool.join()
    print(f"wrote {len(present)} lines to {out}; matched {n_matched}/{len(present)}", flush=True)


_G_KEY_TO_ANN = _G_PREDS = _G_IMAGE_FRAME = None


def _convert_key(k):
    gt = _G_KEY_TO_ANN[k]
    image_id = int(gt["image_id"])
    return k, match_and_convert(gt, _G_PREDS.get(image_id, []), _G_IMAGE_FRAME[image_id])


if __name__ == "__main__":
    main()
