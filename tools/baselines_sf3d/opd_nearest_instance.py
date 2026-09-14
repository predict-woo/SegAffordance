"""For a few test keys, export the OPDFormer / MOPD instance to DRAW: the oracle-matched one when any
instance overlaps the GT element (identical to preds.jsonl), else the NEAREST instance (mask centroid
closest to the GT mask centroid), flagged ``"fallback": "nearest"`` with ``centroid_dist_px``. Same schema
as opd_preds_to_jsonl.py (+ ``n_instances``, ``fallback``, ``centroid_dist_px``). For figures only: the
scorer never uses a non-overlapping instance.

  python tools/baselines_sf3d/opd_nearest_instance.py --pred <runs>/test/inference/instances_predictions.pth \
      --data-dir /workspace/datasets/baselines/data/opd_sf3d_512 --keys KEY [KEY ...] --out preds_nearest.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baselines_sf3d import common as C  # noqa: E402
from tools.baselines_sf3d.opd_preds_to_jsonl import (  # noqa: E402
    DEFAULT_FRAME, _decode_rle, _iou, _mtype_to_ours, _to_native, decode_gt_segmentation, load_frame_infos,
    load_predictions, unroll_point,
)


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def convert(inst, mask, rotated):
    axis = C.opd_to_cam(inst["maxis"]).tolist()
    origin = C.opd_to_cam(inst["morigin"]).tolist()
    if rotated:
        axis, origin = unroll_point(axis), unroll_point(origin)
    return {"matched": True, "score": float(inst["score"]), "mask_rle": C.rle_encode(mask),
            "type": _mtype_to_ours(inst["mtype"]), "axis_cam": axis, "origin_cam": origin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = Path(a.data_dir)
    coco = json.load(open(data / "MotionDataset_h5" / "annotations" / "MotionNet_test.json"))
    ann_to_key = json.load(open(data / "annots_test.json"))
    gt_by_id = {int(ann["id"]): ann for ann in coco["annotations"]}
    key_to_ann = {k: gt_by_id[int(i)] for i, k in ann_to_key.items()}
    frame_infos = load_frame_infos(data / "frames_test.json")
    image_frame = {int(img["id"]): frame_infos[img["file_name"]] for img in coco["images"]}
    preds = load_predictions(a.pred)
    with open(a.out, "w") as f:
        for k in a.keys:
            gt = key_to_ann[k]; image_id = int(gt["image_id"]); fi = image_frame.get(image_id, DEFAULT_FRAME)
            rotated = bool(fi.get("rotated", False)); w, h = fi.get("wh", DEFAULT_FRAME["wh"]); hw = (int(h), int(w))
            gt_full = _to_native(decode_gt_segmentation(gt["segmentation"], gt.get("height", 192), gt.get("width", 256)), rotated, hw)
            insts = preds.get(image_id, [])
            best, best_iou, best_m = None, 0.0, None
            near, near_d, near_m = None, float("inf"), None
            gc = centroid(gt_full)
            for inst in insts:
                m = _to_native(_decode_rle(inst["segmentation"]), rotated, hw)
                iou = _iou(gt_full, m)
                if iou > best_iou:
                    best, best_iou, best_m = inst, iou, m
                c = centroid(m)
                if c is not None and gc is not None:
                    d = float(np.linalg.norm(c - gc))
                    if d < near_d:
                        near, near_d, near_m = inst, d, m
            if best is not None:
                rec = {"key": k, **convert(best, best_m, rotated), "n_instances": len(insts), "fallback": None}
            elif near is not None:
                rec = {"key": k, **convert(near, near_m, rotated), "n_instances": len(insts), "fallback": "nearest", "centroid_dist_px": round(near_d, 1)}
            else:
                rec = {"key": k, "matched": False, "score": None, "mask_rle": None, "type": None, "axis_cam": None, "origin_cam": None, "n_instances": 0, "fallback": None}
            f.write(json.dumps(rec) + "\n")
            print(k[:40], "instances", len(insts), "oracle" if best is not None else ("nearest %.0f px score %.2f" % (near_d, near["score"]) if near else "none"))


if __name__ == "__main__":
    main()
