"""Map a baseline's raw outputs on the Fig. 5 hand-video frames (staged by handvideo_stage.py) to
the figure's schema, one record per sample:

  {"dataset", "sample", "key", "matched", "score", "mask_rle" (512x512 stretched frame), "type"
   (1 rot / 0 trans), "axis_cam", "origin_cam"}  + "line_2d" (3DOI, normalised endpoints in the
  stretched frame) + "depth_nominal" (A3VLM rows lifted with the nominal depth range)

Oracle matching against the exported GT mask (best mask IoU in the stretched frame), like the SF3D
exports; the "nearest instance" fallback for the figure is NOT applied here (the paper session does
that from the raw instances_predictions.pth).

  python handvideo_export.py opd      --stage <dir> --pred <inference/instances_predictions.pth> --out X.jsonl
  python handvideo_export.py threedoi --stage <dir> --pred <threedoi_export preds.jsonl> --out X.jsonl
  python handvideo_export.py a3vlm    --stage <dir> --pred <a3vlm_preds_to_jsonl export .jsonl> --out X.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baselines_sf3d import common as C  # noqa: E402
from tools.baselines_sf3d.handvideo_stage import STRETCH, unletterbox_mask, unletterbox_point  # noqa: E402
from tools.baselines_sf3d.opd_preds_to_jsonl import _decode_rle, _iou, _mtype_to_ours, load_predictions  # noqa: E402

EMPTY = {"matched": False, "score": None, "mask_rle": None, "type": None, "axis_cam": None, "origin_cam": None}


def gt_mask(m):
    return cv2.imread(m["mask_png"], cv2.IMREAD_GRAYSCALE) > 127


def base(m):
    return {"dataset": m["dataset"], "sample": m["sample"], "key": m["key"]}


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if xs.size else None


def export_opd(stage, pred, out, skip_no_depth=False, nearest_samples=None):
    """skip_no_depth: RGB-D models saw an all-zero depth for sources without depth (EPIC) -> those
    rows are exported unmatched with "no_depth": true instead of a prediction on junk input.
    nearest_samples: restrict to these samples and, when no instance overlaps the GT mask, export the
    detection whose mask centroid is nearest to the GT centroid (512-px stretched frame), flagged
    "fallback": "nearest" with "centroid_dist_px" -- figure use only (paper session's convention,
    cf. opd_nearest_instance.py on SF3D); never for scoring."""
    meta = json.load(open(Path(stage) / "hv_meta.json"))
    if nearest_samples:
        meta = {s: m for s, m in meta.items() if s in set(nearest_samples)}
    by_image = load_predictions(pred)  # {image_id: [instances]}
    n_ok = n_near = 0
    with open(out, "w") as f:
        for s, m in meta.items():
            rec = {**base(m), **EMPTY}
            if skip_no_depth and not m["has_depth"]:
                rec["no_depth"] = True
                f.write(json.dumps(rec) + "\n")
                continue
            gt = gt_mask(m)
            best, best_iou, best_mask = None, 0.0, None
            near, near_d, near_mask = None, np.inf, None
            gc = _centroid(gt)
            for inst in by_image.get(m["image_id"], []):
                mk = unletterbox_mask(_decode_rle(inst["segmentation"]), m["lb"])
                iou = _iou(gt, mk)
                if iou > best_iou:
                    best, best_iou, best_mask = inst, iou, mk
                if nearest_samples and gc is not None:
                    pc = _centroid(mk)
                    if pc is not None:
                        d = float(np.linalg.norm(pc - gc))
                        if d < near_d:
                            near, near_d, near_mask = inst, d, mk
            if best is None and nearest_samples and near is not None:
                best, best_mask = near, near_mask
                rec["fallback"] = "nearest"
                rec["centroid_dist_px"] = near_d
                n_near += 1
            if best is not None:
                rec.update(matched=True, score=float(best["score"]), mask_rle=C.rle_encode(best_mask),
                           type=_mtype_to_ours(best["mtype"]), axis_cam=C.opd_to_cam(best["maxis"]).tolist(),
                           origin_cam=C.opd_to_cam(best["morigin"]).tolist(), iou=_iou(gt, best_mask))
                n_ok += 1
            f.write(json.dumps(rec) + "\n")
    print(f"opd: {len(meta)} samples, {n_ok} exported ({n_near} nearest fallbacks) -> {out}")


def export_threedoi(stage, pred, out):
    meta = json.load(open(Path(stage) / "hv_meta.json"))
    rows = {json.loads(l)["key"]: json.loads(l) for l in open(pred)}
    n_mask = n_joint = 0
    with open(out, "w") as f:
        for s, m in meta.items():
            r = rows.get(s)
            rec = {**base(m), **EMPTY, "line_2d": None, "kin_id": None}
            if r is not None:
                lb = m["lb"]
                if r.get("mask_rle") is not None:
                    mk = unletterbox_mask(C.rle_decode(r["mask_rle"]), lb)
                    rec["mask_rle"] = C.rle_encode(mk)
                    rec["iou"] = _iou(gt_mask(m), mk)
                    n_mask += 1
                if r.get("line_2d") is not None and r["line_2d"][0] >= 0:
                    l = r["line_2d"]
                    rec["line_2d"] = unletterbox_point(l[:2], lb) + unletterbox_point(l[2:], lb)
                rec["kin_id"] = r.get("kin_id")
                rec["score"] = r.get("score")
                rec["matched"] = bool(r.get("matched"))
                rec["type"] = r.get("type")
                rec["axis_cam"] = r.get("axis_cam")
                rec["origin_cam"] = r.get("origin_cam")
                n_joint += int(bool(r.get("matched")))
            f.write(json.dumps(rec) + "\n")
    print(f"threedoi: {len(meta)} samples, {n_mask} with mask, {n_joint} with a lifted joint -> {out}")


def export_a3vlm(stage, pred, out):
    meta = json.load(open(Path(stage) / "hv_meta.json"))
    rows = {json.loads(l)["key"]: json.loads(l) for l in open(pred)}
    n_ok = 0
    with open(out, "w") as f:
        for s, m in meta.items():
            r = rows.get(s)
            rec = {**base(m), **EMPTY, "depth_nominal": bool(m["depth_nominal"])}
            if r is not None:
                if r.get("mask_rle") is not None:
                    mk = C.rle_decode(r["mask_rle"])  # (ch, cw) native-aspect box hull
                    mk = C.nearest_resize(mk.astype(np.uint8), (STRETCH, STRETCH)).astype(bool)
                    rec["mask_rle"] = C.rle_encode(mk)
                    rec["iou"] = _iou(gt_mask(m), mk)
                rec.update(matched=bool(r.get("matched")), score=r.get("score"), type=r.get("type"),
                           axis_cam=r.get("axis_cam"), origin_cam=r.get("origin_cam"))
                n_ok += int(bool(r.get("matched")))
            f.write(json.dumps(rec) + "\n")
    print(f"a3vlm: {len(meta)} samples, {n_ok} with a parsed joint -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["opd", "threedoi", "a3vlm"])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-no-depth", action="store_true", help="opd: blank the rows of sources without depth (RGB-D models)")
    ap.add_argument("--nearest", default=None, help="opd: comma-separated samples; nearest-centroid fallback when nothing overlaps")
    a = ap.parse_args(argv)
    if a.cmd == "opd":
        export_opd(a.stage, a.pred, a.out, skip_no_depth=a.skip_no_depth,
                   nearest_samples=a.nearest.split(",") if a.nearest else None)
    else:
        {"threedoi": export_threedoi, "a3vlm": export_a3vlm}[a.cmd](a.stage, a.pred, a.out)


if __name__ == "__main__":
    main()
