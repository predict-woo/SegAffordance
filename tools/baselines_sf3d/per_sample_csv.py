"""Per-sample metrics CSV for the external baselines, in tools/sf3d_mao_probe.py's schema.

One row per (model, test element) with everything the paper metrics need, so tables can be
recomputed and metric conventions swapped (signed vs unsigned MA, thresholds, PDet gating,
confidence thresholds) without re-running inference:

  model,idx,key,gt_type,pred_type,p_rev,axis_signed_deg,axis_unsigned_deg,origin_line_err_m,
  origin_qstar_err_m,point3d_err_m,point2d_err_frac,mask_iou,z_p_m,radius_m
  + matched,confidence          (extra columns for the oracle-matched detector protocol)

Values follow score_predictions.py exactly: IoU at 512x512 vs the fast-pipeline GT mask; an
unmatched row (or a null axis/type) is "no articulation prediction" -> pred_type -1, axis errors
90 deg; origin errors only for revolute GT rows with an origin prediction, else NaN. Baselines
have no interaction-point, z_p or radius outputs, so those columns are NaN; p_rev is NaN (they
emit hard types). `python tools/sf3d_mao_probe.py --summarize FILE.csv` runs on the output
unchanged (its M/MA use pred_type and axis_signed_deg; extra columns are ignored). PDet from the
CSV can be a few rows lower than metrics.json: see `_fmt` (exact IoU ties).

  python tools/baselines_sf3d/per_sample_csv.py \\
      --model opd_c_rgbd /workspace/datasets/baselines/results/opd_c_rgbd/preds.jsonl OUT.csv \\
      --model a3vlm_gtbox .../preds_gtbox.jsonl OUT2.csv      # one GT pass for all models
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.baselines_sf3d import common as C  # noqa: E402
from tools.baselines_sf3d.score_predictions import (  # noqa: E402
    UNMATCHED_AXIS_ERR_DEG, _angle_deg, _foot, _line_dist, _mask_iou, gt_from_repo, load_preds,
)

HEADER = ("model,idx,key,gt_type,pred_type,p_rev,axis_signed_deg,axis_unsigned_deg,origin_line_err_m,"
          "origin_qstar_err_m,point3d_err_m,point2d_err_frac,mask_iou,z_p_m,radius_m,matched,confidence")
NAN = float("nan")


def row(model, idx, key, pr, m_gt, t_gt, a_gt, o_gt, p_gt):
    """One CSV row (as a dict) for prediction record `pr` (may be None/{}) against the GT tuple."""
    pr = pr or {}
    t_gt = int(t_gt)
    m_gt = np.asarray(m_gt).astype(bool)
    if pr.get("mask_rle") is not None:
        m_pr = C.rle_decode(pr["mask_rle"])
        if m_pr.shape != m_gt.shape:
            m_pr = C.nearest_resize(m_pr.astype(np.uint8), m_gt.shape).astype(bool)
        iou = _mask_iou(m_pr, m_gt)
    else:
        iou = 0.0
    has_joint = bool(pr.get("matched")) and pr.get("axis_cam") is not None and pr.get("type") is not None
    if has_joint:
        pred_type = int(pr["type"])
        signed = _angle_deg(pr["axis_cam"], a_gt, signed=True)
        unsigned = _angle_deg(pr["axis_cam"], a_gt, signed=False)
    else:
        pred_type = -1
        signed = unsigned = UNMATCHED_AXIS_ERR_DEG
    if t_gt == 1 and has_joint and pr.get("origin_cam") is not None:
        q = np.asarray(pr["origin_cam"], dtype=np.float64)
        qstar_err = float(np.linalg.norm(q - _foot(o_gt, a_gt, p_gt)))
        line_err = _line_dist(q, o_gt, a_gt)
    else:
        qstar_err = line_err = NAN
    conf = pr.get("score")
    return {
        "model": model, "idx": idx, "key": key, "gt_type": t_gt, "pred_type": pred_type, "p_rev": NAN,
        "axis_signed_deg": signed, "axis_unsigned_deg": unsigned, "origin_line_err_m": line_err,
        "origin_qstar_err_m": qstar_err, "point3d_err_m": NAN, "point2d_err_frac": NAN, "mask_iou": iou,
        "z_p_m": NAN, "radius_m": NAN, "matched": int(bool(pr.get("matched"))),
        "confidence": NAN if conf is None else float(conf),
    }


def _fmt(v):
    # 6 decimals. Known residual vs metrics.json: the harness IoU is (inter+1e-7)/(union+1e-7), so an
    # exact tie (2*inter == union) is 0.5 + ~1e-12 and the scorer counts it as > 0.5, while the CSV
    # value prints as 0.500000 and fails a strict `> 0.5`. On the A3VLM files that is 5 of 5,088 rows
    # (PDet 34.04 from the CSV vs 34.14 in metrics.json); metrics.json stays the number of record.
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6f}"
    return str(v)


def write_csv(path, rows):
    cols = HEADER.split(",")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in cols])


def export_all(models, gt_iter):
    """models: [(name, preds_dict, out_path)]. One pass over the GT, one CSV per model."""
    rows = {name: [] for name, _, _ in models}
    for idx, (key, m_gt, t_gt, a_gt, o_gt, p_gt) in enumerate(gt_iter):
        for name, preds, _ in models:
            rows[name].append(row(name, idx, key, preds.get(key), m_gt, t_gt, a_gt, o_gt, p_gt))
    for name, _, out in models:
        write_csv(out, rows[name])
        print(f"{name}: {len(rows[name])} rows -> {out}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", nargs=3, action="append", required=True, metavar=("NAME", "PREDS_JSONL", "OUT_CSV"))
    ap.add_argument("--lmdb-root", default=C.LMDB_ROOT)
    ap.add_argument("--key-cache", default=C.KEY_CACHE)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    models = [(n, load_preds(p), o) for n, p, o in a.model]
    export_all(models, gt_from_repo(a.lmdb_root, key_cache=a.key_cache, limit=a.limit))


if __name__ == "__main__":
    main()
