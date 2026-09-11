"""Score shared-JSONL baseline predictions with SegAffordance's SF3D test metrics.

Metric definitions mirror train_SF3D_better.py::test_step / on_test_epoch_end
(2026-09-12), so the numbers drop straight next to our own rows:

  IoU            at 512x512 vs the fast-pipeline GT mask (PIL-grid nearest);
                 (inter + 1e-7) / (union + 1e-7) like _mask_iou.
  p_det          % rows with IoU > 0.5.
  pass_rate_m    % rows with the type correct.
  pass_rate_ma   % rows with the type correct AND unsigned axis error
                 <= 10 deg -- over ALL rows, no IoU gate.
  pass_rate_ma_signed   same with the signed (true-angle) error.
  err_adir_all_deg      mean unsigned axis error, all rows.
  err_adir_matched_deg  mean unsigned axis error over the IoU-matched rows
                        (IoU > 0.5) -- the harness's _test_axis_errors_matched.
  err_adir_signed_all_deg / axis_flip_rate / axis_flip_rate_rot
                 signed error mean; flip = signed error > 90 deg, over all
                 rows / rotational GT rows.
  origin_err_m   ||q_hat - q*||, q* = perpendicular foot of traj[0] on the GT
                 axis (model.losses.split.perpendicular_foot); rotational GT
                 rows with an origin prediction only.
  origin_line_err_m  distance of q_hat to the GT axis line
                 (model.losses.twist.point_to_line_distance); same rows.

Unmatched (matched=false, or key absent): IoU 0, type wrong, axis error 90
deg (unsigned and signed), origin metrics absent.

CLI: python tools/baselines_sf3d/score_predictions.py --preds P.jsonl --out
metrics.json [--lmdb-root R --frame-cache F --key-cache K --limit N]
Plan: docs/superpowers/plans/2026-09-12-sf3d-external-baselines.md (Task 2).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Runnable both as `python -m tools.baselines_sf3d.score_predictions` and as a
# plain script from anywhere: put the repo root on sys.path for `tools`,
# `datasets` and `model`.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.baselines_sf3d import common as C  # noqa: E402

THRESH_DEG = 10.0   # config.test_motion_threshold_deg
IOU_THRESH = 0.5    # config.test_iou_threshold
GT_HW = (512, 512)  # image_size_for_mask_reconstruction of the SF3D test configs
FRAME_CACHE = "/workspace/datasets/sf3d_frames_512.lmdb"
UNMATCHED_AXIS_ERR_DEG = 90.0


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _angle_deg(pred_axis, gt_axis, signed):
    """train_OPDReal_better.py::_axis_error_deg: acos(cos) with |cos| unless signed."""
    c = float(np.dot(_unit(pred_axis), _unit(gt_axis)))
    if not signed:
        c = abs(c)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _foot(origin, direction, point):
    """model.losses.split.perpendicular_foot: q* = o + ((p - o) . d_hat) d_hat."""
    d = _unit(direction)
    origin = np.asarray(origin, dtype=np.float64)
    return origin + float(np.dot(np.asarray(point, dtype=np.float64) - origin, d)) * d


def _line_dist(point, line_point, direction):
    """model.losses.twist.point_to_line_distance with a unit direction."""
    d = _unit(direction)
    rel = np.asarray(point, dtype=np.float64) - np.asarray(line_point, dtype=np.float64)
    return float(np.linalg.norm(rel - float(np.dot(rel, d)) * d))


def _mask_iou(pred, gt):
    """train_OPDReal_better.py::_mask_iou (epsilon-regularised)."""
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return (inter + 1e-7) / (union + 1e-7)


def _pct(xs):
    return 100.0 * float(np.mean(xs)) if len(xs) else 0.0


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def score(preds, gt_iter):
    """preds: {key: jsonl record}. gt_iter yields
    (key, mask512 bool[512,512], type_gt int, axis_gt[3], origin_gt[3], traj0_gt[3]).
    Returns the metrics dict (percent where our harness uses percent)."""
    ious, type_ok, ma, ma_s = [], [], [], []
    err_all, err_matched, err_s_all, err_s_rot = [], [], [], []
    oerr, olerr = [], []
    n_matched = n_rot = 0
    for key, m_gt, t_gt, a_gt, o_gt, p_gt in gt_iter:
        t_gt = int(t_gt)
        is_rot = t_gt == 1
        if is_rot:
            n_rot += 1
        pr = preds.get(key)
        if not pr or not pr.get("matched"):
            ious.append(0.0)
            type_ok.append(False)
            ma.append(False)
            ma_s.append(False)
            err_all.append(UNMATCHED_AXIS_ERR_DEG)
            err_s_all.append(UNMATCHED_AXIS_ERR_DEG)
            if is_rot:
                err_s_rot.append(UNMATCHED_AXIS_ERR_DEG)
            continue

        m_gt = np.asarray(m_gt).astype(bool)
        m_pr = C.rle_decode(pr["mask_rle"])
        if m_pr.shape != m_gt.shape:
            m_pr = C.nearest_resize(m_pr.astype(np.uint8), m_gt.shape).astype(bool)
        iou = _mask_iou(m_pr, m_gt)
        ious.append(iou)
        iou_matched = iou > IOU_THRESH
        if iou_matched:
            n_matched += 1

        e = _angle_deg(pr["axis_cam"], a_gt, signed=False)
        es = _angle_deg(pr["axis_cam"], a_gt, signed=True)
        err_all.append(e)
        err_s_all.append(es)
        if iou_matched:
            err_matched.append(e)
        if is_rot:
            err_s_rot.append(es)

        ok_t = int(pr["type"]) == t_gt
        type_ok.append(ok_t)
        ma.append(ok_t and e <= THRESH_DEG)
        ma_s.append(ok_t and es <= THRESH_DEG)

        if is_rot and pr.get("origin_cam") is not None:
            q = np.asarray(pr["origin_cam"], dtype=np.float64)
            q_star = _foot(o_gt, a_gt, p_gt)
            oerr.append(float(np.linalg.norm(q - q_star)))
            olerr.append(_line_dist(q, o_gt, a_gt))

    n = len(ious)
    return {
        "n": n,
        "p_det": _pct([i > IOU_THRESH for i in ious]),
        "mean_iou": float(np.mean(ious)) if n else 0.0,
        "pass_rate_m": _pct(type_ok),
        "pass_rate_ma": _pct(ma),
        "pass_rate_ma_signed": _pct(ma_s),
        "err_adir_all_deg": _mean(err_all),
        "err_adir_matched_deg": _mean(err_matched),
        "err_adir_signed_all_deg": _mean(err_s_all),
        "axis_flip_rate": _pct([e > 90.0 for e in err_s_all]),
        "axis_flip_rate_rot": _pct([e > 90.0 for e in err_s_rot]),
        "origin_err_m": _mean(oerr),
        "origin_line_err_m": _mean(olerr),
        "n_matched": n_matched,
        "n_rot": n_rot,
    }


def gt_from_repo(lmdb_root=C.LMDB_ROOT, frame_cache=FRAME_CACHE, key_cache=C.KEY_CACHE,
                 limit=None):
    """Yield GT exactly as our test loader sees it: SF3DDataset fast pipeline at
    512x512, test split = split_dataset_by_scene(ratio 0.1, seed 42), in the
    split's order (the same order as common.test_keys / the shared JSONL)."""
    from datasets.scenefun3d import SF3DDataset, get_default_transforms, split_dataset_by_scene

    rgb_t, mask_t, depth_t = get_default_transforms(GT_HW)
    ds = SF3DDataset(
        lmdb_data_root=lmdb_root,
        rgb_transform=rgb_t,
        mask_transform=mask_t,
        depth_transform=depth_t,
        image_size_for_mask_reconstruction=GT_HW,
        key_cache_path=key_cache,
        min_revolute_radius=0.1,
        min_mask_area_frac=0.001,
        edge_margin_frac=0.05,
        return_trajectory_2d=True,
        point_source="element",
        frame_cache_path=frame_cache,
        fast_pipeline=True,
        load_depth=False,
    )
    _, val = split_dataset_by_scene(ds, 0.1, 42)
    for j, idx in enumerate(val.indices):
        if limit is not None and j >= limit:
            break
        s = ds[idx]  # 15-tuple (return_trajectory_2d=True)
        key = ds.item_keys[idx]
        key = key.decode() if isinstance(key, bytes) else str(key)
        mask512 = s[3][0].numpy() > 0.5      # (1, H, W) float {0,1}
        type_gt = int(s[7])                  # 0 trans / 1 rot
        axis_gt = s[6].numpy().astype(np.float64)
        origin_gt = s[10].numpy().astype(np.float64)
        traj0_gt = s[12][0].numpy().astype(np.float64)  # (20, 3) -> first point
        yield key, mask512, type_gt, axis_gt, origin_gt, traj0_gt


def load_preds(path):
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                preds[d["key"]] = d
    return preds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", required=True, help="shared-schema predictions JSONL")
    ap.add_argument("--out", required=True, help="metrics.json to write")
    ap.add_argument("--lmdb-root", default=C.LMDB_ROOT)
    ap.add_argument("--frame-cache", default=FRAME_CACHE)
    ap.add_argument("--key-cache", default=C.KEY_CACHE)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N test rows")
    a = ap.parse_args(argv)
    preds = load_preds(a.preds)
    r = score(preds, gt_from_repo(a.lmdb_root, a.frame_cache, a.key_cache, a.limit))
    r["preds_file"] = a.preds
    r["n_preds_in_file"] = len(preds)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
