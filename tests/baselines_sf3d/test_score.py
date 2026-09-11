"""Unit tests for tools/baselines_sf3d/score_predictions.py (synthetic GT, no volume)."""
import numpy as np

from tools.baselines_sf3d import common as C
from tools.baselines_sf3d.score_predictions import score

AXIS = np.array([0.0, 1.0, 0.0])
ORIGIN = np.array([0.0, 0.0, 2.0])
TRAJ0 = np.array([0.5, 0.3, 2.0])


def _gt(n=4):
    """4 rows: types 0,1,0,1; axis y; origin (0,0,2); traj0 (0.5,0.3,2)."""
    for i in range(n):
        m = np.zeros((512, 512), bool)
        m[100:200, 100 + i:300] = True
        yield f"s/v/{i}.0/a", m, 1 if i % 2 else 0, AXIS.copy(), ORIGIN.copy(), TRAJ0.copy()


def _full_mask(m512):
    return C.nearest_resize(m512.astype(np.uint8), (1440, 1920)).astype(bool)


def _pred(m512, t, axis, origin):
    return {
        "matched": True,
        "score": 1.0,
        "mask_rle": C.rle_encode(_full_mask(m512)),
        "type": int(t),
        "axis_cam": list(map(float, axis)),
        "origin_cam": list(map(float, origin)),
    }


def _unmatched():
    return {"matched": False, "score": None, "mask_rle": None, "type": None,
            "axis_cam": None, "origin_cam": None}


def test_perfect_predictions_score_100():
    preds = {}
    for key, m, t, a, o, p in _gt():
        # origin slid 0.3 m along the axis: equals the foot of traj0 -> err 0
        preds[key] = _pred(m, t, a, o + a * 0.3)
    r = score(preds, _gt())
    assert r["n"] == 4 and r["n_matched"] == 4 and r["n_rot"] == 2
    assert r["pass_rate_ma"] == 100.0 and r["pass_rate_m"] == 100.0 and r["p_det"] == 100.0
    assert r["pass_rate_ma_signed"] == 100.0
    assert abs(r["mean_iou"] - 1.0) < 1e-6
    assert r["err_adir_all_deg"] < 1e-4 and r["err_adir_matched_deg"] < 1e-4
    assert r["err_adir_signed_all_deg"] < 1e-4
    assert r["axis_flip_rate"] == 0.0 and r["axis_flip_rate_rot"] == 0.0
    assert r["origin_line_err_m"] < 1e-6  # sliding along the axis is free
    assert abs(r["origin_err_m"]) < 1e-6  # q* = foot of traj0 = origin + 0.3*axis
    for k in ("n", "p_det", "mean_iou", "pass_rate_m", "pass_rate_ma", "pass_rate_ma_signed",
              "err_adir_all_deg", "err_adir_matched_deg", "err_adir_signed_all_deg",
              "axis_flip_rate", "axis_flip_rate_rot", "origin_err_m", "origin_line_err_m",
              "n_matched", "n_rot"):
        assert k in r


def test_unmatched_counts_as_failure():
    preds = {key: _unmatched() for key, *_ in _gt()}
    r = score(preds, _gt())
    assert r["n"] == 4 and r["n_matched"] == 0 and r["n_rot"] == 2
    assert r["p_det"] == 0.0 and r["pass_rate_ma"] == 0.0 and r["mean_iou"] == 0.0
    assert r["pass_rate_m"] == 0.0 and r["pass_rate_ma_signed"] == 0.0
    assert r["err_adir_all_deg"] == 90.0 and r["err_adir_signed_all_deg"] == 90.0
    assert r["axis_flip_rate"] == 0.0 and r["axis_flip_rate_rot"] == 0.0  # 90 is not > 90
    # origin metrics absent (no matched rotational rows), never 0.0
    assert r["origin_err_m"] is None and r["origin_line_err_m"] is None
    assert r["err_adir_matched_deg"] is None


def test_missing_key_is_unmatched():
    preds = {}
    for key, m, t, a, o, p in _gt():
        preds[key] = _pred(m, t, a, o)
    del preds["s/v/0.0/a"]
    r = score(preds, _gt())
    assert r["n"] == 4 and r["n_matched"] == 3
    assert r["p_det"] == 75.0 and r["pass_rate_ma"] == 75.0 and r["pass_rate_m"] == 75.0
    assert abs(r["err_adir_all_deg"] - 90.0 / 4) < 1e-6
    assert r["err_adir_matched_deg"] < 1e-6  # only matched rows enter


def test_flipped_axis_is_unsigned_ok_signed_flip():
    preds = {}
    for key, m, t, a, o, p in _gt():
        preds[key] = _pred(m, t, -a, o)
    r = score(preds, _gt())
    assert r["pass_rate_ma"] == 100.0 and r["err_adir_all_deg"] < 1e-4
    assert r["pass_rate_ma_signed"] == 0.0
    assert abs(r["err_adir_signed_all_deg"] - 180.0) < 1e-4
    assert r["axis_flip_rate"] == 100.0 and r["axis_flip_rate_rot"] == 100.0


def test_wrong_type_fails_ma_but_not_axis():
    preds = {}
    for key, m, t, a, o, p in _gt():
        preds[key] = _pred(m, 1 - t, a, o)
    r = score(preds, _gt())
    assert r["pass_rate_m"] == 0.0 and r["pass_rate_ma"] == 0.0
    assert r["p_det"] == 100.0 and r["err_adir_all_deg"] < 1e-4


def test_axis_error_and_matched_iou_gate():
    # 15 deg off the axis -> type right but MA fails; matched-error uses IoU>0.5 rows only
    th = np.deg2rad(15.0)
    tilted = np.array([np.sin(th), np.cos(th), 0.0])
    preds = {}
    for j, (key, m, t, a, o, p) in enumerate(_gt()):
        if j == 0:
            bad = np.zeros_like(m)
            bad[400:500, 400:500] = True  # IoU 0 with GT -> not matched
            preds[key] = _pred(bad, t, a, o)
        else:
            preds[key] = _pred(m, t, tilted, o)
    r = score(preds, _gt())
    assert r["n_matched"] == 3 and r["p_det"] == 75.0
    assert r["pass_rate_m"] == 100.0 and r["pass_rate_ma"] == 25.0
    assert abs(r["err_adir_all_deg"] - 45.0 / 4) < 1e-6
    assert abs(r["err_adir_matched_deg"] - 15.0) < 1e-6
    assert abs(r["mean_iou"] - 0.75) < 1e-6


def test_partial_iou():
    preds = {}
    for key, m, t, a, o, p in _gt():
        part = m.copy()
        part[:, 190:] = False  # keep cols [100+i, 190) of [100+i, 300): IoU < 0.5
        preds[key] = _pred(part, t, a, o)
    r = score(preds, _gt())
    exp = np.mean([(90 - i) / (200 - i) for i in range(4)])
    assert abs(r["mean_iou"] - exp) < 1e-6
    assert r["p_det"] == 0.0 and r["n_matched"] == 0
    assert r["pass_rate_ma"] == 100.0  # MA has no IoU gate
    assert r["err_adir_matched_deg"] is None  # no IoU-matched rows


def test_origin_metrics_rotational_rows_only():
    preds = {}
    for key, m, t, a, o, p in _gt():
        # rows of type 0 get a wild origin: must not enter the origin means
        q = o + np.array([0.4, 0.0, 0.0]) if t == 1 else np.array([9.0, 9.0, 9.0])
        preds[key] = _pred(m, t, a, q)
    r = score(preds, _gt())
    assert r["n_rot"] == 2
    assert abs(r["origin_line_err_m"] - 0.4) < 1e-6
    # q* = (0, 0.3, 2); q_hat = (0.4, 0, 2) -> sqrt(0.16 + 0.09) = 0.5
    assert abs(r["origin_err_m"] - 0.5) < 1e-6


def test_origin_absent_when_null():
    preds = {}
    for key, m, t, a, o, p in _gt():
        d = _pred(m, t, a, o)
        d["origin_cam"] = None
        preds[key] = d
    r = score(preds, _gt())
    assert r["pass_rate_ma"] == 100.0
    assert r["origin_err_m"] is None and r["origin_line_err_m"] is None
