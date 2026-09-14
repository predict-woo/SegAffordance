"""tools/baselines_sf3d/per_sample_csv.py: rows agree with score_predictions.py and the CSV is
readable by tools/sf3d_mao_probe.py --summarize unchanged (synthetic GT, no volume)."""
import csv
import math

import numpy as np

from tests.baselines_sf3d.test_score import _gt, _pred, _unmatched
from tools.baselines_sf3d.per_sample_csv import HEADER, export_all, row
from tools.baselines_sf3d.score_predictions import score


def _preds(perfect=True):
    preds = {}
    for key, m, t, a, o, p in _gt():
        preds[key] = _pred(m, t, a, o + a * 0.3) if perfect else _unmatched()
    return preds


def test_header_extends_probe_schema():
    from tools.sf3d_mao_probe import HEADER as PROBE_HEADER
    assert HEADER.startswith(PROBE_HEADER), "the probe's columns must come first, unchanged"
    assert HEADER.split(",")[len(PROBE_HEADER.split(",")):] == ["matched", "confidence"]


def test_rows_match_scorer_on_perfect_and_unmatched():
    for perfect in (True, False):
        preds = _preds(perfect)
        rows = [row("m", i, key, preds.get(key), m, t, a, o, p) for i, (key, m, t, a, o, p) in enumerate(_gt())]
        r = score(preds, _gt())
        g = np.array([x["gt_type"] for x in rows]); pt = np.array([x["pred_type"] for x in rows])
        s_ = np.array([x["axis_signed_deg"] for x in rows]); iou = np.array([x["mask_iou"] for x in rows])
        assert 100.0 * np.mean(g == pt) == r["pass_rate_m"]
        assert 100.0 * np.mean((g == pt) & (s_ <= 10.0)) == r["pass_rate_ma_signed"]
        assert 100.0 * np.mean(iou > 0.5) == r["p_det"]
        assert abs(float(np.mean(iou)) - r["mean_iou"]) < 1e-9
        if perfect:
            assert all(x["pred_type"] in (0, 1) for x in rows) and all(x["matched"] == 1 for x in rows)
            assert all(x["confidence"] == 1.0 for x in rows)
            rot = [x for x in rows if x["gt_type"] == 1]
            assert all(x["origin_line_err_m"] < 1e-6 and x["origin_qstar_err_m"] < 1e-6 for x in rot)
            assert all(math.isnan(x["origin_line_err_m"]) for x in rows if x["gt_type"] == 0)
        else:
            assert all(x["pred_type"] == -1 and x["axis_signed_deg"] == 90.0 for x in rows)
            assert all(math.isnan(x["origin_line_err_m"]) and math.isnan(x["confidence"]) for x in rows)
        assert all(math.isnan(x["p_rev"]) and math.isnan(x["z_p_m"]) and math.isnan(x["radius_m"]) for x in rows)


def test_export_writes_csv_the_probe_can_summarize(tmp_path, capsys):
    out = tmp_path / "a.csv"
    export_all([("mA", _preds(True), str(out)), ("mB", _preds(False), str(tmp_path / "b.csv"))], _gt())
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and rows[0]["model"] == "mA" and rows[0]["idx"] == "0"
    assert rows[0]["p_rev"] == "nan" and float(rows[0]["mask_iou"]) == 1.0
    from tools.sf3d_mao_probe import summarize
    summarize(str(out), 10.0, 0.5)
    txt = capsys.readouterr().out
    assert "mA" in txt and "100.00" in txt
