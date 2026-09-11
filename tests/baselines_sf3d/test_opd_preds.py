import json

import numpy as np

from tools.baselines_sf3d import common as C
from tools.baselines_sf3d.opd_preds_to_jsonl import decode_gt_segmentation, match_and_convert

H, W = 192, 256


def _square(y0, x0, s=10):
    m = np.zeros((H, W), bool)
    m[y0 : y0 + s, x0 : x0 + s] = True
    return m


def _gt_ann(mask):
    return {"id": 3, "image_id": 1, "segmentation": C.rle_encode(mask), "object_key": "v_000000_3"}


def _inst(mask, score, mtype, axis, origin):
    return {
        "image_id": 1,
        "category_id": 0,
        "bbox": [0, 0, 1, 1],
        "score": score,
        "segmentation": C.rle_encode(mask),
        "mtype": mtype,
        "morigin": origin,
        "maxis": axis,
    }


def test_picks_overlapping_instance_and_converts_frames():
    gt = _gt_ann(_square(50, 60))
    overlapping = _inst(_square(52, 62), 0.4, 0.0, [0.0, 1.0, 0.0], [0.1, 0.2, -3.0])
    far = _inst(_square(150, 200), 0.9, 1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    out = match_and_convert(gt, [far, overlapping])
    assert out["matched"] is True and np.isclose(out["score"], 0.4)
    assert out["type"] == 1  # their 0 = rotation -> our 1
    assert np.allclose(out["axis_cam"], [0.0, -1.0, 0.0])  # y/z flipped
    assert np.allclose(out["origin_cam"], [0.1, -0.2, 3.0])
    assert out["mask_rle"]["size"] == [C.FRAME_H, C.FRAME_W]
    pred_full = C.rle_decode(out["mask_rle"])
    gt_full = C.nearest_resize(_square(50, 60), (C.FRAME_H, C.FRAME_W))
    exp_full = C.nearest_resize(_square(52, 62), (C.FRAME_H, C.FRAME_W))
    assert (pred_full == exp_full).all() and (pred_full & gt_full).any()
    json.dumps(out)


def test_translation_type_and_bytes_counts():
    gt = _gt_ann(_square(50, 60))
    inst = _inst(_square(50, 60), 0.7, 1.0, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    inst["segmentation"]["counts"] = inst["segmentation"]["counts"].encode()  # as pycocotools emits
    out = match_and_convert(gt, [inst])
    assert out["matched"] and out["type"] == 0
    assert np.allclose(out["axis_cam"], [0.0, 0.0, -1.0])


def test_unmatched_when_no_overlap_or_no_instances():
    gt = _gt_ann(_square(50, 60))
    far = _inst(_square(150, 200), 0.9, 0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    for insts in ([far], []):
        out = match_and_convert(gt, insts)
        assert out["matched"] is False
        assert all(out[k] is None for k in ("score", "mask_rle", "type", "axis_cam", "origin_cam"))


def test_highest_iou_wins_not_highest_score():
    gt = _gt_ann(_square(50, 60))
    exact = _inst(_square(50, 60), 0.1, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
    partial = _inst(_square(55, 65), 0.99, 1.0, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
    out = match_and_convert(gt, [partial, exact])
    assert np.isclose(out["score"], 0.1) and out["type"] == 1


def test_decode_gt_segmentation_accepts_rle_and_polygon():
    m = _square(50, 60)
    assert (decode_gt_segmentation(C.rle_encode(m), H, W) == m).all()
    poly = [[60.0, 50.0, 70.0, 50.0, 70.0, 60.0, 60.0, 60.0]]
    pm = decode_gt_segmentation(poly, H, W)
    assert pm.shape == (H, W) and pm[55, 65] and not pm[10, 10]


def test_rotated_frame_round_trip():
    """Portrait frame rolled by the converter: motion dir (0,1,0) native -> (-1,0,0) in the
    dataset frame -> back to (0,1,0); the mask comes back at the native (h, w) = (1920, 1440)."""
    from tools.baselines_sf3d.sf3d_to_opd import R_ROLL

    native_axis, native_origin = np.array([0.0, 1.0, 0.0]), np.array([0.3, -0.2, 2.0])
    maxis = C.cam_to_opd(R_ROLL @ native_axis).tolist()
    morigin = C.cam_to_opd(R_ROLL @ native_origin).tolist()
    assert np.allclose(maxis, [-1.0, 0.0, 0.0])
    gt_small = _square(50, 60)
    gt = _gt_ann(gt_small)
    inst = _inst(_square(52, 62), 0.5, 0.0, maxis, morigin)
    frame = {"key": "s/v/1.0", "rotated": True, "wh": [1440, 1920]}
    out = match_and_convert(gt, [inst], frame)
    assert out["matched"] and out["type"] == 1
    assert np.allclose(out["axis_cam"], native_axis) and np.allclose(out["origin_cam"], native_origin)
    assert out["mask_rle"]["size"] == [1920, 1440]
    pred_full = C.rle_decode(out["mask_rle"])
    assert (pred_full == C.nearest_resize(np.rot90(_square(52, 62), 1).astype(np.uint8), (1920, 1440)).astype(bool)).all()
    # the same instances are unmatched if the frame is treated as un-rotated and the GT lies elsewhere
    assert match_and_convert(_gt_ann(_square(150, 200)), [inst], frame)["matched"] is False


def test_unroll_helpers():
    from tools.baselines_sf3d.opd_preds_to_jsonl import unroll_mask, unroll_point
    from tools.baselines_sf3d.sf3d_to_opd import R_ROLL

    p = np.array([0.1, 0.2, 0.3])
    assert np.allclose(unroll_point(R_ROLL @ p), p)
    m = np.zeros((192, 256), bool)
    m[0, 255] = True  # top-right of the rolled image == top-left of the native (portrait) image
    u = unroll_mask(m)
    assert u.shape == (256, 192) and u[0, 0]


def test_load_frame_infos_accepts_both_forms(tmp_path):
    from tools.baselines_sf3d.opd_preds_to_jsonl import load_frame_infos

    p = tmp_path / "frames_test.json"
    p.write_text(json.dumps({"a-000000.png": "s/v/1.0", "b-000001.png": {"key": "s/v/2.0", "rotated": True, "wh": [1440, 1920]}}))
    fi = load_frame_infos(p)
    assert fi["a-000000.png"] == {"key": "s/v/1.0", "rotated": False, "wh": [1920, 1440]}
    assert fi["b-000001.png"]["rotated"] is True and fi["b-000001.png"]["wh"] == [1440, 1920]
