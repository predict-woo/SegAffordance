"""Synthetic tests for tools/baselines_sf3d/usdnet_preds_to_jsonl.py (no LMDB, no volume)."""
import json

import numpy as np

from tools.baselines_sf3d import common as C
from tools.baselines_sf3d import sf3d_to_usdnet as U
from tools.baselines_sf3d import usdnet_preds_to_jsonl as P

K100 = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
I4 = np.eye(4)


def t_up():
    th = np.deg2rad(40.0)
    L = np.eye(4)
    L[:3, :3] = [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
    L[:3, 3] = [0.3, -1.2, 2.5]
    return U.R_YUP_TO_ZUP @ L


def apply(T, p):
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    return (T[:3, :3] @ p.T).T + T[:3, 3]


def test_project_instance_centred_and_depth_gate():
    m = P.project_instance(np.array([[0.0, 0.0, 1.0]]), I4, I4, K100, hw=(100, 100), radius_px=4)
    assert m.shape == (100, 100) and m.dtype == bool and m[50, 50]
    ys, xs = np.nonzero(m)
    assert abs(ys.mean() - 50) < 0.6 and abs(xs.mean() - 50) < 0.6
    assert 40 <= m.sum() <= 70  # disc of radius 4
    # x right / y down: a point at +x, +y lands right of and below the centre
    m2 = P.project_instance(np.array([[0.2, 0.1, 1.0]]), I4, I4, K100, hw=(100, 100), radius_px=0)
    assert m2.sum() == 1 and m2[60, 70]
    # behind the camera / at the depth gate -> nothing; far outside the image -> nothing
    assert not P.project_instance(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.04]]), I4, I4, K100, (100, 100)).any()
    assert not P.project_instance(np.array([[5.0, 0.0, 1.0]]), I4, I4, K100, (100, 100)).any()
    assert not P.project_instance(np.zeros((0, 3)), I4, I4, K100, (100, 100)).any()


def test_project_instance_undoes_T_up():
    T = t_up()
    p_laser = np.array([[0.1, -0.2, 1.5], [0.3, 0.2, 2.0], [-0.4, 0.0, 3.0]])
    w2c = np.eye(4)
    w2c[:3, 3] = [0.05, 0.0, -0.5]
    direct = P.project_instance(p_laser, I4, w2c, K100, (100, 100), 2)
    via_up = P.project_instance(apply(T, p_laser), T, w2c, K100, (100, 100), 2)
    assert direct.any() and np.array_equal(direct, via_up)


def test_to_camera_roundtrip():
    T = t_up()
    a = np.array([0.0, 1.0, 0.0])
    o = np.array([0.5, -0.3, 2.0])
    axis, origin = P.to_camera(T[:3, :3] @ (3.0 * a), apply(T, o)[0], T, I4)
    assert np.allclose(axis, a) and np.allclose(origin, o)
    w2c = np.eye(4)
    th = np.deg2rad(25.0)
    w2c[:3, :3] = [[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]
    w2c[:3, 3] = [0.1, 0.2, 0.3]
    axis, origin = P.to_camera(T[:3, :3] @ a, apply(T, o)[0], T, w2c)
    assert np.allclose(axis, w2c[:3, :3] @ a) and np.allclose(origin, apply(w2c, o)[0])


def test_candidate_instances_gates():
    masks = np.zeros((100, 4))
    masks[:50, 0] = 0.9   # ok
    masks[:50, 1] = 0.9   # low score
    masks[:10, 2] = 0.9   # too few points
    masks[:, 3] = 0.3     # below mask threshold everywhere
    pred = {"pred_masks": masks, "pred_scores": np.array([0.8, 0.01, 0.9, 0.9])}
    c = P.candidate_instances(pred)
    assert [k for k, _ in c] == [0] and len(c[0][1]) == 50


def _blob(center, n, rng, r=0.08):
    return np.asarray(center, float) + rng.uniform(-r, r, size=(n, 3))


def test_convert_predictions_end_to_end(tmp_path):
    rng = np.random.RandomState(0)
    T = t_up()
    visit = "420693"
    A = _blob([0.0, 0.0, 2.0], 200, rng)     # projects around (50, 50)
    B = _blob([0.6, 0.0, 2.0], 200, rng)     # projects around (50, 80)
    bg = rng.uniform(-3, 3, size=(600, 3)) + np.array([0, 0, 8.0])
    p_laser = np.vstack([A, B, bg])
    n = len(p_laser)
    pts = np.zeros((n, 13), np.float32)
    pts[:, :3] = apply(T, p_laser)
    (tmp_path / "test").mkdir()
    np.save(tmp_path / "test" / f"{visit}.npy", pts)
    (tmp_path / "test" / f"{visit}_T_up.json").write_text(json.dumps({"T_up": T.tolist()}))

    masks = np.zeros((n, 4), np.float32)
    masks[200:400, 0] = 1.0                 # B, k=0
    masks[:200, 1] = 1.0                    # A duplicate with a low score, k=1
    masks[:200, 2] = 1.0                    # A, k=2
    masks[400:410, 3] = 1.0                 # tiny, k=3
    axes_up = np.stack([T[:3, :3] @ v for v in ([1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1])])
    origins_up = apply(T, [[0.6, 0, 2], [0, 0, 2], [0, 0, 2], [0, 0, 8]])
    preds = {visit: {"pred_masks": masks, "pred_scores": np.array([0.9, 0.01, 0.7, 0.95]),
                     "pred_classes": np.array([2, 1, 1, 2]),
                     "pred_origins": origins_up, "pred_axises": axes_up}}

    def disc(cy, cx, r):
        yy, xx = np.mgrid[:100, :100]
        return np.argwhere((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).tolist()

    frame = f"{visit}/42445255/1234.567"
    records = {
        f"{frame}/annot-a": {"camera_extrinsics_world_to_cam": I4.tolist(), "camera_intrinsics": K100.tolist(),
                             "mask_coordinates_yx": disc(50, 50, 8), "image_dimensions_wh": (100, 100)},
        f"{frame}/annot-b": {"camera_extrinsics_world_to_cam": I4.tolist(), "camera_intrinsics": K100.tolist(),
                             "mask_coordinates_yx": disc(50, 80, 8), "image_dimensions_wh": (100, 100)},
        f"{frame}/annot-none": {"camera_extrinsics_world_to_cam": I4.tolist(), "camera_intrinsics": K100.tolist(),
                                "mask_coordinates_yx": disc(10, 10, 5), "image_dimensions_wh": (100, 100)},
        "999999/1/0.0/annot-x": {"camera_extrinsics_world_to_cam": I4.tolist(), "camera_intrinsics": K100.tolist(),
                                 "mask_coordinates_yx": disc(50, 50, 8), "image_dimensions_wh": (100, 100)},
    }
    keys = list(records)
    lines = list(P.convert_predictions(keys, records.__getitem__, preds, tmp_path, verbose=False))
    assert [l["key"] for l in lines] == keys
    assert set(lines[0]) == {"key", "matched", "score", "mask_rle", "type", "axis_cam", "origin_cam"}

    a = lines[0]
    assert a["matched"] and a["score"] == 0.7 and a["type"] == 1  # k=2 (A), not the low-score duplicate
    assert np.allclose(a["axis_cam"], [0, 1, 0], atol=1e-6) and np.allclose(a["origin_cam"], [0, 0, 2], atol=1e-6)
    m = C.rle_decode(a["mask_rle"])
    assert m.shape == (100, 100) and m[50, 50] and not m[50, 80]
    gt = C.mask_from_coords(records[keys[0]]["mask_coordinates_yx"], 100, 100).astype(bool)
    assert np.logical_and(m, gt).sum() / np.logical_or(m, gt).sum() > 0.5

    b = lines[1]
    assert b["matched"] and b["score"] == 0.9 and b["type"] == 0
    assert np.allclose(b["axis_cam"], [1, 0, 0], atol=1e-6) and np.allclose(b["origin_cam"], [0.6, 0, 2], atol=1e-6)

    for l in lines[2:]:
        assert l == P.unmatched_line(l["key"])
    assert json.loads(json.dumps(lines[0]))["matched"] is True  # JSON-serialisable
