import numpy as np

from tools.baselines_sf3d import common as C
from tools.baselines_sf3d.sf3d_to_opd import R_ROLL
from tools.baselines_sf3d.threedoi_export import (
    OUT_H,
    OUT_W,
    backproject_mask,
    fit_plane_ransac,
    lift_line,
    prediction,
    unroll_mask,
    unroll_vec,
)

K = np.array([[1600.0 * OUT_W / 1920, 0, 960.0 * OUT_W / 1920], [0, 1600.0 * OUT_H / 1440, 720.0 * OUT_H / 1440], [0, 0, 1]])


def _plane_depth(n, d, holes=True):
    """Depth image of the plane n.p = d (n oriented towards the camera, so n[2] < 0)."""
    ys, xs = np.mgrid[0:OUT_H, 0:OUT_W]
    rays = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs, dtype=float)], -1)
    z = d / (rays @ n)
    if holes:
        z[::7, ::5] = -1.0
    return z


def test_plane_fit_and_line_lift_recover_a_planted_axis():
    n = np.array([0.2, 0.0, -1.0])
    n /= np.linalg.norm(n)
    d = float(np.dot(n, [0, 0, 2.5]))  # plane through (0,0,2.5)
    depth = _plane_depth(n, d)
    mask = np.zeros((OUT_H, OUT_W), bool)
    mask[300:500, 400:700] = True
    pts = backproject_mask(mask, depth, K)
    assert len(pts) > 1000
    plane = fit_plane_ransac(pts)
    assert plane is not None and abs(abs(np.dot(plane[0], n)) - 1) < 1e-3 and abs(plane[1] - d) < 1e-3 and plane[0][2] < 0
    # a vertical 3D axis on the plane through the mask projects to a vertical 2D line; lifting gives it back
    p0 = np.array([0.1, -0.3, 0.0])
    p0[2] = (d - n[0] * p0[0] - n[1] * p0[1]) / n[2]
    p1 = np.array([0.1, 0.3, 0.0])
    p1[2] = (d - n[0] * p1[0] - n[1] * p1[1]) / n[2]
    uv = [((K[0, 0] * p[0] / p[2] + K[0, 2]) / OUT_W, (K[1, 1] * p[1] / p[2] + K[1, 2]) / OUT_H) for p in (p0, p1)]
    seg = lift_line([uv[0][0], uv[0][1], uv[1][0], uv[1][1]], plane, K)
    assert np.allclose(seg[0], p0, atol=2e-3) and np.allclose(seg[1], p1, atol=2e-3)
    rec = prediction("k", 1, [uv[0][0], uv[0][1], uv[1][0], uv[1][1]], mask, depth, K, {"wh": [1920, 1440], "rotated": False})
    assert rec["matched"] and rec["type"] == 1
    ax = np.array(rec["axis_cam"])
    assert np.degrees(np.arccos(abs(np.dot(ax, (p1 - p0) / np.linalg.norm(p1 - p0))))) < 0.5
    assert abs(np.dot(np.array(rec["origin_cam"]) - p0, np.cross(ax, n))) < 1e-2  # origin on the lifted line
    assert C.rle_decode(rec["mask_rle"]).shape == (1440, 1920)
    # translation: axis = plane normal, origin = element centroid
    rec_t = prediction("k", 2, None, mask, depth, K, {"wh": [1920, 1440], "rotated": False})
    assert rec_t["matched"] and rec_t["type"] == 0 and abs(abs(np.dot(rec_t["axis_cam"], n)) - 1) < 1e-3
    # freeform -> unmatched but the mask is kept
    rec_f = prediction("k", 0, None, mask, depth, K, {"wh": [1920, 1440], "rotated": False})
    assert not rec_f["matched"] and rec_f["mask_rle"] is not None
    # too few depth points -> unmatched
    rec_e = prediction("k", 1, [0.1, 0.1, 0.9, 0.9], mask, np.full((OUT_H, OUT_W), -1.0), K, {"wh": [1920, 1440], "rotated": False})
    assert not rec_e["matched"]


def test_unroll():
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(unroll_vec(R_ROLL @ v, True), v) and np.allclose(unroll_vec(v, False), v)
    m = np.zeros((OUT_H, OUT_W), bool)
    m[0:10, OUT_W - 20:OUT_W] = True  # top-right corner in the rolled (landscape) frame
    back = unroll_mask(m, [1920, 1440], True)
    assert back.shape == (1920, 1440)
    ys, xs = np.nonzero(back)
    assert ys.min() == 0 and xs.min() == 0 and xs.max() < 60 and ys.max() < 60  # -> top-left corner of the native portrait frame
    same = unroll_mask(m, [1920, 1440], False)
    assert same.shape == (1440, 1920) and same[0, -1]
