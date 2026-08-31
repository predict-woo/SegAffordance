import importlib.util, pathlib, sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ann", ROOT / "tools" / "hoi4d_annotate_articulation.py")
ann = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ann)


def test_parse_key():
    seq, w, f = ann.parse_key("ZY20210800001_H1_C4_N01/S19_s05_T1_w2_f142")
    assert seq == "ZY20210800001_H1_C4_N01_S19_s05_T1"
    assert (w, f) == (2, 142)


def test_wxyz_identity_and_z_axis():
    R = ann.wxyz_to_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
    s = np.sqrt(0.5)
    R = ann.wxyz_to_matrix(np.array([s, s, 0.0, 0.0]))
    np.testing.assert_allclose(R @ [0, 0, 1], [0, -1, 0], atol=1e-12)


def test_matrix_wxyz_roundtrip():
    rng = np.random.default_rng(7)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    R = ann.wxyz_to_matrix(q)
    q2 = ann.matrix_to_wxyz(R)
    np.testing.assert_allclose(ann.wxyz_to_matrix(q2), R, atol=1e-9)


def test_backproject_roundtrip():
    K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
    depth = np.zeros((48, 64), np.uint16)
    depth[24, 32] = 2000              # 2 m at the principal point
    depth[10, 50] = 1000
    rgb = np.full((48, 64, 3), 255, np.uint8)
    pts, cols = ann.backproject(depth, rgb, K, stride=1)
    assert pts.shape[0] == 2          # zero-depth pixels dropped
    center = pts[np.argmin(np.abs(pts[:, 2] - 2.0))]
    np.testing.assert_allclose(center, [0, 0, 2.0], atol=1e-9)
    other = pts[np.argmin(np.abs(pts[:, 2] - 1.0))]
    np.testing.assert_allclose(other, [(50 - 32) / 100.0, (10 - 24) / 100.0, 1.0], atol=1e-9)
    assert cols.max() <= 1.0 and cols.shape == pts.shape


def test_world_axis_to_camera_roundtrip():
    rng = np.random.default_rng(0)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    Rwc = ann.wxyz_to_matrix(q)
    pose = np.eye(4); pose[:3, :3] = Rwc; pose[:3, 3] = [0.3, -0.2, 1.1]  # cam-to-world
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    axis_w = np.array([0.0, 0.0, 1.0])
    origin_w = pose[:3, :3] @ [0.1, 0.0, 2.0] + pose[:3, 3]   # known camera coords
    out = ann.world_axis_to_camera(axis_w, origin_w, pose, K)
    np.testing.assert_allclose(out["motion_origin_3d_camera_coords"], [0.1, 0.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(
        out["motion_dir_3d_camera_coords"], pose[:3, :3].T @ axis_w, atol=1e-9)
    u, v = out["motion_origin_2d_image_coords"]
    np.testing.assert_allclose([u, v], [320 + 500 * 0.1 / 2.0, 240.0], atol=1e-6)
    assert abs(np.linalg.norm(out["motion_dir_3d_camera_coords"]) - 1.0) < 1e-9


def test_sweep_trans_and_rot():
    pts = np.array([[1.0, 0.0, 0.0]])
    out = ann.sweep_points(pts, "trans", np.array([0, 0, 1.0]), np.zeros(3), 0.5)
    np.testing.assert_allclose(out, [[1.0, 0.0, 0.5]])
    out = ann.sweep_points(pts, "rot", np.array([0, 0, 1.0]), np.zeros(3), np.pi / 2)
    np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-12)  # right-handed
    out = ann.sweep_points(pts, "rot", np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.pi)
    np.testing.assert_allclose(out, [[1.0, 0.0, 0.0]], atol=1e-12)


def test_principal_direction_sign():
    t = np.linspace(0, 1, 20)[:, None]
    traj = t * np.array([[0.0, -2.0, 0.0]]) + np.random.default_rng(1).normal(scale=1e-3, size=(20, 3))
    d = ann.principal_direction(traj)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9
    assert d[1] < -0.99   # points from first toward last sample
