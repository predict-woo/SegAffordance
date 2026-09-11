"""Synthetic-scene tests for tools/baselines_sf3d/sf3d_to_usdnet.py (no volume, no open3d)."""
import json
import pickle
import struct

import h5py
import numpy as np
import pytest
import yaml

from tools.baselines_sf3d import sf3d_to_usdnet as U

VISIT, VIDEO = "999001", "55550001"


def write_ply(path, xyz, rgb, fmt="binary_little_endian"):
    n = len(xyz)
    header = (
        f"ply\nformat {fmt} 1.0\ncomment synthetic\nelement vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        if fmt == "ascii":
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n".encode())
        else:
            for p, c in zip(xyz, rgb):
                f.write(struct.pack("<fffBBB", *map(float, p), *map(int, c)))


def laser_to_arkit_matrix():
    th = np.deg2rad(30.0)
    L = np.eye(4)
    L[:3, :3] = [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    L[:3, 3] = [1.0, -2.0, 0.5]
    return L


def make_scene(root, n=2000, seed=0, with_motions=True):
    """Cube scan in [0,1]^3 with a 'rot' element (x < 0.2) and a 'trans' element
    (x > 0.8), plus an 'exclude' annotation and one annotation without a motion."""
    rng = np.random.RandomState(seed)
    xyz = rng.uniform(0.0, 1.0, size=(n, 3))
    rgb = rng.randint(0, 256, size=(n, 3))
    folder = root / VISIT
    (folder / VIDEO).mkdir(parents=True)
    write_ply(folder / f"{VISIT}_laser_scan.ply", xyz, rgb)
    np.save(folder / VIDEO / f"{VIDEO}_transform.npy", laser_to_arkit_matrix())
    rot_idx = np.flatnonzero(xyz[:, 0] < 0.2).tolist()
    trans_idx = np.flatnonzero(xyz[:, 0] > 0.8).tolist()
    mid_idx = np.flatnonzero(np.abs(xyz[:, 1] - 0.5) < 0.05).tolist()
    annotations = {
        "annotations": [
            {"annot_id": "rot-uuid", "label": "door", "indices": rot_idx},
            {"annot_id": "trans-uuid", "label": "drawer", "indices": trans_idx},
            {"annot_id": "excl-uuid", "label": "exclude", "indices": mid_idx[:10]},
            {"annot_id": "nomotion-uuid", "label": "button", "indices": mid_idx[10:20]},
        ]
    }
    motions = {
        "motions": [
            {"annot_id": "rot-uuid", "motion_type": "rot", "motion_dir": [0.0, 2.0, 0.0],
             "motion_origin_idx": rot_idx[0], "motion_viz_orient": "inwards"},
            {"annot_id": "trans-uuid", "motion_type": "trans", "motion_dir": [1.0, 0.0, 0.0],
             "motion_origin_idx": trans_idx[0], "motion_viz_orient": "outwards"},
            {"annot_id": "excl-uuid", "motion_type": "rot", "motion_dir": [0.0, 0.0, 1.0],
             "motion_origin_idx": mid_idx[0]},
        ] if with_motions else []
    }
    (folder / f"{VISIT}_annotations.json").write_text(json.dumps(annotations))
    (folder / f"{VISIT}_motions.json").write_text(json.dumps(motions))
    return xyz, rgb, rot_idx, trans_idx


@pytest.fixture
def scene(tmp_path):
    scans = tmp_path / "scans"
    xyz, rgb, rot_idx, trans_idx = make_scene(scans)
    return {"scans": scans, "out": tmp_path / "usd", "xyz": xyz, "rgb": rgb,
            "rot_idx": rot_idx, "trans_idx": trans_idx}


def test_numpy_ply_reader_binary_and_ascii(tmp_path):
    xyz = np.array([[0.0, 1.0, 2.0], [0.5, 0.25, -1.0]])
    rgb = np.array([[1, 2, 3], [250, 0, 128]])
    for fmt in ("binary_little_endian", "ascii"):
        p = tmp_path / f"{fmt}.ply"
        write_ply(p, xyz, rgb, fmt=fmt)
        x, c = U._read_ply_numpy(p)
        assert np.allclose(x, xyz) and np.allclose(c, rgb)
        x2, c2 = U.read_ply(p)
        assert np.allclose(x2, xyz) and np.allclose(c2, rgb)


def test_voxel_downsample_centroids_and_colors():
    xyz = np.array([[0.001, 0.001, 0.001], [0.009, 0.001, 0.001], [0.05, 0.05, 0.05]])
    rgb = np.array([[0, 0, 0], [200, 100, 50], [10, 10, 10]], dtype=float)
    ds, cds, inv = U.voxel_downsample(xyz, rgb, 0.02)
    assert ds.shape == (2, 3) and inv.shape == (3,) and inv[0] == inv[1] != inv[2]
    i = inv[0]
    assert np.allclose(ds[i], [0.005, 0.001, 0.001]) and np.allclose(cds[i], [100, 50, 25])


def test_grid_tree_matches_brute_force():
    rng = np.random.RandomState(1)
    pts = rng.uniform(0, 1, (300, 3))
    q = rng.uniform(0, 1, (50, 3))
    tree = U._GridTree(pts, 0.1)
    d, idx = tree.query(q, k=1, distance_upper_bound=0.1)
    D = np.linalg.norm(q[:, None] - pts[None], axis=2)
    bf = D.argmin(1)
    ok = D[np.arange(len(q)), bf] <= 0.1
    assert np.array_equal(idx[ok], bf[ok]) and np.all(idx[~ok] == len(pts))
    for p in q[:5]:
        got = sorted(tree.query_ball_point(p, 0.1))
        exp = sorted(np.flatnonzero(np.linalg.norm(pts - p, axis=1) <= 0.1).tolist())
        assert got == exp
    assert np.all(U.nearest_within(pts, q, 0.1) == np.where(ok, bf, -1))


def test_convert_scene_layout_and_labels(scene):
    fb = U.convert_scene(VISIT, scene["scans"], scene["out"], "train", voxel=0.05)
    out = scene["out"]
    pts = np.load(out / "train" / f"{VISIT}.npy")
    assert pts.dtype == np.float32 and pts.ndim == 2 and pts.shape[1] == 13
    sem, inst, seg, inter = pts[:, 9], pts[:, 10], pts[:, 11], pts[:, 12]
    assert set(np.unique(sem).tolist()) <= {0, 1, 2}
    assert set(np.unique(inst).tolist()) == {0, 1, 2}
    assert np.array_equal(inst, inter) and not seg.any()
    assert not pts[:, 6:9].any()  # normals unused (add_normals false) -> zeros
    assert pts[:, 3:6].min() >= 0 and pts[:, 3:6].max() <= 255
    # rot instance first (motions.json order) with sem 1, trans second with sem 2
    assert set(np.unique(sem[inst == 1]).tolist()) == {1}
    assert set(np.unique(sem[inst == 2]).tolist()) == {2}

    # T_up json and geometry: inverse-transform inst-1 points -> laser x < 0.2 (+voxel)
    T = json.loads((out / "train" / f"{VISIT}_T_up.json").read_text())
    T_up = np.asarray(T["T_up"])
    assert np.allclose(T_up, U.R_YUP_TO_ZUP @ laser_to_arkit_matrix())
    assert T["instances"] == {"1": "rot-uuid", "2": "trans-uuid"}
    inv = np.linalg.inv(T_up)
    p_laser = (inv[:3, :3] @ pts[inst == 1, :3].astype(np.float64).T).T + inv[:3, 3]
    assert p_laser[:, 0].max() < 0.2 + 0.05 and len(p_laser) > 10
    p_laser2 = (inv[:3, :3] @ pts[inst == 2, :3].astype(np.float64).T).T + inv[:3, 3]
    assert p_laser2[:, 0].min() > 0.8 - 0.05

    # instance_gt = sem*1000 + inst + 1, background -> 1
    gt = np.loadtxt(out / "instance_gt" / "train" / f"{VISIT}.txt", dtype=np.int64)
    assert set(np.unique(gt).tolist()) == {1, 1002, 2003}
    assert np.array_equal(gt, (sem * 1000 + inst + 1).astype(np.int64))

    # articulation h5: keys "1","2"; axis = R_up @ dir (unit); origin = T_up @ laser point
    with h5py.File(out / "train" / f"{VISIT}_articulation.h5", "r") as h:
        assert set(h.keys()) == {"1", "2"}
        g = h["1"]
        assert set(g.keys()) == {"axis", "inter_mask", "origin", "sem_id"}
        assert int(g["sem_id"][()]) == 1 and int(h["2"]["sem_id"][()]) == 2
        R = T_up[:3, :3]
        assert np.allclose(g["axis"][()], R @ np.array([0.0, 1.0, 0.0]))
        assert np.allclose(h["2"]["axis"][()], R @ np.array([1.0, 0.0, 0.0]))
        p0 = scene["xyz"][scene["rot_idx"][0]]
        assert np.allclose(g["origin"][()], R @ p0 + T_up[:3, 3])
        im = g["inter_mask"][()]
        assert im.dtype == bool and im.shape == ((inst == 1).sum(),) and im.all()

    # expand dict (upstream function) records background points only
    with open(out / "expand_dict" / f"{VISIT}.pkl", "rb") as f:
        ed = pickle.load(f)
    assert set(ed) == {"expand_idx_records", "expand_inst_records", "expand_sem_records", "expand_distances"}
    if len(ed["expand_idx_records"]):
        assert not inst[ed["expand_idx_records"]].any()
        assert set(np.unique(ed["expand_inst_records"]).tolist()) <= {1, 2}
        assert ed["expand_distances"].max() <= 0.1 + 1e-9

    # filebase entry (upstream key names; color_std is E[x^2])
    assert set(fb) == {"filepath", "raw_filepath", "scene", "color_mean", "color_std",
                       "instance_gt_filepath", "expand_dict_file", "articulation_gt_file"}
    assert fb["scene"] == VISIT and fb["filepath"].endswith(f"train/{VISIT}.npy")
    c = pts[:, 3:6].astype(np.float64) / 255.0
    assert np.allclose(fb["color_mean"], c.mean(0)) and np.allclose(fb["color_std"], (c ** 2).mean(0))


def test_scene_with_no_motions_has_zero_instances(tmp_path):
    scans = tmp_path / "scans"
    make_scene(scans, n=500, with_motions=False)
    fb = U.convert_scene(VISIT, scans, tmp_path / "usd", "test", voxel=0.1)
    assert U.n_instances(fb) == 0
    pts = np.load(fb["filepath"])
    assert not pts[:, 9:].any()
    gt = np.loadtxt(fb["instance_gt_filepath"], dtype=np.int64)
    assert set(np.unique(gt).tolist()) == {1}


def test_convert_job_is_idempotent(scene):
    args = (VISIT, str(scene["scans"]), str(scene["out"]), "validation", 0.05, False)
    v, m, fb1, err = U._convert_job(args)
    assert err is None and m == "validation"
    npy = scene["out"] / "validation" / f"{VISIT}.npy"
    mtime = npy.stat().st_mtime_ns
    v, m, fb2, err = U._convert_job(args)
    assert err is None and fb2 == fb1 and npy.stat().st_mtime_ns == mtime
    v, m, fb3, err = U._convert_job((VISIT, str(scene["scans"]), str(scene["out"]), "validation", 0.05, True))
    assert err is None and fb3 == fb1


def test_cli_writes_databases(scene, tmp_path):
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test": [VISIT], "train": [VISIT], "bvalid": []}))
    rc = U.main(["--scans", str(scene["scans"]), "--out", str(scene["out"]), "--splits", str(splits),
                 "--workers", "1", "--voxel", "0.05"])
    assert rc == 0
    out = scene["out"]
    for name in ("train", "validation", "test", "train_validation"):
        assert (out / f"{name}_database.yaml").is_file()
    train_db = yaml.safe_load((out / "train_database.yaml").read_text())
    assert len(train_db) == 1 and train_db[0]["scene"] == VISIT
    assert train_db[0]["filepath"] == str(out / "train" / f"{VISIT}.npy")
    assert yaml.safe_load((out / "validation_database.yaml").read_text()) == []
    assert len(yaml.safe_load((out / "test_database.yaml").read_text())) == 1
    assert yaml.safe_load((out / "train_validation_database.yaml").read_text()) == train_db
    lab = yaml.safe_load((out / "label_database.yaml").read_text())
    assert set(lab) == {0, 1, 2}
    assert lab[1]["name"] == "rotation" and lab[1]["validation"] is True
    assert lab[2]["name"] == "translation" and lab[2]["validation"] is True
    assert lab[0]["name"] == "background" and lab[0]["validation"] is False
    cms = yaml.safe_load((out / "color_mean_std.yaml").read_text())
    assert set(cms) == {"mean", "std"} and len(cms["mean"]) == 3
    assert all(0.0 <= v <= 1.0 for v in cms["mean"]) and all(0.0 <= v <= 0.5 for v in cms["std"])
    # the test-mode outputs are written too (we have GT for the test split)
    assert (out / "test" / f"{VISIT}_articulation.h5").is_file()
    assert (out / "instance_gt" / "test" / f"{VISIT}.txt").is_file()


def test_cli_reports_failures(tmp_path):
    (tmp_path / "scans").mkdir()
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test": [], "train": ["000000"], "bvalid": []}))
    rc = U.main(["--scans", str(tmp_path / "scans"), "--out", str(tmp_path / "usd"), "--splits", str(splits)])
    assert rc == 1
