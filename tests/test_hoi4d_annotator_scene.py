import importlib.util, pathlib, pickle
import numpy as np
import cv2, lmdb, pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ann", ROOT / "tools" / "hoi4d_annotate_articulation.py")
ann = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ann)

SEQ = "ZY20210800001_H1_C4_N01_S19_s05_T1"
KEYS = [f"ZY20210800001_H1_C4_N01/S19_s05_T1_w0_f{f:03d}" for f in (10, 12)]
SIZE = 64


@pytest.fixture()
def data_root(tmp_path):
    K = np.array([[80.0, 0, 32], [0, 80.0, 32], [0, 0, 1]])
    env_d = lmdb.open(str(tmp_path / "data.lmdb"), map_size=1 << 24)
    env_f = lmdb.open(str(tmp_path / "frames.lmdb"), map_size=1 << 26)
    rgb = np.full((SIZE, SIZE, 3), 128, np.uint8)
    depth = np.full((SIZE, SIZE), 1500, np.uint16)  # 1.5 m everywhere
    ok, jpeg = cv2.imencode(".jpg", rgb); ok2, dpng = cv2.imencode(".png", depth)
    for key in KEYS:
        rec = {"camera_intrinsics": K.tolist(),
               "mask_coordinates_yx": [[32, 32], [40, 40]],
               "trajectory_3d_camera_coords": [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
               "hoi4d": {"seq": SEQ, "event": "open", "window": 0,
                          "category": "C4", "wrist_frame_0based": int(key[-3:])}}
        with env_d.begin(write=True) as txn:
            txn.put(key.encode(), pickle.dumps(rec))
        with env_f.begin(write=True) as txn:
            txn.put(key.encode(), pickle.dumps(
                {"jpeg": jpeg.tobytes(), "depth_png": dpng.tobytes(),
                 "orig_size": (SIZE, SIZE)}))
    env_d.close(); env_f.close()
    hands = tmp_path / "hands" / SEQ / "camera"
    hands.mkdir(parents=True)
    poses = np.tile(np.eye(4), (300, 1, 1))
    poses[:, 0, 3] = np.arange(300) * 0.01        # x shifts per frame
    np.save(hands / "official_poses.npy", poses)
    np.save(hands / "intrinsic.npy", K)
    return tmp_path


def test_dataset_grouping_and_frame(data_root):
    ds = ann.Dataset(data_root, data_root / "hands")
    assert list(ds.sequences) == [SEQ] and ds.sequences[SEQ] == sorted(KEYS)
    rgb, depth, K, orig_size = ds.frame(KEYS[0])
    assert rgb.shape == (SIZE, SIZE, 3) and depth.dtype == np.uint16
    assert orig_size == (SIZE, SIZE)
    np.testing.assert_allclose(K[0, 0], 80.0)     # orig_size == stored size here
    pose = ds.pose(SEQ, 10)
    np.testing.assert_allclose(pose[0, 3], 0.10)


def test_build_scene_world_frame_and_cache(data_root, tmp_path):
    ds = ann.Dataset(data_root, data_root / "hands")
    cache = tmp_path / "cache"
    scene = ann.build_scene(ds, SEQ, cache)
    assert scene["points"].shape[1] == 3 and len(scene["points"]) > 0
    xs = scene["points"][:, 0]
    assert xs.min() >= -1.0 and xs.max() <= 2.0    # sane world range
    w0 = scene["windows"][0]
    assert w0["event"] == "open" and w0["mask_points"].shape[1] == 3
    assert w0["traj"].shape == (2, 3)
    # trajectory lifted with the sample frame's pose: cam [0,0,1] + x-shift
    np.testing.assert_allclose(w0["traj"][0], [0.10, 0.0, 1.0], atol=1e-9)
    assert (cache / f"{SEQ}.npz").exists()
    scene2 = ann.build_scene(ds, SEQ, cache)       # cache hit
    np.testing.assert_allclose(scene2["points"], scene["points"])
    assert scene2["windows"][0]["event"] == "open"
