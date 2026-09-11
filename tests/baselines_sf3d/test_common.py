import random

import numpy as np

from tools.baselines_sf3d import common as C


def test_split_scenes_is_scene_disjoint_and_deterministic():
    keys = [f"s{i:03d}/v{i}/1.0/a{j}" for i in range(30) for j in range(3)]
    a = C.split_scenes(keys)
    b = C.split_scenes(keys)
    assert a == b
    assert set(a["test"]).isdisjoint(a["train"]) and set(a["bvalid"]).isdisjoint(a["train"])
    assert set(a["test"]).isdisjoint(a["bvalid"])
    assert len(a["test"]) == 3 and len(a["bvalid"]) == 3 and len(a["train"]) == 24
    kb = C.keys_by_split(keys, a)
    assert sum(len(v) for v in kb.values()) == len(keys)


def test_split_reproduces_repo_split():
    keys = [f"s{i:03d}/v/1.0/a" for i in range(224)]
    ids = sorted({k.split("/")[0] for k in keys})
    rng = random.Random(42)
    rng.shuffle(ids)
    assert C.split_scenes(keys)["test"] == sorted(ids[:22])


def test_cam_to_opd_is_involution():
    v = np.array([0.3, -0.2, 2.0])
    assert np.allclose(C.opd_to_cam(C.cam_to_opd(v)), v)
    assert np.allclose(C.cam_to_opd(v), [0.3, 0.2, -2.0])


def test_nearest_resize_matches_pil_grid():
    m = np.zeros((8, 8), np.uint8)
    m[4:, 4:] = 1
    small = C.nearest_resize(m, (4, 4))
    assert small.shape == (4, 4) and small[2:, 2:].all() and not small[:2, :2].any()


def test_rle_roundtrip():
    m = np.zeros((6, 5), bool)
    m[1:4, 2:5] = True
    assert (C.rle_decode(C.rle_encode(m)) == m).all()


def test_mask_from_coords_and_key_parsing():
    m = C.mask_from_coords([[1, 2], [3, 4]], 5, 6)
    assert m.sum() == 2 and m[1, 2] == 1 and m[3, 4] == 1
    k = "420693/42445255/1234.567/annot"
    assert C.scene_of(k) == "420693" and C.video_of(k) == "42445255"
    assert C.frame_of(k) == "420693/42445255/1234.567"
