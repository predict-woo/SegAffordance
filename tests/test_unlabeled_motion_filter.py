"""motion_type "none" records (HOI4D dump / stapler / pliers / scissors, 2026-09-11)
stay in the LMDB but are left out of the training key list by default."""
import pickle

import lmdb
import pytest

from datasets.scenefun3d import SF3DDataset


def _rec(mt):
    return {"motion_info": {"original_motion_data": {"motion_type": mt}}, "trajectory_2d_image_coords": [[100.0, 100.0]] * 3}


def test_motion_unlabeled_predicate():
    assert SF3DDataset._motion_unlabeled(_rec("none"))
    assert not SF3DDataset._motion_unlabeled(_rec("rot"))
    assert not SF3DDataset._motion_unlabeled(_rec("trans"))
    assert not SF3DDataset._motion_unlabeled({"motion_info": {}})      # missing field: kept


def test_key_scan_drops_none_and_cache_key_tracks_the_flag(tmp_path):
    env = lmdb.open(str(tmp_path / "data.lmdb"), map_size=1 << 24)
    with env.begin(write=True) as t:
        for i, mt in enumerate(["rot", "none", "trans", "none"]):
            t.put(f"k{i}".encode(), pickle.dumps(_rec(mt)))
    env.close()

    class Stub(SF3DDataset):                       # bypass the full constructor
        def __init__(self, path, skip, cache):
            self.env = lmdb.open(str(path), readonly=True, lock=False)
            self.sensor_max_occluded_frac = None; self.min_revolute_radius = 0.0
            self.min_mask_area_frac = 0.0; self.edge_margin_frac = 0.0
            self.skip_unlabeled_motion = skip; self.key_cache_path = cache

    cache = tmp_path / "keys.pkl"
    keys = Stub(tmp_path / "data.lmdb", True, cache)._get_item_keys()
    assert sorted(keys) == [b"k0", b"k2"]
    cached = pickle.loads(cache.read_bytes())
    assert cached["skip_unlabeled_motion"] is True
    # a cache built with the flag on is NOT reused when the flag is off
    keys_all = Stub(tmp_path / "data.lmdb", False, cache)._get_item_keys()
    assert sorted(keys_all) == [b"k0", b"k1", b"k2", b"k3"]
