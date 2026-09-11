"""Shared helpers for the external SF3D baselines (OPDFormer / MOPD / USDNet).

Pure numpy + pycocotools; no torch and no repo imports, so the converters can
run inside the upstream repos' environments. Plan:
docs/superpowers/plans/2026-09-12-sf3d-external-baselines.md
"""
import pickle
import random
from pathlib import Path

import numpy as np
from pycocotools import mask as _rle

KEY_CACHE = "/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl"
LMDB_ROOT = "/workspace/datasets/sf3d_processed_v3"
BASE = "/workspace/datasets/baselines"
FRAME_H, FRAME_W = 1440, 1920
# OPDMulti stores camera-frame geometry in an OpenGL-style camera (y up, -z
# forward): their preprocessing applies diag(1,-1,-1,1) to the OpenCV frame.
F_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


def load_keys(key_cache_path=KEY_CACHE):
    cache = pickle.loads(Path(key_cache_path).read_bytes())
    return [k.decode() if isinstance(k, bytes) else k for k in cache["keys"]]


def scene_of(key):
    return key.split("/")[0]


def video_of(key):
    return key.split("/")[1]


def frame_of(key):
    return "/".join(key.split("/")[:3])


def split_scenes(keys, val_ratio=0.1, seed=42, bval_ratio=0.1, bval_seed=4242):
    """test = the repo's val split (datasets.scenefun3d.split_dataset_by_scene);
    bvalid = 10% of the remaining scenes for the baselines' own checkpoint
    selection; train = the rest."""
    ids = sorted({scene_of(k) for k in keys})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * val_ratio))
    test, rest = ids[:n_test], sorted(ids[n_test:])
    rng2 = random.Random(bval_seed)
    rest_sh = list(rest)
    rng2.shuffle(rest_sh)
    n_bv = int(round(len(rest) * bval_ratio))
    bvalid = sorted(rest_sh[:n_bv])
    train = sorted(set(rest) - set(bvalid))
    return {"test": sorted(test), "train": train, "bvalid": bvalid}


def keys_by_split(keys, splits):
    where = {s: name for name, ss in splits.items() if isinstance(ss, list) for s in ss}
    out = {name: [] for name, ss in splits.items() if isinstance(ss, list)}
    for k in keys:
        out[where[scene_of(k)]].append(k)
    return out


def open_lmdb(root=LMDB_ROOT):
    import lmdb

    return lmdb.open(str(Path(root) / "data.lmdb"), readonly=True, lock=False, max_readers=64)


def read_record(txn, key):
    raw = txn.get(key.encode())
    assert raw is not None, key
    return pickle.loads(raw)


def cam_to_opd(v):
    """OpenCV camera (x right, y down, z fwd) -> OPDMulti camera (x, -y, -z).
    Works for points and directions; it is its own inverse."""
    v = np.asarray(v, dtype=np.float64)
    out = v.copy()
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


opd_to_cam = cam_to_opd


def mask_from_coords(coords_yx, h=FRAME_H, w=FRAME_W):
    m = np.zeros((h, w), np.uint8)
    if coords_yx is not None and len(coords_yx):
        c = np.asarray(coords_yx, dtype=np.int64)
        m[c[:, 0], c[:, 1]] = 1
    return m


def rle_encode(mask):
    r = _rle.encode(np.asfortranarray(np.asarray(mask).astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode()}


def rle_decode(rle):
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode()
    return _rle.decode({"size": list(rle["size"]), "counts": counts}).astype(bool)


def nearest_resize(mask, out_hw):
    """Nearest resample on PIL's grid (src = floor((dst+0.5)*scale)), which is
    what datasets/scenefun3d.py's fast pipeline uses for GT masks."""
    mask = np.asarray(mask)
    h, w = mask.shape[:2]
    th, tw = out_hw
    r = np.minimum(((np.arange(th) + 0.5) * (h / th)).astype(np.int64), h - 1)
    c = np.minimum(((np.arange(tw) + 0.5) * (w / tw)).astype(np.int64), w - 1)
    return mask[np.ix_(r, c)]
