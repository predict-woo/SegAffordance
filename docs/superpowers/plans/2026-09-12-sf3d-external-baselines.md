# SF3D External Baselines (OPDFormer, MOPD, USDNet) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain OPDFormer-C (RGB-D), OPDFormer-P (RGB-D), OPDFormer-P (RGB), MOPD (RGB) and USDNet on the SF3D train split with each method's own released code and recipe, and score all of them on our fixed 5,088-sample SF3D test split with our metric definitions, so their rows drop straight into `experiments/INDEX.md`.

**Architecture:** Each baseline runs in its upstream repo, cloned unmodified at a pinned commit under `/workspace/datasets/baselines/repos/` on the volume (sync-ignored) and patched only where the data format forces it. Two converters (`sf3d_to_opd.py`, `sf3d_to_usdnet.py`) write the upstream native training formats from our LMDB plus (for USDNet) the SceneFun3D laser scans. Each method's predictions are exported into one shared JSONL schema and scored by one scorer that imports our `SF3DDataset`, our scene split and the metric formulas from `train_SF3D_better.py::test_step`. Four A100 pods run in parallel; the dev pod only runs the scorer (<30 min jobs).

**Tech Stack:** OPDMulti/MOPD (detectron2 + Mask2Former, torch 2.1.1 cu121, built from source), USDNet (Mask3D + MinkowskiEngine, torch 2.1.1 cu121, previous session's `runpod/external/usdnet/setup_env.sh`), our repo's `datasets.scenefun3d`, `pycocotools`, `open3d`, RunPod A100 80GB PCIe pods on the main volume `bckt1t9uuf` (EU-RO-1).

**Spec:** the conversation of 2026-09-11/12 with the user (no separate spec file). Decisions: retrain faithfully with upstream code; edit upstream only to fit our data; OPDFormer C+P RGB-D, P RGB (MOPD init), MOPD, USDNet; run in parallel; user asleep, autonomous.

## Global Constraints

- Never edit `model/`, `datasets/`, `train_*_better.py`, `config/opd_train.py`, `config/joint4_*.yaml`, `config/sf3d_test_decoder_rgb_scalefree.yaml`, `config/sf3d_train_runpod_cf_*.yaml`, repo-root `run_*_chain.sh`, `tools/hoi4d_*.py`, `tools/arctic_axis_probe.py`, `tools/epic_vlm_label_types.py`, `experiments/20260911_*`, `experiments/20260912_*`, the top of `STATE.md`, the main table of `experiments/INDEX.md`, `viz/INDEX.md` (owned by session ethz-workspace-c6). Import from `datasets/` and `model/` freely.
- My paths only: `tools/baselines_sf3d/`, `runpod/baselines/`, `experiments/baselines_sf3d/`, `config/baselines/`, `docs/superpowers/plans/2026-09-12-sf3d-external-baselines.md`, one new section at the bottom of `STATE.md` titled `## External baselines on SF3D (session ethz-workspace-34)`, one new table `## Baselines (external models retrained on SF3D)` at the bottom of `experiments/INDEX.md`.
- Volume layout (all sync-ignored because under `datasets/`): `/workspace/datasets/baselines/{repos,data,runs,logs,ckpt}`.
- Test split = `split_dataset_by_scene(ratio 0.1, seed 42)` on key cache `/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl` (59,174 keys) = 22 scenes / 5,088 samples; train = 202 scenes / 54,086 samples. Baseline "valid" for checkpoint selection = 10% of the 202 train scenes (`random.Random(4242)` shuffle of the sorted list, first 20).
- Metrics exactly as `train_SF3D_better.py::test_step`: mask IoU at 512x512 vs the fast-pipeline GT mask, PDet = IoU>0.5, type accuracy, MA = type correct AND unsigned axis error <= 10 deg (over all test samples, no IoU gate), signed axis error and flip rate, `origin_err_m` = ||q_hat - q*|| with q* = perpendicular foot of traj[0] on the GT axis, `origin_line_err_m` = distance of q_hat to the GT axis line, rotational rows only.
- Unmatched detection (no prediction overlapping the GT element) counts as IoU 0, type wrong, axis error 90 deg, origin error absent.
- Pods: A100 80GB PCIe only (torch 2.1 has no sm_120), image `runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04`, `export TMPDIR=/workspace/tmp`, delete each pod right after its export; reconcile `runpodctl pod list` after every create; dev pod stays running.
- Git: Mac-side only, `git add` explicit paths, message ethz-workspace-c6 before committing STATE/INDEX, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_01XfvLBRm9NN9i4jBFQCbfJs`.

---

## File map

| file | responsibility |
|---|---|
| `tools/baselines_sf3d/common.py` | key list, scene split (test / train / baseline-valid), frame grouping, record reading, RLE helpers, camera-frame conversions. Pure functions, no torch. |
| `tools/baselines_sf3d/score_predictions.py` | JSONL predictions -> `metrics.json` with our metric definitions (imports `SF3DDataset`). |
| `tools/baselines_sf3d/sf3d_to_opd.py` | LMDB -> OPDMulti `MotionDataset_h5` layout + `obj_info.json` + `stats.json`. |
| `tools/baselines_sf3d/opd_preds_to_jsonl.py` | OPDFormer `instances_predictions.pth` -> shared JSONL. |
| `tools/baselines_sf3d/sf3d_to_usdnet.py` | laser scans + annotations + motions -> USDNet processed layout (`.npy`, `instance_gt`, `expand_dict`, `_articulation.h5`, database yamls). |
| `tools/baselines_sf3d/usdnet_preds_to_jsonl.py` | USDNet `preds.pkl` -> per-frame shared JSONL (projection + IoU matching). |
| `tools/baselines_sf3d/mopd_compose_ckpt.py` | OPDFormer-P RGB weights + EfficientSAM ViT-S -> MOPD full state dict. |
| `runpod/baselines/pods.sh` | thin wrapper over `runpod/external/common/pods.sh` with A100 + main volume defaults. |
| `runpod/baselines/opd/setup_env.sh`, `chain.sh` | env build (detectron2, MSDeformAttn), convert (owner pod), train, evaluate, export. |
| `runpod/baselines/mopd/setup_env.sh`, `chain.sh` | same for MOPD (after OPDFormer-P RGB). |
| `runpod/baselines/usdnet/download_scans.sh`, `chain.sh` | SceneFun3D asset download, convert, train, export. |
| `experiments/baselines_sf3d/splits.json` | the three scene lists (tracked). |
| `experiments/baselines_sf3d/<id>/{notes.md,config.yaml,metrics.json,train.log.tail}` | tracked results per baseline. |
| `experiments/baselines_sf3d/.gitignore` | `*/preds/`, `*.pth`, `*.pkl`. |
| `tests/baselines_sf3d/test_common.py`, `test_score.py`, `test_opd_convert.py`, `test_usdnet_convert.py` | unit tests (synthetic LMDB, no volume). |

## Shared prediction JSONL schema (all methods emit this)

One line per test key, in the order of `common.test_keys()`:

```json
{"key": "420693/42445255/1234.567/annot-uuid",
 "matched": true,
 "score": 0.83,
 "mask_rle": {"size": [1440, 1920], "counts": "..."},
 "type": 1,
 "axis_cam": [0.0, 0.85, 0.53],
 "origin_cam": [-1.27, -1.10, 3.27]}
```

`type`: 0 = translation, 1 = rotation (our convention). `axis_cam`/`origin_cam`: OpenCV camera frame of that frame (x right, y down, z forward), metres. `mask_rle`: pycocotools RLE at the frame's native resolution (H, W) = (1440, 1920). `matched=false` -> the other fields are `null`.

---

### Task 1: `common.py` — keys, splits, records, conversions

**Files:**
- Create: `tools/baselines_sf3d/__init__.py` (empty), `tools/baselines_sf3d/common.py`
- Create: `experiments/baselines_sf3d/splits.json`, `experiments/baselines_sf3d/.gitignore`
- Test: `tests/baselines_sf3d/__init__.py` (empty), `tests/baselines_sf3d/test_common.py`

**Interfaces:**
- Produces:
  - `KEY_CACHE = "/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl"`, `LMDB_ROOT = "/workspace/datasets/sf3d_processed_v3"`, `BASE = "/workspace/datasets/baselines"`
  - `load_keys(key_cache_path: str) -> list[str]` (decoded key strings, cache order)
  - `scene_of(key: str) -> str`, `frame_of(key: str) -> str` (`visit/video/timestamp`), `video_of(key) -> str`
  - `split_scenes(keys: list[str], val_ratio=0.1, seed=42, bval_ratio=0.1, bval_seed=4242) -> dict` with `{"test": [...], "train": [...], "bvalid": [...]}` sorted scene-id lists; `bvalid` is carved out of `train`; `train` excludes `bvalid`.
  - `keys_by_split(keys, splits) -> dict[str, list[str]]` preserving cache order.
  - `open_lmdb(root) -> lmdb.Environment`, `read_record(txn, key: str) -> dict`
  - `F_YZ = np.diag([1.0, -1.0, -1.0, 1.0])` and `cam_to_opd(v: np.ndarray) -> np.ndarray` (`(x, y, z) -> (x, -y, -z)`, works for points and directions), `opd_to_cam = cam_to_opd`
  - `mask_from_coords(coords_yx: list, h: int, w: int) -> np.ndarray[uint8]`
  - `rle_encode(mask: np.ndarray[bool]) -> dict` (`counts` as str), `rle_decode(rle: dict) -> np.ndarray[bool]`
  - `nearest_resize(mask: np.ndarray, out_hw: tuple) -> np.ndarray` using the PIL-NEAREST grid `src = floor((dst+0.5)*scale)` exactly as `datasets/scenefun3d.py` fast pipeline does.

- [ ] **Step 1: Write the failing tests**

```python
# tests/baselines_sf3d/test_common.py
import numpy as np
from tools.baselines_sf3d import common as C

def test_split_scenes_is_scene_disjoint_and_deterministic():
    keys = [f"s{i:03d}/v{i}/1.0/a{j}" for i in range(30) for j in range(3)]
    a = C.split_scenes(keys); b = C.split_scenes(keys)
    assert a == b
    assert set(a["test"]).isdisjoint(a["train"]) and set(a["bvalid"]).isdisjoint(a["train"])
    assert set(a["test"]).isdisjoint(a["bvalid"])
    assert len(a["test"]) == 3 and len(a["bvalid"]) == 3 and len(a["train"]) == 24

def test_split_reproduces_repo_split():
    # same algorithm as datasets.scenefun3d.split_dataset_by_scene: sorted ids, Random(42).shuffle, round(n*0.1)
    import random
    keys = [f"s{i:03d}/v/1.0/a" for i in range(224)]
    ids = sorted({k.split("/")[0] for k in keys}); rng = random.Random(42); rng.shuffle(ids)
    assert C.split_scenes(keys)["test"] == sorted(ids[:22])

def test_cam_to_opd_is_involution():
    v = np.array([0.3, -0.2, 2.0])
    assert np.allclose(C.opd_to_cam(C.cam_to_opd(v)), v)
    assert np.allclose(C.cam_to_opd(v), [0.3, 0.2, -2.0])

def test_nearest_resize_matches_pil_grid():
    m = np.zeros((8, 8), np.uint8); m[4:, 4:] = 1
    small = C.nearest_resize(m, (4, 4))
    assert small.shape == (4, 4) and small[2:, 2:].all() and not small[:2, :2].any()

def test_rle_roundtrip():
    m = np.zeros((6, 5), bool); m[1:4, 2:5] = True
    assert (C.rle_decode(C.rle_encode(m)) == m).all()
```

- [ ] **Step 2: Run to verify failure**

Run (dev pod, our venv): `bash runpod/dev.sh run "python -m pytest tests/baselines_sf3d/test_common.py -q"`
Expected: FAIL with `ModuleNotFoundError: tools.baselines_sf3d`

- [ ] **Step 3: Implement `common.py`**

```python
# tools/baselines_sf3d/common.py
"""Shared helpers for the external SF3D baselines (no torch, no repo imports)."""
import json, pickle, random
from pathlib import Path
import numpy as np
from pycocotools import mask as _rle

KEY_CACHE = "/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl"
LMDB_ROOT = "/workspace/datasets/sf3d_processed_v3"
BASE = "/workspace/datasets/baselines"
FRAME_H, FRAME_W = 1440, 1920
F_YZ = np.diag([1.0, -1.0, -1.0, 1.0])

def load_keys(key_cache_path=KEY_CACHE):
    cache = pickle.loads(Path(key_cache_path).read_bytes())
    return [k.decode() if isinstance(k, bytes) else k for k in cache["keys"]]

def scene_of(key): return key.split("/")[0]
def video_of(key): return key.split("/")[1]
def frame_of(key): return "/".join(key.split("/")[:3])

def split_scenes(keys, val_ratio=0.1, seed=42, bval_ratio=0.1, bval_seed=4242):
    ids = sorted({scene_of(k) for k in keys})
    rng = random.Random(seed); rng.shuffle(ids)
    n_test = int(round(len(ids) * val_ratio))
    test, rest = ids[:n_test], sorted(ids[n_test:])
    rng2 = random.Random(bval_seed); rest_sh = list(rest); rng2.shuffle(rest_sh)
    n_bv = int(round(len(rest) * bval_ratio))
    bvalid = sorted(rest_sh[:n_bv]); train = sorted(set(rest) - set(bvalid))
    return {"test": sorted(test), "train": train, "bvalid": bvalid}

def keys_by_split(keys, splits):
    where = {s: name for name, ss in splits.items() for s in ss}
    out = {name: [] for name in splits}
    for k in keys: out[where[scene_of(k)]].append(k)
    return out

def open_lmdb(root=LMDB_ROOT):
    import lmdb
    return lmdb.open(str(Path(root) / "data.lmdb"), readonly=True, lock=False, max_readers=64)

def read_record(txn, key):
    raw = txn.get(key.encode()); assert raw is not None, key
    return pickle.loads(raw)

def cam_to_opd(v):
    v = np.asarray(v, dtype=np.float64); out = v.copy(); out[..., 1] *= -1; out[..., 2] *= -1; return out
opd_to_cam = cam_to_opd

def mask_from_coords(coords_yx, h=FRAME_H, w=FRAME_W):
    m = np.zeros((h, w), np.uint8)
    if coords_yx:
        c = np.asarray(coords_yx, dtype=np.int64); m[c[:, 0], c[:, 1]] = 1
    return m

def rle_encode(mask):
    r = _rle.encode(np.asfortranarray(mask.astype(np.uint8))); r["counts"] = r["counts"].decode()
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"]}

def rle_decode(rle):
    return _rle.decode({"size": rle["size"], "counts": rle["counts"].encode()}).astype(bool)

def nearest_resize(mask, out_hw):
    h, w = mask.shape[:2]; th, tw = out_hw
    r = np.minimum(((np.arange(th) + 0.5) * (h / th)).astype(np.int64), h - 1)
    c = np.minimum(((np.arange(tw) + 0.5) * (w / tw)).astype(np.int64), w - 1)
    return mask[np.ix_(r, c)]
```

- [ ] **Step 4: Run tests**

Run: `bash runpod/dev.sh run "python -m pytest tests/baselines_sf3d/test_common.py -q"`
Expected: 5 passed

- [ ] **Step 5: Write `splits.json` from the real key cache (dev pod) and `.gitignore`**

```bash
bash runpod/dev.sh run "python -c \"
from tools.baselines_sf3d import common as C; import json
keys = C.load_keys(); s = C.split_scenes(keys); kb = C.keys_by_split(keys, s)
s['counts'] = {k: len(v) for k, v in kb.items()}
json.dump(s, open('experiments/baselines_sf3d/splits.json','w'), indent=1); print(s['counts'])\""
printf '*/preds/\n*.pth\n*.pkl\n*.h5\n' > experiments/baselines_sf3d/.gitignore
```
Expected counts: test 5088, train + bvalid = 54086.

- [ ] **Step 6: Commit**

```bash
git add tools/baselines_sf3d/__init__.py tools/baselines_sf3d/common.py tests/baselines_sf3d experiments/baselines_sf3d/splits.json experiments/baselines_sf3d/.gitignore docs/superpowers/plans/2026-09-12-sf3d-external-baselines.md
git commit -m "baselines: shared helpers, scene splits and plan for external SF3D baselines"
```

---

### Task 2: `score_predictions.py` — our metrics on the shared JSONL

**Files:**
- Create: `tools/baselines_sf3d/score_predictions.py`
- Test: `tests/baselines_sf3d/test_score.py`

**Interfaces:**
- Consumes: `common.rle_decode`, `common.nearest_resize`, `datasets.scenefun3d.SF3DDataset`, `split_dataset_by_scene`, `model.losses.split.perpendicular_foot`, `model.losses.twist.point_to_line_distance`.
- Produces: CLI `python tools/baselines_sf3d/score_predictions.py --preds P.jsonl --out metrics.json [--lmdb-root R --frame-cache F --key-cache K --limit N]`; function `score(preds_by_key: dict, gt_iter) -> dict` where `gt_iter` yields `(key, mask512: np.bool[512,512], type_gt: int, axis_gt: np.float[3], origin_gt: np.float[3], traj0_gt: np.float[3])`.
- Output keys (percent where our harness uses percent): `n, p_det, mean_iou, pass_rate_m, pass_rate_ma, pass_rate_ma_signed, err_adir_all_deg, err_adir_matched_deg, err_adir_signed_all_deg, axis_flip_rate, axis_flip_rate_rot, origin_err_m, origin_line_err_m, n_matched, n_rot`.

- [ ] **Step 1: Write the failing test (pure-function part, synthetic GT)**

```python
# tests/baselines_sf3d/test_score.py
import numpy as np
from tools.baselines_sf3d import common as C
from tools.baselines_sf3d.score_predictions import score

def _gt(n=4):
    for i in range(n):
        m = np.zeros((512, 512), bool); m[100:200, 100 + i:300] = True
        axis = np.array([0.0, 1.0, 0.0]); origin = np.array([0.0, 0.0, 2.0]); traj0 = np.array([0.5, 0.3, 2.0])
        yield f"s/v/{i}.0/a", m, 1 if i % 2 else 0, axis, origin, traj0

def test_perfect_predictions_score_100():
    preds = {}
    for key, m, t, a, o, p in _gt():
        full = C.nearest_resize(m.astype(np.uint8), (1440, 1920)).astype(bool)
        preds[key] = {"matched": True, "score": 1.0, "mask_rle": C.rle_encode(full), "type": t, "axis_cam": a.tolist(), "origin_cam": (o + a * 0.3).tolist()}
    r = score(preds, _gt())
    assert r["pass_rate_ma"] == 100.0 and r["pass_rate_m"] == 100.0 and r["p_det"] == 100.0
    assert r["err_adir_all_deg"] < 1e-4 and r["axis_flip_rate"] == 0.0
    assert r["origin_line_err_m"] < 1e-6      # sliding along the axis is free
    assert abs(r["origin_err_m"] - 0.0) < 1e-6  # q* is the foot of traj0; origin+0.3*axis: foot of traj0 has y=0.3 -> equal

def test_unmatched_counts_as_failure():
    preds = {key: {"matched": False, "score": None, "mask_rle": None, "type": None, "axis_cam": None, "origin_cam": None} for key, *_ in _gt()}
    r = score(preds, _gt())
    assert r["p_det"] == 0.0 and r["pass_rate_ma"] == 0.0 and r["mean_iou"] == 0.0
    assert r["err_adir_all_deg"] == 90.0
```

- [ ] **Step 2: Run to verify failure**

Run: `bash runpod/dev.sh run "python -m pytest tests/baselines_sf3d/test_score.py -q"` -> FAIL (module missing)

- [ ] **Step 3: Implement**

```python
# tools/baselines_sf3d/score_predictions.py
"""Score shared-JSONL baseline predictions with SegAffordance's SF3D test metrics.

Metric definitions mirror train_SF3D_better.py::test_step (2026-09-12):
IoU at 512x512 vs the fast-pipeline GT mask; PDet = IoU > 0.5; MA = type correct
AND unsigned axis error <= 10 deg, over ALL samples; flip = signed error > 90;
origin_err_m = ||q_hat - q*||, q* = foot of traj[0] on the GT axis; rot rows only.
"""
import argparse, json, math
import numpy as np
from tools.baselines_sf3d import common as C

THRESH_DEG, IOU_THRESH = 10.0, 0.5

def _angle(a, b, signed):
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    if not signed: c = abs(c)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))

def _foot(origin, d, p):
    d = d / (np.linalg.norm(d) + 1e-12); return origin + np.dot(p - origin, d) * d

def _line_dist(q, origin, d):
    d = d / (np.linalg.norm(d) + 1e-12); rel = q - origin; return float(np.linalg.norm(rel - np.dot(rel, d) * d))

def score(preds, gt_iter):
    ious, type_ok, ma, ma_s, err_all, err_matched, err_s_all, err_s_rot = [], [], [], [], [], [], [], []
    oerr, olerr, n_matched, n_rot = [], [], 0, 0
    for key, m_gt, t_gt, a_gt, o_gt, p_gt in gt_iter:
        pr = preds.get(key) or {"matched": False}
        if not pr.get("matched"):
            ious.append(0.0); type_ok.append(False); ma.append(False); ma_s.append(False)
            err_all.append(90.0); err_s_all.append(90.0)
            if t_gt == 1: err_s_rot.append(90.0); n_rot += 1
            continue
        n_matched += 1
        m_pr = C.nearest_resize(C.rle_decode(pr["mask_rle"]).astype(np.uint8), m_gt.shape).astype(bool)
        inter = np.logical_and(m_pr, m_gt).sum(); union = np.logical_or(m_pr, m_gt).sum()
        iou = float(inter / union) if union else 0.0; ious.append(iou)
        e = _angle(pr["axis_cam"], a_gt, False); es = _angle(pr["axis_cam"], a_gt, True)
        err_all.append(e); err_s_all.append(es)
        ok_t = int(pr["type"]) == int(t_gt); type_ok.append(ok_t)
        ma.append(ok_t and e <= THRESH_DEG); ma_s.append(ok_t and es <= THRESH_DEG)
        if ok_t and e <= THRESH_DEG: err_matched.append(e)
        if t_gt == 1:
            n_rot += 1; err_s_rot.append(es)
            if pr.get("origin_cam") is not None:
                q = np.asarray(pr["origin_cam"], float)
                oerr.append(float(np.linalg.norm(q - _foot(np.asarray(o_gt), np.asarray(a_gt), np.asarray(p_gt)))))
                olerr.append(_line_dist(q, np.asarray(o_gt), np.asarray(a_gt)))
    n = len(ious); pct = lambda xs: 100.0 * float(np.mean(xs)) if len(xs) else 0.0
    mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
    return {"n": n, "n_matched": n_matched, "n_rot": n_rot,
            "p_det": pct([i > IOU_THRESH for i in ious]), "mean_iou": mean(ious),
            "pass_rate_m": pct(type_ok), "pass_rate_ma": pct(ma), "pass_rate_ma_signed": pct(ma_s),
            "err_adir_all_deg": mean(err_all), "err_adir_matched_deg": mean(err_matched),
            "err_adir_signed_all_deg": mean(err_s_all),
            "axis_flip_rate": pct([e > 90.0 for e in err_s_all]), "axis_flip_rate_rot": pct([e > 90.0 for e in err_s_rot]),
            "origin_err_m": mean(oerr), "origin_line_err_m": mean(olerr)}

def gt_from_repo(lmdb_root, frame_cache, key_cache, limit=None):
    """Yield GT exactly as our test loader sees it (SF3DDataset fast pipeline, 512x512)."""
    import torch
    from datasets.scenefun3d import SF3DDataset, split_dataset_by_scene, get_default_transforms
    rgb_t, mask_t, depth_t = get_default_transforms((512, 512))
    ds = SF3DDataset(lmdb_data_root=lmdb_root, rgb_transform=rgb_t, mask_transform=mask_t, depth_transform=depth_t,
                     image_size_for_mask_reconstruction=(512, 512), key_cache_path=key_cache,
                     min_revolute_radius=0.1, min_mask_area_frac=0.001, edge_margin_frac=0.05,
                     return_trajectory_2d=True, point_source="element", frame_cache_path=frame_cache,
                     fast_pipeline=True, load_depth=False)
    _, val = split_dataset_by_scene(ds, 0.1, 42)
    for j, idx in enumerate(val.indices):
        if limit and j >= limit: break
        s = ds[idx]
        key = ds.item_keys[idx].decode()
        yield key, s[3][0].numpy() > 0.5, int(s[7]), s[6].numpy(), s[10].numpy(), s[12][0].numpy()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--preds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lmdb-root", default=C.LMDB_ROOT); ap.add_argument("--frame-cache", default="/workspace/datasets/sf3d_frames_512.lmdb")
    ap.add_argument("--key-cache", default=C.KEY_CACHE); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    preds = {}
    for line in open(a.preds):
        if line.strip(): d = json.loads(line); preds[d["key"]] = d
    r = score(preds, gt_from_repo(a.lmdb_root, a.frame_cache, a.key_cache, a.limit)); r["preds_file"] = a.preds
    json.dump(r, open(a.out, "w"), indent=1); print(json.dumps(r, indent=1))

if __name__ == "__main__": main()
```

- [ ] **Step 4: Run tests** -> 2 passed.

- [ ] **Step 5: Oracle check on the real split (dev pod)**: build a GT-as-prediction JSONL for 200 test keys from the LMDB (mask from `mask_coordinates_yx`, GT type/axis/origin) and score it; expect `pass_rate_ma 100`, `mean_iou` close to 1 (the only loss is the 1920x1440 -> 512 nearest path, identical on both sides so exactly 1.0), `origin_err_m` = distance from the annotated origin to q* (not 0; record the number in notes as the "annotation gauge" floor).

```bash
bash runpod/dev.sh run "python - <<'EOF'
import json, numpy as np
from tools.baselines_sf3d import common as C
keys = C.load_keys(); s = C.split_scenes(keys); test = C.keys_by_split(keys, s)['test'][:200]
env = C.open_lmdb()
with env.begin() as txn, open('/workspace/datasets/baselines/oracle200.jsonl','w') as f:
    for k in test:
        r = C.read_record(txn, k); m = r['motion_info']['frame_specific_motion_data']
        f.write(json.dumps({'key': k, 'matched': True, 'score': 1.0, 'mask_rle': C.rle_encode(C.mask_from_coords(r['mask_coordinates_yx'])),
            'type': 1 if r['motion_info']['original_motion_data']['motion_type']=='rot' else 0,
            'axis_cam': m['motion_dir_3d_camera_coords'], 'origin_cam': m['motion_origin_3d_camera_coords']})+'\n')
EOF
mkdir -p /workspace/datasets/baselines && python tools/baselines_sf3d/score_predictions.py --preds /workspace/datasets/baselines/oracle200.jsonl --out /workspace/datasets/baselines/oracle200.json --limit 200"
```

- [ ] **Step 6: Commit** `git add tools/baselines_sf3d/score_predictions.py tests/baselines_sf3d/test_score.py && git commit -m "baselines: scorer with our SF3D test metrics on the shared JSONL"`

---

### Task 3: `sf3d_to_opd.py` — OPDMulti native format from our LMDB

**Files:**
- Create: `tools/baselines_sf3d/sf3d_to_opd.py`
- Test: `tests/baselines_sf3d/test_opd_convert.py`

**Interfaces:**
- Consumes: `common.*`.
- Produces: `<out>/MotionDataset_h5/{train,valid,test}.h5` (datasets `{split}_images` uint8 (N,192,256,3), `{split}_filenames` str), `<out>/MotionDataset_h5/depth.h5` (`depth_images` float32 (N,192,256,1) millimetres, `depth_filenames`), `<out>/MotionDataset_h5/annotations/MotionNet_{train,valid,test}.json`, `<out>/obj_info.json`, `<out>/stats.json` (`pixel_mean` [r,g,b,depth_mm], `pixel_std`, `categories`), `<out>/frames_{split}.json` (`file_name -> frame key`), `<out>/annots_{split}.json` (`annotation id -> our LMDB key`). Also function `convert_record(rec: dict, key: str, ann_id: int, image_id: int, file_name: str) -> dict` returning one COCO annotation.
- Filename convention: `f"{visit}-{frame_idx:06d}.png"`, depth `f"{visit}-{frame_idx:06d}_d.png"` with `frame_idx` = index of the frame in the sorted list of unique frame keys of that visit (the mapper needs an integer before `-`).
- Categories: SF3D affordance labels sorted alphabetically, ids 1..K (K = 8 in the filtered set: foot_push, hook_pull, hook_turn, key_press, pinch_pull, plug_in, tip_push, unplug; the converter derives them from the data and asserts against this list).
- Per-annotation `motion`: `type` = `"rotation"` if `motion_type == "rot"` else `"translation"`; `axis` = `cam_to_opd(motion_dir_3d_camera_coords)` unit-normalised; `origin` = `cam_to_opd(motion_origin_3d_camera_coords)`; `partId` = ann id; `part_label` = SF3D label; `isClosed` true; `rangeMin` 0; `rangeMax` 1.5707963 (rot) / 0.7 (trans); `state` 0; `pixel_num` = mask pixel count at 256x192; `bbox` = xywh at 256x192.
- `object_key` = `f"{visit}_{frame_idx:06d}_{ann_id}"`; `obj_info[object_key] = {"object_pose": c2w_opd 16 col-major, "diagonal": d, "min_bound": [...], "max_bound": [...]}` where `c2w_opd = camera_extrinsics_cam_to_world @ F_YZ` and `d` = bbox diagonal of the element's camera-frame points (mask pixels at native res back-projected with the frame depth and K), clamped to `>= 0.05`.
- Image entry: `file_name`, `depth_file_name`, `height 192`, `width 256`, `id`, `camera: {"intrinsic": K_scaled col-major 9, "extrinsic": c2w_opd col-major 16}` with `K_scaled = diag(256/1920, 192/1440, 1) @ K`.
- Segmentation: RLE dict (pycocotools) at 192x256 from `nearest_resize(mask_from_coords(...), (192, 256))`; `area` = pixel count; `bbox` xywh.
- Depth: `cv2.resize(uint16, (256,192), INTER_NEAREST).astype(float32)[..., None]` (millimetres, as in their writer).
- RGB: `cv2.resize(rgb, (256,192), INTER_AREA)`.
- `stats.json`: means/stds over the train split (RGB per channel over all pixels; depth over pixels > 0).
- CLI: `python tools/baselines_sf3d/sf3d_to_opd.py --out /workspace/datasets/baselines/data/opd_sf3d [--workers 32] [--limit-frames N]`; h5 written uncompressed with chunk (1,192,256,C); multiprocessing over frames with `imap` (each worker returns the two small arrays + the annotation dicts); idempotent per split via a `.done_<split>` marker.

- [ ] **Step 1: Write the failing test** (synthetic record, no volume)

```python
# tests/baselines_sf3d/test_opd_convert.py
import numpy as np
from tools.baselines_sf3d.sf3d_to_opd import convert_record, scale_intrinsics, colmajor16
from tools.baselines_sf3d import common as C

def _rec():
    return {"mask_coordinates_yx": [[y, x] for y in range(700, 760) for x in range(900, 1000)],
            "label_info": {"label": "hook_pull"},
            "motion_info": {"original_motion_data": {"motion_type": "rot"},
                            "frame_specific_motion_data": {"motion_dir_3d_camera_coords": [0.0, 2.0, 0.0],
                                                           "motion_origin_3d_camera_coords": [0.1, 0.2, 3.0]}},
            "camera_intrinsics": [[1600.0, 0, 960.0], [0, 1600.0, 720.0], [0, 0, 1]],
            "camera_extrinsics_cam_to_world": np.eye(4).tolist(), "image_dimensions_wh": (1920, 1440)}

def test_convert_record_geometry_and_mask():
    ann = convert_record(_rec(), "s/v/1.0/a", ann_id=7, image_id=3, categories={"hook_pull": 2})
    assert ann["category_id"] == 2 and ann["image_id"] == 3 and ann["id"] == 7
    assert ann["motion"]["type"] == "rotation"
    assert np.allclose(ann["motion"]["axis"], [0.0, -1.0, 0.0])          # unit + y/z flipped
    assert np.allclose(ann["motion"]["origin"], [0.1, -0.2, -3.0])
    assert ann["segmentation"]["size"] == [192, 256]
    x, y, w, h = ann["bbox"]; assert 115 <= x <= 122 and 90 <= y <= 95 and 10 <= w <= 15 and 6 <= h <= 10
    assert ann["area"] == ann["motion"]["pixel_num"] > 0

def test_scale_intrinsics_and_colmajor():
    K = scale_intrinsics(np.array([[1600.0, 0, 960.0], [0, 1600.0, 720.0], [0, 0, 1]]), (1920, 1440), (256, 192))
    assert np.allclose(K[0, 0], 1600 * 256 / 1920) and np.allclose(K[1, 2], 720 * 192 / 1440)
    M = np.arange(16.0).reshape(4, 4); flat = colmajor16(M)
    assert flat[1] == 4.0 and np.allclose(np.array(flat).reshape(4, 4).T, M)
```

- [ ] **Step 2: Run** -> FAIL (module missing).

- [ ] **Step 3: Implement** (full script; key functions shown, the CLI loops over frames per split and writes h5/json)

```python
# tools/baselines_sf3d/sf3d_to_opd.py
"""SF3D LMDB -> OPDMulti MotionDataset_h5 layout (see plan Task 3 for the contract)."""
import argparse, json, os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import cv2, h5py, numpy as np
from tools.baselines_sf3d import common as C

OUT_W, OUT_H = 256, 192
CATS = ["foot_push", "hook_pull", "hook_turn", "key_press", "pinch_pull", "plug_in", "tip_push", "unplug"]

def scale_intrinsics(K, in_wh, out_wh):
    S = np.diag([out_wh[0] / in_wh[0], out_wh[1] / in_wh[1], 1.0]); return S @ np.asarray(K, float)

def colmajor16(M): return np.asarray(M, float).flatten(order="F").tolist()

def convert_record(rec, key, ann_id, image_id, categories):
    w, h = rec["image_dimensions_wh"]
    m_full = C.mask_from_coords(rec["mask_coordinates_yx"], h, w)
    m = C.nearest_resize(m_full, (OUT_H, OUT_W)).astype(bool)
    ys, xs = np.where(m)
    bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)] if xs.size else [0.0, 0.0, 0.0, 0.0]
    mi = rec["motion_info"]; fs = mi["frame_specific_motion_data"]; rot = mi["original_motion_data"]["motion_type"] == "rot"
    axis = C.cam_to_opd(fs["motion_dir_3d_camera_coords"]); axis = axis / (np.linalg.norm(axis) + 1e-12)
    origin = C.cam_to_opd(fs["motion_origin_3d_camera_coords"])
    label = rec["label_info"]["label"]
    return {"id": ann_id, "image_id": image_id, "category_id": categories[label], "bbox": bbox,
            "area": int(m.sum()), "iscrowd": 0, "height": OUT_H, "width": OUT_W,
            "segmentation": C.rle_encode(m), "object_key": None,   # filled by caller
            "motion": {"type": "rotation" if rot else "translation", "axis": axis.tolist(), "origin": origin.tolist(),
                       "partId": ann_id, "part_label": label, "isClosed": True, "rangeMin": 0,
                       "rangeMax": 1.5707963267948966 if rot else 0.7, "state": 0, "pixel_num": float(m.sum()), "bbox": bbox}}

def element_diagonal(rec, depth_mm):
    """bbox diagonal (m) of the element's camera-frame points, >= 0.05."""
    w, h = rec["image_dimensions_wh"]; K = np.asarray(rec["camera_intrinsics"], float)
    c = np.asarray(rec["mask_coordinates_yx"], np.int64)
    z = depth_mm[c[:, 0], c[:, 1]].astype(np.float64) / 1000.0; ok = z > 1e-3
    if ok.sum() < 3: return 0.05
    x = (c[ok, 1] - K[0, 2]) * z[ok] / K[0, 0]; y = (c[ok, 0] - K[1, 2]) * z[ok] / K[1, 1]
    P = np.stack([x, y, z[ok]], 1); d = float(np.linalg.norm(P.max(0) - P.min(0)))
    return max(d, 0.05), P.min(0).tolist(), P.max(0).tolist()

def process_frame(args):
    """One frame: returns (file_name, rgb256, depth256, image_entry, [annotations], obj_info_items, frame_key)."""
    frame_key, keys, visit, frame_idx, image_id, ann_ids = args
    env = C.open_lmdb()
    with env.begin() as txn:
        recs = [C.read_record(txn, k) for k in keys]
    rec0 = recs[0]; root = Path(C.LMDB_ROOT)
    rgb = cv2.cvtColor(cv2.imread(str(root / "images" / rec0["rgb_image_path"])), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(root / "depth" / rec0["depth_image_path"]), cv2.IMREAD_UNCHANGED)
    rgb_s = cv2.resize(rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    depth_s = cv2.resize(depth, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST).astype(np.float32)[..., None]
    file_name = f"{visit}-{frame_idx:06d}.png"
    c2w_opd = np.asarray(rec0["camera_extrinsics_cam_to_world"], float) @ C.F_YZ
    K_s = scale_intrinsics(rec0["camera_intrinsics"], rec0["image_dimensions_wh"], (OUT_W, OUT_H))
    image = {"id": image_id, "file_name": file_name, "depth_file_name": f"{visit}-{frame_idx:06d}_d.png",
             "height": OUT_H, "width": OUT_W, "license": 1, "coco_url": "", "flickr_url": "", "date_captured": "",
             "camera": {"intrinsic": colmajor16(K_s)[:0] or np.asarray(K_s).flatten(order="F").tolist(), "extrinsic": colmajor16(c2w_opd)}}
    anns, obj = [], {}
    cats = {c: i + 1 for i, c in enumerate(CATS)}
    for rec, key, ann_id in zip(recs, keys, ann_ids):
        a = convert_record(rec, key, ann_id, image_id, cats); ok = f"{visit}_{frame_idx:06d}_{ann_id}"; a["object_key"] = ok; a["motion"]["object_key"] = ok
        d, mn, mx = element_diagonal(rec, depth) if len(rec["mask_coordinates_yx"]) else (0.05, [0, 0, 0], [0, 0, 0])
        obj[ok] = {"object_pose": colmajor16(c2w_opd), "diagonal": d, "min_bound": mn, "max_bound": mx}
        anns.append(a)
    return file_name, rgb_s, depth_s, image, anns, obj, frame_key, keys

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit-frames", type=int, default=None); a = ap.parse_args()
    out = Path(a.out); h5dir = out / "MotionDataset_h5"; (h5dir / "annotations").mkdir(parents=True, exist_ok=True)
    keys = C.load_keys(); splits = C.split_scenes(keys); kb = C.keys_by_split(keys, splits)
    name_map = {"train": "train", "bvalid": "valid", "test": "test"}
    obj_info, stats_rgb, stats_depth, all_depth = {}, [], [], []
    next_image_id, next_ann_id = 1, 1
    for split, opd_split in name_map.items():
        if (out / f".done_{opd_split}").exists(): continue
        frames = defaultdict(list)
        for k in kb[split]: frames[C.frame_of(k)].append(k)
        per_visit = defaultdict(list)
        for fk in frames: per_visit[C.scene_of(fk)].append(fk)
        jobs = []
        for visit, fks in per_visit.items():
            for idx, fk in enumerate(sorted(fks)):
                ks = frames[fk]; jobs.append((fk, ks, visit, idx, next_image_id, list(range(next_ann_id, next_ann_id + len(ks)))))
                next_image_id += 1; next_ann_id += len(ks)
        if a.limit_frames: jobs = jobs[: a.limit_frames]
        n = len(jobs); images, annotations, fmap, amap = [], [], {}, {}
        with h5py.File(h5dir / f"{opd_split}.h5", "w") as hf, h5py.File(h5dir / f"depth_{opd_split}.h5", "w") as hd:
            di = hf.create_dataset(f"{opd_split}_images", (n, OUT_H, OUT_W, 3), np.uint8, chunks=(1, OUT_H, OUT_W, 3))
            dn = hf.create_dataset(f"{opd_split}_filenames", (n,), h5py.string_dtype("utf-8"))
            dd = hd.create_dataset("depth_images", (n, OUT_H, OUT_W, 1), np.float32, chunks=(1, OUT_H, OUT_W, 1))
            ddn = hd.create_dataset("depth_filenames", (n,), h5py.string_dtype("utf-8"))
            with Pool(a.workers) as pool:
                for i, (fn, rgb, dep, image, anns, obj, fk, ks) in enumerate(pool.imap(process_frame, jobs, chunksize=8)):
                    di[i] = rgb; dn[i] = fn; dd[i] = dep; ddn[i] = image["depth_file_name"]
                    images.append(image); annotations.extend(anns); obj_info.update(obj); fmap[fn] = fk
                    for ann, k in zip(anns, ks): amap[ann["id"]] = k
                    if split == "train" and i % 50 == 0:
                        stats_rgb.append(rgb.reshape(-1, 3).astype(np.float64)); all_depth.append(dep[dep > 0].astype(np.float64))
                    if i % 1000 == 0: print(opd_split, i, "/", n, flush=True)
        json.dump({"images": images, "annotations": annotations, "info": {}, "licenses": [{"id": 1, "name": "sf3d"}],
                   "categories": [{"id": i + 1, "name": c, "supercategory": "sf3d"} for i, c in enumerate(CATS)]},
                  open(h5dir / "annotations" / f"MotionNet_{opd_split}.json", "w"))
        json.dump(fmap, open(out / f"frames_{opd_split}.json", "w")); json.dump(amap, open(out / f"annots_{opd_split}.json", "w"))
        (out / f".done_{opd_split}").touch()
    # merge the three depth h5 into depth.h5 (the mapper opens one file)
    with h5py.File(h5dir / "depth.h5", "w") as hd:
        parts = [h5py.File(h5dir / f"depth_{s}.h5", "r") for s in ["train", "valid", "test"] if (h5dir / f"depth_{s}.h5").exists()]
        n = sum(p["depth_images"].shape[0] for p in parts)
        dd = hd.create_dataset("depth_images", (n, OUT_H, OUT_W, 1), np.float32, chunks=(1, OUT_H, OUT_W, 1)); dn = hd.create_dataset("depth_filenames", (n,), h5py.string_dtype("utf-8"))
        i = 0
        for p in parts:
            m = p["depth_images"].shape[0]; dd[i:i + m] = p["depth_images"][:]; dn[i:i + m] = p["depth_filenames"][:]; i += m; p.close()
    json.dump(obj_info, open(out / "obj_info.json", "w"))
    if stats_rgb:
        R = np.concatenate(stats_rgb); D = np.concatenate(all_depth)
        json.dump({"pixel_mean": R.mean(0).tolist() + [float(D.mean())], "pixel_std": R.std(0).tolist() + [float(D.std())], "categories": CATS},
                  open(out / "stats.json", "w"), indent=1)
    print("done")

if __name__ == "__main__": main()
```

Note the `camera.intrinsic` value must be the 9-float column-major list (`np.asarray(K_s).flatten(order="F").tolist()`); remove the `colmajor16(K_s)[:0] or` no-op when implementing (kept here only to show the intended value).

- [ ] **Step 4: Run tests** -> 2 passed.
- [ ] **Step 5: Smoke on 20 frames on the dev pod**: `python tools/baselines_sf3d/sf3d_to_opd.py --out /workspace/datasets/baselines/data/opd_smoke --workers 8 --limit-frames 20` then verify with h5py that `train.h5` has 20 images, annotation json parses, every annotation's `object_key` is in `obj_info.json`, and `stats.json` exists. Delete `opd_smoke` afterwards.
- [ ] **Step 6: Commit** `git add tools/baselines_sf3d/sf3d_to_opd.py tests/baselines_sf3d/test_opd_convert.py && git commit -m "baselines: SF3D -> OPDMulti MotionDataset_h5 converter"`

---

### Task 4: `opd_preds_to_jsonl.py`

**Files:** Create `tools/baselines_sf3d/opd_preds_to_jsonl.py`; Test `tests/baselines_sf3d/test_opd_preds.py`.

**Interfaces:**
- Consumes: OPDFormer `<output-dir>/inference/instances_predictions.pth` (list of `{"image_id", "instances": [{"image_id","category_id","bbox","score","segmentation"(RLE 192x256),"mtype"(0 rot / 1 trans in THEIR convention: `MOTION_TYPE = {"rotation":0,"translation":1}`),"morigin","maxis"}]}`), `annots_test.json`, `frames_test.json`, `MotionNet_test.json`.
- Produces: shared JSONL. Function `match_and_convert(gt_ann: dict, instances: list) -> dict`: decode each instance RLE, upsample to 1440x1920 with `nearest_resize`, IoU vs GT annotation mask (decoded from its RLE at 192x256 then upsampled the same way so both sides share the resampling), pick max IoU; if max IoU == 0 -> unmatched. `type = 1 if mtype == 0 else 0`; `axis_cam = opd_to_cam(maxis)`, `origin_cam = opd_to_cam(morigin)`; `mask_rle` = RLE of the upsampled predicted mask.

- [ ] **Step 1: Failing test**: synthetic GT ann with a 10x10 square at 192x256 and two instances (one overlapping, one not) -> picks the overlapping one; mtype 0 -> type 1; axis flips y,z.
- [ ] **Step 2: Implement** (CLI `--pred PTH --data-dir /workspace/datasets/baselines/data/opd_sf3d --out preds.jsonl`; iterate `MotionNet_test.json` annotations grouped by image id; write one line per GT annotation in `common.test_keys` order using `annots_test.json`).
- [ ] **Step 3: Test passes; commit** `git add tools/baselines_sf3d/opd_preds_to_jsonl.py tests/baselines_sf3d/test_opd_preds.py && git commit -m "baselines: OPDFormer predictions -> shared JSONL"`

---

### Task 5: OPD pods — env, chain, launch (C RGB-D, P RGB-D, P RGB)

**Files:** Create `runpod/baselines/pods.sh`, `runpod/baselines/opd/setup_env.sh`, `runpod/baselines/opd/chain.sh`, `config/baselines/opd_sf3d_base.yaml` (documentation copy of the overrides used).

**Interfaces:**
- `pods.sh`: `bash runpod/baselines/pods.sh gpu <name>` creates an A100 80GB PCIe pod on `bckt1t9uuf` with `EXT_IMAGE=runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04` by delegating to `runpod/external/common/pods.sh` (`EXT_DC=EU-RO-1`); `run`, `scp-to`, `delete`, `list` pass through.
- `setup_env.sh` (runs on pod, idempotent markers in `/root`): apt `libgl1 ninja-build`; `pip install "numpy<2" opencv-python-headless h5py timm scipy shapely scikit-image cython pycocotools git+https://github.com/cocodataset/panopticapi.git`; `pip install 'git+https://github.com/facebookresearch/detectron2.git'` (source build ~10 min); clone `https://github.com/3dlg-hcvc/OPDMulti` into `/workspace/datasets/baselines/repos/OPDMulti` (once, shared on the volume; pin the commit hash printed by `git rev-parse HEAD` into notes); build `opdformer/mask2former/modeling/pixel_decoder/ops` with `python setup.py build install`; verify `python -c "import MultiScaleDeformableAttention"`.
- `chain.sh <variant>` with variant in `c_rgbd | p_rgbd | p_rgb`: sets `TMPDIR=/workspace/tmp`, waits for `/workspace/datasets/baselines/data/opd_sf3d/.done_all` (the `c_rgbd` pod is the converter owner: it runs `sf3d_to_opd.py` first and touches `.done_all`), then

```bash
cd /workspace/datasets/baselines/repos/OPDMulti/opdformer
CFG=configs/opd_${V}_real.yaml   # V=c|p
FMT=RGBD|RGB
python train.py --config-file $CFG --output-dir $RUNS/$variant --data-path $DATA/MotionDataset_h5 --input-format $FMT --model_attr_path $DATA/obj_info.json \
  --opts MODEL.SEM_SEG_HEAD.NUM_CLASSES 8 MODEL.PIXEL_MEAN "[$mean]" MODEL.PIXEL_STD "[$std]" SOLVER.IMS_PER_BATCH 16 DATASETS.TEST "('MotionNet_valid',)" > $LOGS/train_$variant.log 2>&1
python evaluate_on_log.py --config-file $CFG --output-dir $RUNS/$variant/test --data-path $DATA/MotionDataset_h5 --input-format $FMT --model_attr_path $DATA/obj_info.json \
  --opts MODEL.WEIGHTS $RUNS/$variant/model_final.pth MODEL.SEM_SEG_HEAD.NUM_CLASSES 8 MODEL.PIXEL_MEAN "[$mean]" MODEL.PIXEL_STD "[$std]" DATASETS.TEST "('MotionNet_test',)" > $LOGS/test_$variant.log 2>&1
python /workspace/SegAffordance/tools/baselines_sf3d/opd_preds_to_jsonl.py --pred $RUNS/$variant/test/inference/instances_predictions.pth --data-dir $DATA --out $RUNS/$variant/preds.jsonl
touch $RUNS/$variant/CHAIN_DONE
```
  where `$mean`/`$std` are read from `stats.json` (first 3 values for RGB, all 4 for RGBD) and the checkpoint used is the best `valid` one if `evaluate_on_log` supports picking, else `model_final.pth` (their recipe: fixed 60k iterations, final model; record which). Their eval only runs at CHECKPOINT_PERIOD; use `model_final.pth` (faithful to "train 60k iters, report").
- Launch order: create `bl-opd-c` first, build env, run converter (~40 min at 32 workers), then start `bl-opd-p` and `bl-opd-prgb` whose chains wait on `.done_all`. All three run `setup_env.sh` at start.

- [ ] **Step 1: Write `pods.sh`, `setup_env.sh`, `chain.sh` as specified; `bash -n` both.**
- [ ] **Step 2: Create `bl-opd-c`; scp the two scripts (mirror lag: the Mac tree reaches the volume via the dev pod; scripts also live in the repo which IS on the volume at `/workspace/SegAffordance/runpod/baselines/`, run `mutagen sync flush ethz-workspace` first); run `setup_env.sh` in the background with nohup; verify `nvidia-smi` shows an A100 and `import detectron2, MultiScaleDeformableAttention` succeeds.**
- [ ] **Step 3: Run the converter on `bl-opd-c` (`--workers $(nproc)`) detached; watch `/workspace/datasets/baselines/logs/convert_opd.log`; on finish touch `.done_all`; sanity: `MotionNet_train.json` annotations count == 54086 - len(bvalid keys), test count == 5088.**
- [ ] **Step 4: 200-iteration smoke on `bl-opd-c`** with `SOLVER.MAX_ITER 200 TEST.EVAL_PERIOD 200` into `$RUNS/smoke_c`; expect no crash, `inference/instances_predictions.pth` written, `opd_preds_to_jsonl.py` produces 5,088 lines; score 300 of them with the scorer on the dev pod (numbers meaningless, pipeline proven).
- [ ] **Step 5: Launch the three real chains** (`nohup bash chain.sh c_rgbd > ... &` etc.); create the other two pods; record pod ids in `experiments/baselines_sf3d/pods.md` (tracked, tiny) and `runpodctl pod list` reconciled.
- [ ] **Step 6: Commit** `git add runpod/baselines config/baselines/opd_sf3d_base.yaml experiments/baselines_sf3d/pods.md && git commit -m "baselines: OPDFormer pod scripts and launch record"`

---

### Task 6: MOPD — compose checkpoint, fine-tune (after `p_rgb`)

**Files:** Create `tools/baselines_sf3d/mopd_compose_ckpt.py`, `runpod/baselines/mopd/setup_env.sh`, `runpod/baselines/mopd/chain.sh`.

**Interfaces:**
- `mopd_compose_ckpt.py --opd $RUNS/p_rgb/model_final.pth --esam /workspace/datasets/baselines/ckpt/efficient_sam_vits.pt --out $RUNS/mopd/init.pth`: loads the OPDFormer state dict (`["model"]` key from detectron2's checkpointer), loads the EfficientSAM ViT-S checkpoint (zip from `https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vits.pt.zip`), remaps its `image_encoder.*` keys to `image_encoder.module.*`, adds the geffnet `tf_efficientnet_b5_ap` ImageNet weights under `normal_encoder.original_model.*` by instantiating MOPD's `Encoder(B=5, pretrained=True)` (or, simpler, patch MOPD's constructor to load with `strict=False` and let `Encoder(pretrained=True)` fetch its own weights; record which). Saves a plain state dict.
- MOPD env: same as OPD + `pip install geffnet pot matplotlib` + `uotod` (`pip install uotod`; if it fails, patch `matcher.py` to skip the Sinkhorn call since its result is discarded — record the patch). Clone `https://github.com/lisiqi-zju/MOPD` to `/workspace/datasets/baselines/repos/MOPD`, build its `ops` too.
- `chain.sh`: waits for `$RUNS/p_rgb/CHAIN_DONE`; compose ckpt; train with `configs/opd_p_real.yaml` (their BMOC_V1 config with their `opd_base.yaml` overrides: lr 5e-6, 1000 iters), `--input-format RGB`, `--opts MODEL.WEIGHTS $RUNS/mopd/init.pth MODEL.SEM_SEG_HEAD.NUM_CLASSES 8 MODEL.PIXEL_MEAN [...] MODEL.PIXEL_STD [...]`; then `evaluate_on_log.py` on `MotionNet_test`; then `opd_preds_to_jsonl.py`; `CHAIN_DONE`.

- [ ] **Step 1: Implement compose script with a unit test** that builds a fake OPD state dict + fake ESAM dict and checks key prefixes in the output.
- [ ] **Step 2: Env + 20-iteration smoke on the `bl-opd-prgb` pod** as soon as `p_rgb` has a first checkpoint (`model_0009999.pth`), using it as a stand-in init: proves the strict `load_state_dict` passes.
- [ ] **Step 3: Chain launch** on `bl-opd-prgb` right after `p_rgb` finishes (same pod, sequential).
- [ ] **Step 4: Commit** `git add tools/baselines_sf3d/mopd_compose_ckpt.py tests/baselines_sf3d/test_mopd_compose.py runpod/baselines/mopd && git commit -m "baselines: MOPD checkpoint composition and chain"`

---

### Task 7: `sf3d_to_usdnet.py` + scan download

**Files:** Create `tools/baselines_sf3d/sf3d_to_usdnet.py`, `runpod/baselines/usdnet/download_scans.sh`; Test `tests/baselines_sf3d/test_usdnet_convert.py`.

**Interfaces:**
- Download (`download_scans.sh`, runs on the USDNet pod): clone `https://github.com/SceneFun3D/scenefun3d` at `e37c166` into `$BASE/repos/scenefun3d`; write `$BASE/data/sf3d_scans/list.csv` with one `visit_id,video_id` per visit from `splits.json` + the first video of each visit (from the key list); run `python -m data_downloader.data_asset_download --split custom --download_dir $BASE/data/sf3d_scans --video_id_csv list.csv --dataset_assets laser_scan_5mm annotations motions transform`; verify 224 visits have `<visit>_laser_scan.ply`, `<visit>_annotations.json`, `<visit>_motions.json`, `<video>/<video>_transform.npy`.
- Converter function `convert_scene(visit: str, scan_dir: Path, out_dir: Path, mode: str, voxel: float = 0.02) -> dict` (the `filebase` entry) doing:
  1. read PLY (open3d), `voxel_down_sample(voxel)`, `estimate_normals()`; colors 0..255.
  2. `T_up = R_yup_to_zup @ laser_to_arkit` with `R_yup_to_zup = [[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]` (ARKit y-up -> z-up); transform points, save `T_up` to `<out>/<mode>/<visit>_T_up.json`.
  3. for each motion in `motions.json` whose annotation exists and label != `exclude`: sem = 1 (rot) / 2 (trans), inst = running int from 1, full-res `indices` -> nearest downsampled point within 2 cm (KD-tree from open3d/scipy) -> set `sem/inst/inter` columns; axis = `R_up @ motion_dir` (unit), origin = `T_up @ laser_points[motion_origin_idx]`.
  4. write `<out>/<mode>/<visit>.npy` (float32 N x 13: xyz, rgb, normals, sem, inst, 0, inter), `<out>/instance_gt/<mode>/<visit>.txt` (`sem*1000 + inst + 1`, 0 for background... exactly `sem*1000 + inst + 1` with inst 0 for background as upstream), `<out>/expand_dict/<visit>.pkl` via a copy of upstream `expand_instances_and_semantics` (import it from the cloned repo path with `sys.path`), `<out>/<mode>/<visit>_articulation.h5` via upstream `save_dict_to_h5file` with `{str(inst): {"sem_id", "axis", "origin", "inter_mask"}}`.
  5. return filebase `{"filepath": npy, "raw_filepath": ply, "scene": visit, "color_mean": [...], "color_std": [...] (upstream's E[x^2] convention), "instance_gt_filepath", "expand_dict_file", "articulation_gt_file"}`.
- CLI writes `train_database.yaml`, `validation_database.yaml`, `test_database.yaml`, `train_validation_database.yaml`, `label_database.yaml` (copy upstream's from `data/processed` layout: `{1: {"color": [..], "name": "rotation", "validation": true}, 2: {... "translation"}}` as upstream writes at lines 509-519), `color_mean_std.yaml` from train.
- Modes: `train` = `splits.train`, `validation` = `splits.bvalid`, `test` = `splits.test`.

- [ ] **Step 1: Failing test** with a synthetic 2,000-point cube scan + one rot and one trans annotation (write a tiny PLY with open3d in tmp): output npy has 13 columns, inst ids {0,1,2}, sem in {0,1,2}, articulation h5 has keys "1","2", `T_up` json parses, instance_gt values in {0, 1002, 2003} pattern (`sem*1000+inst+1`).
- [ ] **Step 2: Implement; run test (open3d needed: run on the USDNet pod's env or install open3d in the dev venv? -> run on the USDNet pod, not the dev pod).**
- [ ] **Step 3: Convert all 224 scenes on the USDNet pod (`--workers 8`), verify counts (train 182, validation 20, test 22) and that every train scene has >= 1 instance; log scenes with 0 instances.**
- [ ] **Step 4: Commit** `git add tools/baselines_sf3d/sf3d_to_usdnet.py tests/baselines_sf3d/test_usdnet_convert.py runpod/baselines/usdnet/download_scans.sh && git commit -m "baselines: SF3D scans -> USDNet processed layout"`

---

### Task 8: USDNet pod — env, train, export

**Files:** Create `runpod/baselines/usdnet/setup_env.sh` (copy of `runpod/external/usdnet/setup_env.sh` with `W=/workspace/datasets/baselines/usdnet`, no `ours/` screw-loss block, no data symlink block), `runpod/baselines/usdnet/chain.sh`.

**Interfaces / recipe (upstream `scripts/train_mov.sh` verbatim except paths, epochs and the val cadence):**

```bash
python main_instance_segmentation_articulation.py general.experiment_name=sf3d_mov general.project_name=sf3d_baselines \
 data/datasets=articulate3d_challenge_mov data.train_dataset.data_dir=$D data.validation_dataset.data_dir=$D data.test_dataset.data_dir=$D \
 data.train_dataset.label_db_filepath=$D/label_database.yaml data.validation_dataset.label_db_filepath=$D/label_database.yaml data.test_dataset.label_db_filepath=$D/label_database.yaml \
 data.train_dataset.color_mean_std=$D/color_mean_std.yaml data.validation_dataset.color_mean_std=$D/color_mean_std.yaml data.test_dataset.color_mean_std=$D/color_mean_std.yaml \
 general.num_targets=4 general.eval_on_segments=false general.train_on_segments=false general.save_dir=$RUNS/usdnet general.eval_articulation=true general.eval_hierarchy_inter=false \
 data.train_mode=train data.num_labels=3 data.batch_size=1 data.voxel_size=0.02 data.load_articulation=true data.use_hierarchy=false \
 data.cropping=true data.crop_length=5.5 data.crop_min_size=75000 data.use_coarse_to_fine=true data.c2f_rad=0.1 data.c2f_decay=0.4 data.c2f_alpha=100 \
 model.num_queries=100 model.predict_articulation_mode=2 model.predict_hierarchy_interaction=false model.predict_articulation=true \
 loss.regular_arti_loss=false "loss.losses=[labels,masks,articulations]" trainer.check_val_every_n_epoch=20 trainer.max_epochs=200 optimizer.lr=0.0001 \
 general.checkpoint=$CKPT/scannet200_benchmark.ckpt logging=minimal
```
  `trainer.max_epochs=200` is our choice (upstream leaves 10,000 in the yaml; Mask3D's ScanNet recipe is 601; 200 epochs is ~11 h on an A100 at ~3.3 min/epoch). Record this deviation in notes. The `data.*_dataset.data_dir` overrides avoid editing their yaml; if hydra rejects them, symlink `$D` to `repos/USDNet/data/processed/articulate3d_challenge_mov` instead (as the previous session did).
- Export: `general.train_mode=false general.debug=true general.checkpoint=<best val ckpt by val_mean_ap_50, else last.ckpt> data.validation_mode=test data.cropping=false` -> `$RUNS/usdnet_test/debug/val_preds/preds.pkl`.
- Then `usdnet_preds_to_jsonl.py` (Task 9), `CHAIN_DONE`.

- [ ] **Step 1: Create `bl-usdnet`, run `setup_env.sh` (MinkowskiEngine build ~15 min), then `download_scans.sh`, then the converter (Task 7 Step 3), then a 40-step smoke (`+trainer.limit_train_batches=40 +trainer.limit_val_batches=2 trainer.max_epochs=1`), then launch the chain detached.**
- [ ] **Step 2: Commit** `git add runpod/baselines/usdnet && git commit -m "baselines: USDNet env and chain on SF3D scans"`

---

### Task 9: `usdnet_preds_to_jsonl.py` — scene predictions -> per-frame JSONL

**Files:** Create `tools/baselines_sf3d/usdnet_preds_to_jsonl.py`; Test `tests/baselines_sf3d/test_usdnet_preds.py`.

**Interfaces:**
- Consumes: `preds.pkl` `{visit: {"pred_masks": (N,K) float, "pred_scores": (K,), "pred_classes": (K,) 1 rot/2 trans, "pred_origins": (K,3), "pred_axises": (K,3)}}`, `<out>/test/<visit>.npy` coords (rows align with `pred_masks`), `<out>/test/<visit>_T_up.json`, our LMDB test records (`camera_extrinsics_world_to_cam`, `camera_intrinsics`, `mask_coordinates_yx`).
- Function `project_instance(points_up: np.ndarray(M,3), T_up: np.ndarray(4,4), w2c: np.ndarray(4,4), K: np.ndarray(3,3), hw=(1440,1920), radius_px=4) -> np.ndarray[bool]`: `p_laser = inv(T_up) p`, `p_cam = w2c p_laser`, keep `z > 0.05`, project, splat discs of `radius_px` (cv2.circle) -> mask.
- Per test key: candidates = instances with `score >= 0.05` and `pred_masks[:, k] > 0.5` having >= 20 points; IoU vs GT mask (native res); pick max IoU; unmatched if max IoU == 0. `type = 1 if class == 1 else 0`; `axis_cam = R_w2c @ R_up^-1 @ axis`, `origin_cam = w2c @ inv(T_up) @ origin`. Score = pred score.
- CLI: `--preds preds.pkl --usd-dir <out> --out preds.jsonl`.

- [ ] **Step 1: Failing test** with a synthetic cube instance projected with identity extrinsics and a simple K -> mask centred where expected; axis rotation through T_up round-trips.
- [ ] **Step 2: Implement; test; commit** `git add tools/baselines_sf3d/usdnet_preds_to_jsonl.py tests/baselines_sf3d/test_usdnet_preds.py && git commit -m "baselines: USDNet scene predictions -> per-frame JSONL"`

---

### Task 10: Monitoring, scoring, wrap-up

- [ ] **Step 1: Monitors.** One persistent `Monitor` per pod polling every 10 min over ssh: prints the last progress line (`iter:` / `Epoch`), any `Traceback|Error|OOM|Killed`, and `CHAIN_DONE`. Plus a Mac-side watcher per pod (`nohup ... & disown`) that deletes the pod when `CHAIN_DONE` exists and copies `preds.jsonl`, `train.log`, `test.log`, `config` to `/workspace/datasets/baselines/results/<variant>/` (volume) first.
- [ ] **Step 2: Score each `preds.jsonl` on the dev pod** with `score_predictions.py` (<10 min each) -> `experiments/baselines_sf3d/<id>/metrics.json`.
- [ ] **Step 3: Per-baseline `notes.md`** (goal, upstream commit, exact command, deviations, their own evaluator's numbers from `test_*.log` for reference, our metrics table, cost), `config.yaml` (the resolved config dump from their output dir), `train.log.tail` (last 50 lines).
- [ ] **Step 4: `experiments/INDEX.md`**: append section `## Baselines (external models retrained on SF3D)` with one row per baseline: id | method | input | recipe | PDet | mIoU | type% | MA | axis all/matched | origin | notes. Reference row: our best `20260912_joint4_decoder_cfframe_seed7` numbers copied from the STATE table.
- [ ] **Step 5: `STATE.md`** bottom section `## External baselines on SF3D (session ethz-workspace-34)`: what ran, pod ids, spend, results table, deviations, what is left. Message ethz-workspace-c6 before committing.
- [ ] **Step 6: Reconcile `runpodctl pod list`: only `segaffordance-dev` may remain. Commit and push.**

---

## Self-review

- Spec coverage: OPDFormer-C RGB-D (Task 5), OPDFormer-P RGB-D (Task 5), OPDFormer-P RGB (Task 5, MOPD init), MOPD (Task 6), USDNet (Tasks 7-9), faithful recipes (their train scripts; deviations = 8 classes, per-dataset pixel stats, our splits, USDNet 200 epochs), parallel pods (Task 5 Step 5, Task 8), our metrics (Task 2), clone-and-fit (all).
- Placeholders: none; the `colmajor16(K_s)[:0] or` no-op in Task 3 is called out to be removed.
- Type consistency: JSONL schema used identically in Tasks 2, 4, 9; `common` function names consistent; `splits` keys `test/train/bvalid` everywhere; OPD split names `train/valid/test`; USDNet modes `train/validation/test`.
