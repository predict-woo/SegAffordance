# HOI4D Articulation Annotator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A viser-based web tool to annotate per-sequence articulation parameters (prismatic/revolute axis + origin, world frame) on the HOI4D processed-2D dataset, with SF3D-convention camera-frame export.

**Architecture:** One tool module `tools/hoi4d_annotate_articulation.py` with three layers — pure-numpy geometry/IO core (unit-tested locally, no viser/LMDB needed), LMDB scene builder (tested against tiny synthetic LMDBs), and a viser UI + CLI that only runs on the dev pod (manual smoke test). Annotations are per-sequence JSONs; export projects them into every sample's camera frame.

**Tech Stack:** numpy, opencv-python-headless (test/scene encode-decode), lmdb, viser (runtime only, import guarded), pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-hoi4d-articulation-annotator-design.md`

## Global Constraints

- Never write to `data.lmdb` / `frames.lmdb` — the tool is read-only on the dataset.
- Annotation saves are atomic: write `<file>.tmp`, then `os.replace`.
- Depth is uint16 **millimeters**; convert to meters at back-projection; drop zero-depth pixels.
- Gizmo **+z axis = articulation axis**. Revolute: gizmo position = origin. Prismatic: origin display-only.
- Sign convention: trans motion along +axis for "open"-family events (`open`, `pull`, `pullout`); rot right-hand-positive opening.
- `viser` is imported **only** inside `serve()`/`main()` so the test suite never needs it.
- Repo convention: run scripts with `bash`/`python3` explicitly; commits Mac-side only; commit messages end with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.
- Test venv (Mac, lives outside the mirror):
  `python3 -m venv /private/tmp/claude-501/-Users-andyye-dev-ethz-workspace/beb49655-be7d-4a7b-9f25-2063c1e9071b/scratchpad/annenv && annenv/bin/pip install numpy opencv-python-headless lmdb pytest`
  Run all tests below with `<venv>/bin/python -m pytest`.
- Record key format (from `tools/hoi4d_process_2d.py`): `"<cam>_<H>_<C>_<N>/<S>_<s>_<T>_w<i>_f<frame>"`; sequence id = the 7 fields joined with `_` (matches `hands354/` dir names, e.g. `ZY20210800001_H1_C4_N01_S19_s05_T1`).

---

### Task 1: Geometry core

**Files:**
- Create: `tools/hoi4d_annotate_articulation.py` (geometry section)
- Test: `tests/test_hoi4d_annotator_geometry.py`

**Interfaces:**
- Produces (later tasks call these exact signatures):
  - `parse_key(key: str) -> tuple[str, int, int]` — returns `(seq_id, window, frame)`
  - `wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray` — (4,) quat → (3,3) rotation
  - `matrix_to_wxyz(R: np.ndarray) -> np.ndarray` — (3,3) rotation → (4,) unit quat (inverse of the above up to sign)
  - `backproject(depth_u16: np.ndarray, rgb_bgr: np.ndarray, K: np.ndarray, stride: int = 4) -> tuple[np.ndarray, np.ndarray]` — → (N,3) camera-frame meters, (N,3) float RGB in [0,1]; K is at the depth map's resolution
  - `to_world(points_cam: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray`
  - `world_axis_to_camera(axis_w, origin_w, pose_c2w, K) -> dict` with keys `motion_dir_3d_camera_coords`, `motion_origin_3d_camera_coords`, `motion_origin_2d_image_coords` (pixels at K's resolution)
  - `sweep_points(points: np.ndarray, mtype: str, axis: np.ndarray, origin: np.ndarray, t: float) -> np.ndarray` — trans: `points + t*axis` (t meters); rot: rotate by `t` radians about (axis, origin), right-handed
  - `principal_direction(points: np.ndarray) -> np.ndarray` — unit first principal component, sign flipped so it points from first to last point

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hoi4d_annotator_geometry.py
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
    # 90deg about x sends +z to -y... check via known quat (w=cos45, x=sin45)
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
    # random rigid pose
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
    # origin shift matters
    out = ann.sweep_points(pts, "rot", np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.pi)
    np.testing.assert_allclose(out, [[1.0, 0.0, 0.0]], atol=1e-12)


def test_principal_direction_sign():
    t = np.linspace(0, 1, 20)[:, None]
    traj = t * np.array([[0.0, -2.0, 0.0]]) + np.random.default_rng(1).normal(scale=1e-3, size=(20, 3))
    d = ann.principal_direction(traj)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9
    assert d[1] < -0.99   # points from first toward last sample
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `<venv>/bin/python -m pytest tests/test_hoi4d_annotator_geometry.py -v` (from `SegAffordance/`)
Expected: FAIL at module load (file doesn't exist) or attribute errors.

- [ ] **Step 3: Implement the geometry section**

```python
"""Web-based articulation annotator for the HOI4D processed-2D dataset.

Spec: docs/superpowers/specs/2026-09-01-hoi4d-articulation-annotator-design.md
Serve (dev pod):  python3 tools/hoi4d_annotate_articulation.py serve \
    --data /workspace/hoi4d_processed_2d --hands /workspace/datasets/hands354 \
    --out /workspace/hoi4d_processed_2d/annotations [--port 8080]
Export:           python3 tools/hoi4d_annotate_articulation.py export \
    --data ... --hands ... --out ...   (writes annotations/export_sf3d.pkl)

viser is imported only inside serve(); the geometry/IO core is
importable and unit-tested without it.
"""
import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np

OPEN_EVENTS = {"open", "pull", "pullout"}
_KEY_RE = re.compile(r"^(.+)/(.+)_w(\d+)_f(\d+)$")


def parse_key(key: str):
    m = _KEY_RE.match(key)
    pre, mid, w, f = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    return f"{pre}_{mid}", w, f


def wxyz_to_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, float) / np.linalg.norm(wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_wxyz(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-8:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                         (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(R)))          # w ~ 0: pick dominant axis
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4.0
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def backproject(depth_u16, rgb_bgr, K, stride: int = 4):
    d = depth_u16[::stride, ::stride].astype(np.float64) / 1000.0
    ys, xs = np.nonzero(d > 0)
    z = d[ys, xs]
    xs_full, ys_full = xs * stride, ys * stride
    x = (xs_full - K[0, 2]) / K[0, 0] * z
    y = (ys_full - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=1)
    cols = rgb_bgr[ys_full, xs_full, ::-1].astype(np.float64) / 255.0
    return pts, cols


def to_world(points_cam, pose_c2w):
    return points_cam @ pose_c2w[:3, :3].T + pose_c2w[:3, 3]


def world_axis_to_camera(axis_w, origin_w, pose_c2w, K):
    Rcw = pose_c2w[:3, :3].T
    dir_c = Rcw @ (np.asarray(axis_w, float) / np.linalg.norm(axis_w))
    org_c = Rcw @ (np.asarray(origin_w, float) - pose_c2w[:3, 3])
    uvw = K @ org_c
    return {
        "motion_dir_3d_camera_coords": dir_c,
        "motion_origin_3d_camera_coords": org_c,
        "motion_origin_2d_image_coords": np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]]),
    }


def sweep_points(points, mtype, axis, origin, t):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    if mtype == "trans":
        return points + t * axis
    c, s = np.cos(t), np.sin(t)
    Kx = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    R = np.eye(3) * c + s * Kx + (1 - c) * np.outer(axis, axis)
    return (points - origin) @ R.T + origin


def principal_direction(points):
    p = np.asarray(points, float)
    centered = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])
    if d @ (p[-1] - p[0]) < 0:
        d = -d
    return d
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `<venv>/bin/python -m pytest tests/test_hoi4d_annotator_geometry.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Run the full repo suite to check nothing broke**

Run: `<venv>/bin/python -m pytest tests/test_hoi4d_annotator_geometry.py -q` (full suite needs torch; run at least the new file plus `python3 -m pyflakes tools/hoi4d_annotate_articulation.py` if pyflakes is available — otherwise `python3 -c "import ast; ast.parse(open('tools/hoi4d_annotate_articulation.py').read())"`).

- [ ] **Step 6: Commit**

```bash
git add tools/hoi4d_annotate_articulation.py tests/test_hoi4d_annotator_geometry.py
git commit -m "feat(annotator): geometry core for HOI4D articulation annotator"
```

---

### Task 2: Annotation store

**Files:**
- Modify: `tools/hoi4d_annotate_articulation.py` (append store section)
- Test: `tests/test_hoi4d_annotator_store.py`

**Interfaces:**
- Produces:
  - `class AnnotationStore(out_dir: Path)` with:
    - `.load(seq: str) -> dict | None`
    - `.save(seq: str, category: str, parts: list[dict]) -> Path` — atomic; stamps `annotator` (env `USER` or "unknown") and ISO `time`
    - `.status() -> dict[str, str]` — `{seq: "annotated" | "flagged"}` for every JSON present (flagged when **all** parts carry a non-null `flag`)
  - Part dict shape (spec): `{"window_events": [str], "type": "trans"|"rot", "axis_world": [3], "origin_world": [3], "flag": None|str}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hoi4d_annotator_store.py
import importlib.util, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ann", ROOT / "tools" / "hoi4d_annotate_articulation.py")
ann = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ann)

PART = {"window_events": ["open", "close"], "type": "rot",
        "axis_world": [0.0, 0.0, 1.0], "origin_world": [1.0, 2.0, 3.0], "flag": None}


def test_save_load_roundtrip(tmp_path):
    store = ann.AnnotationStore(tmp_path)
    p = store.save("SEQX", "C6", [PART])
    assert p.exists() and not list(tmp_path.glob("*.tmp"))
    got = store.load("SEQX")
    assert got["seq"] == "SEQX" and got["category"] == "C6"
    assert got["parts"] == [PART]
    assert "time" in got and "annotator" in got


def test_missing_returns_none(tmp_path):
    assert ann.AnnotationStore(tmp_path).load("NOPE") is None


def test_status(tmp_path):
    store = ann.AnnotationStore(tmp_path)
    store.save("A", "C4", [PART])
    bad = dict(PART); bad["flag"] = "bad-poses"
    store.save("B", "C4", [bad])
    assert store.status() == {"A": "annotated", "B": "flagged"}


def test_save_is_atomic_json(tmp_path):
    store = ann.AnnotationStore(tmp_path)
    path = store.save("A", "C4", [PART])
    json.load(open(path))   # valid JSON on disk
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `<venv>/bin/python -m pytest tests/test_hoi4d_annotator_store.py -v`
Expected: FAIL — `AnnotationStore` not defined.

- [ ] **Step 3: Implement the store**

```python
class AnnotationStore:
    """Per-sequence articulation annotations (atomic JSON files)."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, seq):
        return self.dir / f"{seq}.json"

    def load(self, seq):
        p = self._path(seq)
        return json.load(open(p)) if p.exists() else None

    def save(self, seq, category, parts):
        rec = {"seq": seq, "category": category,
               "annotator": os.environ.get("USER", "unknown"),
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "parts": parts}
        p = self._path(seq)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, p)
        return p

    def status(self):
        out = {}
        for p in sorted(self.dir.glob("*.json")):
            rec = json.load(open(p))
            flagged = rec["parts"] and all(pt.get("flag") for pt in rec["parts"])
            out[rec["seq"]] = "flagged" if flagged else "annotated"
        return out
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `<venv>/bin/python -m pytest tests/test_hoi4d_annotator_store.py tests/test_hoi4d_annotator_geometry.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/hoi4d_annotate_articulation.py tests/test_hoi4d_annotator_store.py
git commit -m "feat(annotator): atomic per-sequence annotation store"
```

---

### Task 3: Dataset access + scene builder

**Files:**
- Modify: `tools/hoi4d_annotate_articulation.py` (append dataset section)
- Test: `tests/test_hoi4d_annotator_scene.py`

**Interfaces:**
- Consumes: `parse_key`, `backproject`, `to_world` (Task 1).
- Produces:
  - `class Dataset(data_dir: Path, hands_dir: Path)` with:
    - `.sequences: dict[str, list[str]]` — seq id → sorted sample keys (built by iterating `data.lmdb` keys once)
    - `.record(key) -> dict` (unpickled data.lmdb value)
    - `.frame(key) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — (rgb_bgr at stored size, depth_u16 same size, K_scaled) where K from the record is rescaled from `orig_size` to the stored jpeg/depth resolution
    - `.pose(seq, frame) -> np.ndarray` — (4,4) from `hands_dir/<seq>/camera/official_poses.npy`
  - `build_scene(ds: Dataset, seq: str, cache_dir: Path, max_frames: int = 15, target_points: int = 300_000) -> dict` — keys: `points (N,3) world`, `colors (N,3)`, `windows: {w: {"event": str, "mask_points": (M,3) world, "traj": (T,3) world, "sample_key": str}}`, `frusta: [(4,4) pose]`; cached to `cache_dir/<seq>.npz` (windows stored via pickle inside the npz `meta` array)

- [ ] **Step 1: Write the failing test with a synthetic LMDB fixture**

```python
# tests/test_hoi4d_annotator_scene.py
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
    rgb, depth, K = ds.frame(KEYS[0])
    assert rgb.shape == (SIZE, SIZE, 3) and depth.dtype == np.uint16
    np.testing.assert_allclose(K[0, 0], 80.0)     # orig_size == stored size here
    pose = ds.pose(SEQ, 10)
    np.testing.assert_allclose(pose[0, 3], 0.10)


def test_build_scene_world_frame_and_cache(data_root, tmp_path):
    ds = ann.Dataset(data_root, data_root / "hands")
    cache = tmp_path / "cache"
    scene = ann.build_scene(ds, SEQ, cache)
    # depth 1.5m at principal point, pose f10 shifts x by 0.10
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `<venv>/bin/python -m pytest tests/test_hoi4d_annotator_scene.py -v`
Expected: FAIL — `Dataset` not defined.

- [ ] **Step 3: Implement dataset + scene builder**

```python
class Dataset:
    """Read-only view over the processed-2D LMDBs + hands package."""

    def __init__(self, data_dir, hands_dir):
        import lmdb
        self.data_dir, self.hands_dir = Path(data_dir), Path(hands_dir)
        self._env_d = lmdb.open(str(self.data_dir / "data.lmdb"),
                                readonly=True, lock=False)
        self._env_f = lmdb.open(str(self.data_dir / "frames.lmdb"),
                                readonly=True, lock=False)
        self.sequences = {}
        with self._env_d.begin() as txn:
            for key, _ in txn.cursor():
                k = key.decode()
                if k.startswith("__"):
                    continue
                seq, _, _ = parse_key(k)
                self.sequences.setdefault(seq, []).append(k)
        for v in self.sequences.values():
            v.sort()
        self._poses = {}

    def record(self, key):
        with self._env_d.begin() as txn:
            return pickle.loads(txn.get(key.encode()))

    def frame(self, key):
        import cv2
        with self._env_f.begin() as txn:
            fr = pickle.loads(txn.get(key.encode()))
        rgb = cv2.imdecode(np.frombuffer(fr["jpeg"], np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(fr["depth_png"], np.uint8),
                             cv2.IMREAD_UNCHANGED)
        K = np.array(self.record(key)["camera_intrinsics"], float)
        ow, oh = fr["orig_size"]
        K = K.copy()
        K[0] *= rgb.shape[1] / ow
        K[1] *= rgb.shape[0] / oh
        return rgb, depth, K

    def pose(self, seq, frame):
        if seq not in self._poses:
            self._poses[seq] = np.load(
                self.hands_dir / seq / "camera" / "official_poses.npy")
        return self._poses[seq][frame]


def build_scene(ds, seq, cache_dir, max_frames=15, target_points=300_000):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"{seq}.npz"
    if cpath.exists():
        z = np.load(cpath, allow_pickle=True)
        return {"points": z["points"], "colors": z["colors"],
                "windows": pickle.loads(z["meta"].tobytes())["windows"],
                "frusta": [p for p in z["frusta"]]}

    keys = ds.sequences[seq]
    step = max(1, len(keys) // max_frames)
    chosen = keys[::step][:max_frames]
    pts_all, col_all, frusta = [], [], []
    for key in chosen:
        rgb, depth, K = ds.frame(key)
        _, _, f = parse_key(key)
        pose = ds.pose(seq, f)
        pts, cols = backproject(depth, rgb, K, stride=4)
        pts_all.append(to_world(pts, pose)); col_all.append(cols)
        frusta.append(pose)
    points = np.concatenate(pts_all); colors = np.concatenate(col_all)
    if len(points) > target_points:
        idx = np.random.default_rng(0).choice(len(points), target_points, replace=False)
        points, colors = points[idx], colors[idx]

    windows = {}
    for key in keys:            # one representative sample per window
        rec = ds.record(key)
        w = rec["hoi4d"]["window"]
        if w in windows:
            continue
        rgb, depth, K = ds.frame(key)
        _, _, f = parse_key(key)
        pose = ds.pose(seq, f)
        ow, oh = pickle.loads(  # mask coords are orig-res; rescale to stored res
            ds._env_f.begin().get(key.encode()))["orig_size"], None
        ow, ohh = ow[0], ow[1]
        sy, sx = depth.shape[0] / ohh, depth.shape[1] / ow
        mask_pts = []
        for y, x in rec["mask_coordinates_yx"]:
            yy, xx = int(y * sy), int(x * sx)
            z = depth[yy, xx] / 1000.0
            if z > 0:
                mask_pts.append([(xx - K[0, 2]) / K[0, 0] * z,
                                 (yy - K[1, 2]) / K[1, 1] * z, z])
        mask_w = (to_world(np.array(mask_pts), pose)
                  if mask_pts else np.zeros((0, 3)))
        traj_w = to_world(np.array(rec["trajectory_3d_camera_coords"], float), pose)
        windows[w] = {"event": rec["hoi4d"]["event"], "mask_points": mask_w,
                      "traj": traj_w, "sample_key": key}

    meta = np.frombuffer(pickle.dumps({"windows": windows}), np.uint8)
    np.savez_compressed(cpath, points=points, colors=colors,
                        frusta=np.array(frusta), meta=meta)
    return {"points": points, "colors": colors, "windows": windows,
            "frusta": frusta}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `<venv>/bin/python -m pytest tests/test_hoi4d_annotator_scene.py -v`
Expected: 2 PASS. If the `orig_size` unpacking above proves awkward, refactor `Dataset.frame` to also return `orig_size` instead of re-reading — tests must still pass unchanged except signature-dependent lines; keep the Interfaces block authoritative by updating both together.

- [ ] **Step 5: Commit**

```bash
git add tools/hoi4d_annotate_articulation.py tests/test_hoi4d_annotator_scene.py
git commit -m "feat(annotator): LMDB dataset access + cached world-frame scene builder"
```

---

### Task 4: SF3D export

**Files:**
- Modify: `tools/hoi4d_annotate_articulation.py` (append export section)
- Test: `tests/test_hoi4d_annotator_export.py`

**Interfaces:**
- Consumes: `Dataset`, `AnnotationStore`, `world_axis_to_camera`, `parse_key`.
- Produces:
  - `export_sf3d(ds: Dataset, store: AnnotationStore, out_path: Path) -> int` — for every sample key of every **non-flagged** annotated sequence, writes `{key: {"motion_type": str, "motion_dir_3d_camera_coords": [3], "motion_origin_3d_camera_coords": [3], "motion_origin_2d_image_coords": [2]}}` as a pickle; returns number of keys exported. Uses the **first part whose `window_events` contains the sample's event** (fallback: first part). 2D coords are pixels at the record's `orig_size` resolution (use the record's stored, unscaled K).

- [ ] **Step 1: Write the failing test** (reuses the Task 3 fixture — import it)

```python
# tests/test_hoi4d_annotator_export.py
import importlib.util, pathlib, pickle
import numpy as np
from test_hoi4d_annotator_scene import data_root, SEQ, KEYS   # fixture reuse

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ann", ROOT / "tools" / "hoi4d_annotate_articulation.py")
ann = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ann)


def test_export_projects_world_to_each_camera(data_root, tmp_path):
    ds = ann.Dataset(data_root, data_root / "hands")
    store = ann.AnnotationStore(tmp_path / "annos")
    store.save(SEQ, "C4", [{"window_events": ["open"], "type": "trans",
                            "axis_world": [0.0, 0.0, 1.0],
                            "origin_world": [0.5, 0.0, 1.0], "flag": None}])
    out = tmp_path / "export_sf3d.pkl"
    n = ann.export_sf3d(ds, store, out)
    assert n == len(KEYS)
    data = pickle.loads(out.read_bytes())
    for key in KEYS:
        e = data[key]
        assert e["motion_type"] == "trans"
        # pose of frame f is x-shifted by 0.01*f; axis z is rotation-invariant here
        np.testing.assert_allclose(e["motion_dir_3d_camera_coords"], [0, 0, 1], atol=1e-12)
        f = int(key[-3:])
        np.testing.assert_allclose(
            e["motion_origin_3d_camera_coords"], [0.5 - 0.01 * f, 0.0, 1.0], atol=1e-9)


def test_flagged_sequences_skipped(data_root, tmp_path):
    ds = ann.Dataset(data_root, data_root / "hands")
    store = ann.AnnotationStore(tmp_path / "annos")
    store.save(SEQ, "C4", [{"window_events": ["open"], "type": "rot",
                            "axis_world": [0, 0, 1.0], "origin_world": [0, 0, 0],
                            "flag": "bad-poses"}])
    n = ann.export_sf3d(ds, store, tmp_path / "e.pkl")
    assert n == 0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `<venv>/bin/python -m pytest tests/test_hoi4d_annotator_export.py -v`
Expected: FAIL — `export_sf3d` not defined.

- [ ] **Step 3: Implement export**

```python
def export_sf3d(ds, store, out_path):
    out = {}
    status = store.status()
    for seq, keys in ds.sequences.items():
        if status.get(seq) != "annotated":
            continue
        rec0 = store.load(seq)
        for key in keys:
            r = ds.record(key)
            event = r["hoi4d"]["event"]
            part = next((p for p in rec0["parts"]
                         if event in p["window_events"]), rec0["parts"][0])
            _, _, f = parse_key(key)
            pose = ds.pose(seq, f)
            K = np.array(r["camera_intrinsics"], float)   # orig-res K
            cam = world_axis_to_camera(np.array(part["axis_world"], float),
                                       np.array(part["origin_world"], float),
                                       pose, K)
            out[key] = {"motion_type": part["type"],
                        "motion_dir_3d_camera_coords": cam["motion_dir_3d_camera_coords"].tolist(),
                        "motion_origin_3d_camera_coords": cam["motion_origin_3d_camera_coords"].tolist(),
                        "motion_origin_2d_image_coords": cam["motion_origin_2d_image_coords"].tolist()}
    out_path = Path(out_path)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    os.replace(tmp, out_path)
    return len(out)
```

- [ ] **Step 4: Run all annotator tests, verify they pass**

Run: `<venv>/bin/python -m pytest tests/test_hoi4d_annotator_*.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/hoi4d_annotate_articulation.py tests/test_hoi4d_annotator_export.py
git commit -m "feat(annotator): SF3D-convention camera-frame export"
```

---

### Task 5: viser UI + CLI

**Files:**
- Modify: `tools/hoi4d_annotate_articulation.py` (append `serve()` + `main()`)
- Test: manual smoke on the dev pod (no automated test — UI layer only; all logic it calls is covered by Tasks 1–4)

**Interfaces:**
- Consumes: everything above.
- Produces: `python3 tools/hoi4d_annotate_articulation.py serve|export --data <dir> --hands <dir> --out <dir> [--port 8080]`.

- [ ] **Step 1: Implement `serve()` and `main()`**

```python
def serve(ds, store, cache_dir, port):
    import viser
    server = viser.ViserServer(port=port)
    state = {"seq": None, "scene": None, "window": None, "parts": []}

    status = store.status()
    seq_names = sorted(ds.sequences)

    def label(s):
        mark = {"annotated": "[done] ", "flagged": "[flag] "}.get(status.get(s), "")
        return mark + s

    seq_dd = server.gui.add_dropdown("Sequence", [label(s) for s in seq_names],
                                     initial_value=label(seq_names[0]))
    win_dd = server.gui.add_dropdown("Window", ["0"], initial_value="0")
    type_dd = server.gui.add_dropdown("Type", ["trans", "rot"], initial_value="trans")
    align_btn = server.gui.add_button("Align axis to trajectory")
    preview = server.gui.add_slider("Motion preview", min=-1.0, max=1.0,
                                    step=0.02, initial_value=0.0)
    save_btn = server.gui.add_button("Save")
    flag_dd = server.gui.add_dropdown("Flag", ["none", "bad-poses", "ambiguous"],
                                      initial_value="none")
    info = server.gui.add_markdown("")

    tc = server.scene.add_transform_controls("/gizmo", scale=0.25)

    def axis_vec():
        return wxyz_to_matrix(np.array(tc.wxyz)) @ np.array([0.0, 0.0, 1.0])

    def draw_axis():
        a, o = axis_vec(), np.array(tc.position)
        seg = np.stack([o - 2.0 * a, o + 2.0 * a])[None]
        server.scene.add_line_segments("/axis_line", seg,
                                       colors=np.array([[[255, 60, 60]] * 2]))

    def load_seq(_=None):
        seq = seq_names[[label(s) for s in seq_names].index(seq_dd.value)]
        state["seq"], state["window"] = seq, None
        state["scene"] = build_scene(ds, seq, cache_dir)
        sc = state["scene"]
        server.scene.add_point_cloud("/cloud", sc["points"],
                                     colors=sc["colors"], point_size=0.004)
        for i, pose in enumerate(sc["frusta"]):   # small orientation frusta
            server.scene.add_camera_frustum(
                f"/frusta/{i}", fov=1.0, aspect=1.0, scale=0.06,
                wxyz=matrix_to_wxyz(pose[:3, :3]), position=tuple(pose[:3, 3]))
        wins = sorted(sc["windows"])
        win_dd.options = [str(w) for w in wins]
        win_dd.value = str(wins[0])
        cat = seq.split("_")[2]
        type_dd.value = "trans" if cat == "C4" else "rot"
        existing = store.load(seq)
        if existing and existing["parts"]:
            p = existing["parts"][0]
            tc.position = tuple(p["origin_world"])
            type_dd.value = p["type"]
        else:  # start the gizmo at the first window's mask centroid
            m = sc["windows"][wins[0]]["mask_points"]
            if len(m):
                tc.position = tuple(m.mean(axis=0))
        load_window()

    def load_window(_=None):
        sc, w = state["scene"], int(win_dd.value)
        state["window"] = w
        wd = sc["windows"][w]
        server.scene.add_point_cloud("/mask", wd["mask_points"],
                                     colors=np.tile([[1.0, 0.15, 0.15]],
                                                    (len(wd["mask_points"]), 1)),
                                     point_size=0.007)
        if len(wd["traj"]) >= 2:
            segs = np.stack([wd["traj"][:-1], wd["traj"][1:]], axis=1)
            server.scene.add_line_segments("/traj", segs,
                colors=np.tile([[40, 120, 255]], (len(segs), 2, 1)))
        info.content = f"**{state['seq']}** — window {w}: *{wd['event']}*"
        draw_axis()

    def do_align(_):
        wd = state["scene"]["windows"][state["window"]]
        if len(wd["traj"]) >= 2:
            d = principal_direction(wd["traj"])
            # build quat rotating +z onto d
            z = np.array([0.0, 0.0, 1.0]); v = np.cross(z, d)
            w = 1.0 + float(z @ d)
            q = np.array([w, *v]); q /= np.linalg.norm(q)
            tc.wxyz = tuple(q)
            draw_axis()

    def do_preview(_):
        wd = state["scene"]["windows"][state["window"]]
        t = preview.value * (0.4 if type_dd.value == "trans" else np.pi / 2)
        moved = sweep_points(wd["mask_points"], type_dd.value,
                             axis_vec(), np.array(tc.position), t)
        server.scene.add_point_cloud("/mask", moved,
                                     colors=np.tile([[1.0, 0.15, 0.15]],
                                                    (len(moved), 1)),
                                     point_size=0.007)

    def do_save(_):
        wd_events = sorted({w["event"] for w in state["scene"]["windows"].values()})
        part = {"window_events": wd_events, "type": type_dd.value,
                "axis_world": axis_vec().tolist(),
                "origin_world": list(map(float, tc.position)),
                "flag": None if flag_dd.value == "none" else flag_dd.value}
        store.save(state["seq"], state["seq"].split("_")[2], [part])
        status.update(store.status())
        info.content = f"saved {state['seq']}"

    seq_dd.on_update(load_seq)
    win_dd.on_update(load_window)
    align_btn.on_click(do_align)
    preview.on_update(do_preview)
    save_btn.on_click(do_save)
    tc.on_update(lambda _: draw_axis())

    load_seq()
    print(f"annotator up: forward the port and open http://localhost:{port}")
    while True:
        time.sleep(3600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["serve", "export"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--hands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    ds = Dataset(args.data, args.hands)
    store = AnnotationStore(args.out)
    if args.mode == "export":
        n = export_sf3d(ds, store, Path(args.out) / "export_sf3d.pkl")
        print(f"exported {n} keys")
    else:
        serve(ds, store, Path(args.out) / "cache", args.port)


if __name__ == "__main__":
    main()
```

Note for the implementer: viser API names above (`add_transform_controls`, `add_point_cloud`, `add_line_segments`, `gui.add_dropdown/button/slider/markdown`, `.on_update/.on_click`, `.wxyz/.position`) are current as of viser 0.2.x — verify against the installed version's docs (`https://viser.studio`) and adapt call signatures if they drifted; keep the behavior contract of the spec (gizmo z = axis, preview sweeps mask points, save writes through `AnnotationStore`).

- [ ] **Step 2: Syntax + tests still green**

Run: `python3 -c "import ast; ast.parse(open('tools/hoi4d_annotate_articulation.py').read())" && <venv>/bin/python -m pytest tests/test_hoi4d_annotator_*.py -q`
Expected: parse OK, all tests PASS (viser never imported by tests).

- [ ] **Step 3: Manual smoke on the dev pod** (requires the pod up and the processed LMDBs present — if `hoi4d_processed_2d` doesn't exist yet, smoke against `--limit`-built LMDBs from `hoi4d_process_2d.py` or defer this step and note it in the commit message)

```bash
bash runpod/dev.sh run "pip install viser numpy opencv-python-headless lmdb"
bash runpod/dev.sh run "cd /workspace/SegAffordance && nohup python3 tools/hoi4d_annotate_articulation.py serve --data /workspace/hoi4d_processed_2d --hands /workspace/datasets/hands354 --out /workspace/hoi4d_processed_2d/annotations > /tmp/annotator.log 2>&1 & sleep 5; tail /tmp/annotator.log"
ssh -L 8080:localhost:8080 segaff-dev   # then open http://localhost:8080
```

Checklist: cloud renders and is navigable; red mask points sit on the drawer/door; gizmo moves; align-to-trajectory points the axis along the wrist track; preview slider sweeps the mask plausibly (open direction positive); Save writes `annotations/<seq>.json`; reopening the sequence restores the gizmo.

- [ ] **Step 4: Commit**

```bash
git add tools/hoi4d_annotate_articulation.py
git commit -m "feat(annotator): viser UI + CLI (serve/export)"
```

---

### Task 6: Wrap-up

**Files:**
- Modify: `STATE.md` (new dated section + open-threads entry)

- [ ] **Step 1: STATE.md entry** — new section "HOI4D articulation annotator (2026-09-01)": tool path, run/export commands, annotation dir, what's tested vs manual-only, smoke status (done or deferred pending the processed LMDBs).
- [ ] **Step 2: Commit + push**

```bash
git add STATE.md && git commit -m "STATE: HOI4D articulation annotator landed" && git push
```
