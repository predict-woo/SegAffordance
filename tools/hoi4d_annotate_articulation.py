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
    """LMDB key -> (sequence id, window index, 0-based frame)."""
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
    """Depth (uint16 mm) + BGR image + K -> (N,3) camera-frame metres, (N,3) RGB [0,1]."""
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
    """World axis/origin -> SF3D-convention camera-frame dict (2D in pixels at K's res)."""
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
    """Move points along the screw: trans = t metres along axis; rot = t rad about (axis, origin)."""
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
    """Unit first principal component, oriented from first toward last point."""
    p = np.asarray(points, float)
    centered = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])
    if d @ (p[-1] - p[0]) < 0:
        d = -d
    return d


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
