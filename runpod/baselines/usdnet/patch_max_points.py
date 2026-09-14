"""Patch USDNet's datasets/semseg.py with a TRAINING-ONLY cap on points per sample (env
SF3D_MAX_POINTS, off when unset/0): after their RandomCuboid crop, samples with more points are
uniformly subsampled to the cap, all per-point arrays (incl. articulation inter_ids) kept aligned.

Why (2026-09-14): USDNet's articulation head (models/mask3d.py, predict_articulation_mode 2)
materialises (points x queries x 3) tensors per decoder layer, so memory is linear in the points
of a crop. At 1 cm voxels SF3D crops reach 0.7 M points even with crop_length 3.0 (2.2 M at the
recipe's 5.5 m) and OOMed an 80 GB A100 three times; no single GPU takes the recipe's crops at
1 cm, so a cap is unavoidable. Validation / test are untouched (whole scenes). Idempotent.

  python patch_max_points.py <path to datasets/semseg.py>
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
if "SF3D_MAX_POINTS" in s:
    print("already patched")
    sys.exit(0)

anchor = """                if self.dataset_name == 'articulate3d':
                    inter_ids = inter_ids[new_idx]
                # inter_ids = inter_ids[new_idx]
"""
assert s.count(anchor) == 1, "cropping block not found"
insert = anchor + """
            # SF3D_MAX_POINTS (runpod/baselines/usdnet/patch_max_points.py): training-only cap on the
            # points of a sample; the articulation head is O(points x queries) per decoder layer.
            _mp = int(__import__("os").environ.get("SF3D_MAX_POINTS", "0"))
            if _mp > 0 and len(coordinates) > _mp:
                _keep = np.sort(np.random.choice(len(coordinates), _mp, replace=False))
                coordinates = coordinates[_keep]
                color = color[_keep]
                labels = labels[_keep]
                segments = segments[_keep]
                raw_color = raw_color[_keep]
                raw_normals = raw_normals[_keep]
                normals = normals[_keep]
                points = points[_keep]
                if self.dataset_name == 'articulate3d':
                    inter_ids = inter_ids[_keep]
"""
s = s.replace(anchor, insert)
p.write_text(s)
print("patched", p)
