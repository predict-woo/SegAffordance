"""SceneFun3D laser scans + annotations + motions -> USDNet processed layout.

Reproduces what USDNet's ``datasets/preprocessing/articulate3d_preprocessing_challenge.py``
(``Articulate3DPreprocessing`` / ``SceneParser``) writes for Articulate3D, but
from the SceneFun3D raw assets that our SF3D split is built on:

    <scans>/<visit>/<visit>_laser_scan.ply          (laser frame, 5 mm)
    <scans>/<visit>/<visit>_annotations.json        {"annotations": [{annot_id, indices, label}]}
    <scans>/<visit>/<visit>_motions.json            {"motions": [{annot_id, motion_type,
                                                                   motion_dir, motion_origin_idx}]}
    <scans>/<visit>/<video>/<video>_transform.npy   4x4 laser -> ARKit (y up)

Output (``<out>`` = ``data.*_dataset.data_dir`` in USDNet's hydra config):

    <out>/<mode>/<visit>.npy                float32 (N, 13): xyz | rgb 0..255 | normals |
                                            sem | inst | segments(0) | inter
    <out>/<mode>/<visit>_articulation.h5    {str(inst): {sem_id, axis, origin, inter_mask}}
    <out>/<mode>/<visit>_T_up.json          {"T_up": 4x4 laser -> z-up, ...}
    <out>/<mode>/<visit>_filebase.json      the database entry (used for idempotent reruns)
    <out>/instance_gt/<mode>/<visit>.txt    sem*1000 + inst + 1 per point (0 -> 1 for background,
                                            exactly as upstream)
    <out>/expand_dict/<visit>.pkl           upstream ``expand_instances_and_semantics`` records
    <out>/{train,validation,test,train_validation}_database.yaml, label_database.yaml,
    color_mean_std.yaml

Conventions:
  * Points are expressed in a z-up frame: ``T_up = R_YUP_TO_ZUP @ laser_to_arkit``;
    the same transform maps axes (rotation part) and origins.  ``T_up`` is stored
    next to the scene so ``usdnet_preds_to_jsonl.py`` can map predictions back to
    the laser frame our LMDB extrinsics expect.
  * sem 1 = rotation ('rot'), 2 = translation ('trans'); inst = running int from 1
    in ``motions.json`` order over motions whose annotation exists and is not
    labelled ``exclude``; ``inter`` == ``inst`` (SceneFun3D annotates the
    interactable element itself, there is no separate movable part).
  * The voxel downsample (default 2 cm) averages coordinates and colours per
    voxel, like ``open3d.voxel_down_sample``.  Full-res annotation indices are
    mapped to the nearest downsampled point within the voxel size (upstream uses
    a 2 cm KD-tree tolerance).
  * Normals are all zero: USDNet's Articulate3D config sets ``add_normals: false``
    so the three normal columns are never consumed.  We therefore do not need
    open3d's ``estimate_normals``.
  * Dependencies: numpy, h5py, yaml.  ``plyfile`` and ``scipy`` are used when
    importable; otherwise a numpy PLY reader (binary_little_endian / ascii,
    vertex x,y,z,red,green,blue) and a grid-hash neighbour index are used.
    The fallbacks are correct but slower on real scans; install scipy for the
    full conversion.

CLI:
    python tools/baselines_sf3d/sf3d_to_usdnet.py --scans <scans> --out <out> \
        --splits experiments/baselines_sf3d/splits.json --workers 8

Plan: docs/superpowers/plans/2026-09-12-sf3d-external-baselines.md (Task 7).
"""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import yaml

try:  # optional, used when available
    from scipy.spatial import cKDTree as _SciKDTree
except Exception:  # pragma: no cover - exercised only where scipy is missing
    _SciKDTree = None

# ARKit is y-up; USDNet / Articulate3D scenes are z-up.
R_YUP_TO_ZUP = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)

# Upstream label database (articulate3d_preprocessing_challenge.py).
CLASS_LABELS = ["background", "rotation", "translation"]
CLASS_IDS = [0, 1, 2]
VALID_CLASS_IDS = [1, 2]
ARTICULATE3D_COLOR_MAP = {0: (0.0, 0.0, 0.0), 1: (191, 246, 112), 2: (110, 239, 148)}
SEM_OF_MOTION = {"rot": 1, "rotation": 1, "trans": 2, "translation": 2}
MODE_OF_SPLIT = {"train": "train", "bvalid": "validation", "test": "test"}
MODES = ("train", "validation", "test")


# --------------------------------------------------------------------------
# PLY reading
# --------------------------------------------------------------------------
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
_COLOR_NAMES = (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue"))


def _rgb_from_fields(names, get):
    for trio in _COLOR_NAMES:
        if all(n in names for n in trio):
            rgb = np.stack([np.asarray(get(n), dtype=np.float64) for n in trio], 1)
            if rgb.size and rgb.max() <= 1.0 and np.issubdtype(np.asarray(get(trio[0])).dtype, np.floating):
                rgb = rgb * 255.0
            return rgb
    n = len(np.asarray(get("x")))
    return np.zeros((n, 3), dtype=np.float64)


def _read_ply_numpy(path):
    """Minimal PLY vertex reader (ascii / binary_little_endian / binary_big_endian)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            header.append(line.decode("ascii", "replace").strip())
            if header[-1] == "end_header":
                break
        data_start = f.tell()
    if header[0] != "ply":
        raise ValueError(f"{path}: not a PLY file")
    fmt = None
    elements = []  # (name, count, [(prop_name, np_type)]) in file order
    for line in header[1:]:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append((tok[1], int(tok[2]), []))
        elif tok[0] == "property":
            if tok[1] == "list":
                elements[-1][2].append((tok[4], None))
            else:
                elements[-1][2].append((tok[2], _PLY_TYPES[tok[1]]))
    if not elements or elements[0][0] != "vertex":
        raise ValueError(f"{path}: expected 'vertex' as the first element, got {elements[:1]}")
    _, n_vert, props = elements[0]
    if any(t is None for _, t in props):
        raise ValueError(f"{path}: list properties on vertex are unsupported")
    if fmt == "ascii":
        with open(path, "rb") as f:
            f.seek(data_start)
            arr = np.loadtxt(f, max_rows=n_vert, ndmin=2) if n_vert else np.zeros((0, len(props)))
        cols = {name: arr[:, i] for i, (name, _) in enumerate(props)}
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + t) for name, t in props])
        with open(path, "rb") as f:
            f.seek(data_start)
            rec = np.fromfile(f, dtype=dtype, count=n_vert)
        cols = {name: rec[name] for name, _ in props}
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    rgb = _rgb_from_fields(set(cols), cols.__getitem__)
    return xyz, rgb


def read_ply(path):
    """-> (xyz float64 (N,3), rgb float64 (N,3) in 0..255). Uses plyfile if installed."""
    path = str(path)
    try:
        from plyfile import PlyData
    except ImportError:
        return _read_ply_numpy(path)
    v = PlyData.read(path)["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = _rgb_from_fields(set(v.dtype.names), lambda n: v[n])
    return xyz, rgb


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def voxel_downsample(xyz, rgb, voxel):
    """One point per occupied voxel: centroid of coordinates and mean colour
    (open3d.voxel_down_sample semantics). Returns (xyz_ds, rgb_ds, voxel_of_point)
    with ``voxel_of_point`` mapping each input row to its downsampled row."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return xyz.reshape(0, 3), np.zeros((0, 3)), np.zeros(0, np.int64)
    key = np.floor(xyz / voxel).astype(np.int64)
    key -= key.min(0)
    span = key.max(0) + 1
    h = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    _, inv = np.unique(h, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv).astype(np.float64)
    xyz_ds = np.stack([np.bincount(inv, weights=xyz[:, i]) for i in range(3)], 1) / cnt[:, None]
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb_ds = np.stack([np.bincount(inv, weights=rgb[:, i]) for i in range(3)], 1) / cnt[:, None]
    return xyz_ds, rgb_ds, inv


class _GridTree:
    """numpy stand-in for the two cKDTree calls we use (``query`` with a distance
    bound, ``query_ball_point``). Cells are ``cell`` wide; a query touches 27
    cells, so ``cell`` must be >= the largest radius queried."""

    def __init__(self, pts, cell):
        self.pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        self.cell = float(cell)
        self.n = len(self.pts)
        self.cells = {}
        if self.n:
            key = np.floor(self.pts / self.cell).astype(np.int64)
            order = np.lexsort(key.T[::-1])
            ks = key[order]
            change = np.flatnonzero(np.any(ks[1:] != ks[:-1], axis=1)) + 1
            starts = np.concatenate([[0], change])
            ends = np.concatenate([change, [self.n]])
            for s, e in zip(starts, ends):
                self.cells[tuple(ks[s])] = order[s:e]
        self._offsets = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])

    def _candidates(self, p):
        c = np.floor(p / self.cell).astype(np.int64)
        parts = [self.cells[t] for t in map(tuple, c + self._offsets) if t in self.cells]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    def query(self, q, k=1, distance_upper_bound=np.inf):
        assert k == 1
        q = np.asarray(q, dtype=np.float64).reshape(-1, 3)
        dist = np.full(len(q), np.inf)
        idx = np.full(len(q), self.n, dtype=np.int64)
        for i, p in enumerate(q):
            cand = self._candidates(p)
            if len(cand) == 0:
                continue
            d = np.linalg.norm(self.pts[cand] - p, axis=1)
            j = int(np.argmin(d))
            if d[j] <= distance_upper_bound:
                dist[i], idx[i] = d[j], cand[j]
        return dist, idx

    def query_ball_point(self, p, r):
        cand = self._candidates(np.asarray(p, dtype=np.float64))
        if len(cand) == 0:
            return []
        d = np.linalg.norm(self.pts[cand] - p, axis=1)
        return cand[d <= r].tolist()


def make_tree(pts, cell):
    """cKDTree when scipy is importable, else the grid index (cell >= query radius)."""
    if _SciKDTree is not None:
        return _SciKDTree(np.asarray(pts, dtype=np.float64).reshape(-1, 3))
    return _GridTree(pts, cell)


def nearest_within(tree_pts, query_pts, radius, tree=None):
    """Index of the nearest ``tree_pts`` row within ``radius`` for each query (-1 if none)."""
    query_pts = np.asarray(query_pts, dtype=np.float64).reshape(-1, 3)
    if len(query_pts) == 0 or len(tree_pts) == 0:
        return np.full(len(query_pts), -1, dtype=np.int64)
    tree = tree if tree is not None else make_tree(tree_pts, radius)
    d, idx = tree.query(query_pts, k=1, distance_upper_bound=radius)
    idx = np.asarray(idx, dtype=np.int64)
    idx[~np.isfinite(d)] = -1
    idx[idx >= len(tree_pts)] = -1
    return idx


# --------------------------------------------------------------------------
# Vendored from USDNet datasets/preprocessing/articulate3d_preprocessing_challenge.py
# (only ``cKDTree(points_bg)`` -> ``make_tree(points_bg, radius)``; the rest is verbatim).
# --------------------------------------------------------------------------
def expand_instances_and_semantics(points, instance_ids, sem_ids, radius=0.1, ignore_index=0):
    expand_idx_records = []
    expand_inst_records = []
    expand_sem_records = []
    expand_distances = []

    # Initialize expanded instance and semantic IDs with original assignments
    expanded_instance_ids = instance_ids.copy()
    expanded_sem_ids = sem_ids.copy()
    # Create a k-d tree for neighborhood queries, only expand points to background
    bg_mask = instance_ids == ignore_index
    points_bg = points[bg_mask]
    tree = make_tree(points_bg, radius)
    # Iterate over each unique instance ID
    for instance_id in np.unique(instance_ids):
        if instance_id == ignore_index:
            continue
        # Mask for points in the current instance
        instance_mask = instance_ids == instance_id
        instance_points = points[instance_mask]
        instance_sem_id = sem_ids[instance_mask][0]  # Assume all points in an instance have the same sem_id
        # Expand points for this instance
        for idx, point in zip(np.where(instance_mask)[0], instance_points):
            neighbors_idx = tree.query_ball_point(point, radius)
            for neighbor_idx in neighbors_idx:
                distance = np.min(np.linalg.norm(points[instance_mask] - points_bg[neighbor_idx], axis=1))
                cur_id = expanded_instance_ids[bg_mask][neighbor_idx]
                # If the neighbor is closer to the current instance than its current assigned instance
                not_assigned = cur_id == ignore_index
                if not_assigned:
                    # Update instance and semantic IDs
                    indices_to_update = np.where(bg_mask)[0][neighbor_idx]
                    expanded_instance_ids[indices_to_update] = instance_id
                    expanded_sem_ids[indices_to_update] = instance_sem_id
                    ## record
                    expand_idx_records.append(indices_to_update)
                    expand_inst_records.append(instance_id)
                    expand_sem_records.append(instance_sem_id)
                    expand_distances.append(distance)
                else:
                    # If the neighbor is closer to the current instance than its current assigned instance
                    ## calculate the distance to the current instance
                    # Calculate distances
                    if cur_id != instance_id:
                        cur_instance_mask = instance_ids == cur_id
                        cur_instance_points = points[cur_instance_mask]
                        distance_cur = np.min(np.linalg.norm(cur_instance_points - points_bg[neighbor_idx], axis=1))
                        if distance < distance_cur:
                            # Update instance and semantic IDs
                            indices_to_update = np.where(bg_mask)[0][neighbor_idx]
                            expanded_instance_ids[indices_to_update] = instance_id
                            expanded_sem_ids[indices_to_update] = instance_sem_id

                            ## update records
                            record_idx = expand_idx_records.index(indices_to_update)
                            expand_inst_records[record_idx] = instance_id
                            expand_sem_records[record_idx] = instance_sem_id
                            expand_distances[record_idx] = distance
    expand_idx_records = np.array(expand_idx_records)
    expand_inst_records = np.array(expand_inst_records)
    expand_sem_records = np.array(expand_sem_records)
    expand_distances = np.array(expand_distances)
    expand_dict = {
        "expand_idx_records": expand_idx_records,
        "expand_inst_records": expand_inst_records,
        "expand_sem_records": expand_sem_records,
        "expand_distances": expand_distances
    }
    return expand_dict


# Vendored verbatim from the same upstream file.
def save_dict_to_h5group(h5file, data_dict, h5group):
    for key, value in data_dict.items():
        # Convert non-string keys to strings (like integers)
        if not isinstance(key, str):
            key = str(key)

        if isinstance(value, dict):
            # If the value is a dictionary, create a subgroup
            subgroup = h5group.create_group(key)
            save_dict_to_h5group(h5file, value, subgroup)  # Recursively save the nested dictionary
        elif isinstance(value, list):
            # If the value is a list, convert it to a NumPy array
            h5group.create_dataset(key, data=np.array(value))
        elif isinstance(value, str):
            # Handle string data separately
            dt = h5py.string_dtype(encoding='utf-8')
            h5group.create_dataset(key, data=value, dtype=dt)
        else:
            # Save arrays or scalar values directly
            h5group.create_dataset(key, data=value)


def save_dict_to_h5file(data_dict, file_path):
    with h5py.File(file_path, 'w') as h5file:
        save_dict_to_h5group(h5file, data_dict, h5file)


# --------------------------------------------------------------------------
# Scene conversion
# --------------------------------------------------------------------------
def scene_paths(visit, scan_dir):
    """Locate the raw assets of ``visit``. ``scan_dir`` is the scans root
    (``<scans>/<visit>/...``) or the visit folder itself."""
    scan_dir = Path(scan_dir)
    folder = scan_dir / visit if (scan_dir / visit).is_dir() else scan_dir
    ply = folder / f"{visit}_laser_scan.ply"
    ann = folder / f"{visit}_annotations.json"
    mot = folder / f"{visit}_motions.json"
    transforms = sorted(folder.glob("*/*_transform.npy"))
    if not transforms:
        raise FileNotFoundError(f"{visit}: no <video>/<video>_transform.npy under {folder}")
    for p in (ply, ann, mot):
        if not p.is_file():
            raise FileNotFoundError(p)
    return {"folder": folder, "ply": ply, "annotations": ann, "motions": mot, "transform": transforms[0]}


def load_gt(annotations_path, motions_path):
    """-> list of (annot_id, label, indices, motion) for motions whose annotation
    exists and is not labelled 'exclude', in motions.json order."""
    with open(annotations_path) as f:
        annots = json.load(f)["annotations"]
    with open(motions_path) as f:
        motions = json.load(f)["motions"]
    by_id = {a["annot_id"]: a for a in annots}
    out = []
    for m in motions:
        a = by_id.get(m.get("annot_id"))
        if a is None or a.get("label") == "exclude":
            continue
        out.append((a["annot_id"], a.get("label"), a.get("indices") or [], m))
    return out


def _outputs(out_dir, mode, visit):
    out_dir = Path(out_dir)
    return {
        "npy": out_dir / mode / f"{visit}.npy",
        "h5": out_dir / mode / f"{visit}_articulation.h5",
        "T_up": out_dir / mode / f"{visit}_T_up.json",
        "filebase": out_dir / mode / f"{visit}_filebase.json",
        "instance_gt": out_dir / "instance_gt" / mode / f"{visit}.txt",
        "expand": out_dir / "expand_dict" / f"{visit}.pkl",
    }


def convert_scene(visit, scan_dir, out_dir, mode, voxel=0.02, expand_radius=0.1, verbose=True):
    """Convert one visit; returns the database ("filebase") entry."""
    visit = str(visit)
    paths = scene_paths(visit, scan_dir)
    outs = _outputs(out_dir, mode, visit)
    for p in outs.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    xyz_full, rgb_full = read_ply(paths["ply"])
    laser_to_arkit = np.asarray(np.load(paths["transform"]), dtype=np.float64).reshape(4, 4)
    T_up = R_YUP_TO_ZUP @ laser_to_arkit
    R_up = T_up[:3, :3]

    xyz_ds, rgb_ds, _ = voxel_downsample(xyz_full, rgb_full, voxel)
    n = len(xyz_ds)
    xyz_up = (T_up[:3, :3] @ xyz_ds.T).T + T_up[:3, 3]
    colors = rgb_ds.astype(np.int32)  # upstream: (o3d colors * 255).astype(int32)
    normals = np.zeros((n, 3), dtype=np.float64)  # unused by USDNet (add_normals: false)

    sem_gt = np.zeros(n, dtype=np.int32)
    inst_gt = np.zeros(n, dtype=np.int32)
    tree = make_tree(xyz_ds, voxel)

    # First pass: gather instances, then assign in motion order (later motions overwrite).
    gathered = []
    for annot_id, label, indices, m in load_gt(paths["annotations"], paths["motions"]):
        sem = SEM_OF_MOTION.get(str(m.get("motion_type", "")).lower())
        if sem is None:
            if verbose:
                print(f"[{visit}] {annot_id}: unknown motion_type {m.get('motion_type')!r}, skipped")
            continue
        idx_full = np.asarray(indices, dtype=np.int64).reshape(-1)
        idx_full = idx_full[(idx_full >= 0) & (idx_full < len(xyz_full))]
        origin_idx = m.get("motion_origin_idx")
        mdir = m.get("motion_dir")
        if origin_idx is None or mdir is None or not (0 <= int(origin_idx) < len(xyz_full)):
            if verbose:
                print(f"[{visit}] {annot_id}: missing/invalid motion_origin_idx or motion_dir, skipped")
            continue
        ds_idx = nearest_within(xyz_ds, xyz_full[idx_full], voxel, tree=tree)
        ds_idx = np.unique(ds_idx[ds_idx >= 0])
        if len(ds_idx) == 0:
            if verbose:
                print(f"[{visit}] {annot_id} ({label}): no downsampled points, skipped")
            continue
        axis = R_up @ np.asarray(mdir, dtype=np.float64).reshape(3)
        nrm = np.linalg.norm(axis)
        axis = axis / nrm if nrm > 0 else axis
        origin = R_up @ xyz_full[int(origin_idx)] + T_up[:3, 3]
        gathered.append((sem, ds_idx, axis, origin, annot_id, label))

    articulation_gt = {}
    for k, (sem, ds_idx, axis, origin, annot_id, label) in enumerate(gathered, start=1):
        sem_gt[ds_idx] = sem
        inst_gt[ds_idx] = k
    # Drop instances fully overwritten by later ones and renumber contiguously.
    keep = [k for k in range(1, len(gathered) + 1) if np.any(inst_gt == k)]
    remap = {k: i + 1 for i, k in enumerate(keep)}
    new_inst = np.zeros_like(inst_gt)
    for k, i in remap.items():
        new_inst[inst_gt == k] = i
    inst_gt = new_inst
    inter_gt = inst_gt.copy()  # interactable == movable for SceneFun3D elements
    for k, i in remap.items():
        sem, _, axis, origin, annot_id, label = gathered[k - 1]
        mask = inst_gt == i
        articulation_gt[i] = {
            "sem_id": int(sem),
            "axis": axis.astype(np.float64),
            "origin": origin.astype(np.float64),
            "inter_mask": inter_gt[mask] != 0,
        }
    inst_to_annot = {i: str(gathered[k - 1][4]) for k, i in remap.items()}

    segments = np.zeros(n, dtype=np.int32)
    points = np.hstack(
        [xyz_up, colors, normals, sem_gt[:, None], inst_gt[:, None], segments[:, None], inter_gt[:, None]]
    ).astype(np.float32)
    np.save(outs["npy"], points)

    gt_labels = sem_gt * 1000 + inst_gt + 1
    np.savetxt(outs["instance_gt"], gt_labels.astype(np.int32), fmt="%d")

    expand_dict = expand_instances_and_semantics(xyz_up, inter_gt, sem_gt, radius=expand_radius)
    with open(outs["expand"], "wb") as f:
        pickle.dump(expand_dict, f)

    save_dict_to_h5file(articulation_gt, str(outs["h5"]))

    with open(outs["T_up"], "w") as f:
        json.dump(
            {
                "T_up": T_up.tolist(),
                "laser_to_arkit": laser_to_arkit.tolist(),
                "R_yup_to_zup": R_YUP_TO_ZUP.tolist(),
                "video_id": paths["transform"].parent.name,
                "voxel": voxel,
                "n_points": int(n),
                "instances": inst_to_annot,
            },
            f,
            indent=1,
        )

    c = colors.astype(np.float64) / 255.0
    filebase = {
        "filepath": str(outs["npy"]),
        "raw_filepath": str(paths["ply"]),
        "scene": visit,
        "color_mean": [float(c[:, i].mean()) for i in range(3)] if n else [0.0, 0.0, 0.0],
        "color_std": [float((c[:, i] ** 2).mean()) for i in range(3)] if n else [0.0, 0.0, 0.0],
        "instance_gt_filepath": str(outs["instance_gt"]),
        "expand_dict_file": str(outs["expand"]),
        "articulation_gt_file": str(outs["h5"]),
    }
    with open(outs["filebase"], "w") as f:
        json.dump(filebase, f, indent=1)
    if verbose:
        print(f"[{visit}] {mode}: {n} points, {len(articulation_gt)} instances, "
              f"{len(expand_dict['expand_idx_records'])} expanded points")
    return filebase


def _convert_job(args):
    visit, scan_dir, out_dir, mode, voxel, force = args
    outs = _outputs(out_dir, mode, visit)
    if not force and all(p.is_file() for p in outs.values()):
        with open(outs["filebase"]) as f:
            fb = json.load(f)
        return visit, mode, fb, None
    try:
        return visit, mode, convert_scene(visit, scan_dir, out_dir, mode, voxel=voxel), None
    except Exception as e:  # report, do not kill the pool
        return visit, mode, None, f"{type(e).__name__}: {e}"


def n_instances(filebase):
    with h5py.File(filebase["articulation_gt_file"], "r") as h:
        return len(h.keys())


# --------------------------------------------------------------------------
# Database yamls (BasePreprocessing._save_yaml format)
# --------------------------------------------------------------------------
def save_yaml(path, obj):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, default_style=None, default_flow_style=False)


def write_label_database(out_dir):
    db = {}
    for cid in CLASS_IDS:
        db[cid] = {
            "color": [float(v) for v in ARTICULATE3D_COLOR_MAP[cid]],
            "name": CLASS_LABELS[cid],
            "validation": cid in VALID_CLASS_IDS,
        }
    save_yaml(Path(out_dir) / "label_database.yaml", db)
    return db


def write_color_mean_std(out_dir, train_database):
    if not train_database:
        obj = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        save_yaml(Path(out_dir) / "color_mean_std.yaml", obj)
        return obj
    color_mean = np.array([s["color_mean"] for s in train_database], dtype=np.float64)
    color_std = np.array([s["color_std"] for s in train_database], dtype=np.float64)
    mean = color_mean.mean(axis=0)
    std = np.sqrt(np.maximum(color_std.mean(axis=0) - mean ** 2, 0.0))
    obj = {"mean": [float(v) for v in mean], "std": [float(v) for v in std]}
    save_yaml(Path(out_dir) / "color_mean_std.yaml", obj)
    return obj


def write_databases(out_dir, databases):
    """databases: {mode: [filebase, ...]}. Writes the five database yamls,
    label_database.yaml and color_mean_std.yaml."""
    out_dir = Path(out_dir)
    for mode in MODES:
        save_yaml(out_dir / f"{mode}_database.yaml", databases.get(mode, []))
    save_yaml(
        out_dir / "train_validation_database.yaml",
        list(databases.get("train", [])) + list(databases.get("validation", [])),
    )
    write_label_database(out_dir)
    return write_color_mean_std(out_dir, databases.get("train", []) or sum(databases.values(), []))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scans", required=True, help="root with <visit>/ folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", required=True, help="splits.json with test/train/bvalid scene lists")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--force", action="store_true", help="reconvert even if outputs exist")
    a = ap.parse_args(argv)

    with open(a.splits) as f:
        splits = json.load(f)
    jobs = []
    for split_name, mode in MODE_OF_SPLIT.items():
        if mode not in a.modes:
            continue
        for visit in sorted(splits.get(split_name, [])):
            jobs.append((str(visit), a.scans, a.out, mode, a.voxel, a.force))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} scenes -> {a.out} (workers={a.workers}, voxel={a.voxel})", flush=True)

    if a.workers > 1:
        with Pool(a.workers) as pool:
            results = list(pool.imap_unordered(_convert_job, jobs))
    else:
        results = [_convert_job(j) for j in jobs]

    databases = {m: [] for m in MODES}
    failed, zero = [], []
    for visit, mode, fb, err in results:
        if err is not None:
            failed.append((visit, mode, err))
            continue
        databases[mode].append(fb)
        if n_instances(fb) == 0:
            zero.append((visit, mode))
    for m in databases:
        databases[m].sort(key=lambda fb: fb["scene"])
    stats = write_databases(a.out, databases)

    for m in MODES:
        print(f"{m}: {len(databases[m])} scenes")
    print(f"color_mean_std: {stats}")
    if zero:
        print(f"scenes with zero instances ({len(zero)}): " + ", ".join(f"{v} [{m}]" for v, m in zero))
    if failed:
        print(f"FAILED ({len(failed)}):")
        for v, m, e in failed:
            print(f"  {v} [{m}]: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
