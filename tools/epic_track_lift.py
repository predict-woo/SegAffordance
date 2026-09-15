"""Add the record's 2D hand track to the EPIC annotator clouds, lifted into the depth cloud.

For each test record: the LMDB's trajectory_2d_image_coords (native pixels, with trajectory_2d_valid) is scaled
to the render size, and every valid track pixel is lifted to 3D by taking the depth-cloud point that projects
nearest to it (within --radius px; the cloud is the per-pixel back-projection subsampled to 400k points). Writes
track_px (N x 2, render px), track_valid (N bool: valid AND lifted) and track_xyz (N x 3, camera frame, metres)
into the record's npz; the viewer draws the track on the thumbnail and as a polyline in the cloud.

  /opt/venv/bin/python tools/epic_track_lift.py --clouds /workspace/datasets/epic_gt_annot/clouds
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.scenefun3d import SF3DDataset, split_dataset_by_scene  # noqa: E402

ROOT = "/workspace/datasets/epic_processed_2d"
KEYS = "/workspace/cache/epic_2d_keys_v1.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", required=True)
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--split-config", default="config/epic_v1_rgb_scalefree.yaml")
    a = ap.parse_args()
    dcfg = yaml.safe_load(open(a.split_config))["data"]
    ds = SF3DDataset(
        lmdb_data_root=ROOT, lmdb_path=f"{ROOT}/data.lmdb", frame_cache_path=f"{ROOT}/frames.lmdb", key_cache_path=KEYS,
        image_size_for_mask_reconstruction=(512, 512), return_trajectory_2d=True, point_source="element", fast_pipeline=True,
        load_depth=False, min_revolute_radius=0.0, min_mask_area_frac=0.0, edge_margin_frac=0.0, sensor_max_occluded_frac=0.5,
    )
    _, va = split_dataset_by_scene(ds, dcfg.get("val_split_ratio", 0.15), dcfg.get("manual_seed", 42))
    by_key = {ds.item_keys[i].decode(): i for i in va.indices}
    recs = json.load(open(os.path.join(a.clouds, "index.json")))["records"]
    for r in recs:
        fn = os.path.join(a.clouds, r["file"]); z = dict(np.load(fn, allow_pickle=True))
        idx = by_key[r["key"]]
        it = ds[idx]
        img_size, traj2d_px, valid2d = it[8], it[13], it[14]
        W, H = float(img_size[0]), float(img_size[1]); Wr, Hr = [int(v) for v in z["size"]]
        px = traj2d_px.float().numpy() * np.array([Wr / W, Hr / H])
        valid = valid2d.numpy().astype(bool)
        K = z["K_render"]; xyz = z["xyz"]
        u = K[0, 0] * xyz[:, 0] / xyz[:, 2] + K[0, 2]; v = K[1, 1] * xyz[:, 1] / xyz[:, 2] + K[1, 2]
        track_xyz = np.zeros((len(px), 3), np.float32); lifted = np.zeros(len(px), bool)
        for j, (tu, tv) in enumerate(px):
            if not valid[j]:
                continue
            d2 = (u - tu) ** 2 + (v - tv) ** 2; k = int(np.argmin(d2))
            if d2[k] <= a.radius ** 2:
                track_xyz[j] = xyz[k]; lifted[j] = True
        z["track_px"] = px.astype(np.float32); z["track_valid"] = valid & lifted; z["track_xyz"] = track_xyz
        np.savez(fn, **z)
        print(f"{r['key']:28s} track {len(px)} pts, valid {int(valid.sum())}, lifted {int((valid & lifted).sum())}")


if __name__ == "__main__":
    main()
