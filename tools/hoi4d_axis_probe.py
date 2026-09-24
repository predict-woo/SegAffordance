"""HOI4D / EPIC 3D axis probe (pass --root/--keys/--split-config/--gt for EPIC; GT from tools/epic_axis_annotator.py export): score checkpoints on the HOI4D test records that have an official ground-truth
articulation axis (experiments/hoi4d_test_gt_articulation.json, from tools/hoi4d_gt_articulation.py: the
laptop, storage-furniture, safe and trash-can records, 325 of 438). Same protocol as the ARCTIC probe:
unsigned angle between the GT-routed axis head (rot head for revolute GT, trans head for prismatic GT)
and the GT axis in the camera frame; type accuracy against the GT type (from the part motion); for
revolute records the image distance between the projected GT hinge and the record's interaction point
as a sanity check of the pose convention (should be small: the hinge sits on the part).

  python tools/hoi4d_axis_probe.py --model NAME CONFIG CKPT [...] --gt experiments/hoi4d_test_gt_articulation.json \
      --out experiments/<exp>/hoi4d_axis_probe.csv
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.scenefun3d import SF3DDataset, split_dataset_by_scene  # noqa: E402
from model.losses.geometric import normalized_intrinsics, project_points  # noqa: E402
from sf3d_vis_predictions import load_model  # noqa: E402

ROOT = "/workspace/datasets/hoi4d_processed_2d_v2"
KEYS = "/workspace/cache/hoi4d_2d_keys_v2.pkl"
CAT = {"C3": "laptop", "C4": "furniture", "C6": "safe", "C14": "trashcan"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", required=True, metavar=("NAME", "CONFIG", "CKPT"))
    ap.add_argument("--gt", default="experiments/hoi4d_test_gt_articulation.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-config", default="config/hoi4d_v2_rgb_scalefree.yaml")
    ap.add_argument("--root", default=ROOT, help="LMDB root (EPIC: /workspace/datasets/epic_processed_2d)")
    ap.add_argument("--keys", default=KEYS, help="key cache (EPIC: /workspace/cache/epic_2d_keys_v1.pkl)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = json.load(open(a.gt))
    dcfg = yaml.safe_load(open(a.split_config))["data"]
    ds = SF3DDataset(
        lmdb_data_root=a.root, lmdb_path=f"{a.root}/data.lmdb", frame_cache_path=f"{a.root}/frames.lmdb",
        key_cache_path=a.keys, image_size_for_mask_reconstruction=(512, 512), return_trajectory_2d=True,
        point_source="element", fast_pipeline=True, load_depth=True,
        min_revolute_radius=0.0, min_mask_area_frac=0.0, edge_margin_frac=0.0, sensor_max_occluded_frac=0.5,
    )
    _, va = split_dataset_by_scene(ds, dcfg.get("val_split_ratio", 0.15), dcfg.get("manual_seed", 42))
    idxs = [i for i in va.indices if gt.get(ds.item_keys[i].decode(), {}).get("valid")][: a.limit or None]
    print(f"{len(idxs)} test records with a GT axis")
    models = [(n, *load_model(c, k, device)) for n, c, k in a.model]
    rows = {n: [] for n, _, _ in models}
    agree = []
    for j, idx in enumerate(idxs):
        key = ds.item_keys[idx].decode(); g = gt[key]
        (img_t, depth_t, desc, _m, _b, point_gt, _mgt, tgt, img_size, fname, _o3, K, *_rest) = ds[idx]
        K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
        dg = np.asarray(g["axis_cam"], np.float64); dg /= max(np.linalg.norm(dg), 1e-8)
        og = np.asarray(g["origin_cam"], np.float64)
        gt_rot = g["type"] == "rot"
        agree.append(int(tgt) == (1 if gt_rot else 0))
        # convention check: projected GT hinge vs the record's interaction point (image fraction)
        hinge_dist = float("nan")
        if gt_rot and og[2] > 0.05:
            uv = project_points(K_norm, torch.from_numpy(og[None, None]).float())[0, 0].numpy()
            hinge_dist = float(np.linalg.norm(uv - point_gt.numpy()))
        for name, model, mp in models:
            with torch.no_grad():
                word = model.tokenize([desc], 77).to(device)
                out = model(img_t[None].to(device), depth_t[None].to(device), word, None, None, None, None, K_norm.to(device).float())
            head = out.motion_pred_rot if gt_rot else out.motion_pred_trans
            d = head[0].cpu().float().numpy().astype(np.float64); d /= max(np.linalg.norm(d), 1e-8)
            c = float(np.clip(np.dot(d, dg), -1, 1)); signed = math.degrees(math.acos(c)); unsigned = min(signed, 180.0 - signed)
            p_rev = float(torch.softmax(out.motion_type_logits[0].float(), -1)[1])
            hinge_line_m = float("nan")   # predicted hinge (metres, at the model's own depth) to the GT axis LINE; revolute GT only
            if gt_rot and out.origin_pred is not None:
                q = out.origin_pred[0].cpu().float().numpy().astype(np.float64); v = q - og
                hinge_line_m = float(np.linalg.norm(v - np.dot(v, dg) * dg))
            rows[name].append((key, g.get("category", key.split("/")[0].split("_")[0]), g["type"], g.get("moving", ""), unsigned, signed, p_rev, int((p_rev > 0.5) == gt_rot), hinge_dist, hinge_line_m))
        if j % 50 == 0:
            print(f"{j}/{len(idxs)}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("model,key,category,gt_type,moving,axis_unsigned_deg,axis_signed_deg,p_rev,type_ok,gt_hinge_to_point_frac,hinge_line_m\n")
        for name, rs in rows.items():
            for r in rs:
                f.write(",".join([name, r[0], r[1], r[2], r[3]] + [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r[4:]]) + "\n")
    print(f"GT type agrees with the record's rule-based type on {100 * np.mean(agree):.1f}% of records")
    hd = np.array([r[8] for r in next(iter(rows.values()))]); hd = hd[~np.isnan(hd)]
    print(f"projected GT hinge to interaction point: median {np.median(hd):.3f}, mean {hd.mean():.3f} of the image (revolute records)")
    cats = sorted({r[1] for rs in rows.values() for r in rs})
    print(f"{'model':12s} {'n':>4s} {'uns mean':>8s} {'median':>7s} {'<10':>5s} {'<20':>5s} {'flip%':>6s} {'type%':>6s} | " + " ".join(f"{CAT.get(c, c):>9s}" for c in cats))
    for name, rs in rows.items():
        u = np.array([r[4] for r in rs]); s = np.array([r[5] for r in rs]); t = np.array([r[7] for r in rs])
        per = " ".join(f"{np.mean([r[4] for r in rs if r[1] == c]):9.1f}" for c in cats)
        print(f"{name:12s} {len(rs):4d} {u.mean():8.1f} {np.median(u):7.1f} {100 * (u < 10).mean():5.1f} {100 * (u < 20).mean():5.1f} {100 * (s > 90).mean():6.1f} {100 * t.mean():6.1f} | {per}")
    print("per-category columns: unsigned axis error mean (deg)")


if __name__ == "__main__":
    main()
