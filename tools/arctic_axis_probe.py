"""ARCTIC hinge probe: score checkpoints' PREDICTED 3D axis against ARCTIC's GT
hinge (from the object models) on the held-out split — the only 2D hand source
with a real axis. Reports per model: unsigned axis angle (mean / median / %<10 /
%<20 deg), sign-flip rate (signed angle > 90 deg), the projected hinge-line
offset (perpendicular distance, in the image, from the GT origin's projection
to the predicted hinge line; fraction of the image width — scale-free, since
z_p is unsupervised on hand video), and the predicted radius median.

  python tools/arctic_axis_probe.py --model NAME CONFIG CKPT [...] --out viz/<batch>/arctic_axis_probe.csv
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from datasets.scenefun3d import SF3DDataset, split_dataset_by_scene  # noqa: E402
from model.losses.geometric import normalized_intrinsics, project_points  # noqa: E402
from sf3d_vis_predictions import load_model  # noqa: E402

ROOT = "/workspace/datasets/arctic_processed_2d"
KEYS = "/workspace/cache/arctic_2d_keys_v1.pkl"


def line_offset(K_norm, q, d, o_gt):
    """Perpendicular distance (image fraction) from the projected GT origin to the
    projected predicted hinge line (two points q +/- 0.3 d, in front of the camera)."""
    pts = np.stack([q - 0.3 * d, q + 0.3 * d, o_gt])
    if (pts[:, 2] <= 0.05).any():
        return float("nan")
    uv = project_points(K_norm, torch.from_numpy(pts).float()[None])[0].numpy()
    a, b, o = uv[0], uv[1], uv[2]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-6:
        return float("nan")
    return float(abs(ab[0] * (o[1] - a[1]) - ab[1] * (o[0] - a[0])) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", required=True, metavar=("NAME", "CONFIG", "CKPT"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split-config", default="config/arctic_v1_rgb_scalefree.yaml")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dcfg = yaml.safe_load(open(a.split_config))["data"]
    ds = SF3DDataset(
        lmdb_data_root=ROOT, lmdb_path=f"{ROOT}/data.lmdb", frame_cache_path=f"{ROOT}/frames.lmdb",
        key_cache_path=KEYS, image_size_for_mask_reconstruction=(512, 512), return_trajectory_2d=True,
        point_source="element", fast_pipeline=True, load_depth=True,
        min_revolute_radius=0.0, min_mask_area_frac=0.0, edge_margin_frac=0.0, sensor_max_occluded_frac=0.5,
    )
    _, va = split_dataset_by_scene(ds, dcfg.get("val_split_ratio", 0.15), dcfg.get("manual_seed", 42))
    idxs = list(va.indices)[: a.limit or None]
    models = [(n, *load_model(c, k, device)) for n, c, k in a.model]
    rows = {n: [] for n, _, _ in models}
    for j, idx in enumerate(idxs):
        (img_t, depth_t, desc, _m, _b, _pg, mgt, tgt, img_size, fname, o3, K, *_rest) = ds[idx]
        if int(tgt) != 1:
            continue
        K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
        dg = mgt.numpy().astype(np.float64)
        dg /= max(np.linalg.norm(dg), 1e-8)
        og = o3.numpy().astype(np.float64)
        key = ds.item_keys[idx].decode()
        obj = key.split("_")[1] if "_" in key else "?"
        for name, model, mp in models:
            with torch.no_grad():
                word = model.tokenize([desc], 77).to(device)
                out = model(img_t[None].to(device), depth_t[None].to(device), word,
                            None, None, None, None, K_norm.to(device).float())
            d = out.motion_pred_rot[0].cpu().float().numpy().astype(np.float64)  # the rot head (GT-routed)
            d /= max(np.linalg.norm(d), 1e-8)
            c = float(np.clip(np.dot(d, dg), -1, 1))
            signed = math.degrees(math.acos(c))
            unsigned = min(signed, 180.0 - signed)
            p_rev = float(torch.softmax(out.motion_type_logits[0].float(), -1)[1])
            q = out.origin_pred[0].cpu().float().numpy().astype(np.float64)
            p3 = out.point_3d_pred[0].cpu().float().numpy().astype(np.float64)
            cq = q + np.dot(p3 - q, d) * d
            r = float(np.linalg.norm(p3 - cq))
            # offset measured with the GT-signed axis direction (a line has no sign)
            off = line_offset(K_norm, q, d, og)
            rows[name].append((obj, unsigned, signed, p_rev, r, off, float(p3[2])))
        if j % 50 == 0:
            print(f"{j}/{len(idxs)}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("model,object,axis_unsigned_deg,axis_signed_deg,p_rev,radius_m,hinge_offset_frac,z_p\n")
        for name, rs in rows.items():
            for r in rs:
                f.write(f"{name},{r[0]},{r[1]:.3f},{r[2]:.3f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.3f}\n")
    print(f"\n{len(next(iter(rows.values())))} revolute ARCTIC val records")
    print(f"{'model':14s} {'mean':>6s} {'med':>6s} {'<10':>5s} {'<20':>5s} {'flip':>5s} {'offs':>6s} {'r_med':>6s} {'type':>5s}")
    for name, rs in rows.items():
        u = np.array([r[1] for r in rs]); s = np.array([r[2] for r in rs]); pr = np.array([r[3] for r in rs])
        rad = np.array([r[4] for r in rs]); off = np.array([r[5] for r in rs])
        print(f"{name:14s} {u.mean():6.1f} {np.median(u):6.1f} {100*(u<10).mean():5.1f} {100*(u<20).mean():5.1f} "
              f"{100*(s>90).mean():5.1f} {np.nanmedian(off):6.3f} {np.median(rad):6.2f} {100*(pr>0.5).mean():5.1f}")
    print("per object (unsigned mean / flip %):")
    objs = sorted({r[0] for rs in rows.values() for r in rs})
    print(f"{'model':14s} " + " ".join(f"{o[:9]:>11s}" for o in objs))
    for name, rs in rows.items():
        cells = []
        for o in objs:
            u = np.array([r[1] for r in rs if r[0] == o]); s = np.array([r[2] for r in rs if r[0] == o])
            cells.append(f"{u.mean():5.1f}/{100*(s>90).mean():3.0f}" if len(u) else "     -    ")
        print(f"{name:14s} " + " ".join(f"{c:>11s}" for c in cells))
    print("n per object:", {o: sum(1 for r in next(iter(rows.values())) if r[0] == o) for o in objs})


if __name__ == "__main__":
    main()
