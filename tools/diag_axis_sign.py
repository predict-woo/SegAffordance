"""Signed vs unsigned axis-error probe (and the sigmoid-octant post-mortem).

Written 2026-08-18 to test the hypothesis that MotionMLP's Sigmoid confines
predicted axes to the positive octant while the gen-6+ loss is
sign-SENSITIVE. **The hypothesis was FALSE**: CRIS.forward rescales the
sigmoid output with (x - 0.5) * 2 to (-1, 1) (segmenter.py), so every
direction is representable — the g16 probe showed negative predicted
components and NO error difference between "reachable" and "unreachable"
GT rows. Part A's "floor" statistics are therefore vacuous for the real
model; they are kept only because the tool documents the investigation.

What the probe DOES measure usefully (part B): the per-row SIGNED axis
error next to the sign-agnostic |cos| error the test metric reports — i.e.
the size of the sign-flip tail the axis metric hides (g16 @ ep21: ~10% of
rot rows sit above 120 deg signed while their |cos| error is small).

  /opt/venv/bin/python -u tools/diag_axis_sign.py \
    --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl \
    --config config/sf3d_train_runpod_g16_trajnorm.yaml \
    --ckpt experiments/20260817_sf3d_g16_trajnorm/checkpoints/best-epoch21-valloss1.0123.ckpt \
    --data-root /workspace/datasets/sf3d_processed_v3 \
    --input-size 512 --frame-cache-path /workspace/datasets/sf3d_frames_512.lmdb \
    --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05
"""
import argparse
import os
import pickle
import sys

import lmdb
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.scenefun3d import (  # noqa: E402
    SF3DDataset,
    get_default_transforms,
    split_dataset_by_scene,
)
from model.losses.geometric import normalized_intrinsics  # noqa: E402


def floor_deg(d):
    """Structural signed-error floor (deg) for a positive-octant unit pred."""
    d = np.asarray(d, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-8:
        return float("nan")
    best_cos = np.linalg.norm(np.clip(d / n, 0.0, None))
    return float(np.degrees(np.arccos(np.clip(best_cos, -1.0, 1.0))))


def angle_deg(a, b, signed):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    c = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    if not signed:
        c = abs(c)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def summarize(name, arr):
    a = np.asarray([x for x in arr if np.isfinite(x)])
    if len(a) == 0:
        print(f"  {name}: (empty)")
        return
    print(f"  {name:34s} n={len(a):6d}  mean {a.mean():7.2f}  p50 "
          f"{np.percentile(a, 50):7.2f}  p90 {np.percentile(a, 90):7.2f}  "
          f"max {a.max():7.2f}")


def part_a(args):
    cached = pickle.loads(open(args.key_cache, "rb").read())
    keys = cached["keys"]
    print(f"[A] GT floor over {len(keys)} keys ({args.data_root})", flush=True)
    env = lmdb.open(f"{args.data_root}/data.lmdb", readonly=True, lock=False)
    floors = {"rot": [], "trans": []}
    neg_any = {"rot": 0, "trans": 0}
    with env.begin() as txn:
        for i, key in enumerate(keys):
            if (i + 1) % 20000 == 0:
                print(f"  progress {i + 1}/{len(keys)}", flush=True)
            rec = pickle.loads(txn.get(key))
            mi = rec.get("motion_info") or {}
            fsm = mi.get("frame_specific_motion_data") or {}
            d = fsm.get("motion_dir_3d_camera_coords")
            if not d:
                continue
            mtype = (mi.get("original_motion_data") or {}).get("motion_type", "trans")
            t = "rot" if mtype in ("rot", "rotation") else "trans"
            f = floor_deg(d)
            floors[t].append(f)
            dn = np.asarray(d, np.float64)
            dn = dn / max(np.linalg.norm(dn), 1e-12)
            if dn.min() < -0.05:
                neg_any[t] += 1
    env.close()
    print("\n[A] structural floor angle (deg) by GT type:")
    for t in ("rot", "trans"):
        summarize(f"{t} floor", floors[t])
        n = len(floors[t])
        if n:
            fl = np.asarray(floors[t])
            print(f"    {t}: {100.0 * neg_any[t] / n:.1f}% rows have a "
                  f"negative component; floor>10deg: "
                  f"{100.0 * float((fl > 10).mean()):.1f}%   floor>45deg: "
                  f"{100.0 * float((fl > 45).mean()):.1f}%")
    return floors


def part_b(args):
    from tools.sf3d_vis_predictions import load_model  # noqa: E402 (needs torch env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _mp = load_model(args.config, args.ckpt, device)
    sz = (args.input_size, args.input_size)
    fcache = args.frame_cache_path or os.path.join(args.data_root, "frames.lmdb")
    r, m, d = get_default_transforms(sz)
    ds = SF3DDataset(
        lmdb_data_root=args.data_root,
        key_cache_path=args.key_cache,
        frame_cache_path=fcache,
        rgb_transform=r, mask_transform=m, depth_transform=d,
        image_size_for_mask_reconstruction=sz,
        point_source="element", return_trajectory_2d=True,
        min_revolute_radius=args.min_revolute_radius,
        min_mask_area_frac=args.min_mask_area_frac,
        edge_margin_frac=args.edge_margin_frac,
    )
    _, val = split_dataset_by_scene(ds, val_split_ratio=0.1, manual_seed=42)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(val), size=min(args.probe_samples, len(val)), replace=False)
    print(f"\n[B] probing {len(idx)} of {len(val)} val rows with {args.ckpt}",
          flush=True)

    rows = []
    for j, vi in enumerate(idx):
        s = val[int(vi)]
        (img_t, depth_t, desc, _mask, _bbox, _pt, motion_gt, type_gt,
         img_size, _name, _origin3d, K, _t3d, _t2d, _t2dv) = s
        K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
        with torch.no_grad():
            word = model.tokenize([desc], 77).to(device)
            out = model(img_t[None].to(device), depth_t[None].to(device),
                        word, None, None, None, None, K_norm.to(device).float())
        pred = out.motion_pred[0].float().cpu().numpy()
        gt = motion_gt.float().numpy()
        rows.append({
            "vi": int(vi),
            "type": "rot" if int(type_gt) else "trans",
            "desc": desc,
            "floor": floor_deg(gt),
            "signed": angle_deg(pred, gt, signed=True),
            "unsigned": angle_deg(pred, gt, signed=False),
            "pred_min_comp": float(pred.min()),
        })
        if (j + 1) % 100 == 0:
            print(f"  probe {j + 1}/{len(idx)}", flush=True)

    print(f"\n[B] pred components: min over all rows "
          f"{min(r['pred_min_comp'] for r in rows):.5f} "
          f"(sigmoid => must be > 0)")
    for t in ("rot", "trans"):
        sub = [r for r in rows if r["type"] == t]
        reach = [r for r in sub if r["floor"] <= 10.0]
        unreach = [r for r in sub if r["floor"] > 10.0]
        print(f"\n[B] {t} rows ({len(sub)}): reachable(floor<=10deg) "
              f"{len(reach)}  unreachable {len(unreach)}")
        for label, grp in (("reachable", reach), ("unreachable", unreach)):
            summarize(f"{t}/{label} SIGNED err", [r["signed"] for r in grp])
            summarize(f"{t}/{label} |cos| err", [r["unsigned"] for r in grp])
            summarize(f"{t}/{label} floor", [r["floor"] for r in grp])

    worst = sorted(rows, key=lambda r: -r["signed"])[:args.worst]
    print(f"\n[B] worst {len(worst)} rows by SIGNED error:")
    for r in worst:
        print(f"  val{r['vi']:5d} [{r['type']}] signed {r['signed']:6.1f}  "
              f"|cos| {r['unsigned']:5.1f}  floor {r['floor']:5.1f}  "
              f"{r['desc'][:52]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-cache", required=True)
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v3")
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--frame-cache-path", default=None)
    ap.add_argument("--min-revolute-radius", type=float, default=0.0)
    ap.add_argument("--min-mask-area-frac", type=float, default=0.0)
    ap.add_argument("--edge-margin-frac", type=float, default=0.0)
    ap.add_argument("--probe-samples", type=int, default=400)
    ap.add_argument("--worst", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-gt", action="store_true")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    if not args.skip_gt:
        part_a(args)
    if not args.skip_model:
        if not (args.config and args.ckpt):
            ap.error("--config and --ckpt required unless --skip-model")
        part_b(args)


if __name__ == "__main__":
    main()
