"""Diagnose the large-radius pathology of the twist head on real val data.

For stratified val samples, runs the model twice — type hint NULL (the val
deployment condition) and type hint = GT — and compares, per sample:

  * |omega_pred| (posterior-mean hedging predicts |omega| ~ P(revolute) < 1)
  * decoded radius  = dist(GT trajectory start, predicted axis line)
    vs GT radius    = dist(GT trajectory start, GT axis line)

If mixture-mean hedging is the mechanism (see the synthetic study in the
session notes), GT-revolute samples show |omega| well below 1, a ONE-SIDED
radius ratio (pred/GT > 1 almost always), the ratio anticorrelated with
|omega| — and forcing the GT type hint should recover part of it.

Run on a pod from the repo root:
  python tools/diag_twist_radius.py \
      --config config/sf3d_train_runpod_twist.yaml \
      --ckpt 'experiments/20260804_sf3d_twist_g3/checkpoints/best-*.ckpt' \
      --num 800
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.opd_train import ModelParams  # noqa: E402
from datasets.scenefun3d import (  # noqa: E402
    SF3DDataset, get_default_transforms, split_dataset_by_scene,
)
from model.losses.twist import decode_twist, point_to_line_distance, twist_from_gt  # noqa: E402
from model.segmenter import CRIS  # noqa: E402


def load_model(config_path, ckpt_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mp = ModelParams(**cfg["model"]["model_params"])
    model = CRIS(mp)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    return model.to(device).eval()


def q(a, p):
    return float(np.percentile(np.asarray(a, dtype=np.float64), p))


def summarize(tag, rows):
    """rows: list of dicts with om, ratio (ratio None for trans rows)."""
    om = [r["om"] for r in rows]
    print(f"\n  [{tag}]  n={len(rows)}")
    if not rows:
        return
    print(f"    |omega|          p25 {q(om,25):.3f}   median {q(om,50):.3f}   p75 {q(om,75):.3f}")
    ratios = [r["ratio"] for r in rows if r.get("ratio") is not None]
    if ratios:
        ra = np.array(ratios)
        print(f"    radius pred/GT   p25 {q(ra,25):.2f}   median {q(ra,50):.2f}   p75 {q(ra,75):.2f}")
        print(f"    frac ratio>1: {float((ra > 1).mean()):.2f}   frac ratio>2: {float((ra > 2).mean()):.2f}")
        lo = np.log(np.array([r["om"] for r in rows if r.get("ratio") is not None]))
        c = np.corrcoef(-lo, np.log(ra))[0, 1]
        print(f"    corr( log 1/|omega|, log ratio ) = {c:.2f}")
    dec_rot = np.array([r["dec_rot"] for r in rows])
    print(f"    decoded as rot: {float(dec_rot.mean()):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="path or glob; newest match used")
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v2")
    ap.add_argument("--key-cache", default="/workspace/cache/sf3d_v2_keys_cutoff05.pkl")
    ap.add_argument("--min-revolute-radius", type=float, default=0.0,
                    help="must match the key cache / training config")
    ap.add_argument("--min-mask-area-frac", type=float, default=0.0,
                    help="must match the key cache / training config")
    ap.add_argument("--num", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    matches = sorted(glob.glob(args.ckpt), key=os.path.getmtime)
    if not matches:
        sys.exit(f"no checkpoint matches {args.ckpt}")
    ckpt = matches[-1]
    print(f"checkpoint: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.config, ckpt, device)

    r, m, d = get_default_transforms((256, 256))
    ds = SF3DDataset(
        lmdb_data_root=args.data_root,
        key_cache_path=args.key_cache,
        frame_cache_path=os.path.join(args.data_root, "frames.lmdb"),
        rgb_transform=r, mask_transform=m, depth_transform=d,
        image_size_for_mask_reconstruction=(256, 256),
        point_source="element", return_trajectory_2d=True,
        min_revolute_radius=args.min_revolute_radius,
        min_mask_area_frac=args.min_mask_area_frac,
    )
    _, val = split_dataset_by_scene(ds, val_split_ratio=0.1, manual_seed=42)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(val))

    rows = []
    want = {0: args.num // 2, 1: args.num - args.num // 2}
    for i in order:
        if not want[0] and not want[1]:
            break
        s = val[int(i)]
        (img_t, depth_t, desc, _mask, _bbox, _pt, motion_gt, type_gt,
         _img_size, _rgb, origin_3d, _K, traj3d, _t2d, _t2dv) = s
        t = int(type_gt)
        if want[t] <= 0:
            continue
        want[t] -= 1

        anchor = traj3d[0].float()                       # GT element point, 3D
        gt_dir = F.normalize(motion_gt.float(), dim=-1)
        gt_radius = float(point_to_line_distance(
            anchor[None], origin_3d.float()[None], gt_dir[None])[0]) if t == 1 else None

        with torch.no_grad():
            word = model.tokenize([desc], 77).to(device)
            img = img_t[None].to(device)
            dep = depth_t[None].to(device)
            for hint_name, hint in (
                ("null", None),
                ("gt", torch.tensor([t], device=device)),
            ):
                out = model(img, dep, word, None, None, None, hint)
                tw = out.twist_pred[0].float().cpu()
                is_rev, direction, axis_pt = decode_twist(tw[None])
                om = float(tw[:3].norm())
                pred_radius = float(point_to_line_distance(
                    anchor[None], axis_pt[0][None], direction[0][None])[0])
                rows.append({
                    "idx": int(i), "gt_type": t, "hint": hint_name,
                    "om": om, "dec_rot": bool(is_rev[0]),
                    "gt_radius": gt_radius, "pred_radius": pred_radius,
                    "ratio": (pred_radius / gt_radius)
                    if (t == 1 and gt_radius and gt_radius > 1e-3) else None,
                })
        n_done = len(rows) // 2
        if n_done % 100 == 0:
            print(f"  {n_done} samples done", flush=True)

    print("\n================ results ================")
    for t, tname in ((1, "GT revolute"), (0, "GT prismatic")):
        for hint in ("null", "gt"):
            summarize(f"{tname} | hint {hint}",
                      [r for r in rows if r["gt_type"] == t and r["hint"] == hint])

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-sample rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
