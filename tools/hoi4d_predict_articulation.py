"""Cross-domain probe: run checkpoints on held-out frames of a 2D hand source
(--dataset hoi4d | epic | arctic, optional --category key filter, e.g. C3 =
HOI4D laptops) and draw their PREDICTED 3D articulation on the image. HOI4D
and EPIC carry only a rot/trans label (no 3D axis); ARCTIC's GT hinge from
the object model is drawn in green on the GT panel. The scene split matches
the joint4 datamodule (same key cache, ratio 0.15, seed 42).

Panels per sample: [GT | model A | model B ...]
  GT:    moving-part mask (green), 2D knuckle track (cyan), first point ring
  model: predicted mask (red), point_uv ring, projected 3D trajectory head
         (magenta dots, z_p-scaled for scale-free heads), predicted axis
         (red: hinge line through origin_pred for rot, direction ray from
         the lifted point for trans), 90-deg predicted orbit (yellow) for
         rot, origin heatmap uv (small red circle); text = type call,
         p_rev, z_p (predicted depth of the point, m).

  python tools/hoi4d_predict_articulation.py \
      --model rgb_ft config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml experiments/.../best.ckpt \
      --model depth_tf_plain config/sf3d_train_runpod_g19_dct_ft_hoi4d_tf_plain.yaml experiments/.../best.ckpt \
      --out viz/<dated-batch> --num 8 [--category C3] [--split val]
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from datasets.scenefun3d import SF3DDataset, split_dataset_by_scene  # noqa: E402
from model.losses.geometric import (  # noqa: E402
    apply_trajectory_scale, normalized_intrinsics, project_points, trajectory_scale_factor,
)
from sf3d_vis_predictions import (  # noqa: E402
    draw_axis_3d, draw_points_norm, draw_polyline_norm, load_model, overlay_mask, put_lines,
)
from viz_manifest import write_manifest  # noqa: E402

# root, key cache (same as the joint4 configs -> identical key list and scene split), has real 3D GT axes
SOURCES = {
    "hoi4d": ("/workspace/datasets/hoi4d_processed_2d_v2", "/workspace/cache/hoi4d_2d_keys_v2.pkl", False),
    "epic": ("/workspace/datasets/epic_processed_2d", "/workspace/cache/epic_2d_keys_v1.pkl", False),
    "arctic": ("/workspace/datasets/arctic_processed_2d", "/workspace/cache/arctic_2d_keys_v1.pkl", True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", required=True, metavar=("NAME", "CONFIG", "CKPT"))
    ap.add_argument("--dataset", choices=list(SOURCES), default="hoi4d",
                    help="which 2D hand source to sample (root / key cache / GT-axis availability)")
    ap.add_argument("--hoi4d-root", default=None, help="override the source's LMDB root")
    ap.add_argument("--hoi4d-config", default="config/hoi4d_v2_rgb_scalefree.yaml",
                    help="only its data.val_split_ratio / manual_seed are read (the scene split; 0.15 / 42 = "
                         "what the joint4 datamodule uses for every hand source)")
    ap.add_argument("--category", default="", help="key substring filter, e.g. C3 = HOI4D laptops; empty = all")
    ap.add_argument("--per-seq", type=int, default=1, help="max picks per sequence (key prefix)")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=2.0,
                    help="render the panels at this multiple of the 512-px frame (thin, sharp overlays; PNG output)")
    ap.add_argument("--ray-len", type=float, default=0.5,
                    help="metres of predicted axis drawn: trans = a ray of this length from the point, "
                         "rot = +/- this length about the hinge (drawn at the predicted depth z_p)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dcfg = yaml.safe_load(open(a.hoi4d_config))["data"]
    root, key_cache, has_gt_axis = SOURCES[a.dataset]
    root = a.hoi4d_root or root
    ds = SF3DDataset(
        lmdb_data_root=root, lmdb_path=f"{root}/data.lmdb",
        frame_cache_path=f"{root}/frames.lmdb", key_cache_path=key_cache,
        image_size_for_mask_reconstruction=(512, 512), return_trajectory_2d=True,
        point_source="element", fast_pipeline=True, load_depth=True,
        min_revolute_radius=0.0, min_mask_area_frac=0.0, edge_margin_frac=0.0,
        sensor_max_occluded_frac=0.5,
    )
    tr, va = split_dataset_by_scene(ds, dcfg.get("val_split_ratio", 0.15), dcfg.get("manual_seed", 42))
    pool = {"val": va.indices, "train": tr.indices, "all": list(range(len(ds)))}[a.split]
    tag = f"_{a.category}_" if a.category else ""
    cand = [i for i in pool if tag in ds.item_keys[i].decode()]
    if not cand:
        raise SystemExit(f"no {a.category or 'any'} records in split {a.split}")
    rng = np.random.default_rng(a.seed)
    # spread over sequences (key prefix = physical object/sequence): round-robin over a
    # shuffled sequence list, at most --per-seq picks per sequence
    by_seq = {}
    for i in cand:
        by_seq.setdefault(ds.item_keys[i].decode().split("/")[0], []).append(i)
    seqs = list(by_seq)
    rng.shuffle(seqs)
    picks = []
    for rnd in range(a.per_seq):
        for s in seqs:
            if len(picks) >= a.num:
                break
            lst = by_seq[s]
            if rnd < len(lst):
                j = int(rng.integers(len(lst)))
                picks.append(lst.pop(j))
    print(f"{len(cand)} {a.category or 'any'} records in {a.split} ({len(by_seq)} sequences); rendering {len(picks)}")

    models = [(n, *load_model(c, k, device)) for n, c, k in a.model]
    os.makedirs(a.out, exist_ok=True)
    for n_i, idx in enumerate(picks):
        it = ds[idx]
        (img_t, depth_t, desc, mask_t, _bbox, point_gt, mgt, tgt, img_size, fname,
         o3, K, _traj3d, traj2d_px, valid2d) = it
        gt_type = "rot" if int(tgt) == 1 else "trans"
        W, H = float(img_size[0]), float(img_size[1])
        frame = img_t.permute(1, 2, 0).numpy()[:, :, ::-1].copy()
        if a.scale != 1.0:
            frame = cv2.resize(frame, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
        S = frame.shape[0]
        gt_mask = cv2.resize(mask_t[0].numpy(), (S, S), interpolation=cv2.INTER_NEAREST)
        K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
        # GT panel
        gt = overlay_mask(frame, gt_mask, (0, 200, 0))
        tuv = (traj2d_px / torch.tensor([W, H])).numpy()
        gt = draw_polyline_norm(gt, tuv, valid2d.numpy(), (255, 255, 0), thickness=1)
        gt = draw_points_norm(gt, tuv, valid2d.numpy(), (255, 255, 0), radius=2)
        gp = (int(point_gt[0] * S), int(point_gt[1] * S))
        cv2.circle(gt, gp, 8, (255, 255, 255), 2, cv2.LINE_AA)
        gt_lines = [f"GT {a.dataset}: {gt_type}" + ("" if has_gt_axis else " (label only, no 3D axis)"), desc[:44]]
        if has_gt_axis:
            # ARCTIC: real hinge from the object model (green), drawn like the predictions
            dg = mgt.numpy().astype(np.float64)
            dg = dg / max(float(np.linalg.norm(dg)), 1e-8)
            og = o3.numpy().astype(np.float64)
            if gt_type == "rot":
                gt = draw_axis_3d(gt, K_norm, og, dg, og, (0, 200, 0), t0=-a.ray_len, t1=a.ray_len, thickness=2)
                gt_lines.append(f"axis=({dg[0]:+.2f},{dg[1]:+.2f},{dg[2]:+.2f})  z_o={og[2]:.2f}m")
            else:
                gt = draw_axis_3d(gt, K_norm, og, dg, og, (0, 200, 0), t0=0.0, t1=a.ray_len, thickness=2)
        gt = put_lines(gt, gt_lines)
        panels = [gt]
        for name, model, mp in models:
            with torch.no_grad():
                word = model.tokenize([desc], 77).to(device)
                out = model(img_t[None].to(device), depth_t[None].to(device), word,
                            None, None, None, None, K_norm.to(device).float())
                if getattr(mp, "trajectory_scale_free", False):
                    out = apply_trajectory_scale(out, trajectory_scale_factor("pred_z_p", out, None))
            p = frame.copy()
            pm = torch.sigmoid(out.mask_logits)[0, 0].cpu()
            pm = torch.nn.functional.interpolate(pm[None, None], size=(S, S), mode="bilinear")[0, 0].numpy()
            p = overlay_mask(p, (pm > 0.5).astype(np.float32), (0, 0, 230))
            lines = [name]
            anchor = out.point_3d_pred[0:1].cpu().float() if out.point_3d_pred is not None else None
            if out.point_uv is not None:
                pu = out.point_uv[0].cpu().float()
                cv2.circle(p, (int(pu[0] * S), int(pu[1] * S)), 8, (255, 255, 255), 2, cv2.LINE_AA)
            cls_type = "n/a"
            p_rev = float("nan")
            if out.motion_type_logits is not None:
                p_rev = float(torch.softmax(out.motion_type_logits[0].float(), -1)[1])
                cls_type = "rot" if p_rev > 0.5 else "trans"
            if anchor is not None and out.trajectory_pred is not None:
                traj_abs = anchor.unsqueeze(1) + out.trajectory_pred[0:1].cpu().float()
                tv = (traj_abs[0, :, 2] > 0.05).numpy()
                tuv_pred = project_points(K_norm, traj_abs)[0].clamp(-2, 3).numpy()
                p = draw_polyline_norm(p, tuv_pred, tv, (120, 255, 120), thickness=1)   # light green (BGR), distinct from the red axis
                p = draw_points_norm(p, tuv_pred, tv, (120, 255, 120), radius=2)
            if anchor is not None and out.motion_pred is not None:
                d = out.motion_pred[0].cpu().float().numpy()
                d = d / max(float(np.linalg.norm(d)), 1e-8)
                a3 = anchor[0].numpy()
                if cls_type == "rot" and out.origin_pred is not None:
                    q3 = out.origin_pred[0].cpu().float().numpy()
                    c3 = q3 + np.dot(a3 - q3, d) * d
                    r_vec = a3 - c3
                    if float(np.linalg.norm(r_vec)) > 1e-4:
                        th = np.linspace(0.0, math.pi / 2.0, 48)
                        arc = c3[None] + np.cos(th)[:, None] * r_vec[None] + np.sin(th)[:, None] * np.cross(d, r_vec)[None]
                        p = draw_polyline_norm(p, project_points(K_norm, torch.from_numpy(arc).float()[None])[0].clamp(-2, 3).numpy(),
                                               arc[:, 2] > 0.05, (0, 230, 230), thickness=2)
                    p = draw_axis_3d(p, K_norm, q3, d, a3, (0, 0, 255), t0=-a.ray_len, t1=a.ray_len, thickness=2)
                    lines.append(f"rot  p_rev={p_rev:.2f}  r={float(np.linalg.norm(r_vec)):.2f}m")
                else:
                    p = draw_axis_3d(p, K_norm, a3, d, a3, (0, 0, 255), t0=0.0, t1=a.ray_len, thickness=2)
                    lines.append(f"trans  p_rev={p_rev:.2f}")
                lines.append(f"axis=({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f})  z_p={float(anchor[0, 2]):.2f}m")
            if out.origin_uv is not None:
                ou = out.origin_uv[0].cpu().float()
                cv2.circle(p, (int(ou[0] * S), int(ou[1] * S)), 5, (0, 0, 255), 2, cv2.LINE_AA)
            panels.append(put_lines(p, lines))
        key = ds.item_keys[idx].decode().replace("/", "_")[:80]
        cv2.imwrite(f"{a.out}/{n_i:02d}_{a.dataset}_{gt_type}_{key}.png", np.hstack(panels))  # lossless
        print("wrote", n_i, key, "|", desc)
    write_manifest(a.out, models=[{"name": n, "config": c, "ckpt": k} for n, c, k in a.model])
    print("done ->", a.out)


if __name__ == "__main__":
    main()
