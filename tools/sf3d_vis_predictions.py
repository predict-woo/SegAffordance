"""Side-by-side prediction panels for SF3D twist-arm checkpoints.

For N deterministic validation samples (stratified rot/trans), renders one
composite JPEG per sample:

    [ GT | model A pred | model B pred ... ]

GT panel: mask (green), element point, projected GT trajectory (cyan).
Pred panels: predicted mask (red), predicted point (coords_hat), the decoded
twist's ORBIT through the model's own anchor projected into the frame
(yellow) — an arc for revolute, a straight line for prismatic — plus text:
classifier type, type-from-|omega|, |omega|.

Inference is deployment-condition: CVAE prior sampling (motion_gt=None) and
NO type hint (motion_type_input=None -> NULL token).

Run from the repo root on a pod:
    python tools/sf3d_vis_predictions.py \
        --model twist experiments/20260728_sf3d_twist/config.yaml \
                experiments/20260728_sf3d_twist/checkpoints/best-epoch04-valloss0.9891.ckpt \
        --model 2d_twist experiments/20260728_sf3d_2d_twist/config.yaml \
                experiments/20260728_sf3d_2d_twist/checkpoints/best-epoch15-valloss1.0906.ckpt \
        --out sf3d_viz/pred_compare --num 16
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.opd_train import ModelParams
from datasets.scenefun3d import SF3DDataset, get_default_transforms, split_dataset_by_scene
from model.losses.geometric import backproject_points, normalized_intrinsics, project_points
from model.losses.twist import decode_twist, screw_orbit
from model.segmenter import CRIS

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
PANEL_W = 640  # each panel is the frame resized to this width


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


def overlay_mask(img, mask01, color, alpha=0.45):
    layer = np.zeros_like(img)
    layer[mask01 > 0.5] = color
    return cv2.addWeighted(img, 1.0, layer, alpha, 0)


def put_lines(img, lines, color=(255, 255, 255)):
    y = 26
    for line in lines:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 26
    return img


def draw_polyline_norm(img, uv_norm, valid, color, thickness=3):
    """uv_norm: (M, 2) in [0,1]-ish; draws only consecutive valid pairs."""
    h, w = img.shape[:2]
    pts = np.stack([uv_norm[:, 0] * w, uv_norm[:, 1] * h], axis=1)
    for i in range(len(pts) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        p0 = tuple(np.round(pts[i]).astype(int))
        p1 = tuple(np.round(pts[i + 1]).astype(int))
        if all(-w < c < 2 * w for c in (p0[0], p1[0])) and all(-h < c < 2 * h for c in (p0[1], p1[1])):
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", required=True,
                    metavar=("NAME", "CONFIG", "CKPT"))
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v2")
    ap.add_argument("--key-cache", default="/workspace/cache/sf3d_v2_keys_cutoff05.pkl")
    ap.add_argument("--out", default="sf3d_viz/pred_compare")
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for name, cfg, ckpt in args.model:
        print(f"loading {name}: {ckpt}")
        models.append((name, load_model(cfg, ckpt, device)))

    r, m, d = get_default_transforms((256, 256))
    ds = SF3DDataset(
        lmdb_data_root=args.data_root,
        key_cache_path=args.key_cache,
        frame_cache_path=os.path.join(args.data_root, "frames.lmdb"),
        rgb_transform=r, mask_transform=m, depth_transform=d,
        image_size_for_mask_reconstruction=(256, 256),
        point_source="element", return_trajectory_2d=True,
    )
    # The SAME val split as training (seed 42, ratio 0.1, split by scene).
    _, val = split_dataset_by_scene(ds, val_split_ratio=0.1, manual_seed=42)

    # Stratify: half rot, half trans, deterministic.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(val))
    picks, want = [], {0: args.num // 2, 1: args.num - args.num // 2}
    for i in order:
        s = val[int(i)]
        t = int(s[7])
        if want[t] > 0:
            picks.append((int(i), s))
            want[t] -= 1
        if not want[0] and not want[1]:
            break

    os.makedirs(args.out, exist_ok=True)
    for rank, (vi, s) in enumerate(picks):
        (img_t, depth_t, desc, mask_t, _bbox, point_gt, motion_gt, type_gt,
         img_size, rgb_name, origin_3d, K, traj3d, traj2d_px, traj2d_valid) = s
        w0, h0 = int(img_size[0].item()), int(img_size[1].item())
        frame = cv2.imread(os.path.join(args.data_root, "images", rgb_name))
        if frame is None:
            continue
        scale = PANEL_W / frame.shape[1]
        frame = cv2.resize(frame, (PANEL_W, int(frame.shape[0] * scale)))
        H, W = frame.shape[:2]

        # ---- GT panel
        gt = frame.copy()
        gt_mask = F.interpolate(mask_t[None].float(), size=(H, W), mode="nearest")[0, 0].numpy()
        gt = overlay_mask(gt, gt_mask, (0, 200, 0))
        traj_uv = (traj2d_px / img_size[None, :]).numpy()
        gt = draw_polyline_norm(gt, traj_uv, traj2d_valid.numpy(), (255, 220, 0))
        gp = (int(point_gt[0] * W), int(point_gt[1] * H))
        cv2.circle(gt, gp, 6, (255, 255, 255), -1)
        cv2.circle(gt, gp, 6, (0, 0, 0), 2)
        gt = put_lines(gt, [f"GT [{ 'rot' if int(type_gt) else 'trans' }]", desc[:52]])

        panels = [gt]
        for name, model in models:
            with torch.no_grad():
                word = model.tokenize([desc], 77).to(device)
                out = model(img_t[None].to(device), depth_t[None].to(device),
                            word, None, None, None, None)
            p = frame.copy()
            pm = torch.sigmoid(out.mask_logits)[0, 0].cpu()
            pm = F.interpolate(pm[None, None], size=(H, W), mode="bilinear")[0, 0].numpy()
            p = overlay_mask(p, (pm > 0.5).astype(np.float32), (0, 0, 230))
            coords = out.coords_hat[0].cpu()
            pp = (int(coords[0] * W), int(coords[1] * H))
            cv2.circle(p, pp, 6, (255, 255, 255), -1)
            cv2.circle(p, pp, 6, (0, 0, 230), 2)

            lines = [name]
            cls_type = "rot" if int(out.motion_type_logits[0].argmax()) == 1 else "trans"
            if out.twist_pred is not None:
                tw = out.twist_pred[0].cpu().float()
                is_rev, direction, _ = decode_twist(tw[None])
                om = tw[:3].norm().item()
                # anchor: own point lifted with the input depth (as the loss does)
                K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
                grid = (coords[None] * 2.0 - 1.0).view(1, 1, 1, 2)
                z = F.grid_sample(depth_t[None].float(), grid, align_corners=False).view(1)
                anchor = backproject_points(K_norm, coords[None].float(), z)
                ts = torch.linspace(-math.pi, math.pi, 96)[None] / max(om, 1.0)
                orbit = screw_orbit(tw[None], anchor, ts)[0]
                ov = (orbit[:, 2] > 0.05).numpy()
                ouv = project_points(K_norm, orbit[None])[0].clamp(-2, 3).numpy()
                p = draw_polyline_norm(p, ouv, ov, (0, 230, 230))
                lines.append(f"cls={cls_type}  |w|={om:.2f} -> {'rot' if bool(is_rev[0]) else 'trans'}")
            else:
                lines.append(f"cls={cls_type}")
            p = put_lines(p, lines)
            panels.append(p)

        combo = np.concatenate(panels, axis=1)
        t_lbl = "rot" if int(type_gt) else "trans"
        cv2.imwrite(os.path.join(args.out, f"{rank:02d}_{t_lbl}_val{vi}.jpg"),
                    combo, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"wrote {rank:02d}_{t_lbl}_val{vi}.jpg  ({desc[:48]})")

    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
