"""Standalone checkpoint visualizer for OPDReal/OPDMulti.

Loads a Lightning checkpoint (state_dict with "model." prefix), runs pure
inference (motion_gt=None -> CVAE prior sampling) on N deterministic samples
of a split, and writes one composite PNG per sample:
    [ GT: mask+point+axis | Pred mask overlay | Pred point heatmap ]
plus a summary.txt with per-sample IoU / type / axis error and aggregates.

Run from the repo root on a pod:
    python tools/vis_predictions.py \
        --config config/opdreal_train_runpod.yaml \
        --checkpoint /workspace/checkpoints/OPDReal_RUNPOD/last.ckpt \
        --out /workspace/vis_out/opdreal --num 24
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.opd_train import ModelParams
from datasets.opdreal import OPDRealDataset, get_default_transforms
from model.segmenter import CRIS
from utils.dataset import tokenize

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denorm_image(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(STD * img + MEAN, 0, 1)
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def overlay_mask(img_bgr, mask01, color, alpha=0.45):
    out = img_bgr.copy()
    layer = np.zeros_like(out)
    layer[mask01 > 0.5] = color
    return cv2.addWeighted(out, 1.0, layer, alpha, 0)


def draw_point_axis(img, px, py, axis3d, color, length=45):
    cv2.circle(img, (px, py), 5, color, -1)
    cv2.circle(img, (px, py), 5, (0, 0, 0), 1)
    xy = np.array(axis3d[:2], dtype=np.float64)
    n = np.linalg.norm(xy)
    if n > 1e-6:
        xy = xy / n
        end = (int(px + xy[0] * length), int(py + xy[1] * length))
        cv2.arrowedLine(img, (px, py), end, color, 2, tipLength=0.3)
    return img


def put_label(img, text, y=18, color=(255, 255, 255)):
    cv2.rectangle(img, (0, y - 15), (img.shape[1], y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return img


def axis_error_deg(pred, gt):
    p = pred / (np.linalg.norm(pred) + 1e-8)
    g = gt / (np.linalg.norm(gt) + 1e-8)
    c = np.clip(abs(float(np.dot(p, g))), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=24)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mp = ModelParams(**cfg["model"]["model_params"])
    data_cfg = cfg["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRIS(mp)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: missing={missing} unexpected={unexpected}")
    model.to(device).eval()

    rgb_t, mask_t, depth_t = get_default_transforms(
        image_size=tuple(data_cfg["input_size"]), is_train=False
    )
    ds = OPDRealDataset(
        data_path=data_cfg["data_path"],
        dataset_key=f"MotionNet_{args.split}",
        rgb_transform=rgb_t,
        mask_transform=mask_t,
        depth_transform=depth_t,
        is_multi=data_cfg.get("is_multi", False),
        use_depth=data_cfg.get("use_depth", True),
    )
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(ds), size=min(args.num, len(ds)), replace=False)

    os.makedirs(args.out, exist_ok=True)
    ious, axerrs, type_hits = [], [], 0
    lines = []
    type_names = {0: "trans", 1: "rot"}

    for rank, idx in enumerate(sorted(indices.tolist())):
        item = ds[idx]
        (img, depth, desc, mask_gt, _bbox, point_gt, motion_gt, mtype_gt, _size) = item[:9]
        tok = tokenize([desc], mp.word_len, truncate=True).to(device)
        with torch.no_grad():
            out = model(
                img.unsqueeze(0).to(device),
                depth.unsqueeze(0).to(device),
                tok,
                None,
                None,
                None,
            )
        mask_logits, point_logits, coords_hat, motion_pred, type_logits = out[:5]

        H, W = img.shape[-2:]
        mask_prob = torch.sigmoid(mask_logits)[0, 0].cpu()
        mask_up = (
            F.interpolate(mask_prob[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        )
        point_prob = torch.sigmoid(point_logits)[0, 0].cpu().numpy()

        pred_bin = (mask_up > 0.5).float()
        gt_bin = mask_gt[0].float()
        inter = float((pred_bin * gt_bin).sum())
        union = float(((pred_bin + gt_bin) > 0).float().sum())
        iou = inter / union if union > 0 else 0.0
        ious.append(iou)

        pred_type = int(type_logits.argmax(dim=1).item())
        gt_type = int(mtype_gt.item())
        type_ok = pred_type == gt_type
        type_hits += int(type_ok)

        axis_pred = motion_pred[0].cpu().numpy()
        axis_gt = motion_gt.numpy()
        aerr = axis_error_deg(axis_pred, axis_gt)
        axerrs.append(aerr)

        img_bgr = denorm_image(img)

        gt_panel = overlay_mask(img_bgr, gt_bin.numpy(), (0, 200, 0))
        gpx, gpy = int(point_gt[0] * W), int(point_gt[1] * H)
        gt_panel = draw_point_axis(gt_panel, gpx, gpy, axis_gt, (0, 255, 255))
        gt_panel = put_label(gt_panel, f"GT {type_names[gt_type]} | {desc[:44]}")

        pred_panel = overlay_mask(img_bgr, mask_up.numpy(), (0, 80, 255))
        ppx, ppy = int(coords_hat[0, 0] * W), int(coords_hat[0, 1] * H)
        pred_panel = draw_point_axis(pred_panel, ppx, ppy, axis_pred, (255, 200, 0))
        pred_panel = put_label(
            pred_panel,
            f"PRED {type_names[pred_type]}{'' if type_ok else ' (X)'} IoU {iou:.2f} ax {aerr:.0f}d",
        )

        hm = cv2.resize(point_prob, (W, H), interpolation=cv2.INTER_LINEAR)
        hm = cv2.applyColorMap((255 - hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm_panel = cv2.addWeighted(img_bgr, 0.55, hm, 0.45, 0)
        hm_panel = put_label(hm_panel, "pred point heatmap")

        panel = np.concatenate([gt_panel, pred_panel, hm_panel], axis=1)
        name = f"{rank:02d}_idx{idx}_iou{iou:.2f}.png"
        cv2.imwrite(os.path.join(args.out, name), panel)
        lines.append(
            f"{name}  iou={iou:.3f} type_gt={type_names[gt_type]} type_pred={type_names[pred_type]}"
            f" ax_err={aerr:.1f}deg  desc={desc}"
        )

    n = len(ious)
    summary = [
        f"checkpoint: {args.checkpoint}",
        f"split: {args.split}  samples: {n}",
        f"mean IoU: {np.mean(ious):.3f}   IoU>0.5: {sum(i > 0.5 for i in ious)}/{n}",
        f"type acc: {type_hits}/{n}",
        f"mean axis err: {np.mean(axerrs):.1f} deg  (rot-only: "
        f"{np.mean([a for a, l in zip(axerrs, lines) if 'type_gt=rot' in l] or [0]):.1f} deg)",
        "",
    ] + lines
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary[:5]))


if __name__ == "__main__":
    main()
