"""GT | prediction panels for HOI4D 2D checkpoints.

The SF3D vis tool's sampling filters skip HOI4D records silently, so
this is the bespoke equivalent: N stratified val samples (scene split
matched to training: ratio + seed from the config), one composite JPEG
per sample:

  left  (GT):   moving-part mask (green), wrist point (white ring),
                wrist track (cyan)
  right (pred): predicted mask (red), point_uv (white ring), projected
                predicted trajectory (magenta), text + p_rev readout

Run on a pod:
  /opt/venv/bin/python tools/hoi4d_vis_2d_panels.py \
      --config config/hoi4d_train_runpod_2d_dct.yaml \
      --ckpt experiments/20260901_hoi4d_2d_dct/checkpoints/<best>.ckpt \
      --out viz/<dated-batch> --num 12
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
from datasets.scenefun3d import SF3DDataset, split_dataset_by_scene
from model.losses.geometric import (backproject_points, normalized_intrinsics,
                                    project_points)
from model.segmenter import CRIS
from tools.viz_manifest import write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42421)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dcfg = cfg["data"]
    mp = ModelParams(**{k: v for k, v in cfg["model"]["model_params"].items()
                        if k in ModelParams.__dataclass_fields__})
    model = CRIS(mp).cuda().eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k[6:] if k.startswith("model.") else k: v
          for k, v in ck["state_dict"].items()}
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    root = dcfg["train_data_dir"]
    ds = SF3DDataset(
        lmdb_data_root=root, lmdb_path=f"{root}/data.lmdb",
        frame_cache_path=dcfg["frame_cache_path"],
        image_size_for_mask_reconstruction=(512, 512),
        return_trajectory_2d=True, point_source="element",
        fast_pipeline=True,
    )
    _, va = split_dataset_by_scene(
        ds, dcfg.get("val_split_ratio", 0.15), dcfg.get("manual_seed", 42))
    rng = np.random.default_rng(args.seed)
    # stratify by type (C4 trans / C6 rot) using the eval-only labels
    keys = [ds.item_keys[i].decode() for i in va.indices]
    rot_idx = [j for j, k in enumerate(keys) if "_C6_" in k]
    trans_idx = [j for j, k in enumerate(keys) if "_C4_" in k]
    picks = ([int(x) for x in rng.choice(trans_idx, args.num // 2, replace=False)]
             + [int(x) for x in rng.choice(rot_idx, args.num - args.num // 2,
                                           replace=False)])

    os.makedirs(args.out, exist_ok=True)
    for n, j in enumerate(picks):
        it = va[j]
        key = keys[j]
        img = it[0].permute(1, 2, 0).numpy()[:, :, ::-1].copy()  # BGR 512
        W0, H0 = it[8].tolist()
        sx, sy = 512.0 / W0, 512.0 / H0
        gt = img.copy()
        m = it[3][0].numpy() > 0.5
        gt[m] = gt[m] * 0.5 + np.array([0, 200, 0]) * 0.5
        tr2 = it[13].numpy()
        pts = np.stack([tr2[:, 0] * sx, tr2[:, 1] * sy], 1).astype(int)
        cv2.polylines(gt, [pts], False, (255, 255, 0), 2)
        px, py = int(it[5][0] * 512), int(it[5][1] * 512)
        cv2.circle(gt, (px, py), 7, (255, 255, 255), 2)
        cv2.putText(gt, f"GT  {it[2]}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        with torch.no_grad():
            word = model.tokenize([it[2]], 77).cuda()
            # keep uint8: CRIS.forward normalizes ONLY uint8 inputs
            # (segmenter.py:393) — a float cast here fed the model raw
            # 0-255 values and produced garbage panels on 2026-09-02.
            out = model(it[0].unsqueeze(0).cuda(),
                        it[1].unsqueeze(0).cuda(), word, None, None,
                        intrinsics_norm=it[11].unsqueeze(0).cuda())
        pr = img.copy()
        pm = torch.sigmoid(F.interpolate(out.mask_logits.float(), size=(512, 512),
                                         mode="bilinear"))[0, 0].cpu().numpy() > 0.5
        pr[pm] = pr[pm] * 0.5 + np.array([0, 0, 220]) * 0.5
        if out.point_uv is not None:
            qx, qy = (out.point_uv[0].detach().cpu().numpy() * 512).astype(int)
            cv2.circle(pr, (qx, qy), 7, (255, 255, 255), 2)
        if out.trajectory_pred is not None and out.point_uv is not None:
            # EXACTLY the trainer's projection (train_SF3D_better.py:625):
            # normalized K, anchor = point_uv lifted with the INPUT depth.
            # Any other convention renders garbage — the 2D arm's curve is
            # only defined under this projection (learned 2026-09-02).
            K_n = normalized_intrinsics(
                it[11].unsqueeze(0).cuda(), it[8].unsqueeze(0).cuda())
            uv = out.point_uv.detach().float()
            grid = (uv * 2.0 - 1.0).view(-1, 1, 1, 2)
            z = F.grid_sample(it[1].unsqueeze(0).cuda().float(), grid,
                              align_corners=False).view(-1)
            anchor = backproject_points(K_n, uv, z)
            curve = anchor.unsqueeze(1) + out.trajectory_pred.detach().float()
            proj = project_points(K_n, curve)[0].cpu().numpy()  # uv in [0,1]
            in_front = curve[0, :, 2].cpu().numpy() > 0.05
            if z.item() > 1e-3 and in_front.any():
                pts_p = (proj[in_front] * 512).astype(int)
                cv2.polylines(pr, [pts_p], False, (255, 0, 255), 2)
        p_rev = torch.softmax(out.motion_type_logits, -1)[0, 1].item() \
            if out.motion_type_logits is not None else float("nan")
        cv2.putText(pr, f"pred  p_rev={p_rev:.2f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        panel = np.concatenate([gt, pr], axis=1)
        cat = "rot" if "_C6_" in key else "trans"
        name = f"{n:02d}_{cat}_{key.split('/')[0]}.jpg"
        cv2.imwrite(os.path.join(args.out, name), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("wrote", name, flush=True)
    write_manifest(args.out, argv=sys.argv)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
