"""SF3D validation panels in the sharp style: [GT | model A | model B ...] at
2x resolution with thin overlays, PNG.

GT panel: GT mask (green), GT 2D track (cyan), GT interaction point (white
ring), GT axis (green line: hinge line through the GT origin for rot, a
direction ray from the track's first point for trans). Prediction panels:
tools/predict_image.py's draw_prediction (mask red, point ring, trajectory
light green, axis red, orbit yellow) + the axis error vs GT in the text.

  python tools/sf3d_vis_val.py --model joint4_dct <cfg> <ckpt> [--model ...] \
      --out viz/<batch> --num 16 [--seed 7] [--scale 2]
Sampling: stratified rot/trans over the g19 val split (same filters and
seed as the SF3D configs).
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from datasets.scenefun3d import SF3DDataset, get_default_transforms, split_dataset_by_scene  # noqa: E402
from model.losses.geometric import apply_trajectory_scale, normalized_intrinsics, trajectory_scale_factor  # noqa: E402
from predict_image import draw_prediction  # noqa: E402
from sf3d_vis_predictions import draw_axis_3d, draw_points_norm, draw_polyline_norm, load_model, overlay_mask, put_lines  # noqa: E402
from viz_manifest import write_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", required=True, metavar=("NAME", "CONFIG", "CKPT"))
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v3")
    ap.add_argument("--frame-cache-path", default="/workspace/datasets/sf3d_frames_512.lmdb")
    ap.add_argument("--key-cache", default="/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--ray-len", type=float, default=0.5)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r, m, d = get_default_transforms(image_size=(512, 512))
    ds = SF3DDataset(
        lmdb_data_root=a.data_root, lmdb_path=f"{a.data_root}/data.lmdb", rgb_transform=r, mask_transform=m,
        depth_transform=d, image_size_for_mask_reconstruction=(512, 512), point_source="element",
        key_cache_path=a.key_cache, return_trajectory_2d=True, frame_cache_path=a.frame_cache_path,
        fast_pipeline=True, load_depth=False, min_revolute_radius=0.10, min_mask_area_frac=0.001, edge_margin_frac=0.05,
    )
    _, va = split_dataset_by_scene(ds, 0.1, 42)
    rng = np.random.default_rng(a.seed)
    # stratify by GT type without loading every sample: read the records
    import lmdb, pickle
    types = []
    with ds.env.begin() as t:
        for i in va.indices:
            rec = pickle.loads(t.get(ds.item_keys[i]))
            types.append(rec["motion_info"]["original_motion_data"].get("motion_type", "trans"))
    rot = [j for j, ty in enumerate(types) if ty in ("rot", "rotation")]
    trans = [j for j, ty in enumerate(types) if ty not in ("rot", "rotation")]
    picks = [int(x) for x in rng.choice(trans, a.num // 2, replace=False)] + [int(x) for x in rng.choice(rot, a.num - a.num // 2, replace=False)]
    models = [(n, *load_model(c, k, device)) for n, c, k in a.model]
    os.makedirs(a.out, exist_ok=True)
    for n_i, j in enumerate(picks):
        it = va[j]
        (img_t, depth_t, desc, mask_t, _bbox, point_gt, motion_gt, type_gt, img_size, fname,
         origin_3d, K, traj3d, traj2d_px, valid2d) = it
        W, H = float(img_size[0]), float(img_size[1])
        frame = img_t.permute(1, 2, 0).numpy()[:, :, ::-1].copy()
        if a.scale != 1.0:
            frame = cv2.resize(frame, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
        S = frame.shape[0]
        K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
        gt_type = int(type_gt)
        gt_dir = motion_gt.float().numpy(); gt_dir = gt_dir / max(float(np.linalg.norm(gt_dir)), 1e-8)
        # GT panel
        gt = overlay_mask(frame, cv2.resize(mask_t[0].numpy(), (S, S), interpolation=cv2.INTER_NEAREST), (0, 200, 0))
        tuv = (traj2d_px / torch.tensor([W, H])).numpy()
        gt = draw_polyline_norm(gt, tuv, valid2d.numpy(), (255, 255, 0), thickness=2)
        gt = draw_points_norm(gt, tuv, valid2d.numpy(), (255, 255, 0), radius=3)
        p0 = traj3d[0].float().numpy()
        if gt_type == 1:
            gt = draw_axis_3d(gt, K_norm, origin_3d.float().numpy(), gt_dir, p0, (0, 200, 0), t0=-a.ray_len, t1=a.ray_len, thickness=3)
        else:
            gt = draw_axis_3d(gt, K_norm, p0, gt_dir, p0, (0, 200, 0), t0=0.0, t1=a.ray_len, thickness=3)
        cv2.circle(gt, (int(point_gt[0] * S), int(point_gt[1] * S)), 10, (255, 255, 255), 2, cv2.LINE_AA)
        gt = put_lines(gt, [f"GT [{'rot' if gt_type == 1 else 'trans'}]  val {j}", desc[:52]])
        panels = [gt]
        for name, model, mp in models:
            with torch.no_grad():
                word = model.tokenize([desc], 77).to(device)
                out = model(img_t[None].to(device), depth_t[None].to(device), word, None, None, None, None, K_norm.to(device).float())
                if getattr(mp, "trajectory_scale_free", False):
                    out = apply_trajectory_scale(out, trajectory_scale_factor("pred_z_p", out, None))
            panel, cls_type, dpred = draw_prediction(frame, out, K_norm, name, a.ray_len)
            if dpred is not None:
                ang = math.degrees(math.acos(float(np.clip(np.dot(dpred, gt_dir), -1.0, 1.0))))
                ok = "type OK" if (cls_type == ("rot" if gt_type == 1 else "trans")) else "type WRONG"
                cv2.putText(panel, f"axis err {ang:.0f} deg  {ok}", (8, 26 * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(panel, f"axis err {ang:.0f} deg  {ok}", (8, 26 * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)
        cv2.imwrite(f"{a.out}/{n_i:02d}_{'rot' if gt_type == 1 else 'trans'}_val{j}.png", np.hstack(panels))
        print("wrote", n_i, j, "|", desc[:60])
    write_manifest(a.out, models=[{"name": n, "config": c, "ckpt": k} for n, c, k in a.model])
    print("done ->", a.out)


if __name__ == "__main__":
    main()
