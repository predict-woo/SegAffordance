"""Paper Fig. 4 panels: SF3D validation frames as GT | baseline (oracle-matched instance) | ours.

The baseline column reads a baselines-session `preds.jsonl` (one record per test key:
matched, score, mask_rle at native (H, W), type, axis_cam, origin_cam in the frame's
camera, metres) and draws the oracle-matched instance the way the scorer scored it:
mask + axis line through its origin (revolute) or direction ray from the GT point
(prismatic). Ours is drawn by predict_image.draw_prediction, as in sf3d_vis_val.py.

  python tools/sf3d_vis_baseline_vs_ours.py --model dense CONFIG CKPT \
      --baseline OPDFormer-C /workspace/datasets/baselines/results/opd_c_rgbd/preds.jsonl \
      --out viz/<batch> [--num 12 --seed 3 | --val-idx 1696 1773 ...]
"""
import argparse
import json
import math
import os
import pickle
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.scenefun3d import SF3DDataset, get_default_transforms, split_dataset_by_scene  # noqa: E402
from model.losses.geometric import apply_trajectory_scale, normalized_intrinsics, trajectory_scale_factor  # noqa: E402
from predict_image import draw_prediction  # noqa: E402
from sf3d_vis_predictions import draw_axis_3d, draw_points_norm, draw_polyline_norm, load_model, overlay_mask, put_lines  # noqa: E402
from viz_manifest import write_manifest  # noqa: E402


def load_preds(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["key"]] = r
    return out


def baseline_panel(frame, name, rec, K_norm, S, p0, ray_len):
    """Draw the oracle-matched instance; grey note when the detector produced nothing overlapping."""
    from pycocotools import mask as mask_utils
    panel = frame.copy()
    if not rec or not rec.get("matched"):
        return put_lines(panel, [name, "no overlapping instance"]), None, None
    m = mask_utils.decode(rec["mask_rle"]).astype(np.uint8)
    m = cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)
    panel = overlay_mask(panel, m, (0, 0, 220))
    t = int(rec["type"]); d = np.asarray(rec["axis_cam"], dtype=np.float64); d /= max(np.linalg.norm(d), 1e-8)
    if t == 1 and rec.get("origin_cam") is not None:
        o = np.asarray(rec["origin_cam"], dtype=np.float64)
        panel = draw_axis_3d(panel, K_norm, o, d, p0, (0, 0, 220), t0=-ray_len, t1=ray_len, thickness=3)
    else:
        panel = draw_axis_3d(panel, K_norm, p0, d, p0, (0, 0, 220), t0=0.0, t1=ray_len, thickness=3)
    lines = [name, f"{'rot' if t == 1 else 'trans'}  score={rec.get('score', 0) or 0:.2f}"]
    return put_lines(panel, lines), t, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", default=[], metavar=("NAME", "CONFIG", "CKPT"))
    ap.add_argument("--baseline", nargs=2, action="append", default=[], metavar=("NAME", "PREDS_JSONL"))
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v3")
    ap.add_argument("--frame-cache-path", default="/workspace/datasets/sf3d_frames_512.lmdb")
    ap.add_argument("--key-cache", default="/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--val-idx", type=int, nargs="*", default=None, help="explicit val indices (as printed by sf3d_vis_val.py)")
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
    if a.val_idx:
        picks = list(a.val_idx)
    else:
        rng = np.random.default_rng(a.seed)
        types = []
        with ds.env.begin() as t:
            for i in va.indices:
                rec = pickle.loads(t.get(ds.item_keys[i]))
                types.append(rec["motion_info"]["original_motion_data"].get("motion_type", "trans"))
        rot = [j for j, ty in enumerate(types) if ty in ("rot", "rotation")]
        trans = [j for j, ty in enumerate(types) if ty not in ("rot", "rotation")]
        picks = [int(x) for x in rng.choice(trans, a.num // 2, replace=False)] + [int(x) for x in rng.choice(rot, a.num - a.num // 2, replace=False)]
    models = [(n, *load_model(c, k, device)) for n, c, k in a.model]
    baselines = [(n, load_preds(p)) for n, p in a.baseline]
    os.makedirs(a.out, exist_ok=True)
    for n_i, j in enumerate(picks):
        it = va[j]
        key = ds.item_keys[va.indices[j]].decode()
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
        p0 = traj3d[0].float().numpy()
        gt = overlay_mask(frame, cv2.resize(mask_t[0].numpy(), (S, S), interpolation=cv2.INTER_NEAREST), (0, 200, 0))
        tuv = (traj2d_px / torch.tensor([W, H])).numpy()
        gt = draw_polyline_norm(gt, tuv, valid2d.numpy(), (255, 255, 0), thickness=2)
        gt = draw_points_norm(gt, tuv, valid2d.numpy(), (255, 255, 0), radius=3)
        if gt_type == 1:
            gt = draw_axis_3d(gt, K_norm, origin_3d.float().numpy(), gt_dir, p0, (0, 200, 0), t0=-a.ray_len, t1=a.ray_len, thickness=3)
        else:
            gt = draw_axis_3d(gt, K_norm, p0, gt_dir, p0, (0, 200, 0), t0=0.0, t1=a.ray_len, thickness=3)
        cv2.circle(gt, (int(point_gt[0] * S), int(point_gt[1] * S)), 10, (255, 255, 255), 2, cv2.LINE_AA)
        gt = put_lines(gt, [f"GT [{'rot' if gt_type == 1 else 'trans'}]  val {j}", desc[:52]])
        panels = [gt]
        for name, preds in baselines:
            panel, bt, bd = baseline_panel(frame, name, preds.get(key), K_norm, S, p0, a.ray_len)
            if bd is not None:
                ang = math.degrees(math.acos(float(np.clip(np.dot(bd, gt_dir), -1.0, 1.0))))
                ok = "type OK" if bt == gt_type else "type WRONG"
                cv2.putText(panel, f"axis err {ang:.0f} deg  {ok}", (8, 26 * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(panel, f"axis err {ang:.0f} deg  {ok}", (8, 26 * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)
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
        print("wrote", n_i, j, key, "|", desc[:60])
    write_manifest(a.out, models=[{"name": n, "config": c, "ckpt": k} for n, c, k in a.model],
                   baselines=[{"name": n, "preds": p} for n, p in a.baseline], picks=[int(x) for x in picks])
    print("done ->", a.out)


if __name__ == "__main__":
    main()
