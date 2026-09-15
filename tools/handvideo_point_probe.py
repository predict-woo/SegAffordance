"""Hand-video interaction-point and trajectory-endpoint probe (Table III columns).

Runs checkpoints on the FULL test split of each 2D hand source (hoi4d / epic / arctic; same key cache,
ratio 0.15, seed 42 as the joint4 datamodule) and scores, per record, in normalised image coordinates
(u / W, v / H, so an error of 0.05 is 5 % of the image on each axis):

  point_err   |point_uv_pred - point_uv_gt|            predicted interaction point vs the hand's first position
  end_err     |proj(traj_pred)[-1] - track_gt[last]|   the decoded trajectory's end point (projected with the model's
                                                        own depth, z_p-scaled for scale-free heads) vs the track's
                                                        last valid point: extent, direction and shape in one number
  disp_err    |(end_pred - start_pred) - (end_gt - start_gt)|   the same with the start error removed
  gt_len      length of the GT track (sum of valid segments), for reference
  mask_iou    predicted vs GT part mask at 512 x 512, so the hand-video mIoU columns come from the same pass
  p_rev, type_ok

  python tools/handvideo_point_probe.py --model dense CONFIG CKPT [--model ...] --sources hoi4d epic arctic --out /root/x.csv
  python tools/handvideo_point_probe.py --summarize FILE.csv        # per model x source means, no GPU
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.scenefun3d import SF3DDataset, split_dataset_by_scene  # noqa: E402
from model.losses.geometric import apply_trajectory_scale, normalized_intrinsics, project_points, trajectory_scale_factor  # noqa: E402
from sf3d_vis_predictions import load_model  # noqa: E402

SOURCES = {
    "hoi4d": ("/workspace/datasets/hoi4d_processed_2d_v2", "/workspace/cache/hoi4d_2d_keys_v2.pkl", "config/hoi4d_v2_rgb_scalefree.yaml"),
    "epic": ("/workspace/datasets/epic_processed_2d", "/workspace/cache/epic_2d_keys_v1.pkl", "config/epic_v1_rgb_scalefree.yaml"),
    "arctic": ("/workspace/datasets/arctic_processed_2d", "/workspace/cache/arctic_2d_keys_v1.pkl", "config/arctic_v1_rgb_scalefree.yaml"),
}
HEADER = "model,source,key,gt_type,p_rev,type_ok,point_err,end_err,disp_err,gt_len,mask_iou"


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    models = sorted({r["model"] for r in rows}); sources = [s for s in SOURCES if any(r["source"] == s for r in rows)]
    print(f"{'model':12s} {'source':7s} {'n':>4s} {'point':>7s} {'end':>7s} {'disp':>7s} {'gt_len':>7s} {'mIoU':>6s} {'type%':>6s}")
    for m in models:
        for s in sources:
            rs = [r for r in rows if r["model"] == m and r["source"] == s]
            if not rs:
                continue
            f = lambda k: np.array([float(r[k]) for r in rs])
            print(f"{m:12s} {s:7s} {len(rs):4d} {np.nanmean(f('point_err')):7.3f} {np.nanmean(f('end_err')):7.3f} {np.nanmean(f('disp_err')):7.3f} "
                  f"{np.nanmean(f('gt_len')):7.3f} {f('mask_iou').mean():6.3f} {100 * f('type_ok').mean():6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", metavar=("NAME", "CONFIG", "CKPT"))
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize); return
    if not a.model or not a.out:
        ap.error("--model and --out are required")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [(n, *load_model(c, k, device)) for n, c, k in a.model]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w"); fout.write(HEADER + "\n"); fout.flush()
    for src in a.sources:
        root, keys, cfg = SOURCES[src]
        dcfg = yaml.safe_load(open(cfg))["data"]
        ds = SF3DDataset(
            lmdb_data_root=root, lmdb_path=f"{root}/data.lmdb", frame_cache_path=f"{root}/frames.lmdb", key_cache_path=keys,
            image_size_for_mask_reconstruction=(512, 512), return_trajectory_2d=True, point_source="element", fast_pipeline=True,
            load_depth=False, min_revolute_radius=0.0, min_mask_area_frac=0.0, edge_margin_frac=0.0, sensor_max_occluded_frac=0.5,
        )
        _, va = split_dataset_by_scene(ds, dcfg.get("val_split_ratio", 0.15), dcfg.get("manual_seed", 42))
        idxs = list(va.indices)[: a.limit or None]
        print(f"{src}: {len(idxs)} test records", flush=True)
        for j, idx in enumerate(idxs):
            (img_t, depth_t, desc, mask_t, _b, point_gt, _mgt, tgt, img_size, _fn, _o3, K, _t3, traj2d_px, valid2d) = ds[idx]
            key = ds.item_keys[idx].decode()
            W, H = float(img_size[0]), float(img_size[1])
            K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
            tuv = (traj2d_px.float() / torch.tensor([W, H])).numpy(); valid = valid2d.numpy().astype(bool)
            vidx = np.flatnonzero(valid)
            pg = point_gt.float().numpy()
            gt_mask = mask_t[0].numpy() > 0.5
            if len(vidx) >= 2:
                g_start, g_end = tuv[vidx[0]], tuv[vidx[-1]]
                seg = np.diff(tuv[vidx], axis=0); gt_len = float(np.linalg.norm(seg, axis=1).sum())
            else:
                g_start = g_end = None; gt_len = float("nan")
            for name, model, mp in models:
                with torch.no_grad():
                    word = model.tokenize([desc], 77).to(device)
                    out = model(img_t[None].to(device), depth_t[None].to(device), word, None, None, None, None, K_norm.to(device).float())
                    if getattr(mp, "trajectory_scale_free", False):
                        out = apply_trajectory_scale(out, trajectory_scale_factor("pred_z_p", out, None))
                p_rev = float(torch.softmax(out.motion_type_logits[0].float(), -1)[1])
                type_ok = int((p_rev > 0.5) == (int(tgt) == 1))
                pu = out.point_uv[0].cpu().float().numpy()
                point_err = float(np.linalg.norm(pu - pg))
                end_err = disp_err = float("nan")
                if g_end is not None and out.point_3d_pred is not None and out.trajectory_pred is not None:
                    anchor = out.point_3d_pred[0:1].cpu().float()
                    traj_abs = anchor.unsqueeze(1) + out.trajectory_pred[0:1].cpu().float()
                    tp = project_points(K_norm, traj_abs)[0].numpy()
                    end_err = float(np.linalg.norm(tp[-1] - g_end))
                    disp_err = float(np.linalg.norm((tp[-1] - tp[0]) - (g_end - g_start)))
                pm = torch.sigmoid(out.mask_logits)[0, 0].float().cpu()
                pm = torch.nn.functional.interpolate(pm[None, None], size=gt_mask.shape, mode="bilinear")[0, 0].numpy() > 0.5
                inter = float((pm & gt_mask).sum()); union = float((pm | gt_mask).sum())
                iou = inter / union if union > 0 else 0.0
                fout.write(",".join([name, src, key, str(int(tgt)), f"{p_rev:.4f}", str(type_ok), f"{point_err:.5f}", f"{end_err:.5f}",
                                     f"{disp_err:.5f}", f"{gt_len:.5f}", f"{iou:.4f}"]) + "\n")
            if j % 100 == 0:
                fout.flush(); print(f"  {j}/{len(idxs)}", flush=True)
    fout.close()
    summarize(a.out)


if __name__ == "__main__":
    main()
