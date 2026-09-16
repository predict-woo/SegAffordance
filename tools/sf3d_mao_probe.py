"""OPD-style joint pass rates on the SF3D held-out split, per checkpoint.

OPD reports mAP under nested motion constraints: +M (type), +MA (type + axis within
10 deg), +MAO (type + axis + origin within 0.25 of the object's 3D bbox diagonal;
origin ignored for prismatic parts). SF3D has no per-object diagonal, so the origin
criterion here is ABSOLUTE: the distance in metres from the predicted hinge to the
annotated axis LINE (the trainer's origin_line_err) and to q*, the axis point nearest
the interaction point (the trainer's origin_err), at several thresholds. Everything
is one prediction per frame (no detection / IoU matching), like the paper's MA.

  python tools/sf3d_mao_probe.py --model dense CONFIG CKPT [--model ...] --out FILE.csv
  python tools/sf3d_mao_probe.py --summarize FILE.csv [--axis-deg 10] [--iou 0.5]   # recompute from the CSV, no GPU

The CSV holds one row per (model, frame) with everything the metrics need: key, GT and predicted
type, p_rev, signed and unsigned axis angle, origin errors (to the GT axis line and to q*), 3D point
error, 2D point error (image fraction), mask IoU, predicted z_p and lever radius.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.scenefun3d import SF3DDataset, get_default_transforms, split_dataset_by_scene  # noqa: E402
from model.losses.geometric import apply_trajectory_scale, normalized_intrinsics, trajectory_scale_factor  # noqa: E402
from sf3d_vis_predictions import load_model  # noqa: E402

THRESH_M = (0.05, 0.10, 0.25)
HEADER = "model,idx,key,gt_type,pred_type,p_rev,axis_signed_deg,axis_unsigned_deg,origin_line_err_m,origin_qstar_err_m,point3d_err_m,point2d_err_frac,mask_iou,z_p_m,radius_m"


def summarize(csv_path, axis_deg=10.0, iou_t=0.5):
    """Recompute M / MA / MAO (and their PDet-gated versions) from the per-sample CSV."""
    import csv
    by = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            by.setdefault(r["model"], []).append(r)
    print(f"{os.path.basename(csv_path)}: axis <= {axis_deg:.0f} deg (signed), PDet gate IoU >= {iou_t}")
    print(f"{'model':12s} {'n':>5s} {'PDet':>6s} {'M':>6s} {'MA':>6s}" + "".join(f" {'MAO_l' + str(t):>9s}" for t in THRESH_M)
          + "".join(f" {'MAO_q' + str(t):>9s}" for t in THRESH_M) + f" {'PDet+MA':>8s} {'PDet+MAO_l.25':>13s} {'MAO_rot_l.10':>12s}")
    for name, rs in by.items():
        g = np.array([int(r["gt_type"]) for r in rs]); p = np.array([int(r["pred_type"]) for r in rs])
        s_ = np.array([float(r["axis_signed_deg"]) for r in rs]); le = np.array([float(r["origin_line_err_m"]) for r in rs])
        qe = np.array([float(r["origin_qstar_err_m"]) for r in rs]); iou = np.array([float(r["mask_iou"]) for r in rs])
        M = g == p; MA = M & (s_ <= axis_deg); rot = g == 1; det = iou >= iou_t   # >= : the harness IoU is (inter+eps)/(union+eps), so an exact tie counts as detected
        def mao(err, t):
            return MA & np.where(rot, err <= t, True)
        pct = lambda x: 100.0 * float(np.mean(x))
        line = " ".join(f"{pct(mao(le, t)):9.2f}" for t in THRESH_M); qs = " ".join(f"{pct(mao(qe, t)):9.2f}" for t in THRESH_M)
        print(f"{name:12s} {len(rs):5d} {pct(det):6.2f} {pct(M):6.2f} {pct(MA):6.2f} {line} {qs} {pct(det & MA):8.2f} {pct(det & mao(le, 0.25)):13.2f} {pct(mao(le, 0.10)[rot]):12.2f}")
    print("MAO_l<t>: origin-to-GT-axis-line distance <= t m; MAO_q<t>: distance to q* <= t m; origin ignored for prismatic rows (OPD convention); MAO_rot: revolute rows only")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", default=None, metavar=("NAME", "CONFIG", "CKPT"))
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v3")
    ap.add_argument("--frame-cache-path", default="/workspace/datasets/sf3d_frames_512.lmdb")
    ap.add_argument("--key-cache", default="/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-depth", action="store_true", help="feed the frame caches' depth to the model (RGB-D checkpoints, e.g. the depth-input ablation); default zeros")
    ap.add_argument("--axis-deg", type=float, default=10.0)
    ap.add_argument("--iou", type=float, default=0.5, help="PDet threshold for the gated columns")
    ap.add_argument("--summarize", default=None, help="recompute the table from an existing CSV and exit")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize, a.axis_deg, a.iou)
        return
    if not a.model or not a.out:
        ap.error("--model and --out are required unless --summarize is given")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r, m, d = get_default_transforms(image_size=(512, 512))
    ds = SF3DDataset(
        lmdb_data_root=a.data_root, lmdb_path=f"{a.data_root}/data.lmdb", rgb_transform=r, mask_transform=m,
        depth_transform=d, image_size_for_mask_reconstruction=(512, 512), point_source="element",
        key_cache_path=a.key_cache, return_trajectory_2d=True, frame_cache_path=a.frame_cache_path,
        fast_pipeline=True, load_depth=a.load_depth, min_revolute_radius=0.10, min_mask_area_frac=0.001, edge_margin_frac=0.05,
    )
    _, va = split_dataset_by_scene(ds, 0.1, 42)
    idx = list(range(len(va)))
    if a.limit:
        idx = idx[: a.limit]
    models = [(n, *load_model(c, k, device)) for n, c, k in a.model]
    rows = {n: [] for n, _, _ in models}
    fout = open(a.out, "w"); fout.write(HEADER + "\n"); fout.flush()   # streamed: a killed run keeps its rows
    for j in idx:
        it = va[j]
        (img_t, depth_t, desc, mask_t, _bbox, pt_gt, motion_gt, type_gt, img_size, _fn, origin_3d, K, traj3d, _t2, _v2) = it
        key = va.dataset.item_keys[va.indices[j]].decode()
        gt_mask = mask_t[0].numpy() > 0.5
        K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
        gt_type = int(type_gt)
        n_gt = motion_gt.float().numpy(); n_gt = n_gt / max(float(np.linalg.norm(n_gt)), 1e-8)
        o_gt = origin_3d.float().numpy(); p0 = traj3d[0].float().numpy()
        q_star = o_gt + np.dot(p0 - o_gt, n_gt) * n_gt
        for name, model, mp in models:
            with torch.no_grad():
                word = model.tokenize([desc], 77).to(device)
                out = model(img_t[None].to(device), depth_t[None].to(device), word, None, None, None, None, K_norm.to(device).float())
                if getattr(mp, "trajectory_scale_free", False):
                    out = apply_trajectory_scale(out, trajectory_scale_factor("pred_z_p", out, None))
            p_rev = float(torch.softmax(out.motion_type_logits[0].float(), -1)[1])
            pred_type = 1 if p_rev >= 0.5 else 0
            dpred = out.motion_pred[0].cpu().float().numpy(); dpred = dpred / max(float(np.linalg.norm(dpred)), 1e-8)
            signed = math.degrees(math.acos(float(np.clip(np.dot(dpred, n_gt), -1, 1))))
            q = out.origin_pred[0].cpu().float().numpy() if out.origin_pred is not None else None
            if q is not None and gt_type == 1:
                rel = q - o_gt
                line_err = float(np.linalg.norm(rel - np.dot(rel, n_gt) * n_gt))
                qstar_err = float(np.linalg.norm(q - q_star))
            else:
                line_err = qstar_err = float("nan")
            unsigned = min(signed, 180.0 - signed)
            pm = torch.sigmoid(torch.nn.functional.interpolate(out.mask_logits.float(), size=(512, 512), mode="bilinear"))[0, 0].cpu().numpy() > 0.5
            inter = float(np.logical_and(pm, gt_mask).sum()); union = float(np.logical_or(pm, gt_mask).sum())
            iou = inter / union if union > 0 else 0.0
            pt_err = float(np.linalg.norm(out.point_uv[0].cpu().float().numpy() - pt_gt.float().numpy())) if out.point_uv is not None else float("nan")
            p3 = out.point_3d_pred[0].cpu().float().numpy() if out.point_3d_pred is not None else None
            p3_err = float(np.linalg.norm(p3 - p0)) if p3 is not None else float("nan")
            z_p = float(p3[2]) if p3 is not None else float("nan")
            if q is not None and p3 is not None:
                rel_p = p3 - q; radius = float(np.linalg.norm(rel_p - np.dot(rel_p, dpred) * dpred))
            else:
                radius = float("nan")
            r_ = (j, key, gt_type, pred_type, p_rev, signed, unsigned, line_err, qstar_err, p3_err, pt_err, iou, z_p, radius)
            rows[name].append(r_)
            fout.write(",".join([name, str(r_[0]), r_[1], str(r_[2]), str(r_[3])] + [f"{v:.4f}" for v in r_[4:]]) + "\n")
        if (j + 1) % 100 == 0:
            fout.flush()
        if (j + 1) % 500 == 0:
            print(f"  {j + 1}/{len(idx)}", flush=True)

    fout.close()
    summarize(a.out, a.axis_deg, a.iou)



if __name__ == "__main__":
    main()
