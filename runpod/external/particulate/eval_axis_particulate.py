"""Axis/origin metrics for Particulate predictions (paper reports none).
Parts matched pred<->GT by Hungarian assignment on part centroids (as their
evaluate.py does). Plücker (l, m) with m = p x l  ->  p = l x m / |l|^2.
  python eval_axis_particulate.py --gt_dir GT --result_dir RES --out metrics_axis.json
"""
import os, sys, json, argparse, glob
import numpy as np, trimesh
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screw_loss import axis_metrics


def plucker_point(pl):
    l, m = pl[:3], pl[3:6]
    return np.cross(l, m) / (np.dot(l, l) + 1e-12)


def centroids(points, part_ids, k):
    return np.stack([points[part_ids == i].mean(0) if (part_ids == i).any() else np.full(3, np.nan) for i in range(k)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gt_dir", required=True); ap.add_argument("--result_dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []; n_obj = 0; n_gt_rev = 0; n_matched_rev = 0
    for pred_dir in sorted(glob.glob(os.path.join(a.result_dir, "*"))):
        name = os.path.basename(pred_dir); pnpz = os.path.join(pred_dir, "pred.npz"); gnpz = os.path.join(a.gt_dir, f"{name}.npz")
        obj = os.path.join(pred_dir, "original.obj") if os.path.exists(os.path.join(pred_dir, "original.obj")) else None
        if not (os.path.exists(pnpz) and os.path.exists(gnpz)):
            continue
        P, G = dict(np.load(pnpz, allow_pickle=True)), dict(np.load(gnpz, allow_pickle=True))
        if "points" in P and "part_ids" in P:
            pp, ppid = P["points"], P["part_ids"]
        else:
            if obj is None:
                objs = glob.glob(os.path.join(pred_dir, "*.obj")); obj = objs[0] if objs else None
            if obj is None: continue
            mesh = trimesh.load(obj, force="mesh")
            pp, fidx = trimesh.sample.sample_surface(mesh, 20000); ppid = P["face_part_ids"][fidx]
        kp = int(P["revolute_plucker"].shape[0]); kg = int(G["revolute_plucker"].shape[0])
        cp = centroids(pp, ppid, kp); cg = centroids(G["points"], G["part_ids"], kg)
        cost = np.linalg.norm(cp[:, None] - cg[None], axis=-1); cost = np.nan_to_num(cost, nan=1e3)
        ri, ci = linear_sum_assignment(cost)
        n_obj += 1; n_gt_rev += int(np.sum(G["is_part_revolute"]))
        for i, j in zip(ri, ci):
            if not G["is_part_revolute"][j]:
                if G["is_part_prismatic"][j]:
                    m = axis_metrics(P["prismatic_axis"][i][:3], G["prismatic_axis"][j][:3]); m["is_rot"] = False
                    m["type_ok"] = float(bool(P["is_part_prismatic"][i])); rows.append(m)
                continue
            n_matched_rev += 1
            pg, pl = G["revolute_plucker"][j], P["revolute_plucker"][i]
            m = axis_metrics(pl[:3], pg[:3], plucker_point(pl), plucker_point(pg)); m["is_rot"] = True
            m["type_ok"] = float(bool(P["is_part_revolute"][i])); rows.append(m)
    keys = ["angle_unsigned_deg", "angle_signed_deg", "flip", "origin_line_dist", "type_ok"]
    out = {"n_objects": n_obj, "n_gt_revolute": n_gt_rev, "n_matched_revolute": n_matched_rev,
           "revolute": {k: float(np.nanmean([r[k] for r in rows if r["is_rot"] and k in r])) for k in keys},
           "prismatic": {k: float(np.nanmean([r[k] for r in rows if not r["is_rot"] and k in r])) for k in ["angle_unsigned_deg", "angle_signed_deg", "flip", "type_ok"]}}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out))


if __name__ == "__main__":
    main()
