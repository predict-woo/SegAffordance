"""Axis/origin metrics for SINGAPO test outputs (the paper reports none).
Pred nodes are index-aligned with GT (test uses the GT graph), so no matching.
  python eval_axis.py --test_dir exps/<name>/<ver>/output/test/epoch_XXX_w=0.5_pm \
                      --gt_root ../../data --out metrics_axis.json
"""
import os, sys, json, argparse, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screw_loss import axis_metrics

ROT = {"revolute", "continuous"}; TRANS = {"prismatic", "screw"}


def canon(d):
    d = np.asarray(d, dtype=np.float64)
    return -d if (d > 0).sum() < (-d > 0).sum() else d   # SINGAPO's GT convention


def per_object(pred, gt):
    rows = []
    for i, gnode in enumerate(gt["diffuse_tree"]):
        gtype = gnode["joint"]["type"]
        if gtype not in ROT | TRANS or i >= len(pred["diffuse_tree"]):
            continue
        pnode = pred["diffuse_tree"][i]
        n_g = canon(gnode["joint"]["axis"]["direction"]); n_p = np.asarray(pnode["joint"]["axis"]["direction"], float)
        if gtype in ROT:
            m = axis_metrics(n_p, n_g, pnode["joint"]["axis"]["origin"], gnode["joint"]["axis"]["origin"])
        else:
            m = axis_metrics(n_p, n_g)
        m["is_rot"] = gtype in ROT
        m["type_ok"] = float(pnode["joint"]["type"] == gtype)
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True); ap.add_argument("--gt_root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    inputs = sorted(d for d in glob.glob(os.path.join(a.test_dir, "*@*")) if os.path.isdir(d))
    agg = {"avg": [], "best": []}
    n_inputs = 0
    for d in inputs:
        toks = os.path.basename(d).split("@")
        gt_path = os.path.join(a.gt_root, *toks[1:], "object.json")
        if not os.path.exists(gt_path):
            continue
        gt = json.load(open(gt_path))
        samples = sorted(glob.glob(os.path.join(d, "*", "object.json")))
        per_sample = [per_object(json.load(open(s)), gt) for s in samples]
        per_sample = [r for r in per_sample if r]
        if not per_sample:
            continue
        n_inputs += 1
        # avg over samples; best = sample with lowest mean unsigned angle
        def mean_key(rows, k, rot_only=False):
            v = [r[k] for r in rows if (r["is_rot"] or not rot_only) and k in r]
            return float(np.mean(v)) if v else np.nan
        keys = [("angle_unsigned_deg", False), ("angle_signed_deg", False), ("flip", False),
                ("origin_line_dist", True), ("type_ok", False)]
        per_sample_means = [{k: mean_key(rows, k, ro) for k, ro in keys} for rows in per_sample]
        agg["avg"].append({k: float(np.nanmean([m[k] for m in per_sample_means])) for k, _ in keys})
        best = min(per_sample_means, key=lambda m: m["angle_unsigned_deg"])
        agg["best"].append(best)
    out = {"n_inputs": n_inputs}
    for mode in ("avg", "best"):
        out[mode] = {k: float(np.nanmean([m[k] for m in agg[mode]])) for k in agg[mode][0]} if agg[mode] else {}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
