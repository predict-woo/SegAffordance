"""Per-input sample-distribution analysis for SINGAPO axis metrics (paired across ckpts).
For each test input and ckpt: over the 5 samples, the mean / std / min of the mean
unsigned axis error over the object's non-fixed joints. Then paired comparisons."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, "/workspace/repo/singapo")
from eval_axis import per_object, canon  # noqa
R = "/workspace/repo/singapo/exps"
runs = {"init": "dryrun", "theirs": "eval_ft_theirs_early", "ours": "eval_ft_ours_early", "ours_l2": "eval_ft_ours_l2"}
gt_root = "/workspace/data"
per = {}  # key -> {ckpt: (mean, std, min, flipfrac)}
for name, exp in runs.items():
    tdir = [p for p in glob.glob(f"{R}/{exp}/v1/output/test/epoch_*") if os.path.isdir(p)][0]
    for d in sorted(glob.glob(os.path.join(tdir, "*@*"))):
        toks = os.path.basename(d).split("@"); key = "@".join(toks[1:]) + "#" + toks[0]
        gt = json.load(open(os.path.join(gt_root, *toks[1:], "object.json")))
        vals, flips = [], []
        for s in sorted(glob.glob(os.path.join(d, "*", "object.json"))):
            rows = per_object(json.load(open(s)), gt)
            if rows:
                vals.append(np.mean([r["angle_unsigned_deg"] for r in rows])); flips.append(np.mean([r["flip"] for r in rows]))
        if vals:
            per.setdefault(key, {})[name] = (float(np.mean(vals)), float(np.std(vals)), float(np.min(vals)), float(np.mean(flips)))
keys = [k for k in per if len(per[k]) == 4]
A = {n: np.array([per[k][n] for k in keys]) for n in runs}
print("n_inputs", len(keys))
for n in runs:
    a = A[n]; print(f"{n:7s} mean-of-means {a[:,0].mean():6.2f}  median {np.median(a[:,0]):6.2f}  mean within-input std {a[:,1].mean():5.2f}  mean best {a[:,2].mean():5.2f}  median best {np.median(a[:,2]):5.2f}  flip {a[:,3].mean():.3f}")
def paired(x, y, label):
    d = x - y; print(f"  {label}: ours-theirs mean {d.mean():+.2f}, median {np.median(d):+.2f}, ours better on {np.mean(d<0)*100:.0f}% of inputs")
print("paired ours vs theirs:")
paired(A["ours"][:,0], A["theirs"][:,0], "per-input mean angle")
paired(A["ours"][:,2], A["theirs"][:,2], "per-input best-of-5 angle")
paired(A["ours"][:,1], A["theirs"][:,1], "within-input std (sharpness)")
paired(A["ours"][:,3], A["theirs"][:,3], "flip fraction")
print("paired ours_l2 vs theirs:")
paired(A["ours_l2"][:,0], A["theirs"][:,0], "per-input mean angle")
paired(A["ours_l2"][:,2], A["theirs"][:,2], "per-input best-of-5 angle")
paired(A["ours_l2"][:,1], A["theirs"][:,1], "within-input std")
paired(A["ours_l2"][:,3], A["theirs"][:,3], "flip fraction")
print("paired ours vs init:")
paired(A["ours"][:,0], A["init"][:,0], "per-input mean angle")
paired(A["ours"][:,2], A["init"][:,2], "per-input best-of-5 angle")
paired(A["ours"][:,1], A["init"][:,1], "within-input std")
# tails: fraction of inputs with mean angle > 20 deg, and > 45
for n in runs:
    a=A[n][:,0]; print(f"{n:7s} frac mean>20°: {np.mean(a>20):.2f}  >45°: {np.mean(a>45):.2f}   frac best<3°: {np.mean(A[n][:,2]<3):.2f}")
