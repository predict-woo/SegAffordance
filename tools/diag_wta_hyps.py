"""Diagnose WTA hypothesis behavior of a gen-4 checkpoint on val samples.

Per sample: the K hypotheses' |omega|, sweep-sign agreement with GT, the
body-metric distance of each to GT, the oracle (best-of-K) vs the
argmax-logit selection. Aggregates answer three questions:
  1. spread: are the hypotheses functionally different? (mean pairwise
     body-distance between hypotheses)
  2. coverage: does SOME hypothesis fit GT? (oracle distance)
  3. selection: does the logit head find it? (selected vs oracle gap,
     winner-prediction accuracy)

Run from repo root on a pod:
  python tools/diag_wta_hyps.py --config config/sf3d_train_runpod_twist.yaml \
      --ckpt experiments/20260811_sf3d_twist_g4/checkpoints/last.ckpt --num 300
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.opd_train import ModelParams  # noqa: E402
from datasets.scenefun3d import (  # noqa: E402
    SF3DDataset, get_default_transforms, split_dataset_by_scene,
)
from model.losses.twist import twist_body_distance, twist_from_gt  # noqa: E402
from model.segmenter import CRIS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v2")
    ap.add_argument("--key-cache", default="/workspace/cache/sf3d_v2_keys_cutoff05.pkl")
    ap.add_argument("--min-revolute-radius", type=float, default=0.0,
                    help="must match the key cache / training config")
    ap.add_argument("--num", type=int, default=300)
    args = ap.parse_args()

    ckpt = sorted(glob.glob(args.ckpt), key=os.path.getmtime)[-1]
    print(f"checkpoint: {ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mp = ModelParams(**cfg["model"]["model_params"])
    model = CRIS(mp)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    r, m, d = get_default_transforms((256, 256))
    ds = SF3DDataset(
        lmdb_data_root=args.data_root, key_cache_path=args.key_cache,
        frame_cache_path=os.path.join(args.data_root, "frames.lmdb"),
        rgb_transform=r, mask_transform=m, depth_transform=d,
        image_size_for_mask_reconstruction=(256, 256),
        point_source="element", return_trajectory_2d=True,
        min_revolute_radius=args.min_revolute_radius,
    )
    _, val = split_dataset_by_scene(ds, val_split_ratio=0.1, manual_seed=42)
    rng = np.random.default_rng(0)

    spread, oracle, selected, win_by_type = [], [], [], {0: [], 1: []}
    om_sel, om_oracle = {0: [], 1: []}, {0: [], 1: []}
    sel_is_oracle = 0
    n = 0
    for i in rng.permutation(len(val)):
        if n >= args.num:
            break
        s = val[int(i)]
        (img_t, depth_t, desc, _m, _b, _p, motion, ty, _sz, _n2, origin,
         _K, traj3d, _t2, _tv) = s
        t = int(ty)
        gt = twist_from_gt(motion[None].float(), torch.tensor([t]),
                           origin[None].float())[0]
        p0 = traj3d[0].float()
        with torch.no_grad():
            word = model.tokenize([desc], 77).to(device)
            out = model(img_t[None].to(device), depth_t[None].to(device),
                        word, None, None, None, None)
        hyps = out.twist_hyps[0].float().cpu()          # (K, 6)
        logits = out.twist_logits[0].float().cpu()
        K = hyps.shape[0]
        d_gt = twist_body_distance(hyps, gt[None], p0[None], 0.25)  # (K,)
        pair = [twist_body_distance(hyps[a], hyps[b], p0, 0.25).item()
                for a in range(K) for b in range(a + 1, K)]
        spread.append(float(np.mean(pair)))
        oracle.append(d_gt.min().item())
        sel = int(logits.argmax())
        selected.append(d_gt[sel].item())
        sel_is_oracle += int(sel == int(d_gt.argmin()))
        win_by_type[t].append(int(d_gt.argmin()))
        om_sel[t].append(hyps[sel][:3].norm().item())
        om_oracle[t].append(hyps[int(d_gt.argmin())][:3].norm().item())
        n += 1

    print(f"\nn={n}")
    print(f"hypothesis spread (mean pairwise body-dist): {np.mean(spread):.4f}")
    print(f"oracle best-of-K distance:  mean {np.mean(oracle):.4f}  median {np.median(oracle):.4f}")
    print(f"argmax-logit distance:      mean {np.mean(selected):.4f}  median {np.median(selected):.4f}")
    print(f"selector picks the oracle:  {sel_is_oracle/n:.2f}  (random = {1/4:.2f})")
    for t, name in ((1, "revolute"), (0, "prismatic")):
        w = np.bincount(win_by_type[t], minlength=4) / max(1, len(win_by_type[t]))
        print(f"{name:9s} winner histogram {np.round(w,2)}  "
              f"|omega| selected med {np.median(om_sel[t]):.2f}  oracle med {np.median(om_oracle[t]):.2f}")


if __name__ == "__main__":
    main()
