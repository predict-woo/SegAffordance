"""Batched quick-eval of a checkpoint on N fixed val samples.

Reports mean mask IoU, detection rate (IoU>0.5), motion-type accuracy and
mean axis error (direction-agnostic, degrees; CVAE motion is sampled from
the prior, so axis numbers carry sampling noise). Same fixed sample set for
a given (--split, --num, --seed), so results are comparable across
checkpoints.

    python tools/eval_checkpoint.py --config <cfg.yaml> \
        --checkpoint <ckpt|checkpoint-dir> --num 300
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.opd_train import ModelParams
from datasets.opdreal import OPDRealDataset, get_default_transforms
from model.segmenter import CRIS
from utils.dataset import tokenize


def pick_best_checkpoint(path):
    """path may be a .ckpt file or a directory of best-*valloss*.ckpt files."""
    if os.path.isfile(path):
        return path
    cands = glob.glob(os.path.join(path, "best-*.ckpt"))
    if not cands:
        raise SystemExit(f"no best-*.ckpt in {path}")

    def valloss(p):
        m = re.search(r"valloss([0-9.]+?)\.ckpt", p)
        return float(m.group(1)) if m else float("inf")

    return min(cands, key=valloss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help=".ckpt file or checkpoints/ dir")
    ap.add_argument("--num", type=int, default=300)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ckpt_path = pick_best_checkpoint(args.checkpoint)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mp = ModelParams(**cfg["model"]["model_params"])
    dc = cfg["data"]

    rgb_t, mask_t, depth_t = get_default_transforms(
        image_size=tuple(dc["input_size"]), is_train=False
    )
    ds = OPDRealDataset(
        data_path=dc["data_path"],
        dataset_key=f"MotionNet_{args.split}",
        rgb_transform=rgb_t,
        mask_transform=mask_t,
        depth_transform=depth_t,
        is_multi=dc.get("is_multi", False),
        use_depth=dc.get("use_depth", True),
    )
    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(ds), size=min(args.num, len(ds)), replace=False)
    samples = [ds[i] for i in idxs]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRIS(mp)
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = st.get("state_dict", st)
    st = {k[len("model."):]: v for k, v in st.items() if k.startswith("model.")}
    model.load_state_dict(st, strict=False)
    model.to(device).eval()

    ious, type_hits, axerrs = [], 0, []
    n = len(samples)
    with torch.no_grad():
        for s in range(0, n, args.batch):
            chunk = samples[s : s + args.batch]
            img = torch.stack([c[0] for c in chunk]).to(device)
            dep = torch.stack([c[1] for c in chunk]).to(device)
            tok = tokenize([c[2] for c in chunk], mp.word_len, truncate=True).to(device)
            mgt = torch.stack([c[3] for c in chunk]).to(device)
            tgt = torch.stack([c[7] for c in chunk])
            agt = torch.stack([c[6] for c in chunk])
            out = model(img, dep, tok, None, None, None)
            mup = F.interpolate(
                torch.sigmoid(out[0]), size=mgt.shape[-2:], mode="bilinear", align_corners=False
            )
            pb = (mup > 0.5).float()
            gb = mgt.float()
            inter = (pb * gb).sum(dim=(1, 2, 3))
            union = ((pb + gb) > 0).float().sum(dim=(1, 2, 3))
            ious += (inter / union.clamp(min=1)).cpu().tolist()
            type_hits += int((out[4].argmax(1).cpu() == tgt).sum())
            ap_ = F.normalize(out[3], dim=1).cpu()
            ag = F.normalize(agt, dim=1)
            cos = (ap_ * ag).sum(1).abs().clamp(max=1.0)
            axerrs += torch.rad2deg(torch.acos(cos)).tolist()

    ious = np.array(ious)
    result = {
        "checkpoint": ckpt_path,
        "split": args.split,
        "num": n,
        "seed": args.seed,
        "mIoU": round(float(ious.mean()), 4),
        "det_at_0.5": round(float((ious > 0.5).mean()), 4),
        "type_acc": round(type_hits / n, 4),
        "axis_err_deg": round(float(np.mean(axerrs)), 2),
    }
    print(
        f"{os.path.basename(ckpt_path)}: mIoU={result['mIoU']:.3f} "
        f"det={result['det_at_0.5']*100:.1f}% type={result['type_acc']*100:.1f}% "
        f"axis={result['axis_err_deg']:.1f}deg  (n={n}, split={args.split}, seed={args.seed})"
    )
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
