"""Visual sanity check for datasets/augment.py on real records: for N random
train samples of a multi-source config, render [raw | aug 1 | aug 2 | aug 3]
with the GT overlays drawn FROM THE TRANSFORMED FIELDS (mask green, point
white ring, 2D track cyan, and — when the record's 3D track is metric —
project(K, traj3d) in magenta, which must sit on the cyan track).

  python tools/augment_check.py --config config/multi3_rgb_scalefree.yaml \
      --out viz/20260909_augment_check --num 8 [--seed 0]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from datasets.augment import AugmentSpec, augment_sample  # noqa: E402
from datasets.multisource_datamodule import MultiSourceDataModule, SourceSpec  # noqa: E402
from viz_manifest import write_manifest  # noqa: E402


def draw(sample, title):
    img, _d, desc, mask, bbox, point, _m, _t, img_size, fname, _o, K, traj3d, traj2d, valid = sample
    T = img.shape[-1]
    W, H = float(img_size[0]), float(img_size[1])
    bgr = img.permute(1, 2, 0).numpy()[:, :, ::-1].copy()
    m = mask[0].numpy() > 0.5
    bgr[m] = bgr[m] * 0.5 + np.array([0, 200, 0]) * 0.5
    sx, sy = T / W, T / H
    pts = np.stack([traj2d[:, 0].numpy() * sx, traj2d[:, 1].numpy() * sy], 1).astype(int)
    v = valid.numpy()
    if v.sum() >= 2:
        cv2.polylines(bgr, [pts[v]], False, (255, 255, 0), 2)
    # projection of the stored 3D track (metric on SF3D/ARCTIC; a placeholder elsewhere)
    z = traj3d[:, 2]
    if bool((z > 0.05).all()) and float(z.median()) < 5.0:
        uv = (K @ traj3d.T).T
        uv = (uv[:, :2] / uv[:, 2:3]).numpy()
        pp = np.stack([uv[:, 0] * sx, uv[:, 1] * sy], 1).astype(int)
        cv2.polylines(bgr, [pp], False, (255, 0, 255), 1)
    x, y, w, h = bbox.tolist()
    cv2.rectangle(bgr, (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy)), (0, 165, 255), 1)
    cv2.circle(bgr, (int(point[0] * T), int(point[1] * T)), 7, (255, 255, 255), 2)
    cv2.putText(bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bgr, desc[:48], (8, T - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["data"]
    spec = AugmentSpec(**cfg["augment"])
    dm = MultiSourceDataModule(
        sources=[SourceSpec(**s) for s in cfg["sources"]],
        val_split_ratio=cfg["val_split_ratio"], input_size=tuple(cfg["input_size"]),
        batch_size_train=1, batch_size_val=1, num_workers_train=0, num_workers_val=0,
        manual_seed=cfg["manual_seed"], load_depth=cfg.get("load_depth", False),
        augment=None,
    )
    dm.setup("fit")
    base = dm.train_dataset
    rng = np.random.default_rng(a.seed)
    picks = rng.choice(len(base), a.num, replace=False)
    os.makedirs(a.out, exist_ok=True)
    for n, i in enumerate(picks):
        s = base[int(i)]
        panels = [draw(s, "raw")]
        for k in range(3):
            torch.manual_seed(a.seed * 1000 + int(i) * 10 + k)
            panels.append(draw(augment_sample(s, spec), f"aug {k + 1}"))
        cv2.imwrite(f"{a.out}/{n:02d}_{s[9].replace('/', '_')[:40]}.jpg", np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", n, s[9])
    write_manifest(a.out)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
