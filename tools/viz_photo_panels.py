"""Paper-style panels (the Fig. 4 / Fig. 5 look of tools/viz_fig4_panels.py) for in-the-wild PHOTOS.

Input: the jsonl written by `tools/predict_image.py --dump` (one record per case x model: mask RLE at the 512 input,
type logits, point_uv, motion, origin, point_3d, decoded trajectory, plus image path, prompt, K_norm, native size).
Per case: photo | one panel per model (red mask + outline, yellow axis with the hinge foot, green decoded motion with
an arrowhead, white interaction point, type badge; no ground truth, so no axis-error label). Every panel of a row is
the same crop (--aspect around the LAST model's predicted point, i.e. ours when it is listed last).

  python tools/viz_photo_panels.py --dump viz/<batch>/preds.jsonl --models sf3d_only,dense --idx 56 77 82 85 \\
      --out viz/<batch>/panels [--aspect 1.0] [--panel-w 1024] [--k 2.0]
Then compose with tools/compose_panels.py (--row-captions for the prompts).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from viz_fig4_panels import axis_uv, render, traj_uv  # noqa: E402
from model.losses.geometric import project_points  # noqa: E402
from viz_manifest import write_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--models", required=True, help="comma-separated model names in panel order (as in the dump's `model` field)")
    ap.add_argument("--idx", type=int, nargs="+", required=True, help="case indices (the dump's val_idx)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--aspect", type=float, default=1.0, help="crop aspect (w/h) around the last model's point; 0 = native")
    ap.add_argument("--panel-w", type=int, default=1024)
    ap.add_argument("--ray-len", type=float, default=0.5)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    from pycocotools import mask as mask_utils
    models = a.models.split(",")
    recs = {}
    for line in open(a.dump):
        if line.strip():
            r = json.loads(line); recs[(r["model"], int(r["val_idx"]))] = r
    os.makedirs(a.out, exist_ok=True)
    for n_i, j in enumerate(a.idx):
        rs = [recs[(m, j)] for m in models]
        r0 = rs[-1]
        bgr = cv2.imread(r0["image"], cv2.IMREAD_COLOR)   # honours EXIF orientation, as predict_image.py did
        Wn, Hn = r0["native_wh"]
        assert bgr.shape[1] == Wn and bgr.shape[0] == Hn, (bgr.shape, r0["native_wh"])
        PW = a.panel_w; PH = int(round(PW * Hn / Wn))
        frame = cv2.resize(bgr, (PW, PH), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
        cw, ch = PW, PH
        if a.aspect > 0:
            if PW / PH > a.aspect: cw = int(round(PH * a.aspect))
            else: ch = int(round(PW / a.aspect))
        px, py = float(r0["point_uv"][0]) * PW, float(r0["point_uv"][1]) * PH
        x0 = int(min(max(px - cw / 2, 0), PW - cw)); y0 = int(min(max(py - ch / 2, 0), PH - ch))
        crop = (x0, y0, x0 + cw, y0 + ch)
        K = torch.tensor(r0["K_norm"], dtype=torch.float32)[None]
        stem = os.path.splitext(os.path.basename(r0["image"]))[0]
        tag = f"{n_i:02d}_{j:02d}_{stem}"
        panels = []
        f = f"{a.out}/{tag}_photo.png"
        render(frame, None, None, None, None, None, None, f, a.k, crop=crop); panels.append(f)
        for m, r in zip(models, rs):
            rle = dict(r["mask_rle"]); rle["counts"] = rle["counts"].encode("ascii")
            om = cv2.resize(mask_utils.decode(rle).astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST)
            t = int(np.argmax(r["type_logits"])); d = np.asarray(r["motion"], np.float64); d /= max(np.linalg.norm(d), 1e-8)
            anchor = np.asarray(r["point_3d"], np.float64)
            if t == 1 and r.get("origin") is not None:
                uv, c = axis_uv(K, np.asarray(r["origin"], np.float64), d, anchor, -a.ray_len, a.ray_len)
                axis = (uv, project_points(K, torch.from_numpy(c[None, None]).float())[0, 0].numpy())
            else:
                uv, _ = axis_uv(K, anchor, d, anchor, 0.0, a.ray_len); axis = (uv, None)
            tr = traj_uv(K, anchor[None] + np.asarray(r["trajectory"], np.float64)) if r.get("trajectory") else None
            f = f"{a.out}/{tag}_{m}.png"
            render(frame, om, axis, tr, r["point_uv"], None if a.clean else t, None, f, a.k, crop=crop); panels.append(f)
        ims = [cv2.imread(p) for p in panels]
        ims = [cv2.resize(im, (cw, ch), interpolation=cv2.INTER_AREA) if im.shape[:2] != (ch, cw) else im for im in ims]
        cv2.imwrite(f"{a.out}/{tag}.png", np.hstack(ims))
        print("wrote", tag, "|", r0["prompt"], "|", [("rot" if int(np.argmax(r["type_logits"])) == 1 else "trans") for r in rs])
    write_manifest(a.out, dump=a.dump, models=models, picks=[int(x) for x in a.idx], aspect=a.aspect)


if __name__ == "__main__":
    main()
