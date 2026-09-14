"""Re-render SF3D validation panels from dumped predictions, NO GPU and no checkpoint: reads the JSONL
written by `tools/sf3d_vis_val.py --dump` (tools/sf3d_preds_io.py records), fetches the frame and GT
from the LMDB on CPU, and draws [GT | model A | model B ...] with exactly the same code as the live tool.

  python tools/sf3d_render_preds.py --preds viz/<batch>/preds.jsonl [--preds other.jsonl ...] \
      --out viz/<new batch> [--idx 1696 1773 ...] [--models dense,dense_seed7] [--scale 2]
Each --preds file may hold several models; --models keeps a subset (in that order); --idx keeps a subset
of frames (default: every frame present in the first file, in file order).
"""
import argparse
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from sf3d_preds_io import read_jsonl, record_to_out  # noqa: E402
from sf3d_vis_val import add_data_args, gt_panel, load_frame, load_val, panel_filename, pred_panel  # noqa: E402
from viz_manifest import write_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=True)
    add_data_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--idx", type=int, nargs="*", default=None)
    ap.add_argument("--models", default=None, help="comma-separated model names to draw, in order (default: all, file order)")
    a = ap.parse_args()
    recs = {}
    order_models, order_idx = [], []
    for p in a.preds:
        for r in read_jsonl(p):
            recs[(r["model"], r["val_idx"])] = r
            if r["model"] not in order_models:
                order_models.append(r["model"])
            if r["val_idx"] not in order_idx:
                order_idx.append(r["val_idx"])
    models = a.models.split(",") if a.models else order_models
    picks = list(a.idx) if a.idx else order_idx
    ds, va = load_val(a)
    os.makedirs(a.out, exist_ok=True)
    for n_i, j in enumerate(picks):
        fr = load_frame(va, j, a.scale)
        panels = [gt_panel(fr, a.ray_len)]
        for name in models:
            r = recs.get((name, j))
            if r is None:
                print(f"missing {name} val {j}"); continue
            panels.append(pred_panel(fr, record_to_out(r), name, a.ray_len))
        cv2.imwrite(f"{a.out}/{panel_filename(n_i, fr)}", np.hstack(panels))
        print("wrote", n_i, j, "|", fr["desc"][:60])
    write_manifest(a.out, preds=a.preds, models=models, picks=[int(x) for x in picks])
    print("done ->", a.out)


if __name__ == "__main__":
    main()
