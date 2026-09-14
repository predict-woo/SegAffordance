"""Stack horizontal panel strips (as written by the viz tools) into one figure image.

Each input PNG is a row of equal-width square panels (GT | model A | model B ...). Options select
which panel columns to keep, in which order, add a header row with column labels, and stack the
rows with a thin gap. Output: PNG (and JPG if --jpg) for \\includegraphics.

  python tools/compose_panels.py --out figures/fig5.png --cols 0,1,2 --labels "Ground truth,SceneFun3D only,ARTHUR" \
      viz/<batch>/hoi4d/03_*.png viz/<batch>/epic/01_*.png viz/<batch>/arctic/07_*.png
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", default=None, help="comma-separated panel column indices to keep, in order (default all)")
    ap.add_argument("--labels", default=None, help="comma-separated column labels for a header row")
    ap.add_argument("--panel-width", type=int, default=None, help="width of one panel in the strips (default: height, i.e. square panels)")
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--row-height", type=int, default=360, help="resize every row to this height")
    ap.add_argument("--strip-header", type=int, default=0, help="crop this many pixels off the top of each panel (removes the burnt-in text)")
    ap.add_argument("--jpg", action="store_true")
    a = ap.parse_args()
    rows = []
    for f in a.inputs:
        im = cv2.imread(f)
        if im is None:
            raise SystemExit(f"cannot read {f}")
        h, w = im.shape[:2]
        pw = a.panel_width or h
        n = w // pw
        panels = [im[:, i * pw:(i + 1) * pw] for i in range(n)]
        if a.strip_header:
            panels = [p[a.strip_header:] for p in panels]
        if a.cols:
            keep = [int(c) for c in a.cols.split(",")]
            panels = [panels[c] for c in keep]
        ph = panels[0].shape[0]
        scale = a.row_height / ph
        panels = [cv2.resize(p, (int(round(p.shape[1] * scale)), a.row_height), interpolation=cv2.INTER_AREA) for p in panels]
        rows.append(panels)
    ncol = len(rows[0])
    colw = [max(r[c].shape[1] for r in rows) for c in range(ncol)]
    W = sum(colw) + a.gap * (ncol + 1)
    header = 0
    if a.labels:
        header = 44
    H = header + a.gap + sum(a.row_height + a.gap for _ in rows)
    canvas = np.full((H, W, 3), 255, np.uint8)
    if a.labels:
        labels = a.labels.split(",")
        x = a.gap
        for c in range(ncol):
            txt = labels[c] if c < len(labels) else ""
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(canvas, txt, (x + (colw[c] - tw) // 2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
            x += colw[c] + a.gap
    y = header + a.gap
    for panels in rows:
        x = a.gap
        for c, p in enumerate(panels):
            canvas[y:y + p.shape[0], x:x + p.shape[1]] = p
            x += colw[c] + a.gap
        y += a.row_height + a.gap
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(a.out, canvas)
    if a.jpg:
        cv2.imwrite(os.path.splitext(a.out)[0] + ".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", a.out, canvas.shape)


if __name__ == "__main__":
    main()
