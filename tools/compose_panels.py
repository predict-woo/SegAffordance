"""Stack horizontal panel strips (as written by the viz tools) into one figure image.

Each input PNG is a row of equal-width panels (GT | model A | model B ...). Options select which panel
columns to keep, in which order, add a header row with column labels, per-row labels in a left margin,
column / row GROUPS (a spanning label plus a thin black divider between groups, e.g. "Baselines" vs "Ours",
or the dataset of each block of rows), and stack the rows with a thin gap. Output: PNG (and JPG if --jpg).

  python tools/compose_panels.py --out figures/fig5.png --panel-aspect 1.3333 \
      --labels "Ground truth,OPDFormer-C,...,ARTHUR" --row-labels "out of distribution,in distribution,..." \
      --col-groups "Baselines:1-6,Ours:7-8" --row-groups "ARCTIC:0-1,EPIC:2,HOI4D:3-4" row1.png row2.png ...
Group ranges are 0-based inclusive indices over the KEPT columns / the input rows.
"""
import argparse
import os

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
INK = (20, 20, 20)


def parse_groups(spec):
    """'Name:a-b,Name2:c' -> [(name, a, b), ...]"""
    out = []
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        name, rng = part.rsplit(":", 1)
        a, _, b = rng.partition("-")
        out.append((name.strip(), int(a), int(b or a)))
    return out


def text_w(txt, scale, thick):
    return cv2.getTextSize(txt, FONT, scale, thick)[0]


def vertical_label(txt, height, width, scale, thick):
    strip = np.full((width, height, 3), 255, np.uint8)   # drawn horizontally, rotated to read bottom-up
    tw, th = text_w(txt, scale, thick)
    cv2.putText(strip, txt, ((height - tw) // 2, (width + th) // 2), FONT, scale, INK, thick, cv2.LINE_AA)
    return cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", default=None, help="comma-separated panel column indices to keep, in order (default all)")
    ap.add_argument("--labels", default=None, help="comma-separated column labels for a header row")
    ap.add_argument("--panel-width", type=int, default=None, help="width of one panel in the strips (default: height x --panel-aspect)")
    ap.add_argument("--panel-aspect", type=float, default=1.0, help="panel width / height when --panel-width is not given (4:3 crops: 1.3333)")
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--group-gap", type=int, default=22, help="gap at a group boundary (the divider line sits in its middle)")
    ap.add_argument("--line", type=int, default=3, help="divider line thickness")
    ap.add_argument("--row-height", type=int, default=360, help="resize every row to this height")
    ap.add_argument("--strip-header", type=int, default=0, help="crop this many pixels off the top of each panel (removes the burnt-in text)")
    ap.add_argument("--row-labels", default=None, help="comma-separated per-row labels, drawn rotated in a left margin (e.g. in / out of distribution)")
    ap.add_argument("--col-groups", default=None, help='column groups "Name:a-b,..." (kept-column indices): spanning label above the column labels + divider')
    ap.add_argument("--row-groups", default=None, help='row groups "Name:a-b,..." (input-row indices): label in the outer left margin + divider')
    ap.add_argument("--row-captions", default=None, help="'|'-separated per-row captions drawn centred under each row (e.g. the prompt)")
    ap.add_argument("--jpg", action="store_true")
    a = ap.parse_args()
    rows = []
    for f in a.inputs:
        im = cv2.imread(f)
        if im is None:
            raise SystemExit(f"cannot read {f}")
        h, w = im.shape[:2]
        pw = a.panel_width or int(round(h * a.panel_aspect))
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
    ncol = len(rows[0]); nrow = len(rows)
    colw = [max(r[c].shape[1] for r in rows) for c in range(ncol)]
    col_groups = parse_groups(a.col_groups); row_groups = parse_groups(a.row_groups)
    col_bounds = {b for _, _, b in col_groups if b < ncol - 1} | {s - 1 for _, s, _ in col_groups if s > 0}
    row_bounds = {b for _, _, b in row_groups if b < nrow - 1} | {s - 1 for _, s, _ in row_groups if s > 0}
    # x offsets of the columns (extra gap after a group boundary), y offsets of the rows
    label_margin = 40 if a.row_labels else 0
    group_margin = 52 if row_groups else 0
    x_left = group_margin + label_margin + a.gap
    xs = []; x = x_left
    for c in range(ncol):
        xs.append(x); x += colw[c] + (a.group_gap if c in col_bounds else a.gap)
    W = x - (a.group_gap if (ncol - 1) in col_bounds else a.gap) + a.gap
    header_labels = 44 if a.labels else 0
    header_groups = 46 if col_groups else 0
    y_top = header_groups + header_labels + a.gap
    captions = a.row_captions.split("|") if a.row_captions else []
    cap_h = 46 if captions else 0
    ys = []; y = y_top
    for r in range(nrow):
        ys.append(y); y += a.row_height + cap_h + (a.group_gap if r in row_bounds else a.gap)
    H = y - (a.group_gap if (nrow - 1) in row_bounds else a.gap) + a.gap
    canvas = np.full((H, W, 3), 255, np.uint8)
    # panels
    for r, panels in enumerate(rows):
        for c, p in enumerate(panels):
            canvas[ys[r]:ys[r] + p.shape[0], xs[c]:xs[c] + p.shape[1]] = p
    # row captions (under each row, centred over the panel span)
    for r in range(nrow):
        if captions and r < len(captions) and captions[r]:
            txt = captions[r]; tw, th = text_w(txt, 0.95, 2)
            cv2.putText(canvas, txt, ((x_left + W - a.gap - tw) // 2, ys[r] + a.row_height + 34), FONT, 0.95, INK, 2, cv2.LINE_AA)
    # column labels
    if a.labels:
        labels = a.labels.split(",")
        for c in range(ncol):
            txt = labels[c] if c < len(labels) else ""
            tw, th = text_w(txt, 0.9, 2)
            cv2.putText(canvas, txt, (xs[c] + (colw[c] - tw) // 2, header_groups + 32), FONT, 0.9, INK, 2, cv2.LINE_AA)
    # column groups: spanning label + divider between groups
    for name, s, e in col_groups:
        x0, x1 = xs[s], xs[e] + colw[e]
        tw, th = text_w(name, 1.0, 2)
        cv2.putText(canvas, name, ((x0 + x1 - tw) // 2, 30), FONT, 1.0, INK, 2, cv2.LINE_AA)
        cv2.line(canvas, (x0, 40), (x1, 40), INK, 2, cv2.LINE_AA)
    for c in col_bounds:
        xm = xs[c] + colw[c] + a.group_gap // 2
        cv2.line(canvas, (xm, y_top), (xm, H - a.gap), INK, a.line, cv2.LINE_AA)
    # row labels (inner margin) and row groups (outer margin + divider between groups)
    row_labels = a.row_labels.split(",") if a.row_labels else []
    for r in range(nrow):
        if row_labels:
            txt = row_labels[r] if r < len(row_labels) else ""
            canvas[ys[r]:ys[r] + a.row_height, group_margin:group_margin + label_margin] = vertical_label(txt, a.row_height, label_margin, 0.8, 2)
    for name, s, e in row_groups:
        y0, y1 = ys[s], ys[e] + a.row_height
        canvas[y0:y1, 0:group_margin] = vertical_label(name, y1 - y0, group_margin, 1.0, 2)
    for r in row_bounds:
        ym = ys[r] + a.row_height + cap_h + a.group_gap // 2
        cv2.line(canvas, (x_left - a.gap, ym), (W - a.gap, ym), INK, a.line, cv2.LINE_AA)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(a.out, canvas)
    if a.jpg:
        cv2.imwrite(os.path.splitext(a.out)[0] + ".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", a.out, canvas.shape)


if __name__ == "__main__":
    main()
