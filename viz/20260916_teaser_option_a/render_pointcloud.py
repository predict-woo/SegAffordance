#!/usr/bin/env python3
"""Oblique point-cloud view of one SceneFun3D record with its ground-truth articulation.

Back-projects the record's metric depth (panels/sf3d/depth.npy, dumped by
viz/20260914_arthur_method_figure/dump_sample.py) with the frame's intrinsics, colours the points
from the RGB frame, rotates the cloud for an oblique view and draws the annotated hinge axis (yellow),
the hinge (yellow dot), the interaction point (white ring) and the sweep arc (green) in 3D.
Runs on the Mac (numpy + matplotlib only).

    python3 render_pointcloud.py --src panels/sf3d --out src/pointcloud.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="panels/sf3d")
    ap.add_argument("--out", default="src/pointcloud.png")
    ap.add_argument("--yaw", type=float, default=-38.0, help="degrees, view rotation about the vertical axis")
    ap.add_argument("--pitch", type=float, default=22.0, help="degrees, view tilt (positive = look down)")
    ap.add_argument("--points", type=int, default=90000, help="random subsample of valid depth pixels")
    ap.add_argument("--zoom", type=float, default=1.0, help="1 = whole cloud; >1 zooms toward the hinge")
    ap.add_argument("--size", type=int, default=1100, help="output width in px (3:2)")
    a = ap.parse_args()
    src = Path(a.src)
    depth = np.load(src / "depth.npy").astype(np.float64)
    rgb = np.asarray(Image.open(src / "frame.png").convert("RGB")).astype(np.float64) / 255.0
    meta = json.load(open(src / "meta3d.json"))
    H, W = depth.shape
    K = np.asarray(meta["K_norm"], np.float64) * np.array([[W, W, W], [H, H, H], [1, 1, 1]])  # pixel intrinsics
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    ys, xs = np.mgrid[0:H, 0:W]
    z = depth[ys, xs]
    ok = np.isfinite(z) & (z > 0.05)
    xs, ys, z = xs[ok].astype(np.float64), ys[ok].astype(np.float64), z[ok]
    rng = np.random.default_rng(0)
    if len(z) > a.points:
        sel = rng.choice(len(z), a.points, replace=False); xs, ys, z = xs[sel], ys[sel], z[sel]
    xs = xs + rng.uniform(-0.5, 0.5, len(xs)); ys = ys + rng.uniform(-0.5, 0.5, len(ys))   # break the pixel lattice
    pts = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)           # camera frame, +z forward, +y down
    col = rgb[ys.astype(int), xs.astype(int)]

    # ---- ground truth in 3D
    o = np.asarray(meta["origin_3d"]); n = np.asarray(meta["axis_dir"]); n = n / np.linalg.norm(n)
    p = np.asarray(meta["point_3d"]); traj = np.asarray(meta["traj3d"])
    ext = 0.55
    axis = np.stack([o - ext * n, o + ext * n])

    # ---- oblique view: rotate about the vertical (image y) axis, then tilt; keep +y down for the image
    R = rot_x(np.radians(a.pitch)) @ rot_y(np.radians(a.yaw))
    ctr = np.median(pts, 0)
    view = lambda X: (X - ctr) @ R.T + ctr

    P = view(pts); A = view(axis); Pp = view(p[None])[0]; T = view(traj); O = view(o[None])[0]
    # weak perspective: scale by depth so the far wall shrinks a little
    def proj(X):
        zz = X[:, 2]
        s = 1.0 / (0.35 + 0.65 * zz / np.median(P[:, 2]))
        return X[:, 0] * s, X[:, 1] * s, zz
    px, py, pz = proj(P)
    order = np.argsort(-pz)                                  # far points first
    ax_x, ax_y, _ = proj(A); pp_x, pp_y, _ = proj(Pp[None]); tx, ty, _ = proj(T); ox, oy, _ = proj(O[None])

    fig = plt.figure(figsize=(a.size / 200, a.size / 300), dpi=200)
    axp = fig.add_axes([0, 0, 1, 1]); axp.set_facecolor("#14171b"); fig.patch.set_facecolor("#14171b")
    axp.scatter(px[order], py[order], c=col[order], s=0.9, marker="o", linewidths=0, rasterized=True)
    halo_dark = [pe.Stroke(linewidth=7, foreground=(0, 0, 0, 0.6)), pe.Normal()]
    halo_white = [pe.Stroke(linewidth=7, foreground="white"), pe.Normal()]
    axp.plot(ax_x, ax_y, color="#ffd43b", lw=3.4, solid_capstyle="round", path_effects=halo_dark, zorder=5)
    axp.scatter(ox, oy, s=110, facecolor="#ffd43b", edgecolor="black", linewidths=1.3, zorder=6)
    axp.plot(tx, ty, color="#2ecc40", lw=3.2, solid_capstyle="round", path_effects=halo_white, zorder=5)
    d = np.array([tx[-1] - tx[-2], ty[-1] - ty[-2]]); d = d / (np.linalg.norm(d) + 1e-9) * 0.03 * (px.max() - px.min())
    axp.annotate("", xy=(tx[-1] + d[0], ty[-1] + d[1]), xytext=(tx[-1], ty[-1]),
                 arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.8", color="#2ecc40", lw=3.0, mutation_scale=14,
                                 path_effects=halo_white), zorder=6)
    axp.scatter(pp_x, pp_y, s=150, facecolor="white", edgecolor="#2ecc40", linewidths=3.0, zorder=7)

    # frame the door region: centre on the annotated part with a 3:2 window
    lo_x, hi_x = np.percentile(px, 1), np.percentile(px, 99); lo_y, hi_y = np.percentile(py, 1), np.percentile(py, 99)
    w = (hi_x - lo_x) * 1.04 / a.zoom; h = w * 2 / 3
    if h < (hi_y - lo_y) * 1.04 / a.zoom: h = (hi_y - lo_y) * 1.04 / a.zoom; w = h * 3 / 2
    t = min(1.0, a.zoom - 1.0)                               # zooming pulls the window centre toward the hinge
    cxv = (1 - t) * (lo_x + hi_x) / 2 + t * ox; cyv = (1 - t) * (lo_y + hi_y) / 2 + t * oy
    axp.set_xlim(cxv - w / 2, cxv + w / 2); axp.set_ylim(cyv + h / 2, cyv - h / 2); axp.axis("off")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=fig.get_facecolor()); plt.close(fig)
    print("wrote", a.out, "| points", len(pts))


if __name__ == "__main__":
    main()
