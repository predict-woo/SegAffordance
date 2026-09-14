"""Paper Fig. 4 panels in the Fig. 3 style (tools/viz_data_samples.py): GT | baselines ... | ours on SF3D
val frames, drawn with matplotlib from PREDICTION DUMPS, so no GPU and no checkpoint:

  ours      tools/sf3d_vis_val.py --dump records (tools/sf3d_preds_io.py)
  baseline  baselines-session preds.jsonl (matched, mask_rle at native (H, W), type, axis_cam, origin_cam);
            pass the literal path "pending" for a model still training -> grey placeholder panel

Per panel: frame at its native aspect, cropped (not stretched) to --aspect around the GT interaction point;
moving-part mask as translucent red fill + crisp outline; articulation axis in yellow with a dark halo (hinge
line with a foot dot for revolute, direction ray from the interaction point for prismatic); motion in green with
a white halo and an arrowhead: the annotated 2D track on the GT panel and, on ours, the motion IMPLIED by the
predicted axis / hinge / point over the GROUND-TRUTH extent (the sweep angle or travel length of the GT 3D track;
--extent decoded draws the model's own decoded trajectory instead, whose learned extent is meaningless on SF3D
where only the articulation parameters are supervised); the interaction point as a white dot with a green rim
(GT / ours); a type badge top-left; on prediction panels a small axis-error label bottom-left. Baseline panels
show the oracle-matched instance the scorer used (mask + axis; they predict no point and no motion).

Writes one strip PNG per frame (all panels side by side) plus the single panels, and (unless --no-save-preds)
the exact records used: preds_ours.jsonl and preds_<baseline>.jsonl restricted to the drawn frames, so the
figure can be re-rendered in any style later from this directory alone (+ the LMDB for the frames).

  CUDA_VISIBLE_DEVICES= python tools/viz_fig4_panels.py --ours dense viz/<batch>/preds.jsonl \
      --baseline OPDFormer-C /workspace/datasets/baselines/results/opd512_c_rgbd/preds.jsonl \
      --baseline MOPD pending --baseline A3VLM /workspace/datasets/baselines/results/a3vlm/preds_chain.jsonl \
      --idx 1684 113 3726 403 --out viz/<batch> [--k 2.0] [--panel-w 1024] [--extent gt|decoded]
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from model.losses.geometric import project_points  # noqa: E402
from sf3d_preds_io import read_jsonl  # noqa: E402
from sf3d_vis_val import add_data_args, load_frame, load_val  # noqa: E402
from viz_data_samples import GREEN, RED, TYPE_COLOR, TYPE_NAME, mask_polys  # noqa: E402
from viz_manifest import write_manifest  # noqa: E402

YELLOW = (1.0, 0.83, 0.0)


def unit(d):
    d = np.asarray(d, np.float64)
    return d / max(float(np.linalg.norm(d)), 1e-8)


def axis_uv(K_norm, pt, d, center_at, t0, t1, n=33):
    """Normalised uv of the 3D segment pt + t d, t in [t0, t1], centred at the axis point closest to center_at."""
    d = unit(d); pt = np.asarray(pt, np.float64)
    c = pt + float(np.dot(np.asarray(center_at, np.float64) - pt, d)) * d
    p3 = c[None] + np.linspace(t0, t1, n)[:, None] * d[None]
    ok = p3[:, 2] > 0.05
    uv = project_points(K_norm, torch.from_numpy(p3[None]).float())[0].numpy()
    return uv[ok], c


def traj_uv(K_norm, p3):
    p3 = np.asarray(p3, np.float64); ok = p3[:, 2] > 0.05
    return project_points(K_norm, torch.from_numpy(p3[None]).float())[0].numpy()[ok]


def gt_extent(traj3d, gt_type, origin, d):
    """Sweep angle (rad, revolute) or travel length (m, prismatic) of the GT 3D track."""
    p = np.asarray(traj3d, np.float64)
    if gt_type != 1:
        return float(np.linalg.norm(p[-1] - p[0]))
    d = unit(d); o = np.asarray(origin, np.float64)
    r0 = p[0] - o; r0 -= np.dot(r0, d) * d
    r1 = p[-1] - o; r1 -= np.dot(r1, d) * d
    return float(math.atan2(np.linalg.norm(np.cross(r0, r1)), np.dot(r0, r1)))


def implied_motion(point, d, origin, mtype, extent, n=40):
    """3D motion of `point` under the articulation (d, origin) over `extent`: arc (revolute) or segment (prismatic)."""
    p = np.asarray(point, np.float64); d = unit(d)
    if mtype == 1 and origin is not None:
        o = np.asarray(origin, np.float64); c = o + np.dot(p - o, d) * d; r = p - c
        if np.linalg.norm(r) < 1e-4:
            return None
        th = np.linspace(0.0, extent, n)
        return c[None] + np.cos(th)[:, None] * r[None] + np.sin(th)[:, None] * np.cross(d, r)[None]
    return p[None] + np.linspace(0.0, extent, n)[:, None] * d[None]


def render(frame_rgb, mask, axis, track, point, mtype, err_deg, out_png, k=2.0, dpi=150, crop=None, placeholder=None):
    """frame_rgb (H, W, 3) uint8; mask (H, W) 0/1 or None; axis: (uv array (N,2) normalised, hinge uv or None) or None;
    track: uv (N,2) normalised or None; point: uv normalised or None; mtype 0/1 or None; err_deg float or None;
    placeholder: text -> grey panel with that text and nothing else."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    H, W = frame_rgb.shape[:2]
    x0, y0, x1, y1 = crop if crop is not None else (0, 0, W, H)   # pixel window shown (aspect crop, no stretch)
    fig = plt.figure(figsize=((x1 - x0) / dpi, (y1 - y0) / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    ox, oy, W, H = x0, y0, x1 - x0, y1 - y0   # text positions below are relative to the window
    W_full, H_full = frame_rgb.shape[1], frame_rgb.shape[0]
    if placeholder is not None:
        ax.imshow(np.full((H_full, W_full, 3), 235, np.uint8))
        ax.text(ox + W / 2, oy + H / 2, placeholder, ha="center", va="center", fontsize=11 * k, color="#555555", style="italic")
        fig.savefig(out_png, dpi=dpi, facecolor="white"); plt.close(fig); return
    ax.imshow(frame_rgb)
    if mask is not None:
        ov = np.zeros((H_full, W_full, 4), np.float32); ov[mask > 0] = (*RED, 0.40); ax.imshow(ov)
        for poly in mask_polys((mask > 0).astype(np.uint8)):
            ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=RED, lw=1.5 * k, solid_joinstyle="round")
    halo_dark = [pe.Stroke(linewidth=5.0 * k, foreground=(0, 0, 0, 0.55)), pe.Normal()]
    halo_white = [pe.Stroke(linewidth=5.5 * k, foreground="white"), pe.Normal()]
    if axis is not None and len(axis[0]) >= 2:
        uv, hinge = axis
        ax.plot(uv[:, 0] * W_full, uv[:, 1] * H_full, color=YELLOW, lw=2.8 * k, solid_capstyle="round", path_effects=halo_dark, zorder=3)
        if hinge is not None:
            ax.scatter([hinge[0] * W_full], [hinge[1] * H_full], s=90 * k * k, marker="o", facecolor=YELLOW, edgecolor="black", linewidths=1.2 * k, zorder=4)
    if track is not None and len(track) >= 2:
        pts = np.asarray(track) * [W_full, H_full]
        ax.plot(pts[:, 0], pts[:, 1], color=GREEN, lw=3.0 * k, solid_capstyle="round", path_effects=halo_white, zorder=5)
        d = pts[-1] - pts[-2]; n = np.linalg.norm(d)
        if n > 1e-3:
            d = d / n * max(0.025 * W_full * k ** 0.5, 6)
            ax.annotate("", xy=tuple(pts[-1] + d), xytext=tuple(pts[-1]),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.55,head_length=0.9", color=GREEN, lw=3.0 * k,
                                        mutation_scale=10 * k ** 0.5, path_effects=halo_white), zorder=5)
    if point is not None:
        ax.scatter([point[0] * W_full], [point[1] * H_full], s=140 * k * k, facecolor="white", edgecolor=GREEN, linewidths=3.0 * k, zorder=6)
    if mtype is not None:
        ax.text(ox + 0.03 * W, oy + 0.045 * H, TYPE_NAME[int(mtype)], ha="left", va="top", fontsize=11 * k, color="white", fontweight="bold",
                zorder=7, bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.8", fc=TYPE_COLOR[int(mtype)], ec="none", alpha=0.95))
    if err_deg is not None:
        ax.text(ox + 0.03 * W, oy + 0.955 * H, f"axis error {err_deg:.0f}°", ha="left", va="bottom", fontsize=9.5 * k, color="white",
                zorder=7, bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.8", fc=(0, 0, 0, 0.55), ec="none"))
    fig.savefig(out_png, dpi=dpi, facecolor="white"); plt.close(fig)


def angle(d, gt):
    return math.degrees(math.acos(float(np.clip(np.dot(unit(d), gt), -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs=2, action="append", default=[], metavar=("NAME", "PREDS_JSONL"), help="our dump; NAME = model field in it")
    ap.add_argument("--baseline", nargs=2, action="append", default=[], metavar=("NAME", "PREDS_JSONL|pending"))
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    add_data_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--panel-w", type=int, default=1024)
    ap.add_argument("--aspect", type=float, default=4 / 3, help="crop every panel to this aspect around the GT point (0 = native)")
    ap.add_argument("--extent", choices=["gt", "decoded"], default="gt", help="motion drawn on our panels: GT extent through the predicted parameters, or the decoded trajectory")
    ap.add_argument("--no-save-preds", action="store_true")
    a = ap.parse_args()
    ours = {}
    for name, p in a.ours:
        for r in read_jsonl(p):
            if r["model"] == name:
                ours[(name, r["val_idx"])] = r
    bl = [(n, None if p == "pending" else {r["key"]: r for r in read_jsonl(p)}) for n, p in a.baseline]
    ds, va = load_val(a)
    os.makedirs(a.out, exist_ok=True)
    from pycocotools import mask as mask_utils
    used = {n: [] for n, _ in a.baseline}; used_ours = []
    for n_i, j in enumerate(a.idx):
        fr = load_frame(va, j, 1.0)
        key = ds.item_keys[va.indices[j]].decode()
        PW = a.panel_w; PH = int(round(PW * fr["H"] / fr["W"]))
        frame = cv2.resize(fr["frame"], (PW, PH), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].copy()  # native aspect, RGB
        cw, ch = PW, PH
        if a.aspect > 0:
            if PW / PH > a.aspect: cw = int(round(PH * a.aspect))
            else: ch = int(round(PW / a.aspect))
        px, py = float(fr["point_gt"][0]) * PW, float(fr["point_gt"][1]) * PH
        x0 = int(min(max(px - cw / 2, 0), PW - cw)); y0 = int(min(max(py - ch / 2, 0), PH - ch))
        crop = (x0, y0, x0 + cw, y0 + ch)
        K, gt_dir, gt_type = fr["K_norm"], fr["gt_dir"], fr["gt_type"]
        traj3d = fr["traj3d"].float().numpy(); p0 = traj3d[0]
        gt_origin = fr["origin_3d"].float().numpy()
        extent = gt_extent(traj3d, gt_type, gt_origin, gt_dir)
        tag = f"{n_i:02d}_{'rot' if gt_type == 1 else 'trans'}_val{j}"
        panels = []

        def axis_for(t, d, origin, anchor):
            if t == 1 and origin is not None:
                uv, c = axis_uv(K, origin, d, anchor, -a.ray_len, a.ray_len)
                return uv, project_points(K, torch.from_numpy(c[None, None]).float())[0, 0].numpy()
            uv, _ = axis_uv(K, anchor, d, anchor, 0.0, a.ray_len)
            return uv, None

        # GT
        m = cv2.resize(fr["mask_t"][0].numpy(), (PW, PH), interpolation=cv2.INTER_NEAREST)
        tr = (fr["traj2d_px"] / torch.tensor([fr["W"], fr["H"]])).numpy()[fr["valid2d"].numpy()]
        f = f"{a.out}/{tag}_gt.png"
        render(frame, m, axis_for(gt_type, gt_dir, gt_origin, p0), tr, fr["point_gt"].numpy(), gt_type, None, f, a.k, crop=crop); panels.append(f)
        # baselines: oracle-matched instance, or placeholder
        for name, preds in bl:
            f = f"{a.out}/{tag}_{name}.png"
            if preds is None:
                render(frame, None, None, None, None, None, None, f, a.k, crop=crop, placeholder=f"{name}\n(pending)"); panels.append(f); continue
            r = preds.get(key); used[name].append(r if r else {"key": key, "matched": False})
            if not r or not r.get("matched"):
                render(frame, None, None, None, None, None, None, f, a.k, crop=crop, placeholder=f"{name}\nno instance"); panels.append(f); continue
            bm = cv2.resize(mask_utils.decode(r["mask_rle"]).astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST)
            t = int(r["type"]); d = np.asarray(r["axis_cam"], np.float64)
            render(frame, bm, axis_for(t, d, r.get("origin_cam"), p0), None, None, t, angle(d, gt_dir), f, a.k, crop=crop); panels.append(f)
        # ours
        for name, _ in a.ours:
            r = ours[(name, j)]; used_ours.append(r); f = f"{a.out}/{tag}_{name}.png"
            rle = dict(r["mask_rle"]); rle["counts"] = rle["counts"].encode("ascii")
            om = cv2.resize(mask_utils.decode(rle).astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST)
            t = int(np.argmax(r["type_logits"])); d = np.asarray(r["motion"], np.float64); anchor = np.asarray(r["point_3d"], np.float64)
            if a.extent == "gt":
                mo = implied_motion(anchor, d, r["origin"] if t == 1 else None, t, extent)
                tr = traj_uv(K, mo) if mo is not None else None
            else:
                tr = traj_uv(K, anchor[None] + np.asarray(r["trajectory"], np.float64)) if r["trajectory"] else None
            render(frame, om, axis_for(t, d, r["origin"], anchor), tr, r["point_uv"], t, angle(d, gt_dir), f, a.k, crop=crop); panels.append(f)
        ims = [cv2.imread(p) for p in panels]
        ims = [cv2.resize(im, (cw, ch), interpolation=cv2.INTER_AREA) if im.shape[:2] != (ch, cw) else im for im in ims]
        cv2.imwrite(f"{a.out}/{tag}.png", np.hstack(ims))
        print("wrote", tag, "|", fr["desc"][:60], f"| GT extent {math.degrees(extent):.0f} deg" if gt_type == 1 else f"| GT travel {extent:.2f} m")
    if not a.no_save_preds:
        with open(f"{a.out}/preds_ours.jsonl", "w") as fo:
            for r in used_ours: fo.write(json.dumps(r) + "\n")
        for name, rs in used.items():
            if rs:
                with open(f"{a.out}/preds_{name}.jsonl", "w") as fo:
                    for r in rs: fo.write(json.dumps(r) + "\n")
    write_manifest(a.out, ours=a.ours, baselines=a.baseline, picks=[int(x) for x in a.idx], extent=a.extent)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
