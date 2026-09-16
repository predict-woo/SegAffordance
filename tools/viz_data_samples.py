"""Paper Fig. 2 candidates: render ground-truth records of the 2D hand-video
sources (hoi4d | epic | arctic) as clean panels: frame, moving-part mask (red
fill + outline), hand track (green, start dot, arrowhead at the end), a type
badge and the instruction as a caption strip. No model involved.

Held-out (val) split by default, one record per scene, type-balanced where
both types exist. Writes per-record PNGs plus a contact sheet per source.

  python tools/viz_data_samples.py --dataset hoi4d --num 10 --out viz/<batch>/hoi4d
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.scenefun3d import SF3DDataset, split_dataset_by_scene  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_manifest import write_manifest  # noqa: E402

SOURCES = {
    "hoi4d": ("/workspace/datasets/hoi4d_processed_2d_v2", "/workspace/cache/hoi4d_2d_keys_v2.pkl"),
    "epic": ("/workspace/datasets/epic_processed_2d", "/workspace/cache/epic_2d_keys_v1.pkl"),
    "arctic": ("/workspace/datasets/arctic_processed_2d", "/workspace/cache/arctic_2d_keys_v1.pkl"),
}
RED = (0.90, 0.16, 0.16)
GREEN = (0.20, 0.85, 0.30)
TYPE_NAME = {0: "prismatic", 1: "revolute"}
TYPE_COLOR = {0: "#2b6cb0", 1: "#c05621"}


def pick(ds, indices, num, rng, balance):
    """One record per scene (key prefix), type-balanced when both types exist."""
    by_scene = {}
    for i in indices:
        by_scene.setdefault(ds.item_keys[i].decode().split("/")[0], []).append(i)
    scenes = list(by_scene)
    rng.shuffle(scenes)
    cand = [int(rng.choice(by_scene[s])) for s in scenes]
    if not balance:
        return cand[:num]
    picks, n_rot, n_tr, half = [], 0, 0, (num + 1) // 2
    for i in cand:
        t = int(ds[i][7])
        if t == 1 and n_rot < half:
            picks.append(i); n_rot += 1
        elif t == 0 and n_tr < num - half:
            picks.append(i); n_tr += 1
        if len(picks) >= num:
            break
    if len(picks) < num:  # one type is scarce: top up with whatever is left
        picks += [i for i in cand if i not in picks][: num - len(picks)]
    return picks


def mask_polys(mask_u8):
    cs, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c[:, 0, :] for c in cs if len(c) >= 3 and cv2.contourArea(c) > 30]


def render(frame_rgb, mask, track_px, valid, point_xy, mtype, text, out_png, dpi=200, k=1.0, caption=True, badge=True, draw_mask=True):
    """k scales fonts and line widths (k ~ 2.5 for a panel printed ~3 cm wide)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch

    H, W = frame_rgb.shape[:2]
    cap_h = int((0.16 + 0.05 * (k - 1)) * H) if caption else 0
    fig_w = 6.0
    fig = plt.figure(figsize=(fig_w, fig_w * (H + cap_h) / W), dpi=dpi)
    ax = fig.add_axes([0, cap_h / (H + cap_h), 1, H / (H + cap_h)])
    ax.imshow(frame_rgb)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

    # mask: translucent fill + crisp outline
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    if draw_mask:
        overlay[mask > 0] = (*RED, 0.38)
    ax.imshow(overlay)
    for poly in (mask_polys((mask > 0).astype(np.uint8)) if draw_mask else []):
        ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
                color=RED, lw=1.6 * k, solid_joinstyle="round")

    # hand track: white halo under a green line, start dot, arrowhead at the end
    pts = track_px[valid]
    if len(pts) >= 2:
        ax.plot(pts[:, 0], pts[:, 1], color=GREEN, lw=3.0 * k, solid_capstyle="round",
                path_effects=[pe.Stroke(linewidth=5.5 * k, foreground="white"), pe.Normal()])
        d = pts[-1] - pts[-2]
        n = np.linalg.norm(d)
        if n > 1e-3:
            d = d / n * max(0.025 * W * k ** 0.5, 6)
            ax.annotate("", xy=tuple(pts[-1] + d), xytext=tuple(pts[-1]),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.55,head_length=0.9",
                                        color=GREEN, lw=3.0 * k, mutation_scale=10 * k ** 0.5,
                                        path_effects=[pe.Stroke(linewidth=5.5 * k, foreground="white"), pe.Normal()]))
    ax.scatter([point_xy[0]], [point_xy[1]], s=140 * k * k, facecolor="white", edgecolor=GREEN,
               linewidths=3.0 * k, zorder=5)

    # type badge
    label = TYPE_NAME[int(mtype)]
    if badge:
      ax.text(0.03 * W, 0.045 * H, label, ha="left", va="top", fontsize=13 * k, color="white",
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc=TYPE_COLOR[int(mtype)],
                      ec="none", alpha=0.95))

    # caption strip
    if not caption:
        fig.savefig(out_png, dpi=dpi, facecolor="white")
        plt.close(fig)
        return
    cax = fig.add_axes([0, 0, 1, cap_h / (H + cap_h)])
    cax.set_xlim(0, 1); cax.set_ylim(0, 1); cax.axis("off")
    cax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0", fc="white", ec="none"))
    t = text.strip().replace("\n", " ")
    if len(t) > 60:
        t = t[:57].rstrip() + "..."
    cax.text(0.5, 0.5, f"“{t}”", ha="center", va="center", fontsize=14 * k,
             style="italic", color="#222222")
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)


def contact_sheet(pngs, out_jpg, cols=5):
    ims = [cv2.imread(p) for p in pngs]
    ims = [im for im in ims if im is not None]
    if not ims:
        return
    h = min(im.shape[0] for im in ims)
    ims = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA) for im in ims]
    w = max(im.shape[1] for im in ims)
    pad = 12
    rows = (len(ims) + cols - 1) // cols
    sheet = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 235, np.uint8)
    for k, im in enumerate(ims):
        r, c = divmod(k, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
    cv2.imwrite(out_jpg, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SOURCES), required=True)
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--val-split-ratio", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", default="", help="comma-separated record keys to render instead of random picks")
    ap.add_argument("--style-scale", type=float, default=1.0, help="font / line scale (2 for ~3 cm print width)")
    ap.add_argument("--no-caption", action="store_true", help="omit the instruction strip (put it in the LaTeX caption instead)")
    ap.add_argument("--no-badge", action="store_true", help="omit the revolute / prismatic badge")
    ap.add_argument("--no-mask", action="store_true", help="omit the part mask (hand track only)")
    a = ap.parse_args()

    root, key_cache = SOURCES[a.dataset]
    ds = SF3DDataset(
        lmdb_data_root=root, lmdb_path=f"{root}/data.lmdb",
        frame_cache_path=f"{root}/frames.lmdb", key_cache_path=key_cache,
        image_size_for_mask_reconstruction=(512, 512), return_trajectory_2d=True,
        point_source="element", fast_pipeline=True, load_depth=False,
        min_revolute_radius=0.0, min_mask_area_frac=0.0, edge_margin_frac=0.0,
        sensor_max_occluded_frac=0.5,
    )
    tr, va = split_dataset_by_scene(ds, a.val_split_ratio, 42)
    pool = {"val": va.indices, "train": tr.indices, "all": list(range(len(ds)))}[a.split]
    rng = np.random.default_rng(a.seed)
    if a.keys:
        want = [k for k in a.keys.split(",") if k]
        index = {ds.item_keys[i].decode(): i for i in range(len(ds))}
        picks = [index[k] for k in want]
    else:
        picks = pick(ds, pool, a.num, rng, balance=(a.dataset != "arctic"))
    print(f"{a.dataset}: {len(pool)} records in {a.split}; rendering {len(picks)}")

    os.makedirs(a.out, exist_ok=True)
    pngs, rows = [], []
    for n, idx in enumerate(picks):
        it = ds[idx]
        key = ds.item_keys[idx].decode()
        img = it[0].permute(1, 2, 0).numpy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        W0, H0 = [float(v) for v in it[8].tolist()]
        # undo the square resize so the panel has the source's aspect ratio
        if W0 >= H0:
            W, H = 1024, int(round(1024 * H0 / W0))
        else:
            H, W = 1024, int(round(1024 * W0 / H0))
        frame = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(it[3][0].numpy(), (W, H), interpolation=cv2.INTER_NEAREST)
        tr2 = it[13].numpy()
        track = np.stack([tr2[:, 0] * W / W0, tr2[:, 1] * H / H0], 1)
        valid = it[14].numpy().astype(bool)
        point = (float(it[5][0]) * W, float(it[5][1]) * H)
        mtype = int(it[7])
        safe = key.replace("/", "__")
        out_png = os.path.join(a.out, f"{n:02d}_{safe}.png")
        render(frame, mask, track, valid, point, mtype, it[2], out_png, k=a.style_scale, caption=not a.no_caption, badge=not a.no_badge, draw_mask=not a.no_mask)
        pngs.append(out_png)
        rows.append(f"| {n:02d} | `{key}` | {TYPE_NAME[mtype]} | {it[2].strip().splitlines()[0]} |")
        print(f"  {n:02d} {key} {TYPE_NAME[mtype]} :: {it[2].strip().splitlines()[0]}")
    contact_sheet(pngs, os.path.join(a.out, f"contact_{a.dataset}.jpg"))
    write_manifest(a.out, dataset=a.dataset, split=a.split, seed=a.seed, style_scale=a.style_scale, keys=[ds.item_keys[i].decode() for i in picks])
    with open(os.path.join(a.out, "samples.md"), "w") as f:
        f.write(f"# {a.dataset} Fig. 2 candidates (split={a.split}, seed={a.seed})\n\n")
        f.write("| # | key | type | instruction |\n|---|---|---|---|\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
