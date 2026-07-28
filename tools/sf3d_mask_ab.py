"""Side-by-side comparison of two mask rasterisations of the SAME records.

Intended use: run tools/sf3d_process.py twice over an identical --test-file,
once with --mask-method hull (the original convex hull) and once with
--mask-method splat, then point this at both output dirs.

Per sample it renders [clean zoom | hull | splat] and reports how the mask
area changed, so over-coverage (concave hardware) and sliver cases (thin
bars) can be checked by eye rather than argued about.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import lmdb
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
C_HULL = (60, 60, 255)    # red
C_SPLAT = (80, 220, 80)   # green


def build_mask(rec, h, w):
    m = np.zeros((h, w), np.uint8)
    c = np.asarray(rec.get("mask_coordinates_yx", []), dtype=np.int32)
    if c.size:
        m[np.clip(c[:, 0], 0, h - 1), np.clip(c[:, 1], 0, w - 1)] = 255
    return m


def draw(img, mask, colour, panel_w):
    out = img.copy()
    ov = out.copy()
    ov[mask > 0] = colour
    out = cv2.addWeighted(ov, 0.40, out, 0.60, 0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, colour, max(1, panel_w // 300))
    return out


def zoom_box(masks, w, h, margin=3.0, min_side=140):
    ys, xs = [], []
    for m in masks:
        ny, nx = np.nonzero(m)
        if nx.size:
            xs += [nx.min(), nx.max()]
            ys += [ny.min(), ny.max()]
    if not xs:
        return 0, 0, w, h
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * margin
    side = float(np.clip(max(side, min_side), min_side, min(w, h)))
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return x0, y0, int(x0 + side), int(y0 + side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hull-dir", type=Path, required=True)
    ap.add_argument("--splat-dir", type=Path, required=True)
    ap.add_argument("--images-from", type=Path, default=None,
                    help="dir holding images/ (defaults to --splat-dir)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--panel-width", type=int, default=460)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    img_root = args.images_from or args.splat_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels = args.out_dir / "panels"
    panels.mkdir(exist_ok=True)

    ea = lmdb.open(str(args.hull_dir / "data.lmdb"), readonly=True, lock=False)
    eb = lmdb.open(str(args.splat_dir / "data.lmdb"), readonly=True, lock=False)
    with ea.begin() as ta, eb.begin() as tb:
        keys_a = {k for k in ta.cursor().iternext(keys=True, values=False) if k != b"__metadata__"}
        keys_b = {k for k in tb.cursor().iternext(keys=True, values=False) if k != b"__metadata__"}
        common = sorted(keys_a & keys_b)
        print(f"hull records {len(keys_a)}, splat records {len(keys_b)}, common {len(common)}")
        if keys_a - keys_b or keys_b - keys_a:
            print(f"  NOTE only-in-hull {len(keys_a - keys_b)}, only-in-splat {len(keys_b - keys_a)}")

        rows, thumbs = [], []
        for i, k in enumerate(common[: args.limit], 1):
            ra, rb = pickle.loads(ta.get(k)), pickle.loads(tb.get(k))
            rgb = cv2.imread(str(img_root / "images" / rb["rgb_image_path"]))
            if rgb is None:
                print(f"  [{i}] missing image for {k.decode()}")
                continue
            h, w = rgb.shape[:2]
            ma, mb = build_mask(ra, h, w), build_mask(rb, h, w)
            inter = int(((ma > 0) & (mb > 0)).sum())
            union = int(((ma > 0) | (mb > 0)).sum())
            area_a, area_b = int((ma > 0).sum()), int((mb > 0).sum())

            x0, y0, x1, y1 = zoom_box([ma, mb], w, h)
            s = args.panel_width / max(1, x1 - x0)
            def crop(im, interp=cv2.INTER_LANCZOS4):
                sub = im[y0:y1, x0:x1]
                return cv2.resize(sub, (args.panel_width, int(sub.shape[0] * s)), interpolation=interp)
            base = crop(rgb)
            ca = crop(ma, cv2.INTER_NEAREST)
            cb = crop(mb, cv2.INTER_NEAREST)
            pa, pb = draw(base, ca, C_HULL, args.panel_width), draw(base, cb, C_SPLAT, args.panel_width)
            cv2.putText(base, "RGB", (10, 22), FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(pa, f"hull {area_a} px", (10, 22), FONT, 0.55, C_HULL, 2, cv2.LINE_AA)
            cv2.putText(pb, f"splat {area_b} px", (10, 22), FONT, 0.55, C_SPLAT, 2, cv2.LINE_AA)
            gap = np.full((base.shape[0], 4, 3), 22, np.uint8)
            body = np.hstack([base, gap, pa, gap, pb])

            label = (rb.get("label_info") or {}).get("label", "?")
            desc = (rb.get("description") or "").replace("\n---\n", " | ")[:95]
            vr = rb.get("visibility_ratio")
            hdr = np.full((66, body.shape[1], 3), 22, np.uint8)
            cv2.putText(hdr, k.decode(), (12, 20), FONT, 0.45, (120, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(hdr, f"[{label}] {desc}", (12, 40), FONT, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
            stat = (f"area x{area_b / max(1, area_a):.2f}   "
                    f"IoU(hull,splat) {inter / max(1, union):.2f}   "
                    f"pts {rb.get('visible_point_count')}/{rb.get('total_point_count')}")
            if vr is not None:
                stat += f"   vis {vr:.2f}"
            cv2.putText(hdr, stat, (12, 58), FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
            panel = np.vstack([hdr, body])
            cv2.imwrite(str(panels / f"{i:03d}_{k.decode().replace('/', '_')}.jpg"), panel,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            thumbs.append(np.hstack([pa, gap, pb]))
            rows.append({"key": k.decode(), "hull": area_a, "splat": area_b,
                         "iou": inter / max(1, union),
                         "pts": rb.get("visible_point_count") or 0})
    ea.close(); eb.close()

    if thumbs:
        tw = 460
        cells = [cv2.resize(t, (tw, int(t.shape[0] * tw / t.shape[1]))) for t in thumbs]
        th = max(c.shape[0] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, th - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(22, 22, 22))
                 for c in cells]
        grid = [np.hstack(cells[r:r + 5] + [np.full_like(cells[0], 22)] * (5 - len(cells[r:r + 5])))
                for r in range(0, len(cells), 5)]
        cv2.imwrite(str(args.out_dir / "contact_hull_vs_splat.jpg"), np.vstack(grid),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82])

    with open(args.out_dir / "ab.tsv", "w") as f:
        f.write("key\thull_px\tsplat_px\tratio\tiou\tvisible_pts\n")
        for r in rows:
            f.write(f"{r['key']}\t{r['hull']}\t{r['splat']}\t{r['splat']/max(1,r['hull']):.3f}"
                    f"\t{r['iou']:.3f}\t{r['pts']}\n")

    if rows:
        ratio = np.array([r["splat"] / max(1, r["hull"]) for r in rows])
        iou = np.array([r["iou"] for r in rows])
        hull_px = np.array([r["hull"] for r in rows], float)
        splat_px = np.array([r["splat"] for r in rows], float)
        print("\n--- hull -> splat ---")
        print(f"  samples            : {len(rows)}")
        print(f"  area ratio         : median {np.median(ratio):.2f}, "
              f"p10 {np.percentile(ratio,10):.2f}, p90 {np.percentile(ratio,90):.2f}")
        print(f"  shrunk (ratio<0.9) : {(ratio<0.9).sum()}   grew (>1.1): {(ratio>1.1).sum()}")
        print(f"  IoU(hull,splat)    : median {np.median(iou):.2f}")
        print(f"  median area        : hull {np.median(hull_px):.0f} px -> splat {np.median(splat_px):.0f} px")
        print(f"  tiny masks (<100px): hull {(hull_px<100).sum()} -> splat {(splat_px<100).sum()}")


if __name__ == "__main__":
    main()
