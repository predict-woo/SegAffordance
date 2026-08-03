"""Render GT visualisations for random samples of the processed SF3D LMDB.

Reads records written by ``tools/sf3d_process.py`` directly from the LMDB and
draws, per sample, a three-panel figure:

    [ full frame + overlays | zoom on the element | zoom on depth ]

Most SF3D functional elements are small and far away (a wall socket at 4 m is
a few hundred mask pixels, and its 0.1 m GT trajectory is sub-pixel at full
frame), so the zoom panel is where the GT is actually checkable. The yellow
box on the full frame marks the zoom region.

Overlays (projected with the record's stored camera_intrinsics):

    green fill / white contour  segmentation mask (convex hull of the visible
                                laser-scan points, as written by sf3d_process)
    magenta dot                 motion_origin_2d_image_coords (the interaction
                                point the model regresses)
    cyan->red polyline          trajectory_3d_camera_coords, coloured by index
                                so ordering / sweep direction is visible;
                                ring = first point, cross = last point
    yellow arrow                motion_dir_3d_camera_coords for "trans"
    orange double arrow         rotation axis for "rot"

Usage (on the pod, from /workspace/SegAffordance). Copy the LMDB to shm first
-- a full key scan against the MooseFS-backed volume page-faults at ~1.4 MB/s
(hours), while a sequential copy runs at ~155 MB/s:

    cp /workspace/datasets/sf3d_processed/data.lmdb/data.mdb /dev/shm/data.lmdb/
    python tools/sf3d_vis_samples.py --lmdb-path /dev/shm/data.lmdb \
        --out-dir viz/YYYYMMDD_<subject>_dataset_audit --num-samples 100 --seed 42

--out-dir must be a dated batch directory under viz/ (see CLAUDE.md,
"Visualization organization"); a manifest.yaml is written next to the images.
"""

import argparse
import pickle
import random
import textwrap
from pathlib import Path

import cv2
import lmdb
import numpy as np

from viz_manifest import write_manifest

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR
C_MASK = (80, 220, 80)
C_CONTOUR = (255, 255, 255)
C_ORIGIN = (255, 0, 255)
C_TRANS = (0, 255, 255)     # yellow
C_ROT = (0, 165, 255)       # orange
C_CROPBOX = (0, 220, 255)
C_TEXT = (235, 235, 235)
C_DIM = (150, 150, 150)

DEGENERATE_LEN_M = 0.01     # sf3d_process fallback segment length


def _pt(uv_row):
    """OpenCV wants plain python ints, not numpy scalars."""
    return (int(uv_row[0]), int(uv_row[1]))


def _ramp(t):
    """Cyan (t=0, trajectory start) -> red (t=1, trajectory end), BGR."""
    return (int(255 * (1 - t)), int(255 * (1 - t)), int(255 * t))


def project(points_cam, K):
    """Project (N,3) camera-frame points with K -> (uv (N,2), valid (N,))."""
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    z = pts[:, 2]
    valid = z > 1e-6
    homo = (K @ pts.T).T
    uv = np.zeros((len(pts), 2))
    uv[valid] = homo[valid, :2] / homo[valid, 2:3]
    return uv, valid


def build_mask(rec, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    coords = np.asarray(rec.get("mask_coordinates_yx", []), dtype=np.int32)
    if coords.size:
        mask[np.clip(coords[:, 0], 0, h - 1), np.clip(coords[:, 1], 0, w - 1)] = 255
    return mask


def adjust_K(K, x0, y0, s):
    """Intrinsics for an image cropped at (x0,y0) then scaled by s."""
    Ka = K.copy()
    Ka[0, 0] *= s
    Ka[1, 1] *= s
    Ka[0, 2] = (K[0, 2] - x0) * s
    Ka[1, 2] = (K[1, 2] - y0) * s
    return Ka


def zoom_rect(rec, mask, K, w, h, margin=2.6, min_side=180):
    """Square-ish crop covering the mask, the interaction point and the
    in-frame trajectory, with breathing room."""
    xs, ys = [], []
    ny, nx = np.nonzero(mask)
    if nx.size:
        xs += [nx.min(), nx.max()]
        ys += [ny.min(), ny.max()]

    fsm = (rec.get("motion_info") or {}).get("frame_specific_motion_data") or {}
    o2d = fsm.get("motion_origin_2d_image_coords")
    if o2d:
        xs.append(o2d[0])
        ys.append(o2d[1])

    traj = np.asarray(rec.get("trajectory_3d_camera_coords") or [],
                      dtype=np.float64).reshape(-1, 3)
    if len(traj):
        uv, valid = project(traj, K)
        inb = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if inb.any():
            xs += [uv[inb, 0].min(), uv[inb, 0].max()]
            ys += [uv[inb, 1].min(), uv[inb, 1].max()]

    if not xs:
        return 0, 0, w, h

    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * margin
    side = float(np.clip(max(side, min_side), min_side, min(w, h)))

    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return x0, y0, int(x0 + side), int(y0 + side)


def draw_view(img, mask, rec, K_eff, x0, y0, s):
    """Draw all GT overlays onto an already cropped+scaled image."""
    h, w = img.shape[:2]
    info = {}

    overlay = img.copy()
    overlay[mask > 0] = C_MASK
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, C_CONTOUR, max(1, w // 600))

    thick = max(2, w // 450)
    dot_r = max(3, w // 240)

    fsm = (rec.get("motion_info") or {}).get("frame_specific_motion_data") or {}
    omd = (rec.get("motion_info") or {}).get("original_motion_data") or {}
    info["motion_type"] = omd.get("motion_type", "?")

    # --- trajectory ---
    traj = np.asarray(rec.get("trajectory_3d_camera_coords") or [],
                      dtype=np.float64).reshape(-1, 3)
    if len(traj):
        uv, valid = project(traj, K_eff)
        inb = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        for i in range(len(traj) - 1):
            if inb[i] and inb[i + 1]:
                cv2.line(img, _pt(uv[i]), _pt(uv[i + 1]),
                         _ramp(i / max(1, len(traj) - 1)), thick)
        for i in np.where(inb)[0][:: max(1, len(traj) // 25)]:
            cv2.circle(img, _pt(uv[i]), max(1, dot_r - 1),
                       _ramp(i / max(1, len(traj) - 1)), -1)
        if inb[0]:
            cv2.circle(img, _pt(uv[0]), dot_r + 3, (255, 255, 0), thick)
        if inb[-1]:
            cv2.drawMarker(img, _pt(uv[-1]), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, dot_r * 4, thick)

    # --- motion vector / rotation axis ---
    origin_3d = fsm.get("motion_origin_3d_camera_coords")
    dir_3d = fsm.get("motion_dir_3d_camera_coords")
    if origin_3d is not None and dir_3d is not None:
        o = np.asarray(origin_3d, dtype=np.float64)
        d = np.asarray(dir_3d, dtype=np.float64)
        nd = np.linalg.norm(d)
        if nd > 1e-8:
            d = d / nd
            if info["motion_type"] == "rot":
                uv, valid = project(np.stack([o - d * 0.08, o + d * 0.08]), K_eff)
                if valid.all():
                    cv2.arrowedLine(img, _pt(uv[0]), _pt(uv[1]), C_ROT, thick, tipLength=0.15)
                    cv2.arrowedLine(img, _pt(uv[1]), _pt(uv[0]), C_ROT, thick, tipLength=0.15)
            else:
                uv, valid = project(np.stack([o, o + d * 0.12]), K_eff)
                if valid.all():
                    cv2.arrowedLine(img, _pt(uv[0]), _pt(uv[1]), C_TRANS, thick, tipLength=0.18)

    # --- interaction point ---
    o2d = fsm.get("motion_origin_2d_image_coords")
    if o2d is not None:
        p = (int(round((o2d[0] - x0) * s)), int(round((o2d[1] - y0) * s)))
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(img, p, dot_r + 4, (0, 0, 0), -1)
            cv2.circle(img, p, dot_r + 2, C_ORIGIN, -1)

    return img


def sample_stats(rec, mask, K, w, h):
    """Numbers worth having next to the picture."""
    info = {"mask_px": int((mask > 0).sum())}
    omd = (rec.get("motion_info") or {}).get("original_motion_data") or {}
    fsm = (rec.get("motion_info") or {}).get("frame_specific_motion_data") or {}
    info["motion_type"] = omd.get("motion_type", "?")

    traj = np.asarray(rec.get("trajectory_3d_camera_coords") or [],
                      dtype=np.float64).reshape(-1, 3)
    n_in = 0
    if len(traj):
        uv, valid = project(traj, K)
        inb = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        n_in = int(inb.sum())
        info["traj_len_m"] = float(np.linalg.norm(traj[-1] - traj[0]))
        info["traj_span_px"] = (float(np.ptp(uv[inb, 0])) if n_in > 1 else 0.0)
    info["traj_in_frame"] = f"{n_in}/{len(traj)}"

    origin_3d = fsm.get("motion_origin_3d_camera_coords")
    if origin_3d is not None:
        info["origin_depth_m"] = float(np.asarray(origin_3d)[2])
        if len(traj):
            info["origin_to_traj0_m"] = float(np.linalg.norm(traj[0] - np.asarray(origin_3d)))

    # A "rot" whose arc collapsed to the straight fallback in sf3d_process:
    # radius below 0.01 m -> 0.01 m straight segment starting at the origin.
    info["degenerate_rot"] = bool(
        info["motion_type"] == "rot"
        and abs(info.get("traj_len_m", 0.0) - DEGENERATE_LEN_M) < 1e-6
        and info.get("origin_to_traj0_m", 1.0) < 1e-6
    )
    return info


def render_sample(key, rec, lmdb_root, panel_w):
    rgb_path = lmdb_root / "images" / rec["rgb_image_path"]
    img_full = cv2.imread(str(rgb_path))
    if img_full is None:
        return None, None, {"error": f"missing rgb {rgb_path}"}

    h, w = img_full.shape[:2]
    K = np.asarray(rec["camera_intrinsics"], dtype=np.float64)
    mask_full = build_mask(rec, h, w)
    info = sample_stats(rec, mask_full, K, w, h)

    # --- depth, resampled onto the RGB grid so one crop rect serves both ---
    depth_name = rec.get("depth_image_path")
    depth_path = (lmdb_root / "depth" / depth_name) if depth_name else None
    depth_m = None
    if depth_path is not None and depth_path.is_file():
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            depth_m = cv2.resize(raw.astype(np.float32) / 1000.0, (w, h),
                                 interpolation=cv2.INTER_NEAREST)

    # --- panel 1: full frame ---
    s_full = panel_w / w
    full = cv2.resize(img_full, (panel_w, int(h * s_full)), interpolation=cv2.INTER_AREA)
    mask_f = cv2.resize(mask_full, (full.shape[1], full.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    full = draw_view(full, mask_f, rec, adjust_K(K, 0, 0, s_full), 0, 0, s_full)

    # --- panel 2: zoom on the element ---
    x0, y0, x1, y1 = zoom_rect(rec, mask_full, K, w, h)
    cv2.rectangle(full, (int(x0 * s_full), int(y0 * s_full)),
                  (int(x1 * s_full), int(y1 * s_full)), C_CROPBOX, max(1, panel_w // 340))

    sub = img_full[y0:y1, x0:x1]
    s_zoom = panel_w / max(1, sub.shape[1])
    zoom = cv2.resize(sub, (panel_w, int(sub.shape[0] * s_zoom)),
                      interpolation=cv2.INTER_LANCZOS4 if s_zoom > 1 else cv2.INTER_AREA)
    mask_z = cv2.resize(mask_full[y0:y1, x0:x1], (zoom.shape[1], zoom.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    zoom = draw_view(zoom, mask_z, rec, adjust_K(K, x0, y0, s_zoom), x0, y0, s_zoom)
    cv2.putText(zoom, f"zoom x{s_zoom:.1f}", (10, 24), FONT, 0.6, C_CROPBOX, 2, cv2.LINE_AA)
    info["zoom_factor"] = float(s_zoom)

    # --- panel 3: depth over the same zoom region ---
    if depth_m is not None:
        dsub = depth_m[y0:y1, x0:x1]
        finite = dsub[dsub > 0]
        lo, hi = ((float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
                  if finite.size else (0.0, 1.0))
        norm = np.clip((dsub - lo) / max(1e-6, hi - lo), 0, 1)
        dpan = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        dpan[dsub <= 0] = 0
        dpan = cv2.resize(dpan, (panel_w, int(dsub.shape[0] * panel_w / max(1, dsub.shape[1]))),
                          interpolation=cv2.INTER_NEAREST)
        cv2.drawContours(dpan, cv2.findContours(mask_z, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)[0],
                         -1, C_CONTOUR, max(1, panel_w // 400))
        cv2.putText(dpan, f"depth {lo:.2f}-{hi:.2f} m  (valid {100 * (dsub > 0).mean():.0f}%)",
                    (10, 24), FONT, 0.55, C_TEXT, 2, cv2.LINE_AA)
        info["depth_valid_frac"] = float((dsub > 0).mean())
    else:
        dpan = np.zeros((zoom.shape[0], panel_w, 3), dtype=np.uint8)
        cv2.putText(dpan, "depth missing", (12, dpan.shape[0] // 2), FONT, 0.8, (0, 0, 255), 2)

    # --- assemble ---
    ph = max(full.shape[0], zoom.shape[0], dpan.shape[0])
    def pad(p):
        return cv2.copyMakeBorder(p, 0, ph - p.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(22, 22, 22))
    gap = np.full((ph, 4, 3), 22, dtype=np.uint8)
    body = np.hstack([pad(full), gap, pad(zoom), gap, pad(dpan)])

    label = (rec.get("label_info") or {}).get("label", "?")
    desc = (rec.get("description") or "").replace("\n---\n", " | ").strip()
    desc_lines = textwrap.wrap(desc, width=max(40, body.shape[1] // 11))[:2] or ["(no description)"]

    geom = (f"traj in-frame {info.get('traj_in_frame', '-')}   "
            f"len {info.get('traj_len_m', float('nan')):.3f} m   "
            f"|traj0-origin| {info.get('origin_to_traj0_m', float('nan')):.3f} m   "
            f"origin z {info.get('origin_depth_m', float('nan')):.2f} m   "
            f"mask {info.get('mask_px', 0)} px"
            + ("   [DEGENERATE rot -> straight fallback]" if info.get("degenerate_rot") else ""))
    legend = ("mask=green  interaction pt=magenta  trajectory=cyan(start)->red(end)  "
              + ("axis=orange double arrow" if info.get("motion_type") == "rot"
                 else "motion dir=yellow arrow")
              + "  yellow box=zoom region")

    lines = [
        (key, (120, 220, 255), 0.55),
        (f"[{info.get('motion_type', '?')}]  label: {label}", (150, 255, 150), 0.55),
    ]
    lines += [(d, C_TEXT, 0.52) for d in desc_lines]
    lines += [(geom, (80, 160, 255) if info.get("degenerate_rot") else C_DIM, 0.48)]

    # Provenance + the ARKit sensor cross-check, when the record carries them.
    prov = []
    if rec.get("visible_point_count") is not None:
        prov.append(f"pts {rec['visible_point_count']}/{rec.get('total_point_count')}")
    if rec.get("visibility_ratio") is not None:
        prov.append(f"vis {rec['visibility_ratio']:.2f}")
    sc = rec.get("sensor_check")
    if sc:
        prov.append(
            f"sensor occluded {sc['sensor_occluded_frac']:.2f} "
            f"gap {sc['sensor_median_gap_m']:+.2f}m (n={sc['sensor_points_evaluated']})"
        )
    elif "sensor_check" in rec:
        prov.append("sensor: no paired frame")
    if prov:
        reject = bool(sc and sc["sensor_occluded_frac"] > 0.5)
        text = "  ".join(prov) + ("   <<< WOULD BE REJECTED" if reject else "")
        lines.append((text, (80, 120, 255) if reject else (170, 200, 170), 0.48))

    lines.append((legend, C_DIM, 0.45))

    line_h, pad_px = 26, 10
    bar = np.full((pad_px * 2 + line_h * len(lines), body.shape[1], 3), 22, dtype=np.uint8)
    for i, (text, colour, scale) in enumerate(lines):
        cv2.putText(bar, text, (14, pad_px + line_h * i + 18), FONT, scale, colour, 1, cv2.LINE_AA)

    return np.vstack([bar, body]), zoom, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmdb-root", type=Path,
                    default=Path("/workspace/datasets/sf3d_processed"),
                    help="dir containing images/ and depth/ (and data.lmdb by default)")
    ap.add_argument("--lmdb-path", type=Path, default=None,
                    help="override the LMDB location (default <lmdb-root>/data.lmdb). "
                         "Point at a /dev/shm copy: a key scan against the MooseFS "
                         "volume page-faults at ~1.4 MB/s.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="dated batch dir under viz/, e.g. viz/YYYYMMDD_<subject>_dataset_audit")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--panel-width", type=int, default=620)
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--contact-cols", type=int, default=10)
    ap.add_argument("--key-cache", type=Path,
                    default=Path("/workspace/cache/sf3d_lmdb_keys.pkl"))
    args = ap.parse_args()

    lmdb_path = args.lmdb_path or (args.lmdb_root / "data.lmdb")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = args.out_dir / "panels"
    panels_dir.mkdir(exist_ok=True)

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        # Run-wide settings live in __metadata__, not on every record.
        meta_raw = txn.get(b"__metadata__")
        meta = pickle.loads(meta_raw) if meta_raw else {}
        if meta:
            print("  metadata: " + "  ".join(
                f"{k}={meta[k]}" for k in
                ("mask_method", "min_visibility_ratio", "sensor_depth_check",
                 "sensor_max_occluded_frac") if k in meta))
        if args.key_cache and args.key_cache.is_file():
            keys = pickle.loads(args.key_cache.read_bytes())
            print(f"LMDB {lmdb_path}: {len(keys)} records (keys from cache {args.key_cache})")
        else:
            print(f"scanning keys in {lmdb_path} (reads the whole DB) ...")
            keys = sorted(k for k in txn.cursor().iternext(keys=True, values=False)
                          if k != b"__metadata__")
            print(f"LMDB {lmdb_path}: {len(keys)} records")
            if args.key_cache:
                args.key_cache.parent.mkdir(parents=True, exist_ok=True)
                args.key_cache.write_bytes(pickle.dumps(keys))
                print(f"cached key list -> {args.key_cache}")

        rng = random.Random(args.seed)
        chosen = rng.sample(keys, min(args.num_samples, len(keys)))

        thumbs, rows, stats = [], [], {"rot": 0, "trans": 0, "other": 0, "failed": 0}
        for i, key in enumerate(chosen, 1):
            rec = pickle.loads(txn.get(key))
            kstr = key.decode()
            panel, thumb, info = render_sample(kstr, rec, args.lmdb_root, args.panel_width)
            if panel is None:
                stats["failed"] += 1
                print(f"  [{i:3d}] FAIL {kstr}: {info.get('error')}")
                continue
            mt = info.get("motion_type", "?")
            stats[mt if mt in ("rot", "trans") else "other"] += 1
            name = f"{i:03d}_{mt}_{kstr.replace('/', '_')}.jpg"
            cv2.imwrite(str(panels_dir / name), panel,
                        [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            thumbs.append(thumb)
            rows.append({"idx": i, "key": kstr, "file": name, **info})
            if i % 10 == 0:
                print(f"  rendered {i}/{len(chosen)}")
    env.close()

    if thumbs:
        tw = 320
        cells = [cv2.resize(t, (tw, int(t.shape[0] * tw / t.shape[1])),
                            interpolation=cv2.INTER_AREA) for t in thumbs]
        th = max(c.shape[0] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, th - c.shape[0], 0, 0,
                                    cv2.BORDER_CONSTANT, value=(22, 22, 22)) for c in cells]
        cols = args.contact_cols
        grid = []
        for r in range(0, len(cells), cols):
            row = cells[r: r + cols]
            while len(row) < cols:
                row.append(np.full_like(cells[0], 22))
            grid.append(np.hstack(row))
        cv2.imwrite(str(args.out_dir / "contact_sheet.jpg"), np.vstack(grid),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    cols = ["idx", "file", "key", "motion_type", "traj_in_frame", "traj_len_m",
            "origin_to_traj0_m", "origin_depth_m", "mask_px", "zoom_factor",
            "depth_valid_frac", "degenerate_rot"]
    with open(args.out_dir / "index.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(
                f"{r.get(c):.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                for c in cols) + "\n")

    print("\n--- summary ---")
    print(f"  panels written : {len(rows)} -> {panels_dir}")
    print(f"  contact sheet  : {args.out_dir / 'contact_sheet.jpg'}")
    print(f"  motion types   : trans={stats['trans']} rot={stats['rot']} "
          f"other={stats['other']} failed={stats['failed']}")
    if rows:
        inframe = [int(r["traj_in_frame"].split("/")[0]) for r in rows]
        print(f"  traj pts in frame : mean {np.mean(inframe):.1f}, "
              f"fully out-of-frame {sum(1 for v in inframe if v == 0)}")
        deg = [r for r in rows if r.get("degenerate_rot")]
        n_rot = max(1, stats["rot"])
        print(f"  degenerate rot arcs: {len(deg)}/{stats['rot']} "
              f"({100 * len(deg) / n_rot:.0f}% of rot) -> straight 0.01 m fallback")
        mp = sorted(r["mask_px"] for r in rows)
        print(f"  mask px    : median {mp[len(mp) // 2]}, min {mp[0]}, max {mp[-1]}")
        dz = sorted(r["origin_depth_m"] for r in rows if "origin_depth_m" in r)
        if dz:
            print(f"  origin depth m: median {dz[len(dz) // 2]:.2f}, "
                  f"min {dz[0]:.2f}, max {dz[-1]:.2f}")
        dv = [r["depth_valid_frac"] for r in rows if "depth_valid_frac" in r]
        if dv:
            print(f"  depth valid in zoom: mean {100 * np.mean(dv):.0f}%, "
                  f"samples <50% valid: {sum(1 for v in dv if v < 0.5)}")

    write_manifest(
        args.out_dir,
        lmdb=str(args.lmdb_path or args.lmdb_root),
        num_samples=args.num_samples, seed=args.seed,
    )


if __name__ == "__main__":
    main()
