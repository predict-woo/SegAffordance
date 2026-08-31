"""Build SF3D-format LMDBs from HOI4D furniture sequences (2D-only training).

Joins three sources (spec: docs/superpowers/specs/2026-08-31-hoi4d-2d-
training-design.md):
  * extracted raw HOI4D (--ext-root): HOI4D_release/.../align_rgb/image.mp4,
    HOI4D_annotations/.../{2Dseg/mask/*.png, action/color.json},
    HOI4D_depth_video/.../align_depth/depth_video.avi
  * the WiLoR hands package (--hands-root): <seq>/wilor/hands.npz +
    <seq>/camera/intrinsic.npy (frame_number is 1-BASED)

Emits records readable by datasets/scenefun3d.py:SF3DDataset unmodified:
  data.lmdb    pickled dicts keyed "<cam>_<H>_<C>_<N>/<S>_<s>_<T>_w<i>_f<frame>"
               (scene prefix before '/' = the physical object, so
               split_dataset_by_scene keeps an object in one split)
  frames.lmdb  {"jpeg", "depth_png" (16-bit mm), "orig_size"} at --size

Sample construction: one record per stride-2 frame inside an open/close
action window with a right-hand WiLoR detection and >= --min-future
detected wrist frames remaining in the window. The mask is the 2Dseg
color whose area changes most over the window (the moving part — color
indices are per-part, not per-role; verified on the kitted sequences,
notes 2026-08-31). The 2D trajectory is the wrist track (pixels) from
the sample frame to the window end; the 3D trajectory field is filled
with WiLoR joints_3d_cam wrist (REAL xy / NOISY z — placeholder for the
reader's non-empty requirement; every 3D loss is off in the 2D recipe).
Type labels (C4=trans drawer, C6=rot safe door) are written for EVAL
only. Mask coords are thinned to one representative per --size grid
cell to keep records small; the reader's scatter reconstruction is
unaffected at training resolution.

Run on the extraction pod:
  python3 tools/hoi4d_process_2d.py --ext-root /workspace/ext \
      --hands-root /workspace/hands/all_354_furniture_hands \
      --out /workspace/hoi4d_processed_2d [--limit N] [--size 512]
"""

import argparse
import io
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

FPS = 15.0
INTERACTION_EVENTS = {"open", "close", "pull", "push", "pullout", "pushin"}
PART_COLORS = [(0, 0, 128), (0, 128, 0), (0, 128, 128), (128, 0, 0),
               (128, 128, 0), (128, 0, 128), (0, 0, 64), (64, 0, 0)]

def seq_to_relpath(seq: str) -> str:
    cam, h, c, n, s, room, t = seq.split("_")
    return f"{cam}/{h}/{c}/{n}/{s}/{room}/{t}"


def load_windows(action_json: Path):
    """Interaction windows as 0-based frame ranges.

    Some HOI4D action JSONs run on a 10-second clock while every video
    is 300 frames / 15 fps = 20.0 s (13/354 furniture seqs have
    info.duration == 10.0, 8 lack the field; found by ethz-workspace-17
    2026-09-01, frame-verified on a C6 seq). Scale event times by
    video_duration / info.duration; when the field is missing, infer
    from the last event's end (tiles ~0-10 or ~0-20).
    """
    data = json.load(open(action_json))
    evs = data["events"]
    video_dur = 300.0 / FPS
    dur = (data.get("info") or {}).get("duration")
    if not dur:
        max_end = max((e["endTime"] for e in evs), default=video_dur)
        dur = 10.0 if max_end < 12.0 else video_dur
    scale = video_dur / float(dur)
    out = []
    for e in evs:
        name = e["event"].strip().lower()
        if name in INTERACTION_EVENTS:
            f0 = max(0, int(np.ceil(e["startTime"] * scale * FPS)))
            f1 = min(299, int(np.floor(e["endTime"] * scale * FPS)))
            if f1 - f0 >= 6:
                out.append((name, f0, f1))
    return out, scale


def moving_color(mask_dir: Path, f0: int, f1: int):
    """The part color whose mask changes most across the window."""
    a = cv2.imread(str(mask_dir / f"{f0:05d}.png"))
    b = cv2.imread(str(mask_dir / f"{f1:05d}.png"))
    if a is None or b is None:
        return None
    best, best_score = None, 0.0
    for col in PART_COLORS:
        ma = (a == col).all(-1)
        mb = (b == col).all(-1)
        area = max(ma.sum(), mb.sum())
        if area < 2000:
            continue
        change = np.logical_xor(ma, mb).sum() / float(area)
        if change > best_score:
            best, best_score = col, change
    return best


def thin_coords(mask: np.ndarray, grid: int):
    """Original-res (y, x) coords restricted to the reader's gather grid.

    SF3DDataset's fast path reconstructs the mask by GATHERING one source
    pixel per target cell (src = floor((dst+0.5)*scale), scenefun3d.py
    fast_pipeline block) — coords stored anywhere else are invisible. So
    store exactly the set pixels the gather will sample: bit-identical
    downsampled mask at ~mask_frac * grid^2 coords instead of the full
    original-res splat.
    """
    h, w = mask.shape
    r_idx = np.minimum(((np.arange(grid) + 0.5) * (h / grid)).astype(np.int64), h - 1)
    c_idx = np.minimum(((np.arange(grid) + 0.5) * (w / grid)).astype(np.int64), w - 1)
    sub = mask[np.ix_(r_idx, c_idx)]
    ys, xs = np.nonzero(sub)
    if len(ys) == 0:
        return None
    return np.stack([r_idx[ys], c_idx[xs]], axis=1).astype(np.int32)


def encode_frame(bgr, depth_u16, size):
    ok, jpeg = cv2.imencode(
        ".jpg", cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA),
        [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    dep = cv2.resize(depth_u16, (size, size), interpolation=cv2.INTER_NEAREST)
    ok, dpng = cv2.imencode(".png", dep)
    assert ok
    return {"jpeg": jpeg.tobytes(), "depth_png": dpng.tobytes(),
            "orig_size": (bgr.shape[1], bgr.shape[0])}


def description_for(cat: str, event: str) -> str:
    if cat == "C4":
        return {"open": "open the drawer", "pull": "pull out the drawer",
                "close": "close the drawer", "push": "push in the drawer"
                }.get(event, f"{event} the drawer")
    return {"open": "open the safe door",
            "close": "close the safe door"}.get(event, f"{event} the safe door")


def read_video_frames(path: Path, wanted: set, raw: bool = False):
    """Decode only wanted frame indices (0-based) from a video.

    raw=True disables BGR conversion — HOI4D depth is 16-bit FFV1 and
    decodes natively to a (H, W) uint16 millimetre map this way
    (verified 2026-08-31: centre pixel 765 mm at a drawer).
    """
    cap = cv2.VideoCapture(str(path))
    if raw:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    out, idx = {}, 0
    last = max(wanted) if wanted else -1
    while idx <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in wanted:
            out[idx] = fr
        idx += 1
    cap.release()
    return out


def process_sequence(seq: str, ext: Path, hands_root: Path, size: int):
    rel = seq_to_relpath(seq)
    cat = seq.split("_")[2]
    ann = ext / "HOI4D_annotations" / rel
    rgb_mp4 = ext / "HOI4D_release" / rel / "align_rgb" / "image.mp4"
    depth_avi = ext / "HOI4D_depth_video" / rel / "align_depth" / "depth_video.avi"
    hands_npz = hands_root / seq / "wilor" / "hands.npz"
    intr = hands_root / seq / "camera" / "intrinsic.npy"
    if not (ann.exists() and rgb_mp4.exists() and hands_npz.exists() and intr.exists()):
        return None, "missing-input"
    action = ann / "action" / "color.json"
    if not action.exists():
        return None, "missing-action"
    windows, time_scale = load_windows(action)
    if time_scale != 1.0:
        print(f"  {seq}: action time scale {time_scale:g}", flush=True)
    if not windows:
        return None, "no-windows"

    h = np.load(hands_npz)
    right = h["is_right"] == 1
    fr0 = h["frame_number"][right].astype(int) - 1          # -> 0-based
    j2d = h["joints_2d"][right][:, 0, :]                    # wrist px
    j3c = h["joints_3d_cam"][right][:, 0, :]
    order = np.argsort(fr0)
    fr0, j2d, j3c = fr0[order], j2d[order], j3c[order]
    frame_to_i = {int(f): i for i, f in enumerate(fr0)}     # last det wins
    K = np.load(intr).astype(np.float32)

    samples = []
    needed_frames = set()
    for wi, (event, f0, f1) in enumerate(windows):
        col = moving_color(ann / "2Dseg" / "mask", f0, f1)
        if col is None:
            continue
        wf = [f for f in range(f0, f1 + 1) if f in frame_to_i]
        for f in wf[::2]:
            future = [g for g in wf if g >= f]
            if len(future) < 5:
                continue
            samples.append((wi, event, f, future, col))
            needed_frames.add(f)
    if not samples:
        return None, "no-samples"

    rgb = read_video_frames(rgb_mp4, needed_frames)
    depth = (read_video_frames(depth_avi, needed_frames, raw=True)
             if depth_avi.exists() else {})

    cam, hh, c, n, s, room, t = seq.split("_")
    records, frames = {}, {}
    for wi, event, f, future, col in samples:
        if f not in rgb:
            continue
        m = cv2.imread(str(ann / "2Dseg" / "mask" / f"{f:05d}.png"))
        if m is None:
            continue
        coords = thin_coords((m == col).all(-1), size)
        if coords is None or len(coords) < 50:
            continue
        dep = depth.get(f)
        if dep is None:
            dep_u16 = np.zeros(rgb[f].shape[:2], np.uint16)
        elif dep.ndim == 2 and dep.dtype == np.uint16:
            dep_u16 = dep                     # native FFV1 16-bit mm map
        else:
            raise ValueError(
                f"{seq} f{f}: unexpected depth decode "
                f"{dep.shape}/{dep.dtype} — expected (H,W) uint16"
            )
        key = f"{cam}_{hh}_{c}_{n}/{s}_{room}_{t}_w{wi}_f{f:03d}"
        idxs = [frame_to_i[g] for g in future]
        records[key] = {
            "rgb_image_path": key,
            "mask_coordinates_yx": coords.tolist(),
            "description": description_for(cat, event),
            "camera_intrinsics": K.tolist(),
            "motion_info": {
                "frame_specific_motion_data": {
                    "motion_origin_2d_image_coords": [0.0, 0.0],
                    "motion_dir_3d_camera_coords": [0.0, 0.0, 0.0],
                    "motion_origin_3d_camera_coords": [0.0, 0.0, 0.0],
                },
                "original_motion_data": {
                    "motion_type": "trans" if cat == "C4" else "rot",
                },
            },
            "trajectory_3d_camera_coords": j3c[idxs].tolist(),
            "trajectory_2d_image_coords": j2d[idxs].tolist(),
            "trajectory_2d_valid": [True] * len(idxs),
            "hoi4d": {"seq": seq, "event": event, "window": wi,
                      "category": cat, "wrist_frame_0based": f},
        }
        frames[key] = encode_frame(rgb[f], dep_u16, size)
    if not records:
        return None, "no-records"
    return (records, frames), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-root", required=True)
    ap.add_argument("--hands-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import lmdb
    ext, hands_root = Path(args.ext_root), Path(args.hands_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p.name for p in hands_root.iterdir() if p.is_dir())
    if args.limit:
        seqs = seqs[: args.limit]

    env_d = lmdb.open(str(out / "data.lmdb"), map_size=1 << 36)
    env_f = lmdb.open(str(out / "frames.lmdb"), map_size=1 << 38)
    stats, n_rec = {}, 0
    for i, seq in enumerate(seqs):
        try:
            got, status = process_sequence(seq, ext, hands_root, args.size)
        except Exception as e:  # keep going; report at the end
            got, status = None, f"error:{type(e).__name__}"
        stats[status] = stats.get(status, 0) + 1
        if got is None:
            print(f"[{i+1}/{len(seqs)}] {seq}: {status}", flush=True)
            continue
        records, frames = got
        with env_d.begin(write=True) as txn:
            for k, v in records.items():
                txn.put(k.encode(), pickle.dumps(v, protocol=4))
        with env_f.begin(write=True) as txn:
            for k, v in frames.items():
                txn.put(k.encode(), pickle.dumps(v, protocol=4))
        n_rec += len(records)
        print(f"[{i+1}/{len(seqs)}] {seq}: {len(records)} records "
              f"(total {n_rec})", flush=True)
    with env_f.begin(write=True) as txn:
        txn.put(b"__metadata__", pickle.dumps(
            {"entries": n_rec, "depth_size": args.size, "jpeg_quality": 92,
             "source": "hoi4d_process_2d"}, protocol=4))
    print("DONE", n_rec, "records;", stats)


if __name__ == "__main__":
    main()
