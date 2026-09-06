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

Sample construction (v2, 2026-09-06, all 16 categories): ONE record per
interaction window (collaborator CSV windows, KEEP_VERBS) at the window's
first frame with a WiLoR detection of the window's hand (the side with
more detections), carrying the FULL hand trajectory of the window (SF3D
semantics: the image shows the object in its start state; >= 5 detected
frames required). `--per-frame` restores the v1 sampling (a record every
2nd frame with the remaining trajectory). The
mask is the 2Dseg class chosen per window by the VLM sweep
(--selections; single-candidate windows are forced, NONE dropped) — the
palette index is per-part, not per-role, but stable within a video
(verified 2026-09-06). Without --selections the old motion-energy
heuristic runs (v1 behaviour, known-bad: ~57% hand masks). The 2D
trajectory is the MIDDLE-KNUCKLE (MANO joint 9) track in pixels from the
sample frame to the window end; the 3D field is the same joint from
joints_3d_cam (placeholder — every 3D loss is off in the 2D recipe).
Description = the VLM's DESC field when present, else a category/verb
template. motion_info is a content-free stub (2D-only: no motion
supervision). Mask coords are thinned to one representative per --size
grid cell; the reader's scatter reconstruction is unaffected.

Run on the HOI4D-volume pod:
  python3 tools/hoi4d_process_2d.py --ext-root /workspace/ext \
      --hands-root /workspace/hands2973/hands \
      --selections /workspace/vlm_select_v2/selections.json \
      --out /workspace/hoi4d_processed_2d_v2 [--limit N] [--size 512]
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
HAND_COLOR = (0, 128, 0)          # palette index 2 = primary hand, every seq
ANCHOR_JOINT = 9                  # MANO middle-finger MCP (verified 2026-09-06)
MIN_WINDOW_FRAMES = 6

# Official fine-grained verbs kept as interaction windows (user-approved
# 2026-09-06 after the verb survey). Dropped: rest / Reachout / Stop /
# Grasp / go / Lookaround (no manipulation), carry / Carrywithbothhands
# (hand path dominated by the person walking), and — after the v2 review —
# cut / paper-cut / binding (the VLM split between the tool and the
# material being cut/stapled; dropped rather than fixed) and Pickup /
# putdown (75% of windows but short, near-static lifts — "nothing
# interesting to train on", user 2026-09-06).
KEEP_VERBS = {
    "open", "close", "Press", "push", "pull", "turn&on&the&switch",   # articulated
    "dump",                                                           # pouring
}
# The verb set the 2026-09-06 VLM sweep ENUMERATED windows with. Window
# indices in selections.json ("<seq>|<wi>") are positions in THIS list, so
# it is FROZEN: windows are always enumerated with SWEEP_VERBS and then
# filtered by KEEP_VERBS (a 2026-09-06 rebuild that enumerated with the
# reduced set mis-assigned masks — caught by the event check below).
SWEEP_VERBS = KEEP_VERBS | {"paper-cut", "binding", "cut", "Pickup", "putdown"}
# trajectory sanity (v2 review: 0.2% of records had a WiLoR outlier —
# knuckle far from the mask or jumping between frames)
MAX_START_TO_MASK_PX = 300.0
MAX_FRAME_JUMP_PX = 300.0
VERB_PHRASE = {"turn&on&the&switch": "switch on", "paper-cut": "cut with",
               "binding": "staple with", "Pickup": "pick up", "putdown": "put down",
               "Press": "press", "dump": "pour from"}
CATEGORY_NAMES = {
    "C1": "toy car", "C2": "mug", "C3": "laptop", "C4": "storage furniture",
    "C5": "bottle", "C6": "safe", "C7": "bowl", "C8": "bucket", "C9": "scissors",
    "C11": "pliers", "C12": "kettle", "C13": "knife", "C14": "trash can",
    "C17": "lamp", "C18": "stapler", "C20": "chair",
}


def seq_to_relpath(seq: str) -> str:
    cam, h, c, n, s, room, t = seq.split("_")
    return f"{cam}/{h}/{c}/{n}/{s}/{room}/{t}"


def load_segments_csv(path: Path) -> dict:
    """Collaborator's hoi4d_action_segments.csv -> {seq: [(verb, f0, f1)]}
    enumerated with SWEEP_VERBS (index-stable vs selections.json); the
    KEEP_VERBS filter is applied per window in process_sequence. Frames are
    already 0-based 15 fps."""
    import csv
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["event"] not in SWEEP_VERBS:
                continue
            f0 = max(0, int(r["start_frame_15fps"]))
            f1 = min(299, int(r["end_frame_15fps"]))
            if f1 - f0 >= MIN_WINDOW_FRAMES:
                out.setdefault(r["sequence_id"], []).append((r["event"], f0, f1))
    return out


def template_description(cat: str, verb: str) -> str:
    return f"{VERB_PHRASE.get(verb, verb)} the {CATEGORY_NAMES.get(cat, 'object')}"


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
    # Two coexisting formats in the release: {"events": [{event, startTime,
    # endTime}]} and {"markResult": {"marks": [{event, hdTimeStart,
    # hdTimeEnd}]}} (8/354 furniture seqs use the second).
    if "events" in data:
        evs = [(e["event"], e["startTime"], e["endTime"]) for e in data["events"]]
    else:
        evs = [(m["event"], m["hdTimeStart"], m["hdTimeEnd"])
               for m in data["markResult"]["marks"]]
    video_dur = 300.0 / FPS
    dur = (data.get("info") or {}).get("duration")
    if not dur:
        max_end = max((t1 for _, _, t1 in evs), default=video_dur)
        dur = 10.0 if max_end < 12.0 else video_dur
    scale = video_dur / float(dur)
    out = []
    for name, t0, t1 in evs:
        name = name.strip().lower()
        if name in INTERACTION_EVENTS:
            f0 = max(0, int(np.ceil(t0 * scale * FPS)))
            f1 = min(299, int(np.floor(t1 * scale * FPS)))
            if f1 - f0 >= 6:
                out.append((name, f0, f1))
    return out, scale


def hand_colors(mask_dir: Path, frame_wrists, radius: int = 22,
                min_hit_frac: float = 0.4):
    """2Dseg classes that ARE the hand: colors whose mask reaches within
    `radius` px of the WiLoR wrist in >= `min_hit_frac` of the checked
    frames. The hand class identifies itself this way — the 2026-09-01
    build's motion-energy pick chose it as "the moving part" in ~57% of
    records (audit 2026-09-02), because the hand IS the most-changing
    class during a manipulation window.

    frame_wrists: iterable of (frame_idx, (x_px, y_px)).
    """
    hits = {tuple(c): 0 for c in PART_COLORS}
    checked = 0
    for f, (wx, wy) in frame_wrists:
        m = cv2.imread(str(mask_dir / f"{f:05d}.png"))
        if m is None:
            continue
        checked += 1
        y0, y1 = max(0, int(wy) - radius), int(wy) + radius + 1
        x0, x1 = max(0, int(wx) - radius), int(wx) + radius + 1
        patch = m[y0:y1, x0:x1]
        for col in PART_COLORS:
            if (patch == col).all(-1).any():
                hits[tuple(col)] += 1
    if checked == 0:
        return set()
    return {c for c, h in hits.items() if h / checked >= min_hit_frac}


def moving_color(mask_dir: Path, f0: int, f1: int, exclude=()):
    """The part color whose mask changes most across the window,
    excluding `exclude` (the hand classes from hand_colors)."""
    a = cv2.imread(str(mask_dir / f"{f0:05d}.png"))
    b = cv2.imread(str(mask_dir / f"{f1:05d}.png"))
    if a is None or b is None:
        return None
    best, best_score = None, 0.0
    for col in PART_COLORS:
        if tuple(col) in exclude:
            continue
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


def process_sequence(seq: str, ext: Path, hands_root: Path, size: int,
                     selections: dict | None = None,
                     segments: dict | None = None,
                     per_frame: bool = False):
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
    # Camera 1 ships 2Dseg/mask/; cameras 2-4 ship 2Dseg/shift_mask/
    # (masks re-aligned to align_rgb) — same 300-frame palette format.
    mask_dir = ann / "2Dseg" / "mask"
    if not mask_dir.exists():
        mask_dir = ann / "2Dseg" / "shift_mask"
    if not mask_dir.exists():
        return None, "missing-2dseg"
    if segments is not None and seq in segments:
        windows = segments[seq]
    else:
        # fallback for sequences absent from the collaborator CSV (the one
        # such seq, ZY20210800003_H3_C3_N44_S284_s01_T2, ships an EMPTY
        # action file upstream -> no windows)
        try:
            windows, time_scale = load_windows(action)
        except (KeyError, ValueError):
            return None, "no-windows"
        if time_scale != 1.0:
            print(f"  {seq}: action time scale {time_scale:g}", flush=True)
    if not windows:
        return None, "no-windows"

    h = np.load(hands_npz)
    K = np.load(intr).astype(np.float32)
    sides = {}
    for name, flag in (("right", 1), ("left", 0)):
        sel = h["is_right"] == flag
        fr0 = h["frame_number"][sel].astype(int) - 1        # -> 0-based
        j2d = h["joints_2d"][sel][:, ANCHOR_JOINT, :]        # knuckle px
        j3c = h["joints_3d_cam"][sel][:, ANCHOR_JOINT, :]
        order = np.argsort(fr0)
        fr0, j2d, j3c = fr0[order], j2d[order], j3c[order]
        sides[name] = (j2d, j3c, {int(f): i for i, f in enumerate(fr0)})

    samples = []
    needed_frames = set()
    for wi, (event, f0, f1) in enumerate(windows):
        if event not in KEEP_VERBS:
            continue
        if selections is not None and f"{seq}|{wi}" in selections:
            sel_event = selections[f"{seq}|{wi}"].get("event")
            if sel_event is not None and sel_event != event:
                raise RuntimeError(
                    f"{seq} window {wi}: selection event {sel_event!r} != CSV "
                    f"event {event!r} — window enumeration drifted from the sweep")
        # per-window hand: the VLM's HAND field when it names a side that
        # WiLoR actually detected in the window, else the side with more
        # detections
        ndet = {s: sum(f in sides[s][2] for f in range(f0, f1 + 1)) for s in sides}
        hand = max(ndet, key=ndet.get)
        vlm_hand = (selections or {}).get(f"{seq}|{wi}", {}).get("hand")
        if vlm_hand in ndet and ndet[vlm_hand] >= 5:
            hand = vlm_hand
        j2d, j3c, frame_to_i = sides[hand]
        if selections is not None:
            # VLM-chosen part (tools/hoi4d_vlm_select_all.py); windows the
            # VLM declined (NONE/ERROR) are dropped rather than guessed.
            sel = selections.get(f"{seq}|{wi}")
            # v2 records carry "colors" (several parts = whole object for
            # pick/put-down); v1 records only "color".
            cols = (sel.get("colors") or ([sel["color"]] if sel.get("color") else [])) if sel else []
            col = [tuple(c) for c in cols] or None
        else:
            det = [f for f in range(f0, f1 + 1) if f in frame_to_i]
            probe = det[:: max(1, len(det) // 5)][:5]
            wrists = [(f, tuple(j2d[frame_to_i[f]])) for f in probe]
            col = moving_color(mask_dir, f0, f1,
                               exclude=hand_colors(mask_dir, wrists))
        if col is None:
            continue
        desc = (sel.get("desc") if selections is not None and sel else None) \
            or template_description(cat, event)
        wf = [f for f in range(f0, f1 + 1) if f in frame_to_i]
        # ONE sample per window (user decision 2026-09-06, SF3D semantics:
        # the frame shows the object in its start state and carries the
        # FULL hand trajectory of the window). --per-frame restores the
        # v1 stride-2 suffix sampling.
        starts = wf[::2] if per_frame else wf[:1]
        for f in starts:
            future = [g for g in wf if g >= f]
            if len(future) < 5:
                continue
            samples.append((wi, event, f, future, col, hand, desc, j2d, j3c, frame_to_i))
            needed_frames.add(f)
    if not samples:
        return None, "no-samples"

    rgb = read_video_frames(rgb_mp4, needed_frames)
    depth = (read_video_frames(depth_avi, needed_frames, raw=True)
             if depth_avi.exists() else {})

    cam, hh, c, n, s, room, t = seq.split("_")
    records, frames = {}, {}
    for wi, event, f, future, col, hand, desc, j2d, j3c, frame_to_i in samples:
        if f not in rgb:
            continue
        m = cv2.imread(str(mask_dir / f"{f:05d}.png"))
        if m is None:
            continue
        cols = col if isinstance(col, list) else [col]
        part = np.zeros(m.shape[:2], bool)
        for part_col in cols:
            part |= (m == part_col).all(-1)
        coords = thin_coords(part, size)
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
        tr = j2d[idxs]
        if (np.sqrt(((coords[:, ::-1] - tr[0]) ** 2).sum(1)).min() > MAX_START_TO_MASK_PX
                or (len(tr) > 1 and np.sqrt((np.diff(tr, axis=0) ** 2).sum(1)).max() > MAX_FRAME_JUMP_PX)):
            continue                      # WiLoR outlier (see constants above)
        records[key] = {
            "rgb_image_path": key,
            "mask_coordinates_yx": coords.tolist(),
            "description": desc,
            "camera_intrinsics": K.tolist(),
            # 2D-only training carries NO motion supervision (user decision
            # 2026-09-06); the reader indexes this key unconditionally, so
            # a content-free stub stays.
            "motion_info": {
                "frame_specific_motion_data": {
                    "motion_origin_2d_image_coords": [0.0, 0.0],
                    "motion_dir_3d_camera_coords": [0.0, 0.0, 0.0],
                    "motion_origin_3d_camera_coords": [0.0, 0.0, 0.0],
                },
                "original_motion_data": {"motion_type": "trans"},
            },
            "trajectory_3d_camera_coords": j3c[idxs].tolist(),
            "trajectory_2d_image_coords": j2d[idxs].tolist(),
            "trajectory_2d_valid": [True] * len(idxs),
            "hoi4d": {"seq": seq, "event": event, "window": wi,
                      "category": cat, "frame_0based": f, "hand": hand,
                      "anchor_joint": ANCHOR_JOINT},
        }
        frames[key] = encode_frame(rgb[f], dep_u16, size)
    if not records:
        return None, "no-records"
    return (records, frames), "ok"


def merge_shards(out: Path, size: int):
    """Copy every record of <out>/shard_*/{data,frames}.lmdb into
    <out>/{data,frames}.lmdb (skips each shard's __metadata__)."""
    import lmdb
    env_d = lmdb.open(str(out / "data.lmdb"), map_size=1 << 38)
    env_f = lmdb.open(str(out / "frames.lmdb"), map_size=1 << 40)
    n = 0
    for d in sorted(out.glob("shard_*")):
        for name, env in (("data.lmdb", env_d), ("frames.lmdb", env_f)):
            src = lmdb.open(str(d / name), readonly=True, lock=False)
            with src.begin() as rt, env.begin(write=True) as wt:
                for k, v in rt.cursor():
                    if k == b"__metadata__":
                        continue
                    wt.put(k, v)
                    if name == "data.lmdb":
                        n += 1
            src.close()
        print(f"merged {d.name}: total records {n}", flush=True)
    with env_f.begin(write=True) as txn:
        txn.put(b"__metadata__", pickle.dumps(
            {"entries": n, "depth_size": size, "jpeg_quality": 92,
             "source": "hoi4d_process_2d"}, protocol=4))
    print("MERGE DONE", n, "records", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-root", default=None, help="required unless --merge-shards")
    ap.add_argument("--hands-root", default=None, help="required unless --merge-shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selections", default=None,
                    help="selections.json from hoi4d_vlm_select_all.py")
    ap.add_argument("--segments-csv", default=None,
                    help="collaborator hoi4d_action_segments.csv (default: "
                         "<hands-root>/../hoi4d_action_segments.csv); JSON "
                         "action files are the fallback per sequence")
    ap.add_argument("--shard", default=None,
                    help="'i/n': process seqs[i::n] into --out (run n "
                         "processes with --out <root>/shard_i, then --merge-shards)")
    ap.add_argument("--merge-shards", action="store_true",
                    help="merge <out>/shard_*/{data,frames}.lmdb into <out>/ and exit")
    ap.add_argument("--per-frame", action="store_true",
                    help="v1 sampling: a record every 2nd frame of the window "
                         "with the remaining trajectory (default: ONE record per "
                         "window at its first frame with the full trajectory)")
    args = ap.parse_args()

    import lmdb
    if args.merge_shards:
        merge_shards(Path(args.out), args.size)
        return
    if not (args.ext_root and args.hands_root):
        ap.error("--ext-root and --hands-root are required")
    selections = None
    if args.selections:
        selections = json.load(open(args.selections))
        print("using", len(selections), "VLM selections", flush=True)
    ext, hands_root = Path(args.ext_root), Path(args.hands_root)
    seg_csv = Path(args.segments_csv) if args.segments_csv \
        else hands_root.parent / "hoi4d_action_segments.csv"
    segments = load_segments_csv(seg_csv) if seg_csv.exists() else None
    print("segments csv:", seg_csv if segments is not None else "NONE (json fallback)",
          "windows:", sum(len(v) for v in (segments or {}).values()), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p.name for p in hands_root.iterdir() if p.is_dir())
    if args.limit:
        seqs = seqs[: args.limit]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        seqs = seqs[i::n]

    env_d = lmdb.open(str(out / "data.lmdb"), map_size=1 << 36)
    env_f = lmdb.open(str(out / "frames.lmdb"), map_size=1 << 38)
    stats, n_rec = {}, 0
    for i, seq in enumerate(seqs):
        try:
            got, status = process_sequence(seq, ext, hands_root, args.size,
                                           selections=selections,
                                           segments=segments,
                                           per_frame=args.per_frame)
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
