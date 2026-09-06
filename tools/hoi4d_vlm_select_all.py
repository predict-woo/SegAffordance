"""VLM part selection + description for every interaction window of the
FULL HOI4D package (2,973 seqs, 16 categories) — v2 of the furniture sweep.

Per window the VLM (gpt-5.6-luna via codex app-server, Set-of-Mark
composite, build_composite_v2) answers three fields:
  ANSWER: <n>[,<n>..] | NONE  segment(s) the hand acts on (several = whole
                              object for pick-up/put-down; builder unions)
  DESC: <sentence>            short imperative for the window (piggybacked so
                              descriptions cost no extra call)
  HAND: left | right | both   acting hand (builder uses it to pick the WiLoR
                              side; falls back to most detections)
Windows come from the collaborator CSV (hoi4d_process_2d.KEEP_VERBS).
Single-candidate windows can skip the VLM (--forced-desc template) or be
asked anyway for an image-grounded description (--forced-desc vlm).

Output: selections.json mapping "<seq>|<window_idx>" ->
  {"color": [b,g,r] | null, "answer": "<n>|NONE|FORCED|ERROR|NOMASK|NORGB",
   "desc": str | null, "event", "f0", "f1", "raw"}
NONE/ERROR windows get color null (the rebuild drops them).

Phases (never share a small pod's CPUs between decode and codex):
  python3 hoi4d_vlm_select_all.py --prepare-only --out /workspace/vlm_select_v2
  python3 hoi4d_vlm_select_all.py --consume-jobs --workers 14 --fast \
      --out /workspace/vlm_select_v2
Resume-safe: existing entries in selections.json are not re-asked.
"""
import argparse
import csv
import faulthandler
import json
import os
import queue
import re
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codex_client import CodexClient, CodexError
from hoi4d_process_2d import (CATEGORY_NAMES, HAND_COLOR, PART_COLORS,
                              load_segments_csv, load_windows,
                              read_video_frames, seq_to_relpath)
from hoi4d_vlm_part_pilot import CODEX_CWD, interior_point

MIN_AREA = 2000
TINT = {1: (60, 60, 230), 2: (230, 160, 60), 3: (60, 200, 230),
        4: (200, 60, 200), 5: (40, 220, 220), 6: (230, 60, 230),
        7: (120, 120, 230), "H": (60, 230, 60)}

PROMPT = """These four images show a person interacting with a {category} \
(official action label: "{event}"). Top row: the first and last video frames \
of the action, with the zoom region outlined. Bottom row: a zoomed crop of \
the same two frames with segmentation overlays — each candidate segment \
carries a NUMBER (a short line connects a number to its segment when they \
are small); the person's main hand/arm is labeled H. A numbered segment may \
also be the person's OTHER hand, or an unrelated object in the scene (a \
sheet of paper, a second container, an item being carried) — never choose \
those.

Task 1 — which numbered segment(s) is the hand directly acting on in this \
action? If the action operates a PART of the object (opening, closing, \
pressing, switching, pulling out — e.g. a drawer, lid, door, button, \
screen), answer ONLY that moving part's number. If the whole object is \
lifted, set down, poured, or used as a tool (Pickup, putdown, dump, cut), \
answer ALL numbers that belong to that object, comma-separated. Compare the \
two frames to see what changed.

Task 2 — write ONE short imperative sentence (max 12 words) describing what \
to do to that object/part, naming it with a visual or spatial qualifier if \
several similar ones are visible (e.g. "pull open the top drawer", \
"pick up the blue mug on the left"). Describe the goal, not the motion \
mechanics or direction; no mention of numbers, overlay colors, or hands.

Task 3 — which of the person's hands performs the action: their LEFT or \
RIGHT hand (the camera is worn on the person's head, so the image left is \
the person's left), or BOTH if both hands act on the object together.

Reply in exactly this format, nothing else:
ANSWER: <number(s) or NONE>
DESC: <sentence>
HAND: <left, right, or both>"""


def overlay_v2(img, m, mapping):
    """Tint + outline every mapped class; place labels at interior points,
    pushing colliding labels apart (with a leader line)."""
    ov = img.copy()
    pts = []
    for col, label in mapping.items():
        mask = (m == np.array(col)).all(-1)
        if not mask.any():
            continue
        t = TINT.get(label, (180, 180, 180))
        ov[mask] = ov[mask] * 0.45 + np.array(t) * 0.55
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cs, -1, t, 3)
        pts.append((label, interior_point(mask)))
    h, w = ov.shape[:2]
    placed = []
    for label, (x, y) in pts:
        lx, ly = x, y
        for _ in range(12):
            clash = [p for p in placed if abs(p[0] - lx) < 110 and abs(p[1] - ly) < 90]
            if not clash:
                break
            px, py = clash[0]
            dx, dy = lx - px, ly - py
            n = (dx * dx + dy * dy) ** 0.5 or 1.0
            if n < 1.5:
                dx, dy, n = 1.0, -1.0, 2 ** 0.5
            lx, ly = int(lx + dx / n * 110), int(ly + dy / n * 90)
        lx, ly = int(np.clip(lx, 40, w - 60)), int(np.clip(ly, 60, h - 20))
        placed.append((lx, ly))
        if (lx, ly) != (x, y):
            cv2.line(ov, (x, y), (lx, ly), (0, 0, 0), 6)
            cv2.line(ov, (x, y), (lx, ly), (255, 255, 255), 2)
        for th, c in ((10, (0, 0, 0)), (4, (255, 255, 255))):
            cv2.putText(ov, str(label), (lx - 25, ly + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.6, c, th)
    return ov


def zoom_box(masks, shape, margin=0.35, min_w=720):
    """16:9 crop box around the union of the given masks (both frames)."""
    h, w = shape[:2]
    ys, xs = np.nonzero(np.any(masks, axis=0)) if masks else (np.array([0, h - 1]), np.array([0, w - 1]))
    if len(ys) == 0:
        return 0, 0, w, h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    bw = max(bw, min_w, bh * 16 / 9); bh = max(bh, bw * 9 / 16)
    bw, bh = min(bw, w), min(bh, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0 = int(np.clip(cx - bw / 2, 0, w - bw)); Y0 = int(np.clip(cy - bh / 2, 0, h - bh))
    return X0, Y0, int(bw), int(bh)


def build_composite_v2(rgb_a, rgb_b, ma, mb, mapping, title):
    """2x2 grid: top = full first/last frames with the zoom box drawn,
    bottom = zoomed overlays with numbered segments. Output 1920x1080."""
    # zoom on the candidate parts only — the hand/arm mask always runs to
    # the frame border and would defeat the crop
    masks = []
    for m in (ma, mb):
        for col, label in mapping.items():
            if label != "H":
                masks.append((m == np.array(col)).all(-1))
    X0, Y0, bw, bh = zoom_box(masks, ma.shape)
    a, b = rgb_a.copy(), rgb_b.copy()
    for im in (a, b):
        cv2.rectangle(im, (X0, Y0), (X0 + bw, Y0 + bh), (0, 0, 0), 6)
        cv2.rectangle(im, (X0, Y0), (X0 + bw, Y0 + bh), (255, 255, 255), 2)
    oa = overlay_v2(rgb_a, ma, mapping)[Y0:Y0 + bh, X0:X0 + bw]
    ob = overlay_v2(rgb_b, mb, mapping)[Y0:Y0 + bh, X0:X0 + bw]
    tiles = []
    for im, lab in ((a, f"FIRST frame ({title})"), (b, "LAST frame"),
                    (oa, "FIRST, zoomed + segments"), (ob, "LAST, zoomed + segments")):
        im = cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.putText(im, lab, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,) * 3, 6)
        cv2.putText(im, lab, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,) * 3, 2)
        tiles.append(im)
    return np.concatenate([np.concatenate(tiles[:2], 1),
                           np.concatenate(tiles[2:], 1)], 0)


def prepare_sequence(seq, ext, out, existing, segments, forced_desc):
    """Return (jobs, forced, skipped) for one sequence's windows."""
    rel = seq_to_relpath(seq)
    cat = seq.split("_")[2]
    catname = CATEGORY_NAMES.get(cat, "object")
    ann = ext / "HOI4D_annotations" / rel
    mask_dir = ann / "2Dseg" / "mask"
    if not mask_dir.exists():
        mask_dir = ann / "2Dseg" / "shift_mask"
    if not mask_dir.exists():
        return [], {}, 1
    if seq in segments:
        windows = segments[seq]
    else:
        action = ann / "action" / "color.json"
        try:
            windows, _ = load_windows(action) if action.exists() else ([], 1.0)
        except Exception:
            windows = []
    jobs, forced = [], {}
    need_frames, metas = set(), []
    for wi, (event, f0, f1) in enumerate(windows):
        key = f"{seq}|{wi}"
        if key in existing:
            continue
        base = {"event": event, "f0": f0, "f1": f1, "desc": None}
        ma = cv2.imread(str(mask_dir / f"{f0:05d}.png"))
        mb = cv2.imread(str(mask_dir / f"{f1:05d}.png"))
        if ma is None or mb is None:
            forced[key] = {"color": None, "answer": "NOMASK", **base}
            continue
        mapping, cands = {}, []
        for col in PART_COLORS:
            area = max((ma == col).all(-1).sum(), (mb == col).all(-1).sum())
            if area < MIN_AREA:
                continue
            if tuple(col) == HAND_COLOR:
                mapping[tuple(col)] = "H"
            else:
                cands.append(col)
                mapping[tuple(col)] = len(cands)
        if not cands:
            forced[key] = {"color": None, "answer": "NONE", **base}
        elif len(cands) == 1 and forced_desc == "template":
            forced[key] = {"color": list(cands[0]), "colors": [list(cands[0])],
                           "answer": "FORCED", **base}
        else:
            metas.append((key, wi, event, f0, f1, ma, mb, mapping, len(cands)))
            need_frames.update((f0, f1))
    if metas:
        rgbs = read_video_frames(
            ext / "HOI4D_release" / rel / "align_rgb" / "image.mp4",
            need_frames)
        for key, wi, event, f0, f1, ma, mb, mapping, ncand in metas:
            if f0 not in rgbs or f1 not in rgbs:
                forced[key] = {"color": None, "answer": "NORGB",
                               "event": event, "f0": f0, "f1": f1, "desc": None}
                continue
            comp = build_composite_v2(rgbs[f0], rgbs[f1], ma, mb, mapping,
                                      f"{catname}: {event}")
            cpath = str(out / "comps" / f"{seq}_w{wi}.jpg")
            cv2.imwrite(cpath, comp, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jobs.append({"key": key, "event": event, "f0": f0, "f1": f1,
                         "category": catname, "ncand": ncand,
                         "comp": cpath, "mapping": mapping})
    return jobs, forced, 0


def parse_reply(reply: str):
    """-> (answer, numbers, desc). answer: '1' | '1,3' | 'NONE' | 'UNPARSED'."""
    m = re.search(r"ANSWER\s*:\s*([\d,\s]+?|NONE)\s*(?:\n|$)", reply, re.I)
    if not m:  # tolerate a bare number / NONE on the first line
        m = re.match(r"\s*([\d,\s]+?|NONE)\s*(?:\n|$)", reply, re.I)
    nums = []
    if m and m.group(1).strip().upper() == "NONE":
        answer = "NONE"
    elif m:
        nums = sorted({int(x) for x in re.findall(r"\d+", m.group(1))})
        answer = ",".join(map(str, nums)) if nums else "UNPARSED"
    else:
        answer = "UNPARSED"
    d = re.search(r"DESC\s*:\s*(.+)", reply, re.I)
    desc = d.group(1).strip().strip('"').rstrip(".") if d else None
    hm = re.search(r"HAND\s*:\s*(left|right|both)", reply, re.I)
    hand = hm.group(1).lower() if hm else None
    return answer, nums, (desc or None), hand


def vlm_worker(jobs, selections, args, lock, prep_done):
    client, calls = None, 0
    tier = "priority" if args.fast else None
    while True:
        try:
            job = jobs.get(timeout=3)
        except queue.Empty:
            if prep_done.is_set():
                if client: client.close()
                return
            continue
        reply = None
        for attempt in (0, 1):
            try:
                if client is None or calls >= 25:
                    if client: client.close()
                    client = CodexClient(model=args.model, effort=args.effort,
                                         cwd=CODEX_CWD, service_tier=tier)
                    calls = 0
                client.new_thread(); calls += 1
                reply = client.describe(
                    PROMPT.format(category=job["category"], event=job["event"]),
                    image=job["comp"]).strip()
                break
            except CodexError:
                client = None
                reply = None
        if reply is None:
            answer, colors, desc, hand = "ERROR", [], None, None
        else:
            answer, nums, desc, hand = parse_reply(reply)
            num_to_col = {v: k for k, v in job["mapping"].items() if v != "H"}
            colors = [list(num_to_col[n]) for n in nums if n in num_to_col]
            if nums and not colors:          # numbers outside the mapping
                answer = "UNPARSED"
        rec = {"color": colors[0] if colors else None, "colors": colors,
               "answer": answer, "desc": desc, "hand": hand,
               "event": job["event"], "f0": job["f0"], "f1": job["f1"],
               "ncand": job["ncand"], "raw": (reply or "")[:200]}
        with lock:
            selections[job["key"]] = rec
            n = len(selections)
            if n % 25 == 0:
                print(f"[{n}] selections so far", flush=True)
                try:
                    Path(args.out).joinpath("selections.json").write_text(
                        json.dumps(selections, indent=0))
                except Exception:
                    pass


def dump_jobs(path, all_jobs):
    path.write_text(json.dumps(
        [{**j, "mapping": {str(k): v for k, v in j["mapping"].items()}}
         for j in all_jobs]))


def merge_shards(out: Path):
    """Merge <out>/shard_*/{selections,jobs}.json into <out>/ (composite
    paths inside jobs stay absolute, so shard dirs must be kept)."""
    sel, jobs = {}, []
    for d in sorted(out.glob("shard_*")):
        if (d / "selections.json").exists():
            sel.update(json.loads((d / "selections.json").read_text()))
        if (d / "jobs.json").exists():
            jobs.extend(json.loads((d / "jobs.json").read_text()))
    (out / "selections.json").write_text(json.dumps(sel, indent=0))
    (out / "jobs.json").write_text(json.dumps(jobs))
    print(f"merged {len(sel)} selections, {len(jobs)} jobs into {out}")


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/vlm_select_v2")
    ap.add_argument("--ext-root", default="/workspace/ext")
    ap.add_argument("--hands-root", default="/workspace/hands2973/hands")
    ap.add_argument("--segments-csv", default=None,
                    help="default <hands-root>/../hoi4d_action_segments.csv")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--fast", action="store_true",
                    help="codex 'Fast' service tier (priority; ~1.5x speed)")
    ap.add_argument("--forced-desc", choices=("vlm", "template"), default="vlm",
                    help="single-candidate windows: ask the VLM anyway for a "
                         "description (vlm) or skip with a template (template)")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="first N seqs only")
    ap.add_argument("--shard", default=None,
                    help="'i/n': prepare seqs[i::n] only (run n processes with "
                         "separate --out dirs, then merge_shards)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="render composites + forced picks, dump jobs.json, "
                         "no VLM (avoids codex CPU contention on small pods)")
    ap.add_argument("--consume-jobs", action="store_true",
                    help="skip prepare; run VLM workers over jobs.json")
    ap.add_argument("--max-jobs", type=int, default=0,
                    help="consume at most N pending jobs (mini-pilot)")
    ap.add_argument("--merge-shards", action="store_true",
                    help="merge <out>/shard_*/ results into <out>/ and exit")
    args = ap.parse_args()
    if args.merge_shards:
        merge_shards(Path(args.out))
        return

    ext, hands = Path(args.ext_root), Path(args.hands_root)
    out = Path(args.out)
    (out / "comps").mkdir(parents=True, exist_ok=True)
    os.makedirs(CODEX_CWD, exist_ok=True)
    sel_path = out / "selections.json"
    selections = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    print("resuming with", len(selections), "existing selections", flush=True)

    jobs = queue.Queue()
    lock = threading.Lock()
    prep_done = threading.Event()
    jobs_path = out / "jobs.json"

    all_jobs = []
    if not args.consume_jobs:
        seg_csv = Path(args.segments_csv) if args.segments_csv \
            else hands.parent / "hoi4d_action_segments.csv"
        segments = load_segments_csv(seg_csv)
        seqs = sorted(p.name for p in hands.iterdir() if p.is_dir())
        if args.limit:
            seqs = seqs[: args.limit]
        if args.shard:
            i, n = (int(x) for x in args.shard.split("/"))
            seqs = seqs[i::n]
        print(f"{len(seqs)} seqs; csv windows for {len(segments)} seqs "
              f"({sum(len(v) for v in segments.values())} windows)", flush=True)
        n_forced = n_skip = 0
        for i, seq in enumerate(seqs):
            sjobs, forced, skipped = prepare_sequence(
                seq, ext, out, selections, segments, args.forced_desc)
            selections.update(forced)
            all_jobs.extend(sjobs)
            n_forced += len(forced); n_skip += skipped
            if (i + 1) % 50 == 0:
                print(f"prepared {i+1}/{len(seqs)} seqs "
                      f"({len(all_jobs)} vlm jobs, {n_forced} forced/none)",
                      flush=True)
                sel_path.write_text(json.dumps(selections, indent=0))
                dump_jobs(jobs_path, all_jobs)
        sel_path.write_text(json.dumps(selections, indent=0))
        dump_jobs(jobs_path, all_jobs)
        from collections import Counter
        print(f"prepare complete: {len(all_jobs)} vlm jobs "
              f"({sum(j['ncand'] == 1 for j in all_jobs)} single-candidate), "
              f"{n_forced} forced/none, {n_skip} seqs skipped;",
              dict(Counter(v['answer'] for v in selections.values())), flush=True)
        if args.prepare_only:
            print("PREPARE-ONLY DONE", flush=True)
            return
    else:
        loaded = json.loads(jobs_path.read_text())
        for j in loaded:
            if j["key"] in selections:
                continue
            j["mapping"] = {eval(k): v for k, v in j["mapping"].items()}
            all_jobs.append(j)
        print(f"consume-jobs: {len(all_jobs)} pending of {len(loaded)}",
              flush=True)
    if args.max_jobs:
        all_jobs = all_jobs[: args.max_jobs]

    threads = [threading.Thread(target=vlm_worker,
                                args=(jobs, selections, args, lock, prep_done))
               for _ in range(args.workers)]
    for t in threads: t.start()
    for j in all_jobs:
        jobs.put(j)
    prep_done.set()
    for t in threads: t.join()
    sel_path.write_text(json.dumps(selections, indent=0))
    from collections import Counter
    dist = Counter(v["answer"] for v in selections.values())
    print("DONE", len(selections), "windows;", dict(dist), flush=True)


if __name__ == "__main__":
    main()
