"""VLM pilot: pick the manipulated furniture part via Set-of-Mark prompting.

For N interaction windows, renders the 2x2 composite (raw first/last
frames + numbered segment overlays; hand labeled 'H'; same class = same
number in both frames), asks Codex (gpt-5.6-luna, the OPD-descriptions
pipeline's model) which numbered segment is the part being manipulated,
and writes a verification panel per window: the composite with the VLM's
chosen segment re-highlighted + the answer, alongside what the old
motion-energy pick and the wrist-exclusion pick would have chosen.

Run on the HOI4D-volume pod (raw 2Dseg + videos live only there):
  python3 hoi4d_vlm_part_pilot.py --num 8 --out /workspace/vlm_pilot
Requires: codex on PATH, /root/.codex/auth.json, hoi4d_process_2d.py and
codex_client.py importable (scp'd next to this script).
"""
import argparse
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
from hoi4d_process_2d import (PART_COLORS, hand_colors, load_windows,
                              moving_color, read_video_frames, seq_to_relpath)

HAND_COLOR = (0, 128, 0)
CODEX_CWD = "/root/codex-empty-cwd"

PROMPT = """These four images show a person interacting with a piece of \
furniture ({event} action). Top row: the first and last video frames of the \
interaction. Bottom row: the same two frames with segmentation overlays — \
each candidate part carries a NUMBER, and the person's hand/arm is labeled H.

Which numbered segment is the furniture part being manipulated (the part \
that physically moves — e.g. the drawer being pulled, the door being \
swung)? Compare the two frames to see what moved.

Reply with ONLY the number. If no numbered segment is the manipulated \
part, reply NONE."""


def interior_point(mask):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(x), int(y)


def overlay_numbered(img, m, mapping):
    ov = img.copy()
    tint = {1: (60, 60, 230), 2: (230, 160, 60), 3: (60, 200, 230),
            4: (200, 60, 200), "H": (60, 230, 60)}
    for col, label in mapping.items():
        mask = (m == np.array(col)).all(-1)
        if not mask.any():
            continue
        ov[mask] = ov[mask] * 0.45 + np.array(tint.get(label, (180,) * 3)) * 0.55
        x, y = interior_point(mask)
        for th, c in ((10, (0, 0, 0)), (4, (255, 255, 255))):
            cv2.putText(ov, str(label), (x - 25, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.6, c, th)
    return ov


def build_composite(rgb_a, rgb_b, ma, mb, mapping, event):
    a, b = rgb_a.copy(), rgb_b.copy()
    oa = overlay_numbered(rgb_a, ma, mapping)
    ob = overlay_numbered(rgb_b, mb, mapping)
    for im, lab in ((a, f"FIRST frame ({event})"), (b, "LAST frame"),
                    (oa, "FIRST + segments"), (ob, "LAST + segments")):
        cv2.putText(im, lab, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,) * 3, 8)
        cv2.putText(im, lab, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,) * 3, 4)
    grid = np.concatenate([np.concatenate([a, b], 1),
                           np.concatenate([oa, ob], 1)], 0)
    return cv2.resize(grid, (1920, 1080), interpolation=cv2.INTER_AREA)


def prepare_job(seq, ext, hands, out, done_idx):
    rel = seq_to_relpath(seq)
    ann = ext / "HOI4D_annotations" / rel
    mask_dir = ann / "2Dseg" / "mask"
    if not mask_dir.exists():
        mask_dir = ann / "2Dseg" / "shift_mask"
    action = ann / "action" / "color.json"
    hz = hands / seq / "wilor" / "hands.npz"
    if not (mask_dir.exists() and action.exists() and hz.exists()):
        return None
    try:
        windows, _ = load_windows(action)
    except Exception:
        return None
    if not windows:
        return None
    event, f0, f1 = windows[0]
    ma = cv2.imread(str(mask_dir / f"{f0:05d}.png"))
    mb = cv2.imread(str(mask_dir / f"{f1:05d}.png"))
    if ma is None or mb is None:
        return None
    mapping, n = {}, 0
    for col in PART_COLORS:
        area = max((ma == col).all(-1).sum(), (mb == col).all(-1).sum())
        if area < 2000:
            continue
        if tuple(col) == HAND_COLOR:
            mapping[tuple(col)] = "H"
        else:
            n += 1
            mapping[tuple(col)] = n
    if n < 2:
        return None
    rgbs = read_video_frames(
        ext / "HOI4D_release" / rel / "align_rgb" / "image.mp4", {f0, f1})
    if f0 not in rgbs or f1 not in rgbs:
        return None
    comp = build_composite(rgbs[f0], rgbs[f1], ma, mb, mapping, event)
    comp_path = str(out / f"comp_{done_idx}_{seq}.jpg")
    cv2.imwrite(comp_path, comp, [cv2.IMWRITE_JPEG_QUALITY, 90])
    rgb1_path = str(out / f"_rgb1_{done_idx}.jpg")
    cv2.imwrite(rgb1_path, rgbs[f1], [cv2.IMWRITE_JPEG_QUALITY, 92])

    h = np.load(hz); right = h["is_right"] == 1
    fr0 = h["frame_number"][right].astype(int) - 1
    j2d = h["joints_2d"][right][:, 0, :]
    f2i = {int(f): i for i, f in enumerate(fr0)}
    det = [f for f in range(f0, f1 + 1) if f in f2i]
    probe = det[:: max(1, len(det) // 5)][:5] if det else []
    wrists = [(f, tuple(j2d[f2i[f]])) for f in probe]
    old_pick = moving_color(mask_dir, f0, f1)
    wx_pick = moving_color(mask_dir, f0, f1,
                           exclude=hand_colors(mask_dir, wrists))
    return {"idx": done_idx, "seq": seq, "event": event, "f0": f0, "f1": f1,
            "mapping": mapping, "comp": comp_path, "rgb1": rgb1_path,
            "mask_last": str(mask_dir / f"{f1:05d}.png"),
            "old": old_pick, "wx": wx_pick}


def vlm_worker(jobs, results, args, out, lock, prep_done):
    client, calls = None, 0
    while True:
        try:
            job = jobs.get(timeout=3)
        except queue.Empty:
            if prep_done.is_set():
                if client: client.close()
                return
            continue
        try:
            if client is None or calls >= 25:
                if client: client.close()
                client = CodexClient(model=args.model, effort=args.effort,
                                     cwd=CODEX_CWD)
                calls = 0
            client.new_thread(); calls += 1
            reply = client.describe(PROMPT.format(event=job["event"]),
                                    image=job["comp"]).strip()
        except CodexError as e:
            reply = f"ERROR: {e}"
            client = None
        mnum = re.search(r"\b(\d+|NONE)\b", reply, re.I)
        answer = mnum.group(1).upper() if mnum else "UNPARSED"
        mapping = job["mapping"]
        num_to_col = {v: k for k, v in mapping.items() if v != "H"}
        vlm_col = num_to_col.get(int(answer)) if answer.isdigit() else None

        comp = cv2.imread(job["comp"])
        if vlm_col is not None:
            mb = cv2.imread(job["mask_last"])
            rgb1 = cv2.imread(job["rgb1"])
            mv = (mb == np.array(vlm_col)).all(-1)
            hl = rgb1.copy()
            hl[mv] = hl[mv] * 0.35 + np.array([0, 220, 255]) * 0.65
            hl = cv2.resize(hl, (960, 540), interpolation=cv2.INTER_AREA)
            comp[540:, 960:] = hl
            for th, c in ((8, (0, 0, 0)), (4, (0, 220, 255))):
                cv2.putText(comp, "VLM PICK", (974, 592),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, c, th)
        def lab(col):
            return mapping.get(tuple(col), "?") if col is not None else "-"
        footer = (f"VLM: {answer}   |   old: {lab(job['old'])}"
                  f"   |   wrist-excl: {lab(job['wx'])}")
        cv2.rectangle(comp, (0, 1040), (1920, 1080), (0, 0, 0), -1)
        cv2.putText(comp, footer, (14, 1070), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"verify_{job['idx']}_{job['seq']}.jpg"), comp,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        rec = {"seq": job["seq"], "window": [job["event"], job["f0"], job["f1"]],
               "vlm_raw": reply[:200], "vlm": answer,
               "old": lab(job["old"]), "wrist_excl": lab(job["wx"])}
        with lock:
            results.append(rec)
            print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--out", default="/workspace/vlm_pilot")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    ext, hands = Path("/workspace/ext"), Path("/workspace/hands354")
    out = Path(args.out); out.mkdir(exist_ok=True)
    os.makedirs(CODEX_CWD, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    seqs = sorted(p.name for p in hands.iterdir() if p.is_dir())
    rng.shuffle(seqs)

    jobs = queue.Queue()
    results, lock = [], threading.Lock()
    prep_done = threading.Event()
    threads = [threading.Thread(target=vlm_worker,
                                args=(jobs, results, args, out, lock, prep_done))
               for _ in range(args.workers)]
    for t in threads: t.start()   # workers consume while we keep preparing

    prepared = 0
    for seq in seqs:
        if prepared >= args.num:
            break
        job = prepare_job(seq, ext, hands, out, prepared)
        if job is None:
            continue
        jobs.put(job); prepared += 1
        if prepared % 10 == 0:
            print(f"prepared {prepared}/{args.num}", flush=True)
    prep_done.set()
    print(f"prepared {prepared} jobs; {args.workers} VLM workers draining",
          flush=True)
    for t in threads: t.join()
    with open(out / "pilot_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("done", len(results))


if __name__ == "__main__":
    main()
