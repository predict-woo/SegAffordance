"""VLM moving-part selection for EVERY interaction window of the HOI4D
furniture set (the production run behind the pilot,
tools/hoi4d_vlm_part_pilot.py — same composites, same prompt, effort
high, user-approved 2026-09-02).

Output: selections.json mapping "<seq>|<window_idx>" ->
  {"color": [b,g,r] | null, "answer": "<n>|NONE|FORCED|ERROR", "event",
   "f0", "f1"}
Windows with exactly ONE non-hand candidate class skip the VLM (answer
FORCED). NONE/ERROR windows get color null (the rebuild drops them).
One transient retry per window; a failed retry records ERROR.

Run on the HOI4D-volume pod:
  python3 hoi4d_vlm_select_all.py --workers 24 --effort high \
      --out /workspace/vlm_select_all
Resume-safe: existing entries in selections.json are not re-asked.
"""
import argparse
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
from hoi4d_process_2d import (PART_COLORS, load_windows, read_video_frames,
                              seq_to_relpath)
from hoi4d_vlm_part_pilot import (CODEX_CWD, HAND_COLOR, PROMPT,
                                  build_composite)


def prepare_sequence(seq, ext, hands, out, existing):
    """Return (jobs, forced, skipped) for one sequence's windows."""
    rel = seq_to_relpath(seq)
    ann = ext / "HOI4D_annotations" / rel
    mask_dir = ann / "2Dseg" / "mask"
    if not mask_dir.exists():
        mask_dir = ann / "2Dseg" / "shift_mask"
    action = ann / "action" / "color.json"
    if not (mask_dir.exists() and action.exists()):
        return [], {}, 1
    try:
        windows, _ = load_windows(action)
    except Exception:
        return [], {}, 1
    jobs, forced = [], {}
    need_frames, metas = set(), []
    for wi, (event, f0, f1) in enumerate(windows):
        key = f"{seq}|{wi}"
        if key in existing:
            continue
        ma = cv2.imread(str(mask_dir / f"{f0:05d}.png"))
        mb = cv2.imread(str(mask_dir / f"{f1:05d}.png"))
        if ma is None or mb is None:
            forced[key] = {"color": None, "answer": "NOMASK",
                           "event": event, "f0": f0, "f1": f1}
            continue
        mapping, cands = {}, []
        for col in PART_COLORS:
            area = max((ma == col).all(-1).sum(), (mb == col).all(-1).sum())
            if area < 2000:
                continue
            if tuple(col) == HAND_COLOR:
                mapping[tuple(col)] = "H"
            else:
                cands.append(col)
                mapping[tuple(col)] = len(cands)
        if not cands:
            forced[key] = {"color": None, "answer": "NONE",
                           "event": event, "f0": f0, "f1": f1}
        elif len(cands) == 1:
            forced[key] = {"color": list(cands[0]), "answer": "FORCED",
                           "event": event, "f0": f0, "f1": f1}
        else:
            metas.append((key, wi, event, f0, f1, ma, mb, mapping))
            need_frames.update((f0, f1))
    if metas:
        rgbs = read_video_frames(
            ext / "HOI4D_release" / rel / "align_rgb" / "image.mp4",
            need_frames)
        for key, wi, event, f0, f1, ma, mb, mapping in metas:
            if f0 not in rgbs or f1 not in rgbs:
                forced[key] = {"color": None, "answer": "NORGB",
                               "event": event, "f0": f0, "f1": f1}
                continue
            comp = build_composite(rgbs[f0], rgbs[f1], ma, mb, mapping, event)
            cpath = str(out / "comps" / f"{seq}_w{wi}.jpg")
            cv2.imwrite(cpath, comp, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jobs.append({"key": key, "event": event, "f0": f0, "f1": f1,
                         "comp": cpath, "mapping": mapping})
    return jobs, forced, 0


def vlm_worker(jobs, selections, args, lock, prep_done):
    client, calls = None, 0
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
                                         cwd=CODEX_CWD)
                    calls = 0
                client.new_thread(); calls += 1
                reply = client.describe(PROMPT.format(event=job["event"]),
                                        image=job["comp"]).strip()
                break
            except CodexError:
                client = None
                reply = None
        if reply is None:
            answer, color = "ERROR", None
        else:
            m = re.search(r"\b(\d+|NONE)\b", reply, re.I)
            answer = m.group(1).upper() if m else "UNPARSED"
            num_to_col = {v: k for k, v in job["mapping"].items() if v != "H"}
            col = num_to_col.get(int(answer)) if answer.isdigit() else None
            color = list(col) if col is not None else None
        rec = {"color": color, "answer": answer, "event": job["event"],
               "f0": job["f0"], "f1": job["f1"],
               "raw": (reply or "")[:160]}
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


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/vlm_select_all")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--prepare-only", action="store_true",
                    help="render composites + forced picks, dump jobs.json, "
                         "no VLM (avoids codex CPU contention on small pods)")
    ap.add_argument("--consume-jobs", action="store_true",
                    help="skip prepare; run VLM workers over jobs.json")
    args = ap.parse_args()

    ext, hands = Path("/workspace/ext"), Path("/workspace/hands354")
    out = Path(args.out)
    (out / "comps").mkdir(parents=True, exist_ok=True)
    os.makedirs(CODEX_CWD, exist_ok=True)
    sel_path = out / "selections.json"
    selections = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    print("resuming with", len(selections), "existing selections", flush=True)

    seqs = sorted(p.name for p in hands.iterdir() if p.is_dir())
    jobs = queue.Queue()
    lock = threading.Lock()
    prep_done = threading.Event()
    jobs_path = out / "jobs.json"

    all_jobs = []
    if not args.consume_jobs:
        n_forced = n_skip = 0
        for i, seq in enumerate(seqs):
            sjobs, forced, skipped = prepare_sequence(seq, ext, hands, out,
                                                      selections)
            selections.update(forced)
            all_jobs.extend(sjobs)
            n_forced += len(forced); n_skip += skipped
            if (i + 1) % 10 == 0:
                print(f"prepared {i+1}/{len(seqs)} seqs "
                      f"({len(all_jobs)} vlm jobs, {n_forced} forced)",
                      flush=True)
                sel_path.write_text(json.dumps(selections, indent=0))
                jobs_path.write_text(json.dumps(
                    [{**j, "mapping": {str(k): v for k, v in j["mapping"].items()}}
                     for j in all_jobs]))
        sel_path.write_text(json.dumps(selections, indent=0))
        jobs_path.write_text(json.dumps(
            [{**j, "mapping": {str(k): v for k, v in j["mapping"].items()}}
             for j in all_jobs]))
        print(f"prepare complete: {len(all_jobs)} vlm jobs, {n_forced} forced, "
              f"{n_skip} seqs skipped", flush=True)
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
