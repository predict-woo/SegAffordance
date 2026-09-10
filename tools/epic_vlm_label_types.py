"""VLM rot/trans labels for the EPIC 2D LMDB (tools/epic_process_2d.py output).

The EPIC records carry a placeholder motion_type ("trans" for every record).
This tool asks a VLM (gpt-5.6-luna via the codex app-server, like
tools/hoi4d_vlm_select_all.py) whether the marked part swings about a hinge
(revolute) or slides (prismatic), from ONE composite image per record: the
onset frame with the part mask outlined in red and the hand's 2D path over the
clip drawn in green (cyan = start, magenta = end), full frame on the left and
a zoomed crop on the right, plus the narration.

Phases (resume-safe; existing labels are not re-asked):
  python tools/epic_vlm_label_types.py prepare --lmdb /workspace/datasets/epic_processed_2d --out /workspace/cache/vlm_epic_types
  python tools/epic_vlm_label_types.py run     --out /workspace/cache/vlm_epic_types [--workers 4] [--fast] [--limit N]
  python tools/epic_vlm_label_types.py summary --out /workspace/cache/vlm_epic_types
  python tools/epic_vlm_label_types.py apply   --lmdb ... --out ...   # writes motion_type into the LMDB (backup first)

labels.json maps "<video_id>/<narration_id>" -> {"type": "rot"|"trans"|"ERROR"|"UNPARSED",
  "confidence": "high"|"medium"|"low"|null, "why": str|null, "noun", "verb", "narration", "raw"}.
The noun rule (drawer -> trans, everything else -> rot) is reported alongside as a sanity check.
"""
import argparse
import json
import os
import pickle
import queue
import re
import shutil
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CODEX_CWD = "/root/codex-empty-cwd"

PROMPT = """This image shows the first frame of a short first-person kitchen video clip in \
which a person {verb}s a {noun} (narration: "{narration}"). The RED outline marks the \
moving part the person operates. The GREEN line is the path of the person's hand over \
the clip, from the CYAN dot (start) to the MAGENTA dot (end). Left: the full frame. \
Right: a zoomed crop of the same frame around the part and the hand path.

Task: decide how the marked part moves when it is operated.
REVOLUTE: it swings about a hinge — a door, lid or flap that rotates (a fridge, \
cupboard, oven, dishwasher, microwave or freezer door, a hinged window).
PRISMATIC: it slides along a straight line — a drawer, a pull-out shelf or basket, a \
sliding door or sliding window.
Use the object's construction (visible hinges, where the handle sits, the part's shape \
and how it is mounted) and the hand path as evidence. Do not decide from the noun \
alone: a "cupboard" can have a sliding door, a "drawer" label can be a hinged flap, a \
"door" can slide.

Reply in exactly this format, nothing else:
TYPE: <revolute or prismatic>
CONFIDENCE: <high, medium, or low>
WHY: <one short sentence naming the visual evidence>"""


def noun_rule(noun: str) -> str:
    return "trans" if "drawer" in (noun or "") else "rot"


def parse_reply(reply: str):
    m = re.search(r"TYPE\s*:\s*(revolute|prismatic|rot|trans)", reply, re.I)
    if not m:
        return "UNPARSED", None, None
    t = m.group(1).lower()
    typ = "rot" if t in ("revolute", "rot") else "trans"
    c = re.search(r"CONFIDENCE\s*:\s*(high|medium|low)", reply, re.I)
    w = re.search(r"WHY\s*:\s*(.+)", reply, re.I)
    return typ, (c.group(1).lower() if c else None), (w.group(1).strip() if w else None)


# ------------------------------------------------------------------ composite

def _track_px(rec, frame):
    uv = np.asarray(rec["trajectory_2d_image_coords"], np.float64)
    W, H = frame["orig_size"]
    S = frame_side(frame)
    return uv * np.array([S / W, S / H])


def frame_side(frame):
    im = cv2.imdecode(np.frombuffer(frame["jpeg"], np.uint8), cv2.IMREAD_COLOR)
    return im.shape[0]


def overlay(im, rec, frame):
    S = im.shape[0]
    W, H = frame["orig_size"]
    sx, sy = S / W, S / H
    ov = im.copy()
    mask = np.zeros(im.shape[:2], np.uint8)
    for y, x in rec["mask_coordinates_yx"]:
        mask[min(int(y * sy), S - 1), min(int(x * sx), S - 1)] = 1
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    # outline only (user 2026-09-11: a filled tint hides the part's construction)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 0, 0), 5)
    cv2.drawContours(ov, cnts, -1, (0, 0, 255), 3)
    uv = _track_px(rec, frame)
    pts = uv.round().astype(int)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(ov, tuple(a), tuple(b), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(ov, tuple(a), tuple(b), (60, 230, 60), 3, cv2.LINE_AA)
    cv2.circle(ov, tuple(pts[0]), 7, (230, 230, 40), -1, cv2.LINE_AA)   # cyan start (BGR)
    cv2.circle(ov, tuple(pts[-1]), 7, (230, 40, 230), -1, cv2.LINE_AA)  # magenta end
    return ov, mask, uv


def build_composite(rec, frame, title):
    im = cv2.imdecode(np.frombuffer(frame["jpeg"], np.uint8), cv2.IMREAD_COLOR)
    ov, mask, uv = overlay(im, rec, frame)
    S = im.shape[0]
    ys, xs = np.nonzero(mask)
    xs = np.concatenate([xs, uv[:, 0]]); ys = np.concatenate([ys, uv[:, 1]])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = max(x1 - x0, y1 - y0) * 1.5
    side = float(np.clip(side, 0.45 * S, S))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0 = int(np.clip(cx - side / 2, 0, S - side)); Y0 = int(np.clip(cy - side / 2, 0, S - side))
    crop = ov[Y0:Y0 + int(side), X0:X0 + int(side)]
    crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC)
    left = ov.copy()
    cv2.rectangle(left, (X0, Y0), (X0 + int(side), Y0 + int(side)), (255, 255, 255), 2)
    canvas = np.full((S + 44, 2 * S + 12, 3), 255, np.uint8)
    canvas[44:, :S] = left; canvas[44:, S + 12:] = crop
    cv2.putText(canvas, title, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def load_records(lmdb_dir):
    import lmdb
    env = lmdb.open(f"{lmdb_dir}/data.lmdb", readonly=True, lock=False)
    envf = lmdb.open(f"{lmdb_dir}/frames.lmdb", readonly=True, lock=False)
    with env.begin() as t, envf.begin() as tf:
        for k, v in t.cursor():
            yield k.decode(), pickle.loads(v), pickle.loads(tf.get(k))


def cmd_prepare(a):
    out = Path(a.out); (out / "comp").mkdir(parents=True, exist_ok=True)
    jobs = []
    for key, rec, frame in load_records(a.lmdb):
        e = rec["epic"]
        comp = out / "comp" / (key.replace("/", "__") + ".jpg")
        if not comp.exists():
            title = f'{e["verb"]} {e["noun"]}  |  "{rec["description"]}"  |  {key}'
            cv2.imwrite(str(comp), build_composite(rec, frame, title), [cv2.IMWRITE_JPEG_QUALITY, 90])
        jobs.append({"key": key, "comp": str(comp), "noun": e["noun"], "verb": e["verb"],
                     "narration": rec["description"]})
    (out / "jobs.json").write_text(json.dumps(jobs, indent=0))
    print(f"prepared {len(jobs)} composites -> {out}/comp")


# ------------------------------------------------------------------ VLM

def vlm_worker(jobs, labels, args, lock):
    from codex_client import CodexClient, CodexError
    client, calls = None, 0
    tier = "priority" if args.fast else None
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            if client: client.close()
            return
        reply = None
        for attempt in (0, 1):
            try:
                if client is None or calls >= 25:
                    if client: client.close()
                    client = CodexClient(model=args.model, effort=args.effort, cwd=CODEX_CWD, service_tier=tier)
                    calls = 0
                client.new_thread(); calls += 1
                reply = client.describe(PROMPT.format(verb=job["verb"], noun=job["noun"].replace(":", " "),
                                                      narration=job["narration"]), image=job["comp"]).strip()
                break
            except CodexError:
                client = None; reply = None
        if reply is None:
            typ, conf, why = "ERROR", None, None
        else:
            typ, conf, why = parse_reply(reply)
        rec = {"type": typ, "confidence": conf, "why": why, "noun": job["noun"], "verb": job["verb"],
               "narration": job["narration"], "noun_rule": noun_rule(job["noun"]), "raw": (reply or "")[:300]}
        with lock:
            labels[job["key"]] = rec
            n = len(labels)
            if n % 20 == 0:
                print(f"[{n}] labels so far", flush=True)
                Path(args.out).joinpath("labels.json").write_text(json.dumps(labels, indent=1))


def cmd_run(a):
    os.makedirs(CODEX_CWD, exist_ok=True)
    out = Path(a.out)
    jobs_all = json.loads((out / "jobs.json").read_text())
    lp = out / "labels.json"
    labels = json.loads(lp.read_text()) if lp.exists() else {}
    todo = [j for j in jobs_all if labels.get(j["key"], {}).get("type") in (None, "ERROR", "UNPARSED")]
    if a.limit:
        # round-robin over nouns so a small pilot covers the rare fixtures too
        from collections import defaultdict
        byn = defaultdict(list)
        for j in todo: byn[j["noun"]].append(j)
        nouns = sorted(byn, key=lambda n: -len(byn[n])); pick = []
        while len(pick) < a.limit and any(byn.values()):
            for n in nouns:
                if byn[n] and len(pick) < a.limit: pick.append(byn[n].pop(0))
        todo = pick
    print(f"{len(jobs_all)} jobs, {len(labels)} labelled, {len(todo)} to ask")
    q = queue.Queue()
    for j in todo: q.put(j)
    lock = threading.Lock()
    threads = [threading.Thread(target=vlm_worker, args=(q, labels, a, lock), daemon=True) for _ in range(a.workers)]
    for t in threads: t.start()
    for t in threads: t.join()
    lp.write_text(json.dumps(labels, indent=1))
    print(f"done: {len(labels)} labels -> {lp}")
    cmd_summary(a)


def cmd_summary(a):
    labels = json.loads(Path(a.out).joinpath("labels.json").read_text())
    from collections import Counter, defaultdict
    by = defaultdict(Counter); conf = Counter(); dis = []
    for k, r in labels.items():
        by[r["noun"]][r["type"]] += 1; conf[r["confidence"]] += 1
        if r["type"] in ("rot", "trans") and r["type"] != r["noun_rule"]:
            dis.append((k, r["noun"], r["verb"], r["type"], r["confidence"], r["why"]))
    print(f"{len(labels)} labels; confidence {dict(conf)}")
    for noun in sorted(by, key=lambda n: -sum(by[n].values())):
        print(f"  {noun:16s} {dict(by[noun])}")
    print(f"{len(dis)} disagree with the noun rule:")
    for d in dis: print("   ", d)


def cmd_sheet(a):
    """Review sheet: each labelled record's composite (scaled) with the VLM answer under it."""
    labels = json.loads(Path(a.out).joinpath("labels.json").read_text())
    jobs = {j["key"]: j for j in json.loads(Path(a.out).joinpath("jobs.json").read_text())}
    keys = [k for k in labels if k in jobs]
    if a.limit: keys = keys[:a.limit]
    tiles = []
    for k in keys:
        r = labels[k]; im = cv2.imread(jobs[k]["comp"])
        im = cv2.resize(im, None, fx=a.sheet_scale, fy=a.sheet_scale, interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]
        tile = np.full((h + 62, w, 3), 255, np.uint8); tile[:h] = im
        flag = "" if r["type"] == r["noun_rule"] else "   <-- differs from the noun rule"
        cv2.putText(tile, f'VLM: {r["type"]} ({r["confidence"]}){flag}', (6, h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 200) if flag else (0, 120, 0), 2, cv2.LINE_AA)
        cv2.putText(tile, (r["why"] or "")[:110], (6, h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles: print("no labels"); return
    cols = a.sheet_cols; w = tiles[0].shape[1]; h = max(t.shape[0] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols); sheet[r * h:r * h + t.shape[0], c * w:c * w + w] = t
    outp = Path(a.sheet_out or (Path(a.out) / "review_sheet.jpg")); outp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(f"sheet: {len(tiles)} tiles -> {outp}")


def cmd_apply(a):
    import lmdb
    labels = json.loads(Path(a.out).joinpath("labels.json").read_text())
    src = Path(a.lmdb) / "data.lmdb"
    bak = Path(a.lmdb) / "data.lmdb.bak_pre_vlm_types"
    if not bak.exists():
        shutil.copytree(src, bak); print(f"backup -> {bak}")
    env = lmdb.open(str(src), map_size=1 << 36)
    n = c = 0
    with env.begin(write=True) as t:
        for k, v in list(t.cursor()):
            key = k.decode(); lab = labels.get(key, {}).get("type")
            if lab not in ("rot", "trans"):
                continue
            rec = pickle.loads(v)
            if rec["motion_info"]["original_motion_data"].get("motion_type") != lab: c += 1
            rec["motion_info"]["original_motion_data"]["motion_type"] = lab
            rec["motion_info"]["original_motion_data"]["motion_type_source"] = f"vlm-{a.model}-v1"
            t.put(k, pickle.dumps(rec, protocol=4)); n += 1
    print(f"applied {n} labels ({c} changed from the placeholder)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "summary", "sheet", "apply"])
    ap.add_argument("--lmdb", default="/workspace/datasets/epic_processed_2d")
    ap.add_argument("--out", default="/workspace/cache/vlm_epic_types")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sheet-out", default=None)
    ap.add_argument("--sheet-scale", type=float, default=0.5)
    ap.add_argument("--sheet-cols", type=int, default=2)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summary": cmd_summary, "sheet": cmd_sheet, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
