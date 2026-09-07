"""Exact-onset fixture masks from VISOR dense interpolations.

Per-video worker (`--video VID`): map each ok interaction's EPIC onset frame
to VISOR frame numbering (offset of the nearest sparse frame in
frame_mapping.json), look up the dense-interpolation record at that frame
(tolerance ±2, trajectory stride is 2), keep the fixture polygons (entity
name matching the noun), write $OUT/VID.json. `--merge` collects the
per-video files into /workspace/datasets/visor/coverage_dense.json and
prints the coverage summary. Run the workers with xargs -P (no
multiprocessing.Pool: it hung on this data).
"""
import json, os, re, sys, collections, zipfile
V = "/workspace/datasets/visor"
Z = os.environ.get("INTERP_DIR", f"{V}/interpolations")
OUT = os.environ.get("DENSE_OUT", "/dev/shm/visor_dense_out")
SYN = {"cupboard": ["cupboard", "cabinet"], "cabinet": ["cabinet", "cupboard"], "fridge": ["fridge", "refrigerator"]}

def match(ann_name, noun):
    t = [x for x in noun.lower().split(":") if x != "machine"][0]
    return any(c in ann_name.lower() for c in SYN.get(t, [t]))

def fnum(name): return int(re.search(r"frame_(\d+)", name).group(1))

def do_video(vid):
    cov = [o for o in json.load(open(f"{V}/coverage.json")) if o["video_id"] == vid]
    split = cov[0]["split"]
    fm = json.load(open(f"{V}/frame_mapping.json")).get(vid, {})
    pairs = sorted((fnum(k), fnum(v)) for k, v in fm.items())  # (visor, epic)
    def to_visor(epic):
        if not pairs: return epic
        vv, ee = min(pairs, key=lambda p: abs(p[1] - epic)); return epic + (vv - ee)
    # Frames we need: onset±2 for every interaction, plus the window frames
    # (counted only, annotations dropped) — the container cgroup is 31 GB and a
    # full json.load of a 1.4 GB file costs ~15 GB, so stream one record at a
    # time with raw_decode and keep only what is needed.
    need = set(); win = []
    for it in cov:
        vo = to_visor(it["onset"]); need.update(vo + d for d in (-2, -1, 0, 1, 2))
        w0, w1 = it["window"]; win.append((to_visor(w0), to_visor(w1)))
    with zipfile.ZipFile(f"{Z}/{split}/{vid}_interpolations.zip") as z:
        txt = z.open(z.namelist()[0]).read().decode("utf-8")
    dec = json.JSONDecoder(); i = txt.index('"video_annotations"'); i = txt.index("[", i) + 1
    by_frame = {}; frames_with_fixture = collections.defaultdict(set); seen_frames = set()  # noun -> frames
    nouns = {it["noun"] for it in cov}
    n_recs = 0
    while True:
        while i < len(txt) and txt[i] in " \t\r\n,": i += 1
        if i >= len(txt) or txt[i] == "]": break
        r, i = dec.raw_decode(txt, i); n_recs += 1
        f = fnum(r["image"]["name"]); seen_frames.add(f)
        if f in need: by_frame[f] = r
        for nn in nouns:
            if any(match(a["name"], nn) for a in r["annotations"]): frames_with_fixture[nn].add(f)
    del txt
    out = []
    for it in cov:
        vo = to_visor(it["onset"]); best = None
        for d in (0, -1, 1, -2, 2):
            r = by_frame.get(vo + d)
            if r is None: continue
            fx = [a for a in r["annotations"] if match(a["name"], it["noun"])]
            if fx:
                best = {"visor_frame": vo + d, "d": d, "interpolation": r["image"].get("interpolation"),
                        "types": [a["type"] for a in fx], "fixture": [a["name"] for a in fx],
                        "segments": [a["segments"] for a in fx],
                        "others": sorted({a["name"] for a in r["annotations"]} - {a["name"] for a in fx})}
                break
        w0, w1 = it["window"]; vw0, vw1 = to_visor(w0), to_visor(w1)
        n_win = sum(1 for f in frames_with_fixture[it["noun"]] if vw0 <= f <= vw1)
        out.append({**{k: it[k] for k in ("narration_id", "noun", "verb", "onset", "window")}, "video_id": vid,
                    "dense_at_onset": best, "n_dense_fixture_frames_in_window": n_win,
                    "dense_frame_at_onset_exists": (vo in seen_frames), "n_dense_records": n_recs})
    os.makedirs(OUT, exist_ok=True)
    tmp = f"{OUT}/{vid}.json.tmp"; json.dump(out, open(tmp, "w")); os.replace(tmp, f"{OUT}/{vid}.json")
    print(vid, len(out), sum(1 for o in out if o["dense_at_onset"]), flush=True)

def merge():
    cov = json.load(open(f"{V}/coverage.json"))
    vids = sorted({o["video_id"] for o in cov})
    res = []; missing = []
    for v in vids:
        p = f"{OUT}/{v}.json"
        if os.path.exists(p): res += json.load(open(p))
        else: missing.append(v)
    json.dump(res, open(f"{V}/coverage_dense.json", "w"))
    hit = [r for r in res if r["dense_at_onset"]]
    print("videos merged:", len(vids) - len(missing), " missing:", missing)
    print("interactions in VISOR videos:", len(res), " dense frame exists at onset:", sum(r["dense_frame_at_onset_exists"] for r in res),
          " FIXTURE MASK AT ONSET (±2):", len(hit), " exact:", sum(r["dense_at_onset"]["d"] == 0 for r in hit))
    print("types at onset:", collections.Counter(t for r in hit for t in r["dense_at_onset"]["types"]))
    print("per noun:", collections.Counter(r["noun"] for r in hit).most_common())
    print("sparse-only hits (coverage.json):", sum(1 for o in cov if o["hits"]),
          " any dense fixture frame in window:", sum(1 for r in res if r["n_dense_fixture_frames_in_window"] > 0))

if __name__ == "__main__":
    if sys.argv[1] == "--merge": merge()
    elif sys.argv[1] == "--list":
        cov = json.load(open(f"{V}/coverage.json"))
        for v in sorted({o["video_id"] for o in cov}):
            if not os.path.exists(f"{OUT}/{v}.json"): print(v)
    else: do_video(sys.argv[2] if sys.argv[1] == "--video" else sys.argv[1])
