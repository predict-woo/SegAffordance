"""Intersect VISOR sparse annotations with the EPIC hands package interactions.

For every interaction with an ok trajectory: which VISOR frames (mapped to
EPIC frame numbers via frame_mapping.json) fall in its span / window, how far
the nearest one is from the contact onset, and whether that frame carries a
mask whose entity name matches the interaction's fixture noun.
Writes /workspace/datasets/visor/coverage.json (per-interaction hits).
"""
import csv, json, os, re, collections, glob
V = "/workspace/datasets/visor"
PKG = "/workspace/datasets/epic_hands_package"

rows = [r for r in csv.DictReader(open(f"{PKG}/epic_interaction_index.csv")) if r["trajectory_status"] == "ok"]
fm = json.load(open(f"{V}/frame_mapping.json"))
# VISOR frames per video: {video: [(epic_frame, image_name, annotations)]}
visor = {}
split_of = {}
for split in ("train", "val"):
    for p in glob.glob(f"{V}/annotations/{split}/*.json"):
        vid = os.path.basename(p)[:-5]
        split_of[vid] = split
        d = json.load(open(p))["video_annotations"]
        lst = []
        for rec in d:
            name = rec["image"]["name"]
            mapped = fm.get(vid, {}).get(name)
            fr = int(re.search(r"frame_(\d+)", mapped or name).group(1))
            lst.append((fr, name, rec["annotations"]))
        visor[vid] = sorted(lst)

def noun_tokens(noun):
    return [t for t in noun.split(":") if t not in ("machine",)]  # "door:washing:machine" -> door, washing

def matches(ann_name, noun):
    ann = ann_name.lower(); toks = noun_tokens(noun.lower())
    # primary token first ("drawer" in "drawer:fridge"); accept synonyms
    syn = {"cupboard": ["cupboard", "cabinet"], "cabinet": ["cabinet", "cupboard"],
           "fridge": ["fridge", "refrigerator"], "freezer": ["freezer"],
           "bin": ["bin"], "door": ["door"]}
    cands = syn.get(toks[0], [toks[0]])
    return any(c in ann for c in cands)

out = []
stats = collections.Counter()
near = collections.Counter()
names_at_hits = collections.Counter()
for r in rows:
    vid = r["video_id"]; noun = r["noun"]; onset = int(r["contact_onset_frame"])
    s0, s1 = int(r["span_start"]), int(r["span_end"]); w0, w1 = int(r["window_start"]), int(r["window_end"])
    stats["ok"] += 1
    if vid not in visor:
        stats["video_not_in_visor"] += 1; continue
    stats["video_in_visor"] += 1
    frames = visor[vid]
    in_win = [f for f in frames if w0 <= f[0] <= w1]
    in_span = [f for f in frames if s0 <= f[0] <= s1]
    if in_win: stats["has_visor_frame_in_window"] += 1
    if in_span: stats["has_visor_frame_in_span"] += 1
    # fixture masks on window frames
    hits = []
    for fr, name, anns in in_win:
        fx = [a for a in anns if matches(a["name"], noun)]
        if fx:
            hands = [a for a in anns if "hand" in a["name"] or "glove" in a["name"]]
            contact = any(a.get("in_contact_object") in {x["id"] for x in fx} for a in hands)
            hits.append({"epic_frame": fr, "image": name, "d_onset": fr - onset, "fixture": [x["name"] for x in fx],
                         "n_fixture_masks": len(fx), "hand_in_contact_with_fixture": contact,
                         "others": sorted({a["name"] for a in anns} - {x["name"] for x in fx})})
    if hits:
        stats["fixture_mask_in_window"] += 1
        best = min(hits, key=lambda h: abs(h["d_onset"]))
        for k in (0, 5, 15, 30, 60, 90):
            if abs(best["d_onset"]) <= k: near[f"|d|<={k}"] += 1
        if any(h["hand_in_contact_with_fixture"] for h in hits): stats["fixture_mask_with_hand_contact"] += 1
        if any(s0 <= h["epic_frame"] <= s1 for h in hits): stats["fixture_mask_in_span"] += 1
        for h in hits:
            for n in h["fixture"]: names_at_hits[n] += 1
    elif in_win:
        stats["window_frame_but_no_fixture_mask"] += 1
        for fr, name, anns in in_win:
            for a in anns: names_at_hits["(non-fixture) " + a["name"]] += 1
    out.append({"video_id": vid, "narration_id": r["narration_id"], "noun": noun, "verb": r["verb"], "onset": onset,
                "span": [s0, s1], "window": [w0, w1], "split": split_of[vid], "n_visor_frames_in_window": len(in_win),
                "n_visor_frames_in_span": len(in_span), "hits": hits})

json.dump(out, open(f"{V}/coverage.json", "w"), indent=1)
print("VISOR videos with sparse json:", len(visor), " our ok videos:", len({r['video_id'] for r in rows}),
      " overlap:", len({r['video_id'] for r in rows} & set(visor)))
for k, v in stats.items(): print(f"{k:40s} {v}")
print("nearest fixture-mask frame to onset:", dict(near))
print("fixture names at hits:", names_at_hits.most_common(30))
per_noun = collections.Counter(o["noun"] for o in out if o["hits"])
print("hits per noun:", per_noun.most_common())
