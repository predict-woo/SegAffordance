"""Render VISOR fixture masks for a spread of our EPIC interactions.

Picks up to N interactions per fixture noun (nearest VISOR frame to the
contact onset, distinct videos), downloads each needed video's VISOR
rgb_frames zip (sparse frames only), and draws: fixture masks filled red,
hands green, other entities blue outline; header = narration / d_onset /
in-contact flag. Output: viz/20260907_epic_visor_masks/.
"""
import json, os, sys, csv, subprocess, zipfile, collections, re
from PIL import Image, ImageDraw, ImageFont
V = "/workspace/datasets/visor"; PKG = "/workspace/datasets/epic_hands_package"
OUT = "/workspace/SegAffordance/viz/20260907_epic_visor_masks"; os.makedirs(OUT, exist_ok=True)
B = "https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/rgb_frames"
PER_NOUN = int(sys.argv[1]) if len(sys.argv) > 1 else 3

cov = json.load(open(f"{V}/coverage.json"))
narr = {r["narration_id"]: r for r in csv.DictReader(open(f"{PKG}/epic_interaction_index.csv"))}
picks = []; used_videos = collections.Counter()
for noun in ["drawer", "fridge", "cupboard", "oven", "dishwasher", "freezer", "microwave", "cabinet", "door"]:
    cands = [o for o in cov if o["noun"] == noun and o["hits"]]
    cands.sort(key=lambda o: min(abs(h["d_onset"]) for h in o["hits"]))
    n = 0
    for o in cands:
        if used_videos[o["video_id"]] >= 2: continue
        picks.append(o); used_videos[o["video_id"]] += 1; n += 1
        if n >= PER_NOUN: break
print("picks", len(picks), "videos", len(used_videos))

# fetch zips
for vid in used_videos:
    split = next(o["split"] for o in picks if o["video_id"] == vid)
    z = f"{V}/rgb_frames/{split}/{vid}.zip"; os.makedirs(os.path.dirname(z), exist_ok=True)
    if not os.path.exists(z) or not zipfile.is_zipfile(z):
        url = f"{B}/{split}/{vid.split('_')[0]}/{vid}.zip"
        for attempt in range(4):
            rc = subprocess.run(["curl", "-s", "-m", "900", "--retry", "3", "-o", z, url]).returncode
            if rc == 0 and zipfile.is_zipfile(z): break
            if os.path.exists(z): os.remove(z)
        else:
            print("DOWNLOAD_FAILED", vid); picks[:] = [o for o in picks if o["video_id"] != vid]; continue
    print(vid, split, os.path.getsize(z) // 2**20, "MB")

# annotations by (video, image name)
ann_by = {}
for o in picks:
    d = json.load(open(f"{V}/annotations/{o['split']}/{o['video_id']}.json"))["video_annotations"]
    for rec in d: ann_by[(o["video_id"], rec["image"]["name"])] = rec["annotations"]

font = ImageFont.load_default()
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
except Exception: pass
sheet = []
for o in picks:
    hit = min(o["hits"], key=lambda h: abs(h["d_onset"]))
    vid = o["video_id"]; img_name = hit["image"]
    zf = zipfile.ZipFile(f"{V}/rgb_frames/{o['split']}/{vid}.zip")
    member = next(m for m in zf.namelist() if m.endswith(img_name))
    im = Image.open(zf.open(member)).convert("RGB"); W, H = im.size
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    anns = ann_by[(vid, img_name)]
    fixture_ids = set()
    for a in anns:
        nm = a["name"].lower(); is_fix = any(t in nm for t in re.split("[:]", o["noun"]) if t != "machine") or nm in hit["fixture"]
        is_hand = "hand" in nm or "glove" in nm
        for poly in a["segments"]:
            pts = [(x, y) for x, y in poly]
            if len(pts) < 3: continue
            if nm in [f.lower() for f in hit["fixture"]]:
                dr.polygon(pts, fill=(255, 40, 40, 110), outline=(255, 0, 0, 255)); fixture_ids.add(a["id"])
            elif is_hand:
                dr.polygon(pts, fill=(40, 255, 40, 70), outline=(0, 200, 0, 255))
            else:
                dr.polygon(pts, outline=(60, 120, 255, 255))
        if a["segments"] and a["segments"][0]:
            x, y = a["segments"][0][0]
            dr.text((x + 4, y + 4), a["name"] + (f" [touch]" if a.get("in_contact_object") in fixture_ids else ""), fill=(255, 255, 0, 255), font=font)
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    hdr = Image.new("RGB", (W, 90), (20, 20, 20)); hd = ImageDraw.Draw(hdr)
    r = narr[o["narration_id"]]
    hd.text((10, 6), f"{o['narration_id']}  '{r['narration']}'  noun={o['noun']}  side={r['sides']}  VISOR frame {hit['epic_frame']} = onset{hit['d_onset']:+d}  (span {o['span'][0]}-{o['span'][1]})", fill=(255, 255, 255), font=font)
    hd.text((10, 48), f"fixture masks: {hit['fixture']}  hand-in-contact-with-fixture: {hit['hand_in_contact_with_fixture']}  others: {', '.join(hit['others'][:8])}", fill=(200, 200, 200), font=font)
    panel = Image.new("RGB", (W, H + 90)); panel.paste(hdr, (0, 0)); panel.paste(im, (0, 90))
    fn = f"{o['noun'].replace(':','-')}_{o['narration_id']}_f{hit['epic_frame']}.jpg"
    panel.save(f"{OUT}/{fn}", quality=88); sheet.append(panel.resize((W // 3, (H + 90) // 3)))
    print(fn, hit["fixture"], hit["d_onset"], hit["hand_in_contact_with_fixture"])
# contact sheet, 3 per row
cols = 3; rows = (len(sheet) + cols - 1) // cols; w, h = sheet[0].size
cs = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
for i, p in enumerate(sheet): cs.paste(p, ((i % cols) * w, (i // cols) * h))
cs.save(f"{OUT}/contact_sheet.jpg", quality=85)
json.dump([{"narration_id": o["narration_id"], "noun": o["noun"], "video": o["video_id"], "hit": min(o["hits"], key=lambda h: abs(h["d_onset"]))} for o in picks], open(f"{OUT}/picks.json", "w"), indent=1)
print("VIZ_DONE", len(picks))
