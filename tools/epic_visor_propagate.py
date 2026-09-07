"""VISOR mask -> contact-onset frame via SAM2 video propagation (test harness).

For each picked interaction: cut the frames between the nearest VISOR sparse
frame and the onset from the full-HD video (EPIC-55 ids on the 60 fps grid,
extension ids at 50 fps), seed SAM2's video predictor with the VISOR fixture
polygon(s) on the sparse frame, propagate to the onset, save the mask and a
panel (sparse frame + GT mask | onset frame + propagated mask + knuckle track).
`--validate`: where the window holds two sparse fixture frames <= 40 frames
apart, propagate first -> second and report IoU against the second's GT.
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, zipfile, collections
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
V = "/workspace/datasets/visor"; PKG = "/workspace/datasets/epic_hands_package"
VID = "/workspace/datasets/epic_videos"; OUT = "/workspace/SegAffordance/viz/20260907_epic_visor_propagate"
CKPT = "/workspace/models/sam2.1_hiera_large.pt"; CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
W, H = 1920, 1080

def grid_fps(vid): return 50.0 if len(vid.split("_")[1]) == 3 else 60.0

def extract(vid, a, b, dst):
    """frames a..b (EPIC annotation numbering, 1-based) -> dst/00000.jpg ..."""
    os.makedirs(dst, exist_ok=True); fps = grid_fps(vid)
    t0 = (a - 1) / fps
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t0:.6f}", "-i", f"{VID}/{vid}.MP4",
                    "-vf", f"fps={fps:g}", "-frames:v", str(b - a + 1), "-q:v", "2", f"{dst}/%05d.jpg"], check=True)
    n = len(os.listdir(dst)); assert n == b - a + 1, (vid, a, b, n)

def poly_mask(segs):
    m = Image.new("L", (W, H), 0); d = ImageDraw.Draw(m)
    for seg in segs:
        for poly in seg:
            if len(poly) >= 3: d.polygon([tuple(p) for p in poly], fill=255)
    return np.array(m) > 0

def sparse_anns(vid, split, image_name):
    d = json.load(open(f"{V}/annotations/{split}/{vid}.json"))["video_annotations"]
    return next(r["annotations"] for r in d if r["image"]["name"] == image_name)

def iou(a, b): return float((a & b).sum()) / max(1, float((a | b).sum()))

def propagate(predictor, frames_dir, seed_mask, reverse):
    """seed at the first (or last, if reverse) frame; returns list of masks per frame index."""
    files = sorted(os.listdir(frames_dir))
    if reverse:  # SAM2 propagates forward from the prompt; reorder so the seed is frame 0
        tmp = frames_dir + "_rev"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
        for i, f in enumerate(reversed(files)): os.symlink(os.path.abspath(f"{frames_dir}/{f}"), f"{tmp}/{i:05d}.jpg")
        frames_dir = tmp
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True)
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=torch.from_numpy(seed_mask))
        masks = {}
        for fi, oids, logits in predictor.propagate_in_video(state):
            masks[fi] = (logits[0, 0] > 0).cpu().numpy()
        predictor.reset_state(state)
    return [masks[i] for i in range(len(files))]  # index 0 = seed frame

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-video", type=int, default=3); ap.add_argument("--max-d", type=int, default=30)
    ap.add_argument("--videos", default="P01_09,P22_07,P04_05,P28_103"); ap.add_argument("--validate", action="store_true"); a = ap.parse_args()
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(CFG, CKPT, device="cuda")
    cov = json.load(open(f"{V}/coverage.json")); narr = {r["narration_id"]: r for r in csv.DictReader(open(f"{PKG}/epic_interaction_index.csv"))}
    intr = {r["video_id"]: r for r in csv.DictReader(open(f"{PKG}/camera_intrinsics.csv"))}
    os.makedirs(OUT, exist_ok=True); work = "/workspace/tmp_propagate"; os.makedirs(work, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    results = []; sheet = []
    for vid in a.videos.split(","):
        if not os.path.exists(f"{VID}/{vid}.MP4"): print("no video", vid); continue
        cands = [o for o in cov if o["video_id"] == vid and o["hits"] and abs(min(o["hits"], key=lambda h: abs(h["d_onset"]))["d_onset"]) <= a.max_d]
        # spread: different nouns, both signs of d
        cands.sort(key=lambda o: (o["noun"], abs(min(o["hits"], key=lambda h: abs(h["d_onset"]))["d_onset"])))
        picks, seen = [], collections.Counter()
        for o in cands:
            if seen[o["noun"]] >= 2: continue
            picks.append(o); seen[o["noun"]] += 1
            if len(picks) >= a.per_video: break
        for o in picks:
            hit = min(o["hits"], key=lambda h: abs(h["d_onset"])); sp, on = hit["epic_frame"], o["onset"]
            split = o["split"]; anns = sparse_anns(vid, split, hit["image"])
            fx = [x for x in anns if x["name"] in hit["fixture"]]
            seed = poly_mask([x["segments"] for x in fx])
            lo, hi = min(sp, on), max(sp, on); fd = f"{work}/{o['narration_id']}"; shutil.rmtree(fd, ignore_errors=True)
            extract(vid, lo, hi, fd)
            masks = propagate(predictor, fd, seed, reverse=(sp > on))
            m_on = masks[-1]; m_seed = masks[0]
            # the trajectory: middle knuckle (joint 9) projected with the video intrinsics
            r = narr[o["narration_id"]]; K = intr[vid]; fx_, fy_, cx, cy = (float(K[k]) for k in ("fx", "fy", "cx", "cy"))
            z = np.load(f"{PKG}/trajectories/{vid}/{o['narration_id']}.npz")
            side = "right" if f"right_joints3d" in z.files and (r["sides"] == "right" or f"left_joints3d" not in z.files) else "left"
            J = z[f"{side}_joints3d"][:, 9, :]; uv = np.stack([fx_ * J[:, 0] / J[:, 2] + cx, fy_ * J[:, 1] / J[:, 2] + cy], 1)
            # panel
            im_sp = Image.open(f"{fd}/{(sp - lo) + 1:05d}.jpg").convert("RGBA"); im_on = Image.open(f"{fd}/{(on - lo) + 1:05d}.jpg").convert("RGBA")
            def overlay(im, m, col):
                ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); arr = np.zeros((H, W, 4), np.uint8); arr[m] = (*col, 110); ov = Image.fromarray(arr)
                return Image.alpha_composite(im, ov)
            L = overlay(im_sp, seed, (255, 40, 40)); Rr = overlay(im_on, m_on, (40, 255, 40))
            d = ImageDraw.Draw(Rr); pts = [tuple(p) for p in uv if np.all(np.isfinite(p))]
            if len(pts) > 1: d.line(pts, fill=(0, 220, 255, 255), width=5)
            if pts: d.ellipse([pts[0][0] - 12, pts[0][1] - 12, pts[0][0] + 12, pts[0][1] + 12], fill=(0, 220, 255, 255)); d.ellipse([pts[-1][0] - 10, pts[-1][1] - 10, pts[-1][0] + 10, pts[-1][1] + 10], fill=(255, 0, 255, 255))
            hdr = Image.new("RGBA", (2 * W, 80), (20, 20, 20, 255)); hd = ImageDraw.Draw(hdr)
            area_ratio = float(m_on.sum()) / max(1, float(seed.sum()))
            hd.text((10, 8), f"{o['narration_id']} '{r['narration']}' side={r['sides']}  LEFT: VISOR frame {sp} (GT {hit['fixture']})   RIGHT: onset {on} (d={sp-on:+d} -> propagated {abs(sp-on)} frames, area x{area_ratio:.2f}, seed-frame IoU {iou(m_seed, seed):.2f}); cyan = knuckle track (dot = onset)", fill=(255, 255, 255, 255), font=font)
            hd.text((10, 44), f"{'EPIC-55 60fps grid' if grid_fps(vid)==60 else 'EK100 ext 50fps'}   noun={o['noun']}   sparse->onset direction: {'forward' if sp < on else 'reverse'}", fill=(200, 200, 200, 255), font=font)
            panel = Image.new("RGBA", (2 * W, H + 80)); panel.paste(hdr, (0, 0)); panel.paste(L, (0, 80)); panel.paste(Rr, (W, 80))
            fn = f"{o['noun'].replace(':', '-')}_{o['narration_id']}_d{sp-on:+d}.jpg"; panel.convert("RGB").save(f"{OUT}/{fn}", quality=85)
            Image.fromarray((m_on * 255).astype(np.uint8)).save(f"{OUT}/{o['narration_id']}_onset_mask.png")
            sheet.append(panel.convert("RGB").resize((2 * W // 4, (H + 80) // 4)))
            results.append({"narration_id": o["narration_id"], "video": vid, "noun": o["noun"], "sparse_frame": sp, "onset": on, "d": sp - on, "area_ratio": area_ratio, "seed_iou": iou(m_seed, seed), "panel": fn})
            print(fn, f"area x{area_ratio:.2f}", flush=True)
            shutil.rmtree(fd, ignore_errors=True); shutil.rmtree(fd + "_rev", ignore_errors=True)
        if a.validate:  # sparse -> sparse round trips inside this video
            d = json.load(open(f"{V}/annotations/{split}/{vid}.json"))["video_annotations"]; fm = json.load(open(f"{V}/frame_mapping.json")).get(vid, {})
            fr = sorted((int(re.search(r"frame_(\d+)", fm.get(r["image"]["name"], r["image"]["name"])).group(1)), r) for r in d)
            nouns = ["drawer", "fridge", "cupboard", "oven", "dishwasher"]; done = 0
            for (f1, r1), (f2, r2) in zip(fr, fr[1:]):
                if not (0 < f2 - f1 <= 40): continue
                for nn in nouns:
                    a1 = [x for x in r1["annotations"] if nn in x["name"]]; a2 = [x for x in r2["annotations"] if nn in x["name"]]
                    if not a1 or not a2: continue
                    fd = f"{work}/val_{vid}_{f1}"; shutil.rmtree(fd, ignore_errors=True); extract(vid, f1, f2, fd)
                    m = propagate(predictor, fd, poly_mask([x["segments"] for x in a1]), reverse=False)[-1]
                    g2 = poly_mask([x["segments"] for x in a2]); v = iou(m, g2); results.append({"validate": True, "video": vid, "noun": nn, "f1": f1, "f2": f2, "gap": f2 - f1, "iou": v})
                    # validation panel: frame f1 + seed (red) | frame f2: GT red, propagated green (overlap -> yellow)
                    i1 = Image.open(f"{fd}/00001.jpg").convert("RGBA"); i2 = Image.open(f"{fd}/{f2 - f1 + 1:05d}.jpg").convert("RGBA")
                    arr = np.zeros((H, W, 4), np.uint8); arr[g2] = (255, 40, 40, 110); arr[m] = (40, 255, 40, 110); arr[g2 & m] = (255, 255, 0, 130)
                    i2 = Image.alpha_composite(i2, Image.fromarray(arr)); arr1 = np.zeros((H, W, 4), np.uint8); arr1[poly_mask([x["segments"] for x in a1])] = (255, 40, 40, 110); i1 = Image.alpha_composite(i1, Image.fromarray(arr1))
                    pan = Image.new("RGBA", (2 * W, H + 80)); hd = ImageDraw.Draw(pan); pan.paste(i1, (0, 80)); pan.paste(i2, (W, 80))
                    hd.text((10, 8), f"VALIDATION {vid} {nn}: seed frame {f1} (red = VISOR GT)  ->  frame {f2} after {f2-f1} frames: red = VISOR GT, green = propagated, yellow = overlap; IoU {v:.3f}", fill=(255, 255, 255, 255), font=font)
                    pan.convert("RGB").save(f"{OUT}/validate_{vid}_{nn}_{f1}_{f2}_iou{v:.2f}.jpg", quality=80)
                    print(f"VALIDATE {vid} {nn} {f1}->{f2} gap {f2-f1}: IoU {v:.3f}", flush=True); shutil.rmtree(fd, ignore_errors=True); done += 1
                    break
                if done >= 6: break
    if sheet:
        cols = 2; rows = (len(sheet) + 1) // 2; w, h = sheet[0].size; cs = Image.new("RGB", (cols * w, rows * h))
        for i, p in enumerate(sheet): cs.paste(p, ((i % cols) * w, (i // cols) * h))
        cs.save(f"{OUT}/contact_sheet.jpg", quality=85)
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
    vals = [x["iou"] for x in results if x.get("validate")]
    if vals: print("VALIDATION IoU: n", len(vals), "mean", np.mean(vals), "min", min(vals))
    print("PROPAGATE_DONE", len([x for x in results if not x.get("validate")]))

if __name__ == "__main__": main()
