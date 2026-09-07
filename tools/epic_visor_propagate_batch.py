"""Production step: VISOR mask -> contact-onset frame for every covered
interaction of ONE EPIC video (SAM2 video propagation), written as work items.

Per interaction (coverage.json hit with |d| <= --max-d, nearest sparse frame):
    <out>/<narration_id>/onset.jpg     full-res onset frame (JPEG q95)
    <out>/<narration_id>/mask.png      propagated fixture mask at the onset (0/255)
    <out>/<narration_id>/meta.json     provenance + QA numbers (d, area ratio, seed IoU, ...)
    <out>/<narration_id>/panel.jpg     QA panel (VISOR frame + GT | onset + mask + knuckle track), 1/2 res
Skips items whose meta.json exists (resumable); writes <out>/_video_done/<video>
when the whole video is processed. The video must already be at
/workspace/datasets/epic_videos/<video>.MP4 (tools/epic_fetch_video.py).
Usage: python tools/epic_visor_propagate_batch.py --video P01_09 --out /workspace/datasets/epic_processed_2d/work
"""
import argparse, csv, json, os, shutil, sys, time, traceback
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epic_visor_propagate import (V, PKG, VID, CKPT, CFG, W, H, grid_fps, extract, poly_mask,
                                  sparse_anns, iou, propagate)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-d", type=int, default=60); ap.add_argument("--work", default="/workspace/tmp_propagate")
    a = ap.parse_args(); vid = a.video
    if not os.path.exists(f"{VID}/{vid}.MP4"): print("NO_VIDEO", vid); sys.exit(2)
    cov = [o for o in json.load(open(f"{V}/coverage.json")) if o["video_id"] == vid and o["hits"]]
    items = []
    for o in cov:
        hit = min(o["hits"], key=lambda h: abs(h["d_onset"]))
        if abs(hit["d_onset"]) <= a.max_d: items.append((o, hit))
    print(f"{vid}: {len(items)} interactions (|d|<={a.max_d}) of {len(cov)} with hits", flush=True)
    narr = {r["narration_id"]: r for r in csv.DictReader(open(f"{PKG}/epic_interaction_index.csv"))}
    intr = {r["video_id"]: r for r in csv.DictReader(open(f"{PKG}/camera_intrinsics.csv"))}
    todo = [(o, h) for o, h in items if not os.path.exists(f"{a.out}/{o['narration_id']}/meta.json")]
    if not todo:
        os.makedirs(f"{a.out}/_video_done", exist_ok=True); open(f"{a.out}/_video_done/{vid}", "w").write("ok\n"); print("VIDEO_DONE", vid, "(nothing to do)"); return
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(CFG, CKPT, device="cuda")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    os.makedirs(a.work, exist_ok=True); n_ok = n_fail = 0
    for o, hit in todo:
        nid = o["narration_id"]; t0 = time.time(); od = f"{a.out}/{nid}"; fd = f"{a.work}/{nid}"
        try:
            sp, on = hit["epic_frame"], o["onset"]; split = o["split"]
            anns = sparse_anns(vid, split, hit["image"]); fx = [x for x in anns if x["name"] in hit["fixture"]]
            seed = poly_mask([x["segments"] for x in fx])
            lo, hi = min(sp, on), max(sp, on); shutil.rmtree(fd, ignore_errors=True); extract(vid, lo, hi, fd)
            masks = propagate(predictor, fd, seed, reverse=(sp > on)); m_on, m_seed = masks[-1], masks[0]
            r = narr[nid]; K = intr[vid]; fx_, fy_, cx, cy = (float(K[k]) for k in ("fx", "fy", "cx", "cy"))
            z = np.load(f"{PKG}/trajectories/{vid}/{nid}.npz")
            sides = [s for s in ("left", "right") if f"{s}_joints3d" in z.files]
            side = r["sides"] if r["sides"] in sides else (sides[0] if sides else None)
            J = z[f"{side}_joints3d"][:, 9, :] if side else np.zeros((0, 3))
            uv = np.stack([fx_ * J[:, 0] / J[:, 2] + cx, fy_ * J[:, 1] / J[:, 2] + cy], 1) if len(J) else np.zeros((0, 2))
            os.makedirs(od, exist_ok=True)
            shutil.copy(f"{fd}/{(on - lo) + 1:05d}.jpg", f"{od}/onset.jpg")  # ffmpeg -q:v 2 JPEG, full res
            Image.fromarray((m_on * 255).astype(np.uint8)).save(f"{od}/mask.png")
            area_ratio = float(m_on.sum()) / max(1, float(seed.sum()))
            # QA panel at half resolution
            im_sp = Image.open(f"{fd}/{(sp - lo) + 1:05d}.jpg").convert("RGBA"); im_on = Image.open(f"{od}/onset.jpg").convert("RGBA")
            def overlay(im, m, col):
                arr = np.zeros((H, W, 4), np.uint8); arr[m] = (*col, 110); return Image.alpha_composite(im, Image.fromarray(arr))
            L = overlay(im_sp, seed, (255, 40, 40)); Rr = overlay(im_on, m_on, (40, 255, 40)); d = ImageDraw.Draw(Rr)
            pts = [tuple(p) for p in uv if np.all(np.isfinite(p))]
            if len(pts) > 1: d.line(pts, fill=(0, 220, 255, 255), width=5)
            if pts: d.ellipse([pts[0][0] - 12, pts[0][1] - 12, pts[0][0] + 12, pts[0][1] + 12], fill=(0, 220, 255, 255))
            pan = Image.new("RGBA", (2 * W, H + 70), (20, 20, 20, 255)); pan.paste(L, (0, 70)); pan.paste(Rr, (W, 70))
            ImageDraw.Draw(pan).text((10, 10), f"{nid} '{r['narration']}' side={r['sides']}  VISOR {sp} -> onset {on} (d={sp-on:+d})  area x{area_ratio:.2f}  seedIoU {iou(m_seed, seed):.2f}  fixture={hit['fixture']}", fill=(255, 255, 255, 255), font=font)
            pan.convert("RGB").resize((W, (H + 70) // 2)).save(f"{od}/panel.jpg", quality=80)
            meta = {"video_id": vid, "narration_id": nid, "narration": r["narration"], "verb": o["verb"], "noun": o["noun"],
                    "sides": r["sides"], "side_used": side, "scale_regime": r["scale_regime"], "onset": on, "sparse_frame": sp,
                    "d": sp - on, "span": o["span"], "window": o["window"], "visor_image": hit["image"], "visor_split": split,
                    "fixture_names": hit["fixture"], "hand_in_contact_with_fixture": hit["hand_in_contact_with_fixture"],
                    "seed_area_px": int(seed.sum()), "onset_area_px": int(m_on.sum()), "area_ratio": area_ratio,
                    "seed_iou": iou(m_seed, seed), "grid_fps": grid_fps(vid),
                    "intrinsics": {k: float(K[k]) for k in ("fx", "fy", "cx", "cy")}, "width": W, "height": H,
                    "knuckle_uv_onset": [float(pts[0][0]), float(pts[0][1])] if pts else None, "n_traj": int(len(J)),
                    "seconds": round(time.time() - t0, 1)}
            json.dump(meta, open(f"{od}/meta.json", "w"), indent=1)
            n_ok += 1; print(f"OK {nid} d={sp-on:+d} area x{area_ratio:.2f} {meta['seconds']}s", flush=True)
        except Exception as e:
            n_fail += 1; print(f"FAIL {nid}: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
            os.makedirs(f"{a.out}/_failed", exist_ok=True); open(f"{a.out}/_failed/{nid}", "w").write(traceback.format_exc())
        finally:
            shutil.rmtree(fd, ignore_errors=True); shutil.rmtree(fd + "_rev", ignore_errors=True)
    os.makedirs(f"{a.out}/_video_done", exist_ok=True); open(f"{a.out}/_video_done/{vid}", "w").write(f"ok={n_ok} fail={n_fail}\n")
    print("VIDEO_DONE", vid, "ok", n_ok, "fail", n_fail, flush=True)

if __name__ == "__main__": main()
