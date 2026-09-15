"""Stage the paper's Fig. 5 hand-video frames (HOI4D / EPIC / ARCTIC, 50 each) for the external
baselines, and map their outputs back to the 512x512 stretched frame the figure uses.

Inputs (paper session, 2026-09-15): /workspace/datasets/handvideo_fig5_samples/<src>/ with
NN_<src>_frame_512.png (the 512x512 stretched model input), NN_<src>_gt_mask_512.png,
NN_<src>_depth_512.npy (metres; HOI4D + ARCTIC only) and samples.jsonl (key, desc, gt_type,
native_wh, K_native, point_uv, ...). The baselines were trained on 4:3 SF3D frames, so every model
sees the frame at its NATIVE aspect (the stretched PNG resized back to native_wh's aspect),
letterboxed into the model's canvas with the dataset's mean colour, and with K scaled/shifted
accordingly -- camera-frame geometry is therefore in the native camera. Masks come back to the
512x512 stretched frame by cropping the content region and resizing (nearest).

  python handvideo_stage.py opd   --out <dir>            # OPDFormer / MOPD: MotionDataset_h5 + json
  python handvideo_stage.py threedoi --out <dir>         # 3DOI: images/omnidata/data_test.pt
  python handvideo_stage.py a3vlm --out <dir> --image-root <abs dir as seen by the A3VLM pod>
Every subcommand also writes <out>/hv_meta.json: per sample the letterbox geometry the exporter
(handvideo_export.py) needs.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baselines_sf3d import common as C  # noqa: E402

SAMPLES = Path("/workspace/datasets/handvideo_fig5_samples")
SOURCES = ("hoi4d", "epic", "arctic")
STRETCH = 512
DEPTH_NOMINAL = (0.3, 3.0)  # metres, A3VLM depth normalisation when a source has no depth (EPIC)


# ---------------------------------------------------------------- samples
def load_samples(root=SAMPLES, sources=SOURCES):
    out = []
    for src in sources:
        d = Path(root) / src
        for line in open(d / "samples.jsonl"):
            r = json.loads(line)
            s = r["sample"]
            r["frame_png"] = str(d / f"{s}_frame_512.png")
            r["mask_png"] = str(d / f"{s}_gt_mask_512.png")
            dp = d / f"{s}_depth_512.npy"
            r["depth_npy"] = str(dp) if dp.exists() else None
            out.append(r)
    return out


def read_frame(r):
    """(rgb uint8 (H,W,3) at the native aspect, gt bool (512,512), depth float32 metres at the native
    aspect or None). Native aspect = native_wh scaled to width 512 (the stretched PNG holds 512 rows
    for the native height, so this loses nothing horizontally)."""
    bgr = cv2.imread(r["frame_png"], cv2.IMREAD_COLOR)
    assert bgr is not None and bgr.shape[:2] == (STRETCH, STRETCH), r["frame_png"]
    W, H = r["native_wh"]
    cw, ch = STRETCH, int(round(STRETCH * H / W))
    rgb = cv2.cvtColor(cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(r["mask_png"], cv2.IMREAD_GRAYSCALE) > 127
    depth = None
    if r["depth_npy"]:
        depth = cv2.resize(np.load(r["depth_npy"]).astype(np.float32), (cw, ch), interpolation=cv2.INTER_NEAREST)
    return rgb, gt, depth


def native_K(r, cw, ch):
    """K_native scaled to the native-aspect image (cw, ch)."""
    W, H = r["native_wh"]
    K = np.asarray(r["K_native"], dtype=np.float64)
    S = np.diag([cw / W, ch / H, 1.0])
    return S @ K


def letterbox(img, out_w, out_h, fill, nearest=False):
    """Scale to fit, centre on a (out_h, out_w) canvas of colour/value `fill` (nearest for masks /
    depth, area for RGB). Returns (canvas, {"s", "ox", "oy", "cw", "ch", "out_w", "out_h"})."""
    h, w = img.shape[:2]
    s = min(out_w / w, out_h / h)
    cw, ch = int(round(w * s)), int(round(h * s))
    interp = cv2.INTER_NEAREST if (nearest or img.dtype != np.uint8) else cv2.INTER_AREA
    content = cv2.resize(img, (cw, ch), interpolation=interp)
    canvas = np.empty((out_h, out_w) + img.shape[2:], img.dtype)
    canvas[...] = fill
    ox, oy = (out_w - cw) // 2, (out_h - ch) // 2
    canvas[oy:oy + ch, ox:ox + cw] = content
    return canvas, {"s": s, "ox": ox, "oy": oy, "cw": cw, "ch": ch, "out_w": out_w, "out_h": out_h}


def letterbox_K(K, lb):
    K = np.asarray(K, dtype=np.float64)
    return np.array([[K[0, 0] * lb["s"], 0.0, K[0, 2] * lb["s"] + lb["ox"]],
                     [0.0, K[1, 1] * lb["s"], K[1, 2] * lb["s"] + lb["oy"]],
                     [0.0, 0.0, 1.0]])


def unletterbox_mask(mask, lb):
    """Canvas-space mask -> 512x512 stretched-frame mask (bool)."""
    m = np.asarray(mask).astype(np.uint8)
    if m.shape != (lb["out_h"], lb["out_w"]):
        m = C.nearest_resize(m, (lb["out_h"], lb["out_w"]))
    content = m[lb["oy"]:lb["oy"] + lb["ch"], lb["ox"]:lb["ox"] + lb["cw"]]
    return C.nearest_resize(content, (STRETCH, STRETCH)).astype(bool)


def unletterbox_point(xy_norm, lb):
    """Normalised canvas point -> normalised stretched-frame point."""
    x = (xy_norm[0] * lb["out_w"] - lb["ox"]) / lb["cw"]
    y = (xy_norm[1] * lb["out_h"] - lb["oy"]) / lb["ch"]
    return [float(x), float(y)]


def letterbox_point(xy_norm_stretched, lb):
    """Normalised stretched-frame point -> normalised canvas point."""
    x = (lb["ox"] + xy_norm_stretched[0] * lb["cw"]) / lb["out_w"]
    y = (lb["oy"] + xy_norm_stretched[1] * lb["ch"]) / lb["out_h"]
    return [float(x), float(y)]


def stretched_to_native_mask(gt512, cw, ch):
    return C.nearest_resize(gt512.astype(np.uint8), (ch, cw)).astype(bool)


# ---------------------------------------------------------------- OPD / MOPD
def stage_opd(out, stats_src, samples):
    """MotionDataset_h5/{test.h5, depth.h5, annotations/MotionNet_test.json} + obj_info.json +
    stats.json (copied from the TRAINING data: the chain reads PIXEL_MEAN/STD from it) + hv_meta.json.
    GT motion is a placeholder (their evaluator needs the fields; only the predictions are used)."""
    import h5py
    from tools.baselines_sf3d.sf3d_to_opd import CATS, CAT_IDS, colmajor16, colmajor9, RANGE_MAX_ROT, RANGE_MAX_TRANS

    OUT_W, OUT_H = 512, 384
    out = Path(out)
    h5dir = out / "MotionDataset_h5"
    (h5dir / "annotations").mkdir(parents=True, exist_ok=True)
    stats = json.load(open(stats_src))
    fill_rgb = np.array(stats["pixel_mean"][:3], dtype=np.float64).round().astype(np.uint8)
    n = len(samples)
    images, anns, obj_info, meta = [], [], {}, {}
    with h5py.File(h5dir / "test.h5", "w") as hf, h5py.File(h5dir / "depth.h5", "w") as hd:
        di = hf.create_dataset("test_images", (n, OUT_H, OUT_W, 3), np.uint8, chunks=(1, OUT_H, OUT_W, 3))
        dn = hf.create_dataset("test_filenames", (n,), h5py.string_dtype("utf-8"))
        dd = hd.create_dataset("depth_images", (n, OUT_H, OUT_W, 1), np.float32, chunks=(1, OUT_H, OUT_W, 1))
        ddn = hd.create_dataset("depth_filenames", (n,), h5py.string_dtype("utf-8"))
        for i, r in enumerate(samples):
            rgb, gt, depth = read_frame(r)
            ch, cw = rgb.shape[:2]
            canvas, lb = letterbox(rgb, OUT_W, OUT_H, fill_rgb)
            if depth is None:
                depth = np.zeros((ch, cw), np.float32)
            dcan, _ = letterbox(depth, OUT_W, OUT_H, 0.0)
            K_out = letterbox_K(native_K(r, cw, ch), lb)
            fn, dfn = f"{r['sample']}.png", f"{r['sample']}_d.png"
            di[i] = canvas
            dn[i] = fn
            dd[i] = (dcan * 1000.0)[..., None]  # mm like sf3d_to_opd
            ddn[i] = dfn
            image_id, ann_id = i + 1, i + 1
            c2w_opd = np.eye(4) @ C.F_YZ
            images.append({"id": image_id, "file_name": fn, "depth_file_name": dfn, "height": OUT_H, "width": OUT_W,
                           "license": 1, "coco_url": "", "flickr_url": "", "date_captured": "",
                           "camera": {"intrinsic": colmajor9(K_out), "extrinsic": colmajor16(c2w_opd)}})
            gt_can = letterbox(stretched_to_native_mask(gt, cw, ch).astype(np.uint8), OUT_W, OUT_H, 0, nearest=True)[0].astype(bool)
            ys, xs = np.where(gt_can)
            bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)] if xs.size else [0.0, 0.0, 0.0, 0.0]
            rot = int(r["gt_type"]) == 1
            ok = f"hv-{r['sample']}"
            anns.append({"id": ann_id, "image_id": image_id, "category_id": CAT_IDS["pinch_pull"], "bbox": bbox,
                         "area": int(gt_can.sum()), "iscrowd": 0, "height": OUT_H, "width": OUT_W,
                         "segmentation": C.rle_encode(gt_can), "object_key": ok,
                         "motion": {"type": "rotation" if rot else "translation", "axis": [0.0, 0.0, 1.0],
                                    "origin": [0.0, 0.0, 0.0], "partId": ann_id, "part_label": "pinch_pull",
                                    "isClosed": True, "rangeMin": 0, "rangeMax": RANGE_MAX_ROT if rot else RANGE_MAX_TRANS,
                                    "state": 0, "pixel_num": float(gt_can.sum()), "bbox": bbox, "object_key": ok}})
            obj_info[ok] = {"object_pose": colmajor16(c2w_opd), "diagonal": 1.0, "min_bound": [-0.5] * 3, "max_bound": [0.5] * 3}
            meta[r["sample"]] = {"sample": r["sample"], "dataset": r["dataset"], "key": r["key"], "image_id": image_id,
                                 "lb": lb, "K_out": K_out.tolist(), "has_depth": r["depth_npy"] is not None,
                                 "mask_png": r["mask_png"], "gt_type": int(r["gt_type"])}
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": CAT_IDS[c], "name": c, "supercategory": "sf3d"} for c in CATS]}
    json.dump(coco, open(h5dir / "annotations" / "MotionNet_test.json", "w"))
    # their dataset registration opens DATASETS.TRAIN/VALID json files too (opd_base.yaml)
    for split in ("train", "valid"):
        json.dump(coco, open(h5dir / "annotations" / f"MotionNet_{split}.json", "w"))
        for suffix in ("", "depth_"):
            link = h5dir / f"{suffix}{split}.h5"
            if not link.exists():
                link.symlink_to(h5dir / ("depth.h5" if suffix else "test.h5"))
    json.dump(obj_info, open(out / "obj_info.json", "w"))
    json.dump(stats, open(out / "stats.json", "w"))
    json.dump(meta, open(out / "hv_meta.json", "w"))
    print(f"opd: {n} frames -> {out}")


# ---------------------------------------------------------------- 3DOI
def stage_threedoi(out, samples):
    """images/ + omnidata_filtered/{depth_zbuffer,mask_valid,point_info}/taskonomy/<visit>/ +
    3doi_sf3d/data_test.pt + frames_test.json (the layout sf3d_to_3doi.py writes, one instance per
    frame prompted with the GT interaction point). GT bbox/mask/axis fields are placeholders for
    their loader; kinematic = gt_type. No-depth sources get an all-zero depth (= all holes)."""
    import torch
    from tools.baselines_sf3d.sf3d_to_3doi import (DEPTH_SCALE, KINEMATIC, bbox_norm, depth_name, hfov_rads, img_name,
                                                   info_name, mask_polygon)

    OUT_W, OUT_H = 1024, 768
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "3doi_sf3d").mkdir(exist_ok=True)
    entries, infos, meta = [], {}, {}
    for i, r in enumerate(samples):
        rgb, gt, depth = read_frame(r)
        ch, cw = rgb.shape[:2]
        canvas, lb = letterbox(rgb, OUT_W, OUT_H, np.array([124, 116, 104], np.uint8))
        K_out = letterbox_K(native_K(r, cw, ch), lb)
        visit, f = f"hv{r['dataset']}", i  # no underscore: their loader splits the name on '_'
        name = img_name(visit, f)
        cv2.imwrite(str(out / "images" / name), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        d_dir = out / "omnidata_filtered" / "depth_zbuffer" / "taskonomy" / visit
        m_dir = out / "omnidata_filtered" / "mask_valid" / "taskonomy" / visit
        i_dir = out / "omnidata_filtered" / "point_info" / "taskonomy" / visit
        for d in (d_dir, m_dir, i_dir):
            d.mkdir(parents=True, exist_ok=True)
        dcan = letterbox(depth, OUT_W, OUT_H, 0.0)[0] if depth is not None else np.zeros((OUT_H, OUT_W), np.float32)
        cv2.imwrite(str(d_dir / depth_name(f)), np.clip(dcan.astype(np.float64) * DEPTH_SCALE, 0, 65535).round().astype(np.uint16))
        cv2.imwrite(str(m_dir / depth_name(f)), ((dcan > 0) * 255).astype(np.uint8))
        json.dump({"field_of_view_rads": hfov_rads(K_out.tolist())}, open(i_dir / info_name(f), "w"))
        gt_can = letterbox(stretched_to_native_mask(gt, cw, ch).astype(np.uint8), OUT_W, OUT_H, 0, nearest=True)[0].astype(bool)
        kp = letterbox_point(r["point_uv"], lb)
        inst = {"keypoint": kp, "movable": "one_hand", "rigid": "yes",
                "kinematic": KINEMATIC["rot" if int(r["gt_type"]) == 1 else "trans"], "pull_or_push": "n/a",
                "affordance": kp, "bbox": bbox_norm(gt_can) if gt_can.any() else [0.0, 0.0, 1.0, 1.0],
                "mask": mask_polygon(gt_can) if gt_can.any() else [], "axis": [-1.0, -1.0, -1.0, -1.0]}
        entries.append({"img_name": name, "instances": [inst]})
        # wh = the canvas itself so threedoi_export's unroll_mask is a no-op; we unletterbox afterwards
        infos[name] = {"key": r["sample"], "rotated": False, "wh": [OUT_W, OUT_H], "K_out": K_out.tolist(), "keys": [r["sample"]]}
        meta[r["sample"]] = {"sample": r["sample"], "dataset": r["dataset"], "key": r["key"], "img_name": name, "lb": lb,
                             "K_out": K_out.tolist(), "has_depth": depth is not None, "mask_png": r["mask_png"],
                             "gt_type": int(r["gt_type"])}
    torch.save(entries, out / "3doi_sf3d" / "data_test.pt")
    json.dump(infos, open(out / "frames_test.json", "w"))
    json.dump(meta, open(out / "hv_meta.json", "w"))
    print(f"threedoi: {len(entries)} frames -> {out}")


# ---------------------------------------------------------------- A3VLM
def stage_a3vlm(out, image_root, samples):
    """rec_test.json (REC question per sample, answer empty), meta_test.json (their PadToSquare
    geometry: wh = native-aspect image, pad, K, d_min/d_max) + 448 px padded images + hv_meta.json.
    Sources without depth get the nominal depth range DEPTH_NOMINAL (flagged in hv_meta)."""
    from tools.baselines_sf3d.sf3d_to_a3vlm import REC_INSTRUCT, depth_range, pad_params, vqa

    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    fill = np.array([122.77, 116.75, 104.09]).round().astype(np.uint8)  # CLIP mean like the converter
    rec, metas, meta = [], {}, {}
    for r in samples:
        rgb, gt, depth = read_frame(r)
        ch, cw = rgb.shape[:2]
        x0, y0, s = pad_params(cw, ch)
        sq = np.empty((s, s, 3), np.uint8)
        sq[...] = fill
        sq[y0:y0 + ch, x0:x0 + cw] = rgb
        img448 = cv2.resize(sq, (448, 448), interpolation=cv2.INTER_AREA)
        name = f"hv_{r['sample']}.jpg"
        cv2.imwrite(str(out / "images" / name), cv2.cvtColor(img448, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if depth is not None and (depth > 0).any():
            dmin, dmax = depth_range(depth)
            nominal = False
        else:
            dmin, dmax = DEPTH_NOMINAL
            nominal = True
        K = native_K(r, cw, ch)
        img_path = str(Path(image_root) / name)
        rec.append(vqa(img_path, REC_INSTRUCT + r["desc"], "", key=r["sample"]))
        metas[name] = {"key_frame": r["sample"], "wh": [cw, ch], "pad": [int(x0), int(y0), int(s)], "K": K.tolist(),
                       "d_min": float(dmin), "d_max": float(dmax)}
        meta[r["sample"]] = {"sample": r["sample"], "dataset": r["dataset"], "key": r["key"], "image": name,
                             "wh": [cw, ch], "depth_nominal": nominal, "mask_png": r["mask_png"], "gt_type": int(r["gt_type"])}
    json.dump(rec, open(out / "rec_test.json", "w"))
    json.dump(metas, open(out / "meta_test.json", "w"))
    json.dump(meta, open(out / "hv_meta.json", "w"))
    print(f"a3vlm: {len(rec)} questions -> {out} (images referenced under {image_root})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["opd", "threedoi", "a3vlm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", default=str(SAMPLES))
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--stats", default="/workspace/datasets/baselines/data/opd_sf3d_512/stats.json", help="opd: training pixel stats")
    ap.add_argument("--image-root", default=None, help="a3vlm: absolute image dir as the A3VLM pod sees it")
    a = ap.parse_args(argv)
    samples = load_samples(a.samples, a.sources.split(","))
    if a.cmd == "opd":
        stage_opd(a.out, a.stats, samples)
    elif a.cmd == "threedoi":
        stage_threedoi(a.out, samples)
    else:
        stage_a3vlm(a.out, a.image_root or str(Path(a.out) / "images"), samples)


if __name__ == "__main__":
    main()
