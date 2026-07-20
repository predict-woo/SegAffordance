"""Dump a few visualized samples from each OPD dataset.

For every dataset (OPDSynth / OPDReal / OPDMulti) this picks N random
annotations from a split, renders the RGB frame with the part mask, the
projected motion origin, and the motion direction arrow, and writes a
JPEG plus a sidecar JSON with the raw annotation next to it.

Run on the pod from the repo root:
    python tools/show_opd_samples.py --out sample_viz --num 3
"""

import argparse
import json
import os
import random
import sys

import cv2
import h5py
import numpy as np
from pycocotools import mask as coco_mask

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.opd_intrinsics import intrinsic_matrix_from_camera  # noqa: E402

DATASETS = {
    "opdsynth": {"root": "/workspace/datasets/MotionDataset_h5_6.11", "is_multi": False},
    "opdreal": {"root": "/workspace/datasets/MotionDataset_h5_real", "is_multi": False},
    "opdmulti": {"root": "/workspace/datasets/OPDMulti/MotionDataset_h5", "is_multi": True},
}


def decode_mask(segm, height, width):
    if isinstance(segm, list):
        rles = coco_mask.frPyObjects(segm, height, width)
        return coco_mask.decode(coco_mask.merge(rles))
    if isinstance(segm, dict) and "counts" in segm:
        return coco_mask.decode(segm)
    return np.zeros((height, width), dtype=np.uint8)


def render(img_bgr, mask, origin_xy, motion_dir, caption_lines):
    out = img_bgr.copy()
    overlay = out.copy()
    overlay[mask > 0] = (0.4 * overlay[mask > 0] + 0.6 * np.array([0, 200, 0])).astype(np.uint8)
    out = cv2.addWeighted(overlay, 0.7, out, 0.3, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)

    if origin_xy is not None:
        ox, oy = int(round(origin_xy[0])), int(round(origin_xy[1]))
        cv2.circle(out, (ox, oy), 7, (0, 0, 255), -1)
        if motion_dir is not None:
            v = np.array(motion_dir[:2], dtype=float)
            n = np.linalg.norm(v)
            if n > 1e-6:
                v = v / n * 60
                cv2.arrowedLine(out, (ox, oy), (ox + int(v[0]), oy + int(v[1])), (255, 100, 0), 3, tipLength=0.3)

    pad = 22 * len(caption_lines) + 10
    canvas = np.zeros((out.shape[0] + pad, out.shape[1], 3), dtype=np.uint8)
    canvas[: out.shape[0]] = out
    for i, line in enumerate(caption_lines):
        cv2.putText(canvas, line, (8, out.shape[0] + 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_viz")
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    for name, cfg in DATASETS.items():
        root, is_multi = cfg["root"], cfg["is_multi"]
        with open(os.path.join(root, "annotations_bwdf", f"MotionNet_{args.split}.json")) as f:
            data = json.load(f)
        img_by_id = {im["id"]: im for im in data["images"]}
        cat_by_id = {c["id"]: c["name"] for c in data.get("categories", [])}

        h5 = h5py.File(os.path.join(root, f"{args.split}.h5"), "r")
        images_dset = h5[f"{args.split}_images"]
        fname_map = {n.decode(): i for i, n in enumerate(list(h5[f"{args.split}_filenames"]))}

        annos = rng.sample(data["annotations"], args.num)
        for k, anno in enumerate(annos):
            im = img_by_id[anno["image_id"]]
            base = os.path.basename(im["file_name"])
            if base not in fname_map:
                print(f"[{name}] skip {base}: not in h5")
                continue
            arr = images_dset[fname_map[base]][:, :, :3]
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            mask = decode_mask(anno["segmentation"], im["height"], im["width"])
            K = intrinsic_matrix_from_camera(im, is_multi)
            motion = anno["motion"]
            origin3d = motion.get("current_origin", motion.get("origin"))
            axis3d = motion.get("current_axis", motion.get("axis"))
            homo = K @ np.array(origin3d[:3])
            origin_xy = (homo[0] / homo[2], homo[1] / homo[2]) if homo[2] != 0 else None

            cat = cat_by_id.get(anno["category_id"], str(anno["category_id"]))
            caption = [
                f"{name}/{args.split}  img={base}  cat={cat}",
                f"motion={motion['type']}  desc={anno.get('description', 'MISSING')!r}",
            ]
            jpg = os.path.join(args.out, f"{name}_{k}.jpg")
            cv2.imwrite(jpg, render(img_bgr, mask, origin_xy, axis3d, caption))

            meta = {kk: vv for kk, vv in anno.items() if kk != "segmentation"}
            meta["_image"] = {kk: im[kk] for kk in ("file_name", "width", "height", "camera")}
            meta["_category_name"] = cat
            meta["_projected_origin_xy"] = origin_xy
            with open(os.path.join(args.out, f"{name}_{k}.json"), "w") as f:
                json.dump(meta, f, indent=2, default=str)
            print(f"[{name}] wrote {jpg}  ({cat}, {motion['type']})")
        h5.close()


if __name__ == "__main__":
    main()
