"""Render annotated/original image pairs for VLM description generation.

Rendering scheme (per target annotation):
  - ALL movable-part masks in the frame get a thin orange outline, so the
    model can see which parts compete for a referring expression.
  - The TARGET part is filled semi-transparent green with a bold outline.
  - The motion is drawn as an arrow from the projected 3D origin along the
    projected 3D axis (red dot = origin, blue arrow = direction).
  - Both the annotated image and the untouched original are returned
    (2x upscaled), to be sent together in one VLM call.

Demo (run on the pod from the repo root):
    python tools/label_render.py --num 6 --out label_viz
renders pairs from OPDReal/OPDMulti train and asks Codex
(gpt-5.6-luna, low effort) for a description of each.
"""

import argparse
import collections
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
    "opdreal": {"root": "/workspace/datasets/MotionDataset_h5_real", "is_multi": False},
    "opdmulti": {"root": "/workspace/datasets/OPDMulti/MotionDataset_h5", "is_multi": True},
}

PROMPT_TEMPLATE = """\
You are writing language labels for a robot-interaction dataset.

Image 1 is the ORIGINAL photo of a scene. Image 2 is the SAME photo with
annotations added.

Step 1 — find the target. In image 2, exactly ONE part is FILLED solid green:
that green-filled part is the target, and it is the ONLY part you describe.
Orange outlines mark other movable parts — they are context to disambiguate
against, never the subject. The red dot marks where the target's motion
originates. The blue arrow is the target's 3D motion axis projected into the
image: use it to understand the motion in 3D, but do NOT describe the arrow's
on-screen direction — use the natural action verb for the motion.
Metadata: the target is a "{category}"; its motion type is "{motion_type}"
({motion_hint}).

Step 2 — write the label. ONE short imperative instruction naming WHICH part
to act on and WHAT the goal is — but NOT how the part moves. A person who
sees ONLY the original photo (and never the annotated one) must be able to
identify the exact target part, so disambiguate with spatial or appearance
cues (e.g. "the bottom drawer", "the left cabinet door under the sink")
whenever similar parts exist. Take the target's current state in the photo
into account (an already-open drawer gets closed or pulled out, not opened).

IMPORTANT — no motion mechanics in the text: do not name the motion kind
(no "rotate", "swing", "slide", "hinge") and use no direction words (no
"inward", "outward", "left", "right", "up", "down", "toward you") in the
action. Good: "Close the upper-left cabinet door.", "Push the open drawer
closed.", "Pull out the middle drawer." Bad: "Close the door by rotating it
shut.", "Push the drawer inward.", "Slide the drawer to the left."
(Direction/appearance words are still fine when they IDENTIFY the part,
e.g. "the left door".)

Never mention the annotations, colors, outlines, arrows, dots, markers, or
the word "highlighted". Reply with the sentence only."""

MOTION_HINTS = {
    "rotation": "it rotates around an axis, e.g. a hinged door or lid",
    "translation": "it slides along a direction, e.g. a drawer",
}


def decode_mask(segm, height, width):
    if isinstance(segm, list):
        rles = coco_mask.frPyObjects(segm, height, width)
        return coco_mask.decode(coco_mask.merge(rles))
    if isinstance(segm, dict) and "counts" in segm:
        return coco_mask.decode(segm)
    return np.zeros((height, width), dtype=np.uint8)


def project(K, p3d):
    homo = K @ np.asarray(p3d, dtype=float)[:3]
    if homo[2] == 0:
        return None
    return np.array([homo[0] / homo[2], homo[1] / homo[2]])


def render_pair(img_bgr, all_annos, target_anno, K, upscale=2):
    """Return (original, annotated) BGR images, both upscaled."""
    h, w = img_bgr.shape[:2]
    ann = img_bgr.copy()

    # thin outline around every movable part in the frame
    for a in all_annos:
        if a is target_anno:
            continue
        m = decode_mask(a["segmentation"], h, w)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ann, contours, -1, (0, 165, 255), 1)  # orange

    # target: green fill + bold outline
    tm = decode_mask(target_anno["segmentation"], h, w)
    overlay = ann.copy()
    overlay[tm > 0] = (0.45 * overlay[tm > 0] + 0.55 * np.array([0, 210, 0])).astype(np.uint8)
    ann = cv2.addWeighted(overlay, 0.75, ann, 0.25, 0)
    contours, _ = cv2.findContours(tm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ann, contours, -1, (0, 255, 0), 2)

    # motion arrow: project origin and origin + axis into the image
    motion = target_anno["motion"]
    origin = motion.get("current_origin", motion.get("origin"))
    axis = motion.get("current_axis", motion.get("axis"))
    p0 = project(K, origin)
    p1 = project(K, np.asarray(origin, dtype=float) + 0.25 * np.asarray(axis, dtype=float))
    if p0 is not None:
        p0i = (int(round(p0[0])), int(round(p0[1])))
        if p1 is not None and np.linalg.norm(p1 - p0) > 1e-3:
            d = (p1 - p0) / np.linalg.norm(p1 - p0) * min(50, 0.25 * max(h, w))
            p1i = (int(round(p0[0] + d[0])), int(round(p0[1] + d[1])))
            cv2.arrowedLine(ann, p0i, p1i, (255, 80, 0), 2, tipLength=0.35)
        cv2.circle(ann, p0i, 5, (0, 0, 255), -1)

    if upscale != 1:
        dim = (w * upscale, h * upscale)
        img_bgr = cv2.resize(img_bgr, dim, interpolation=cv2.INTER_CUBIC)
        ann = cv2.resize(ann, dim, interpolation=cv2.INTER_CUBIC)
    return img_bgr, ann


class FrameSource:
    """Loads frames + grouped annotations for one dataset split."""

    def __init__(self, name, split="train"):
        cfg = DATASETS[name]
        self.name, self.is_multi = name, cfg["is_multi"]
        root = cfg["root"]
        with open(os.path.join(root, "annotations_bwdf", f"MotionNet_{split}.json")) as f:
            self.data = json.load(f)
        self.img_by_id = {im["id"]: im for im in self.data["images"]}
        self.cat_by_id = {c["id"]: c["name"] for c in self.data.get("categories", [])}
        self.by_image = collections.defaultdict(list)
        for a in self.data["annotations"]:
            self.by_image[a["image_id"]].append(a)
        self.h5 = h5py.File(os.path.join(root, f"{split}.h5"), "r")
        self.images_dset = self.h5[f"{split}_images"]
        self.fname_map = {n.decode(): i for i, n in
                          enumerate(list(self.h5[f"{split}_filenames"]))}

    def load_bgr(self, image_id):
        base = os.path.basename(self.img_by_id[image_id]["file_name"])
        arr = self.images_dset[self.fname_map[base]][:, :, :3]
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def prompt_for(self, anno):
        cat = self.cat_by_id.get(anno["category_id"], "part")
        mtype = anno["motion"]["type"]
        return PROMPT_TEMPLATE.format(category=cat, motion_type=mtype,
                                      motion_hint=MOTION_HINTS.get(mtype, ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="label_viz")
    ap.add_argument("--num", type=int, default=6, help="samples per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--no-codex", action="store_true", help="render only")
    args = ap.parse_args()

    from codex_client import CodexClient, CodexError

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    outdir = os.path.abspath(args.out)

    jobs = []
    for name in DATASETS:
        src = FrameSource(name)
        # prefer frames with several parts — that's where disambiguation matters
        multi_part = [i for i, annos in src.by_image.items() if len(annos) >= 2]
        single = [i for i, annos in src.by_image.items() if len(annos) == 1]
        picks = (rng.sample(multi_part, min(args.num - 1, len(multi_part)))
                 + rng.sample(single, min(1, len(single))))
        for k, image_id in enumerate(picks):
            annos = src.by_image[image_id]
            target = rng.choice(annos)
            im = src.img_by_id[image_id]
            K = intrinsic_matrix_from_camera(im, src.is_multi)
            orig, ann = render_pair(src.load_bgr(image_id), annos, target, K)
            p_orig = os.path.join(outdir, f"{name}_{k}_orig.jpg")
            p_ann = os.path.join(outdir, f"{name}_{k}_ann.jpg")
            cv2.imwrite(p_orig, orig)
            cv2.imwrite(p_ann, ann)
            jobs.append((name, k, len(annos), src.prompt_for(target), p_orig, p_ann))
        src.h5.close()

    if args.no_codex:
        for j in jobs:
            print(f"[{j[0]}_{j[1]}] rendered ({j[2]} parts)")
        return

    for name, k, n_parts, prompt, p_orig, p_ann in jobs:
        try:
            with CodexClient(model=args.model, effort=args.effort) as c:
                desc = c.describe(prompt, images=[p_orig, p_ann])
        except CodexError as e:
            desc = f"ERROR: {e}"
        print(f"[{name}_{k}] ({n_parts} parts in frame) -> {desc}")


if __name__ == "__main__":
    main()
