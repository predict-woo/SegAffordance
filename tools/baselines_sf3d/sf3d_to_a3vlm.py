"""SF3D LMDB -> A3VLM (ManipVQA) training / test JSONs.

Reproduces the released data generator (A3VLM/data_gen/vqa_task_construction.py,
partnet_label.py, point_render.py::BBox3D) on our frames:

* image: the native frame padded to a centred square with the CLIP mean colour
  (exactly what their ``padded_resize`` transform does) and stored at 448 px, so
  every normalised coordinate below is w.r.t. the padded square = what the
  model sees;
* a manipulable part = one SF3D element; its 3D box = axis-aligned camera-frame
  box around the element's back-projected mask pixels (input depth, 2-98 %
  bounds), serialised as their 8 projected vertices ``[[u,v,d], ...]`` with
  ``d = (Z - d_min) / (d_max - d_min)`` and (d_min, d_max) the per-image
  min / max of the valid input depth (partnet_label.py:692-699);
* the joint axis = two 3D points on the GT axis around the element
  (``[x0,y0,z0,x1,y1,z1]`` in the same (u,v,d) space), joint type revolute /
  prismatic;
* tasks (their 3D set minus the robot "grounding" task, which has no SF3D
  counterpart): DET (all parts + boxes), REC (description -> box, their
  single_link_3d_rec with our description as the link name), REG-Joint
  (box -> type + axis).  Test JSONs carry ``gpt: null`` and our ``key``.

Usage (inside the A3VLM env or ours; numpy + PIL + lmdb only):
  python tools/baselines_sf3d/run.py tools/baselines_sf3d/sf3d_to_a3vlm.py \
      --out /workspace/datasets/baselines/stage/a3vlm --workers 32 [--limit-frames N]
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baselines_sf3d import common as C  # noqa: E402

IMG_SIZE = 448
CLIP_MEAN_RGB = (int(0.48145466 * 255), int(0.4578275 * 255), int(0.40821073 * 255))
MIN_VALID_PX = 10
MIN_EXTENT = 0.02
AXIS_LENGTHS = (0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
POD_IMAGE_ROOT = "/workspace/bl/data/a3vlm/images"

DET_INSTRUCT = "Detect all manipulable object parts and provide their 3D bounding boxes."
REC_INSTRUCT = "Please provide the 3D bounding box of the region this sentence describes: "
JOINT_INSTRUCT = "Please provide the joint's type and its 3D axis linked to the object part {REF}."
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
MAX_DET = 10
JOINT_TYPE = {"rot": "revolute", "trans": "prismatic"}

_NUM = r"(-?\d+(?:\.\d+)?)"
BOX_RE = re.compile(r"\[\s*" + r"\s*,\s*".join([r"\[\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\]"] * 8) + r"\s*\]")
AXIS_RE = re.compile(r"<axis>\s*(\w+)\s*</axis>\s*\[\s*" + r"\s*,\s*".join([_NUM] * 6) + r"\s*\]")


# ------------------------------------------------------------------ geometry
def pad_params(w, h):
    """PadToSquare (LLaVA): landscape -> paste at (0, (w-h)//2); portrait -> ((h-w)//2, 0)."""
    s = max(w, h)
    return (s - w) // 2, (s - h) // 2, s


def project_uvd_raw(pts, K, pad, dmin, dmax):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64)
    x0, y0, s = pad
    z = pts[:, 2]
    u = (K[0, 0] * pts[:, 0] / z + K[0, 2] + x0) / s
    v = (K[1, 1] * pts[:, 1] / z + K[1, 2] + y0) / s
    d = (np.abs(z) - dmin) / (dmax - dmin + 1e-6)
    return np.stack([u, v, d], 1)


def project_uvd(pts, K, pad, dmin, dmax):
    """Camera points (N,3) -> their [u, v, d] in the padded square, clipped to [0,1]
    (point_render.py::BBox3D.project_points in OpenCV signs)."""
    return np.clip(project_uvd_raw(pts, K, pad, dmin, dmax), 0.0, 1.0)


def unproject_uvd(uvd, K, pad, dmin, dmax):
    """Inverse of project_uvd (for un-clipped points)."""
    uvd = np.asarray(uvd, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64)
    x0, y0, s = pad
    z = uvd[:, 2] * (dmax - dmin + 1e-6) + dmin
    x = (uvd[:, 0] * s - x0 - K[0, 2]) * z / K[0, 0]
    y = (uvd[:, 1] * s - y0 - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1)


def box_points(center, extent):
    """Axis-aligned box vertices in point_render.py::BBox3D.get_points order (R = I)."""
    c = np.asarray(center, dtype=np.float64)
    hx, hy, hz = np.asarray(extent, dtype=np.float64) / 2
    x, y, z = np.array([hx, 0, 0]), np.array([0, hy, 0]), np.array([0, 0, hz])
    return np.stack([c - x - y - z, c + x - y - z, c - x + y - z, c - x - y + z,
                     c + x + y + z, c - x + y + z, c + x - y + z, c + x + y - z])


def fmt_box(uvd):
    uvd = np.asarray(uvd).reshape(8, 3)
    return "[" + ",".join("[{:.2f},{:.2f},{:.2f}]".format(*p) for p in uvd) + "]"


def fmt_axis(joint_type, uvd):
    uvd = np.asarray(uvd).reshape(2, 3)
    return "<axis>{}</axis>[{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]".format(joint_type, *uvd.ravel())


def parse_box(text):
    """First 8-vertex box in ``text`` -> (8,3) array, or None."""
    m = BOX_RE.search(text or "")
    return None if m is None else np.array([float(g) for g in m.groups()], dtype=np.float64).reshape(8, 3)


def parse_axis(text):
    """'<axis>type</axis>[6 numbers]' -> (type, (2,3) array), or None."""
    m = AXIS_RE.search(text or "")
    if m is None:
        return None
    return m.group(1).lower(), np.array([float(g) for g in m.groups()[1:]], dtype=np.float64).reshape(2, 3)


def backproject(mask, depth_m, K):
    """Valid mask pixels -> camera points (N,3)."""
    ys, xs = np.nonzero(mask & (depth_m > 0))
    if xs.size == 0:
        return np.zeros((0, 3))
    z = depth_m[ys, xs]
    K = np.asarray(K, dtype=np.float64)
    x = (xs + 0.0 - K[0, 2]) * z / K[0, 0]
    y = (ys + 0.0 - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1)


def element_geometry(rec, depth_m):
    """One record -> box (center, extent), axis endpoints and joint type, in the camera frame.
    Returns None when fewer than MIN_VALID_PX mask pixels have depth."""
    w, h = rec["image_dimensions_wh"]
    mask = C.mask_from_coords(rec["mask_coordinates_yx"], h, w).astype(bool)
    pts = backproject(mask, depth_m, rec["camera_intrinsics"])
    if len(pts) < MIN_VALID_PX:
        return None
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    center = (lo + hi) / 2
    extent = np.maximum(hi - lo, MIN_EXTENT)
    fs = rec["motion_info"]["frame_specific_motion_data"]
    n = np.asarray(fs["motion_dir_3d_camera_coords"], dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    o = np.asarray(fs["motion_origin_3d_camera_coords"], dtype=np.float64)
    mtype = rec["motion_info"]["original_motion_data"]["motion_type"]
    centroid = pts.mean(0)
    if mtype == "rot":
        anchor = o + np.dot(centroid - o, n) * n  # foot of the centroid on the GT axis
    else:
        anchor = centroid
    axis_pts, length = axis_segment(anchor, n, rec["camera_intrinsics"], (w, h), depth_range(depth_m))
    return {"center": center, "extent": extent, "axis_pts": axis_pts, "axis_len": length,
            "joint_type": JOINT_TYPE[mtype], "n_valid": int(len(pts)), "n_mask": int(mask.sum())}


def axis_segment(anchor, n, K, wh, drange):
    """Two points ``anchor -+ L/2 n`` with the longest L in AXIS_LENGTHS whose (u, v, d) all stay
    inside [0, 1] (their generator clips to [0, 1], which would corrupt a segment leaving the frame).
    Their PartNet segments span the whole link; a long segment keeps the 2-decimal quantisation of
    the answer format (~2-4 cm per step at SF3D depths) from dominating the direction error."""
    pad = pad_params(*wh)
    for length in AXIS_LENGTHS:
        pts = np.stack([anchor - 0.5 * length * n, anchor + 0.5 * length * n])
        if (pts[:, 2] > 0.05).all():
            uvd = project_uvd_raw(pts, K, pad, *drange)
            if (uvd >= 0).all() and (uvd <= 1).all():
                return pts, length
    length = AXIS_LENGTHS[-1]
    return np.stack([anchor - 0.5 * length * n, anchor + 0.5 * length * n]), length


def depth_range(depth_m):
    v = depth_m[depth_m > 0]
    if v.size == 0:
        return 0.0, 1.0
    return float(v.min()), float(v.max())


# ------------------------------------------------------------------ tasks
def det_answer(parts):
    """parts: list of (label, box_str) already sorted by 2D area (largest first)."""
    parts = parts[:MAX_DET]
    n = len(parts)
    if n == 1:
        s = "There is one manipulable object part with its 3d bounding box: "
    else:
        s = f"There are {NUMBER_WORDS[n]} manipulable object parts with their 3d bounding boxes: "
    body = ",".join(f"<box>{lab}</box>{box}" for lab, box in parts)
    return s + body + "."


def vqa(image, question, answer, key=None):
    d = {"image": image, "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer}]}
    if key is not None:
        d["key"] = key
    return d


def label_text(rec):
    return rec["label_info"]["label"].replace("_", " ")


# ------------------------------------------------------------------ per frame
_ENV = None


def _env():
    global _ENV
    if _ENV is None:
        _ENV = C.open_lmdb()
    return _ENV


def image_name(frame_key):
    visit, video, ts = frame_key.split("/")
    return f"{visit}_{video}_{ts}.jpg"


def process_frame(args):
    frame_key, keys, out_dir = args
    with _env().begin() as txn:
        recs = [C.read_record(txn, k) for k in keys]
    rec0 = recs[0]
    root = Path(C.LMDB_ROOT)
    img = Image.open(root / "images" / rec0["rgb_image_path"]).convert("RGB")
    depth = np.asarray(Image.open(root / "depth" / rec0["depth_image_path"]))
    assert depth.dtype == np.uint16, rec0["depth_image_path"]
    w, h = rec0["image_dimensions_wh"]
    assert img.size == (w, h) and depth.shape == (h, w), (frame_key, img.size, depth.shape)
    depth_m = depth.astype(np.float64) / 1000.0
    pad = pad_params(w, h)
    name = image_name(frame_key)
    dst = Path(out_dir) / "images" / name
    if not dst.exists():
        sq = Image.new("RGB", (pad[2], pad[2]), CLIP_MEAN_RGB)
        sq.paste(img, (pad[0], pad[1]))
        sq.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC).save(dst, quality=95)
    dmin, dmax = depth_range(depth_m)
    K = rec0["camera_intrinsics"]
    meta = {"key_frame": frame_key, "wh": [int(w), int(h)], "pad": [int(pad[0]), int(pad[1]), int(pad[2])],
            "K": np.asarray(K, dtype=np.float64).tolist(), "d_min": dmin, "d_max": dmax}
    elems = []
    for rec, key in zip(recs, keys):
        g = element_geometry(rec, depth_m)
        mask_area = int(len(rec["mask_coordinates_yx"] or []))
        if g is None:
            elems.append({"key": key, "ok": False, "description": rec["description"], "label": label_text(rec),
                          "joint_type": JOINT_TYPE[rec["motion_info"]["original_motion_data"]["motion_type"]],
                          "area": mask_area})
            continue
        box = fmt_box(project_uvd(box_points(g["center"], g["extent"]), K, pad, dmin, dmax))
        axis = fmt_axis(g["joint_type"], project_uvd(g["axis_pts"], K, pad, dmin, dmax))
        elems.append({"key": key, "ok": True, "description": rec["description"], "label": label_text(rec),
                      "joint_type": g["joint_type"], "box": box, "axis": axis, "area": mask_area,
                      "n_valid": g["n_valid"]})
    return name, meta, elems


def plan_jobs(keys, limit_frames=None):
    by_frame = defaultdict(list)
    for k in keys:
        by_frame[C.frame_of(k)].append(k)
    frames = sorted(by_frame)
    if limit_frames:
        frames = frames[:limit_frames]
    return [(f, by_frame[f]) for f in frames]


def build_split(split, keys, out, workers, limit_frames, image_root):
    jobs = [(f, ks, str(out)) for f, ks in plan_jobs(keys, limit_frames)]
    t0 = time.time()
    metas, det, rec_t, joint = {}, [], [], []
    n_bad = 0
    with Pool(workers) as pool:
        for i, (name, meta, elems) in enumerate(pool.imap_unordered(process_frame, jobs, chunksize=4)):
            metas[name] = meta
            img_path = f"{image_root}/{name}"
            is_test = split == "test"
            parts = sorted([e for e in elems if e["ok"]], key=lambda e: -e["area"])
            if parts and not is_test:
                det.append(vqa(img_path, DET_INSTRUCT, det_answer([(e["label"], e["box"]) for e in parts])))
            for e in elems:
                if not e["ok"]:
                    n_bad += 1
                    if not is_test:
                        continue
                    e = {**e, "box": fmt_box(np.full((8, 3), 0.5)), "axis": fmt_axis(e["joint_type"], np.full((2, 3), 0.5))}
                rec_t.append(vqa(img_path, REC_INSTRUCT + e["description"], e["box"], key=e["key"] if is_test else None))
                joint.append(vqa(img_path, JOINT_INSTRUCT.format(REF=e["box"]), e["axis"], key=e["key"] if is_test else None))
            if (i + 1) % 2000 == 0:
                print(f"  {split}: {i + 1}/{len(jobs)} frames, {time.time() - t0:.0f}s", flush=True)
    json.dump(metas, open(out / f"meta_{split}.json", "w"))
    for tag, lst in (("det", det), ("rec", rec_t), ("joint", joint)):
        if lst:
            json.dump(lst, open(out / f"{tag}_{split}.json", "w"))
    print(f"{split}: {len(jobs)} frames, det {len(det)}, rec {len(rec_t)}, joint {len(joint)}, "
          f"elements without depth {n_bad}, {time.time() - t0:.0f}s", flush=True)
    return {"frames": len(jobs), "det": len(det), "rec": len(rec_t), "joint": len(joint), "no_depth": n_bad}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=str(Path(__file__).resolve().parents[2] / "experiments/baselines_sf3d/splits.json"))
    ap.add_argument("--key-cache", default=C.KEY_CACHE)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--image-root", default=POD_IMAGE_ROOT, help="absolute image dir as seen by the training pod")
    ap.add_argument("--only", default="train,bvalid,test")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    keys = C.load_keys(a.key_cache)
    splits = json.load(open(a.splits))
    per = C.keys_by_split(keys, splits)
    stats = {}
    for split in a.only.split(","):
        stats[split] = build_split(split, per[split], out, a.workers, a.limit_frames, a.image_root)
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
