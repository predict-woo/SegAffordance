"""3DOI (monoarti SAM model) test-split inference -> shared prediction JSONL.

Runs their model exactly as test.py does (batches from their InteractionDataset, the GT
element point as the query, ``pred_masks.sigmoid() > 0.5``, kinematic argmax, the axis
decoded with ``axis_ops.line_angle_to_xyxy`` about the predicted box centre) and lifts
the 2D line to 3D the way their export_video does: RANSAC plane through the predicted
mask's points, rays of the two line endpoints intersected with that plane (rotation), or
the plane normal (translation).  Their lift uses the model's own depth aligned to GT;
ours uses the INPUT depth directly ("3DOI + depth-plane lifting" in the tables).

Run inside the 3DOI env, from anywhere:
  /opt/venv_3doi/bin/python tools/baselines_sf3d/run.py tools/baselines_sf3d/threedoi_export.py \
      --repo /workspace/datasets/baselines/repos/3DOI/monoarti --ckpt <checkpoint.pth> \
      --data /workspace/datasets/baselines/stage/3doi --out preds.jsonl [--limit N] [--batch 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baselines_sf3d import common as C  # noqa: E402
from tools.baselines_sf3d.sf3d_to_opd import R_ROLL  # noqa: E402

OUT_W, OUT_H = 1024, 768
KIN_TYPE = {1: 1, 2: 0}  # their kinematic ids (1 rotation, 2 translation) -> ours (1 rot, 0 trans); 0 freeform -> unmatched
PLANE_THRES = 0.03
MIN_PLANE_PTS = 20


# ---------------------------------------------------------------- geometry (pure numpy, unit-tested)
def backproject_mask(mask, depth, K):
    ys, xs = np.nonzero(mask & (depth > 0))
    if xs.size == 0:
        return np.zeros((0, 3))
    z = depth[ys, xs]
    return np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], 1)


def fit_plane_ransac(pts, thres=PLANE_THRES, iters=200, seed=0):
    """n . p = d with |n| = 1 (their depth_ops.fit_plane, deterministic seed, thres in metres)."""
    rng = np.random.RandomState(seed)
    best_inl, best = None, None
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        d = float(np.dot(n, s[0]))
        inl = np.abs(pts @ n - d) < thres
        if best is None or inl.sum() > best_inl.sum():
            best_inl, best = inl, (n, d)
    if best is None or best_inl.sum() < 3:
        return None
    q = pts[best_inl]
    c = q.mean(0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    n = vt[-1]
    if n[2] > 0:  # orient the normal towards the camera
        n = -n
    return n, float(np.dot(n, c))


def lift_line(line_xyxy, plane, K, w=OUT_W, h=OUT_H):
    """Normalised 2D line endpoints -> 3D points on the plane (their mesh_utils.get_pcd), or None
    when a ray is parallel to the plane or hits it behind the camera."""
    n, d = plane
    pts = []
    for u, v in np.asarray(line_xyxy, dtype=np.float64).reshape(2, 2):
        ray = np.array([(u * w - K[0, 2]) / K[0, 0], (v * h - K[1, 2]) / K[1, 1], 1.0])
        den = float(np.dot(n, ray))
        if abs(den) < 1e-9:
            return None
        t = d / den
        if t <= 0:
            return None
        pts.append(t * ray)
    return np.stack(pts)


def unroll_vec(v, rotated):
    return (R_ROLL.T @ np.asarray(v, dtype=np.float64)) if rotated else np.asarray(v, dtype=np.float64)


def unroll_mask(mask_1024x768, wh_rolled, rotated):
    """Model-resolution mask -> native frame (nearest), undoing the 90 deg clockwise roll."""
    w, h = wh_rolled
    m = cv2.resize(mask_1024x768.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if rotated:
        m = cv2.rotate(m, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return m.astype(bool)


def prediction(key, kin_id, line_xyxy, mask, depth, K, info, score=1.0):
    """One instance -> JSONL record. ``mask`` (H=768, W=1024) bool, ``depth`` metres (-1 holes),
    ``K`` = K_out, ``info`` = frames_test.json entry."""
    rec = {"key": key, "matched": False, "score": float(score), "mask_rle": None, "type": None, "axis_cam": None, "origin_cam": None}
    # the raw predicted 2D line (normalised xyxy in the 1024x768 image; -1s when their head emitted
    # none) and kinematic id, so figures can draw the undirected axis even without depth
    rec["line_2d"] = [float(x) for x in line_xyxy] if line_xyxy is not None else None
    rec["kin_id"] = int(kin_id)
    if mask.any():
        rec["mask_rle"] = C.rle_encode(unroll_mask(mask, info["wh"], info["rotated"]))
    if kin_id not in KIN_TYPE:
        return rec
    pts = backproject_mask(mask, depth, K)
    if len(pts) < MIN_PLANE_PTS:
        return rec
    plane = fit_plane_ransac(pts)
    if plane is None:
        return rec
    centroid = pts.mean(0)
    if KIN_TYPE[kin_id] == 1:
        if line_xyxy is None or line_xyxy[0] < 0:
            return rec
        seg = lift_line(line_xyxy, plane, K)
        if seg is None:
            return rec
        axis = seg[1] - seg[0]
        if np.linalg.norm(axis) < 1e-6:
            return rec
        axis = axis / np.linalg.norm(axis)
        origin = seg[0] + np.dot(centroid - seg[0], axis) * axis  # line point nearest the element
    else:
        axis, origin = plane[0], centroid
    rec["matched"] = True
    rec["type"] = KIN_TYPE[kin_id]
    rec["axis_cam"] = unroll_vec(axis, info["rotated"]).tolist()
    rec["origin_cam"] = unroll_vec(origin, info["rotated"]).tolist()
    return rec


# ---------------------------------------------------------------- their model
def run(repo, ckpt, data, out, split="test", limit=None, batch=2, workers=4, device="cuda"):
    import torch
    from hydra import compose, initialize_config_dir

    sys.path.insert(0, str(repo))
    from monoarti import axis_ops
    import monoarti.dataset as _mds
    from monoarti.dataset import prepare_datasets
    # their __getitem__ joins image paths onto the module-level DEFAULT_DATA_ROOT, not the data_root
    # argument (and retries a random entry on failure -> RecursionError); point it at our stage
    _mds.DEFAULT_DATA_ROOT = str(data)
    from monoarti.detr import box_ops
    from monoarti.model import build_model

    with initialize_config_dir(config_dir=str(Path(repo) / "configs"), version_base="1.2"):
        cfg = compose(config_name="sam_sf3d")
    model = build_model(cfg)
    sd = torch.load(ckpt, map_location="cpu")["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)
    model = model.to(device).eval()
    ds = prepare_datasets("3doi_sf3d", list(cfg.data.image_size), list(cfg.data.output_size), data_root=str(data),
                          load_depth=True, affordance_radius=cfg.data.affordance_radius, num_queries=cfg.data.num_queries, split=split)
    if limit:
        ds._entries = ds._entries[:limit]
    frames = json.load(open(Path(data) / f"frames_{split}.json"))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    t0 = time.time()
    n = n_ok = 0
    with open(out, "w") as f, torch.no_grad():
        for it, b in enumerate(dl):
            names = b.pop("img_name")
            b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()}
            o = model(**b, backward=False)
            masks = (o["pred_masks"].sigmoid() > 0.5).cpu().numpy()
            kin = o["pred_kinematic"].argmax(-1).cpu().numpy()
            for i, name in enumerate(names):
                info = frames[name]
                K = np.asarray(info["K_out"], dtype=np.float64)
                depth = b["depth"][i].cpu().numpy()
                valid = b["valid"][i].cpu().numpy().astype(bool)
                center = box_ops.box_xyxy_to_cxcywh(o["pred_boxes"][i]).clone()
                center[:, 2:] = center[:, :2]
                ax = o["pred_axis"][i]
                ax = torch.cat((torch.nn.functional.normalize(ax[:, :2]), ax[:, 2:]), dim=-1)
                lines = axis_ops.line_angle_to_xyxy(ax, center=center).cpu().numpy()
                for j, key in enumerate(info["keys"]):
                    if not valid[j]:
                        continue
                    rec = prediction(key, int(kin[i, j]), lines[j], masks[i, j], depth, K, info)
                    n += 1
                    n_ok += rec["matched"]
                    f.write(json.dumps(rec) + "\n")
            if (it + 1) % 100 == 0:
                print(f"  {it + 1}/{len(dl)} batches, {n} preds, {n_ok} matched, {time.time() - t0:.0f}s", flush=True)
    print(f"{n} predictions ({n_ok} with a lifted axis) -> {out}, {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    run(a.repo, a.ckpt, a.data, a.out, a.split, a.limit, a.batch, a.workers)


if __name__ == "__main__":
    main()
