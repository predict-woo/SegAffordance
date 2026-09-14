"""Serialise our model's per-frame prediction to a JSONL record and back, so panels can be re-rendered
without a GPU (tools/sf3d_render_preds.py) and metrics recomputed from the raw prediction.

One record per (model, frame) holds exactly what tools/predict_image.draw_prediction consumes:
  model, val_idx, key, desc, gt_type,
  mask_rle      COCO RLE of (sigmoid(mask_logits) > 0.5) at the logits' own resolution (mask_h, mask_w)
  type_logits   [2]      motion_type_logits (softmax -> p_rev)
  point_3d      [3]      lifted interaction point (camera, metres) = the trajectory anchor
  point_uv      [2]      interaction point, normalised image coords
  motion        [3]      axis direction (unit)
  origin        [3]|null hinge point (camera, metres), revolute readout
  origin_uv     [2]|null hinge heatmap argmax, normalised
  trajectory    [N,3]    decoded trajectory RELATIVE to point_3d (after the scale-free rescale)
record_to_out() rebuilds a namespace with the same attributes as tensors (mask_logits = +-10 from the RLE).
"""
import json
from types import SimpleNamespace

import numpy as np
import torch


def _lst(t):
    return None if t is None else [round(float(v), 5) for v in t.detach().cpu().float().reshape(-1)]


def out_to_record(out, model_name, val_idx, key, desc, gt_type):
    from pycocotools import mask as mask_utils
    pm = (torch.sigmoid(out.mask_logits)[0, 0].detach().cpu() > 0.5).numpy().astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(pm))
    rle["counts"] = rle["counts"].decode("ascii")
    traj = None
    if out.trajectory_pred is not None:
        traj = [[round(float(v), 5) for v in p] for p in out.trajectory_pred[0].detach().cpu().float().tolist()]
    return {
        "model": model_name, "val_idx": int(val_idx), "key": key, "desc": desc, "gt_type": int(gt_type),
        "mask_rle": rle, "mask_h": int(pm.shape[0]), "mask_w": int(pm.shape[1]),
        "type_logits": _lst(out.motion_type_logits[0]) if out.motion_type_logits is not None else None,
        "point_3d": _lst(out.point_3d_pred[0]) if out.point_3d_pred is not None else None,
        "point_uv": _lst(out.point_uv[0]) if out.point_uv is not None else None,
        "motion": _lst(out.motion_pred[0]) if out.motion_pred is not None else None,
        "origin": _lst(out.origin_pred[0]) if out.origin_pred is not None else None,
        "origin_uv": _lst(out.origin_uv[0]) if out.origin_uv is not None else None,
        "trajectory": traj,
    }


def record_to_out(rec):
    from pycocotools import mask as mask_utils
    rle = dict(rec["mask_rle"]); rle["counts"] = rle["counts"].encode("ascii")
    m = torch.from_numpy(mask_utils.decode(rle).astype(np.float32))
    t = lambda v: None if v is None else torch.tensor(v, dtype=torch.float32)[None]
    return SimpleNamespace(
        mask_logits=(m * 20.0 - 10.0)[None, None],
        motion_type_logits=t(rec["type_logits"]), point_3d_pred=t(rec["point_3d"]), point_uv=t(rec["point_uv"]),
        motion_pred=t(rec["motion"]), origin_pred=t(rec["origin"]), origin_uv=t(rec["origin_uv"]),
        trajectory_pred=None if rec["trajectory"] is None else torch.tensor(rec["trajectory"], dtype=torch.float32)[None],
    )


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_record(fout, rec):
    fout.write(json.dumps(rec) + "\n"); fout.flush()
