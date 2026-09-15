"""EPIC-KITCHENS test records -> per-record metric point clouds for manual 3D axis annotation.

For every record of the EPIC test split (the `va` subset of config/epic_v1_rgb_scalefree.yaml's scene
split, 53 records) this un-stretches the 512x512 training frame back to its native aspect ratio (default
width 1024), runs a monocular METRIC depth model (Depth Anything V2 metric-depth, indoor, Large —
`depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` via HuggingFace transformers), back-projects
every pixel with the intrinsics scaled to the render size into a camera-frame cloud (OpenCV convention:
x right, y down, z forward, metres) and writes one .npz per record plus an index.json in test order.
The clouds are consumed by tools/epic_axis_annotator.py. Runs on the pod (GPU); transformers is not in
requirements.lock, install it into the venv in the same dev.sh run command:

  uv pip install --python /opt/venv/bin/python transformers && \\
  HF_HOME=/root/hfcache /opt/venv/bin/python tools/epic_cloud_precompute.py --out /workspace/datasets/epic_gt_annot/clouds

npz fields: xyz (float32 N x 3), rgb (uint8 N x 3), in_mask (bool N), point_uv (2, render px), desc (str),
key (str), type_label (int, 1 = revolute / 0 = prismatic rule/VLM label), K_render (3x3), size (W, H).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.scenefun3d import SF3DDataset, split_dataset_by_scene  # noqa: E402

ROOT = "/workspace/datasets/epic_processed_2d"
KEYS = "/workspace/cache/epic_2d_keys_v1.pkl"
MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def load_test_subset(split_config):
    dcfg = yaml.safe_load(open(split_config))["data"]
    ds = SF3DDataset(
        lmdb_data_root=ROOT, lmdb_path=f"{ROOT}/data.lmdb", frame_cache_path=f"{ROOT}/frames.lmdb",
        key_cache_path=KEYS, image_size_for_mask_reconstruction=(512, 512), return_trajectory_2d=True,
        point_source="element", fast_pipeline=True, load_depth=False,
        min_revolute_radius=0.0, min_mask_area_frac=0.0, edge_margin_frac=0.0, sensor_max_occluded_frac=0.5,
    )
    _, va = split_dataset_by_scene(ds, dcfg.get("val_split_ratio", 0.15), dcfg.get("manual_seed", 42))
    return ds, va


def predict_depth(proc, model, rgb, device):
    """Metric depth (metres) at the resolution of `rgb` (H x W x 3 uint8 RGB)."""
    from PIL import Image
    inputs = proc(images=Image.fromarray(rgb), return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    h, w = rgb.shape[:2]
    try:
        depth = proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]["predicted_depth"]
    except AttributeError:  # older transformers: resize the raw prediction ourselves
        depth = torch.nn.functional.interpolate(out.predicted_depth[None], size=(h, w), mode="bicubic", align_corners=False)[0, 0]
    return depth.float().cpu().numpy()


def backproject(depth, K):
    """Every pixel centre -> camera-frame xyz (OpenCV axes). Returns (H*W, 3) float32."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
    z = depth.astype(np.float32)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], -1).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/datasets/epic_gt_annot/clouds")
    ap.add_argument("--width", type=int, default=1024, help="render width; height follows the native aspect")
    ap.add_argument("--max-points", type=int, default=400_000)
    ap.add_argument("--split-config", default="config/epic_v1_rgb_scalefree.yaml")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/root/hfcache")
    os.environ.pop("HF_HUB_OFFLINE", None)  # the model must be downloadable on first use
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
    print(f"loaded {MODEL_ID} on {device} in {time.time() - t0:.0f}s", flush=True)
    ds, va = load_test_subset(a.split_config)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    index = []
    n = min(len(va), a.limit) if a.limit else len(va)
    for j in range(n):
        key = ds.item_keys[va.indices[j]].decode()
        (img_t, _d, desc, mask_t, _b, point_gt, _m, type_gt, img_size, fname, _o, K, *_rest) = va[j]
        W0, H0 = float(img_size[0]), float(img_size[1])
        Wr = a.width
        Hr = int(round(Wr * H0 / W0))
        rgb = cv2.resize(img_t.permute(1, 2, 0).numpy(), (Wr, Hr), interpolation=cv2.INTER_CUBIC)  # RGB uint8
        mask = cv2.resize(mask_t[0].numpy(), (Wr, Hr), interpolation=cv2.INTER_NEAREST) > 0.5
        K_render = K.numpy().astype(np.float64).copy()
        K_render[0] *= Wr / W0
        K_render[1] *= Hr / H0
        depth = predict_depth(proc, model, rgb, device)
        xyz = backproject(depth, K_render)
        keep = np.flatnonzero(np.isfinite(xyz).all(1) & (xyz[:, 2] > 0.05))
        if keep.size > a.max_points:
            keep = np.sort(rng.choice(keep, a.max_points, replace=False))
        rec = dict(
            xyz=xyz[keep].astype(np.float32), rgb=rgb.reshape(-1, 3)[keep], in_mask=mask.reshape(-1)[keep],
            point_uv=(point_gt.numpy() * np.array([Wr, Hr])).astype(np.float32), desc=str(desc), key=key,
            type_label=int(type_gt), K_render=K_render.astype(np.float32), size=np.array([Wr, Hr], np.int32),
        )
        fn = key.replace("/", "__") + ".npz"
        np.savez_compressed(os.path.join(a.out, fn), **rec)
        index.append(dict(i=j, key=key, file=fn, desc=str(desc), type_label=int(type_gt), n_points=int(keep.size),
                          size=[Wr, Hr], depth_min=float(depth.min()), depth_median=float(np.median(depth)), depth_max=float(depth.max())))
        print(f"{j:3d}/{n} {key:28s} {Wr}x{Hr} pts={keep.size:7d} depth med {np.median(depth):.2f} m [{depth.min():.2f}, {depth.max():.2f}] "
              f"mask={int(mask.sum())} type={'rot' if int(type_gt) == 1 else 'trans'} | {desc}", flush=True)
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(dict(model=MODEL_ID, width=a.width, max_points=a.max_points, records=index), f, indent=1)
    print(f"wrote {len(index)} clouds to {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
