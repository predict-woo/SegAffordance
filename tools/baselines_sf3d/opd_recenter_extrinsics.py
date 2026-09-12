"""Recentre the OPD 'world' frame per scene so OPDFormer-O/P's extrinsic head is learnable.

Our SF3D world frame is the laser-scan frame, whose origin can sit hundreds
of metres from the cameras (|t| ~ 390 m in some scenes). OPDFormer-P (BMOC_V1)
regresses the camera->object pose translation directly (EXTRINSIC_WEIGHT 30),
so those magnitudes swamp every other loss (total_loss ~1.6e4). OPDMulti's own
data uses object-centred frames with metre-scale translations. A per-scene
rigid shift of the world frame changes nothing observable (axis/origin are
stored in the camera frame; world coordinates only pass through the
extrinsic in the mapper and are mapped back with the predicted extrinsic at
inference), so we subtract, per scene (visit), the mean camera position.

Rewrites in place (idempotent, offsets recorded in scene_offsets.json):
  MotionDataset_h5/annotations/MotionNet_{train,valid,test}.json  images[*].camera.extrinsic
  obj_info.json                                                   [*].object_pose
Both are 16-float column-major cam->world matrices; only the translation
(elements 12:15) changes.

    python tools/baselines_sf3d/opd_recenter_extrinsics.py --data-dir /workspace/datasets/baselines/data/opd_sf3d
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def scene_of(file_name):
    return file_name.split("-")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    root = Path(a.data_dir)
    ann_dir = root / "MotionDataset_h5" / "annotations"
    off_path = root / "scene_offsets.json"
    if off_path.exists():
        print("already recentred:", off_path)
        return
    splits = {s: json.load(open(ann_dir / f"MotionNet_{s}.json")) for s in ["train", "valid", "test"]}
    per_scene = defaultdict(list)
    for d in splits.values():
        for im in d["images"]:
            per_scene[scene_of(im["file_name"])].append(np.asarray(im["camera"]["extrinsic"][12:15], float))
    offsets = {s: np.mean(v, axis=0) for s, v in per_scene.items()}
    for s, d in splits.items():
        for im in d["images"]:
            e = list(im["camera"]["extrinsic"])
            o = offsets[scene_of(im["file_name"])]
            e[12:15] = (np.asarray(e[12:15], float) - o).tolist()
            im["camera"]["extrinsic"] = e
        json.dump(d, open(ann_dir / f"MotionNet_{s}.json", "w"))
        print(s, "images recentred:", len(d["images"]))
    obj = json.load(open(root / "obj_info.json"))
    n = 0
    for key, info in obj.items():
        o = offsets[key.split("_")[0]]
        p = list(info["object_pose"])
        p[12:15] = (np.asarray(p[12:15], float) - o).tolist()
        info["object_pose"] = p
        n += 1
    json.dump(obj, open(root / "obj_info.json", "w"))
    json.dump({s: o.tolist() for s, o in offsets.items()}, open(off_path, "w"), indent=1)
    mags = np.concatenate([[np.linalg.norm(np.asarray(im["camera"]["extrinsic"][12:15])) for im in d["images"]] for d in splits.values()])
    print(f"object poses recentred: {n}; |t| after: p50 {np.percentile(mags, 50):.2f} m, p95 {np.percentile(mags, 95):.2f} m")


if __name__ == "__main__":
    main()
