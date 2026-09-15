"""Gravity ("real up") direction for the EPIC annotator clouds, from EPIC-Fields camera trajectories.

EPIC records are in a head-mounted camera frame, so the camera's own up tilts with the wearer, and the
EPIC-Fields COLMAP world frame is arbitrary. Estimate: the wearer's mean head-up over ALL registered frames
of the video (world_up = mean of R_c2w (0,-1,0)) is close to gravity; express it in the record's camera with
that frame's pose (up_cam = R_w2c world_up) and store it in the record's npz as `up_cam` (unit, OpenCV camera
frame). The viewer rotates the scene so this direction is vertical. Records without a registered frame within
--max-offset keep the camera up (up_cam = (0,-1,0), flagged up_source = "camera").

  python tools/epic_up_direction.py --clouds /workspace/datasets/epic_gt_annot/clouds --fields /workspace/datasets/epic_fields
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epic_fields_overlay import frame_index, nearest_frame, quat_to_rot  # noqa: E402


def video_up(images):
    """Mean camera-up in the COLMAP world frame over all registered frames (unit)."""
    acc = np.zeros(3)
    for pose in images.values():
        R = quat_to_rot(*pose[:4])           # world -> camera
        acc += R.T @ np.array([0.0, -1.0, 0.0])   # camera up (OpenCV -y) in world
    return acc / max(np.linalg.norm(acc), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", required=True)
    ap.add_argument("--fields", required=True)
    ap.add_argument("--max-offset", type=int, default=30)
    a = ap.parse_args()
    recs = json.load(open(os.path.join(a.clouds, "index.json")))["records"]
    cache = {}
    n_ok = 0
    for r in recs:
        fn = os.path.join(a.clouds, r["file"]); z = dict(np.load(fn, allow_pickle=True))
        video = r["key"].split("/")[0]
        jf = os.path.join(a.fields, video + ".json")
        up_cam, src, tilt = np.array([0.0, -1.0, 0.0]), "camera", float("nan")
        if os.path.exists(jf):
            if video not in cache:
                cache[video] = json.load(open(jf))
            images = cache[video]["images"]
            wup = video_up(images)
            name = str(z["fields_frame"]) if "fields_frame" in z and str(z["fields_frame"]) in images else None
            if name is None and "fields_onset" in z:
                name = nearest_frame(images, int(z["fields_onset"]), a.max_offset)
            if name is None and r.get("onset_frame") is not None:
                name = nearest_frame(images, int(r["onset_frame"]), a.max_offset)
            if name is not None:
                R = quat_to_rot(*images[name][:4])
                up_cam = R @ wup; up_cam /= max(np.linalg.norm(up_cam), 1e-9); src = f"fields:{name}"
                tilt = float(np.degrees(np.arccos(np.clip(np.dot(up_cam, [0, -1, 0]), -1, 1))))
                n_ok += 1
        z["up_cam"] = up_cam.astype(np.float32); z["up_source"] = np.array(src)
        np.savez(fn, **z)
        print(f"{r['key']:28s} up {np.round(up_cam, 3).tolist()}  tilt vs camera-up {tilt:5.1f} deg  ({src})")
    print(f"{n_ok}/{len(recs)} records with a gravity estimate from EPIC-Fields")


if __name__ == "__main__":
    main()
