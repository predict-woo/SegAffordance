#!/usr/bin/env python3
"""Hand pose that points the hand camera straight at a body-frame target from a given distance.

  python3 plan_aim.py --target X Y Z --normal NX NY NZ --dist 0.5 --T-hand-cam '<4x4 json>' -o aim.json

target: the predicted handle centroid in the CURRENT body frame; normal: door normal pointing toward the robot
(camera sits at target + normal*dist and looks along -normal). T_hand_cam: spot/hand -> camera optical frame,
from `spotctl tf spot/hand spot/hand_color_image_sensor`. Output: hand pose (xyz + quat) for `spotctl poseq`.
Optical frame: x right, y down, z forward. The camera's x is kept horizontal (image level).
"""
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, nargs=3, required=True)
    ap.add_argument("--normal", type=float, nargs=3, required=True)
    ap.add_argument("--dist", type=float, default=0.5)
    ap.add_argument("--T-hand-cam", required=True, help="4x4 json, hand -> camera")
    ap.add_argument("-o", "--out", default="aim.json")
    a = ap.parse_args()
    h = np.array(a.target); n = np.array(a.normal, float); n[2] = 0.0; n /= np.linalg.norm(n)
    cam_pos = h + n * a.dist
    z_cam = (h - cam_pos); z_cam /= np.linalg.norm(z_cam)           # look at the handle
    up = np.array([0.0, 0.0, 1.0])
    x_cam = np.cross(z_cam, up); x_cam /= np.linalg.norm(x_cam)      # image right, horizontal
    y_cam = np.cross(z_cam, x_cam)                                   # image down
    R_body_cam = np.stack([x_cam, y_cam, z_cam], 1)
    T_body_cam = np.eye(4); T_body_cam[:3, :3] = R_body_cam; T_body_cam[:3, 3] = cam_pos
    T_hand_cam = np.array(json.loads(a.T_hand_cam))
    T_body_hand = T_body_cam @ np.linalg.inv(T_hand_cam)
    xyz = T_body_hand[:3, 3]; q = R.from_matrix(T_body_hand[:3, :3]).as_quat()
    out = {"hand_xyz": xyz.tolist(), "hand_quat_xyzw": q.tolist(), "cam_xyz": cam_pos.tolist(), "target": h.tolist(),
           "dist": a.dist, "hand_rpy_deg": R.from_matrix(T_body_hand[:3, :3]).as_euler("xyz", degrees=True).tolist(),
           "reach_from_shoulder": float(np.linalg.norm(xyz - np.array([0.29, 0, 0])))}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"camera at {np.round(cam_pos, 3).tolist()} looking at {np.round(h, 3).tolist()} from {a.dist:.2f} m")
    print(f"hand pose: xyz {np.round(xyz, 3).tolist()}  rpy {np.round(out['hand_rpy_deg'], 1).tolist()} deg  reach {out['reach_from_shoulder']:.2f} m")
    print("spotctl poseq " + " ".join(f"{v:.4f}" for v in list(xyz) + list(q)))


if __name__ == "__main__":
    main()
