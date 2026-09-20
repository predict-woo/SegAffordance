#!/usr/bin/env python3
"""rr_egoart: draw the EgoArt web app's outputs into the live Rerun world view (spot_world on spot22), stage by stage.

Runs on the Mac (python with rerun-sdk 0.38, e.g. the pipx venv) as a short subprocess of server.py:

    rr_egoart.py clear                                          remove everything under world/egoart
    rr_egoart.py far      --data far.data.json      --tvb T     far-frame EgoArt prediction (magenta)
    rr_egoart.py standoff --standoff standoff.json  --tvb T     planned base goal (green footprint + arrow)
    rr_egoart.py close    --data close.data.json    --tvb T     close-up recalibrated prediction (cyan)
    rr_egoart.py plan     --traj traj_web.json      --tvb T     hand waypoints about to be executed (orange)

T = 4x4 vision_T_body (JSON) at the time of the stage's snapshot: everything is logged in spot/vision, the frame the live
clouds and the robot model use, so a far prediction stays on the door while the robot walks. Entities live under
world/egoart/<stage>/... and are cleared with `clear` (web app reset / new run).
"""
import argparse
import base64
import json
import os
import time

import numpy as np
import rerun as rr

URL = os.environ.get("RR_URL", "rerun+http://192.168.1.213:9876/proxy")
RECORDING_ID = "spot-world-live"          # must match spot_world.py
WORLD = "spot/vision"
ROOT = "world/egoart"
COL = {"far": [255, 60, 220], "close": [0, 220, 255], "plan": [255, 150, 0], "mask": [255, 230, 0], "goal": [80, 220, 120]}


def tf(T, pts):
    pts = np.atleast_2d(np.asarray(pts, np.float64))
    return pts @ T[:3, :3].T + T[:3, 3]


def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; return [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; return [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; return [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; return [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]


def connect():
    rr.init("spot_world", recording_id=RECORDING_ID)
    rr.connect_grpc(URL)
    rr.set_time("ros", timestamp=time.time())
    rr.log(ROOT, rr.CoordinateFrame(WORLD), static=True)


def log_prediction(stage, data, T_vb):
    """One EgoArt / recalibrated prediction (camera frame in the .data.json) -> world."""
    T_bc = np.asarray(data["T_body_cam"], np.float64)
    T_wc = T_vb @ T_bc
    col = COL[stage]
    base = f"{ROOT}/{stage}"
    rr.log(base, rr.Clear(recursive=True))
    # the snapshot camera: frustum with the photo in it
    K = np.asarray(data["K"], np.float64)
    rr.log(f"{base}/camera", rr.Transform3D(translation=T_wc[:3, 3], quaternion=rot_to_quat(T_wc[:3, :3])))
    rr.log(f"{base}/camera", rr.Pinhole(image_from_camera=K, resolution=[data["w"], data["h"]], camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.12))
    if data.get("img_b64"):
        rr.log(f"{base}/camera/image", rr.EncodedImage(contents=np.frombuffer(base64.b64decode(data["img_b64"]), np.uint8), media_type="image/jpeg"))
    preds = data.get("preds") or []
    if not preds:
        return f"{stage}: camera only (no prediction)"
    p = preds[0]
    pts = np.frombuffer(base64.b64decode(data["xyz_b64"]), np.float32).reshape(-1, 3).astype(np.float64)
    if p.get("mask_idx"):
        m = tf(T_wc, pts[np.asarray(p["mask_idx"], int)])
        rr.log(f"{base}/mask", rr.Points3D(m.astype(np.float32), colors=COL["mask"], radii=0.01, labels=[f"{stage}: segmented handle ({len(m)} pts)"], show_labels=False))
    contact = tf(T_wc, p["anchor"])[0]
    rr.log(f"{base}/contact", rr.Points3D([contact], colors=col, radii=0.03, labels=[f"{stage}: contact point"], show_labels=False))
    traj = tf(T_wc, p["traj"]) if p.get("traj") else None
    if traj is not None and len(traj) > 1:
        rr.log(f"{base}/trajectory", rr.LineStrips3D([traj.astype(np.float32)], colors=col, radii=0.012,
                                                     labels=[f"{stage}: {p['type']} trajectory" + (f", {p['turn_deg']:.0f} deg" if p.get("turn_deg") else f", {p.get('slide_m', 0):.2f} m")], show_labels=False))
        rr.log(f"{base}/trajectory/waypoints", rr.Points3D(traj.astype(np.float32), colors=col, radii=0.02))
    axis = T_wc[:3, :3] @ np.asarray(p["axis"], np.float64); axis /= np.linalg.norm(axis)
    if p["type"] == "revolute" and p.get("origin") is not None:
        hinge = tf(T_wc, p["origin"])[0]
        span = max(0.4, 2.0 * float(np.linalg.norm(contact - hinge)))
        rr.log(f"{base}/hinge", rr.Points3D([hinge], colors=col, radii=0.016, labels=[f"{stage}: hinge"], show_labels=False))
        rr.log(f"{base}/lever", rr.LineStrips3D([np.stack([hinge, contact]).astype(np.float32)], colors=col, radii=0.003))
        rr.log(f"{base}/axis", rr.Arrows3D(origins=[hinge - axis * span / 2], vectors=[axis * span], colors=col, radii=0.006, labels=[f"{stage}: hinge axis"], show_labels=False))
    else:
        rr.log(f"{base}/axis", rr.Arrows3D(origins=[contact], vectors=[axis * 0.4], colors=col, radii=0.006, labels=[f"{stage}: slide axis"], show_labels=False))
    return f"{stage}: {p['type']} contact {np.round(contact, 3).tolist()} (prompt {p.get('prompt', '')!r})"


def log_standoff(so, T_vb):
    base = f"{ROOT}/standoff"
    rr.log(base, rr.Clear(recursive=True))
    x, y, yaw = so["goal_body_xy_yaw_deg"][0], so["goal_body_xy_yaw_deg"][1], np.radians(so["goal_body_xy_yaw_deg"][2])
    c, s = np.cos(yaw), np.sin(yaw)
    T_bg = np.eye(4); T_bg[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; T_bg[:3, 3] = [x, y, 0.0]
    T_wg = T_vb @ T_bg
    col = COL["goal"]
    rr.log(f"{base}/goal", rr.Transform3D(translation=T_wg[:3, 3], quaternion=rot_to_quat(T_wg[:3, :3]), axis_length=0.3))
    rr.log(f"{base}/goal/footprint", rr.Boxes3D(centers=[[0.0, 0.0, 0.25]], half_sizes=[[0.55, 0.25, 0.25]], colors=col,
                                                labels=[f"stand here: {np.hypot(x, y):.2f} m away, standoff {so['standoff_m']:.2f} m"], show_labels=False))
    here = T_vb[:3, 3].copy(); here[2] = T_wg[2, 3] + 0.02
    rr.log(f"{base}/walk", rr.Arrows3D(origins=[here], vectors=[T_wg[:3, 3] + [0, 0, 0.02] - here], colors=col, radii=0.01))
    h = tf(T_vb, so["handle_body"])[0]; n = T_vb[:3, :3] @ np.asarray(so["door_normal_body"])
    rr.log(f"{base}/door_normal", rr.Arrows3D(origins=[h], vectors=[n * 0.3], colors=col, radii=0.004, labels=["door normal"], show_labels=False))
    return f"standoff: goal ({x:.2f}, {y:.2f}, {np.degrees(yaw):.0f} deg) in the far body frame"


def log_plan(tj, T_vb):
    base = f"{ROOT}/plan"
    rr.log(base, rr.Clear(recursive=True))
    col = COL["plan"]
    wps = np.asarray([w["xyz"] for w in tj["waypoints"]], np.float64)
    W = tf(T_vb, wps)
    pre = tf(T_vb, tj["pre_grasp"]["xyz"])[0]
    rr.log(f"{base}/waypoints", rr.LineStrips3D([W.astype(np.float32)], colors=col, radii=0.012,
                                                 labels=[f"hand path to execute: {tj['type']}, {tj['handle']} handle" + (f", {tj['turn_deg']:.0f} deg" if tj.get("turn_deg") else "")], show_labels=False))
    rr.log(f"{base}/waypoints/points", rr.Points3D(W.astype(np.float32), colors=col, radii=0.018))
    rr.log(f"{base}/pre_grasp", rr.LineStrips3D([np.stack([pre, W[0]]).astype(np.float32)], colors=col, radii=0.003, labels=["approach"], show_labels=False))
    rr.log(f"{base}/pre_grasp/point", rr.Points3D([pre], colors=col, radii=0.012))
    # hand x axis (approach direction) at a few waypoints, from the quaternions
    idx = sorted(set(list(range(0, len(wps), max(1, len(wps) // 6))) + [len(wps) - 1]))
    dirs = []
    for k in idx:
        qx, qy, qz, qw = tj["waypoints"][k]["quat_xyzw"]
        xb = np.array([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw)])   # first column of R(q)
        dirs.append(T_vb[:3, :3] @ xb * 0.06)
    rr.log(f"{base}/hand_x", rr.Arrows3D(origins=W[idx], vectors=np.asarray(dirs), colors=[255, 60, 60], radii=0.002, labels=["hand x (approach direction)"] * len(idx), show_labels=False))
    if tj.get("origin_body") is not None:
        o = tf(T_vb, tj["origin_body"])[0]; ax = T_vb[:3, :3] @ np.asarray(tj["axis_body"])
        rr.log(f"{base}/axis", rr.Arrows3D(origins=[o - ax * 0.4], vectors=[ax * 0.8], colors=col, radii=0.004))
    return f"plan: {len(wps)} waypoints"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["clear", "far", "close", "standoff", "plan"])
    ap.add_argument("--data"); ap.add_argument("--standoff"); ap.add_argument("--traj")
    ap.add_argument("--tvb", help="4x4 vision_T_body as JSON (identity if omitted)")
    a = ap.parse_args()
    T_vb = np.asarray(json.loads(a.tvb), np.float64) if a.tvb else np.eye(4)
    connect()
    if a.stage == "clear":
        rr.log(ROOT, rr.Clear(recursive=True)); msg = "cleared world/egoart"
    elif a.stage in ("far", "close"):
        msg = log_prediction(a.stage, json.load(open(a.data)), T_vb)
    elif a.stage == "standoff":
        msg = log_standoff(json.load(open(a.standoff)), T_vb)
    else:
        msg = log_plan(json.load(open(a.traj)), T_vb)
    rr.disconnect()          # flushes
    print(msg)


if __name__ == "__main__":
    main()
