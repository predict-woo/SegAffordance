#!/usr/bin/env python3
"""Turn an EgoArt prediction (pc_export.py's *.pred.json, camera frame, metric) into a body-frame gripper
trajectory for spotd's `traj` command.

  python3 plan_traj.py snaps/<name>_cloud.pred.json [--turn 60] [--steps 40] [--start-x 0.75] [--real] -o traj.json

Frames: prediction is in the hand colour camera's optical frame at capture time (x right, y down, z forward);
T_body_cam from the snapshot maps it into spot/body (x forward, y left, z up). The robot has not walked
since the capture, so the cabinet is fixed in the body frame.

Gripper: hand x points into the door (fitted door normal, else camera -> contact). --handle vertical (default)
rolls the hand 90 deg so the jaws close horizontally around a vertical bar; --handle horizontal keeps roll 0 so
they close vertically around a horizontal bar. Along the arc the hand rotates with the door (R_k = Rot(axis,
theta_k) R_0), as a hand holding the handle would.

Default is an IN-THE-AIR rehearsal: the whole arc is translated so its first point sits at --start-x in
front of the body (same y / z as predicted, clamped), far from the real handle; --real keeps the true
position (only if it is within reach).
"""
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation as R

SHOULDER = np.array([0.29, 0.0, 0.0])   # arm base in body frame, roughly
REACH_MAX = 0.90                         # conservative, metres from the shoulder


def rodrigues(d, a):
    return R.from_rotvec(np.asarray(d) * a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_json")
    ap.add_argument("-o", "--out", default="traj.json")
    ap.add_argument("--turn", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--start-x", type=float, default=0.75, help="rehearsal: body-frame x of the first waypoint")
    ap.add_argument("--z-clamp", type=float, nargs=2, default=(-0.10, 0.45))
    ap.add_argument("--approach", type=float, default=0.10, help="pre-grasp stand-off along -hand_x (m)")
    ap.add_argument("--real", action="store_true", help="no translation: execute at the predicted location")
    ap.add_argument("--handle", choices=["vertical", "horizontal"], default="vertical",
                    help="bar orientation: vertical -> roll 90 (jaws close horizontally); horizontal -> roll 0 (jaws close vertically)")
    a = ap.parse_args()

    J = json.load(open(a.pred_json))
    p = J["preds"][0]   # far frame: the EgoArt record; close-up: the SAM 3 record with the far hinge carried over
    T = np.array(J["T_body_cam"]); Rbc, tbc = T[:3, :3], T[:3, 3]
    cam2body = lambda x: Rbc @ np.asarray(x) + tbc
    anchor = cam2body(p["anchor"]); axis = Rbc @ np.asarray(p["axis"]); axis /= np.linalg.norm(axis)
    origin = cam2body(p["origin"]) if p["type"] == "revolute" and p["origin"] else None

    # hand orientation at grasp: x toward the door (horizontal camera->contact), rolled 90 deg (jaws horizontal)
    # hand x = into the door. Prefer the door normal (fitted to the cloud by pc_export --recalib, stored in
    # recalib.door_normal_new_body, pointing toward the robot); fall back to the camera -> contact direction.
    rc = p.get("recalib") or {}
    if rc.get("door_normal_new_body"):
        look = -np.asarray(rc["door_normal_new_body"], float); look[2] = 0.0; look /= np.linalg.norm(look)
        yaw_src = "door normal"
    else:
        look = anchor - tbc; look[2] = 0.0; look /= np.linalg.norm(look)
        yaw_src = "camera->contact"
    yaw = np.arctan2(look[1], look[0])
    roll = np.pi / 2 if a.handle == "vertical" else 0.0
    R0 = R.from_euler("z", yaw) * R.from_euler("x", roll)

    th = np.radians(np.linspace(0.0, a.turn, a.steps))
    if origin is not None:
        lever = anchor - origin
        pos = np.stack([origin + rodrigues(axis, t).apply(lever) for t in th])
        rots = [rodrigues(axis, t) * R0 for t in th]
    else:
        pos = np.stack([anchor + axis * (0.3 * k / (a.steps - 1)) for k in range(a.steps)])
        rots = [R0] * a.steps

    real_start = pos[0].copy()
    offset = np.zeros(3)
    if not a.real:
        target0 = np.array([a.start_x, pos[0, 1], float(np.clip(pos[0, 2], *a.z_clamp))])
        offset = target0 - pos[0]
    pos = pos + offset

    hand_x0 = R0.apply([1.0, 0.0, 0.0])
    pre = pos[0] - hand_x0 * a.approach
    reach = np.linalg.norm(pos - SHOULDER, axis=1)
    wps = [{"xyz": pos[k].tolist(), "quat_xyzw": rots[k].as_quat().tolist()} for k in range(a.steps)]
    out = {
        "source": a.pred_json, "approach_m": a.approach, "handle": a.handle, "prompt": p["prompt"], "type": p["type"], "turn_deg": a.turn, "rehearsal": not a.real,
        "offset_body": offset.tolist(), "axis_body": axis.tolist(), "origin_body": (origin + offset).tolist() if origin is not None else None,
        "pre_grasp": {"xyz": pre.tolist(), "quat_xyzw": R0.as_quat().tolist()},
        "waypoints": wps,
    }
    json.dump(out, open(a.out, "w"), indent=1)

    f = lambda v: "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"
    print(f"prompt: {p['prompt']}   type: {p['type']}   turn: {a.turn:.0f} deg over {a.steps} waypoints")
    print(f"REAL contact in body frame:  {f(real_start)}   reach from shoulder {np.linalg.norm(real_start - SHOULDER):.2f} m"
          f"  -> {'in reach' if np.linalg.norm(real_start - SHOULDER) < REACH_MAX else 'OUT OF REACH (needs walking)'}")
    print(f"axis (body):    {f(axis)}   (z up = {axis[2]:+.2f}; sign gives opening direction, right-hand rule)")
    if origin is not None:
        print(f"hinge (body, {'shifted' if not a.real else 'real'}): {f(origin + offset)}   lever {np.linalg.norm(anchor - origin):.2f} m")
    print(f"mode: {'REHEARSAL, arc shifted by ' + f(offset) if not a.real else 'REAL position'}")
    print(f"hand at grasp: yaw {np.degrees(yaw):+.1f} deg ({yaw_src}), roll {np.degrees(roll):+.0f} deg ({a.handle} handle, jaws close {'horizontally' if a.handle == 'vertical' else 'vertically'}); RPY = {np.round(R0.as_euler('xyz', degrees=True), 1).tolist()}")
    print(f"pre-grasp:     {f(pre)}")
    print(f"start:         {f(pos[0])}")
    print(f"end:           {f(pos[-1])}   end hand RPY = {np.round(rots[-1].as_euler('xyz', degrees=True), 1).tolist()}")
    print(f"arc length:    {np.linalg.norm(np.diff(pos, axis=0), axis=1).sum():.2f} m   step {np.linalg.norm(np.diff(pos, axis=0), axis=1).mean() * 100:.1f} cm")
    print(f"reach range:   {reach.min():.2f}..{reach.max():.2f} m from shoulder (limit {REACH_MAX})  x range {pos[:, 0].min():.2f}..{pos[:, 0].max():.2f}"
          f"  y {pos[:, 1].min():+.2f}..{pos[:, 1].max():+.2f}  z {pos[:, 2].min():+.2f}..{pos[:, 2].max():+.2f}")
    if reach.max() > REACH_MAX:
        print("WARNING: some waypoints exceed the reach limit; the driver may reject them or Spot may shift its stance")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
