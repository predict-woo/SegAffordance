#!/usr/bin/env python3
"""Plan where the BASE should stand before the arm can reach the predicted handle.

  python3 plan_standoff.py snaps/<name>_cloud.pred.json [--standoff 1.10] [--lateral 0.15] -o standoff.json

Input: pc_export.py's *.pred.json (EgoArt prediction in the camera frame, metric, + T_body_cam).
Output: a base goal in the CURRENT body frame (x, y, yaw), which is what the driver's /spot/trajectory
action takes (frame_id must be 'body'; the driver converts to odom itself), plus the same numbers for
`spotctl walkto X Y YAW`.

Geometry (all in the current body frame, x forward, y left, z up):
  handle h, hinge o, axis a (vertical-ish), lever l = h - o (hinge -> handle, along the door face).
  door normal n = horizontal unit vector perpendicular to the door face, signed to point toward the robot.
  base goal = h + n * standoff + lhat * lateral    (lhat = horizontal unit lever direction, i.e. away from the
  hinge, so the door, which swings toward the robot on the hinge side, sweeps past the body, not into it)
  yaw = heading of -n (face the door).
Reach check: with the base at the goal, the handle sits `standoff` ahead of the body origin; the arm reaches
~0.9 m from a shoulder ~0.29 m ahead, so standoff <= ~1.15 m keeps the whole 60 deg arc reachable.
"""
import argparse
import json

import numpy as np

SHOULDER_X, REACH_MAX = 0.29, 0.90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_json")
    ap.add_argument("-o", "--out", default="standoff.json")
    ap.add_argument("--standoff", type=float, default=1.10, help="body origin to handle along the door normal (m)")
    ap.add_argument("--lateral", type=float, default=0.15, help="sideways offset away from the hinge (m)")
    ap.add_argument("--turn", type=float, default=60.0, help="arc to check reach for (deg)")
    a = ap.parse_args()

    J = json.load(open(a.pred_json))
    p = J["preds"][0]   # far frame: the EgoArt record; close-up: the SAM 3 record with the far hinge carried over
    T = np.array(J["T_body_cam"]); Rbc, tbc = T[:3, :3], T[:3, 3]
    h = Rbc @ np.asarray(p["anchor"]) + tbc
    ax = Rbc @ np.asarray(p["axis"]); ax /= np.linalg.norm(ax)
    if p["type"] != "revolute" or not p["origin"]:
        raise SystemExit("standoff planning is written for a revolute (door) prediction")
    o = Rbc @ np.asarray(p["origin"]) + tbc
    lever = h - o
    lhat = lever.copy(); lhat[2] = 0; lhat /= np.linalg.norm(lhat)
    n = np.cross(ax, lever); n[2] = 0; n /= np.linalg.norm(n)
    if np.dot(n, -h) < 0:          # point toward the robot (body origin)
        n = -n
    goal = h + n * a.standoff + lhat * a.lateral
    goal[2] = 0.0
    yaw = float(np.arctan2(-n[1], -n[0]))

    # reach check of the arc from the goal pose: express handle path in the goal's frame
    c, s = np.cos(yaw), np.sin(yaw)
    Rg = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    def to_goal_frame(x):
        return Rg.T @ (np.asarray(x) - goal)
    th = np.radians(np.linspace(0, a.turn, 25))
    def rot(v, d, t):
        return v * np.cos(t) + np.cross(d, v) * np.sin(t) + d * np.dot(d, v) * (1 - np.cos(t))
    path_body_at_goal = np.stack([to_goal_frame(o + rot(lever, ax, t)) for t in th])
    reach = np.linalg.norm(path_body_at_goal - np.array([SHOULDER_X, 0, 0]), axis=1)

    out = {"goal_body_xy_yaw_deg": [float(goal[0]), float(goal[1]), float(np.degrees(yaw))],
           "handle_body": h.tolist(), "hinge_body": o.tolist(), "axis_body": ax.tolist(), "door_normal_body": n.tolist(),
           "standoff_m": a.standoff, "lateral_m": a.lateral,
           "handle_in_goal_frame": to_goal_frame(h).tolist(), "hinge_in_goal_frame": to_goal_frame(o).tolist(),
           "arc_reach_from_shoulder_m": [float(reach.min()), float(reach.max())]}
    json.dump(out, open(a.out, "w"), indent=1)

    f = lambda v: "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"
    print(f"handle (body):      {f(h)}    hinge: {f(o)}    lever {np.linalg.norm(lever):.2f} m")
    print(f"axis (body):        {f(ax)}    door normal toward robot: {f(n)}")
    print(f"base goal (body):   x={goal[0]:+.3f}  y={goal[1]:+.3f}  yaw={np.degrees(yaw):+.1f} deg"
          f"   -> walk {np.linalg.norm(goal[:2]):.2f} m forward-ish, turn {np.degrees(yaw):+.1f} deg")
    print(f"after walking, handle would be at {f(to_goal_frame(h))} in the new body frame; hinge at {f(to_goal_frame(o))}")
    print(f"arc reach from shoulder at goal: {reach.min():.2f}..{reach.max():.2f} m (limit {REACH_MAX})"
          + ("   OK" if reach.max() <= REACH_MAX else "   TOO FAR: reduce --standoff"))
    print(f"door sweep: handle moves toward the robot and toward the hinge side ({'right' if o[1] < h[1] else 'left'}); body is offset {a.lateral:.2f} m the other way")
    print(f"spotctl walkto {goal[0]:.3f} {goal[1]:.3f} {np.degrees(yaw):.1f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
