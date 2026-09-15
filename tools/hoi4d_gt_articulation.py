"""Ground-truth 3D articulation for HOI4D records from the OFFICIAL per-frame part poses (objpose/*.json).

For each record (sequence, 0-based frame f) of an articulated category, the moving part and the object body
are identified from the category-level part poses: within the record's action segment, the part whose pose
relative to the body changes most is the moving part. Its relative motion between frame f and the frame of
maximal relative displacement is a screw in the body frame; its rotation axis (revolute) or translation
direction (prismatic) and a point on the axis are mapped into the camera frame of frame f with the body's
pose there. No CAD canonical frames, no camera trajectory needed: every pose is in its own frame's camera
and relative poses are camera-independent.

Runs on the HOI4D volume (CPU): python3 tools/hoi4d_gt_articulation.py --keys hoi4d_test_keys.json \
    --annot /workspace/ext/HOI4D_annotations --out hoi4d_test_gt_articulation.json
Output: {key: {valid, axis_cam, origin_cam, type ("rot"|"trans"), angle_deg, shift_m, moving, body,
frame_g, n_parts, reason}} with axis/origin in OpenCV camera coordinates (metres), unsigned convention.
Sign of the axis: the direction that rotates the part from frame f towards frame g (open/close dependent).
"""
import argparse
import json
import math
import os

import numpy as np

# HOI4D categories with a joint between a body and a moving part (test-split ones). Rigid categories
# (toy car, mug, lamp) have no joint; their records get valid=False with reason "rigid".
ARTICULATED = {"C3": "Laptop", "C4": "StorageFurniture", "C6": "Safe", "C14": "TrashCan"}
BODY_HINTS = ("body", "base", "keyboard", "safebox", "lockerbody")   # label substrings that mark the non-moving part
# (HOI4D labels: Dustbinbase/Dustbincover, Laptopkeyboard/Laptopdisplay, Safebox/Safedoor, Lockerbody/Lockerdrawer/Lockersldingdoor)


def euler_xyz_to_R(x, y, z):
    """scipy Rotation.from_euler('XYZ', [x, y, z]) (intrinsic) = Rx @ Ry @ Rz."""
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def load_parts(path):
    """objpose frame json -> {label: (R, t)} in the frame's camera coordinates."""
    with open(path) as f:
        c = json.load(f)
    lst = c.get("dataList", c.get("objects", []))
    out = {}
    for a in lst:
        r, t = a["rotation"], a["center"]
        out[a["label"]] = (euler_xyz_to_R(r["x"], r["y"], r["z"]), np.array([t["x"], t["y"], t["z"]], float))
    return out


def seq_dir(annot, seq):
    zy, h, c, n, s, s2, t = seq.split("_")
    return os.path.join(annot, zy, h, c, n, s, s2, t)


def rel_pose(body, part):
    """Pose of `part` in the body frame: (Rb^T Rp, Rb^T (tp - tb))."""
    Rb, tb = body; Rp, tp = part
    return Rb.T @ Rp, Rb.T @ (tp - tb)


def rot_angle(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def rot_axis(R):
    """Unit rotation axis of R (Rodrigues), for angles away from 0 and 180."""
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def screw_point(R, t, d):
    """A point on the rotation axis of the rigid motion x -> R x + t (least squares on (I - R) p = t_perp)."""
    t_perp = t - np.dot(t, d) * d
    A = np.eye(3) - R
    p, *_ = np.linalg.lstsq(A, t_perp, rcond=None)
    return p


def action_segment(seq_path, f):
    """(start, end) frame of the action segment containing f, from action/color.json; whole sequence if absent."""
    try:
        with open(os.path.join(seq_path, "action", "color.json")) as fh:
            c = json.load(fh)
        ev = c.get("events", c.get("markResult", {}).get("marks", []))
        dur = float(c.get("info", {}).get("duration", 0) or 0)
        n = 300
        for e in ev:
            st, en = float(e.get("startTime", 0)), float(e.get("endTime", 0))
            if dur > 0:
                a, b = int(st / dur * n), int(en / dur * n)
            else:
                a, b = int(st), int(en)
            if a <= f <= b:
                return max(0, a), min(n - 1, b)
    except Exception:
        pass
    return max(0, f - 30), min(299, f + 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--annot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-rot-deg", type=float, default=5.0)
    ap.add_argument("--min-shift-m", type=float, default=0.02)
    a = ap.parse_args()
    keys = json.load(open(a.keys))["keys"]
    out = {}
    stats = {}
    for key in keys:
        pre, rest = key.split("/")
        seq = pre + "_" + rest.rsplit("_w", 1)[0]
        f = int(rest.rsplit("_f", 1)[1])
        cat = pre.split("_")[2]
        rec = {"valid": False, "seq": seq, "frame": f, "category": cat}
        if cat not in ARTICULATED:
            rec["reason"] = "rigid"; out[key] = rec; stats["rigid"] = stats.get("rigid", 0) + 1; continue
        sp = seq_dir(a.annot, seq)
        pf = os.path.join(sp, "objpose", f"{f}.json")
        if not os.path.exists(pf):
            rec["reason"] = "no objpose"; out[key] = rec; stats["no objpose"] = stats.get("no objpose", 0) + 1; continue
        parts_f = load_parts(pf)
        labels = list(parts_f)
        body = next((l for l in labels if any(h in l.lower() for h in BODY_HINTS)), None)
        if body is None or len(labels) < 2:
            rec.update(reason=f"no body/part split: {labels}"); out[key] = rec; stats["no body"] = stats.get("no body", 0) + 1; continue
        # frame of maximal relative motion of any non-body part within the action segment; if the segment shows
        # (almost) no motion, search the whole sequence (segments are coarse and the part may move later)
        s0, s1 = action_segment(sp, f)
        best = None
        for g in list(range(s0, s1 + 1)) + list(range(0, 300)):
            if best is not None and g < s0 and (best[5] >= a.min_rot_deg or best[6] >= a.min_shift_m):
                break
            if g == f:
                continue
            pg = os.path.join(sp, "objpose", f"{g}.json")
            if not os.path.exists(pg):
                continue
            parts_g = load_parts(pg)
            if body not in parts_g:
                continue
            for lab in labels:
                if lab == body or lab not in parts_g:
                    continue
                Rf, tf = rel_pose(parts_f[body], parts_f[lab]); Rg, tg = rel_pose(parts_g[body], parts_g[lab])
                # motion of the part in the body frame: x_g = M x_f with M = (Rg, tg) (Rf, tf)^-1
                Rm = Rg @ Rf.T; tm = tg - Rm @ tf
                ang = rot_angle(Rm); shift = float(np.linalg.norm(tm))
                score = ang / 90.0 + shift / 0.30
                if best is None or score > best[0]:
                    best = (score, g, lab, Rm, tm, ang, shift)
        if best is None:
            rec["reason"] = "no other frames"; out[key] = rec; stats["no frames"] = stats.get("no frames", 0) + 1; continue
        _, g, lab, Rm, tm, ang, shift = best
        Rb, tb = parts_f[body]
        Rf, tf = rel_pose(parts_f[body], parts_f[lab])   # moving part in the body frame at f
        rec.update(moving=lab, body=body, frame_g=g, angle_deg=round(ang, 2), shift_m=round(shift, 4), n_parts=len(labels))
        if ang >= a.min_rot_deg and (shift < 0.15 or ang > 20):
            d_b = rot_axis(Rm)
            if d_b is None:
                rec["reason"] = "degenerate rotation"; out[key] = rec; continue
            p_b = screw_point(Rm, tm, d_b)
            d_c = Rb @ d_b; p_c = Rb @ p_b + tb
            rec.update(valid=True, type="rot", axis_cam=[float(v) for v in d_c], origin_cam=[float(v) for v in p_c])
            stats["rot"] = stats.get("rot", 0) + 1
        elif shift >= a.min_shift_m:
            d_b = tm / shift
            d_c = Rb @ d_b; p_c = Rb @ tf + tb   # the moving part's centre at f, in the camera
            rec.update(valid=True, type="trans", axis_cam=[float(v) for v in d_c], origin_cam=[float(v) for v in p_c])
            stats["trans"] = stats.get("trans", 0) + 1
        else:
            rec["reason"] = f"too little motion ({ang:.1f} deg, {shift:.3f} m)"; stats["static"] = stats.get("static", 0) + 1
        out[key] = rec
    json.dump(out, open(a.out, "w"))
    print("records", len(keys), "stats", stats)
    for k, r in list(out.items())[:6]:
        print(k, {kk: vv for kk, vv in r.items() if kk not in ("axis_cam", "origin_cam")})


if __name__ == "__main__":
    main()
