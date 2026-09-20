#!/usr/bin/env python3
"""Backproject a spotctl snapshot (rgb jpg + depth png + json with K / T_body_cam) into a self-contained
three.js viewer HTML, optionally with model predictions (tools/predict_image.py --dump records) overlaid.

Points are embedded as base64 Float32 (xyz, optical camera frame: x right, y down, z forward) + Uint8 rgb,
so the file opens from disk with no server.

Predictions are scale-free in depth, so each one is rescaled to the cloud: s = measured depth at the
predicted contact pixel (median of valid depth in a small window) / predicted anchor depth. Anchor and hinge
are multiplied by s; the axis direction is a unit vector and is left alone.

The trajectory is NOT the model's decoded arc (its length is a 2D-training quantity). It is re-decoded from
the predicted screw: revolute = rotate the contact point about the axis through the hinge by --turn-deg
(default 60 deg = pi/3, right-hand rule about the predicted axis sign); prismatic = slide --slide-m along
the axis. Only the models listed in --models are overlaid (default: dense, i.e. EgoArt).

  python3 pc_export.py snaps/hand_X.jpg [--preds preds.jsonl] [-o out.html]
"""
import argparse
import base64
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def rotate(v, d, a):
    """Rodrigues: rotate vector v about unit axis d by angle a (right-hand rule)."""
    return v * np.cos(a) + np.cross(d, v) * np.sin(a) + d * np.dot(d, v) * (1.0 - np.cos(a))


def measured_depth(depth_m, u, v, r=12):
    H, W = depth_m.shape
    x, y = int(round(u * W)), int(round(v * H))
    win = depth_m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    vals = win[win > 0]
    return float(np.median(vals)) if vals.size else None


def door_normal_body(pts_cam, Rbc, band=0.06):
    """Horizontal unit normal (body frame) of the dominant fronto-parallel plane in the cloud, pointing at the robot."""
    z = pts_cam[:, 2]
    zm = np.median(z)
    sel = pts_cam[np.abs(z - zm) < band]
    c = sel.mean(0)
    _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
    n = Rbc @ vt[-1]
    n[2] = 0.0; n /= np.linalg.norm(n)
    if n[0] > 0:            # door is in front of the robot; its normal toward the robot points -x
        n = -n
    return n, int(len(sel))


def recalibrate(pred_new, old_json, T_new, pts_cam):
    """Carry the FAR frame's hinge/axis over to the CLOSE frame using the close frame's handle centroid.

    translation: the new anchor (mask centroid at measured depth) IS the handle.
    rotation:    yaw between the two body frames = angle between the old door normal (from old hinge/axis/handle)
                 and the door normal fitted to the new cloud.  Lever (hinge - handle) and axis are rotated by it.
    """
    old = ([q for q in old_json["preds"] if q["model"] == pred_new["model"]] or [q for q in old_json["preds"] if q["model"] == "dense"])[0]
    To = np.array(old_json["T_body_cam"]); Ro, to = To[:3, :3], To[:3, 3]
    Rn, tn = T_new[:3, :3], T_new[:3, 3]
    h_old = Ro @ np.asarray(old["anchor"]) + to
    a_old = Ro @ np.asarray(old["axis"]); a_old /= np.linalg.norm(a_old)
    revolute = old["type"] == "revolute" and old.get("origin") is not None
    if revolute:
        o_old = Ro @ np.asarray(old["origin"]) + to
        lever_old = h_old - o_old
        n_old = np.cross(a_old, lever_old); n_old[2] = 0; n_old /= np.linalg.norm(n_old)   # door normal from the hinge geometry
    else:
        o_old, lever_old = None, None
        n_old = a_old.copy(); n_old[2] = 0; n_old /= np.linalg.norm(n_old)                   # drawer: the slide axis IS the front normal
    if n_old[0] > 0:
        n_old = -n_old
    n_new, n_pts = door_normal_body(pts_cam, Rn)
    dpsi = float((np.arctan2(n_new[1], n_new[0]) - np.arctan2(n_old[1], n_old[0]) + np.pi) % (2 * np.pi) - np.pi)   # wrap to [-pi, pi)
    c, s = np.cos(dpsi), np.sin(dpsi)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    h_new = Rn @ np.asarray(pred_new["anchor"]) + tn
    a_new = Rz @ a_old
    body2cam = lambda x: Rn.T @ (np.asarray(x) - tn)
    pred_new["axis"] = (Rn.T @ a_new).tolist()
    pred_new["type"] = "revolute" if revolute else "prismatic"
    pred_new["recalibrated_from"] = old_json["snapshot"]
    rc = {"yaw_deg": float(np.degrees(dpsi)), "door_normal_new_body": n_new.tolist(), "door_normal_old_body": n_old.tolist(),
          "plane_points": n_pts, "handle_body": h_new.tolist(), "axis_body": a_new.tolist(),
          "own_hinge_body": (Rn @ np.asarray(pred_new["own_origin"]) + tn).tolist() if pred_new.get("own_origin") else None}
    if revolute:
        o_new = h_new - Rz @ lever_old
        pred_new["origin"] = body2cam(o_new).tolist()
        rc.update({"hinge_body": o_new.tolist(), "lever_m": float(np.linalg.norm(lever_old))})
    else:
        pred_new["origin"] = None
        # a drawer pulls along the front normal: snap the carried axis to the fitted normal (toward the robot)
        pred_new["axis"] = (Rn.T @ n_new).tolist()
        rc.update({"hinge_body": None, "lever_m": None, "axis_body": n_new.tolist()})
    pred_new["recalib"] = rc
    return pred_new


def mask_orientation(m):
    """Principal axis of a binary mask: (angle_deg from the image x axis in [0, 180), aspect = major/minor extent)."""
    my, mx = np.nonzero(m)
    if mx.size < 5:
        return None, None
    pts = np.stack([mx, my], 1).astype(np.float64); pts -= pts.mean(0)
    cov = pts.T @ pts / len(pts)
    w, v = np.linalg.eigh(cov)
    major = v[:, 1]
    ang = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    aspect = float(np.sqrt(max(w[1], 1e-9) / max(w[0], 1e-9)))
    return ang, aspect


def preds_from_masks(npz_path, depth_m, xs, ys, K, T_body_cam, expect_body=None, model="sam3", grasp_bias_m=0.0):
    z = np.load(npz_path, allow_pickle=True)
    masks, scores = z["masks"].astype(bool), z["scores"].astype(float)
    if len(scores) == 0:
        return [], "no instances"
    H, W = depth_m.shape
    cents = []
    for m in masks:
        my, mx = np.nonzero(m)
        cents.append(((mx.mean() + 0.5) / W, (my.mean() + 0.5) / H) if mx.size else (np.nan, np.nan))
    cents = np.array(cents)
    how = "best score"
    idx = 0
    if expect_body is not None and T_body_cam is not None:
        Tc = np.linalg.inv(np.asarray(T_body_cam)); pc = Tc[:3, :3] @ np.asarray(expect_body) + Tc[:3, 3]
        if pc[2] > 0:
            eu, ev = (K[0, 0] * pc[0] / pc[2] + K[0, 2]) / W, (K[1, 1] * pc[1] / pc[2] + K[1, 2]) / H
            d = np.hypot(cents[:, 0] - eu, cents[:, 1] - ev)
            idx = int(np.nanargmin(d)); how = f"nearest to expected handle at uv ({eu:.3f},{ev:.3f}), {d[idx]:.3f} away"
    m = masks[idx]
    u, v = cents[idx]
    z_meas = measured_depth(depth_m, u, v) or measured_depth(depth_m, u, v, r=30)
    if not z_meas:
        return [], "no depth at the mask centroid"
    # grasp bias: place the contact grasp_bias_m closer to the camera along the pixel ray (the mask centroid sits ON
    # the bar surface; the gripper closes better with a little stand-off)
    z_use = max(z_meas - grasp_bias_m, 0.05)
    anchor = np.array([(u * W - K[0, 2]) / K[0, 0] * z_use, (v * H - K[1, 2]) / K[1, 1] * z_use, z_use])
    ang, aspect = mask_orientation(m)
    # bar orientation from the mask's principal axis (image x is horizontal at the aim pose): <45 deg from x = horizontal bar
    orient = None if ang is None else ("horizontal" if min(ang, 180 - ang) < 45 else "vertical")
    pred = {"own_origin": None, "model": model, "prompt": str(z["prompt"]) if "prompt" in z else "", "type": "revolute", "p_rev": None,   # None, not NaN: NaN is not valid JSON for browsers
            "point_uv": [float(u), float(v)], "model_uv": [float(u), float(v)],
            "contact_from": f"sam3 instance #{idx}/{len(scores)} score {scores[idx]:.2f} ({how})",
            "z_pred": float(z_meas), "z_meas": z_meas, "scale": 1.0, "grasp_bias_m": float(grasp_bias_m),
            "anchor": anchor.tolist(), "axis": [0.0, -1.0, 0.0], "origin": None,
            "handle_orient": orient, "handle_angle_deg": ang, "handle_aspect": aspect,
            "traj": [], "turn_deg": None, "slide_m": None, "mask_idx": np.flatnonzero(m[ys, xs]).tolist(),
            "sam3": {"scores": scores.tolist(), "centroids": cents.tolist(), "picked": idx}}
    return [pred], how + (f"; bar {orient} ({ang:.0f} deg, aspect {aspect:.1f})" if orient else "") + (f"; grasp bias {grasp_bias_m * 100:.0f} cm toward the camera" if grasp_bias_m else "")


def load_preds(path, image_stem, depth_m, xs, ys, K_px, models, turn_deg, slide_m, n_steps=24):
    """One entry per (model) record of this frame, rescaled to the cloud. xs/ys = pixel coords of cloud points."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        mask_utils = None
    out = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if image_stem not in os.path.basename(r["image"]) or r["model"] not in models:
            continue
        t = np.asarray(r["type_logits"], np.float64)
        p_rev = float(np.exp(t[1]) / np.exp(t).sum())
        revolute = bool(np.argmax(t) == 1)
        p3d = np.asarray(r["point_3d"], np.float64)
        H, W = depth_m.shape
        K = np.asarray(K_px, np.float64)
        # part mask at native resolution; the interaction point is its centroid (still our prediction, just
        # the mask head's instead of the point head's), so the 3D contact sits on the detected handle
        mask_idx, cen_uv, n_comp_dropped = [], None, 0
        if mask_utils is not None and r.get("mask_rle"):
            rle = dict(r["mask_rle"]); rle["counts"] = rle["counts"].encode("ascii")
            m = cv2.resize(mask_utils.decode(rle).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            # the mask may cover several handles; keep the connected component nearest the point head, which is
            # the model's own disambiguation of the prompt, and take THAT component's centroid
            n_cc, lab, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            if n_cc > 1:
                pu, pv = r["point_uv"][0] * W, r["point_uv"][1] * H
                best = 1 + int(np.argmin([np.hypot(cents[i][0] - pu, cents[i][1] - pv) for i in range(1, n_cc)]))
                m = lab == best
                n_comp_dropped = n_cc - 2
            else:
                n_comp_dropped = 0
            mask_idx = np.flatnonzero(m[ys, xs]).tolist()
            my, mx = np.nonzero(m)
            if mx.size:
                cen_uv = (float((mx.mean() + 0.5) / W), float((my.mean() + 0.5) / H))
        model_uv = (float(r["point_uv"][0]), float(r["point_uv"][1]))
        u, v = cen_uv or model_uv
        z_meas = measured_depth(depth_m, u, v) or measured_depth(depth_m, u, v, r=30)
        s = z_meas / p3d[2] if (z_meas and p3d[2] > 0) else 1.0
        if z_meas:   # metric contact point: centroid pixel backprojected at the measured depth
            anchor = np.array([(u * W - K[0, 2]) / K[0, 0] * z_meas, (v * H - K[1, 2]) / K[1, 1] * z_meas, z_meas])
        else:
            anchor = p3d * s
        d = np.asarray(r["motion"], np.float64); d /= max(np.linalg.norm(d), 1e-9)
        origin = np.asarray(r["origin"], np.float64) * s if (revolute and r.get("origin") is not None) else None
        if revolute and origin is not None:
            th = np.linspace(0.0, np.radians(turn_deg), n_steps)
            lever = anchor - origin
            traj = np.stack([origin + rotate(lever, d, a) for a in th])
        else:
            traj = np.stack([anchor + d * (slide_m * k / (n_steps - 1)) for k in range(n_steps)])
        out.append({
            "own_origin": origin.tolist() if origin is not None else None,
            "model": r["model"], "prompt": r["prompt"], "type": "revolute" if revolute else "prismatic", "p_rev": p_rev,
            "point_uv": [u, v], "model_uv": list(model_uv), "contact_from": (f"mask centroid ({n_comp_dropped} other component{'s' if n_comp_dropped != 1 else ''} dropped)" if n_comp_dropped else "mask centroid") if cen_uv else "point head", "z_pred": float(p3d[2]), "z_meas": z_meas, "scale": float(s),
            "anchor": anchor.tolist(), "axis": d.tolist(), "origin": origin.tolist() if origin is not None else None,
            "traj": traj.tolist(), "turn_deg": turn_deg if revolute else None, "slide_m": None if revolute else slide_m, "mask_idx": mask_idx,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rgb")
    ap.add_argument("-o", "--out")
    ap.add_argument("--preds", help="predict_image.py --dump jsonl; records whose image matches this frame are overlaid")
    ap.add_argument("--max-depth-m", type=float, default=4.0)
    ap.add_argument("--models", default="dense", help="comma-separated model names to overlay (default: dense = EgoArt)")
    ap.add_argument("--turn-deg", type=float, default=80.0, help="revolute trajectory sweep (default 80 deg)")
    ap.add_argument("--slide-m", type=float, default=0.3, help="prismatic trajectory length")
    ap.add_argument("--masks-npz", help="external segmentation (sam3_client.py output) instead of --preds: instance masks + scores")
    ap.add_argument("--expect-body", type=float, nargs=3, help="expected handle position in the body frame; picks the instance nearest its projection")
    ap.add_argument("--grasp-bias-m", type=float, default=0.0, help="move the external-mask contact this much closer to the camera (m)")
    ap.add_argument("--recalib", help="FAR frame's *.pred.json: carry its hinge/axis over to this frame via the handle centroid + door-plane yaw")
    a = ap.parse_args()
    stem = os.path.splitext(a.rgb)[0]
    meta = json.load(open(stem + ".json"))
    rgb = cv2.imread(a.rgb, cv2.IMREAD_COLOR)[:, :, ::-1]
    depth = cv2.imread(stem + "_depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    H, W = depth.shape
    K = np.array(meta["K"], np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v = (depth > 0) & (depth < a.max_depth_m)
    ys, xs = np.nonzero(v)
    z = depth[ys, xs]
    pts = np.stack([(xs + 0.5 - cx) / fx * z, (ys + 0.5 - cy) / fy * z, z], 1).astype(np.float32)
    col = rgb[ys, xs].astype(np.uint8)
    T = np.array(meta["T_body_cam"], np.float64) if "T_body_cam" in meta else None
    preds = load_preds(a.preds, os.path.basename(stem), depth, xs, ys, K, set(a.models.split(',')), a.turn_deg, a.slide_m) if a.preds else []
    if a.masks_npz:
        preds, how = preds_from_masks(a.masks_npz, depth, xs, ys, K, T, a.expect_body, grasp_bias_m=a.grasp_bias_m)
        print(f"external masks: {how}" if preds else f"external masks: FAILED ({how})")
    if a.recalib and preds:
        oldj = json.load(open(a.recalib))
        for pr in preds:
            recalibrate(pr, oldj, T, pts.astype(np.float64))
            anc, ax = np.asarray(pr["anchor"]), np.asarray(pr["axis"])
            if pr["type"] == "revolute":
                th = np.linspace(0.0, np.radians(a.turn_deg), 24)
                org = np.asarray(pr["origin"])
                pr["traj"] = np.stack([org + rotate(anc - org, ax, t_) for t_ in th]).tolist()
                pr["turn_deg"], pr["slide_m"] = a.turn_deg, None
            else:
                pr["traj"] = np.stack([anc + ax * (a.slide_m * k / 23) for k in range(24)]).tolist()
                pr["turn_deg"], pr["slide_m"] = None, a.slide_m
    data = {
        "n": int(len(pts)), "w": W, "h": H, "valid_frac": float(v.mean()),
        "z_med": float(np.median(z)), "z_min": float(z.min()), "z_max": float(z.max()),
        "K": K.tolist(), "T_body_cam": T.tolist() if T is not None else None,
        "stamp": meta.get("stamp"), "name": os.path.basename(stem), "preds": preds,
        "xyz_b64": base64.b64encode(pts.tobytes()).decode(),
        "rgb_b64": base64.b64encode(col.tobytes()).decode(),
        "img_b64": base64.b64encode(open(a.rgb, "rb").read()).decode(),
    }
    tpl = open(os.path.join(HERE, "pc_viewer.html")).read()
    out = a.out or stem + "_cloud.html"
    open(out, "w").write(tpl.replace("/*__DATA__*/null", json.dumps(data)))
    json.dump(data, open(os.path.splitext(out)[0] + ".data.json", "w"))   # same payload for the webapp
    json.dump({"snapshot": os.path.basename(stem), "K": K.tolist(), "T_body_cam": data["T_body_cam"],
               "preds": [{k: v for k, v in p.items() if k != "mask_idx"} for p in preds]},
              open(os.path.splitext(out)[0] + ".pred.json", "w"), indent=1)
    print(f"{out}  {len(pts)} points, valid {100 * v.mean():.1f}%, depth {z.min():.2f}..{z.max():.2f} m (median {np.median(z):.2f})")
    for p in preds:
        prev = f"{p['p_rev']:.2f}" if p.get('p_rev') is not None else "n/a"
        print(f"  {p['model']:10s} {p['type']:9s} p_rev={prev}  z_pred={p['z_pred']:.2f}  z_meas={p['z_meas']}  scale={p['scale']:.2f}  "
              f"contact={p['contact_from']} uv=({p['point_uv'][0]:.3f},{p['point_uv'][1]:.3f})  mask pts={len(p['mask_idx'])}  origin={p['origin'] and [round(x, 2) for x in p['origin']]}")
        if p.get("recalib"):
            rc = p["recalib"]
            print(f"    recalibrated from {p['recalibrated_from']}: yaw between frames {rc['yaw_deg']:+.1f} deg (door plane from {rc['plane_points']} pts)")
            hinge = f"hinge {np.round(rc['hinge_body'], 3).tolist()}  lever {rc['lever_m']:.2f} m" if rc.get("hinge_body") else "prismatic: no hinge, axis = fitted front normal"
            print(f"    body frame now: handle {np.round(rc['handle_body'], 3).tolist()}  {hinge}  axis {np.round(rc['axis_body'], 3).tolist()}  type {p['type']}")
            print(f"    this frame's own hinge (for comparison): {rc['own_hinge_body'] and np.round(rc['own_hinge_body'], 3).tolist()}")


if __name__ == "__main__":
    main()
