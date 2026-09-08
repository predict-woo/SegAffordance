"""Build SF3D-format LMDBs from ARCTIC (collaborator GT-trajectory package +
ARCTIC raw_seqs / meta / egocentric images).

One record per single articulation STROKE (monotone run of the 1-DoF angle,
>= --min-deg and >= --min-frames long) inside a "use" sequence:
  image        the ego frame at the stroke start (2800x2000 or portrait), resized to --size
  mask         the MOVING part ("top" in ARCTIC's templates) rendered with the GT object
               pose at the stroke start: z-buffer of the dense top.obj + bottom.obj, pixels
               whose nearest surface is the top part; hands are NOT rendered (no MANO models)
  depth        z-buffer depth of the object only (mm; 0 elsewhere) — best available, no scene depth
  trajectory   middle-knuckle (OpenPose-21 joint 9) of the hand nearest the moving part at the
               stroke start, over the stroke, re-anchored from the package's onset camera into the
               stroke-start ego camera (world-fixed), 3D metres + distorted 2D pixels
  motion_info  REAL articulation GT: revolute axis = object rotation applied to canonical -z,
               origin = object translation, both in the stroke-start camera (+ 2D origin)
  description  template "<open|close> the <object> <part>" (open = |angle| increasing)
Keys "<subject>_<object>/<seq>_s<k>_f<frame>" (scene prefix = subject+object instance,
so split_dataset_by_scene keeps an object's sequences of one subject together).
Usage (dev pod): python tools/arctic_process_2d.py --out /workspace/datasets/arctic_processed_2d [--seqs s01_box_use_01] [--workers 8] [--viz N]
"""
import argparse, csv, json, os, pickle, re, sys, collections
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi4d_process_2d import thin_coords, encode_frame, MAX_START_TO_MASK_PX, MAX_FRAME_JUMP_PX
ROOT = "/workspace/datasets/arctic"; PKG = "/workspace/datasets/arctic_gt_package"
ANCHOR_JOINT = 9  # OpenPose-21 middle knuckle (package joint order)
PART_WORD = {"lid (hinge)": "lid", "screen (hinge)": "screen", "head (hinge)": "head", "lever/lid (hinge)": "lever",
             "lever (hinge)": "lever", "cover (hinge)": "cover", "flip phone (hinge)": "phone", "flip cap (hinge)": "cap",
             "door (hinge)": "door", "blades (pivot)": "blades"}
OBJ_WORD = {"espressomachine": "espresso machine", "capsulemachine": "capsule machine", "waffleiron": "waffle iron"}

def read_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "): V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "): F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V, np.float64), np.array(F, np.int64)

def load_mesh(obj):
    """ARCTIC's dense PER-PART meshes (top.obj = moving part, bottom.obj = base; mm -> m), concatenated
    with unambiguous per-vertex/per-face labels. (The simplified mesh.obj + a vertex labelling produced
    faces straddling the hinge whose vertices rotate differently -> triangles smeared across the base.)"""
    d = f"{ROOT}/meta/object_vtemplates/{obj}"
    Vt, Ft = read_obj(f"{d}/top.obj"); Vb, Fb = read_obj(f"{d}/bottom.obj")
    V = np.concatenate([Vt, Vb]) / 1000.0; F = np.concatenate([Ft, Fb + len(Vt)])
    parts = np.concatenate([np.ones(len(Vt), np.int64), np.zeros(len(Vb), np.int64)])
    face_top = np.concatenate([np.ones(len(Ft), bool), np.zeros(len(Fb), bool)])
    return V, F, parts, face_top

def pose_object(V, parts, row):
    """ARCTIC 7-D pose row [angle, rotvec(3), trans_mm(3)] -> world-space vertices (m) + axis/origin."""
    ang, rv, tr = float(row[0]), np.asarray(row[1:4], np.float64), np.asarray(row[4:7], np.float64) / 1000.0
    # The moving part rotates by -angle about canonical +z (verified 2026-09-08 against the collaborator's
    # fingertip-to-moving-part distances: -angle reproduces them to ~0.3 cm, +angle is off by up to 14 cm).
    Rz = cv2.Rodrigues(np.array([0, 0, -ang]))[0]; Rg = cv2.Rodrigues(rv)[0]
    Vp = V.copy(); Vp[parts == 1] = Vp[parts == 1] @ Rz.T
    axis = Rg @ np.array([0.0, 0.0, -1.0])  # so that increasing angle = right-hand rotation about `axis`
    return Vp @ Rg.T + tr, axis, tr

def project(Xc, K, dist):
    """camera-space points (N,3) -> distorted pixels (N,2); points behind the camera -> nan"""
    out = np.full((len(Xc), 2), np.nan)
    ok = Xc[:, 2] > 1e-3
    if ok.any():
        uv, _ = cv2.projectPoints(Xc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
        out[ok] = uv.reshape(-1, 2)
    return out

def rasterize(uv, z, F, face_top, W, H, scale, bias=0.012):
    """z-buffer at (W*scale, H*scale) with SEPARATE buffers for the moving (top) and fixed (bottom)
    parts; the top part is visible wherever it is not occluded by the bottom by more than `bias`
    metres (the template fits let a closed lid penetrate the base by a few mm, which otherwise
    hands the whole lid to the base). Returns top-mask and depth (m; 0 = nothing)."""
    w, h = int(round(W * scale)), int(round(H * scale)); zt = np.full((h, w), np.inf); zb = np.full((h, w), np.inf)
    p = uv * scale
    for fi in range(len(F)):
        a, b, c = F[fi]
        if not (np.isfinite(p[a]).all() and np.isfinite(p[b]).all() and np.isfinite(p[c]).all()): continue
        tri = p[[a, b, c]]; x0, y0 = np.floor(tri.min(0)).astype(int); x1, y1 = np.ceil(tri.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0: continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (xa, ya), (xb, yb), (xc, yc) = tri; det = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
        if abs(det) < 1e-9: continue
        l1 = ((xb - xs) * (yc - ys) - (xc - xs) * (yb - ys)) / det; l2 = ((xc - xs) * (ya - ys) - (xa - xs) * (yc - ys)) / det; l3 = 1 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not inside.any(): continue
        zz = l1 * z[a] + l2 * z[b] + l3 * z[c]; buf = zt if face_top[fi] else zb
        sub = buf[y0:y1 + 1, x0:x1 + 1]; upd = inside & (zz < sub); sub[upd] = zz[upd]
    top = np.isfinite(zt) & (zt <= zb + bias)
    depth = np.minimum(zt, zb); depth[~np.isfinite(depth)] = 0
    return top, depth

def strokes_of(arti_rad, min_deg, min_frames):
    a = np.degrees(arti_rad.astype(float)); k = 5; s = np.convolve(a, np.ones(k) / k, mode="same"); v = np.gradient(s)
    sign = np.sign(v); sign[np.abs(v) < 0.15] = 0; out = []; i = 0
    while i < len(a):
        if sign[i] == 0: i += 1; continue
        j = i
        while j + 1 < len(a) and (sign[j + 1] == sign[i] or sign[j + 1] == 0): j += 1
        if abs(s[j] - s[i]) >= min_deg and j - i >= min_frames: out.append((i, j, float(s[i]), float(s[j])))
        i = j + 1
    return out

def process_seq(args):
    seq, a = args; sid, rest = seq.split("_", 1); obj = rest.split("_")[0]
    meta = a["meta"][seq]
    z = np.load(f"{PKG}/trajectories/{seq}.npz"); frames = z["frames"]; offset = int(z["image_index_offset"])
    raw_obj = np.load(f"{ROOT}/raw_seqs/{sid}/{rest}.object.npy", allow_pickle=True)
    cam = np.load(f"{ROOT}/raw_seqs/{sid}/{rest}.egocam.dist.npy", allow_pickle=True).item()
    Rk, Tk = np.asarray(cam["R_k_cam_np"], np.float64), np.asarray(cam["T_k_cam_np"], np.float64).reshape(-1, 3)
    K, dist = z["K_ego"].astype(np.float64), z["dist8"].astype(np.float64)
    w2e_on = z["world2ego_onset"].astype(np.float64); e2w_on = np.linalg.inv(w2e_on)
    V, F, parts, face_top = load_mesh(obj)
    img_dir = f"{ROOT}/images/{sid}/{rest}/0"
    recs, frs, viz = {}, {}, []; stats = collections.Counter()
    for k, (i, j, d0, d1) in enumerate(strokes_of(z["arti_rad"], a["min_deg"], a["min_frames"])):
        f0 = int(frames[i]); img_p = f"{img_dir}/{f0 + offset:05d}.jpg"
        if not os.path.exists(img_p): stats["skip:no_image"] += 1; continue
        # hand nearest the moving part at the stroke start
        sides = [s for s in ("left", "right") if f"{s}_joints3d_onset" in z.files]
        side = min(sides, key=lambda s: float(z[f"{s}_dist_to_moving_part"][i]))
        J_on = z[f"{side}_joints3d_onset"][i:j + 1, ANCHOR_JOINT, :].astype(np.float64)  # onset-camera metres
        J_w = (e2w_on[:3, :3] @ J_on.T).T + e2w_on[:3, 3]
        R0, T0 = Rk[f0], Tk[f0]; J_c = (R0 @ J_w.T).T + T0  # stroke-start camera
        uv = project(J_c, K, dist); ok = np.isfinite(uv).all(1) & (J_c[:, 2] > 1e-3)
        if ok.sum() < 5: stats["skip:short_traj"] += 1; continue
        # object at the stroke start
        Vw, axis_w, org_w = pose_object(V, parts, raw_obj[f0]); Vc = (R0 @ Vw.T).T + T0
        axis_c = R0 @ axis_w; org_c = R0 @ org_w + T0
        img = cv2.imread(img_p); H, W = img.shape[:2]
        top, zb = rasterize(project(Vc, K, dist), Vc[:, 2], F, face_top, W, H, a["raster_scale"])
        mask = cv2.resize(top.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        depth_mm = cv2.resize((zb * 1000).astype(np.uint16), (W, H), interpolation=cv2.INTER_NEAREST)
        coords = thin_coords(mask, a["size"])
        if coords is None or len(coords) < 50: stats["skip:small_mask"] += 1; continue
        uvk, Jk, frk = uv[ok], J_c[ok], frames[i:j + 1][ok]
        if not (0 <= uvk[0][0] < W and 0 <= uvk[0][1] < H): stats["skip:hand_out_of_frame"] += 1; continue
        # HOI4D's 300 px rule was set at fx ~ 1000 px; ARCTIC's ego camera has fx ~ 2400 (2800 px wide),
        # so the same metric slack is ~2.4x more pixels — scale by focal length, not image width.
        px_scale = float(K[0, 0]) / 1000.0
        d_start = float(np.sqrt(((coords[:, ::-1] - uvk[0]) ** 2).sum(1)).min())
        if d_start > MAX_START_TO_MASK_PX * px_scale:
            stats["skip:start_far_from_mask"] += 1
            if a.get("debug"): print(f"  far: {seq} stroke {k} f{f0} hand={side} d_start={d_start:.0f}px dist_to_part={float(z[f'{side}_dist_to_moving_part'][i])*100:.1f}cm uv0={uvk[0].round()} in_frame={(0<=uvk[0][0]<W) and (0<=uvk[0][1]<H)}", flush=True)
            continue
        if len(uvk) > 1 and np.sqrt((np.diff(uvk, axis=0) ** 2).sum(1)).max() > MAX_FRAME_JUMP_PX * px_scale: stats["skip:frame_jump"] += 1; continue
        verb = "open" if abs(d1) > abs(d0) else "close"
        desc = f"{verb} the {OBJ_WORD.get(obj, obj)} {PART_WORD.get(meta['articulation'], 'part')}"
        org2d = project(org_c[None], K, dist)[0]
        key = f"{sid}_{obj}/{rest}_s{k:02d}_f{f0:05d}"
        recs[key] = {
            "rgb_image_path": key, "mask_coordinates_yx": coords.tolist(), "description": desc, "camera_intrinsics": K.tolist(),
            "motion_info": {"frame_specific_motion_data": {"motion_origin_2d_image_coords": [float(org2d[0]), float(org2d[1])],
                                                           "motion_dir_3d_camera_coords": axis_c.tolist(),
                                                           "motion_origin_3d_camera_coords": org_c.tolist()},
                            "original_motion_data": {"motion_type": "rot"}},
            "trajectory_3d_camera_coords": Jk.tolist(), "trajectory_2d_image_coords": uvk.tolist(), "trajectory_2d_valid": [True] * len(uvk),
            "arctic": {"seq": seq, "subject": sid, "object": obj, "articulation": meta["articulation"], "stroke": k, "verb": verb,
                       "start_frame": f0, "end_frame": int(frames[j]), "angle_start_deg": d0, "angle_end_deg": d1, "hand": side,
                       "traj_frames": frk.tolist(), "anchor_joint": ANCHOR_JOINT, "dist8": dist.tolist(), "image_size": [W, H],
                       "depth": "object-only render (mm)", "mask": "rendered top part, hands not removed"},
        }
        frs[key] = encode_frame(img, depth_mm, a["size"]); stats["ok"] += 1
        if len(viz) < a["viz"]:
            ov = img.copy(); ov[mask] = (0.5 * ov[mask] + [0, 0, 128]).astype(np.uint8)
            pts = [tuple(map(int, p)) for p in uvk]
            for p_, q_ in zip(pts, pts[1:]): cv2.line(ov, p_, q_, (0, 255, 0), 6)
            cv2.circle(ov, pts[0], 18, (255, 255, 0), -1); cv2.circle(ov, tuple(map(int, org2d)) if np.isfinite(org2d).all() else pts[0], 18, (0, 165, 255), -1)
            cv2.putText(ov, f"{key} '{desc}' {side} T={len(pts)} {d0:.0f}->{d1:.0f}deg", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
            viz.append((key, cv2.resize(ov, (W // 3, H // 3))))
    return seq, recs, frs, dict(stats), viz

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seqs", default=None)
    ap.add_argument("--size", type=int, default=512); ap.add_argument("--min-deg", type=float, default=10); ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--raster-scale", type=float, default=0.25); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--viz", type=int, default=0); ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(); import lmdb
    meta = {r["sequence_id"]: r for r in csv.DictReader(open(f"{PKG}/arctic_sample_list.csv"))}
    seqs = a.seqs.split(",") if a.seqs else sorted(meta)
    os.makedirs(a.out, exist_ok=True); env_d = lmdb.open(f"{a.out}/data.lmdb", map_size=1 << 36); env_f = lmdb.open(f"{a.out}/frames.lmdb", map_size=1 << 38)
    cfg = {"meta": meta, "min_deg": a.min_deg, "min_frames": a.min_frames, "raster_scale": a.raster_scale, "size": a.size, "viz": a.viz, "debug": a.debug}
    tot = collections.Counter(); n = 0; per_obj = collections.Counter(); vizs = []
    with ProcessPoolExecutor(a.workers) as ex:
        for seq, recs, frs, st, viz in ex.map(process_seq, [(s, cfg) for s in seqs]):
            with env_d.begin(write=True) as t:
                for k, v in recs.items(): t.put(k.encode(), pickle.dumps(v, protocol=4))
            with env_f.begin(write=True) as t:
                for k, v in frs.items(): t.put(k.encode(), pickle.dumps(v, protocol=4))
            n += len(recs); tot.update(st); vizs += viz
            for k in recs: per_obj[k.split("/")[0].split("_", 1)[1]] += 1
            print(f"{seq}: {len(recs)} records {st}", flush=True)
    with env_f.begin(write=True) as t:
        t.put(b"__metadata__", pickle.dumps({"entries": n, "depth_size": a.size, "jpeg_quality": 92, "source": "arctic_process_2d", "depth": "object-only render"}, protocol=4))
    if vizs:
        os.makedirs(f"{a.out}/viz", exist_ok=True)
        for k, im in vizs[: a.viz]: cv2.imwrite(f"{a.out}/viz/{k.replace('/', '__')}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("DONE", n, "records;", dict(tot)); print("per object:", per_obj.most_common())
    json.dump({"records": n, "stats": dict(tot), "per_object": dict(per_obj), "args": {k: v for k, v in vars(a).items()}}, open(f"{a.out}/build_stats.json", "w"), indent=1)

if __name__ == "__main__": main()
