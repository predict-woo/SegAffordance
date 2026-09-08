"""Build SF3D-format LMDBs from the EPIC/VISOR work items
(tools/epic_visor_propagate_batch.py output) — 2D-only training records
readable by datasets/scenefun3d.py:SF3DDataset, same layout as the HOI4D v2
builder (tools/hoi4d_process_2d.py):

  data.lmdb    pickled dicts keyed "<video_id>/<narration_id>" (scene prefix =
               the kitchen video, so split_dataset_by_scene keeps a kitchen in
               one split)
  frames.lmdb  {"jpeg", "depth_png" (16-bit mm — ZEROS: EPIC has no depth;
               the hand's WiLoR z at the onset is kept in the record for a
               later depth-optional model), "orig_size"} at --size

Per record: the contact-onset frame; mask = propagated VISOR fixture mask
thinned to the reader's gather grid; 2D trajectory = middle-knuckle (MANO
joint 9) track of the interacting hand from the onset (stride-2 frames,
onset-frame camera, pixels); 3D = the same joint's joints3d (metric only
for scale_regime=grip; placeholder for the 2D recipe); description = EPIC
narration; motion_info = content-free stub (no motion supervision).
Filters: |d| <= --max-d (default 30), area_ratio in [--min-area-ratio,
--max-area-ratio] (SAM2 sanity), >= 5 trajectory points, first point within
MAX_START_TO_MASK_PX of the mask, no frame jump > MAX_FRAME_JUMP_PX (HOI4D
outlier rules), mask >= 50 grid coords.
Usage: python tools/epic_process_2d.py --work /workspace/datasets/epic_processed_2d/work \
           --out /workspace/datasets/epic_processed_2d [--size 512]
"""
import argparse, csv, json, os, pickle, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi4d_process_2d import thin_coords, encode_frame, MAX_START_TO_MASK_PX, MAX_FRAME_JUMP_PX, ANCHOR_JOINT
PKG = "/workspace/datasets/epic_hands_package"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512); ap.add_argument("--max-d", type=int, default=30)
    ap.add_argument("--min-area-ratio", type=float, default=0.35); ap.add_argument("--max-area-ratio", type=float, default=2.5)
    ap.add_argument("--traj-end", choices=["span", "window"], default="span",
                    help="cut the knuckle track at the narrated action's span_end (default; the collaborator's tracks run "
                         "90 frames past it and include the hand leaving for the next task) or keep the full window")
    ap.add_argument("--traj-end-margin", type=int, default=4, help="frames kept after span_end (stride is 2)")
    ap.add_argument("--traj-frac", type=float, default=1.0,
                    help="keep only this fraction of the onset..span_end interval (1.0 = the full narrated action, the chosen default; 0.5 was tried and reverted 2026-09-08)")
    a = ap.parse_args()
    import lmdb
    os.makedirs(a.out, exist_ok=True)
    env_d = lmdb.open(f"{a.out}/data.lmdb", map_size=1 << 36); env_f = lmdb.open(f"{a.out}/frames.lmdb", map_size=1 << 38)
    items = sorted(d for d in os.listdir(a.work) if os.path.exists(f"{a.work}/{d}/meta.json"))
    stats = {}; n = 0; per_noun = {}
    def bump(k): stats[k] = stats.get(k, 0) + 1
    for nid in items:
        m = json.load(open(f"{a.work}/{nid}/meta.json"))
        if abs(m["d"]) > a.max_d: bump("skip:d"); continue
        if not (a.min_area_ratio <= m["area_ratio"] <= a.max_area_ratio): bump("skip:area_ratio"); continue
        if m["side_used"] is None: bump("skip:no_hand"); continue
        z = np.load(f"{PKG}/trajectories/{m['video_id']}/{nid}.npz"); side = m["side_used"]
        J = z[f"{side}_joints3d"][:, ANCHOR_JOINT, :].astype(np.float64); frames = z[f"{side}_frames"]
        K = m["intrinsics"]; fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
        uv = np.stack([fx * J[:, 0] / J[:, 2] + cx, fy * J[:, 1] / J[:, 2] + cy], 1)
        valid = np.isfinite(uv).all(1) & (J[:, 2] > 1e-3)
        if a.traj_end == "span":
            end = m["onset"] + a.traj_frac * (m["span"][1] - m["onset"]) + a.traj_end_margin if a.traj_frac < 1 else m["span"][1] + a.traj_end_margin
            valid &= frames <= end
        keep = np.nonzero(valid)[0]
        if len(keep) < 5: bump("skip:short_traj"); continue
        uv, J3, fr = uv[keep], J[keep], frames[keep]
        mask = cv2.imread(f"{a.work}/{nid}/mask.png", cv2.IMREAD_GRAYSCALE) > 127
        coords = thin_coords(mask, a.size)
        if coords is None or len(coords) < 50: bump("skip:small_mask"); continue
        if np.sqrt(((coords[:, ::-1] - uv[0]) ** 2).sum(1)).min() > MAX_START_TO_MASK_PX: bump("skip:start_far_from_mask"); continue
        if len(uv) > 1 and np.sqrt((np.diff(uv, axis=0) ** 2).sum(1)).max() > MAX_FRAME_JUMP_PX: bump("skip:frame_jump"); continue
        bgr = cv2.imread(f"{a.work}/{nid}/onset.jpg")
        if bgr is None: bump("skip:no_image"); continue
        Kmat = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        key = f"{m['video_id']}/{nid}"
        rec = {
            "rgb_image_path": key, "mask_coordinates_yx": coords.tolist(), "description": m["narration"],
            "camera_intrinsics": Kmat,
            "motion_info": {"frame_specific_motion_data": {"motion_origin_2d_image_coords": [0.0, 0.0],
                                                           "motion_dir_3d_camera_coords": [0.0, 0.0, 0.0],
                                                           "motion_origin_3d_camera_coords": [0.0, 0.0, 0.0]},
                            "original_motion_data": {"motion_type": "trans"}},
            "trajectory_3d_camera_coords": J3.tolist(), "trajectory_2d_image_coords": uv.tolist(),
            "trajectory_2d_valid": [True] * len(uv),
            "epic": {"video_id": m["video_id"], "narration_id": nid, "verb": m["verb"], "noun": m["noun"], "sides": m["sides"],
                     "hand": side, "scale_regime": m["scale_regime"], "onset_frame": m["onset"], "visor_frame": m["sparse_frame"],
                     "d": m["d"], "traj_frames": fr.tolist(), "hand_z_onset_m": float(J3[0, 2]), "area_ratio": m["area_ratio"],
                     "fixture_names": m["fixture_names"], "anchor_joint": ANCHOR_JOINT, "depth": "none (zeros)",
                     "traj_end": a.traj_end, "traj_frac": a.traj_frac, "span_end": m["span"][1]},
        }
        dep = np.zeros(bgr.shape[:2], np.uint16)
        with env_d.begin(write=True) as txn: txn.put(key.encode(), pickle.dumps(rec, protocol=4))
        with env_f.begin(write=True) as txn: txn.put(key.encode(), pickle.dumps(encode_frame(bgr, dep, a.size), protocol=4))
        n += 1; bump("ok"); per_noun[m["noun"]] = per_noun.get(m["noun"], 0) + 1
    with env_f.begin(write=True) as txn:
        txn.put(b"__metadata__", pickle.dumps({"entries": n, "depth_size": a.size, "jpeg_quality": 92, "source": "epic_process_2d",
                                               "depth": "zeros (EPIC has no depth)"}, protocol=4))
    print("DONE", n, "records;", stats); print("per noun:", sorted(per_noun.items(), key=lambda kv: -kv[1]))
    json.dump({"records": n, "stats": stats, "per_noun": per_noun, "args": vars(a)}, open(f"{a.out}/build_stats.json", "w"), indent=1)

if __name__ == "__main__": main()
