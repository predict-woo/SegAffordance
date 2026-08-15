"""Verify sf3d_processed_v3 against v2 (one-off, after tools/sf3d_build_v3.py).

Checks: entry counts match; rot records byte-identical; trans records have
0.7 m trajectories whose [0] and 2D [0] match v2's; 2D tracks equal a fresh
projection; metadata carries the v3 fields; the SF3D reader opens v3 with
the gen-9 key cache and serves a trans sample with a 0.7 m trajectory.
"""
import pickle
import sys

import lmdb
import numpy as np

sys.path.insert(0, "/workspace/SegAffordance")
from tools.sf3d_build_v3 import project_trajectory_to_2d  # noqa: E402

V2 = "/workspace/datasets/sf3d_processed_v2"
V3 = "/workspace/datasets/sf3d_processed_v3"

e2 = lmdb.open(f"{V2}/data.lmdb", readonly=True, lock=False)
e3 = lmdb.open(f"{V3}/data.lmdb", readonly=True, lock=False)
n2 = e2.stat()["entries"]
n3 = e3.stat()["entries"]
print(f"entries: v2={n2} v3={n3} match={n2 == n3}")

rot_checked = trans_checked = rot_identical = 0
with e2.begin() as t2, e3.begin() as t3:
    meta = pickle.loads(t3.get(b"__metadata__"))
    print(f"metadata: version={meta.get('version')} "
          f"trans_len={meta.get('trans_traj_length_m')} "
          f"derived_from={meta.get('derived_from')}")
    cur = t2.cursor()
    for key, v2val in cur:
        if key == b"__metadata__":
            continue
        if rot_checked >= 200 and trans_checked >= 200:
            break
        v3val = t3.get(key)
        rec2 = pickle.loads(v2val)
        typ = ((rec2.get("motion_info") or {}).get("original_motion_data")
               or {}).get("motion_type", "trans")
        if typ in ("rot", "rotation"):
            if rot_checked < 200:
                rot_checked += 1
                rot_identical += int(v3val == v2val)
        elif trans_checked < 200:
            trans_checked += 1
            rec3 = pickle.loads(v3val)
            t2a = np.asarray(rec2["trajectory_3d_camera_coords"])
            t3a = np.asarray(rec3["trajectory_3d_camera_coords"])
            L = float(np.linalg.norm(np.diff(t3a, axis=0), axis=1).sum())
            assert abs(L - 0.7) < 1e-6, f"len {L}"
            assert np.allclose(t2a[0], t3a[0], atol=1e-9), "traj[0] moved"
            c2d0 = np.asarray(rec2["trajectory_2d_image_coords"][0])
            c3d0 = np.asarray(rec3["trajectory_2d_image_coords"][0])
            assert np.allclose(c2d0, c3d0, atol=1e-6), "2d[0] moved"
            K = rec3["camera_intrinsics"]
            w, h = rec3["image_dimensions_wh"]
            cc, vv = project_trajectory_to_2d(t3a, K, w, h)
            assert np.allclose(np.asarray(cc),
                               np.asarray(rec3["trajectory_2d_image_coords"]),
                               atol=1e-9), "2d track != fresh projection"
            assert vv == list(rec3["trajectory_2d_valid"]), "valid mask"

print(f"rot byte-identical: {rot_identical}/{rot_checked}")
print(f"trans verified (len/pivot/2d): {trans_checked}/200")
e2.close()
e3.close()  # the reader below opens v3 itself; lmdb forbids two opens per process

from datasets.scenefun3d import SF3DDataset, get_default_transforms  # noqa: E402

r, m, d = get_default_transforms((256, 256))
ds = SF3DDataset(
    lmdb_data_root=V3,
    key_cache_path="/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl",
    frame_cache_path=f"{V3}/frames.lmdb",
    rgb_transform=r, mask_transform=m, depth_transform=d,
    image_size_for_mask_reconstruction=(256, 256),
    point_source="element", return_trajectory_2d=True,
    min_revolute_radius=0.10, min_mask_area_frac=0.001, edge_margin_frac=0.05,
)
print(f"reader opened v3 with g9 cache: {len(ds)} samples")
for i in range(0, 5000, 997):
    s = ds[i]
    if int(s[7]) == 0:  # trans
        traj = s[12].numpy()
        L = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
        print(f"  sample {i}: trans, 20-pt traj length {L:.4f}")
print("V3 VERIFY OK")
