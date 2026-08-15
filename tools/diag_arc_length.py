"""Arc-length statistics of the synthesized GT trajectories.

Motivation (2026-08-16): the preprocessor sweeps prismatic trajectories a
fixed, arbitrary 0.1 m while revolute arcs are 90 deg at the element's real
radius. To replace the 0.1 m with something grounded in the data, measure
the polyline length of the stored revolute trajectories over a filtered key
set (pass the training run's key cache so the stats describe what the model
actually sees).

  /opt/venv/bin/python -u tools/diag_arc_length.py \
    --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl
"""
import argparse
import pickle

import lmdb
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v2")
    ap.add_argument("--key-cache", required=True)
    args = ap.parse_args()

    cached = pickle.loads(open(args.key_cache, "rb").read())
    keys = cached["keys"]
    print(f"keys: {len(keys)}  (cache filters: cutoff={cached.get('cutoff')}, "
          f"min_rev_radius={cached.get('min_revolute_radius')}, "
          f"mask_frac={cached.get('min_mask_area_frac')}, "
          f"edge={cached.get('edge_margin_frac')})", flush=True)

    env = lmdb.open(f"{args.data_root}/data.lmdb", readonly=True, lock=False)
    rot_len, trans_n = [], 0
    with env.begin() as txn:
        for i, key in enumerate(keys):
            if (i + 1) % 10000 == 0:
                print(f"progress {i + 1}/{len(keys)}", flush=True)
            rec = pickle.loads(txn.get(key))
            traj = rec.get("trajectory_3d_camera_coords")
            if not traj:
                continue
            motion_info = rec.get("motion_info") or {}
            original = motion_info.get("original_motion_data") or {}
            if original.get("motion_type", "trans") in ("rot", "rotation"):
                t = np.asarray(traj, dtype=np.float64)
                rot_len.append(float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum()))
            else:
                trans_n += 1

    a = np.asarray(rot_len)
    print(f"\nrot rows: {len(a)}   trans rows: {trans_n} "
          f"({100.0 * len(a) / max(1, len(a) + trans_n):.1f}% rot)")
    print(f"90-deg arc length (m):  mean {a.mean():.4f}   median {np.median(a):.4f}")
    for p in (10, 25, 75, 90):
        print(f"  p{p}: {np.percentile(a, p):.4f}")
    print(f"  min {a.min():.4f}   max {a.max():.4f}")
    r = a / (np.pi / 2.0)
    print(f"implied radius (m):     mean {r.mean():.4f}   median {np.median(r):.4f}")


if __name__ == "__main__":
    main()
