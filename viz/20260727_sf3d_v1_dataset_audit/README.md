# SF3D processed-dataset sample visualisations

100 random records from `/workspace/datasets/sf3d_processed/data.lmdb`
(461,334 records), rendered 2026-07-27 by `tools/sf3d_vis_samples.py`
with `--seed 42`. Regenerate with:

```bash
bash runpod/dev.sh start
# key scan / random reads against the MooseFS volume run at ~1.4 MB/s;
# a sequential copy runs at ~155 MB/s, so stage the LMDB in shm first:
bash runpod/dev.sh run "mkdir -p /dev/shm/data.lmdb && \
  cp /workspace/datasets/sf3d_processed/data.lmdb/data.mdb /dev/shm/data.lmdb/"
bash runpod/dev.sh run "python tools/sf3d_vis_samples.py \
  --lmdb-path /dev/shm/data.lmdb --out-dir /workspace/SegAffordance/viz/20260727_sf3d_v1_dataset_audit \
  --num-samples 100 --seed 42"
bash runpod/dev.sh stop
```

## Contents

- `panels/NNN_<motiontype>_<visit>_<video>_<timestamp>_<annot>.jpg` — one per
  sample: **full frame | zoom on the element | depth over the same zoom**.
  The zoom panel is the point: the median mask is 704 px and the median
  element sits 2.2 m away, so a 0.1 m GT trajectory is only a few pixels at
  full frame.
- `contact_sheet.jpg` — all 100 zoom panels, 10x10.
- `index.tsv` — per-sample geometry (mask px, trajectory length, origin
  depth, zoom factor, depth validity, degenerate flag).

## Legend

| colour | meaning |
|---|---|
| green fill / white contour | mask — convex hull of the visible laser-scan points |
| magenta dot | `motion_origin_2d_image_coords` (the regressed interaction point) |
| cyan → red polyline | `trajectory_3d_camera_coords`; ring = first point, cross = last |
| yellow arrow | `motion_dir_3d_camera_coords` for `trans` |
| orange double arrow | rotation axis for `rot` |
| yellow box (full frame) | the zoom region |

## What this run showed

- 100/100 rendered, no missing RGB or depth files; depth valid over 98% of
  the zoom crop on average, none below 50%.
- Motion split: 61 `trans` / 39 `rot`.
- Masks are hull approximations and often tiny: median 704 px, min 18 px
  (`022_rot_...`, 24 px^2 of a handle). Small ones are legitimate — sockets
  and switches at 2-6 m — but the hull visibly over-covers concave parts.
- **4/39 `rot` samples (10%) have a degenerate trajectory**: the arc radius
  fell below `TRAJECTORY_MIN_ROT_RADIUS_M` (0.01 m) in
  `tools/sf3d_process.py`, so the GT collapsed to the 0.01 m straight
  fallback pointing along the rotation axis. Flagged in the banner and in
  `index.tsv` (`degenerate_rot`). These carry no usable rotational
  supervision — the trajectory loss sees a near-zero segment.
- 2/100 have a trajectory that leaves the frame (63/100 and 36/100 points
  in-frame). Expected for 90° arcs on large doors; not a defect.
