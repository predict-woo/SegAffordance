# clip_g3 full panels

Full panels ([GT | clip_g3], 12 samples, best-epoch08) — mask overlay,
predicted point, projected 3D trajectory (magenta line), decoded twist
orbit (yellow). Companion to [[20260804_sf3d_clip_g3_traj_points]].

Notable vs gen-2: trajectories are smooth lines (no scribble — the 1-cos
effect, see the traj-points batch); decoded orbits are clean planar
arcs/lines (pitch-free head: no helical drift representable); predicted
masks are now visible at the 0.5 threshold on the correct handles
(mIoU 0.103, best of any arm); |omega| commits to rot on true rotations
(cls=n/a: gen-3 has no type head — type is emergent).

Tool: tools/sf3d_vis_predictions.py (manifest.yaml). Rendered 2026-08-04.
