# 20260911_sf3d_val_panels_sharp — SF3D validation predictions, sharp style

16 SF3D validation samples (8 trans + 8 rot, seed 7, the g19 val split and
filters), rendered at 2x (1024 px per panel, PNG, thin overlays):
`GT | joint4_dct | dct_chain`.

- `joint4_dct` = `20260910_joint4_dct_rgb_scalefree` best-epoch12 (joint 2D+3D; MA 31.41, mIoU 0.2738).
- `dct_chain` = `20260910_sf3d_g19_dct_rgb_scalefree_ft_multi3dct` best-epoch24 (2D pretrain -> SF3D post-training; MA 32.80).

GT panel: GT mask (green), GT 2D track (cyan), GT interaction point (white
ring), GT axis (green: hinge line through the GT origin for rot, a 0.5 m
direction ray from the track start for trans). Prediction panels: mask
(red), point_uv (white ring), projected trajectory (light green, z_p-scaled),
predicted axis (red: hinge line for rot / 0.5 m ray for trans), 90-deg orbit
(yellow) for rot, origin-heatmap uv (small red circle); text = type, p_rev,
radius, axis direction, z_p, and the axis error vs GT + whether the type
call matches. `contact_sheet.png` = all 16 at 0.4 scale.

Regen (dev pod): `python tools/sf3d_vis_val.py --model joint4_dct
config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml <ckpt> --model dct_chain
config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_multi3dct.yaml <ckpt> --out
viz/20260911_sf3d_val_panels_sharp --num 16 --seed 7` (manifest.yaml has the argv).

What it shows: on SF3D validation both models are in their element — types
right, hinge lines on the door edges, axis errors mostly 8-18°, plausible
z_p (1.0-1.7 m), masks on the described part. The trajectories are smooth
arcs (DCT head) that follow the decoded orbit on most doors (closet 14: both
models sweep along the yellow orbit; the joint model's arc runs the full
90°); the joint model still produces the occasional loop (washing-machine
door 13) where the chain's arc is cleaner, matching the roughness / flip
numbers. On the prismatic samples (keyboard 00) both trajectories run down
the GT direction with the direction ray on top. Contrast with the in-the-wild
photos (`viz/20260911_iphone_probe`): in-domain the sweep and the orbit agree,
out of domain they do not.
