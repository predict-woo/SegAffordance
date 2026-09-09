# 20260911_hoi4d_laptop_probe_joint — articulation generalisation: the joint model on HOI4D laptops

The same 8 held-out HOI4D laptop records (C3, one per sequence, seed 0) as
`viz/20260909_hoi4d_laptop_articulation_probe`, three panels each:
`GT | joint4_dct | dct_chain`.

- `joint4_dct` = `20260910_joint4_dct_rgb_scalefree` best-epoch12 (trained jointly on SF3D + HOI4D + EPIC + ARCTIC; HOI4D laptops seen under the 2D recipe — masks, point, projected track, no articulation labels).
- `dct_chain` = `20260910_sf3d_g19_dct_rgb_scalefree_ft_multi3dct` best-epoch24 (2D pretrain then SF3D post-training; MA 32.80).

Overlays: predicted mask (red), point_uv ring, projected 3D trajectory
(magenta, z_p-scaled), predicted axis (red line: hinge line through
origin_pred for rot, direction ray for trans), origin-heatmap uv (small
red circle); text = type call, p_rev, axis direction, z_p. HOI4D has no 3D
articulation label, so the GT panel shows only mask + 2D track.

Regen (dev pod): `python tools/hoi4d_predict_articulation.py --model joint4_dct
config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml <ckpt> --model dct_chain
config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_multi3dct.yaml <ckpt> --out
viz/20260911_hoi4d_laptop_probe_joint --num 8 --category C3 --split val`.

What it shows: the joint model now SEGMENTS the lid (full-lid masks on
00 / 03 / 06 where every post-trained model had nothing), puts the point at
the hand, and its projected trajectory follows the GT track's direction
(up along the lid for "open", down for "close") — the 2D abilities it kept.
But the 3D articulation does NOT generalise: it calls "trans" on all 8
with p_rev 0.02-0.14 (more confidently wrong than the chain's 0.06-0.28),
the axes point in arbitrary directions, and z_p reads 2.6-3.7 m for a
laptop about 1 m away (the depth head is never supervised on HOI4D batches;
the chain's z_p of 1.0-1.3 m is at least plausible). The DCT chain has
forgotten the lid entirely (no mask) and also calls trans.

Reading: type / axis / origin only learn from SF3D batches, and SF3D has
no laptop-lid articulation; on HOI4D batches the type head is pushed only
by the p_rev prior (target 0.225) and the consistency term, so "trans" is
the learned default there. The direct fix is 3D supervision on laptops:
ARCTIC has 145 laptop strokes with exact mocap axes, origins AND metric 3D
trajectories, so ARCTIC batches can run the FULL 3D profile (type, axis,
origin, trajectory with scale gt_z0) instead of the 2D one — a config
change in the joint recipe, no new code.
