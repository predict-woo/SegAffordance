# 20260909_hoi4d_laptop_articulation_probe — SF3D-post-trained models asked for 3D articulation on HOI4D laptops

8 held-out HOI4D laptop records (category C3, one per sequence, seed 0),
three panels each: `GT | rgb_ft_hoi4d | depth_tf_plain`.

- `rgb_ft_hoi4d` = `20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d` best-epoch24 (RGB-only scale-free, SF3D post-training from the RGB HOI4D arm).
- `depth_tf_plain` = `20260907_sf3d_g19_dct_ft_hoi4d_tf_plain` best-epoch25 (depth input, SF3D post-training from the depth HOI4D arm; gets HOI4D's sensor depth here).

GT panel: HOI4D moving-part mask (green), knuckle track (cyan), first
point ring. HOI4D carries NO 3D articulation label (motion_info is a
"trans" placeholder), so there is no GT axis to draw. Model panels:
predicted mask (red), point_uv ring, projected 3D trajectory head (magenta,
z_p-scaled for the scale-free head), predicted axis (red: hinge line
through origin_pred for rot, direction ray for trans), origin-heatmap uv
(small red circle); text = type call, p_rev, axis direction, z_p.

Regen (dev pod): `python tools/hoi4d_predict_articulation.py --model rgb_ft_hoi4d
<cfg> <ckpt> --model depth_tf_plain <cfg> <ckpt> --out
viz/20260909_hoi4d_laptop_articulation_probe --num 8 --category C3 --split val`
(manifest.yaml has the exact argv and checkpoints).

What it shows: NEITHER post-trained model recovers the laptop hinge. Both
call "trans" on 15 of 16 panels (p_rev 0.12-0.48; the depth model's one
"rot" call puts the hinge 0.76 m away, off the table), the predicted masks
are empty or a few pixels on the lid's sticker, the point lands on the lid
edge or the hand, and the predicted axes point along the table / the arm
rather than along the lid's hinge line. The projected trajectories sweep
downward across the table for "open the laptop screen" where the GT track
goes up along the lid. Reading: (1) SF3D post-training forgets HOI4D
(measured 2026-09-07: HOI4D mIoU 0.727 -> 0.12 after post-training), so
the mask/point are gone before any articulation reasoning starts; (2) SF3D
never contains a laptop-lid articulation ("laptop" appears only as "click on
the laptop screen"), so the axis/type heads have no laptop concept — they
produce SF3D-style drawer/door guesses. A laptop hinge needs 3D
articulation supervision on laptops: ARCTIC v1 has 145 laptop strokes with
real hinge axes (`arctic_processed_2d`, `motion_info` type rot), which the
multi-source line can feed; the in-flight multi3 arm at least keeps the
laptop mask/point in 2D.
