# 20260911_hoi4d_laptop_joint_only — the joint model alone on HOI4D laptops

10 held-out HOI4D laptop records (category C3, one per sequence — the val
split has 10 laptop sequences; seed 1), two panels each: `GT | joint4_dct`.
`joint4_dct` = `20260910_joint4_dct_rgb_scalefree` best-epoch12 (SF3D +
HOI4D + EPIC + ARCTIC in one stream; HOI4D laptops were seen under the 2D
recipe only: mask, point, projected track, no articulation label).

Overlays: predicted mask (red), point_uv ring, projected 3D trajectory
(light green, z_p-scaled), predicted axis (red: hinge line for rot, direction
ray for trans), origin-heatmap uv (small red circle); text = type call,
p_rev, axis direction, z_p. 

Regen (dev pod): `python tools/hoi4d_predict_articulation.py --model joint4_dct
config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml
experiments/20260910_joint4_dct_rgb_scalefree/checkpoints/best-epoch12-sf3dval0.9688.ckpt
--out viz/20260911_hoi4d_laptop_joint_only --num 12 --category C3 --split val --seed 1 --ray-len 0.5 --scale 2` (panels rendered at 2x = 1024 px with thin overlays, PNG; axis rays 0.5 m at the predicted depth; trajectories in light green). `contact_sheet.png` = all 10 at half of that.

What it shows: lid masks in 10/10 (full lid, clean edges), the point at the
hand on the lid rim, projected trajectories in the GT track's direction
(up along the lid for "open", down for "close") with the DCT head's smooth
shape — the 2D ability the joint model kept. And the same 3D failure in
10/10: type "trans" with p_rev 0.03-0.14, axes in arbitrary directions,
z_p 2.5-4.0 m for laptops about 1 m away. The articulation heads only
learn from SF3D batches and SF3D has no lid; the type head's only signal on
HOI4D batches is the p_rev prior, so "trans" is its default there. Fix =
ARCTIC's laptop strokes (145, exact hinge axes/origins, metric tracks)
under the 3D loss profile in the joint recipe.
