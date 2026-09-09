# 20260910_multi3dct_val_panels — the multi-source 2D arm with the DCT-6 head on its three held-out splits

`20260910_multi3_dct_rgb_scalefree` best-epoch11-valloss0.4161.ckpt (HOI4D +
EPIC + ARCTIC, RGB-only scale-free teacher-forcing recipe, DCT-6 head,
augmentation x8, 12 epochs) on 12 held-out records of each source, the same
picks (seed 42421) as `viz/20260909_multi3_val_panels` (the plain-head arm)
so the two batches compare panel by panel. Left GT (moving-part mask green,
2D track cyan, first-point ring), right prediction (mask red, point_uv ring,
projected trajectory magenta anchored at point_uv at depth 1, p_rev).
`contact_sheet.jpg` per source.

Regen (dev pod), one per source:
`python tools/hoi4d_vis_2d_panels.py --config config/<src>_rgb_scalefree.yaml
--ckpt experiments/20260910_multi3_dct_rgb_scalefree/checkpoints/best-epoch11-valloss0.4161.ckpt
--set trajectory_dct_coeffs=6 --out viz/20260910_multi3dct_val_panels/<src> --num 12`
(the single-source configs are plain-headed; `--set` overrides the head).

What it shows, against the plain arm's panels: the predicted trajectories
are single smooth curves — a clean arc along the GT on the ARCTIC laptop
lid, the notebook cover and the espresso lever, short smooth curves on the
HOI4D drawers and safes — where the plain head zigzagged (roughness 0.07 ->
0.005-0.007). Masks and points are the same as the plain arm's on HOI4D
(12/12 on the moving part) and ARCTIC (11/12; the waffle-iron lid mask
also spills onto the lower plate — a new miss vs the plain arm). EPIC
masks are on the right fixture in most panels, a little more spill than the
plain arm (mIoU 0.520 vs 0.591 on 53 records). p_rev stays near the batch
prior on HOI4D/ARCTIC and self-organises on EPIC as before.
