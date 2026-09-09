# 20260909_hoi4d_rgb_scalefree_val_panels — HOI4D v2 held-out, RGB-only scale-free arm

16 stratified validation samples (8 C4 trans / 8 C6 rot, seed 42421) of
`20260909_hoi4d_2d_v2_rgb_scalefree` (best-epoch83-valloss0.2993.ckpt):
the teacher_forcing_PLAIN recipe with NO depth (use_depth false,
load_depth false) and the scale-free trajectory head (Δ̃ = Δ/z0, projection
anchor = GT first pixel at depth 1). Spec: docs/superpowers/specs/
2026-09-09-rgb-only-scale-free-trajectory-design.md.

Panels: left GT (moving-part mask green, wrist point ring, 2D track cyan);
right prediction (mask red, point_uv ring, projected trajectory magenta —
anchored at point_uv at depth 1, the unit convention the loss sees; text =
p_rev). `contact_sheet.jpg` = all 16 at half size, two per row.

Regen (dev pod): `python tools/hoi4d_vis_2d_panels.py --config
config/hoi4d_v2_rgb_scalefree.yaml --ckpt experiments/20260909_hoi4d_2d_v2_rgb_scalefree/checkpoints/best-epoch83-valloss0.2993.ckpt
--out viz/20260909_hoi4d_rgb_scalefree_val_panels --num 16` (manifest.yaml has argv).

What it shows: masks on the correct moving part in 16/16 (the drawer front,
the cabinet door leaf, the safe door), the point at the knuckle, and the
projected trajectories heading in the GT direction with a plausible extent
— with NO depth map anywhere and the anchor fixed at depth 1. The curves
are jittery (plain per-point head, no DCT basis — same as every plain arm;
the SF3D stage uses the DCT head). p_rev sits at ~0.24 for every sample =
the batch prior; the type gate does not self-organize on 2D data (known
from all 2D-only arms). Held-out numbers: mIoU 0.733 / PDet 88.7 / point
err 0.0147 vs the depth tf_plain arm 0.708 / 86.7 / 0.0157 (see the
experiment's notes.md).
