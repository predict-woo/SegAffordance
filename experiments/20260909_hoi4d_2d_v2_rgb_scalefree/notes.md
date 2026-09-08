# 20260909_hoi4d_2d_v2_rgb_scalefree — HOI4D v2, RGB-only, scale-free trajectory head (plain teacher-forcing recipe)

**Question:** does the 2D recipe still train when the depth map is removed from BOTH the input (use_depth false, load_depth false) and the loss (projection anchor = GT first pixel at depth 1, head predicts Δ̃ = Δ/z0, z_p tether off)? This is the RGB-only counterpart of `20260907_hoi4d_2d_v2_teacher_forcing_plain` (plain 20-point head, GT-anchored projection): same data, epochs, lr, batch; only the depth/scale changes. Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md.

**Recipe:** `config/hoi4d_v2_rgb_scalefree.yaml` = teacher_forcing_plain with `use_depth false`, `trajectory_scale_free true`, `trajectory_proj_anchor "unit"`, `trajectory_scale_source "unit"`, `depth_anchor_weight 0`, `pred_pred_art_radius_floor 0.15` (0.10 m in units of anchor depth), `test_trajectory_scale "unit"`, data `load_depth false`. 100 epochs, lr 3e-5, batch 64, 15% val by physical object.

**Comparison row (depth, plain TF):** held-out mIoU 0.708 / PDet 86.7 / point err 0.0157 / traj proj-2D shape 0.0413 / traj_dir 49.2.

**Result:** (pending)

**Decision:** (pending)
