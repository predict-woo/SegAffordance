# 20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d — SF3D 3D-DCT post-training from the RGB-only scale-free HOI4D arm

**Question:** the RGB-only counterpart of `20260907_sf3d_g19_dct_ft_hoi4d_tf_plain` (MA 30.62 / PDet 20.03 / mIoU 0.2555, depth input, metric head): with NO depth anywhere and the scale-free head (metres = GT z0 · Δ̃ at train, z_p · Δ̃ at test), how much of the HOI4D→SF3D transfer survives, and what does depth buy? Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md.

**Recipe:** `config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml` = the g19_dct recipe (30 ep, lr 1e-5, DCT-6 head) with `use_depth false`, `load_depth false`, `trajectory_scale_free true`, `trajectory_scale_source "gt_z0"`, `test_trajectory_scale "pred_z_p"`; `finetune_from_path` = `20260909_hoi4d_2d_v2_rgb_scalefree` best (plain head → DCT head: everything but the trajectory head's output layer loads). Tested twice: pred_z_p (deployable) and gt_z0 (oracle scale, shape only; logs/test_gt_z0.log).

**Comparison rows:** scratch g19_dct (depth) 25.98 / 21.72 / 0.2685; tf_plain init (depth) 30.62 / 20.03 / 0.2555; tf (DCT) init (depth, record) 31.13 / 23.27 / 0.266.

**Result:** (pending)

**Decision:** (pending)
