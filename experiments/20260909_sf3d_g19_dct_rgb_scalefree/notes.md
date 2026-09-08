# 20260909_sf3d_g19_dct_rgb_scalefree — SF3D 3D-DCT from scratch, RGB-only, scale-free head

**Question:** the depth ablation at equal recipe: `20260821_sf3d_g19_dct` (scratch, depth input, MA 25.98 / PDet 21.72 / mIoU 0.2685) vs the same recipe with NO depth (use_depth false, load_depth false) and the scale-free head (metres = GT z0 · Δ̃ at train, z_p · Δ̃ at test). Pairs with `20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d` (same model, HOI4D init) to separate "depth" from "HOI4D transfer". Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md.

**Recipe:** `config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml` (30 ep, lr 1e-5, DCT-6 head, scratch). Tested twice: pred_z_p (deployable) and gt_z0 (oracle scale; logs/test_gt_z0.log).

**Result:** (pending)

**Decision:** (pending)
