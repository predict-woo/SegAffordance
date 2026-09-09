# 20260910_sf3d_g19_dct_rgb_scalefree_ft_multi3dct — SF3D DCT post-training from the multi3 DCT arm

**Question:** the DCT-at-both-stages chain on the RGB-only scale-free line: g19_dct SF3D recipe initialised from `20260910_multi3_dct_rgb_scalefree` (DCT-6 head, so the whole checkpoint loads 1:1 — no readout re-init). Compare with the plain-head post-training from the plain multi3 arm (`20260909_sf3d_plain_rgb_scalefree_ft_multi3`: MA 31.01 / PDet 22.72 / mIoU 0.2625, roughness 0.068) and the DCT post-training from the HOI4D-only plain arm with a re-initialised readout (30.07 / 22.35 / 0.266, roughness 0.0089).

**Recipe:** `config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_multi3dct.yaml` = the RGB g19_dct post-training config with the multi3-DCT init. 30 ep, lr 1e-5, milestones [24, 28]. Tested with pred_z_p and gt_z0.

**Result:** (pending)

**Decision:** (pending)
