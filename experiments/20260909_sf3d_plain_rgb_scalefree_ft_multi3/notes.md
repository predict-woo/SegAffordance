# 20260909_sf3d_plain_rgb_scalefree_ft_multi3 — SF3D post-training, PLAIN head, from the multi-source 2D arm

**Question:** post-train the multi-source 2D arm (`20260909_multi3_rgb_scalefree`: HOI4D + EPIC + ARCTIC, RGB-only scale-free plain-TF recipe, augmentation x8) on SF3D WITHOUT swapping the trajectory head: `trajectory_dct_coeffs 0`, so every weight of the 2D checkpoint — trunk, decoder, mask/point/origin heads, z_p/z_q, axis and type heads, and the full trajectory MLP including its readout — loads 1:1 (model_params verified identical; user 2026-09-09 after learning that the DCT post-training re-initialised the plain readout). First SF3D post-training with a plain head and the first from a multi-source init. Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md.

**Recipe:** `config/sf3d_train_runpod_plain_rgb_scalefree_ft_multi3.yaml` = `sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml` with `trajectory_dct_coeffs 0` and the multi3 init. 30 ep, lr 1e-5, milestones [24, 28]; losses: mask/point/origin heatmaps, point-3D, origin-3D, axis, type, normalized trajectory (metres = GT z0 · Δ̃), consistency 0.1; fdiff / closed-form / analytic / twist / projection / tether all 0. Test with pred_z_p (logs/test.log) and gt_z0 (logs/test_gt_z0.log). Runs on pod C right after the multi3 test (run_multi3_post_chain.sh).

**Comparison rows (all DCT head at the SF3D stage):** RGB ft-from-HOI4D 30.07 / 22.35 / 0.2660 (MA / PDet / mIoU); RGB scratch 24.88 / 17.39 / 0.2425; depth tf_plain init 30.62 / 20.03 / 0.2555. Confounds vs those: plain head AND multi-source init; a plain post-training from the HOI4D-only arm would separate them.

**Result:** (pending)

**Decision:** (pending)
