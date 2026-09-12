# 20260913_field_joint4_l2anchor — joint decoder arm `field`

**Spec.** docs/superpowers/specs/2026-09-13-field-model-design.md

**Goal.** FIELD MODEL (model/field_model.py, train_field_better.py): one decoded map, per-pixel fields for axis / type / hinge offset / log-depth / arc length read out geometrically; predicted-mask weighting; depth field supervised by SF3D depth (0.5); 2D votes off the trunk; no pooled vector, no legacy heads; l2anchor loss recipe

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `loss_params.depth_field_weight=0.5 data.load_depth=true 2d:dense_trunk_detach=true 2d:depth_field_weight=0.0`; `config/joint4_decoder_field.yaml`, `run_joint4dec_field_chain.sh`, pod jdec-field.

**Run 1 (12:11-16:40 local, pod jdec-field #2): DIVERGED.** val/sf3d/loss_total 6.8 (ep0) -> 4.5 (ep3, best) ->
52-112 (ep6-14); val L_mask 4.8 at epoch 0 already 10x the dense arm's 0.46, L_point_map 6.9 vs 0.35,
then 133 / 166. Cause: the vote weights were the predicted mask with no floor — early masks ~0 make
the weighted means 1/wsum amplifiers (wsum ~ 1e-6), so the closed-form / offset gradients into the
field head and the trunk explode in the first steps and the trunk never recovers. Fix (commit
below): a uniform floor worth one cell mixed into the weights and wsum clamped at 1
(`test_empty_predicted_mask_does_not_amplify_gradients`). Logs kept in logs/run1_diverged/.

**Result.** (pending — run 2)
