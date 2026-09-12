# 20260913_field_joint4_l2anchor — joint decoder arm `field`

**Spec.** docs/superpowers/specs/2026-09-13-field-model-design.md

**Goal.** FIELD MODEL (model/field_model.py, train_field_better.py): one decoded map, per-pixel fields for axis / type / hinge offset / log-depth / arc length read out geometrically; predicted-mask weighting; depth field supervised by SF3D depth (0.5); 2D votes off the trunk; no pooled vector, no legacy heads; l2anchor loss recipe

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `loss_params.depth_field_weight=0.5 data.load_depth=true 2d:dense_trunk_detach=true 2d:depth_field_weight=0.0`; `config/joint4_decoder_field.yaml`, `run_joint4dec_field_chain.sh`, pod jdec-field.

**Result.** (pending)
