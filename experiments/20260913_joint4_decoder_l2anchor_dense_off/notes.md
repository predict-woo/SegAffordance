# 20260913_joint4_decoder_l2anchor_dense_off — joint decoder arm `l2anchor_dense_off`

**Goal.** dense hinge voting + the per-pixel offset loss toward q* projection on SF3D (dense_offset_weight 0.5): direct supervision of every vote — placement hypothesis

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `loss_params.dense_offset_weight=0.5`; `config/joint4_decoder_l2anchor_dense_off.yaml`, `run_joint4dec_l2anchor_dense_off_chain.sh`, pod jdec-l2anchor_dense_off.

**Result.** (pending)
