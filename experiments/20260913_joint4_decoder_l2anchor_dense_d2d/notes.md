# 20260913_joint4_decoder_l2anchor_dense_d2d — joint decoder arm `l2anchor_dense_d2d`

**Goal.** dense hinge voting with the votes computed from a DETACHED map on the 2D-only sources (loss profile 2d: dense_trunk_detach) — keep the SF3D record, protect the hand-source masks

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `2d:dense_trunk_detach=true`; `config/joint4_decoder_l2anchor_dense_d2d.yaml`, `run_joint4dec_l2anchor_dense_d2d_chain.sh`, pod jdec-l2anchor_dense_d2d.

**Result.** (pending)
