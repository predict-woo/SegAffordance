# 20260913_joint4_decoder_l2anchor_query_pos — joint decoder arm `l2anchor_query_pos`

**Goal.** OPTION 1 (user, 2026-09-13): location-conditioned queries — the point / hinge locations enter the depth, length and type-axis queries as sine codes (zero-init projection) and the single bilinear grid samples of the depth heads are REMOVED (depth_local_sample false)

**Setup.** Derived from `config/joint4_decoder_l2anchor_query.yaml` (exp `20260913_joint4_decoder_l2anchor_query`) with overrides `model_params.readout_query_pos=true model_params.depth_local_sample=false`; `config/joint4_decoder_l2anchor_query_pos.yaml`, `run_joint4dec_l2anchor_query_pos_chain.sh`, pod jdec-l2anchor_query_pos.

**Result.** (pending)
