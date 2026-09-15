# 20260915_joint4_decoder_dense_noproj — joint decoder arm `dense_noproj`

**Goal.** supervisor ablation: human video WITHOUT the trajectory (projection) loss, i.e. video contributes mask, type and point-heatmap supervision only; everything else = the final dense recipe

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `2d:trajectory_proj_weight=0.0`; `config/joint4_decoder_dense_noproj.yaml`, `run_joint4dec_dense_noproj_chain.sh`, pod jdec-dense_noproj.

**Result.** (pending)
