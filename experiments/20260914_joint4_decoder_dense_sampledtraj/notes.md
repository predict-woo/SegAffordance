# 20260914_joint4_decoder_dense_sampledtraj — joint decoder arm `dense_sampledtraj`

**Goal.** LOSS ABLATION: closed-form loss replaced by the SAMPLED analytic trajectory loss (20 decoded points vs the GT curve, weight 1.0) on SF3D; dense voting recipe otherwise

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `loss_params.closed_form_trajectory_weight=0.0 loss_params.analytic_trajectory_weight=1.0`; `config/joint4_decoder_dense_sampledtraj.yaml`, `run_joint4dec_dense_sampledtraj_chain.sh`, pod jdec-dense_sampledtraj.

**Result.** (pending)
