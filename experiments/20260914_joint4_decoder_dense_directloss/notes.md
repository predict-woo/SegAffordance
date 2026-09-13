# 20260914_joint4_decoder_dense_directloss — joint decoder arm `dense_directloss`

**Goal.** LOSS ABLATION: closed-form loss OFF on SF3D; only the direct parameter losses (axis 1-cos 0.5, hinge-to-q* 0.5, 3D point 0.5); dense voting recipe otherwise

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `loss_params.closed_form_trajectory_weight=0.0`; `config/joint4_decoder_dense_directloss.yaml`, `run_joint4dec_dense_directloss_chain.sh`, pod jdec-dense_directloss.

**Result.** (pending)
