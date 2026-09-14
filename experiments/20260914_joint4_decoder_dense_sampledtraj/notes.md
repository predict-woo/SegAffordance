# 20260914_joint4_decoder_dense_sampledtraj — joint decoder arm `dense_sampledtraj`

**Goal.** LOSS ABLATION: closed-form loss replaced by the SAMPLED analytic trajectory loss (20 decoded points vs the GT curve, weight 1.0) on SF3D; dense voting recipe otherwise

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `loss_params.closed_form_trajectory_weight=0.0 loss_params.analytic_trajectory_weight=1.0`; `config/joint4_decoder_dense_sampledtraj.yaml`, `run_joint4dec_dense_sampledtraj_chain.sh`, pod jdec-dense_sampledtraj.

**Result (2026-09-13, CHAIN_DONE, pod deleted by the launcher).** best-epoch19-sf3dval1.1797. SF3D test: signed MA **42.26** (unsigned 42.57), matched axis **10.2 deg**, origin **0.236** (line 0.216), mIoU 0.224 / PDet 17.2, type 96.1, rot flips 5.7; hand video HOI4D 0.479 / 54.6, EPIC 0.204 / 7.5, ARCTIC 0.479 / 49.9. ARCTIC hinge probe: pending (dev pod OOM-killed the probe while the baselines session's 16-worker USDNet conversion runs there).

**Reading.** The sampled 20-point loss (the closed form's teacher) matches the closed form on articulation (42.3 vs 42.7 MA, 10.2 vs 11.4 deg, 0.236 vs 0.248 m: all within seed noise) but costs the masks (0.224 vs 0.277 mIoU, PDet 17.2 vs 22.5) and the hand-video masks. The closed form = same articulation, no sampling, no extent, no mask cost. Paper Table V (3D loss block) filled 2026-09-14.
