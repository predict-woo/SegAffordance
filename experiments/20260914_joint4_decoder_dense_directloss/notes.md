# 20260914_joint4_decoder_dense_directloss — joint decoder arm `dense_directloss`

**Goal.** LOSS ABLATION: closed-form loss OFF on SF3D; only the direct parameter losses (axis 1-cos 0.5, hinge-to-q* 0.5, 3D point 0.5); dense voting recipe otherwise

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `loss_params.closed_form_trajectory_weight=0.0`; `config/joint4_decoder_dense_directloss.yaml`, `run_joint4dec_dense_directloss_chain.sh`, pod jdec-dense_directloss.

**Result (2026-09-13, CHAIN_DONE, pod deleted by the launcher).** best-epoch16-sf3dval0.7608. SF3D test: signed MA **36.73** (unsigned 36.77), matched axis 16.3 deg, origin 0.305 (line 0.277), mIoU **0.270** / PDet **24.9**, type 95.4, rot flips 8.8; hand video HOI4D 0.613 / 74.4, EPIC 0.267 / 13.2, ARCTIC 0.544 / 60.5 (best hand masks of the dense family). ARCTIC hinge probe: pending (dev pod OOM-killed the probe while the baselines session's 16-worker USDNet conversion runs there).

**Reading.** Direct parameter losses lose 6 MA and 5 deg matched axis / 6 cm hinge to either trajectory loss, but keep the best masks: the trajectory losses' gradients through the trunk cost segmentation. Paper Table V (3D loss block) filled 2026-09-14.
