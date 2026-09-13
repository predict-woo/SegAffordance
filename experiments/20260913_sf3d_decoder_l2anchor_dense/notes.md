# 20260913_sf3d_decoder_l2anchor_dense — SF3D-only control of the final model

**Goal.** Paper gap A: does joint training on 2D human video change 3D articulation on SceneFun3D? Same recipe as the
final model `20260913_joint4_decoder_l2anchor_dense` (dense hinge voting, L2 2pi + axis 0.5, analytic decoder, lr 2e-5,
20 epochs, seed 42) with the three hand sources removed. SF3D sees the same 54,086 samples per epoch for the same 20
epochs; only the interleaved 2D batches are gone (so ~half the optimizer steps).

**Setup.** `config/joint4_decoder_l2anchor_dense_sf3donly.yaml`, `run_joint4dec_l2anchor_dense_sf3donly_chain.sh`, pod jdec-sf3donly (launched 2026-09-13
via the scratchpad launcher). Tests: SF3D (head length + writer length) and HOI4D / EPIC / ARCTIC zero-shot.

**Result.** (pending)
