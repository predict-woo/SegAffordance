# 20260913_sf3d_decoder_l2anchor_dense — SF3D-only control of the final model

**Goal.** Paper gap A: does joint training on 2D human video change 3D articulation on SceneFun3D? Same recipe as the
final model `20260913_joint4_decoder_l2anchor_dense` (dense hinge voting, L2 2pi + axis 0.5, analytic decoder, lr 2e-5,
20 epochs, seed 42) with the three hand sources removed. SF3D sees the same 54,086 samples per epoch for the same 20
epochs; only the interleaved 2D batches are gone (so ~half the optimizer steps).

**Setup.** `config/joint4_decoder_l2anchor_dense_sf3donly.yaml`, `run_joint4dec_l2anchor_dense_sf3donly_chain.sh`, pod jdec-sf3donly (launched 2026-09-13
via the scratchpad launcher). Tests: SF3D (head length + writer length) and HOI4D / EPIC / ARCTIC zero-shot.

**Result (2026-09-13, CHAIN_DONE 16:44 UTC, pod deleted).** best-epoch13-sf3dval1.1144. SF3D test:
signed MA **43.75** (unsigned 44.14; joint final 42.69 / 44.01 -> equal within seed noise), matched axis
**15.2 deg** (joint 11.4), all-axis 22.2, origin 0.262 (joint 0.248), origin-line 0.236, radius 0.127,
mIoU 0.262 / PDet 22.4 (joint 0.277 / 22.5), type 93.6, traj_dir 92.1. Zero-shot hand video: HOI4D mIoU
0.012 / PDet 0.2, EPIC 0.010 / 0.0, ARCTIC 0.034 / 0.0; type acc HOI4D 57 / EPIC 94 / ARCTIC 40; MA 0 on
all three. ARCTIC probe (329 held-out revolute): axis 56.6 deg mean / 60.9 median, <10 deg 1.2 %,
flips 38 %, hinge-line offset 0.084, type 40.4 %.

**Reading.** Human video does NOT raise SF3D signed MA (control is +1.1, inside the +-1 seed band); it
sharpens the matched axis (15.2 -> 11.4) and improves masks / origin slightly. Its real effect is
generalisation: without it the model finds nothing on hand video (masks ~0.01) and its ARCTIC axes are
at chance; with it masks 0.51 / 0.19 / 0.51 and ARCTIC axis 45.8. Paper Section E / Table III written
accordingly (2026-09-14). Note: the control's hinge-line OFFSET on ARCTIC (0.084) is lower than the
joint model's (0.112) despite chance-level axes, i.e. the offset metric rewards hinge lines through the
part centre; do not read it as placement quality (see STATE 'point-to-line offset too lenient').
