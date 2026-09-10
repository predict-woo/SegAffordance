# 20260911_joint4_decoder_rgb_scalefree — joint 2D+3D with the analytic decoder (no trajectory head)

**Design (user, 2026-09-11).** The trajectory is no longer a learned head. On hand-video batches (HOI4D
2,821 labelled windows, EPIC 359 VLM-labelled, ARCTIC 2,559) the analytic decoder renders it from the
GT-routed type, the split axis heads, the lifted origin/point and a learned ARC LENGTH (rot: angle =
L / radius, radius detached, clamped at π; trans: L · direction), in the scale-free frame, and the
unit-anchor projection loss trains it; type CE 0.5 on every source. On SF3D batches the closed-form
position quadratic at sweep 2π is the only trajectory-related loss (no direct axis loss, no rendered
curve). No residual, no consistency term, no derivative terms. Everything else = the joint4 recipe
(source-homogeneous batches, hand ×10, lr 2e-5, 20 ep, monitor `val/sf3d/loss_total`, from scratch).
At test the decoder uses the predicted type; SF3D is tested with the learned length and with the
writer's constants (π/2, 0.7 m) for comparability with the head-based arms.

**Comparison rows.** joint4 (MA 31.41 / signed 30.15, mIoU 0.2738, PDet 22.7, traj_dir 96.4, HOI4D
0.676 / 85.2), joint v2 (32.94 / 32.08, 0.250 / 19.1), DCT chain (32.80 / 32.43), cf_l2_noaxis at π/2
(MA 23.80, rot flips 20.1; the L2-only loss this recipe uses, on the arm-B config) and the 2π arm in
flight (`20260911_sf3d_cf_l2_noaxis_2pi`).

**Status.** launched 2026-09-11 (pod `segaffordance-jdec`, `run_joint4dec_chain.sh`, log `joint4dec_chain.log`).
