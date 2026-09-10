# 20260912_joint4_decoder_cfframe — joint decoder variant `cfframe`

**Goal.** SF3D side = the shape-designed closed-form loss (closed_form_frame: 2(1-k) + (1+k)(1-cos psi) + 0.15 (log lam)^2, unit levers; no L2, no direct axis loss) on the joint recipe.

**Comparison row.** `20260911_joint4_decoder_rgb_scalefree` (L2 at 2pi, no axis loss).

**Status.** prepared 2026-09-12 (autonomous night); launched only if the base joint decoder run checks out.
