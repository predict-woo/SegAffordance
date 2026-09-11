# 20260911_sf3d_cf_frame — the closed-form loss designed from the shape of the master formula

**Question.** Every point-arc loss expands to radius(λ) + λ^p (1−k)[1 + ρ cos(χ−χ₀)] + λ^p (1+k)(1−cos ψ)
(`knowledge/2026-09-11_revolute_loss_with_axis_term.md`). All closed-form arms had p = 1 (axis penalty
scaled by the predicted radius) and ρ > 0 (a flipped axis partly buyable by moving origin/point: floors
0.09 for L2, 0.60 for H1 at π/2). What happens with p = 0, ρ = 0 and a symmetric radius term?

**Recipe.** `config/sf3d_train_runpod_cf_frame.yaml` = cf_h1only with the H1 term and the direct axis
loss OFF and `closed_form_frame_*` ON: rot rows 2(1−k) + (1+k)(1−cos ψ) + 0.15 (log λ)², trans rows
2(1−cos), unit levers floored at 0.1 |r*|. Arm-B config (depth, no trajectory head, no L_pp), 30 ep,
seed 42, lr 1e-5. Pod cfframe (PRO 6000 Workstation, clocks OK), 4 h 05 min.

**Result (2026-09-11 03:45 local).** Best val 1.1587 at epoch 28 (last 29: 1.1593, flat/improving).

| metric | cf_h1only (H1 + axis 0.5) | **cf_frame** | closedform (L2+H1+axis) | cf_noaxis (L2+H1) |
|---|---|---|---|---|
| MA / signed | 30.64 / 30.11 | **31.03 / 30.27** | 29.19 / 28.89 | 27.71 / 27.26 |
| type pass | 95.5 | 95.1 | 93.3 | 93.8 |
| axis matched / all / signed-all | **16.6 / 24.5 / 31.7** | 17.0 / 24.8 / 33.5 | 22.3 / 26.8 / 34.0 | 17.6 / 26.4 / 34.6 |
| flips all / rot | 9.8 / **15.4** | 9.9 / 17.8 | 10.7 / 13.7 | 10.5 / 15.9 |
| origin / line | 0.254 / 0.230 | **0.245 / 0.215** | 0.250 / 0.229 | 0.253 / 0.229 |
| radius / point 3D | 0.130 / 0.234 | **0.123 / 0.229** | 0.128 / 0.238 | 0.123 / 0.231 |
| mIoU / PDet / point 2D | 0.266 / 21.8 / 0.106 | **0.270 / 21.9 / 0.097** | 0.258 / 20.1 / 0.110 | 0.263 / 20.8 / 0.104 |

**Reading.** (1) **MA 31.03 = the new record of the closed-form family** (arm-B config, depth, no 2D
init; +0.4 over cf_h1only, single seed) with the **best origin (0.245, line 0.215), radius, 3D point,
masks (0.270 / 21.9) and 2D point error (0.097) of every closed-form arm** — the p = 0 unit-lever form
lets the trunk keep its masks (no radius-scaled tug) and the log-radius term is a better origin
regulariser than (λ − 1)². (2) The registered prediction FAILED on sign: hinge flips 17.8 % vs 15.4
(all-flips equal, 9.9 vs 9.8) and the signed-all error 33.5 vs 31.7. Why, from the note's own curvature
table: with axis : phase = 2 : 1 the flipped axis has curvature (−1 − w, 1 − w) with w = 1, i.e.
**(−2, 0) — one FLAT direction**, a degenerate maximum, not the strict maximum the anchored H1 arm has
in combination with its radius-scaled tangential term. "w ≥ 1" in the design table should have read
w > 1. Matched axis 17.0 vs 16.6: within noise. (3) Everything the formula said about scale (no radius
scaling, no collapse plateau, symmetric radius) delivered; the sign prediction needed the strict
inequality.

**Follow-up (launched 2026-09-11 night, autonomous): `20260911_sf3d_cf_frame_a3`** — axis : phase = 3 : 1
(curvature (−3, −1) at the flip), everything else identical. Prediction: rot flips ≤ 15, MA ≥ 31,
origin unchanged.
