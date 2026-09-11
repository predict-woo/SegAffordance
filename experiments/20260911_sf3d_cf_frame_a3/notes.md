# 20260911_sf3d_cf_frame_a3 — cf_frame with axis : phase = 3 : 1

**Question.** cf_frame (2 : 1) set the closed-form family MA record (31.03) with the best origin and masks
but more hinge flips (17.8 % vs cf_h1only's 15.4). At axis/phase = 2 the flipped axis has curvature
(−1 − w, 1 − w) with w = 1, i.e. (−2, 0): one flat escape direction. Does 3 : 1 (curvature (−3, −1))
fix the sign without losing the rest?

**Recipe.** `config/sf3d_train_runpod_cf_frame_a3.yaml` = cf_frame with `closed_form_frame_axis 3.0`.

**Comparison rows.** cf_frame: MA 31.03 / 30.27, matched 17.0°, rot flips 17.8, origin 0.245, mIoU 0.270.
cf_h1only: 30.64, 16.6°, 15.4, 0.254, 0.266. Prediction: rot flips ≤ 15, MA ≥ 31, origin unchanged.

**Status.** launched 2026-09-11 night (pod `segaffordance-cfframea3`, `run_cf_frame_a3_chain.sh`).
