# 20260911_sf3d_cf_frame_a3 — cf_frame with axis : phase = 3 : 1

**Question.** cf_frame (2 : 1) set the closed-form family MA record (31.03) with the best origin and masks
but more hinge flips (17.8 % vs cf_h1only's 15.4). At axis/phase = 2 the flipped axis has curvature
(−1 − w, 1 − w) with w = 1, i.e. (−2, 0): one flat escape direction. Does 3 : 1 (curvature (−3, −1))
fix the sign without losing the rest?

**Recipe.** `config/sf3d_train_runpod_cf_frame_a3.yaml` = cf_frame with `closed_form_frame_axis 3.0`.

**Comparison rows.** cf_frame: MA 31.03 / 30.27, matched 17.0°, rot flips 17.8, origin 0.245, mIoU 0.270.
cf_h1only: 30.64, 16.6°, 15.4, 0.254, 0.266. Prediction: rot flips ≤ 15, MA ≥ 31, origin unchanged.

**Result (2026-09-11 08:55 local, pod cfframea3, 3 h 40 min, pod deleted — verified).** Best val 1.3749 at
epoch 26 (val totals not comparable across weightings).

| metric | cf_h1only (H1 + axis 0.5) | cf_frame (2:1) | **cf_frame_a3 (3:1)** |
|---|---|---|---|
| MA / signed | 30.64 / 30.11 | 31.03 / 30.27 | **31.90 / 31.33** |
| type pass | 95.5 | 95.1 | 95.0 |
| axis matched / all / signed-all | 16.6 / **24.5** / **31.7** | 17.0 / 24.8 / 33.5 | **16.2** / 25.1 / 32.3 |
| flips all / rot | 9.8 / 15.4 | 9.9 / 17.8 | **9.2 / 13.0** |
| origin / line | 0.254 / 0.230 | **0.245 / 0.215** | 0.262 / 0.234 |
| radius / point 3D | 0.130 / 0.234 | **0.123** / 0.229 | 0.132 / 0.229 |
| mIoU / PDet / point 2D | 0.266 / 21.8 / 0.106 | **0.270 / 21.9 / 0.097** | 0.249 / 19.0 / 0.103 |

**Reading.** (1) The registered predictions held where they were about sign: hinge flips 17.8 → **13.0**
(the best of any anchored closed-form arm; = cf_noaxis_2pi's 13.0), all-flips **9.2** (record), matched
axis **16.2°**, and **MA 31.90 / signed 31.33 = the closed-form family record** (+0.9 over cf_frame, +1.3
over cf_h1only). The strict curvature condition (axis : phase > 2) was the missing piece. (2) The
prediction "origin unchanged" failed: origin 0.245 → 0.262 (+1.7 cm), and **masks fell 0.270 → 0.249 /
21.9 → 19.0**. Raising the axis weight from 2 to 3 pulled the trunk toward the axis heads at the expense
of the mask / heatmap channels and loosened the lever-phase term's relative weight (origin/radius). The
articulation-vs-mask trade seen with every stronger axis/derivative term reappears here as a single
knob. (3) Signed-all error 32.3 (better than cf_frame, worse than h1only).

**Verdict.** 3 : 1 is the right sign fix and the best MA of the family; 2 : 1 is the better origin/mask
point. The trade suggests 2.5 : 1 (or 3 : 1 with the log-radius weight raised to ~0.3) as the next
knob; the same loss is running on the joint decoder recipe (`20260912_joint4_decoder_cfframe`, at 2 : 1).

