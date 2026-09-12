# 20260913_field_joint4_l2anchor — joint decoder arm `field`

**Spec.** docs/superpowers/specs/2026-09-13-field-model-design.md

**Goal.** FIELD MODEL (model/field_model.py, train_field_better.py): one decoded map, per-pixel fields for axis / type / hinge offset / log-depth / arc length read out geometrically; predicted-mask weighting; depth field supervised by SF3D depth (0.5); 2D votes off the trunk; no pooled vector, no legacy heads; l2anchor loss recipe

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `loss_params.depth_field_weight=0.5 data.load_depth=true 2d:dense_trunk_detach=true 2d:depth_field_weight=0.0`; `config/joint4_decoder_field.yaml`, `run_joint4dec_field_chain.sh`, pod jdec-field.

**Run 1 (12:11-16:40 local, pod jdec-field #2): DIVERGED.** val/sf3d/loss_total 6.8 (ep0) -> 4.5 (ep3, best) ->
52-112 (ep6-14); val L_mask 4.8 at epoch 0 already 10x the dense arm's 0.46, L_point_map 6.9 vs 0.35,
then 133 / 166. Cause: the vote weights were the predicted mask with no floor — early masks ~0 make
the weighted means 1/wsum amplifiers (wsum ~ 1e-6), so the closed-form / offset gradients into the
field head and the trunk explode in the first steps and the trunk never recovers. Fix (commit
below): a uniform floor worth one cell mixed into the weights and wsum clamped at 1
(`test_empty_predicted_mask_does_not_amplify_gradients`). Logs kept in logs/run1_diverged/.

**Result (run 2: 2026-09-13 17:08-21:45 local, pod jdec-field #4; tests + ARCTIC probe on the dev pod; pod deleted — verified).**
Best `val/sf3d/loss_total` 1.0536 at epoch 14 (carries the depth-field + offset terms; dense's 0.98 lacks them).

| metric | dense s42 / s7 | dense_off | query s42 | **field** | prev best of that column |
|---|---|---|---|---|---|
| SF3D MA / signed | 44.0 / 42.7 ; 45.1 / 44.8 | 46.0 / 45.3 | 35.8 / 35.2 | 43.3 / 42.9 | 46.0 (dense_off) |
| type acc | 96.0 / 93.9 | 94.4 | 93.9 | 95.0 | 96.0 |
| matched / all / signed-all (deg) | 11.4 / 19.6 / 25.0 ; 14.7 / 21.0 / 24.8 | 12.3 / 20.3 / 26.3 | 17.9 / 23.6 / 29.4 | **10.7** / 19.9 / **24.0** | 11.4 / 19.6 / 24.8 |
| flips all / rot (%) | 7.6 / 12.7 ; 7.5 / 4.6 | 10.0 / 11.4 | 8.1 / 15.7 | **6.7** / 12.3 | 7.5 / 4.6 |
| origin / line / radius (m) | 0.248 / 0.220 / 0.124 ; 0.252 / 0.221 / 0.124 | 0.281 / 0.235 / 0.140 | 0.308 / 0.283 / 0.152 | 0.269 / 0.227 / 0.137 | 0.245 |
| point3d (m) | 0.310 / 0.315 | 0.308 | 0.302 | 0.324 | 0.248 |
| mIoU / PDet | 0.277 / 22.5 ; 0.263 / 21.1 | 0.247 / 20.3 | 0.270 / 23.4 | 0.259 / 19.1 | 0.277 / 23.9 |
| traj_dir acc / cos | 94.8 / 0.816 ; 94.5 / 0.813 | 94.2 / 0.796 | 93.5 / 0.776 | **96.3 / 0.834** | 96.4 / 0.830 (free head) |
| HOI4D mIoU / PDet | 0.508 / 59.6 ; 0.552 / 68.0 | 0.550 / 66.7 | 0.625 / 77.4 | **0.683 / 83.6** | 0.676 / 85.2 (joint4 head) |
| EPIC mIoU / PDet | 0.186 / 7.5 ; 0.224 / 11.3 | 0.287 / 22.6 | 0.386 / 32.1 | **0.404 / 28.3** | 0.391 |
| ARCTIC mIoU / PDet | 0.509 / 56.5 ; 0.559 / 63.2 | 0.492 / 48.3 | 0.588 / 62.6 | **0.652 / 79.3** | 0.618 / 68.7 (joint4) |
| ARCTIC probe axis / flips / hinge offset / r_med | 45.8 / 30 % / 0.112 ; 51.0 / 46 % / 0.104 | 46.5 / 55 % / **0.029** / 0.59 | 41.2 / 13 % / 0.087 / 0.93 | 55.3 / 41 % / 0.048 / 0.10 | 0.029 |

**Reading.** The clean field model is the first SINGLE model that holds both sides: SF3D articulation in
the dense record band (MA 43.3 vs 44-46, i.e. within the +-1-2 MA seed spread; matched axis 10.7 deg,
signed-all 24.0 and all-flips 6.7 are new records; traj_dir 96.3 ties the free head's record) AND the
best hand-video masks of any run (HOI4D 0.683 / 83.6 = the joint4 head's, EPIC 0.404, ARCTIC 0.652 / 79.3
— records), where every dense arm collapsed them. Hinge placement on hand video transfers (ARCTIC
offset 0.048, second only to dense_off's 0.029) with the hinge radius collapsed there (0.10 m median)
and a worse axis mean (55 deg; the sign is the known coin flip). Costs vs the best dense arm: -1..-3 MA,
+2 cm origin, PDet 19.1 (the lowest of the family; the point heatmap loses a little to the fields). The
model is 55M trainable parameters, no pooled vector, no MLP heads, no legacy modes, predicted-mask
weighting (with the floor). Single seed; replicate + the SF3D-only control are the next runs.
Best ckpt: `checkpoints/best-epoch14-sf3dval1.0536.ckpt` (keys `model.core.*`; load with
`tools/arctic_axis_probe.py --field` / the field trainer).
