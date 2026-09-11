# 20260912_joint4_decoder_h1anchor — joint decoder variant `h1anchor`

**Goal.** SF3D side = the cf_h1only recipe (H1 derivative quadratic 1.0 at pi/2 + direct axis loss 0.5; position L2 off) — the August MA record recipe, on the joint recipe.

**Comparison row.** `20260911_joint4_decoder_rgb_scalefree` (L2 at 2pi, no axis loss).

**Result (2026-09-11 10:50 local, pod jdec-h1anchor, 4 h 15 min + tests, pod deleted — verified).** Best
`val/sf3d/loss_total` 1.1649 at epoch 16.

| SF3D metric | dec_base (L2 2π) | dec_l2anchor (+axis) | **dec_h1anchor (H1 π/2 + axis)** |
|---|---|---|---|
| MA / signed | **31.29 / 30.86** | 31.05 / 30.27 | 31.07 / 29.97 |
| matched / all / signed-all | **18.4** / 25.3 / **32.1** | 18.9 / 24.3 / 34.0 | 19.9 / **24.1** / 35.2 |
| flips all / rot | **10.7 / 13.9** | 12.1 / 19.0 | 13.2 / 18.7 |
| origin / line | 0.273 / 0.247 | 0.294 / 0.257 | **0.269 / 0.244** |
| radius / point 3D | **0.124** / 0.289 | 0.135 / 0.299 | 0.132 / **0.261** |
| mIoU / PDet / point 2D | 0.266 / **22.5** / 0.108 | 0.241 / 18.5 / 0.118 | 0.266 / 22.1 / **0.104** |
| traj_dir acc / cos | **91.1 / 0.745** | 89.6 / 0.732 | 88.3 / 0.710 |
| HOI4D / EPIC / ARCTIC mIoU | 0.615 / 0.267 / 0.607 | 0.576 / 0.288 / 0.553 | 0.573 / **0.390** / 0.599 |

**Reading.** (1) The August record recipe on the SF3D side (H1 + anchor) does not beat L2 2π under the
decoder: MA −0.2, signed −0.9, matched +1.5°. It does give the best origin (0.269), best 3D point (0.261)
and best all-axis error (24.1) of the joint arms, with masks kept — the H1 term's usual precision gains
on placement. (2) **Sign again**: hinge flips 18.7 vs the base's 13.9, and trajectory direction 88.3 vs
91.1 — the second anchor-carrying variant in a row with ~+5 flip points. Two arms, same direction: under
the decoder the direct 1-cos axis loss is not just redundant (l2anchor's val axis term was identical to
the base) but seems to interact badly with the sign the 2D projection loss imposes through the arc. A
plausible mechanism: the hand-track direction fixes the axis sign by the right-hand rule relative to the
OBSERVED motion (opening), while the direct loss pulls toward SF3D's canonical sign — where these differ
per instance the two fight and the tail of flipped hinges grows. The seed replicate (`seed7`) will say
how much of the 14 → 19 is noise. (3) EPIC mIoU 0.390 / PDet 32 (vs 0.27 / 15 elsewhere): 53 val
records, treat as noise until repeated.

**Verdict.** No. The base recipe's SF3D side (L2 2π, no axis term) remains the best joint-decoder
recipe on MA / sign / direction; h1anchor is the best on origin / point. The remaining variant,
`cfframe` (2(1-k) built in), is the third test of "an explicit axis term under the decoder".

