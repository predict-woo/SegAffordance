# 20260912_joint4_decoder_l2anchor — joint decoder variant `l2anchor`

**Goal.** SF3D side = closed-form L2 at 2pi PLUS the direct 1-cos axis loss (vae_weight 0.5) — the untested {L2 + anchor} corner of the August grid, on the joint recipe.

**Comparison row.** `20260911_joint4_decoder_rgb_scalefree` (L2 at 2pi, no axis loss).

**Result (2026-09-11 09:30 local, pod jdec-l2anchor, 4 h 20 min + tests, pod deleted — verified).** Best
`val/sf3d/loss_total` 1.1495 at epoch 17 (not comparable: the total now carries the axis term).

| SF3D metric | dec_base (L2 2π only) | **dec_l2anchor (+ axis 0.5)** | joint4 (head, with axis) |
|---|---|---|---|
| MA / signed | **31.29 / 30.86** | 31.05 / 30.27 | 31.41 / 30.15 |
| matched / all / signed-all | 18.4 / 25.3 / **32.1** | 18.9 / **24.3** / 34.0 | 20.1 / 26.1 / 34.4 |
| flips all / rot | **10.7 / 13.9** | 12.1 / 19.0 | 11.9 / 13.6 |
| origin / line | **0.273 / 0.247** | 0.294 / 0.257 | 0.327 / 0.297 |
| radius / point 3D | **0.124 / 0.289** | 0.135 / 0.299 | 0.163 / 0.285 |
| mIoU / PDet / point 2D | **0.266 / 22.5 / 0.108** | 0.241 / 18.5 / 0.118 | 0.274 / 22.7 / 0.104 |
| traj_dir acc / cos | **91.1 / 0.745** | 89.6 / 0.732 | 96.4 / 0.830 |
| HOI4D / EPIC / ARCTIC mIoU | **0.615 / 0.267 / 0.607** | 0.576 / 0.288 / 0.553 | 0.676 / 0.305 / 0.618 |

**Reading — a NEGATIVE result.** Adding the direct 1-cos axis loss to the decoder recipe's SF3D side
made almost everything worse: hinge flips 13.9 → 19.0, origin +2 cm, masks −0.025 / PDet −4, HOI4D
−0.04, MA −0.2; only the all-axis error improved (−1.0°). The validation axis term (`val/L_vae_total`,
logged unweighted in both runs) ends at 0.279 vs 0.281 — **the anchor did not improve the axis at all**:
the base recipe already learns the axis just as well through the 2π quadratic on SF3D plus the projection
loss on hand video, which reaches the axis heads through the decoded arc. So the extra gradient bought
nothing on the axis and re-balanced the trunk toward it (the trunk-pull signature on masks). This is the
opposite of the arm-B closed-form family, where the anchor was worth +1.5 MA and the sign fix — there, no
2D data supplied sign information. The flip jump (14 → 19) at equal mean axis error is a tail effect and
single-seed, but its direction is the same as every other metric here.

**Verdict.** Under the decoder on the joint recipe the direct axis loss is redundant and costly. Keep the
base (L2 2π only). The other two variants (h1anchor, cfframe) test whether a different SF3D-side loss
beats L2 2π; both carry an axis term (h1anchor: the anchor; cfframe: 2(1-k)), so read them with this
result in mind.

