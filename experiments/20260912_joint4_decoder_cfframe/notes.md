# 20260912_joint4_decoder_cfframe — joint decoder variant `cfframe`

**Goal.** SF3D side = the shape-designed closed-form loss (closed_form_frame: 2(1-k) + (1+k)(1-cos psi) + 0.15 (log lam)^2, unit levers; no L2, no direct axis loss) on the joint recipe.

**Comparison row.** `20260911_joint4_decoder_rgb_scalefree` (L2 at 2pi, no axis loss).

**Result (2026-09-11 11:05 local, pod jdec-cfframe, 4 h 15 min + tests, pod deleted — verified).** Best
`val/sf3d/loss_total` 1.1178 at epoch 17. Test = the standard SF3D test set (5,088; 80 batches), the
standard 10° threshold, the same test config as the other decoder arms; the writer-length pass is
identical on every metric.

| SF3D metric | joint v2 (prev record) | dec_base (L2 2π) | **dec_cfframe** |
|---|---|---|---|
| MA / signed | 32.94 / 32.08 | 31.29 / 30.86 | **36.36 / 35.73** |
| type pass | 91.6 | 91.8 | **92.4** |
| matched / all / signed-all | **16.4** / 25.2 / 32.0 | 18.4 / 25.3 / 32.1 | 18.8 / 24.3 / **30.7** |
| flips all / rot | 10.8 / 15.2 | 10.7 / 13.9 | **9.8 / 13.4** |
| origin / line | 0.355 / 0.320 | **0.273 / 0.247** | 0.313 / 0.284 |
| radius / point 3D | 0.161 / 0.279 | **0.124** / 0.289 | 0.171 / 0.297 |
| mIoU / PDet / point 2D | 0.250 / 19.1 / 0.113 | **0.266 / 22.5** / 0.108 | 0.252 / 20.0 / **0.104** |
| traj_dir acc / cos | 96.0 / 0.828 | 91.1 / 0.745 | 92.4 / 0.757 |
| HOI4D / EPIC / ARCTIC mIoU | 0.591 / 0.299 / 0.546 | 0.615 / 0.267 / 0.607 | 0.578 / 0.313 / 0.562 |

**Reading.** (1) **MA 36.36 / signed 35.73 = new all-time records by +3.4 / +3.7** (previous 32.94 /
32.43), with the best type pass, the best signed-all axis error (30.7), all-flips 9.8 (= the record) and
the best decoder-arm direction accuracy (92.4). The mean all-axis error only moves 25.3 → 24.3, so the
gain is in the CORE of the distribution: 5 more percent of the test set falls under 10°. That is what the
p = 0 unit-lever form promised — an axis term that is scale-free in the lever, so small-radius hinges get
the same axis gradient as big doors. (2) The costs are the same trade the arm-B cf_frame showed in the
other direction: origin 0.313 (base 0.273), radius 0.171 (0.124), masks 0.252 / 20.0 (0.266 / 22.5).
The log-radius term at 0.15 is a weaker origin regulariser than the 2π quadratic's 3:1 radial weighting.
(3) Unlike the two anchor-carrying variants, this one has a built-in 2(1-k) axis term and its hinge
flips DROP (13.4 vs 13.9): the frame loss's axis term is the same 1-cos but sits inside a loss whose
other term, (1+k)(1-cos ψ), is also scale-free — no conflict with the 2D sign this time. So the earlier
"anchor hurts under the decoder" reading is more precisely "an axis term with a radius-scaled partner
(L2 or H1) hurts"; the seed replicates will sharpen this.

**Verdict.** The SF3D-side loss for the joint decoder recipe is the shape-designed loss. Follow-ups
launched: `cfframe_a3` (3:1, the arm-B sign fix) and `cfframe_seed7` (replicate of the record).

