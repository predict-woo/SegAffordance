# 20260912_sf3d_decoder_l2anchor_ft_multi3dec — SF3D post-training from the multi3 decoder arm (chain stage 2)

**Goal.** The decoder CHAIN counterpart of the DCT chain (MA 32.80): SF3D post-training (30 ep, lr 1e-5, RGB-only scale-free) with the analytic decoder, closed-form L2 at 2pi + the direct axis loss 0.5, from `20260912_multi3_decoder_rgb_scalefree` best.

**Comparison rows.** DCT chain 32.80 / 32.43, origin 0.256, mIoU 0.254; joint decoder base 31.29, origin 0.273; joint decoder l2anchor (running).

**Result (2026-09-11 14:30 local, pod mdec, 3 h 35 min for stage 2 + tests, pod deleted — verified).** Best
val 1.2462 at epoch 24, init = `20260912_multi3_decoder_rgb_scalefree` best-epoch09.

| SF3D metric | DCT chain (head) | **decoder chain** | decoder joint base | decoder joint cf_frame |
|---|---|---|---|---|
| MA / signed | 32.80 / 32.43 | 28.81 / 28.36 | 31.29 / 30.86 | **36.36 / 35.73** |
| type pass | 90.8 | 89.7 | 91.8 | **92.4** |
| matched / all / signed-all | 20.5 / 28.5 / 34.2 | 20.4 / 28.1 / 35.9 | 18.4 / 25.3 / 32.1 | 18.8 / 24.3 / **30.7** |
| flips all / rot | 9.9 / **9.8** | 11.6 / 15.0 | 10.7 / 13.9 | **9.8** / 13.4 |
| origin / line | **0.256 / 0.230** | **0.254 / 0.230** | 0.273 / 0.247 | 0.313 / 0.284 |
| radius / point 3D | 0.125 / **0.248** | 0.145 / 0.257 | **0.124** / 0.289 | 0.171 / 0.297 |
| mIoU / PDet | 0.254 / 20.7 | 0.257 / 19.1 | **0.266 / 22.5** | 0.252 / 20.0 |
| traj_dir / roughness | 92.6 / 0.008 | 89.9 / 0.001 | 91.1 / 0 | 92.4 / 0 |

**Reading.** (1) The chain does NOT transfer under the decoder: MA 28.8 vs the DCT chain's 32.8 and the
decoder joint arms' 31.3–36.4, with 5 more hinge flips than the DCT chain and the lowest type pass of
the line. The 2D decoder arm was already a weaker init (union masks 0.626 vs 0.715), and the SF3D stage
carries the direct axis loss, which the joint variants suggest hurts under the decoder — two
confounded reasons, same direction. (2) What it keeps: the DCT chain's origin (0.254) and masks (0.257),
i.e. the chain still places hinges well, it just gets sign/direction wrong more often. (3) Under the
head-based line the chain beat joint training on MA (32.80 vs 31.41); under the decoder the ordering
flips (28.8 vs 31.3 base / 36.4 cf_frame). The decoder wants the 3D data present while the axis heads
are shaped by the hand-video projection loss, not before.

**Verdict.** Joint training is the recipe for the decoder; the chain is not worth a second seed.

