# 20260912_joint4_decoder_cfframe_a3 — joint decoder, cf_frame at 3:1

**Goal.** The cfframe variant set MA 36.36 (record) at 2:1; on the arm-B config 3:1 fixed hinge flips (17.8 -> 13.0) at a mask/origin cost. Same on the joint recipe?

**Comparison row.** `20260912_joint4_decoder_cfframe`: MA 36.36 / 35.73, matched 18.8, rot flips 13.4, origin 0.313, mIoU 0.252 / PDet 20.0.

**Result (2026-09-11 15:40 local, pod jdec-cfa3, 4 h 20 min + tests, pod deleted — verified).** Best
`val/sf3d/loss_total` 1.2559 at epoch 19.

| SF3D metric | cfframe 2:1 (seed 42 / 7) | **cfframe 3:1** |
|---|---|---|
| MA / signed | 36.36 / 35.73, 36.03 / 35.91 | **36.73 / 36.14** |
| type pass | 92.4 / 93.3 | 91.1 |
| matched / all / signed-all | 18.8 / 24.3 / 30.7, 17.6 / 22.5 / 29.6 | **16.1** / 24.9 / 30.0 |
| flips all / rot | 9.8 / 13.4, 10.6 / 9.5 | **7.8** / 11.0 |
| origin / line | 0.313 / 0.284, 0.305 / 0.272 | 0.304 / 0.281 |
| radius / point 3D | 0.171 / 0.297, 0.150 / 0.271 | 0.137 / 0.284 |
| mIoU / PDet | 0.252 / 20.0, 0.264 / 23.9 | 0.247 / 18.5 |
| traj_dir | 92.4 / 91.6 | 92.2 |
| HOI4D / EPIC / ARCTIC mIoU | 0.578 / 0.313 / 0.562, 0.558 / 0.271 / 0.543 | **0.614 / 0.338 / 0.601** |

**Reading.** 3 : 1 on the joint recipe does what it did on the arm-B config: the sharpest matched axis of
the decoder arms (16.1°) and the lowest all-flip rate ever recorded (7.8 %, previous best 9.2), MA a hair
above the 2 : 1 pair (36.73, inside the seed noise of the 36.0–36.4 pair), at the cost of masks / PDet on
SF3D (0.247 / 18.5 — the same trade as arm-B a3) and 2 points of type pass. Unexpectedly it has the best
hand-source masks of any decoder arm (HOI4D 0.614, EPIC 0.338, ARCTIC 0.601) — single seed, but the
direction is the opposite of its SF3D mask cost.

**Verdict.** 2 : 1 and 3 : 1 are two points on the same sign-vs-mask trade; both hold the MA record band
(36.0–36.7). 3 : 1 for sign/axis precision, 2 : 1 (seed 7) for the all-round profile.
