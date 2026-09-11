# 20260912_joint4_decoder_cfframe_seed7 — seed-7 replicate of the joint decoder cfframe variant

**Goal.** Confirm the MA 36.36 record (seed 42) at a second seed.

**Result (2026-09-11 15:40 local, pod jdec-cfs7, 4 h 20 min + tests, pod deleted — verified).** Best
`val/sf3d/loss_total` at epoch 19 (see checkpoints).

| SF3D metric | cfframe seed 42 | **cfframe seed 7** | base seed 42 / 7 |
|---|---|---|---|
| MA / signed | 36.36 / 35.73 | **36.03 / 35.91** | 31.29 / 30.86, 33.26 / 32.21 |
| type pass | 92.4 | **93.3** | 91.8 / 91.9 |
| matched / all / signed-all | 18.8 / 24.3 / 30.7 | 17.6 / **22.5 / 29.6** | 18.4 / 25.3 / 32.1, 18.8 / 25.3 / 33.3 |
| flips all / rot | 9.8 / 13.4 | 10.6 / **9.5** | 10.7 / 13.9, 10.5 / 15.2 |
| origin / line | 0.313 / 0.284 | 0.305 / 0.272 | 0.273 / 0.247, 0.311 / 0.271 |
| radius / point 3D | 0.171 / 0.297 | 0.150 / **0.271** | 0.124 / 0.289, 0.164 / 0.325 |
| mIoU / PDet | 0.252 / 20.0 | 0.264 / **23.9** | 0.266 / 22.5, 0.247 / 20.0 |
| traj_dir | 92.4 | 91.6 | 91.1 / 91.6 |
| HOI4D / EPIC / ARCTIC mIoU | 0.578 / 0.313 / 0.562 | 0.558 / 0.271 / 0.543 | 0.615 / 0.267 / 0.607, 0.573 / 0.244 / 0.573 |

**Reading.** The record REPLICATES: 36.36 and 36.03 at two seeds vs 31.29 / 33.26 for the base recipe —
a +3.9 MA gap against a ~2-point seed noise. This seed adds the best all-axis error ever (22.5°), the
best signed-all (29.6°), the lowest hinge flip rate ever (9.5 %), the best PDet ever (23.9) and the best
decoder-arm 3D point (0.271), with masks at the base's level (0.264). Origin 0.305 sits where the base's
seed-7 also lands (0.311). The hand-source masks stay in the decoder band (HOI4D 0.56–0.62).

**Verdict.** The shape-designed closed-form loss on the SF3D side of the joint decoder recipe is the new
default. Best checkpoint of the night by the all-round profile: this one (seed 7).
