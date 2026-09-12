# 20260913_joint4_decoder_l2anchor_dense_off — joint decoder arm `l2anchor_dense_off`

**Goal.** dense hinge voting + the per-pixel offset loss toward q* projection on SF3D (dense_offset_weight 0.5): direct supervision of every vote — placement hypothesis

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `loss_params.dense_offset_weight=0.5`; `config/joint4_decoder_l2anchor_dense_off.yaml`, `run_joint4dec_l2anchor_dense_off_chain.sh`, pod jdec-l2anchor_dense_off.

**Result (2026-09-13 11:40 local, pod jdec-doff — second pod after a power-capped lemon; Server Edition; pod deleted — verified).** Best val 1.1404 at epoch 17 (not comparable: carries the offset term).

| SF3D metric | dense s42 / s7 | **dense + offset loss 0.5** |
|---|---|---|
| MA / signed | 44.01 / 42.69 ; 45.13 / 44.75 | **45.97 / 45.32** |
| type acc | 96.0 / 93.9 | 94.4 |
| matched / all / signed-all | 11.4 / 19.6 / 25.0 ; 14.7 / 21.0 / 24.8 | 12.3 / 20.3 / 26.3 |
| flips all / rot | 7.6 / 12.7 ; 7.5 / 4.6 | 10.0 / 11.4 |
| origin / line / radius | 0.248 / 0.220 / 0.124 ; 0.252 / 0.221 / 0.124 | 0.281 / 0.235 / 0.140 |
| mIoU / PDet | 0.277 / 22.5 ; 0.263 / 21.1 | 0.247 / 20.3 |
| HOI4D / EPIC / ARCTIC mIoU | 0.508 / 0.186 / 0.509 ; 0.552 / 0.224 / 0.559 | 0.550 / 0.287 / 0.492 |
| ARCTIC probe axis / flips / hinge offset | 45.8 / 30 % / 0.112 ; 51.0 / 46 % / 0.104 | 46.5 / 55 % / **0.029** |

**Reading.** The per-pixel offset loss (every part pixel's vote pulled to q*'s projection on SF3D) is the
first thing that moves hand-video hinge PLACEMENT: the ARCTIC hinge-line offset drops from 0.07-0.16
(every previous arm) to **0.029** of the image width — the votes learn an image-space geometric rule
('the hinge is at that edge of the part') that transfers, unlike the axis sign. SF3D MA 46.0 (the best
of all arms, +1 over dense s7, noise-level), matched 12.3, while the 3D origin is 3 cm WORSE (0.281) and
SF3D masks lower (0.247): the loss trades a little of SF3D's 3D placement for 2D placement that
generalises. Sign on hand video still 55 % flips (the ambiguity is not touched by an offset loss).
Single seed.
