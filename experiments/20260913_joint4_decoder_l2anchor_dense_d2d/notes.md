# 20260913_joint4_decoder_l2anchor_dense_d2d — joint decoder arm `l2anchor_dense_d2d`

**Goal.** dense hinge voting with the votes computed from a DETACHED map on the 2D-only sources (loss profile 2d: dense_trunk_detach) — keep the SF3D record, protect the hand-source masks

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `2d:dense_trunk_detach=true`; `config/joint4_decoder_l2anchor_dense_d2d.yaml`, `run_joint4dec_l2anchor_dense_d2d_chain.sh`, pod jdec-l2anchor_dense_d2d.

**Result (2026-09-13 12:00 local, pod jdec-dd2d, Server Edition, pod deleted — verified).** Best val 1.0920 at epoch 13.

| SF3D metric | dense s42 / s7 | dense_off | **dense_d2d (2D votes off the trunk)** | query s42 |
|---|---|---|---|---|
| MA / signed | 44.01 / 42.69 ; 45.13 / 44.75 | 45.97 / 45.32 | 41.63 / 40.72 | 35.75 / 35.24 |
| type acc | 96.0 / 93.9 | 94.4 | **96.0** | 93.9 |
| matched / all / signed-all | 11.4 / 19.6 / 25.0 ; 14.7 / 21.0 / 24.8 | 12.3 / 20.3 / 26.3 | 12.5 / 20.6 / 25.9 | 17.9 / 23.6 / 29.4 |
| flips all / rot | 7.6 / 12.7 ; 7.5 / 4.6 | 10.0 / 11.4 | 8.1 / 9.1 | 8.1 / 15.7 |
| origin / radius | 0.248 / 0.124 ; 0.252 / 0.124 | 0.281 / 0.140 | 0.302 / 0.124 | 0.308 / 0.152 |
| mIoU / PDet | 0.277 / 22.5 ; 0.263 / 21.1 | 0.247 / 20.3 | 0.269 / 20.5 | 0.270 / 23.4 |
| traj_dir | 94.8 / 94.5 | 94.2 | **94.9** | 93.5 |
| HOI4D / EPIC / ARCTIC mIoU | 0.508 / 0.186 / 0.509 ; 0.552 / 0.224 / 0.559 | 0.550 / 0.287 / 0.492 | 0.582 / 0.214 / 0.522 | 0.625 / 0.386 / 0.588 |
| ARCTIC probe axis / flips / offset | 45.8 / 30 % / 0.112 ; 51.0 / 46 % / 0.104 | 46.5 / 55 % / 0.029 | 51.0 / 68 % / 0.072 | 41.2 / 13 % / 0.087 |

**Reading.** Computing the hand-video votes from a detached map is a partial answer: HOI4D masks recover
half the gap (0.582 vs dense 0.51-0.55; query 0.61-0.63), ARCTIC hinge offset improves (0.072 vs
0.10-0.11), SF3D type 96.0 and traj_dir 94.9 are the best, but EPIC masks do not recover (0.214) and SF3D
pays 3 MA and 5 cm of origin (41.6 / 0.302 vs 44-45 / 0.25). So the mask collapse is only partly the
projection loss reaching the trunk through the votes; the rest is the trunk's capacity being spent on
SF3D's geometry once the fields make it cheap to satisfy the SF3D losses. The field model (running)
carries this rule plus the offset loss and a depth field; a balance knob (projection weight, or the
2D repeat factor) is the remaining lever. Single seed.
