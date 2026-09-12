# 20260913_joint4_decoder_l2anchor_dense_seed7 — joint decoder arm `l2anchor_dense_seed7`

**Goal.** seed-7 replicate of the dense hinge-voting arm (its MA 44.0 record needs a second seed)

**Setup.** Derived from `config/joint4_decoder_l2anchor_dense.yaml` (exp `20260913_joint4_decoder_l2anchor_dense`) with overrides `seed_everything=7`; `config/joint4_decoder_l2anchor_dense_seed7.yaml`, `run_joint4dec_l2anchor_dense_seed7_chain.sh`, pod jdec-l2anchor_dense_seed7.

**Result (2026-09-13 11:20 local, pod jdec-dseed7, Server Edition, 4 h 35 min + tests, pod deleted — verified).**
Best `val/sf3d/loss_total` 1.0205 at epoch 19.

| SF3D metric | dense seed 42 | **dense seed 7** | query s42 / s7 | l2anchor base |
|---|---|---|---|---|
| MA / signed | 44.01 / 42.69 | **45.13 / 44.75** | 35.75 / 33.57 | 31.05 / 30.27 |
| type acc | 96.0 | 93.9 | 93.9 / 93.2 | 92.4 |
| matched / all / signed-all (deg) | 11.4 / 19.6 / 25.0 | 14.7 / 21.0 / 24.8 | 17.9 / 19.0 ; 23.6 / 24.8 | 18.9 / 24.3 / 34.0 |
| flips all / rot (%) | 7.6 / 12.7 | **7.5 / 4.6** | 8.1 / 15.7 ; 7.4 / 7.3 | 12.1 / 19.0 |
| origin / line (m) | 0.248 / 0.220 | 0.252 / 0.221 | 0.308 / 0.300 | 0.294 |
| radius / point3d (m) | 0.124 / 0.310 | 0.124 / 0.315 | 0.152 / 0.139 ; 0.302 / 0.308 | 0.135 / 0.299 |
| mIoU / PDet | 0.2765 / 22.5 | 0.263 / 21.1 | 0.270 / 23.35 ; 0.268 / 23.2 | 0.241 / 18.5 |
| traj_dir acc | 94.8 | 94.5 | 93.5 / 93.2 | 89.6 |
| HOI4D / EPIC / ARCTIC mIoU | 0.508 / 0.186 / 0.509 | 0.552 / 0.224 / 0.559 | 0.625 / 0.386 / 0.588 ; 0.614 / 0.330 / 0.582 | 0.576 / 0.288 / 0.553 |
| ARCTIC probe: axis / flips / offset | 45.8 / 30.4 % / 0.112 | 51.0 / 46.2 % / 0.104 | 41.2 / 13.4 % / 0.087 ; 42.4 / 71.7 % / 0.083 | 48.2 / 26.4 % / 0.070 |

**Reading.** THE RECORD REPLICATES: MA 44.0 / 45.1 across two seeds (previous record 36.7), origin 0.248 /
0.252, radius 0.124 both, traj_dir 94.5-94.8, and seed 7 sets the rot-flip record (4.6 %). The dense
voting head is a +13 MA structural effect, not seed luck. The hand-source cost also replicates (HOI4D
0.51-0.55, EPIC 0.19-0.22 vs 0.58-0.63 / 0.29-0.39 for the query arms), and ARCTIC hinge transfer stays
poor (axis 46-51 deg, flips 30-46 %, offset 0.10-0.11). Two seeds of the same recipe: this is the SF3D
model; the 2D-side handling (`dense_d2d`, running) decides whether it can also be the hand-video model.
