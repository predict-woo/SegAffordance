# 20260913_joint4_decoder_l2anchor_query_seed7 — joint decoder arm `l2anchor_query_seed7`

**Goal.** seed-7 replicate of the query readout arm (noise calibration of its +4.7 MA / ARCTIC flips 13 %)

**Setup.** Derived from `config/joint4_decoder_l2anchor_query.yaml` (exp `20260913_joint4_decoder_l2anchor_query`) with overrides `seed_everything=7`; `config/joint4_decoder_l2anchor_query_seed7.yaml`, `run_joint4dec_l2anchor_query_seed7_chain.sh`, pod jdec-l2anchor_query_seed7.

**Result (2026-09-13 11:20 local, pod jdec-qseed7, Server Edition, pod deleted — verified).** Best val 1.1176 at epoch 19.

| SF3D metric | query seed 42 | **query seed 7** | l2anchor base |
|---|---|---|---|
| MA / signed | 35.75 / 35.24 | 33.57 / 33.29 | 31.05 / 30.27 |
| flips all / rot | 8.1 / 15.7 | **7.4 / 7.3** | 12.1 / 19.0 |
| matched / all / signed-all | 17.9 / 23.6 / 29.4 | 19.0 / 24.8 / 30.6 | 18.9 / 24.3 / 34.0 |
| origin / radius | 0.308 / 0.152 | 0.300 / 0.139 | 0.294 / 0.135 |
| mIoU / PDet | 0.270 / 23.35 | 0.268 / 23.2 | 0.241 / 18.5 |
| HOI4D / EPIC / ARCTIC | 0.625 / 0.386 / 0.588 | 0.614 / 0.330 / 0.582 | 0.576 / 0.288 / 0.553 |
| ARCTIC probe axis / flips / offset | 41.2 / **13.4 %** / 0.087 | 42.4 / **71.7 %** / 0.083 | 48.2 / 26.4 % / 0.070 |

**Reading.** SF3D: the query readout's gain holds at seed 7 (33.6 vs the base 31.05; two-seed mean 34.7,
+3.6) with the SF3D rot-flip rate at a record-level 7.3 % — the readout is real. HAND VIDEO: the ARCTIC
sign result does NOT hold — 13 % flips at seed 42, **72 % at seed 7** with the same 42 deg mean axis
error and the same ARCTIC traj_dir (78-79 %), and per object the flip rate jumps between 0 and 100 %
(laptop 0 % -> 100 %, scissors 12 -> 92 %). Since the decoded arcs follow the tracks equally well in
both seeds, the axis sign and the hinge SIDE flip together: for a short arc near the point, (axis n,
hinge on one side) and (-n, hinge on the other side) project to the same track — the 2D supervision
has an exact two-fold ambiguity that only 3D labels or a part-geometry prior can break. The
'halved ARCTIC flips' of seed 42 was the coin landing right. Consequence: single-seed ARCTIC flip rates
are not evidence; the hinge-line OFFSET (0.083-0.087, seed-stable) and the mean axis error are.
