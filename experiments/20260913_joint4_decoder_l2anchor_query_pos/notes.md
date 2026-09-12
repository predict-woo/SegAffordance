# 20260913_joint4_decoder_l2anchor_query_pos — joint decoder arm `l2anchor_query_pos`

**Goal.** OPTION 1 (user, 2026-09-13): location-conditioned queries — the point / hinge locations enter the depth, length and type-axis queries as sine codes (zero-init projection) and the single bilinear grid samples of the depth heads are REMOVED (depth_local_sample false)

**Setup.** Derived from `config/joint4_decoder_l2anchor_query.yaml` (exp `20260913_joint4_decoder_l2anchor_query`) with overrides `model_params.readout_query_pos=true model_params.depth_local_sample=false`; `config/joint4_decoder_l2anchor_query_pos.yaml`, `run_joint4dec_l2anchor_query_pos_chain.sh`, pod jdec-l2anchor_query_pos.

**Result (2026-09-13 16:00 local, pod jdec-qpos, Server Edition, pod deleted — verified).** Best val 1.1439 at epoch 19.

| SF3D metric | query (grid samples; s42 / s7) | **query_pos (location queries, no grid samples)** |
|---|---|---|
| MA / signed | 35.75 / 35.24 ; 33.57 / 33.29 | 34.98 / 34.34 |
| type acc | 93.9 / 93.2 | 91.3 |
| matched / all / signed-all | 17.9 / 23.6 / 29.4 ; 19.0 / 24.8 / 30.6 | 17.6 / 24.7 / 32.5 |
| flips all / rot | 8.1 / 15.7 ; 7.4 / 7.3 | 11.3 / 9.8 |
| origin / radius / point3d | 0.308 / 0.152 / 0.302 ; 0.300 / 0.139 / 0.308 | 0.304 / 0.150 / **0.292** |
| mIoU / PDet | 0.270 / 23.4 ; 0.268 / 23.2 | 0.251 / 20.4 |
| traj_dir | 93.5 / 93.2 | 91.4 |
| HOI4D / EPIC / ARCTIC mIoU | 0.625 / 0.386 / 0.588 ; 0.614 / 0.330 / 0.582 | **0.630** / 0.261 / **0.606** |
| ARCTIC probe axis / flips / offset | 41.2 / 13 % / 0.087 ; 42.4 / 72 % / 0.083 | 52.8 / 70 % / 0.088 |

**Reading (option 1).** Replacing the two single-pixel grid samples by location-conditioned queries is
NEUTRAL on SF3D articulation (MA 35.0 inside the query arm's 33.6-35.8 seed band; origin 0.304 = ;
point3d 0.292 marginally better), slightly worse on SF3D masks / PDet (0.251 / 20.4), the best HOI4D and
ARCTIC masks of the query family (0.630 / 0.606), and no change in hand-video hinge placement (offset
0.088). So the grid sample was not load-bearing — the depth heads get what they need from a
location-aware query — and the location conditioning does not by itself buy placement (the origin
needs geometry: the offset-loss result). Keep as the cleaner default of the query line; the field
model (running) supersedes both. Single seed.
