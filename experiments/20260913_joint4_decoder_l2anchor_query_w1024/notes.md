# 20260913_joint4_decoder_l2anchor_query_w1024 — joint decoder arm `l2anchor_query_w1024`

**Goal.** query readout + head width 1024 (spatial readout x capacity: the two effects that each bought MA)

**Setup.** Derived from `config/joint4_decoder_l2anchor_query.yaml` (exp `20260913_joint4_decoder_l2anchor_query`) with overrides `model_params.vae_hidden_dim=1024 model_params.trajectory_length_hidden=1024`; `config/joint4_decoder_l2anchor_query_w1024.yaml`, `run_joint4dec_l2anchor_query_w1024_chain.sh`, pod jdec-l2anchor_query_w1024.

**Result (2026-09-13 12:10 local, pod jdec-qw1024 — second pod after a power-capped lemon; pod deleted — verified).** Best val 1.0775 at epoch 19.

| SF3D metric | query 256 (s42 / s7) | mlp1024 | **query x 1024** |
|---|---|---|---|
| MA / signed | 35.75 / 35.24 ; 33.57 / 33.29 | 37.70 / 36.87 | **38.56 / 37.56** |
| type acc | 93.9 / 93.2 | 91.1 | 92.2 |
| matched / all / signed-all | 17.9 / 23.6 / 29.4 ; 19.0 / 24.8 / 30.6 | 18.2 / 24.2 / 31.2 | 16.8 / 23.8 / 30.3 |
| flips all / rot | 8.1 / 15.7 ; 7.4 / 7.3 | 11.3 / 14.8 | 8.2 / 16.6 |
| origin / radius / point3d | 0.308 / 0.152 / 0.302 ; 0.300 / 0.139 / 0.308 | 0.301 / 0.157 / 0.262 | 0.316 / 0.130 / 0.282 |
| mIoU / PDet | 0.270 / 23.4 ; 0.268 / 23.2 | 0.239 / 20.1 | 0.249 / 20.8 |
| HOI4D / EPIC / ARCTIC | 0.625 / 0.386 / 0.588 ; 0.614 / 0.330 / 0.582 | 0.584 / 0.279 / 0.527 | 0.589 / 0.222 / 0.596 |
| ARCTIC probe axis / flips / offset | 41.2 / 13 % / 0.087 ; 42.4 / 72 % / 0.083 | 44.4 / 66 % / 0.073 | 43.0 / 19.5 % / 0.108 |

**Reading.** Width on top of the query readout adds MA (38.6, the best of the query family) but the two
effects do not add cleanly: masks and PDet fall to the mlp1024 level (0.249 / 20.8), EPIC masks 0.222,
and the hinge offset is worse (0.108). Same picture as mlp1024 — the wide heads take the trunk for MA.
Single seed. The query line's best all-round configuration remains the 2-layer / 256 arm.
