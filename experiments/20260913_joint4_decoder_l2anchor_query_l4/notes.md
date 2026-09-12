# 20260913_joint4_decoder_l2anchor_query_l4 — joint decoder arm `l2anchor_query_l4`

**Goal.** query readout with 4 layers instead of 2 (depth of the spatial readout)

**Setup.** Derived from `config/joint4_decoder_l2anchor_query.yaml` (exp `20260913_joint4_decoder_l2anchor_query`) with overrides `model_params.readout_layers=4`; `config/joint4_decoder_l2anchor_query_l4.yaml`, `run_joint4dec_l2anchor_query_l4_chain.sh`, pod jdec-l2anchor_query_l4.

**Result (2026-09-13 11:20 local, pod jdec-ql4, Workstation Edition at full clocks, pod deleted — verified).** Best val 1.1337 at epoch 16.

| SF3D metric | query (2 layers, s42 / s7) | **query 4 layers** |
|---|---|---|
| MA / signed | 35.75 / 35.24 ; 33.57 / 33.29 | **37.17 / 36.30** |
| type acc | 93.9 / 93.2 | **95.3** |
| matched / all / signed-all | 17.9 / 23.6 / 29.4 ; 19.0 / 24.8 / 30.6 | **15.3 / 23.3 / 30.1** |
| flips all / rot | 8.1 / 15.7 ; 7.4 / 7.3 | 10.3 / 15.8 |
| origin / radius / point3d | 0.308 / 0.152 / 0.302 ; 0.300 / 0.139 / 0.308 | 0.300 / 0.135 / **0.292** |
| mIoU / PDet | 0.270 / 23.35 ; 0.268 / 23.2 | 0.269 / 22.1 |
| HOI4D / EPIC / ARCTIC | 0.625 / 0.386 / 0.588 ; 0.614 / 0.330 / 0.582 | 0.599 / 0.250 / 0.587 |
| ARCTIC probe axis / flips / offset | 41.2 / 13 % / 0.087 ; 42.4 / 72 % / 0.083 | 43.4 / 11.9 % / 0.112 |

**Reading.** Depth helps the readout on SF3D: +1.4..3.6 MA over the two 2-layer seeds, matched axis
15.3 deg (best of the query family), type 95.3, point3d 0.292 — at the cost of EPIC masks (0.250) and a
worse ARCTIC hinge offset (0.112). Single seed; the gain is inside the readout family's seed spread
(33.6-35.8), so read it as 'not worse, likely a little better on articulation'.
