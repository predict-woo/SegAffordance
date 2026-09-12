# 20260913_joint4_decoder_l2anchor_query_eps01 — joint decoder arm `l2anchor_query_eps01`

**Goal.** query readout with mask bias floor eps 0.1 (log 0.1 = -2.3 vs -4.6: the surroundings of the part — seam, frame — reachable for the origin query; placement hypothesis)

**Setup.** Derived from `config/joint4_decoder_l2anchor_query.yaml` (exp `20260913_joint4_decoder_l2anchor_query`) with overrides `model_params.readout_mask_eps=0.1`; `config/joint4_decoder_l2anchor_query_eps01.yaml`, `run_joint4dec_l2anchor_query_eps01_chain.sh`, pod jdec-l2anchor_query_eps01.

**Result (2026-09-13 11:20 local, pod jdec-qeps01, Server Edition, pod deleted — verified).** Best val 1.1076 at epoch 13.

| SF3D metric | query eps 0.01 (s42 / s7) | **query eps 0.1** |
|---|---|---|
| MA / signed | 35.75 / 35.24 ; 33.57 / 33.29 | **38.11 / 37.64** |
| type acc | 93.9 / 93.2 | 94.0 |
| matched / all / signed-all | 17.9 / 23.6 / 29.4 ; 19.0 / 24.8 / 30.6 | 16.7 / 22.8 / 30.0 |
| flips all / rot | 8.1 / 15.7 ; 7.4 / 7.3 | 9.3 / 14.9 |
| origin / radius / point3d | 0.308 / 0.152 / 0.302 ; 0.300 / 0.139 / 0.308 | 0.322 / 0.172 / **0.278** |
| mIoU / PDet | 0.270 / 23.35 ; 0.268 / 23.2 | **0.2763** / 22.3 |
| HOI4D / EPIC / ARCTIC | 0.625 / 0.386 / 0.588 ; 0.614 / 0.330 / 0.582 | 0.594 / 0.241 / 0.534 |
| ARCTIC probe axis / flips / offset | 41.2 / 13 % / 0.087 ; 42.4 / 72 % / 0.083 | 48.2 / 59 % / **0.159** |

**Reading.** Letting the queries see the surroundings (bias floor log 0.1 instead of log 0.01) is the
best SF3D MA of the query family (38.1) with record-level SF3D masks (0.276) and the best 3D point
(0.278) — but the PLACEMENT hypothesis is refuted: SF3D origin worse (0.322), radius worse (0.172) and
the ARCTIC hinge-line offset is the worst of all arms (0.159). More context helps the trunk-level
quantities (masks, point, MA) and hurts the hinge; the origin needs geometry (votes / 3D labels), not
a wider receptive field. Single seed.
