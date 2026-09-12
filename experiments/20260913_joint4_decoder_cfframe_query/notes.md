# 20260913_joint4_decoder_cfframe_query — joint decoder arm `cfframe_query`

**Goal.** query readout (4 queries x 2 layers) on the MA-record cf_frame 2:1 SF3D side — does the readout help the record recipe and fix its hand-video sign?

**Setup.** Derived from `config/joint4_decoder_cfframe.yaml` (exp `20260912_joint4_decoder_cfframe`) with overrides `model_params.articulation_readout=query model_params.readout_queries=4 model_params.readout_layers=2 model_params.readout_dim_ffn=1024 model_params.readout_mask_eps=0.01`; `config/joint4_decoder_cfframe_query.yaml`, `run_joint4dec_cfframe_query_chain.sh`, pod jdec-cfframe_query.

**Comparison rows.** `20260912_joint4_decoder_cfframe` (seed 42: MA 36.36 / 35.73, rot flips 13.4, origin 0.313, mIoU
0.252 / 20.0, HOI4D 0.578; ARCTIC probe axis 46.0, flips 48.3 %, offset 0.096) and the l2anchor query arm
(`20260913_joint4_decoder_l2anchor_query`: MA 35.75, ARCTIC flips 13.4 %).

**Result (2026-09-13 06:05 local, pod jdec-cfq, Server Edition, 4 h 20 min + tests, pod deleted by the watcher — verify).**
Best `val/sf3d/loss_total` 1.1911 at epoch 16.

| SF3D metric | cf_frame (no readout, s42) | **cf_frame + query** | l2anchor + query |
|---|---|---|---|
| MA / signed | **36.36 / 35.73** | 33.77 / 33.45 | 35.75 / 35.24 |
| type acc | 92.4 | 92.0 | 93.9 |
| matched / all / signed-all axis (deg) | 17.6 / 24.3 / 30.7 | 18.9 / 26.5 / 32.9 | 17.9 / 23.6 / 29.4 |
| flips all / rot (%) | 9.8 / 13.4 | 12.0 / 13.9 | 8.1 / 15.7 |
| origin / line (m) | 0.313 / — | **0.298 / 0.265** | 0.308 / 0.283 |
| radius / point3d (m) | 0.171 / — | 0.151 / 0.299 | 0.152 / 0.302 |
| mIoU / PDet | 0.252 / 20.0 | 0.256 / 21.0 | 0.270 / 23.35 |
| traj_dir acc | 92.4 | 92.3 | 93.5 |
| HOI4D / EPIC / ARCTIC mIoU | 0.578 / — / — | **0.641 / 0.391** / 0.578 | 0.625 / 0.386 / 0.588 |
| ARCTIC probe: axis / <20 / flips / offset | 46.0 / 23.4 % / 48.3 % / 0.096 | 40.6 / 20.7 % / 37.7 % / 0.095 | 41.2 / 20.1 % / 13.4 % / 0.087 |

**Reading.** On the cf_frame side the query readout COSTS SF3D articulation (MA -2.6, at ~2x the seed
noise; axis errors +1..2 deg) while buying masks on the hand sources (HOI4D 0.641 and EPIC 0.391, the
best of any joint arm) and 1.5 cm of origin. And it does NOT repair cf_frame's hand-video sign: ARCTIC
flips 38 % (cf_frame 48 %, l2anchor+query 13 %). So the readout interacts with the SF3D-side loss: with
the L2 quadratic + 1-cos anchor the spatial readout is worth +4.7 MA and halves ARCTIC flips; with the
scale-free cf_frame term it is worth -2.6 MA and the sign stays wrong. Hypothesis: cf_frame's axis term
already supplies what the readout adds on SF3D (a sharp, scale-free direction signal), so the extra
capacity goes to the mask/trunk trade instead, and the sign convention the 2D arc imposes stays
inconsistent with cf_frame's (object-dependent, see the notebook/laptop inversion). Decision: the
readout line continues on the l2anchor recipe (the user's final loss); cf_frame stays the SF3D-only MA
reference. Single seed.
