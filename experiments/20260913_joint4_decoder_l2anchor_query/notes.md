# 20260913_joint4_decoder_l2anchor_query — articulation readout arm `query`

**Goal.** articulation readout = QUERY (4 learned queries, 2 pre-norm layers, mask-biased cross-attention over the decoded map; one query per head). Tests the head-bottleneck hypothesis (2026-09-12): the articulation heads read one
mask-mean pooled vector through a 256-wide MLP, which keeps nothing of WHERE the hinge sits.
Three arms in parallel on the user's final recipe (`config/joint4_decoder_l2anchor.yaml`, L2 2pi +
axis 0.5 on SF3D, analytic decoder, joint4 data): `query` (option 2), `attnpool` (option 1),
`mlp1024` (capacity control). Spec: docs/superpowers/specs/2026-09-12-articulation-readout-design.md.

**Comparison row.** `20260912_joint4_decoder_l2anchor` (MA 31.05 / 30.27, rot flips 19.0, origin 0.294,
mIoU 0.241 / PDet 18.5, HOI4D 0.576; ARCTIC probe: axis 48.2 deg, flips 26.4 %, hinge-line offset 0.070).

**Setup.** `config/joint4_decoder_l2anchor_query.yaml`, `run_joint4dec_l2anchor_query_chain.sh` (stage,
train 20 ep, best ckpt, SF3D test head + writer length, per-source tests), pod jdec-query.
Judged by SF3D origin / rot flips / MA and the ARCTIC hinge probe (`tools/arctic_axis_probe.py`).

**Result (2026-09-13 06:00 local, pod jdec-query, Server Edition, 4 h 25 min + tests, pod deleted — verified).**
Best `val/sf3d/loss_total` 1.1168 at epoch 16.

| SF3D metric | l2anchor (base) | attnpool | **query** |
|---|---|---|---|
| MA / signed | 31.05 / 30.27 | 33.08 / 32.41 | **35.75 / 35.24** |
| type acc | 92.4 | 92.2 | **93.9** |
| matched / all / signed-all axis (deg) | 18.9 / 24.3 / 34.0 | 17.9 / 25.5 / 32.3 | 17.9 / **23.6** / **29.4** |
| flips all / rot (%) | 12.1 / 19.0 | 10.1 / 17.4 | **8.1 / 15.7** |
| origin / line (m) | **0.294** / 0.257 | 0.345 / 0.314 | 0.308 / 0.283 |
| radius / point3d (m) | 0.135 / 0.299 | 0.168 / 0.308 | 0.152 / 0.302 |
| mIoU / PDet / point2d | 0.241 / 18.5 / 0.118 | 0.273 / 22.0 / 0.111 | 0.270 / **23.35** / 0.117 |
| traj_dir acc / cos | 89.6 / 0.732 | 91.5 / 0.740 | **93.5 / 0.776** |
| HOI4D / EPIC / ARCTIC mIoU | 0.576 / 0.288 / 0.553 | 0.622 / 0.312 / 0.594 | 0.625 / **0.386** / 0.588 |
| ARCTIC probe: axis mean / <20 deg / flips / hinge offset | 48.2 / 18.8 % / 26.4 % / **0.070** | 49.3 / 12.2 % / 34.0 % / 0.082 | **41.2 / 20.1 % / 13.4 %** / 0.087 |

**Reading.** The query readout is the arm that works: **+4.7 MA over the same recipe** (35.75, within
noise of the cf_frame record band 36.0-36.7 — reached with the L2 2pi + axis loss the user chose for
hand-video placement), best type / all-axis / signed-all / flips of the l2anchor family, PDet 23.35
(record band), traj_dir 93.5 (best decoder arm), EPIC masks 0.386 (best of ANY joint arm; joint4 0.305),
and on ARCTIC the sign problem halves (flips 26 -> 13 %, axis 48 -> 41 deg; espresso and laptop 0 %
flips). What it does not do: hinge PLACEMENT — SF3D origin 0.308 (base 0.294, noise-level), ARCTIC
hinge-line offset 0.087 (base 0.070), radius still ~0.9 m on hand video. Giving the heads spatial
access fixes what a pooled vector loses about sign and type; where the hinge sits still has no
supervision on the 2D side. Single seed (+-1 MA noise); the seed-7 replicate is the obvious check.
