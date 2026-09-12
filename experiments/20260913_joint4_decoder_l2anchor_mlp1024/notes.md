# 20260913_joint4_decoder_l2anchor_mlp1024 — articulation readout arm `mlp1024`

**Goal.** WIDENED MLP control (classical mask-mean pooling, head hidden width 256 -> 1024 for the type/axis MLP, the depth heads and the arc-length head). Tests the head-bottleneck hypothesis (2026-09-12): the articulation heads read one
mask-mean pooled vector through a 256-wide MLP, which keeps nothing of WHERE the hinge sits.
Three arms in parallel on the user's final recipe (`config/joint4_decoder_l2anchor.yaml`, L2 2pi +
axis 0.5 on SF3D, analytic decoder, joint4 data): `query` (option 2), `attnpool` (option 1),
`mlp1024` (capacity control). Spec: docs/superpowers/specs/2026-09-12-articulation-readout-design.md.

**Comparison row.** `20260912_joint4_decoder_l2anchor` (MA 31.05 / 30.27, rot flips 19.0, origin 0.294,
mIoU 0.241 / PDet 18.5, HOI4D 0.576; ARCTIC probe: axis 48.2 deg, flips 26.4 %, hinge-line offset 0.070).

**Setup.** `config/joint4_decoder_l2anchor_mlp1024.yaml`, `run_joint4dec_l2anchor_mlp1024_chain.sh` (stage,
train 20 ep, best ckpt, SF3D test head + writer length, per-source tests), pod jdec-mlp1024.
Judged by SF3D origin / rot flips / MA and the ARCTIC hinge probe (`tools/arctic_axis_probe.py`).

**Result (2026-09-13 06:45 local, pod jdec-mlp1024 — second pod after a power-capped Workstation lemon (622 MHz at 597 W, 0.08 it/s) was swapped; 4 h 50 min + tests; pod deleted — verified).**
Best `val/sf3d/loss_total` 1.1247 at epoch 13.

| SF3D metric | l2anchor (base) | attnpool | query | **mlp1024 (control)** | dense |
|---|---|---|---|---|---|
| MA / signed | 31.05 / 30.27 | 33.08 / 32.41 | 35.75 / 35.24 | **37.70 / 36.87** | 44.01 / 42.69 |
| type acc | 92.4 | 92.2 | 93.9 | 91.1 | 96.0 |
| matched / all / signed-all (deg) | 18.9 / 24.3 / 34.0 | 17.9 / 25.5 / 32.3 | 17.9 / 23.6 / 29.4 | 18.2 / 24.2 / 31.2 | 11.4 / 19.6 / 25.0 |
| flips all / rot (%) | 12.1 / 19.0 | 10.1 / 17.4 | 8.1 / 15.7 | 11.3 / 14.8 | 7.6 / 12.7 |
| origin / line (m) | 0.294 / 0.257 | 0.345 / 0.314 | 0.308 / 0.283 | 0.301 / 0.269 | 0.248 / 0.220 |
| radius / point3d (m) | 0.135 / 0.299 | 0.168 / 0.308 | 0.152 / 0.302 | 0.157 / **0.262** | 0.124 / 0.310 |
| mIoU / PDet | 0.241 / 18.5 | 0.273 / 22.0 | 0.270 / 23.35 | 0.239 / 20.1 | 0.2765 / 22.5 |
| traj_dir acc | 89.6 | 91.5 | 93.5 | 90.8 | 94.8 |
| HOI4D / EPIC / ARCTIC mIoU | 0.576 / 0.288 / 0.553 | 0.622 / 0.312 / 0.594 | 0.625 / 0.386 / 0.588 | 0.584 / 0.279 / 0.527 | 0.508 / 0.186 / 0.509 |
| ARCTIC probe: axis / <20 / flips / offset | 48.2 / 18.8 % / 26.4 % / 0.070 | 49.3 / 12.2 % / 34.0 % / 0.082 | 41.2 / 20.1 % / 13.4 % / 0.087 | 44.4 / 17.6 % / **65.7 %** / 0.073 | 45.8 / 16.1 % / 30.4 % / 0.112 |

**Reading.** The capacity control is NOT null: widening the heads 256 -> 1024 on the same pooled vector is
+6.6 MA (37.70, above the previous cf_frame record 36.73) with the mean axis errors unchanged (24.2 /
18.2 deg) — the extra width tightens the under-10-deg tail rather than the mean — while masks stay at the
base's low level (0.239 / 20.1) and the hand sources do not improve. Its ARCTIC probe is the worst sign
transfer of any arm (66 % flips: the wide MLP learns SF3D's sign convention and inverts it on hand
video). So the readout ranking on SF3D MA is base 31 < attnpool 33 < query 35.8 < mlp1024 37.7 < dense
44.0 — capacity buys MA, but only the SPATIAL readouts (query, dense) also buy type, flips, masks, PDet,
traj_dir and hand-video sign; the widened MLP buys MA alone and loses sign transfer. Reads with the
+-1 MA noise; the query seed-7 replicate calibrates the readout arms. The 256 width is clearly too
small for the pooled vector (a 1024-wide query readout is a cheap follow-up).
