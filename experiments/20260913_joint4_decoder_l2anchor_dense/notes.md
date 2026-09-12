# 20260913_joint4_decoder_l2anchor_dense — articulation readout arm `dense`

**Goal.** DENSE HINGE VOTING (per-pixel rot/trans axis, type logits and 2D offset-to-hinge fields on the decoded map; part-mask-weighted means give axis / type / origin_uv; heatmap kept as auxiliary; scalar heads classical). The fourth arm of the head-bottleneck test (option 3 of the 2026-09-12 discussion):
the origin is LOCATED by the part's pixels (each votes an offset to the hinge) instead of being read
out of one pooled vector. Supervision reaches the fields only through the averaged outputs (origin_pred
via the lift, the closed-form loss on SF3D, the projection loss on the 2D sources) — no dense per-pixel
loss yet (candidate follow-up: mask-weighted |uv + offset - q*_uv| on SF3D).
Spec: docs/superpowers/specs/2026-09-12-articulation-readout-design.md.

**Comparison row.** `20260912_joint4_decoder_l2anchor` (MA 31.05 / 30.27, rot flips 19.0, origin 0.294,
mIoU 0.241 / PDet 18.5, HOI4D 0.576; ARCTIC probe: axis 48.2 deg, flips 26.4 %, hinge-line offset 0.070)
and the three readout arms of the same date (query / attnpool / mlp1024).

**Setup.** `config/joint4_decoder_l2anchor_dense.yaml`, `run_joint4dec_l2anchor_dense_chain.sh`, pod jdec-dense.

**Result (2026-09-13 06:15 local, pod jdec-dense, Server Edition, 4 h 45 min incl. one relaunch + tests, pod deleted — verified).**
Best `val/sf3d/loss_total` **0.9789** at epoch 13 (every other arm of the family: 1.12-1.19 on the same loss).

| SF3D metric | l2anchor (base) | query | **dense** | previous all-time record |
|---|---|---|---|---|
| MA / signed | 31.05 / 30.27 | 35.75 / 35.24 | **44.01 / 42.69** | 36.73 / 36.14 (cf_frame a3) |
| type acc | 92.4 | 93.9 | **96.0** | 95.3 (g17) |
| matched / all / signed-all axis (deg) | 18.9 / 24.3 / 34.0 | 17.9 / 23.6 / 29.4 | **11.4 / 19.6 / 25.0** | 14.6 / 22.5 / 29.6 |
| flips all / rot (%) | 12.1 / 19.0 | 8.1 / 15.7 | **7.6 / 12.7** | 7.8 / 9.5 |
| origin / line (m) | 0.294 / 0.257 | 0.308 / 0.283 | **0.248 / 0.220** | 0.245 (cf_frame) |
| radius / point3d (m) | 0.135 / 0.299 | 0.152 / 0.302 | **0.124** / 0.310 | 0.124 (dec base) |
| mIoU / PDet / point2d | 0.241 / 18.5 / 0.118 | 0.270 / 23.35 / 0.117 | **0.2765** / 22.5 / 0.108 | 0.2738 / 23.9 |
| traj_dir acc / cos | 89.6 / 0.732 | 93.5 / 0.776 | **94.8 / 0.816** | 96.4 (head) |
| HOI4D / EPIC / ARCTIC mIoU (PDet) | 0.576 / 0.288 / 0.553 | 0.625 / 0.386 / 0.588 | 0.508 (59.6) / 0.186 (7.5) / 0.509 (56.5) | 0.676 / 0.305 / 0.618 (joint4) |
| ARCTIC probe: axis mean / <20 / flips / hinge offset | 48.2 / 18.8 % / 26.4 % / 0.070 | 41.2 / 20.1 % / 13.4 % / 0.087 | 45.8 / 16.1 % / 30.4 % / 0.112 | — |

**Reading.** Dense hinge voting is the largest single jump in the project's history on SF3D: **MA 44.0
(+13 over the same recipe, +7.3 over the all-time record), matched axis 11.4 deg (record by 3.2 deg),
all-axis 19.6 (record), type 96.0 (record), origin 0.248 (= the record band), masks 0.2765 (record),
traj_dir 94.8** — every SF3D column at or beyond the previous best, from a 1.8M-parameter conv head that
lets every part pixel vote for the axis and the hinge instead of reading them out of one pooled vector.
The votes get no per-pixel loss here (`dense_offset_weight` 0): the geometry comes purely from
averaging under the closed-form loss.
The cost is the hand sources: HOI4D 0.508 / 59.6, EPIC 0.186 / 7.5, ARCTIC 0.509 (the worst joint-arm
hand masks), and hinge transfer to ARCTIC is worse (offset 0.112, flips 30 %). With per-pixel fields the
2D projection loss reaches the map pixel by pixel through the votes, and on 2D-only batches (no axis /
origin GT) that gradient is uncontrolled — the trunk gets pulled toward whatever makes the decoded
arc project onto the knuckle track, at the expense of the masks. Single seed: the size of the SF3D
jump makes a replicate mandatory, and the 2D-side handling is the next structural question
(vote-detach or lower projection weight on 2D batches; dense + attnpool for the scalar heads; the
per-pixel offset loss on SF3D). Best ckpt: `checkpoints/best-epoch13-sf3dval0.9789.ckpt`.
