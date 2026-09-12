# 20260913_joint4_decoder_l2anchor_attnpool — articulation readout arm `attnpool`

**Goal.** articulation readout = ATTNPOOL (one learned query attention-pools the decoded map in place of the mask mean; heads unchanged). Tests the head-bottleneck hypothesis (2026-09-12): the articulation heads read one
mask-mean pooled vector through a 256-wide MLP, which keeps nothing of WHERE the hinge sits.
Three arms in parallel on the user's final recipe (`config/joint4_decoder_l2anchor.yaml`, L2 2pi +
axis 0.5 on SF3D, analytic decoder, joint4 data): `query` (option 2), `attnpool` (option 1),
`mlp1024` (capacity control). Spec: docs/superpowers/specs/2026-09-12-articulation-readout-design.md.

**Comparison row.** `20260912_joint4_decoder_l2anchor` (MA 31.05 / 30.27, rot flips 19.0, origin 0.294,
mIoU 0.241 / PDet 18.5, HOI4D 0.576; ARCTIC probe: axis 48.2 deg, flips 26.4 %, hinge-line offset 0.070).

**Setup.** `config/joint4_decoder_l2anchor_attnpool.yaml`, `run_joint4dec_l2anchor_attnpool_chain.sh` (stage,
train 20 ep, best ckpt, SF3D test head + writer length, per-source tests), pod jdec-attnpool.
Judged by SF3D origin / rot flips / MA and the ARCTIC hinge probe (`tools/arctic_axis_probe.py`).

**Result (2026-09-13 05:23 local, pod jdec-attnpool, Server Edition, 4 h 20 min + tests, pod deleted by the watcher — verify).**
Best `val/sf3d/loss_total` 1.1630 at epoch 19 (last epoch — still improving).

| SF3D metric | l2anchor (base) | **attnpool** |
|---|---|---|
| MA / signed | 31.05 / 30.27 | **33.08 / 32.41** |
| type acc | 92.4 | 92.2 |
| matched / all / signed-all axis (deg) | 18.9 / 24.3 / 34.0 | 17.9 / 25.5 / 32.3 |
| flips all / rot (%) | 12.1 / 19.0 | 10.1 / 17.4 |
| origin / line (m) | **0.294 / 0.257** | 0.345 / 0.314 |
| radius / point3d (m) | 0.135 / 0.299 | 0.168 / 0.308 |
| mIoU / PDet / point2d | 0.241 / 18.5 / 0.118 | **0.273 / 22.0** / 0.111 |
| traj_dir acc / cos | 89.6 / 0.732 | 91.5 / 0.740 |
| HOI4D / EPIC / ARCTIC mIoU | 0.576 / 0.288 / 0.553 | 0.622 / 0.312 / 0.594 |
| ARCTIC probe: axis mean / <20 deg / flips / hinge offset | 48.2 / 18.8 % / 26.4 % / 0.070 | 49.3 / 12.2 % / 34.0 % / 0.082 |

**Reading.** Attention pooling in place of the mask mean is +2.0 MA (above the +-1 noise), takes SF3D masks
back to the base decoder's level (0.273 / 22.0 vs 0.241 / 18.5 — the l2anchor mask cost is gone) and
lifts every hand-source mask (+0.03..0.05), with slightly fewer flips. It does NOT help hinge placement:
SF3D origin +5 cm, radius +3 cm, and the ARCTIC hinge-line offset is worse (0.082 vs 0.070) with more
flips there (34 %). One learned query is a better feature selector for the trunk (masks, MA) but still
hands the heads a single vector — the origin bottleneck is untouched. ARCTIC per object: notebooks and
scissors almost never flip (3 / 5 %), laptops / phones / espresso 39-62 %. Single seed.
