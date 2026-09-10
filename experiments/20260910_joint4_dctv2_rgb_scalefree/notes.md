# 20260910_joint4_dctv2_rgb_scalefree — joint 2D+3D training with the DCT readout conventions v2

**Goal.** The joint4 recipe unchanged (SF3D + HOI4D + EPIC + ARCTIC, source-homogeneous batches,
hand sources x10, lr 2e-5, 20 ep, monitor `val/sf3d/loss_total`) with the DCT readout conventions
v2 (`knowledge/2026-09-10_trajectory_head_synthesis_v2.md`, #2): pinned start, shape/scale split;
SF3D batches get the 3D first-difference trio (0.5 / 0.25 / 0.25) + log path-length 0.5, the 2D
batches the uv-space trio + log path-length via the `2d` loss profile.

**Comparison row.** `20260910_joint4_dct_rgb_scalefree` best-epoch12: MA 31.41 / signed 30.15,
mIoU 0.2738, PDet 22.72, traj_dir 96.4, roughness 0.0090; HOI4D 0.676 / 85.2.

**Result (2026-09-10 07:00 local, pod J `segaffordance-dctv2-j`, 4 h 30 min train + 5 min tests, pod
deleted — verified).** Best `val/sf3d/loss_total` **1.1888 at epoch 13** of 20 (joint4: 0.9688 at
epoch 12 — totals not comparable, the v2 total carries the fdiff trio + log-length terms).

SF3D test (pred_z_p; the gt_z0 pass is identical on every metric, as always for scale-free arms):

| metric | joint4 (ep 12) | **joint4 v2 (ep 13)** | delta |
|---|---|---|---|
| MA (type+axis pass) | 31.41 | **32.94** | +1.5 — beats the DCT chain's 32.80 by a hair |
| MA signed | 30.15 | **32.08** | +1.9 (chain: 32.43) |
| matched axis err | 20.05° | **16.42°** | -3.6° |
| all-axis err / signed | 26.06 / 34.43 | **25.24 / 32.01** | better |
| axis flips all / rot | 11.87 / 13.58 | **10.83** / 15.17 | all better, rot worse |
| origin err / line | 0.327 / 0.297 | 0.355 / 0.320 | -3 cm |
| point 3D err | 0.285 | **0.279** | |
| mIoU / PDet | **0.2738 / 22.72** | 0.2495 / 19.06 | -0.024 / -3.7 |
| traj_dir acc / cos | **96.40 / 0.830** | 95.99 / 0.828 | |
| roughness | 0.0090 | 0.0096 | |

Per-source (single-source configs + the three DCT overrides):

| source | joint4 mIoU / PDet / point / shape / rough | **v2** mIoU / PDet / point / shape / rough |
|---|---|---|
| HOI4D | **0.676 / 85.2** / 0.0144 / 0.0343 / 0.0051 | 0.592 / 73.6 / 0.0175 / 0.0342 / 0.0107 |
| EPIC | 0.305 / 15.1 / 0.048 / **0.091** / 0.0026 | 0.300 / 15.1 / 0.044 / 0.109 / 0.0066 |
| ARCTIC | **0.618 / 68.7** / 0.062 / **0.083** / 0.0058 | 0.547 / 59.6 / 0.070 / 0.091 / 0.0142 |

**Reading.** (1) Articulation improves clearly: MA +1.5 (a new joint-recipe best and marginally the
best MA of any arm), matched axis -3.6° (the sharpest since the fdiff_dir record 14.62°), signed and
all-axis errors and all-flips down. The derivative terms sharpen the axis, exactly the g19_fdiff /
cf_h1only story (velocity supervision = the H1 quadratic that set the earlier records). (2) Masks and
detection REGRESS everywhere: SF3D mIoU -0.024 / PDet -3.7, HOI4D -0.08 / -12, ARCTIC -0.07 / -9.
Same as g19_fdiff's "mask/type dips" and the v2 2D arm: the trajectory-side terms pull the shared
trunk away from the mask/heatmap channels. The joint4 model's headline (best masks + hand sources
kept) is lost. (3) Origin +3 cm worse; rot flips worse (15.2 vs 13.6) while all-flips improve.
(4) Roughness on the hand sources doubles (0.005 -> 0.011): the pinned readout keeps the 6th AC
frequency and the uv derivative terms reward following the jittery WiLoR tracks; on SF3D it is flat.
(5) Best epoch moved 12 -> 13; no overfit signal.

**Verdict.** The v2 conventions are an ARTICULATION trade: +1.5 MA / -3.6° matched axis for -0.02
mIoU / -4 PDet and the hand-source masks. Which of the four ingredients carries which effect is not
separated here (pin + scale split + fdiff trio + log-length all changed at once); the fdiff-family
precedent says the derivative trio owns both the axis gain and the mask loss. Ablation candidates:
v2 head with the trio OFF (pin + scale split + log-length only), and the trio at 0.25/0.1/0.1.

