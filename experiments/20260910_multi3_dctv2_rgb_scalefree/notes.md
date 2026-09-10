# 20260910_multi3_dctv2_rgb_scalefree — multi-source 2D DCT arm with the DCT readout conventions v2

**Goal.** Same recipe as `20260910_multi3_dct_rgb_scalefree` (HOI4D v2 + EPIC v1 + ARCTIC v1,
RGB-only scale-free, DCT-6 head, x8 augmentation, 12 ep) with the four readout fixes from the
2026-09-10 literature sweep (`knowledge/2026-09-10_trajectory_head_synthesis_v2.md`, #2):
pinned first point (AC-only coefficients), shape/scale split (unit-path-length curve x softplus
scale), the uv-space first-difference trio at half the g19 weights (velocity 0.5 / angle 0.25 /
length 0.25) and a log path-length loss (0.5). Stage 1 of the v2 chain; stage 2 =
`20260910_sf3d_g19_dctv2_rgb_scalefree_ft_multi3dctv2`.

**Comparison row.** multi3 DCT best-epoch11 val 0.4161; per-source held-out HOI4D / EPIC / ARCTIC
in its notes.

**Result (2026-09-10 03:30 local, pod M, 66 min).** Best val/loss_total **0.7640 at epoch 7** of 12
(the total now carries the uv fdiff trio + log-length term, so it is not comparable to the DCT arm's
0.4161; the projection term alone is 0.508 at the best epoch vs 0.516 for the DCT arm at its best —
parity on the data term; val L_proj_scale falls 0.46 -> 0.35, val L_proj_fdiff_vel 0.0149 -> 0.0111).

| held-out split | n | DCT arm (ep 11/12) mIoU / PDet / point / shape / rough | **DCT v2 (ep 7/12)** mIoU / PDet / point / shape / rough |
|---|---|---|---|
| HOI4D | 459 | 0.753 / 89.8 / 0.0137 / 0.0337 / 0.0072 | 0.692 / 84.3 / 0.0150 / **0.0325** / 0.0121 |
| EPIC | 53 | 0.520 / 58.5 / 0.053 / 0.090 / 0.0062 | 0.461 / 43.4 / 0.047 / 0.094 / 0.0133 |
| ARCTIC | 329 | 0.694 / 79.6 / 0.062 / 0.085 / 0.0053 | 0.672 / 76.6 / 0.058 / **0.0805** / 0.0116 |
| union | 841 | 0.715 / 83.8 / 0.035 / 0.057 / 0.0064 | 0.670 / 78.7 / 0.034 / **0.0551** / 0.0120 |

**Reading.** (1) The 2D track fit (shape) improves slightly on HOI4D / ARCTIC / union — the pinned,
scale-split, derivative-supervised readout fits the tracks a little better. (2) Masks and detection
REGRESS on every source (union mIoU -0.045, PDet -5): the same pattern as `20260821_sf3d_g19_fdiff`
(fdiff on: mask/type dips) — the extra trajectory-side terms pull the shared trunk. Best-epoch
selection also moved (7 vs 11) because the new terms now dominate val/loss_total. (3) Roughness
0.012: 2x the DCT arm (still 6x better than plain). Two contributions: the pinned readout keeps the
6th AC frequency (DC + 5 AC before), and the uv velocity/angle terms reward following the jittery
WiLoR tracks. (4) EPIC is 53 records — noise territory. The number that matters is the SF3D
post-training (`20260910_sf3d_g19_dctv2_rgb_scalefree_ft_multi3dctv2`) and the joint run.

