# 20260822_sf3d_g17_2d_fdiff2d — 2D arm + uv-space first-difference losses

**Recipe:** the arm-A-detach 2D-only config
(`20260820_sf3d_g17_2d_detach`) + the g19 fdiff losses ported into uv
space inside `TrajectoryProjectionLoss` (`proj_fdiff_velocity_weight 1.0,
proj_fdiff_angle_weight 0.5, proj_fdiff_length_weight 0.5` on diffs of the
projected trajectory vs the GT track, both-endpoints-valid segments).
Question: does the direction/velocity supervision that set articulation
records on the 3D path (g19_fdiff) transfer to the 2D-only path?
30 epochs, best = epoch 10 (val 1.8296) — val loss ROSE after epoch 10
(1.83 → 2.06 by ep29), unlike arm A (best ep24): the fdiff terms
destabilize late training on the 2D path.

## Test (5,088) vs arm A (2D baseline) / g17 (3D reference)

| metric | arm A (detach) | **+uv-fdiff** | g17 (3D) |
|---|---|---|---|
| proj2d total / anchor / shape | 0.175 / 0.129 / **0.0997** | 0.170 / 0.134 / 0.107 | 0.155 / 0.09 / 0.11 |
| mIoU / PDet | **0.242** / 16.1 | 0.237 / 15.8 | 0.265 / 20.6 |
| 2D point | 0.117 | **0.108** | 0.095 |
| traj_dir acc / cos | 87.6 / 0.524 | **89.9 / 0.563** | 94.9 / 0.811 |
| traj roughness (m) | — (predates metric) | 0.177 | 0.091 |
| axis (all) | 63.5° | 62.1° | 25.8° |

## Reading

**A wash — not adopted for the 2D path.** The one real gain is trajectory
direction (+2.3 acc / +0.04 cos), the same signature fdiff showed on 3D —
direction is loss-driven. But it costs a little everywhere else (shape
0.107 vs 0.0997, mIoU 0.237 vs 0.242), the val curve turns over at epoch
10, and 3D-space roughness is very high (0.177, ~2× the 3D baseline —
uv-space diffs are translation-invariant per frame and put no constraint
on the depth profile). Articulation stays deadlocked, as expected.

Decision input for the label-efficiency p90 pretrain recipe: uv-fdiff is
OUT unless the DCT 2D arm's numbers change the picture.

test pass: `logs/test.log` (ckpt best-epoch10-valloss1.8296)
spec: docs/superpowers/specs/2026-08-21-smooth-trajectory-g19-design.md (port section)
