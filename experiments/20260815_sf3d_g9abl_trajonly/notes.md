# 20260815_sf3d_g9abl_trajonly — supervision ablation arm C: TRAJECTORY ONLY

**Recipe:** gen-9 minus every articulation path: `use_motion_head:
false`, `use_motion_type_head: false` (MotionMLP skipped), `use_origin_heatmap:
false` (2-channel projector, no z_q lift, condition = [features, point_uv]),
all articulation loss weights 0, `geometric_loss: "none"`. Mask +
interaction point (heatmap + z_p) + 20-point relative trajectory remain.
Same split/schedule/seed as arms A and B. Spec:
`docs/superpowers/specs/2026-08-15-supervision-ablation-design.md`.

**Run:** ~58 min (3.80 it/s — fastest arm, fewest heads), best = epoch 22
(val 0.4090 — NOT comparable across arms). Clean exit. Cost ≈ $0.9.

## Test (best-epoch22, same 5,088 samples)

Articulation metrics correctly ABSENT (type, axis, origin-head columns all
skipped). `test/mean_origin_error_m` present but it is the LEGACY
pseudo-origin (point_uv + depth patch), not an origin-head output.

| metric | arm C | arm A (joint) | arm B (art-only) |
|---|---|---|---|
| mIoU | **0.1566** | 0.1463 | 0.1479 |
| PDet | **7.11%** | 5.39% | 6.03% |
| 2D point | 0.1590 | **0.1525** | 0.1619 |
| 3D point (m) | 0.3029 | **0.2920** | 0.3082 |
| traj_dir acc | 87.58% | **93.10%** | — |
| traj_dir cos | 0.631 | **0.736** | — |
| legacy pseudo-origin (m) | 1.0936 | 1.0880 | 1.0783 |

## Reading

- **Removing articulation supervision clearly HURTS trajectory
  prediction:** traj_dir 87.6% vs 93.1% joint (−5.5 pts), cos 0.631 vs
  0.736. Knowing the motion type and axis constrains what curve to sweep —
  the largest single effect in the ablation.
- **Masks got slightly BETTER with fewer tasks** (mIoU 0.157, PDet 7.1 —
  best of the three arms): mild task competition on the shared
  decoder/projector. The effect is small next to the trajectory/type gains.
- Point localization ≈ flat (3D point 0.30 vs 0.29) — the point pipeline is
  supervised identically in all arms.
