# 20260815_sf3d_g9abl_artonly — supervision ablation arm B: ARTICULATION ONLY

**Recipe:** gen-9 (`20260815_sf3d_g9_closeup010`) minus the trajectory path:
`use_trajectory_head: false`, `trajectory_weight: 0`, `geometric_loss:
"none"` (L_pp dies with the trajectory). Everything else identical — same
59,174-record split, cache, 30 epochs, milestones [24, 28], seed 42, frozen
CLIP, RTX PRO 4500. Spec:
`docs/superpowers/specs/2026-08-15-supervision-ablation-design.md`.

**Run:** ~1h (3.66–3.71 it/s, slightly faster than arm A — fewer heads),
best = epoch 21 (val 0.9491 — NOT comparable to other arms' val losses:
different term sets). Clean exit. Cost ≈ $0.9.

## Test (best-epoch21, same 5,088 samples as arm A)

Trajectory metrics correctly ABSENT (head removed; skip-guards verified in
production). Δ = arm B − arm A (gen-9 joint); negative Δ on errors = B
better.

| metric | arm B | arm A (joint) | Δ |
|---|---|---|---|
| mIoU | 0.1479 | 0.1463 | +0.002 |
| PDet | 6.03% | 5.39% | +0.6 |
| type acc | 92.00% | 94.97% | **−3.0** |
| axis° all | 27.42 | 27.56 | −0.1 |
| axis° matched | 20.39 | 15.81 | **+4.6** |
| MA pass | 21.74 | 21.91 | −0.2 |
| 2D point | 0.1619 | 0.1525 | +0.009 |
| 3D point (m) | 0.3082 | 0.2920 | +0.016 |
| origin vs q* (m) | 0.3532 | 0.3614 | −0.008 |
| origin line (m) | 0.3176 | 0.3220 | −0.004 |
| radius (m) | 0.1947 | 0.1914 | +0.003 |
| legacy pseudo-origin (m) | 1.0783 | 1.0880 | −0.010 |

## Reading (preliminary — full 3-way table after arm C)

- **Removing trajectory supervision HURTS articulation semantics:** type
  −3.0 pts and matched-axis +4.6° — the strongest joint-training effects.
  Trajectory supervision apparently regularizes the type/axis heads (a
  swept curve implies both the type and the axis direction).
- Overall-axis, origin, radius: flat (≤0.14° / ≤1 cm) — origin estimation
  gains nothing from trajectories.
- Masks/points: flat to trivially different — as expected for shared heads.
- Attribution caveat: arm A also has L_pp; these deltas bundle co-training
  + consistency coupling (spec's accepted limitation).

vis: viz/20260815_sf3d_g9_ablation_panels
