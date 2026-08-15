# Gen-10: Normalized, Full-Strength Consistency Loss (L_pp)

**Date:** 2026-08-15
**Status:** DRAFT — awaiting user review
**Goal:** Make the pred-pred consistency term bite as hard as the trajectory
and articulation losses (user decision), by (1) normalizing both residual
branches to dimensionless relative errors and (2) recalibrating the weight
from measured statistics — then rerun the gen-9 recipe unchanged otherwise.

## Motivation (measured on gen-9 best-epoch23)

- L_pp is the smallest term in the loss: raw 0.0015 train / 0.0045 val,
  ~0.2% of the total — 3.5× below even the trajectory MSE. Median
  per-sample value 0.00018; the inconsistent tail (val93/val1684, both
  ~p97 at 0.022–0.023) feels almost no pressure. Confirmed visually: the
  predicted trajectory cuts inside its own predicted orbit
  (viz/20260815_sf3d_g9_lpp_val93_3d).
- Absolute m² units also mean big-radius doors dominate small-radius ones
  for the same *relative* orbit violation.

## Loss change (`model/losses/geometric.py`, PredPredArticulationLoss)

New constructor params, default-off so every existing config/checkpoint
reproduces bit-identically:

```python
PredPredArticulationLoss(weight, degenerate_threshold=1e-6,
                         trajectory_is_absolute=False,
                         normalized=False, radius_floor=0.10)
```

With `normalized=True`, both branches become dimensionless relative errors
(decisions 2026-08-15: normalize BOTH branches, unit-consistent gate):

- **Prismatic:** `l_line_norm = l_line / mean‖dᵢ‖².clamp(min=1e-8)` — the
  fraction of the trajectory's displacement energy perpendicular to the
  axis. Intrinsically in [0, 1].
- **Revolute:** `l_circle_norm = (radial + axial) / max(r̂, radius_floor)²`
  — orbit residual relative to the predicted radius. `radius_floor = 0.10 m
  = min_revolute_radius`: GT radii below 0.10 m are filtered from the
  dataset, so a predicted r̂ below the floor is itself implausible and is
  measured against the 0.10 m scale instead of exploding the ratio (the
  unfloored p99/max on g9 were 1.48/8.48 with a 0.05 floor — spikes come
  precisely from tiny predicted radii).

Soft type gate, degenerate-energy masking, fp32 island, and the relative /
absolute trajectory handling are all unchanged. `build_geometric_loss`
forwards two new `loss_params` fields: `pred_pred_art_normalized` (default
False) and `pred_pred_art_radius_floor` (default 0.10).

## Weight calibration (measured, 512 val samples, g9 best ckpt)

Normalized per-sample L_pp on the converged gen-9 model (0.05 floor —
0.10-floor values will be slightly lower in the tail):
mean **0.163**, p50 0.037, p90 0.323, p99 1.48.

End-of-training weighted contributions to compare against: trajectory
0.0026, axis 0.006, origin 0.018 (type CE 0.12 is a different currency).

**`pred_pred_art_weight: 0.1`** → expected weighted contribution
≈ 0.1 × 0.16 ≈ **0.016** at g9-level residuals — top of the
trajectory/axis/origin band (≈ the origin term, 6× the trajectory term),
which is what "as important as trajectory and articulation" asks for
without letting one term dominate. Per-sample: val93 would contribute
0.021, a p99 sample ~0.15 — real pressure on the tail, bounded by the
radius floor.

Watch item (first epochs): at init the normalized residual is O(1), so the
term starts at ~0.1 contribution — comparable to the heatmap terms. The
direct losses still anchor every head and degenerate curves are masked, so
mutual collapse is not expected; if training destabilizes, the fallback is
a 5-epoch linear warmup on the weight (NOT in scope unless needed).

## Gen-10 run

`config/sf3d_train_runpod_g10_closeup010.yaml` = the gen-9 config with ONLY:

```yaml
loss_params:
  pred_pred_art_weight: 0.1        # calibrated (was 0.5 on the m² term)
  pred_pred_art_normalized: true   # relative-error branches
  pred_pred_art_radius_floor: 0.10 # = min_revolute_radius
```

plus experiment paths → `experiments/20260815_sf3d_g10_closeup010`. Same
split (59,174 records), 30 epochs, milestones [24, 28], seed 42, frozen
CLIP, RTX PRO 4500. Directly comparable to gen-9 AND to the ablation arms
(the entire gen-9 family shares the split).

## Success criteria & evaluation

1. Standard test pass (same 5,088 samples). Direct metrics must not
   regress materially vs gen-9 (type 95.0, axis 27.6°/15.8°, point3d
   0.292 m, origin 0.361 m, traj_dir 93.1%, mIoU 0.146); axis/origin/
   traj_dir improving is the hoped-for effect.
2. Consistency improvement is judged with `tools/diag_lpp_samples.py`
   (ref-512, same seed) on the g10 best checkpoint: normalized mean must
   drop well below g9's 0.163, and the p97-class samples (93, 1684)
   re-checked individually + re-rendered in 3D.
3. Viz batch on the ablation-panel seed for side-by-side panels vs gen-9.

## Code changes required

1. `model/losses/geometric.py`: the two params + normalized branches (one
   place — the existing forward, branching only on `self.normalized`).
2. `config/opd_train.py` LossParams: `pred_pred_art_normalized: bool =
   False`, `pred_pred_art_radius_floor: float = 0.10`.
3. `build_geometric_loss`: forward both.
4. `tools/diag_lpp_samples.py`: already computes the normalized breakdown;
   update its floor from 0.05 → read a `--radius-floor` arg (default 0.10)
   so probe and loss stay in lockstep.
5. Tests: normalized=False bit-identity with the old loss on random
   inputs; line branch ≤ 1; circle branch invariant to uniform scene
   scaling (s·traj, s·origin, s·point, r̂ above floor); floor engages for
   r̂ < 0.10; config test pinning the three new YAML values against the
   gen-9 base (constants unchanged).

## Out of scope

- Weight warmup schedule (fallback only, if instability is observed).
- Any other gen-9 recipe change; the ablation arms are not rerun.
