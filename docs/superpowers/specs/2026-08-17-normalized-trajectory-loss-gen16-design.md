# Gen-16: GT-Energy-Normalized Trajectory Loss (fix the rot-sweep collapse)

**Date:** 2026-08-17
**Status:** IMPLEMENTED (commits ce41a7a, 939eddd; empirical collapse test pending)
**Goal:** Stop revolute trajectory collapse (predicted rot sweeps are 4–8 cm
against ~0.7 m GT arcs in every trajectory_weight=0.15 arm) by making the
trajectory loss a per-row RELATIVE error — normalized by each row's GT
sweep energy — and restoring the weight to 0.5. One flag, one run on the
gen-13 base.

## Diagnosis being fixed (measured 2026-08-17)

Predicted net rot sweep, val93 / val1684 (GT ≈ 0.7 m):
gen-11 (w=0.5): 0.27 / 0.84 m → every w=0.15 arm (g11b, g12–g15):
0.04–0.08 m. The collapse started exactly at the 0.5→0.15 rebalance.

Two compounding causes:
1. The 0.15 calibration used the trans-dominated aggregate; rot GT never
   changed v2→v3, so rot rows just lost 3.3× supervision pressure.
2. The normalized L_pp circle residual exerts ZERO restoring force on
   sweep extent — a curve shrunk to the anchor has radial = axial = 0 by
   construction (r̂ is the anchor's own axis distance) and a fully
   degenerate curve is energy-masked out — while any axis/origin error
   makes LONGER sweeps costlier. Shrinking rot sweeps is a free descent
   direction once the MSE counterweight weakened. (The line branch is
   scale-invariant, so trans has no such incentive — hence long drawer
   rays next to collapsed door arcs.)

## Change: `trajectory_loss_normalized` (LossParams, default False)

`train_OPDReal_better.py`, the relative branch of the trajectory loss
(~line 434; absolute mode is out of scope — no current arm uses it, and
the flag combination raises to avoid silent nonsense):

```python
                trajectory_gt_relative = (
                    trajectory_gt_device - trajectory_gt_device[:, 0:1, :]
                )
                if getattr(self.loss_params, "trajectory_loss_normalized", False):
                    # Gen-16: per-row RELATIVE error. Each row's squared
                    # error is divided by that row's GT sweep energy, so a
                    # collapsed prediction scores exactly 1.0 and rot/trans
                    # rows exert identical pressure regardless of sweep
                    # scale (the same philosophy as the normalized L_pp;
                    # fixes the rot-sweep collapse of the w=0.15 arms).
                    # eps = (1 cm)^2: degenerate GT stubs are damped, not
                    # amplified.
                    err = (outputs.trajectory_pred - trajectory_gt_relative)
                    per_row_mse = err.pow(2).sum(-1).mean(-1)          # (B,)
                    gt_energy = trajectory_gt_relative.pow(2).sum(-1).mean(-1)
                    L_trajectory = (
                        per_row_mse / gt_energy.clamp(min=1e-4)
                    ).mean()
                    # Continuity: the absolute-m^2 value the family logged.
                    self.log(f"{step_type}/L_trajectory_m2",
                             per_row_mse.detach().mean(), ...)
                else:
                    L_trajectory = self.trajectory_loss_fn(
                        outputs.trajectory_pred, trajectory_gt_relative
                    )
```

(Exact logging kwargs mirror the neighboring `self.log` calls. The
flag-off path is byte-identical. Guard in `__init__`:
`trajectory_loss_normalized` + `trajectory_absolute` → ValueError.)

Properties:
- A fully collapsed prediction has relative error exactly 1.0 → strong,
  scale-free restoring gradient on every row, rot and trans alike.
- Trans rows can no longer dominate the term through sheer GT magnitude —
  the gen-11 "enlarged trajectory term taxes the shared decoder" problem
  (which motivated 0.15) is solved structurally, so the weight returns to
  **`trajectory_weight: 0.5`** on the normalized term. Expected early-
  training contribution ≈ 0.5 (relative error ~1 at init) — the same O(1)
  band as the heatmap terms; the L_pp precedent showed this is stable.
- fp32/fp16: the division mirrors the L_pp pattern (bounded ratios); no
  autocast island needed (values O(1)).

## Gen-16 run

`config/sf3d_train_runpod_g16_trajnorm.yaml` = **gen-13** config plus ONLY:

```yaml
loss_params:
  trajectory_weight: 0.5           # restored on the normalized term
  trajectory_loss_normalized: true
```

paths → `experiments/20260817_sf3d_g16_trajnorm`. Same v3 data, 512 input,
512 frame cache, no taps, no cost map, 30 epochs, seed 42, 6000-class pod.
(The g13-without-taps base is deliberate; the cost-map-without-taps arm
stays a separate parked candidate.)

## Success criteria & evaluation

1. **Rot sweep restored:** probe `net_extent_m` on val93/val1684 back to
   O(GT) (≥ 0.2 m; the w=0.5 gen-11 reference was 0.27 / 0.84).
2. **traj_dir recovers** toward the w=0.5 band (≥ 90%; g13: 86.2).
3. **g13's records hold:** mIoU ≥ ~0.26, PDet ≥ ~21, axis matched ≤ ~17°,
   origin ≤ ~0.30 m — the normalization must not re-tax the shared
   decoder (its whole point).
4. Standard wrap-up: test eval + probe (512 flags) + viz batch (seed
   42421; door/oven rot samples must show real magenta arcs again).

## Code changes required

1. `train_OPDReal_better.py`: the gated branch + `L_trajectory_m2` log +
   the absolute-mode ValueError.
2. `config/opd_train.py`: `trajectory_loss_normalized: bool = False`.
3. Config as above.
4. Tests: flag-off bit-identity (same loss value as `trajectory_loss_fn`
   on random tensors); collapsed-prediction scores 1.0 (pred = zeros ⇒
   loss = 1 exactly for any GT above the eps floor); scale invariance
   (scaling GT and pred together leaves the loss unchanged, GT energy
   above eps); eps floor engages for a 1 cm degenerate GT row; ValueError
   with trajectory_absolute; config test (g16 = g13 + the two loss knobs
   + paths).

## Out of scope

- Any L_pp change (extent restoration belongs to the direct, GT-anchored
  loss; L_pp stays GT-free).
- Absolute-trajectory mode support for the flag.
- The cost-map-without-taps arm and the type-conditional-weight
  alternative (recorded; superseded if this works).
