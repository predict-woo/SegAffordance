# Gen-18: 2D-Only Training on the Gen-17 Stack

**Date:** 2026-08-18
**Status:** APPROVED (user, 2026-08-18)
**Companion:** docs/slides/2026-08-18_g18_2donly_structure.html
**Predecessor proof:** 20260804_sf3d_2donly (twist era: mechanism works,
signal weak; needed ω-prior, type collapsed).

## Design

Zero architecture change — the g17 model verbatim. "2D-only" is a loss
configuration: 3D GT (3D trajectory, axis, origin, type labels) reserved
for eval; supervision is the GT mask, the 2D interaction point, the 2D
track, and the model's internal consistency.

- **Data term:** `TrajectoryProjectionLoss` (exists; weight 0 in all 3D
  arms) at `trajectory_proj_weight: 0.5`, upgraded with
  `trajectory_proj_normalized` — per-row masked MSE divided by that row's
  GT track motion energy (track relative to its first valid point),
  `clamp(min=1e-4)`; rows without valid points are dropped from the row
  mean. Encodes the gen-16 rot-collapse lesson in uv-space.
- **Consistency term:** normalized L_pp @ 0.1, unchanged — sole teacher of
  the split axis heads, origin lift, and p_rev. Type self-organizes as
  model selection (index 1 structurally = circle = rot).
- **z_p tether:** `depth_anchor_weight: 0.5` — masked MSE between the
  predicted lifted-point depth (`point_3d_pred[..., 2]`) and the input
  depth sampled at a DETACHED `point_uv` (mask: sampled depth > 1e-3).
  Gradient reaches only the z_p lift, not point_uv, not the depth.
- **Zeroed (3D GT):** trajectory_weight, vae_weight (axis),
  motion_type_weight, origin_weight, origin_map_weight, point_3d_weight.
- **Kept (2D GT):** mask/point_map/coord @ 0.5, GT-mask pooling.
- **Logging:** batch-mean p_rev (`train/p_rev_mean`) to watch for branch
  collapse.
- **Balance prior (UN-parked after launch 1, 2026-08-18):** the first
  launch collapsed the gate within one epoch (p_rev_mean 0.44 → 0.0015) —
  the line residual is bounded ≤ 1, the circle residual is not, so
  trans-everywhere is the cheapest descent. `p_rev_balance_weight: 0.5` on
  `(mean(p_rev) − 0.225)²` (target = key-set rot fraction) constrains only
  the batch mean. Same launch also showed 30–50× outlier ratios from
  near-degenerate tracks → `trajectory_proj_energy_floor: 0.0025`.

## Config

`config/sf3d_train_runpod_g18_2donly.yaml` = g17 config, loss_params per
the table above, paths → `experiments/20260818_sf3d_g18_2donly`. Same
data/epochs/seed; model_params byte-identical to g17.

## Expectations & eval

Standard test pass (3D GT is eval-only, exactly SF3D's role). Success ≠
g17 parity: (a) type separation and traj_dir far above chance with zero 3D
labels; (b) sign-agnostic axis meaningfully better than random (sign is
UNOBSERVABLE — both L_pp branches quadratic in the axis; expect ~50% flip
rate, read the sign-agnostic columns). Origin metrics expected weak (z_q
trains only through L_pp). The payoff experiment (separate, later):
finetune the g17 recipe from this checkpoint vs from scratch.

## Code changes

1. `model/losses/geometric.py`: `TrajectoryProjectionLoss(weight,
   near_plane, normalized=False)` — normalized branch restructures the
   flat masked mean into per-row means.
2. `config/opd_train.py`: `trajectory_proj_normalized: bool = False`,
   `depth_anchor_weight: float = 0.0`.
3. `train_OPDReal_better.py`: pass the flag through; depth-anchor term +
   `L_depth_anchor` log; `p_rev_mean` log.
4. Config as above. Tests: normalized-projection (collapse scores ~1/row,
   flag-off bit-identity, eps floor, all-masked rows dropped);
   depth-anchor (gradient reaches z_p head only; masked on depth holes);
   config chain (g18 = g17 + loss table + paths); training-step
   integration (finite loss, no 3D-GT gradient: axis/type/origin heads'
   grads come only from L_pp).
