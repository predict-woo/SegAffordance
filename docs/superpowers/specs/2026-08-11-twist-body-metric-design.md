# Twist loss: body-frame kinetic-energy metric (sensitivity fix)

**Status: approved design — NOT yet implemented.** This fixes the loss
*pricing* only; the companion fix for mode-averaging (hedging) is
`2026-08-11-twist-wta-head-design.md` (approved same day). Implement both
together — the WTA loss uses this metric as its distortion measure.

## Problem

The twist head is supervised with a plain 6-vector MSE
`mean(Δω² , Δv²)`. Diagnosis on the clip_g3 checkpoint
(2026-08-08/11 session; `tools/diag_twist_radius.py`, 800 stratified val
samples) showed:

- GT-revolute predictions have `‖ω‖` median **0.295** (target 1.0); 77% fall
  below the 0.5 type threshold and decode as prismatic — this alone explains
  the ~60% emergent-type accuracy.
- Decoded radius is inflated **one-sidedly**: pred/GT ratio > 1 on 96% of
  revolute samples, median 10× (3.3× among samples that decode as revolute;
  GT radius median 0.25 m → predicted 1.43 m). This is the "too-flat arcs"
  symptom in the gen-3 panels.
- Forcing the GT type hint changes nothing (`‖ω‖` 0.295 → 0.307), so type
  ambiguity is not the dominant shrinker; sign hedging (dir acc ~71%) and
  optimization drift along flat loss directions are.

Root cause decomposition (synthetic study, scratchpad `twist_diag.py`):

1. **Hedging** (separate fix, out of scope here): under genuine rot/trans and
   sign ambiguity the MSE minimizer is the posterior-mean twist; the mean of a
   revolute (`‖ω‖=1`) and prismatic (`ω=0`) twist decodes to a far axis.
2. **Sensitivity imbalance** (THIS fix): the decode `q = ω×v/‖ω‖²` is
   hyperbolic exactly where the MSE is flat. Measured: an axis moved 0.5 m
   too far (radius 3×) costs 1.4% of a sign flip; the observed `‖ω‖→0.3`
   collapse costs 2.7%. The loss surface is nearly flat along
   radius-controlling directions, so weight decay and noise park `ω` low.
3. The consistency terms cannot push back: the `1−cos` field residual is
   scale-free in the twist and cannot distinguish radius 0.18 m from 0.61 m
   on a 15° arc.

Key structural insight: the current MSE **is** a kinetic-energy metric — of a
unit-mass, unit-gyration rigid body located **at the camera origin**. `‖Δv‖`
prices axis position relative to a point ~2 m away from anything that matters.

## Design

Replace the Euclidean form inside `TwistLoss` with the left-invariant
kinetic-energy (rigid-body inertia) metric on se(3), with the body placed at
the GT element point:

```
L_twist(Δξ) = ‖Δω × p0 + Δv‖²  +  ρ² ‖Δω‖² ,      Δξ = ξ_pred − ξ_gt
```

- `p0` = `targets.trajectory[:, 0]` — the GT element point in camera frame
  (present in every SF3D batch; the loss already no-ops on batches without
  3D GT, e.g. OPD).
- The first term is exactly the velocity-field error **at the object**,
  `‖f_pred(p0) − f_gt(p0)‖²` in (m/s)², because `f(p) = ω×p + v` is linear
  in the twist.
- `ρ` (`LossParams.twist_metric_rho`, default **0.25 m**) is the body's
  gyration radius — the only hyperparameter, a part-scale constant equal to
  the median GT radius. It prices the angular error independently of how
  close the object sits to the axis (and regularizes the metric for the ~9%
  degenerate stub trajectories).
- Sign-agnostic path (OPD convention) unchanged in structure: `min` over the
  joint `(ω,v) → (−ω,−v)` flip of the same metric.

Properties (why this and not a heuristic reweighting):

- Pullback of the L² metric on velocity fields restricted to the object —
  the classical rigid-body inertia metric (Park; Featherstone spatial
  algebra). `M = A(p0)ᵀA(p0) + ρ²·diag(I,0)`, `A(p) = [−[p]ₓ  I]`.
- **Gauge-invariant**: unlike the current loss, independent of the choice of
  camera/world origin. The only distinguished point is the object.
- Preserves every twist-arm invariant: convex quadratic on ℝ⁶, both motion
  types interior, no branches/type gates, sign-sensitive, smooth through
  `ω = 0`.

Measured rebalancing (GT radius 0.25 m; prices relative to a sign flip in
the same metric):

| error                              | current MSE | body metric (ρ=0.25) |
|------------------------------------|-------------|----------------------|
| axis 0.5 m too far (radius 3×)     | 1.4%        | 50%                  |
| pure `‖ω‖` shrink to 0.3 (observed)| 2.7%        | 330%                 |
| axis tilt ~6°                      | 0.23%       | 0.12%                |

Angular-error pricing is *unchanged in law*: any quadratic metric prices a
tilt δ at ≈ δ²/4 of a flip (the lever arm cancels in the ratio); direction
retains its healthy gradient ∝ δ. Only radius-relevant directions gain
(~40–100× relative).

### Alternatives rejected

1. **Pullback at the GT trajectory points** (`meanᵢ ‖Δω×pᵢ + Δv‖²`):
   parameter-free, identifiability-calibrated — but under-prices radius on
   short arcs and is near-singular on stub trajectories. The ρ-term is
   exactly the regularization it lacks; the body metric dominates.
2. **Decoded-geometry losses** (axis-line distance, `log‖ω‖`, angles):
   meters-calibrated but non-convex, needs a prismatic branch, resurrects
   type gating.

## Scope

- **Fixes**: gradient pricing of radius/`‖ω‖` errors, camera-origin gauge
  artifact, weight-decay drift along flat directions.
- **Does NOT fix** (by proof, not oversight): the posterior-mean hedge under
  genuine ambiguity — the minimizer of any fixed quadratic metric is the
  same affine mean. That is the separate mode-commitment fix, to be designed
  before implementing either.

## Implementation sketch (when unblocked)

- `model/losses/twist.py::TwistLoss`: accept `metric_rho: float` and the
  anchor `p0`; replace `err_pos`/`err_neg` with the body form. Anchor comes
  from `targets.trajectory[:, 0]`; if trajectory is absent, fall back to the
  current Euclidean form (OPD paths are no-ops anyway).
- `config/opd_train.py::LossParams`: add `twist_metric_rho: float = 0.25`.
  `twist_weight` stays **0.5** — measured on 600 stratified val GT rows
  (pred = 0 reference): mean init scale 0.603 (body) vs 0.634 (current
  MSE), ratio 0.95. The metric redistributes across types (old: revolute
  rows 6.6× prismatic via the camera-gauge `‖v‖≈1.76`; new: prismatic
  1.00 vs revolute 0.21, i.e. metres of actual motion at the object —
  the GT time convention prices 1 m of handle travel like 1 rad of
  sweep). Watch-item, not a knob: if revolute twist learning lags under
  the new balance, revisit.
- Normalization convention: the field term is `‖Δf(p0)‖²` SUMMED over
  xyz (a physical squared norm). Make the trajectory MSE use the same
  per-point norm² convention so the two terms — which are in identical
  units (m² at the object) after this change — stay directly comparable.
- SF3D twist configs: set `twist_metric_rho` explicitly.
- Tests (`tests/test_twist.py`): zero at `Δξ=0`; gauge invariance under a
  synthetic origin shift; the pricing table above as regression assertions;
  sign-agnostic min-over-flip; prismatic rows well-behaved.
- Not comparable across checkpoints: a new loss scale means `val/loss_total`
  is not comparable with gen-3 runs; new experiment ID.

## Evidence trail

- Real-model stats: `tools/diag_twist_radius.py` (committed with this spec);
  per-sample CSV `/tmp/diag_twist_radius.csv` on the dev pod.
- Synthetic study: session scratchpad `twist_diag.py` (E1 flatness, E2
  identifiability, E3 posterior-mean mechanism + branched baseline).
