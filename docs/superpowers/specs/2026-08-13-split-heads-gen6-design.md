# Gen-6: split articulation heads, classical per-branch losses

**Status:** DESIGNED (not implemented)
**Date:** 2026-08-13
**Supersedes as the training arm:** the twist parameterization
(2026-08-11 body-metric + WTA specs). Those code paths stay in the
repo, config-gated, so gen-3/4/5 remain reproducible.

## Decision

Retire the se(3) twist as the *predicted* representation. Gen-5 proved
the WTA + body-metric machinery can fight the type-in-|omega| hedge to
|omega| = 0.59, but that is a permanent tax paid to make a regression
target carry a discrete label. Gen-6 moves each quantity to the output
type that suits it:

| Quantity | Head | Output space |
|---|---|---|
| Motion type | `MotionMLP` type head (re-enabled) | 2 logits, CE |
| Axis direction | `MotionMLP` motion head (re-enabled) | 3-vector, normalized at decode |
| Joint origin | **new** `OriginMLP` | absolute 3D point, camera frame |
| Interaction point | **new** 3D point head (SF3D arm only) | absolute 3D point, camera frame |
| Trajectory | `TrajectoryMLP`, K=1 | 20 relative points, delta-cumsum |
| Mask | unchanged | unchanged |

Type commitment cannot fail structurally: CE hedging is calibrated
probability with zero geometric side effect. Radius is computed
linearly from the explicit origin — no hyperbolic decode exists.

All heads are single-hypothesis (user decision). Known accepted risk:
every single-hypothesis run to date converged the trajectory loss to
the zero-motion baseline (0.018) under opening-direction ambiguity.
If gen-6 reproduces that signature, the parameterization-independent
fix (K=4 WTA on the trajectory only) is a recorded follow-up, not part
of this design.

## Baseline: the classical implementation (cee7cd7)

The reference is the code that trained `20260726_sf3d_geo_crossgt`:
CVAE axis+type heads, direct trajectory readout, point supervised at
the projected motion origin, axis loss `1 - cos^2` (sign-agnostic),
type CE (label smoothing 0.1), trajectory MSE, cross-GT geometric
loss, all weights 0.5, no origin prediction, no filters, no hint.

## Gen-6 vs classical — full change list

**Kept verbatim:** type CE (incl. label smoothing 0.1), trajectory
relative-MSE, DiceBCE mask loss and their 0.5 weights.

**Changed, carrying independently-validated fix-era wins:**

- CVAE -> `MotionMLP` (`use_cvae: false`). The CVAE's KLD had
  collapsed (~3e-5); it was a deterministic head with extra plumbing.
- Axis loss `1 - cos^2` -> **sign-sensitive `1 - cos`**. Deviation
  from classical, deliberate: SF3D's stored axis sign is canonical
  (the GT trajectory is derived from it, commit cd51085); antiparallel
  must not score as perfect. Scale-free like the original.
  Note `MotionMLP.motion_head` ends in sigmoid mapped to [-1, 1]; the
  cosine loss is invariant to the magnitude this induces.
- Interaction point is the graspable element centroid, not the
  border-clamping projected hinge — and on this arm it is predicted
  directly in 3D (see "Genuinely new"), so `point_source` only
  matters for arms still on the 2D path.
- `trajectory_delta_cumsum: true` — connected path by construction.
- `min_revolute_radius: 0.10` — no knob/dial/faucet-class rotations;
  applies to train AND val/test (metrics comparable to gen-5, not to
  gen-3/4). Key cache: `sf3d_v2_keys_cutoff05_minrad010.pkl`.
- Fast pipeline + channels_last + compile (infrastructure, not
  modeling).
- CLIP **unfrozen** (gen-5's frozen backbone is the prime suspect for
  its mask regression to mIoU 0.090).
- Cross-GT geometric loss -> **prediction-anchored articulation
  consistency** (user decision: no teacher forcing). New
  `geometric_loss: "pred_pred_art"` variant — the classical line /
  circle residual forms, evaluated entirely on PREDICTED quantities
  in the trajectory's relative frame (first point = 0):

  ```
  d_i   = pred relative trajectory displacements   (N, 3)
  dhat  = pred axis direction, normalized
  c     = q_hat - p_hat        # pred axis point, relative frame
  L_line   = mean_i || d_i x dhat ||^2                    # prismatic
  r_hat    = dist(0, axis line (c, dhat))
  L_circle = mean_i ( dist(d_i, axis line) - r_hat )^2    # revolute
  L_pp     = P(pris) * L_line + P(rev) * L_circle
  ```

  The type gate is the SOFT predicted probability (no GT type
  either): gradient into the geometry scales with the type head's
  confidence, the same self-switching coupling the July
  `PredPredGeometricLoss` used. Degenerate predicted trajectories
  (total displacement below threshold) are masked out of the batch
  mean, reusing that loss's guard, since a zero curve satisfies any
  axis. The old cross-GT and energy-ratio pred-pred variants stay in
  the code, config-gated, unused by this arm. NOTE the July
  crossgt-vs-predpred experiment pair never ran (the twist arm
  superseded it), so this coupling is empirically untested; each
  quantity keeps its own direct GT loss, which is what anchors the
  mutually-consistent-but-wrong failure mode. Toggleable by
  construction: `geometric_loss` remains a config enum
  (`pred_pred_art` | `cross_gt` | `pred_pred` | `screw` |
  `projected` | `none`), so reverting to the teacher-forced variant
  or disabling consistency entirely is a one-line YAML change.
- **Mask-pooling teacher forcing removed, flag-gated** (user
  decision): new `ModelParams.pool_with_predicted_mask` (bool,
  default false = classical GT-mask pooling at train time). When
  true, the condition-vector pooling uses the DETACHED sigmoid of
  the predicted mask in training too — identical to what val/test
  always did — so the articulation heads train on the same feature
  distribution they deploy on. Detached, so articulation losses
  cannot push the mask head around to make pooling convenient; the
  mask keeps learning only from its dense DiceBCE term. Known
  cold-start cost: pooled features are ~global averages until the
  mask sharpens (historically within ~1 epoch). Gen-6 config sets
  true.

**Removed relative to gen-5 (and relative to classical where noted):**

- Twist head, body metric, WTA bundles/annealing/selector, screw
  consistency losses, type-from-|omega| eval. Config-gated off, code
  untouched.
- GT-type input hint (`use_motion_type_input: false`) — twist-era
  crutch; the type head makes it redundant. No embedding concat in
  the condition vector.
- **2D interaction-point machinery (SF3D arm)**: the point heatmap
  channel, point-map BCE, coord L1, and `coords_hat` are REPLACED by
  the direct 3D point head (user decision, Option B — full
  replacement, not add-alongside). `Projector_Mult` runs with
  `out_channels=1` (mask only) on this arm. Flag-gated
  (`point_prediction_3d: bool`, default false): the OPD trainers
  share `_common_step` and have no 3D element GT, so they keep the
  classical 2D path untouched. Accepted risk, on record: this
  removes the dense heatmap localization supervision the point has
  always trained on; if localization degrades, Option A
  (3D head alongside the 2D machinery) is the recorded fallback.

**Genuinely new (no classical counterpart):**

- `OriginMLP`: `vae_condition -> Linear(hidden) -> ReLU -> Linear(hidden)
  -> ReLU -> Linear(3)`, unconstrained absolute 3D point (user
  decision: absolute, not offset-from-element, not 2D+depth).
- Origin loss, **revolute samples only** (prismatic rows contribute
  zero and no gradient): plain MSE against a CANONICAL point on the
  GT axis — the foot of the perpendicular from the GT element point
  `p = trajectory_3d[0]` onto the GT axis line:

  ```
  q_star   = o_gt + ((p - o_gt) . d_gt) d_gt     # d_gt normalized
  L_origin = || q_hat - q_star ||^2
  ```

  (user decision, replacing an earlier distance-to-line form). The
  gauge is fixed by canonicalization rather than invariance: `q_star`
  is the unique axis point where the segment to the interaction point
  is perpendicular to the axis, so all 3 output dimensions are
  constrained (distance-to-line left the along-axis component free to
  drift) and the target sits near the element — bounded and local.
  `q_star` is unchanged if the annotated origin slides along the axis
  (same robustness to annotation gauge as before). Batch-masked mean
  over revolute rows; zero-valued term logged when a batch has none
  (same semantics as the screw losses' degenerate handling).
- **3D interaction-point head**: same MLP shape as `OriginMLP`,
  absolute camera-frame 3D point. Loss: plain MSE against the GT
  element point `trajectory_3d[0]` (present in every SF3D record).
  Sequencing: the point head consumes the base condition vector
  (pooled features + global visual + text state — no `coords_hat`
  anymore); the articulation heads (type, direction, origin,
  trajectory) consume `[base condition, predicted point_3d]`,
  gradient flowing, mirroring how `coords_hat` classically entered
  the condition. At inference the relative trajectory is anchored at
  the predicted 3D point — no depth-map lookup, no intrinsics.
  Viz projects the predicted 3D point with intrinsics for drawing.
- Eval additions: `point_err_3d_m` (metric error of the predicted
  interaction point), `origin_err_m` (distance to `q_star`,
  unsquared), `origin_line_err_m` (distance to the GT axis line —
  the axis-position error the paper cares about, insensitive to
  along-axis slack) and `radius_err` = | dist(element, pred axis
  line) - GT radius | on revolute rows. Axis direction error/type accuracy
  come from their heads directly; `twist_*` and the 2D
  `mean_point_error` metrics disappear.

## Loss total

```
L = 0.5*L_mask
  + 0.5*L_point_3d(MSE, camera coords)
  + 0.5*L_motion(1-cos) + 0.5*L_motion_type(CE)
  + 0.5*L_trajectory(MSE, relative)
  + 0.5*L_pp(prediction-anchored consistency, soft type gate)
  + w_origin * L_origin
```

`w_origin = 0.5` initially (every classical term uses 0.5; the term is
in m^2 like the geometric and point terms). `L_point_3d` replaces the
classical point-map BCE + coord L1 pair at the same total weight
budget. New `LossParams` fields: `origin_weight` (0.5),
`point_3d_weight` (0.5); new `ModelParams` fields: `use_origin_head`
(bool, default false), `point_prediction_3d` (bool, default false),
`pool_with_predicted_mask` (bool, default false). All defaults
preserve classical behaviour; the gen-6 config turns them all on.
The predicted origin feeds the consistency term's
revolute branch (the axis line it constrains the trajectory against
is the fully predicted one); no GT quantity appears anywhere in
`L_pp`. GT supervision reaches each head only through its own direct
term.

## Config

New `config/sf3d_train_runpod_split.yaml`, derived from the twist
config's data/trainer blocks (batch 128, lr 1e-5, 16 epochs,
milestones [13, 15], filtered key cache) with the model/loss blocks
per this spec. Experiment dir: `experiments/2026MMDD_sf3d_split_g6/`.

## Testing

- Unit: `q_star` canonicalization (translate the annotated GT origin
  along the axis -> `q_star` and the loss unchanged; `(q_star - p)`
  perpendicular to `d_gt`); revolute-only masking (prismatic-only
  batch -> zero term, no NaN); sign sensitivity of the axis loss
  (antiparallel prediction scores ~2, not 0); OriginMLP / point-head
  shapes; `point_prediction_3d=false` reproduces the classical 2D
  outputs exactly (OPD-arm regression guard); `pred_pred_art` uses
  no GT tensors (a perfect prediction set scores ~0 regardless of
  GT), soft gate selects the right branch at p in {0, 1}, q_hat
  shifted along d_hat leaves L_circle unchanged (gauge), degenerate
  predicted trajectory masked out without NaN;
  `pool_with_predicted_mask=true` in train mode pools with the
  detached predicted mask (no grad reaches the mask logits through
  pooling) and `=false` reproduces GT pooling bit-exactly.
- Existing twist/WTA tests keep passing (nothing deleted).
- Smoke on the dev pod before any training-pod launch, as always.

## Open questions deferred (recorded, not blocking)

- Trajectory re-hedge -> add trajectory-only WTA (see above).
- DINOv3 mask-projector fix; 2D arm updates (pre-existing items).
