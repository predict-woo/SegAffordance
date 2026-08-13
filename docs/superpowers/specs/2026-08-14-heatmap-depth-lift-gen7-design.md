# Gen-7: heatmap + depth lifts for both articulation points

**Status:** IMPLEMENTED (2026-08-14)
**Date:** 2026-08-14
**Companion visual:** docs/slides/2026-08-14_gen7_structure.html (every
decision below was made interactively on that page; it is the picture,
this is the contract).
**Supersedes:** the gen-6 direct-3D point pipeline (spec 2026-08-13),
whose eval validated the split articulation semantics (axis 22.1 deg,
type 97.5%, traj_dir 92.0% at K=1) and falsified its localization
pipeline (point_err_3d 0.58 m, origin_err 0.69 m, mIoU 0.098).

## Decision

Both 3D points become two-stage heatmap + depth predictions; nothing
about the direction/type heads changes:

| Quantity | How |
|---|---|
| Interaction point | point heatmap -> soft-argmax `point_uv` + scalar depth `z_p` -> `p_hat = unproject(point_uv, z_p, K)` |
| Joint origin | origin heatmap (NEW 3rd projector channel) -> soft-argmax `origin_uv` + scalar depth `z_q` -> `q_hat = unproject(origin_uv, z_q, K)` |
| Axis direction / type | MotionMLP, unchanged (1-cos sign-sensitive / CE ls=0.1) |
| Trajectory | K=1 direct readout of 20 ABSOLUTE camera-frame 3D points — NO delta-cumsum, NO relative frame (user decision; risks on record below) |
| Mask | unchanged |

Measured basis: projected q* is in-frame 99.6-100% on v2 data, both
filtered and unfiltered (tools/diag_origin_inframe.py) — the border-
clamp objection to a 2D origin readout is stale.

## Architecture (all flags default False = current behaviour)

- **Projector**: `out_channels = 3` when `use_origin_heatmap` — mask,
  point heatmap, origin heatmap, all from the shared language-
  conditioned `Projector_Mult`. The gen-6 `point_prediction_3d` flag
  stays and stays FALSE on this arm (the classical 2D point machinery
  is back on).
- **Readouts**: both channels through the same `soft_argmax2d`;
  `point_uv` (existing name) and `origin_uv` (new ModelOutputs field,
  (B, 2), normalized [0, 1]).
- **Condition vector**: pooled features + global visual + text state
  + `point_uv` (2) + `origin_uv` (2). Gradient flows through both
  soft-argmaxes (classical convention). Watch item: on prismatic rows
  the origin channel is unsupervised; downstream heads must learn to
  ignore `origin_uv` there (fallback: detach, one line).
- **Depth heads** (new `ModelParams.predict_point_depth` /
  `use_origin_heatmap` gates):
  - `z_p`: MLP [condition + grid_sample(decoded fq at point_uv) (512)]
    -> 256 -> 1, softplus + min_depth (reuse the `OriginDepthHead`
    module shape). The local sample: the element pixel shows a real
    surface whose appearance carries depth cues.
  - `z_q`: MLP [condition only] -> 256 -> 1, softplus + min_depth. NO
    local sample — the hinge pixel is often featureless wall/frame
    (user decision).
- **Lifts**: `unproject` via the existing `normalized_intrinsics` +
  `backproject_points` helpers. `CRIS.forward` gains optional
  `intrinsics` + `img_size` inputs; when absent (OPD batches, 2D
  pretraining, pure-2D inference) the lifted outputs are None and
  every 3D consumer no-ops — the lift exists only where K exists.
  Lifted points populate the EXISTING `ModelOutputs.point_3d_pred` /
  `origin_pred` fields, so the gen-6 trainer losses and eval metrics
  consume them unchanged.
- **Trajectory**: new `ModelParams.trajectory_absolute` gate —
  TrajectoryMLP emits (B, 1, 20, 3) raw absolute points (no cumsum, no
  zero-pinning). `trajectory_delta_cumsum` must be False when set
  (assert).
- **Pooling**: GT-mask teacher forcing at train
  (`pool_with_predicted_mask: false`), predicted sigmoid at val/test —
  the classical policy, reinstated (user decision; gen-6 evidence:
  predicted-mask pooling diluted every head's features and unfreezing
  CLIP did not rescue masks).
- **Backbone**: CLIP FROZEN (`freeze_backbone: true`, user decision).
  Watch item: gen-5 frozen run had the mask regression; gen-7 leans on
  masks harder than any generation.

## Losses (weights all 0.5, classical defaults, provisional)

```
L = 0.5*L_mask                      # DiceBCE, unchanged
  + 0.5*L_point_map + 0.5*L_coord   # classical pair, point channel
  + 0.5*L_origin_map                # NEW: BCE vs Gaussian at projected q*
  + 0.5*L_point_3d                  # ||p_hat - traj_3d[0]||^2 (composed lift)
  + 0.5*L_origin_3d                 # ||q_hat - q*||^2, revolute rows only
  + 0.5*L_motion(1-cos) + 0.5*L_type(CE)
  + 0.5*L_trajectory                # per-point MSE, ABSOLUTE camera coords
  + 0.5*L_pp                        # unchanged form; d_i computed in-loss
```

- `L_origin_map`: `make_gaussian_map` (same sigma = point_sigma = 8) at
  the NORMALIZED projection of q*; supervised on revolute rows whose
  q* projects in-frame with z > 0 (~99.6%); other rows contribute zero
  (loss-masked — NO dataset filter, eval split identical to gen-6).
  Helper `project_q_star(origin_gt, motion_gt, p_gt, K, img_size) ->
  (uv_norm, valid)` lives in model/losses/split.py.
- `L_point_3d` / `L_origin_3d`: composed — one 3D loss on the lifted
  point (user decision: compose first, price pixel+depth jointly in
  metres; gradients reach the heatmaps through soft-argmax). The
  existing `origin_canonical_loss` (q* MSE + revolute masking) is
  reused verbatim on the lifted `origin_pred`; additionally masked by
  the same in-frame validity as `L_origin_map`.
- `L_trajectory`: `MSE(trajectory_pred, targets.trajectory)` absolute,
  when `trajectory_absolute` (relative path preserved under the old
  flags). Point 0's target IS the element point — the deliberate
  double-prediction; `||p_hat - traj[0]||` becomes the free
  consistency readout `test/point_traj0_gap_m`.
- `L_pp` (`PredPredArticulationLoss`): constructor gains
  `trajectory_is_absolute: bool = False`; when True, `d_i = traj_i -
  traj_0` and the axis point in the relative frame is
  `c = origin_pred - traj_0` (decided) instead of
  `origin_pred - point_3d_pred`. Everything else (soft gate, radial +
  along-axis drift circle residual, degenerate masking, no-GT
  property) unchanged.
- Origin stack on prismatic: FULLY masked (heatmap BCE, L_origin_3d) —
  no auxiliary task (user decision). z_q still runs in forward (its
  output is only consumed by the lift; no loss reaches it on
  prismatic rows).

## Eval

Gen-6 metrics carry over (they read `point_3d_pred`/`origin_pred`):
point_err_3d_m, origin_err_m, origin_line_err_m, radius_err_m + the
classical 2D mean_point_error (coords exist again). New:
`test/point_traj0_gap_m`. Trajectory metrics: traj_dir uses last-first
(frame-agnostic, unchanged); the logged trajectory MSE is against the
absolute GT on this arm.

## Viz

tools/sf3d_vis_predictions.py: absolute-trajectory mode (draw the
curve directly, no anchoring step) and origin marker from `origin_uv`
+ red axis through lifted `q_hat` — extend the existing split-arm
branch; twist and gen-6 checkpoints keep rendering as today.

## Accepted risks (on record, config-flip fallbacks)

1. Direct per-point trajectory readout reverses the delta-cumsum fix
   for zigzag point clouds (viz/20260803_sf3d_twist_traj_points).
   Fallback: `trajectory_delta_cumsum: true` + relative loss.
2. Absolute-coordinate trajectory regression from the condition vector
   is the pattern that scored 0.58 m in gen-6 — mitigated (unproven)
   by point_uv/origin_uv conditioning + restored dense trunk
   supervision. Fallback: relative frame.
3. Frozen CLIP vs masks (see above).

## Config

`config/sf3d_train_runpod_g7.yaml` from the gen-6 config:
`freeze_backbone: true`; model: `point_prediction_3d: false`,
`use_origin_head: false` (the gen-6 MLP head — OFF; `use_origin_heatmap`
replaces it), `use_origin_heatmap: true`, `predict_point_depth: true`,
`trajectory_absolute: true`, `trajectory_delta_cumsum: false`,
`pool_with_predicted_mask: false`; loss: `point_map_weight: 0.5`,
`coord_weight: 0.5`, `origin_map_weight: 0.5`, `point_3d_weight: 0.5`,
`origin_weight: 0.5`, `trajectory_weight: 0.5`, rest as gen-6.
Experiment dir `experiments/20260814_sf3d_g7/`. Same data block
(filtered cache), 16 epochs, batch 128, lr 1e-5, milestones [13, 15].

## Testing

Unit: origin channel shapes + origin_uv readout; lift round-trip
(project(unproject(uv, z)) == uv, z); forward without intrinsics ->
lifted outputs None, no crash; q* projection helper (validity masking,
gauge invariance to sliding the annotated origin); composed-loss
gradients reach heatmap logits AND depth head; absolute trajectory
shape/loss; L_pp absolute mode (perfect prismatic/revolute score ~0
with absolute inputs; c = q_hat - traj_0 gauge along d_hat);
prismatic rows send zero gradient to the whole origin stack; classical
and gen-6 flag combinations bit-preserved; z_p local sample present
(input dim), z_q without. Existing 117 tests keep passing.
