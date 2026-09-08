# RGB-only model + scale-free trajectory head — design

Date: 2026-09-09. Status: agreed with the user in conversation (explainers:
`docs/slides/2026-09-09_model_depth_anchor_explainer.html`,
`..._scale_free_trajectory_explainer.html`, `..._depth_input_explainer.html`).

## Problem

Depth has two unrelated jobs in the current model, and both fail on the new
hand datasets (EPIC: no depth at all; ARCTIC: object-only render, the hand
knuckle lands on a zero pixel for 80% of records — measured 2026-09-09:
523/2559 valid anchors on ARCTIC, 0/359 on EPIC):

1. **Input feature.** `DepthEncoder` → 128/256 channels concatenated onto
   the backbone's H/8 and H/16 maps before the FPN.
2. **The anchor of the 2D recipe.** `TrajectoryProjectionLoss` and the z_p
   tether sample the input depth map at the GT track's first pixel to get a
   metric z₀; rows with z₀ = 0 are silently dropped, so the trajectory head
   receives no gradient.

The SF3D 3D path never uses the depth map for geometry: the point is
`point_uv` lifted with z_p, the origin is `origin_uv` lifted with z_q, the
trajectory loss compares relative offsets to the GT curve directly.

## Decisions (user)

- **No depth anywhere, on every dataset** (SF3D and HOI4D included; no
  partial-depth modes, no modality dropout). `use_depth: false`.
- **The 2D path does not learn scale.** The trajectory head predicts
  Δ̃ = Δ / z₀ (offsets in units of the anchor depth). The 2D projection loss
  anchors the GT first pixel at depth 1, which is exactly invariant to the
  unknown z₀ and needs no depth map. The 3D stage (SF3D) recovers metres
  with one multiply, z₀ · Δ̃: z₀ = GT first-point depth during training
  (teacher forcing of one scalar), z_p at test.
- Fresh training chain (HOI4D pretrain → SF3D post-train, plus an SF3D
  scratch arm) under the new convention; compared against the current
  record table (`STATE.md`) to price the loss of depth. Existing checkpoints
  are not inits (their heads are in metres, their FPN expects depth
  channels) — an optional column-slicing loader is provided for the
  "depth record checkpoint into RGB-only model" experiment only.

## Design

### A. RGB-only

- `ModelParams.use_depth: false` (exists). `CRIS.forward` must accept
  `depth=None` (it already guards the channels_last conversion; the fusion
  block is skipped).
- `SF3DDataset(load_depth=False)` (new, default True) + the same knob on
  `SF3DDataModule`: skip the depth PNG decode and emit
  `torch.zeros((1, H, W))` in the tuple slot, so the batch layout is
  unchanged and nothing downstream needs a None check.
- Trainer: with `depth_anchor_weight: 0`, `trajectory_proj_anchor: "unit"`
  and the scale-free test diagnostics (below), no loss or metric reads the
  depth tensor. The legacy depth-map origin metric already skips on z = 0.
- Optional loader surgery: `ModelParams.finetune_slice_input_channels:
  bool = False`. When true, `load_finetune_weights` loads a 4-D conv weight
  whose shape matches the model's except for MORE input channels by taking
  the leading model-sized slice (`v[:, :C_model]`). The concat order is
  `[rgb, depth]`, so the leading columns are the RGB ones — this is exactly
  the "concatenate zeros" model. Off by default; only `neck.f3_v_proj` and
  `neck.f2_v_proj` differ between a depth and an RGB-only checkpoint.

### B. Scale-free trajectory

- `ModelParams.trajectory_scale_free: bool = False` — documents the head's
  units (Δ̃) and switches the trainer's scale multiply on. The head itself is
  unchanged (linear readout, no range squash).
- `LossParams.trajectory_scale_source: str = "unit"` — train/val multiplier:
  - `"unit"`: no multiply (2D datasets: HOI4D, EPIC, ARCTIC).
  - `"gt_z0"`: `targets.trajectory[:, 0, 2]` (SF3D: exact GT depth of the
    first point; raises if the batch has no 3D trajectory).
- `Config.test_trajectory_scale: str = "pred_z_p"` — test multiplier:
  `"pred_z_p"` = `outputs.point_3d_pred[:, 2]` (the model's own z_p lift;
  raises if absent), `"gt_z0"` (oracle scale, measures shape only),
  `"unit"` (2D datasets).
- Helpers in `model/losses/geometric.py`:
  `trajectory_scale_factor(source, outputs, targets) -> Optional[Tensor]`
  and `apply_trajectory_scale(outputs, scale) -> ModelOutputs` (multiplies
  `trajectory_pred` and `trajectory_hyps`, in place on the dataclass).
  Called right after the forward in `_common_step` and in the SF3D
  `test_step` whenever `trajectory_scale_free` is true.
- `TrajectoryProjectionLoss(anchor_source="unit")`: anchor =
  `backproject(K_norm, uv0, 1)` with uv0 = GT first track point,
  `anchor_ok = trajectory_2d_valid[:, 0]`; the loss no longer requires
  `depth`. The other two sources are unchanged. Near-plane check stays at
  0.05 (now a fraction of the anchor depth).
- The test-time 2D-reprojection diagnostic (`proj2d_*`) lifts the anchor
  with the SAME scale that was applied to `trajectory_pred` instead of the
  depth map (projection is invariant to the joint rescale, so this equals
  the unit-anchor projection).
- Because the normalized trajectory loss divides by GT energy, applying
  z₀ as a constant multiplier gives exactly the normalized loss on the
  dimensionless target — no re-tuning of `trajectory_weight`.
- 2D-recipe constants in metres become fractions of anchor depth:
  `pred_pred_art_radius_floor` 0.10 m → 0.15 (z₀ ≈ 0.7 m on hand data).

### Configs (new)

| file | from | changes |
|---|---|---|
| `config/hoi4d_v2_rgb_scalefree.yaml` | `hoi4d_v2_teacher_forcing.yaml` | `use_depth: false`, `trajectory_scale_free: true`, `trajectory_scale_source: "unit"`, `trajectory_proj_anchor: "unit"`, `depth_anchor_weight: 0.0`, `pred_pred_art_radius_floor: 0.15`, `test_trajectory_scale: "unit"`, data `load_depth: false`; experiment dir `20260909_hoi4d_2d_v2_rgb_scalefree` |
| `config/epic_v1_rgb_scalefree.yaml` | `epic_v1_smoke.yaml` | same |
| `config/arctic_v1_rgb_scalefree.yaml` | `arctic_v1_smoke.yaml` | same |
| `config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml` | `sf3d_train_runpod_g19_dct_ft_hoi4d_tf.yaml` | `use_depth: false`, `trajectory_scale_free: true`, `trajectory_scale_source: "gt_z0"`, `test_trajectory_scale: "pred_z_p"`, data `load_depth: false`, `finetune_from_path` removed (scratch arm); experiment dir `20260909_sf3d_g19_dct_rgb_scalefree` |
| `config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml` | same | + `finetune_from_path` = the HOI4D rgb_scalefree arm's best ckpt (filled in after that run); experiment dir `20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d` |

### Out of scope

Training runs themselves (next step after the code lands and smoke tests
pass), mixed-dataset training, monocular depth estimation, modality dropout.

## Verification

- Unit tests (`tests/test_scale_free_trajectory.py`, run on the dev pod):
  unit anchor is exactly scale-invariant and needs no depth; invalid first
  point drops the row; scale-factor sources; RGB-only forward with
  `depth=None`; loader column slicing.
- Pod smoke: `fast_dev_run` of the HOI4D, EPIC, ARCTIC and SF3D scratch
  configs (one train + one val step each), then a `test` pass of the SF3D
  config on a random-init model to exercise `pred_z_p`.
