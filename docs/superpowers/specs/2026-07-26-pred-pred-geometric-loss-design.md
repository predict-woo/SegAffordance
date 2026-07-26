# Pred↔pred geometric consistency loss

**Date:** 2026-07-26
**Status:** implemented; unit-verified, integration pending a pod

Two deviations from the design as written, both found during implementation:

1. **`StepTargets` and `unpack_batch` live in `model/targets.py`**, not in the
   training module as specified below. `model/losses/geometric.py` names
   `StepTargets` in its signatures while the training module imports
   `model.losses` — keeping the bundle in the trainer is a circular import. It
   also made `unpack_batch` untestable without dragging in the datamodules.
2. **No existing config was edited.** The opt-in ships as a new matched pair,
   `config/sf3d_train_runpod_geo_{crossgt,predpred}.yaml`, so this is a new
   experiment rather than a change of defaults.

## Goal

Add a symmetric prediction-to-prediction geometric consistency loss coupling the
motion-axis head and the trajectory head, **selectable by config**, with the
existing cross-GT scheme preserved unchanged as the default.

Landing this requires a scoped structural fix: geometric losses currently live
as a `@staticmethod` on the LightningModule and are wired into `_common_step`
by hand, so there is no seam at which to swap one for another.

## Background: what exists today

`train_OPDReal_better.py:_common_step` computes two geometric terms, each
pairing one prediction against the *other head's* ground truth:

| term | trajectory arg | axis arg | weight |
|---|---|---|---|
| `L_geometric_pred_vector_gt_traj` | `trajectory_gt_relative` | `motion_pred` | `geometric_weight` |
| `L_geometric_pred_traj_gt_vector` | `trajectory_pred` | `motion_gt` | `trajectory_to_motion_weight` |

Both call `_geometric_consistency_loss` (`train_OPDReal_better.py:781`), which
branches on GT motion type: a line loss for prismatic, a circle loss for
revolute. The circle's centre and radius come from GT (`motion_origin_3d` and
the first GT trajectory point).

Neither term couples the two *predictions*, so nothing makes the heads
self-consistent at inference. Both are gated on `trajectory_gt is not None` and
`motion_origin_3d is not None`, so they are reachable only from the 13-element
SF3D batch; every OPD run to date computes no geometric loss at all.

## The new loss

### Derivation

`trajectory_pred` is already in **relative** coordinates — point 0 is the origin
by construction, and GT is converted to match (`train_OPDReal_better.py:198`).
That makes both motion types expressible without any origin:

- **Prismatic** — the path is a line through the start point, so every
  displacement is **parallel** to the axis.
- **Revolute** — the arc lies in a plane through the start point normal to the
  axis, so every displacement is **perpendicular** to the axis.

The two conditions are exact complements, via
`‖d‖² = (d·n̂)² + ‖d × n̂‖²`.

### Definition

With `n̂` the normalized `motion_pred`, `dᵢ = trajectory_pred[i]`, and
`M = Σᵢ dᵢdᵢᵀ` the uncentered scatter matrix:

```
R = n̂ᵀ M n̂ / tr(M)                    ∈ [0, 1]
p = softmax(motion_type_logits)[:, 1]   = P(revolute), gradient flowing
L = (1 − p)(1 − R) + p·R
```

`R` is the fraction of the trajectory's motion energy lying along the axis.
Prismatic wants `R = 1`, revolute wants `R = 0`.

Equivalently `R` is the displacement-magnitude-weighted mean of `cos²(dᵢ, n̂)`,
so minimizing `1 − R` drives `n̂` toward the top eigenvector of the trajectory's
scatter and minimizing `R` drives it toward the bottom eigenvector — the plane
normal, for a planar arc.

**Uncentered scatter is exact here, not an approximation:** relative coordinates
guarantee the line (prismatic) or plane (revolute) passes through the origin, so
centering would discard the anchoring.

### Properties

Substituting `u = 2p − 1` and `v = 2R − 1`, both in `[−1, 1]`:

```
L = ½(1 + u·v)
```

- **Symmetric in the gradient** — one scalar, gradients to both heads, no
  stop-gradient and no GT tensor in the expression.
- **Symmetric in the form** — invariant under swapping the roles of `u` and `v`;
  neither head is privileged.
- **Sign-invariant** — uses `cos²`, so invariant to `n̂ → −n̂` and to trajectory
  reversal. Matches how `MotionVAELoss` (`utils/tools.py:25`) already treats
  axes as undirected lines.
- **Scale-invariant** — the `tr(M)` denominator makes it a pure angular
  quantity in `[0, 1]`. The existing line/circle losses are unnormalized squared
  distances, i.e. **in square metres**, so `geometric_weight: 0.5` currently
  multiplies a scene-scale-dependent quantity while every other loss term is
  O(1). The new term is dimensionless and directly comparable to the rest.
- **Self-scheduling** — `∂L/∂R = 2p − 1` and `∂L/∂p = 2R − 1`. The gradient into
  each factor is proportional to the other's confidence, so while the type head
  is uncertain (`p ≈ 0.5`) the geometric constraint is near-zero and it ramps in
  as the type head sharpens. Given that the knowledge base records motion type
  as solved by epoch 1, this is a free curriculum.

### Accepted risks

- **Type-gate cheating.** With `p` live, the type head can reduce the loss by
  relabelling a sample instead of fixing the geometry. Chosen deliberately over
  a GT gate to keep the loss fully pred↔pred and to get the self-scheduling
  above. Cheat pressure is bounded by `|∂L/∂p| = |2R − 1| ≤ 1` scaled by the
  loss weight, against a type head anchored by its own CE at
  `motion_type_weight: 0.5`. Mitigations: start `pred_pred_weight` at **0.1**,
  and **monitor val motion-type accuracy against the 96–97% baseline in
  `experiments/INDEX.md`** — degradation there is the tell that the cheat is
  active.
- **Mutual agreement on a wrong answer.** Irreducible for any pred↔pred term.
  This is a regularizer alongside the direct GT losses, never a replacement.
- **Weaker revolute constraint.** With no predicted origin, revolute degenerates
  to coplanarity: a straight line lying in the correct plane scores zero. The
  circle loss additionally pinned radius and centre — but both were GT-supplied,
  so they never tested the model's own consistency, and trajectory position
  remains supervised by the direct MSE. Recovering them would need a
  differentiable in-plane circle fit as a second term; explicitly out of scope.

### Numerical requirements

- **Compute in fp32.** Runs use `precision: 16`; `R` is a ratio of sums of
  squares and should not be formed under autocast. Cast the loss body to float.
- **Degenerate trajectories must be masked, not epsilon'd.** If `tr(M) → 0` then
  `R → 0` and `L → (1 − p)`, which is *not* neutral — it actively pushes
  `p → 1`, a spurious "everything is revolute" signal. Drop samples with
  `tr(M) < τ` (absolute, τ = 1e-6 m²) from the batch mean rather than adding an
  epsilon to the denominator. If every sample in a batch is degenerate, return 0.
- **Clamp `R` to `[0, 1]`** after division; the bound holds mathematically but
  floating point can nudge outside it, and a negative loss is confusing.
- **Class-index convention:** motion type `1` = rotation/revolute, matching the
  `elif motion_type_gt[b] == 1` circle branch at `train_OPDReal_better.py:819`.
  `softmax(-1)[:, 1]` is therefore `P(revolute)`. Getting this backwards
  silently inverts the loss and must be covered by a test.

## Structural changes

Three changes, each the minimum needed for a swappable loss. Ordered by
dependency.

### 1. `ModelOutputs` dataclass

`CRIS.forward` returns an 8-tuple with two `None` slots. Six sites consume it
positionally, and `tools/smoke_backbone.py:119` maintains a hand-written
parallel list of field names for it. Loss modules need named access to pick out
the subset they care about.

Replace with a dataclass in a new `model/outputs.py`:

```python
@dataclass
class ModelOutputs:
    mask_logits: Tensor           # B,1,H/4,W/4
    point_logits: Tensor          # B,1,H/4,W/4
    coords_hat: Tensor            # B,2, normalized to [0,1]
    motion_pred: Tensor           # B,3
    motion_type_logits: Tensor    # B,num_motion_types
    trajectory_pred: Tensor       # B,20,3, relative to point 0
    mu: Tensor | None             # B,latent — CVAE with motion_gt only
    log_var: Tensor | None
```

A `NamedTuple` would preserve positional access and need zero call-site edits,
but that keeps the landmine this change exists to remove. Use a plain dataclass
and update all six sites:

| site | current | change |
|---|---|---|
| `model/segmenter.py:231,245,259` | three tuple returns | construct `ModelOutputs` |
| `train_OPDReal_better.py:143` | 8-name unpack | attribute access |
| `train_OPDReal_better.py:521` | unpack w/ underscores | attribute access |
| `train_SF3D_better.py:269` | unpack w/ underscores | attribute access |
| `tools/vis_predictions.py:132` | `out[:5]` | attribute access |
| `tools/eval_checkpoint.py:99` | `out[0]`, `out[3]`, `out[4]` | attribute access |
| `tools/smoke_backbone.py:118` | `zip(names, out)` | iterate dataclass fields, delete the name list |

No checkpoint impact — this is a return type, nothing in `state_dict`. Nothing
JIT-traces `CRIS` (the only `torch.jit.load` is the CLIP `.pt` at
`model/backbones/clip_rn50.py:26`).

### 2. `StepTargets` bundle

`_common_step` dispatches on `len(batch)` (13 / >10 / else) to unpack, which is
why a loss module cannot be handed "the targets" generically. Add a module-level
helper that keeps the length dispatch but returns a named bundle with `None` for
absent fields:

```python
@dataclass
class StepTargets:
    mask, point_norm, motion, motion_type, img_size   # always present
    trajectory: Tensor | None
    motion_origin_3d: Tensor | None
    camera_intrinsic: Tensor | None
```

Deliberately local to the training module — **no dataset or datamodule changes**,
so the three datasets keep their current tuple contract. Fixing that contract at
source is a separate job.

### 3. `model/losses/geometric.py`

New package holding the geometric losses, which today sit on the LightningModule
rather than beside `DiceBCELoss` / `MotionVAELoss`. Common interface:

```python
class GeometricConsistencyLoss(nn.Module):
    def forward(self, outputs: ModelOutputs, targets: StepTargets
                ) -> tuple[Tensor, dict[str, Tensor]]:
        """Returns (weighted total, unweighted named terms for logging)."""
```

Returning the weighted total *and* unweighted terms keeps weights owned by the
module (one source of truth) while logging values that stay comparable across
weight changes.

Three implementations:

- **`NoGeometricLoss`** — returns `(0, {})`. Used for OPD, and as the ablation.
- **`CrossGTGeometricLoss`** — the two current terms, weights `geometric_weight`
  and `trajectory_to_motion_weight`. **Must be numerically identical** to the
  present code path; the existing `_geometric_consistency_loss` body moves here
  verbatim.
- **`PredPredGeometricLoss`** — the new loss, weight `pred_pred_weight`.

Selected by `build_geometric_loss(loss_params)` on a new
`LossParams.geometric_loss: str = "cross_gt"`.

Both non-trivial variants no-op when the inputs they need are absent, preserving
today's behaviour on OPD batches: `PredPredGeometricLoss` requires
`targets.trajectory` only as a reachability gate (its own maths uses no GT),
while `CrossGTGeometricLoss` additionally requires `targets.motion_origin_3d`,
matching the two nested guards in the current code.

### Wiring in `_common_step`

The trajectory block (`train_OPDReal_better.py:194-257`) collapses to the
direct trajectory MSE plus:

```python
geo_total, geo_terms = self.geometric_loss(outputs, targets)
total_loss += geo_total
for name, value in geo_terms.items():
    self.log(f"{step_type}/{name}", value, ...)
```

This replaces the three hand-written eight-line `self.log` blocks for the
trajectory and geometric terms with generic iteration. The eight blocks for the
mask/point/coord/VAE/type losses are untouched — genericizing those is a
separate cleanup.

## Config

Two additions to `LossParams` (`config/opd_train.py`), both defaulted so that
**every existing YAML behaves exactly as it does today**:

```python
geometric_loss: str = "cross_gt"    # "cross_gt" | "pred_pred" | "none"
pred_pred_weight: float = 0.1
```

`geometric_weight` and `trajectory_to_motion_weight` stay, now read only by
`CrossGTGeometricLoss`. **No existing config file is modified** — the dataclass
gaining two defaulted fields is the whole change.

The opt-in ships instead as a new matched pair of experiment configs, so this
is a new experiment rather than an altered default:

| config | experiment dir | geometric block |
|---|---|---|
| `config/sf3d_train_runpod_geo_crossgt.yaml` | `20260726_sf3d_geo_crossgt` | `cross_gt`, weights 0.5/0.5 |
| `config/sf3d_train_runpod_geo_predpred.yaml` | `20260726_sf3d_geo_predpred` | `pred_pred`, weight 0.1 |

They are identical apart from that block and their output paths, so the loss is
the only variable. Both derive from `sf3d_train_runpod.yaml` (left untouched)
with two bookkeeping fixes it never got: a CSVLogger, since the original sets
`logger: false` and records no metrics; and `auto_insert_metric_name: false`,
without which `val/loss_total` in the checkpoint filename creates nested
directories (`knowledge/infra-lessons.md`).

## Verification

`tests/` currently holds only three LightningCLI shims — there is no unit test
suite. Add `tests/test_geometric_losses.py` as a real pytest module. The loss is
pure geometry, so it is fully testable on synthetic tensors, CPU-only, with no
dataset or GPU:

**Correctness**
- Line along the axis with `p = 0` → `L ≈ 0`.
- Circle in the plane normal to the axis with `p = 1` → `L ≈ 0`.
- Line along the axis with `p = 1` → `L ≈ 1` (catches an inverted class index).
- Circle with `p = 0` → `L ≈ 1`.
- `p = 0.5` → `L = 0.5` and zero gradient into both `motion_pred` and
  `trajectory_pred`, whatever the geometry.

**Invariances**
- `motion_pred → −motion_pred` leaves `L` unchanged.
- Trajectory scaled by any `k > 0` leaves `L` unchanged.
- Rotating axis and trajectory by a shared random rotation leaves `L` unchanged.

**Degeneracy**
- All-zero trajectory is excluded from the mean, not silently scored as
  revolute; an all-degenerate batch returns exactly 0.

**Gradients**
- Both `motion_pred` and `trajectory_pred` receive non-zero gradient for a
  misaligned pair, confirming the coupling is genuinely two-way.

**Regression**
- `CrossGTGeometricLoss` matches the pre-refactor `_geometric_consistency_loss`
  on fixed random input to within float tolerance.

**Integration**
- `tools/smoke_backbone.py` runs clean after the `ModelOutputs` change.
- One short OPD run confirms no behavioural change on the default path (OPD
  reaches no geometric loss either before or after).
- `tools/smoke_losses.py` drives `_common_step` on the real model across all
  four combinations of batch shape and variant, asserting the exact set of
  logged geometric terms in each — which is what catches a variant silently
  no-opping or leaking terms into the wrong dataset. Needs a GPU and
  `pretrain/RN50.pt`, hence `tools/` rather than `tests/`.

## Non-goals

- **Any SF3D training run.** Deferred by request. Note there is no SF3D row in
  `experiments/INDEX.md`, so evaluating the new loss will first need a baseline
  SF3D run on `cross_gt`. Until then this lands verified by unit tests only.
- **Vectorizing the legacy loop.** `_geometric_consistency_loss` iterates the
  batch in Python with a per-sample branch. Vectorizable by masking on motion
  type, but that perturbs float associativity and would break the bit-identical
  regression check. Leave it.
- **Moving `DiceBCELoss` / `MotionVAELoss`** out of `utils/tools.py` (which also
  holds a visualization helper). Real, unrelated.
- **Fixing the dataset tuple contract.** `StepTargets` papers over the
  `len(batch)` dispatch at the point of use; the datasets keep their current
  contract.
- **Head swapping.** Motion-head selection is already flag-based
  (`use_cvae`, `use_depth` in `config/opd_train.py`). Nothing in this change
  needs a third motion head, so the same string-registry treatment should wait
  until something actually requires it.

## Open question for review

`pred_pred_weight: float = 0.1` is a guess anchored on staying well under
`motion_type_weight: 0.5`. There is no evidence for it yet and no SF3D baseline
to calibrate against, so treat it as a starting point for a sweep rather than a
recommendation.
