# RGB-only model + scale-free trajectory head — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the depth map from the model's input and from every loss, so HOI4D/EPIC/ARCTIC/SF3D all train the same RGB-only way, with the trajectory head predicting depth-normalised offsets (Δ̃ = Δ/z₀) that the SF3D stage converts to metres with one scalar.

**Architecture:** (1) `TrajectoryProjectionLoss` gets a third anchor source, `"unit"`: the GT first pixel lifted to depth 1, no depth map. (2) Two helpers in `model/losses/geometric.py` compute a per-sample scale (`None` / GT z₀ / predicted z_p) and multiply `outputs.trajectory_pred` by it; the trainers call them right after the forward when `model_params.trajectory_scale_free` is on. (3) `SF3DDataset(load_depth=False)` emits zeros instead of decoding depth; `use_depth: false` (existing) drops the encoder. (4) An opt-in loader rule slices depth columns off FPN conv weights so a depth checkpoint can init the RGB-only model. New configs for the four datasets.

**Tech Stack:** PyTorch 2 / PyTorch Lightning (LightningCLI YAML configs), pytest. Tests and smoke runs execute on the dev pod through `bash runpod/dev.sh run "<cmd>"` (the pod's `/opt/venv` is on PATH there; the Mac has no venv). Git commits are made on the Mac only.

**Spec:** `docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md`

## Global Constraints

- Edit files on the Mac under `~/dev/ethz-workspace/SegAffordance/`; they mirror to the pod in ~1 s. Never run python on the Mac.
- Run tests as: `bash runpod/dev.sh run "python -m pytest tests/<file> -q"` (abbreviated `$RUN` below). Expected output ends with `N passed`.
- Commit from the Mac with the trailer:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01XL5tB4LrNonfPRfAz1hjRR
  ```
- Default values of every new field keep existing configs and checkpoints behaving exactly as before: `trajectory_scale_free=False`, `finetune_slice_input_channels=False`, `trajectory_scale_source="unit"`, `test_trajectory_scale="pred_z_p"`, `load_depth=True`.
- Do not touch `model/layers.py` (the head is unchanged) or any existing config file.

---

### Task 1: `"unit"` anchor source in `TrajectoryProjectionLoss`

**Files:**
- Modify: `model/losses/geometric.py:1178-1273` (`TrajectoryProjectionLoss.__init__` and `forward`)
- Test: `tests/test_scale_free_trajectory.py` (new)

**Interfaces:**
- Produces: `TrajectoryProjectionLoss(weight, ..., anchor_source="unit")` — `forward(outputs, targets, depth)` accepts `depth=None` in this mode.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scale_free_trajectory.py`:

```python
"""RGB-only + scale-free trajectory (spec 2026-09-09).

The projection loss's "unit" anchor: the GT first pixel lifted to depth 1,
no depth map. Exactly invariant to the unknown anchor depth when the head
predicts Δ̃ = Δ / z0.
"""
import torch

from model.losses.geometric import (
    TrajectoryProjectionLoss,
    backproject_points,
    normalized_intrinsics,
    project_points,
)
from model.outputs import ModelOutputs
from model.targets import StepTargets


def _case(rel_traj_m, z0, uv0=(0.5, 0.5)):
    """GT 2D track = projection of a METRIC curve anchored at depth z0.
    Returns (outputs with the head predicting rel/z0, targets)."""
    B, N, _ = rel_traj_m.shape
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]).expand(B, 3, 3)
    img_size = torch.tensor([[100.0, 100.0]]).expand(B, 2)
    K_norm = normalized_intrinsics(K, img_size)
    uv = torch.tensor([list(uv0)]).expand(B, 2)
    anchor = backproject_points(K_norm, uv, torch.full((B,), float(z0)))
    track = project_points(K_norm, anchor.unsqueeze(1) + rel_traj_m)
    dummy = torch.zeros(B, 1, 4, 4)
    outputs = ModelOutputs(
        mask_logits=dummy, point_logits=dummy, point_uv=uv,
        trajectory_pred=rel_traj_m / z0,          # the scale-free head's output
    )
    targets = StepTargets(
        camera_intrinsic=K, img_size=img_size, trajectory_2d=track,
        trajectory_2d_valid=torch.ones(B, N, dtype=torch.bool),
    )
    return outputs, targets


REL = torch.tensor([[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.02, 0.0], [0.15, 0.05, 0.01]]])


def test_unit_anchor_is_exact_for_any_anchor_depth_and_needs_no_depth_map():
    loss_fn = TrajectoryProjectionLoss(weight=1.0, anchor_source="unit")
    for z0 in (0.4, 1.0, 2.5):
        outputs, targets = _case(REL, z0)
        loss, terms = loss_fn(outputs, targets, depth=None)
        assert loss.item() < 1e-10, z0
        assert "L_traj_proj" in terms


def test_unit_anchor_penalises_metric_offsets_when_z0_is_not_one():
    outputs, targets = _case(REL, 2.5)
    outputs.trajectory_pred = REL.clone()  # metres, not Δ/z0
    loss, _ = TrajectoryProjectionLoss(weight=1.0, anchor_source="unit")(outputs, targets, None)
    assert loss.item() > 1e-4


def test_unit_anchor_drops_rows_whose_first_point_is_invalid():
    outputs, targets = _case(REL, 1.0)
    targets.trajectory_2d_valid[:, 0] = False
    loss, terms = TrajectoryProjectionLoss(weight=1.0, anchor_source="unit")(outputs, targets, None)
    assert loss.item() == 0.0 and terms == {}


def test_unit_anchor_has_no_gradient_path_into_the_anchor():
    outputs, targets = _case(REL, 1.0)
    pred = (REL / 1.0).clone().requires_grad_(True)
    outputs.trajectory_pred = pred
    outputs.point_uv = outputs.point_uv.clone().requires_grad_(True)
    loss, _ = TrajectoryProjectionLoss(weight=1.0, anchor_source="unit")(outputs, targets, None)
    (loss + pred.pow(2).sum() * 0.0).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert outputs.point_uv.grad is None  # the unit anchor never reads point_uv


def test_legacy_sources_still_require_depth():
    outputs, targets = _case(REL, 1.0)
    loss, terms = TrajectoryProjectionLoss(weight=1.0, anchor_source="gt_point")(outputs, targets, None)
    assert loss.item() == 0.0 and terms == {}
```

- [ ] **Step 2: Run to verify they fail**

Run: `$RUN tests/test_scale_free_trajectory.py -q`
Expected: 4 failures with `ValueError: anchor_source must be pred_depth|gt_point, got unit` (the legacy test passes).

- [ ] **Step 3: Implement the unit anchor**

In `model/losses/geometric.py`, `TrajectoryProjectionLoss.__init__` (around line 1188), replace the comment + check:

```python
        # "pred_depth": point_uv lifted with the input depth (g17-2d chain).
        # "gt_point": teacher forcing — anchor at targets.trajectory[:, 0]
        # (GT camera-frame first point); constant, depth-free (2026-09-07).
        if anchor_source not in ("pred_depth", "gt_point"):
            raise ValueError(f"anchor_source must be pred_depth|gt_point, got {anchor_source}")
```
with
```python
        # "pred_depth": point_uv lifted with the input depth (g17-2d chain).
        # "gt_point": teacher forcing — the GT 2D first point lifted with the
        # input depth there; constant (2026-09-07). Still needs a depth map.
        # "unit" (2026-09-09 scale-free spec): the GT 2D first point at depth
        # EXACTLY 1 — no depth map at all. Projection ignores a global
        # rescale of (anchor, curve), so with the head predicting
        # Δ̃ = Δ / z0 this is the same loss as gt_point for every z0; the
        # unknown metric scale is factored out and supplied by the 3D stage
        # (model.losses.geometric.trajectory_scale_factor).
        if anchor_source not in ("pred_depth", "gt_point", "unit"):
            raise ValueError(f"anchor_source must be pred_depth|gt_point|unit, got {anchor_source}")
```

In `forward`, change the early-return condition line `or depth is None` to:

```python
            or (depth is None and self.anchor_source != "unit")
```

and replace the anchor branch (`if self.anchor_source == "gt_point": ... else: ...`) with:

```python
            if self.anchor_source == "unit":
                uv0 = targets.trajectory_2d[:, 0, :].to(device).float()
                z = torch.ones(uv0.shape[0], device=device, dtype=torch.float32)
                anchor = backproject_points(K_norm, uv0, z).detach()
                anchor_ok = torch.ones(uv0.shape[0], device=device, dtype=torch.bool)
                if targets.trajectory_2d_valid is not None:
                    anchor_ok = anchor_ok & targets.trajectory_2d_valid[:, 0].to(device).bool()
            elif self.anchor_source == "gt_point":
                # (existing gt_point body, unchanged)
                ...
            else:
                # (existing pred_depth body, unchanged)
                ...
```

Also update the class docstring's first paragraph: append the sentence "With ``anchor_source="unit"`` the anchor is the GT first pixel at depth 1 and no depth map is read (scale-free 2D recipe, 2026-09-09)."

- [ ] **Step 4: Run the tests**

Run: `$RUN tests/test_scale_free_trajectory.py tests/test_projection_loss.py tests/test_2donly_g17_2d.py -q`
Expected: all pass (the two older files guard the legacy sources).

- [ ] **Step 5: Commit**

```bash
git add model/losses/geometric.py tests/test_scale_free_trajectory.py
git commit -m "TrajectoryProjectionLoss: 'unit' anchor — GT first pixel at depth 1, no depth map (scale-free 2D recipe)"
```

---

### Task 2: scale factor helpers + the three new config fields

**Files:**
- Modify: `model/losses/geometric.py` (after `backproject_points`, ~line 350)
- Modify: `config/opd_train.py` (`ModelParams` after line 115, `LossParams` after line 310, `Config` after line 353)
- Test: `tests/test_scale_free_trajectory.py` (append)

**Interfaces:**
- Produces: `trajectory_scale_factor(source: str, outputs, targets) -> Optional[torch.Tensor]` returning `(B,)` float or `None`; `apply_trajectory_scale(outputs, scale) -> ModelOutputs` (mutates and returns `outputs`).
- Produces config fields: `ModelParams.trajectory_scale_free: bool = False`, `ModelParams.finetune_slice_input_channels: bool = False`, `LossParams.trajectory_scale_source: str = "unit"`, `Config.test_trajectory_scale: str = "pred_z_p"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_scale_free_trajectory.py`)

```python
from model.losses.geometric import apply_trajectory_scale, trajectory_scale_factor
import pytest


def _scale_fixture():
    B = 3
    traj = torch.randn(B, 5, 3)
    hyps = torch.randn(B, 2, 5, 3)
    outputs = ModelOutputs(
        mask_logits=torch.zeros(B, 1, 4, 4), trajectory_pred=traj.clone(),
        trajectory_hyps=hyps.clone(),
        point_3d_pred=torch.tensor([[0.1, 0.2, 0.7], [0.0, 0.0, 1.3], [0.3, 0.1, 2.0]]),
    )
    gt = torch.zeros(B, 5, 3); gt[:, 0, 2] = torch.tensor([0.5, 1.0, 1.5])
    targets = StepTargets(trajectory=gt)
    return outputs, targets, traj, hyps


def test_scale_factor_sources():
    outputs, targets, _, _ = _scale_fixture()
    assert trajectory_scale_factor("unit", outputs, targets) is None
    assert torch.allclose(trajectory_scale_factor("gt_z0", outputs, targets), torch.tensor([0.5, 1.0, 1.5]))
    s = trajectory_scale_factor("pred_z_p", outputs, targets)
    assert torch.allclose(s, torch.tensor([0.7, 1.3, 2.0])) and not s.requires_grad
    with pytest.raises(ValueError):
        trajectory_scale_factor("gt_z0", outputs, StepTargets())
    with pytest.raises(ValueError):
        trajectory_scale_factor("pred_z_p", ModelOutputs(mask_logits=outputs.mask_logits), targets)
    with pytest.raises(ValueError):
        trajectory_scale_factor("metres", outputs, targets)


def test_apply_scale_multiplies_pred_and_hyps_rowwise():
    outputs, targets, traj, hyps = _scale_fixture()
    s = torch.tensor([0.5, 1.0, 1.5])
    out = apply_trajectory_scale(outputs, s)
    assert out is outputs
    assert torch.allclose(out.trajectory_pred, traj * s.view(-1, 1, 1))
    assert torch.allclose(out.trajectory_hyps, hyps * s.view(-1, 1, 1, 1))
    # None scale and a model without a trajectory head are no-ops
    assert apply_trajectory_scale(outputs, None) is outputs
    o2 = ModelOutputs(mask_logits=outputs.mask_logits)
    assert apply_trajectory_scale(o2, s).trajectory_pred is None


def test_new_config_fields_default_off():
    from config.opd_train import Config, LossParams, ModelParams
    assert ModelParams.__dataclass_fields__["trajectory_scale_free"].default is False
    assert ModelParams.__dataclass_fields__["finetune_slice_input_channels"].default is False
    assert LossParams.__dataclass_fields__["trajectory_scale_source"].default == "unit"
    assert Config.__dataclass_fields__["test_trajectory_scale"].default == "pred_z_p"
```

- [ ] **Step 2: Run to verify they fail**

Run: `$RUN tests/test_scale_free_trajectory.py -q`
Expected: ImportError on `apply_trajectory_scale` (collection error).

- [ ] **Step 3: Implement the helpers**

In `model/losses/geometric.py`, directly after `backproject_points` (after its `return rays * depth.unsqueeze(-1)`), add:

```python
def trajectory_scale_factor(source: str, outputs, targets) -> typing.Optional[torch.Tensor]:
    """(B,) multiplier that turns the scale-free trajectory head's output
    (Δ̃ = Δ / z0, offsets in units of the anchor depth — spec 2026-09-09)
    into metres, or None for "no multiply".

    "unit"     -> None. 2D datasets (HOI4D / EPIC / ARCTIC): the projection
                  loss anchors at depth 1, so Δ̃ is compared as is.
    "gt_z0"    -> targets.trajectory[:, 0, 2]: the GT depth of the first
                  point (SF3D train/val — teacher forcing of ONE scalar).
    "pred_z_p" -> outputs.point_3d_pred[:, 2], detached: the model's own z_p
                  lift (test time; needs predict_point_depth + intrinsics).
    """
    if source == "unit":
        return None
    if source == "gt_z0":
        if targets.trajectory is None:
            raise ValueError(
                "trajectory_scale_source='gt_z0' needs a 3D GT trajectory in the "
                "batch — use 'unit' on 2D datasets"
            )
        return targets.trajectory[:, 0, 2].float()
    if source == "pred_z_p":
        if outputs.point_3d_pred is None:
            raise ValueError(
                "trajectory scale 'pred_z_p' needs point_3d_pred "
                "(predict_point_depth + intrinsics in the batch)"
            )
        return outputs.point_3d_pred[:, 2].detach().float()
    raise ValueError(f"unknown trajectory scale source {source!r} (unit|gt_z0|pred_z_p)")


def apply_trajectory_scale(outputs, scale: typing.Optional[torch.Tensor]):
    """Multiply outputs.trajectory_pred (B, N, 3) and trajectory_hyps
    (B, K, N, 3) by a per-row scale. In place on the dataclass; returns it.
    No-op for scale None or a model without a trajectory head."""
    if scale is None or outputs.trajectory_pred is None:
        return outputs
    s = scale.to(outputs.trajectory_pred.device, outputs.trajectory_pred.dtype).view(-1, 1, 1)
    outputs.trajectory_pred = outputs.trajectory_pred * s
    if outputs.trajectory_hyps is not None:
        outputs.trajectory_hyps = outputs.trajectory_hyps * s.unsqueeze(1)
    return outputs
```

In `config/opd_train.py`:

After `trajectory_dct_coeffs: int = 0` in `ModelParams` add:
```python
    # 2026-09-09 (RGB-only / scale-free spec): the trajectory head predicts
    # Δ̃ = Δ / z0 — offsets in units of the anchor depth — instead of metres.
    # The head is unchanged; the trainers multiply trajectory_pred by a
    # per-sample scale right after the forward (loss_params.
    # trajectory_scale_source at train/val, config.test_trajectory_scale at
    # test). Pair with loss_params.trajectory_proj_anchor "unit".
    trajectory_scale_free: bool = False
    # load_finetune_weights: when a checkpoint conv weight has MORE input
    # channels than the model's (a depth-fused FPN checkpoint into a
    # use_depth=false model), load its leading model-sized slice. The
    # concat order is [rgb, depth], so this is exactly the "concatenate
    # zeros for the depth channels" model. Off = such keys stay re-init.
    finetune_slice_input_channels: bool = False
```

After `depth_anchor_source: str = "input"` in `LossParams` add:
```python
    # 2026-09-09 scale-free head: train/val multiplier applied to
    # trajectory_pred. "unit" = none (2D datasets); "gt_z0" = the GT first-
    # point depth (SF3D). See model.losses.geometric.trajectory_scale_factor.
    trajectory_scale_source: str = "unit"
```

After `test_pred_threshold: float = 0.5` in `Config` add:
```python
    # 2026-09-09 scale-free head: test-time multiplier for trajectory_pred.
    # "pred_z_p" = the model's own z_p (the deployable number); "gt_z0" =
    # oracle scale (shape-only metrics); "unit" = none (2D datasets).
    test_trajectory_scale: str = "pred_z_p"
```

- [ ] **Step 4: Run the tests**

Run: `$RUN tests/test_scale_free_trajectory.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add model/losses/geometric.py config/opd_train.py tests/test_scale_free_trajectory.py
git commit -m "Scale-free trajectory: trajectory_scale_factor/apply_trajectory_scale + config fields"
```

---

### Task 3: wire the scale into `_common_step` and the SF3D `test_step`

**Files:**
- Modify: `train_OPDReal_better.py:20-24` (import), `:206-209` (after the forward in `_common_step`)
- Modify: `train_SF3D_better.py:15-19` (import), `:323-336` (no-camera branch), `:353-356` (after the forward in `test_step`), `:631-635` (proj2d diagnostic lift)

**Interfaces:**
- Consumes: `trajectory_scale_factor`, `apply_trajectory_scale` (Task 2); `model_params.trajectory_scale_free`, `loss_params.trajectory_scale_source`, `config.test_trajectory_scale`.

- [ ] **Step 1: `_common_step`** — in `train_OPDReal_better.py` extend the import to

```python
from model.losses.geometric import (
    ScrewConsistencyLoss,
    TrajectoryProjectionLoss,
    apply_trajectory_scale,
    normalized_intrinsics,
    trajectory_scale_factor,
)
```

and right after
```python
        outputs = self(
            img, depth, tokenized_words, mask_gt, point_gt_norm, motion_gt,
            motion_type_input, K_norm,
        )
```
insert
```python
        # Scale-free head (2026-09-09 spec): the head emits Δ̃ = Δ / z0; put
        # the metric scale back BEFORE any loss reads trajectory_pred. "unit"
        # (2D datasets) is a no-op; "gt_z0" (SF3D) multiplies by the GT
        # first-point depth — a constant, so the normalized trajectory loss
        # is exactly the normalized loss on the dimensionless target.
        if getattr(self.model_params, "trajectory_scale_free", False):
            outputs = apply_trajectory_scale(
                outputs,
                trajectory_scale_factor(
                    getattr(self.loss_params, "trajectory_scale_source", "unit"),
                    outputs, targets,
                ),
            )
```

- [ ] **Step 2: `test_step`** — in `train_SF3D_better.py` extend the import to

```python
from model.losses.geometric import (
    apply_trajectory_scale,
    backproject_points,
    normalized_intrinsics,
    project_points,
    trajectory_scale_factor,
)
from model.targets import StepTargets
```

In the no-camera `else:` branch, change `motion_origin_3d_gt, intrinsic_matrix = None, None` to
```python
            motion_origin_3d_gt, intrinsic_matrix, trajectory_gt = None, None, None
```

Right after
```python
        with torch.no_grad():
            outputs = self(
                img, depth, tokenized_words, None, None, None, None, K_norm
            )
```
insert
```python
        # Scale-free head: metres = scale * Δ̃ (config.test_trajectory_scale:
        # the model's own z_p by default; "gt_z0" = oracle scale). Kept for
        # the proj2d diagnostic below, whose anchor must use the SAME scale.
        _traj_scale = None
        if getattr(self.model_params, "trajectory_scale_free", False):
            _traj_scale = trajectory_scale_factor(
                getattr(self.config, "test_trajectory_scale", "pred_z_p"),
                outputs, StepTargets(trajectory=trajectory_gt),
            )
            outputs = apply_trajectory_scale(outputs, _traj_scale)
```

In the proj2d diagnostic replace
```python
                _grid = (_uv * 2.0 - 1.0).view(1, 1, 1, 2)
                _z = F.grid_sample(
                    depth[i:i + 1].float().to(_dev), _grid, align_corners=False
                ).view(1)
```
with
```python
                if _traj_scale is not None:
                    # Scale-free: lift the anchor with the scale applied to
                    # the curve (projection is invariant to the joint rescale,
                    # so this equals the unit-anchor projection of Δ̃).
                    _z = _traj_scale[i:i + 1].to(_dev)
                else:
                    _grid = (_uv * 2.0 - 1.0).view(1, 1, 1, 2)
                    _z = F.grid_sample(
                        depth[i:i + 1].float().to(_dev), _grid, align_corners=False
                    ).view(1)
```

- [ ] **Step 3: Regression tests still pass**

Run: `$RUN tests/test_SF3D_better.py tests/test_OPDReal_better.py tests/test_2donly_g17_2d.py tests/test_step_targets.py -q`
Expected: all pass (defaults keep every path unchanged; the functional check of the new path is the pod smoke in Task 5).

- [ ] **Step 4: Commit**

```bash
git add train_OPDReal_better.py train_SF3D_better.py
git commit -m "Trainers: apply the scale-free trajectory scale after the forward (gt_z0 train, z_p test); proj2d diag uses the same scale"
```

---

### Task 4: RGB-only — `load_depth=False`, `depth=None` forward, loader column slicing

**Files:**
- Modify: `datasets/scenefun3d.py:60-62` (init signature), `:162` (attribute), `:504-508` (depth block head)
- Modify: `datasets/scenefun3d_datamodule.py:32-33` (signature), `:62` (attribute), `:110` (pass-through)
- Modify: `train_OPDReal_better.py:1638-1661` (`load_finetune_weights`) + a module-level `slice_input_channels`
- Test: `tests/test_scale_free_trajectory.py` (append)

**Interfaces:**
- Produces: `SF3DDataset(..., load_depth: bool = True)`, `SF3DDataModule(..., load_depth: bool = True)`; `slice_input_channels(weight, target) -> Optional[Tensor]` in `train_OPDReal_better.py`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_scale_free_trajectory.py`)

```python
from test_split_heads import _inputs, _make_cris  # stub-backbone CRIS (tests/ is on sys.path under pytest)


def test_rgb_only_model_runs_with_depth_none():
    model = _make_cris(use_depth=False, predict_point_depth=False)
    assert model.depth_encoder is None
    img, _depth, word, mask, point, motion = _inputs(B=2, size=64)
    out = model(img, None, word, mask, point, motion)
    assert out.mask_logits.shape[0] == 2 and out.trajectory_pred.shape == (2, 20, 3)


def test_slice_input_channels_rule():
    from train_OPDReal_better import slice_input_channels
    depth_sd = _make_cris(use_depth=True).state_dict()
    rgb_sd = _make_cris(use_depth=False).state_dict()
    # the two FPN convs that see the concat differ only in input channels
    for k in ("neck.f3_v_proj.0.weight", "neck.f2_v_proj.0.weight"):
        assert depth_sd[k].shape[1] > rgb_sd[k].shape[1]
        sliced = slice_input_channels(depth_sd[k], rgb_sd[k])
        assert sliced.shape == rgb_sd[k].shape
        assert torch.equal(sliced, depth_sd[k][:, : rgb_sd[k].shape[1]])
    # not a conv, same shape, or FEWER channels -> None
    assert slice_input_channels(torch.zeros(4, 8), torch.zeros(4, 6)) is None
    assert slice_input_channels(torch.zeros(4, 8, 3, 3), torch.zeros(4, 8, 3, 3)) is None
    assert slice_input_channels(torch.zeros(4, 6, 3, 3), torch.zeros(4, 8, 3, 3)) is None
    assert slice_input_channels(torch.zeros(5, 8, 3, 3), torch.zeros(4, 6, 3, 3)) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `$RUN tests/test_scale_free_trajectory.py -q`
Expected: `test_rgb_only_model_runs_with_depth_none` may already pass (the forward guards `depth is None`; keep it as the regression guard); `test_slice_input_channels_rule` fails with ImportError.

- [ ] **Step 3: Dataset `load_depth`**

`datasets/scenefun3d.py` — add the parameter after `fast_pipeline: bool = False,` in `SF3DDataset.__init__`:
```python
        load_depth: bool = True,
```
and after `self.fast_pipeline = fast_pipeline`:
```python
        # RGB-only runs (2026-09-09 spec): skip the depth PNG decode and emit
        # a zero (1, H, W) map in the tuple slot — the batch layout is
        # unchanged and nothing downstream needs a None check.
        self.load_depth = load_depth
```
In `__getitem__` replace
```python
        depth_image_tensor = None
        if frame_blob is not None:
            depth_np_uint16 = cv2.imdecode(
```
with
```python
        depth_image_tensor = None
        if not self.load_depth:
            depth_image_tensor = torch.zeros((1, target_h, target_w), dtype=torch.float32)
        elif frame_blob is not None:
            depth_np_uint16 = cv2.imdecode(
```
(the later `elif depth_image_filename:` and `if depth_image_tensor is None:` blocks are unchanged — with zeros already set they are skipped).

`datasets/scenefun3d_datamodule.py` — add `load_depth: bool = True,` after `fast_pipeline: bool = False,` in `__init__`; after `self.fast_pipeline = fast_pipeline` add
```python
        # RGB-only runs: no depth decode (see SF3DDataset.load_depth).
        self.load_depth = load_depth
```
and in the `SF3DDataset(...)` call add `load_depth=self.load_depth,` after `fast_pipeline=self.fast_pipeline,`.

- [ ] **Step 4: Loader slicing**

In `train_OPDReal_better.py`, add above `class OPDRealTrainingModule` (module level):

```python
def slice_input_channels(weight: torch.Tensor, target: torch.Tensor) -> typing.Optional[torch.Tensor]:
    """Leading-columns slice of a conv weight whose ONLY mismatch with
    ``target`` is more input channels (dim 1). Used to load a depth-fused
    FPN checkpoint into a use_depth=false model: the fusion concat is
    [rgb, depth], so the first target.shape[1] columns are the RGB ones and
    the result equals feeding zeros for the depth channels. None when the
    rule does not apply (then the key is skipped as before)."""
    if weight.dim() != 4 or target.dim() != 4:
        return None
    if weight.shape[0] != target.shape[0] or weight.shape[2:] != target.shape[2:]:
        return None
    if weight.shape[1] <= target.shape[1]:
        return None
    return weight[:, : target.shape[1]].clone()
```

In `load_finetune_weights` replace
```python
        pretrained_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_state_dict and v.size() == model_state_dict[k].size()
        }
```
with
```python
        slice_in = getattr(self.model_params, "finetune_slice_input_channels", False)
        pretrained_dict = {}
        sliced_keys = []
        for k, v in state_dict.items():
            if k not in model_state_dict:
                continue
            tgt = model_state_dict[k]
            if v.size() == tgt.size():
                pretrained_dict[k] = v
            elif slice_in:
                s = slice_input_channels(v, tgt)
                if s is not None:
                    pretrained_dict[k] = s
                    sliced_keys.append(f"{k} {tuple(v.shape)} -> {tuple(tgt.shape)}")
        if sliced_keys:
            print(f"✂️  Sliced input channels (finetune_slice_input_channels): {sliced_keys}")
```

- [ ] **Step 5: Run the tests**

Run: `$RUN tests/test_scale_free_trajectory.py tests/test_split_heads.py tests/test_OPDReal_better.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add datasets/scenefun3d.py datasets/scenefun3d_datamodule.py train_OPDReal_better.py tests/test_scale_free_trajectory.py
git commit -m "RGB-only: SF3DDataset load_depth=False (zeros, no decode); finetune_slice_input_channels loader rule"
```

---

### Task 5: configs for the four datasets + pod smoke runs

**Files:**
- Create: `config/hoi4d_v2_rgb_scalefree.yaml`, `config/epic_v1_rgb_scalefree.yaml`, `config/arctic_v1_rgb_scalefree.yaml`, `config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml`, `config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml`

**Interfaces:**
- Consumes every field from Tasks 1–4.

- [ ] **Step 1: Generate the 2D configs** — run this on the Mac from `SegAffordance/` (pure text edits, keeps the comments):

```bash
python3 - <<'EOF'
import re, pathlib
def make(src, dst, exp, header):
    t = pathlib.Path(src).read_text().splitlines()
    t[0] = header
    t = "\n".join(t) + "\n"
    subs = [
        (r'use_depth: true[^\n]*', 'use_depth: false                 # RGB-only (2026-09-09 spec): no DepthEncoder, FPN sees RGB features only'),
        (r'(    trajectory_dct_coeffs: 6[^\n]*\n)', r'\1    trajectory_scale_free: true    # head emits Δ̃ = Δ/z0 (units of anchor depth); no metric scale on the 2D path\n'),
        (r'depth_anchor_weight: 0\.5[^\n]*', 'depth_anchor_weight: 0.0       # OFF: no depth map to tether z_p to (RGB-only)'),
        (r'trajectory_proj_anchor: "gt_point"[^\n]*', 'trajectory_proj_anchor: "unit"       # GT 2D first point at depth 1 — scale-free, no depth map, never drops a row'),
        (r'(    depth_anchor_source: "gt_point"[^\n]*\n)', r'\1    trajectory_scale_source: "unit"     # 2D dataset: no multiply (Δ̃ compared as is)\n'),
        (r'pred_pred_art_radius_floor: 0\.10[^\n]*', 'pred_pred_art_radius_floor: 0.15 # 0.10 m expressed in units of anchor depth (z0 ~ 0.7 m on hand data)'),
        (r'(    test_pred_threshold[^\n]*\n)', r'\1    test_trajectory_scale: "unit"    # 2D dataset: MA vs a placeholder 3D track is not meaningful anyway\n'),
        (r'(    manual_seed: 42\n)', r'\1    test_trajectory_scale: "unit"    # 2D dataset: no multiply at test either\n'),
        (r'(  fast_pipeline: true\n)', r'\1  load_depth: false             # RGB-only: skip the depth PNG decode, zeros in the tuple slot\n'),
    ]
    for pat, rep in subs:
        t, n = re.subn(pat, rep, t, count=1)
    t = re.sub(r'experiments/[0-9]{8}_[a-z0-9_]+/', f'experiments/{exp}/', t)
    pathlib.Path(dst).write_text(t)
    print(dst, "ok")
make("config/hoi4d_v2_teacher_forcing.yaml", "config/hoi4d_v2_rgb_scalefree.yaml", "20260909_hoi4d_2d_v2_rgb_scalefree",
     "# HOI4D v2 `rgb_scalefree` (2026-09-09): the teacher_forcing recipe with NO depth (use_depth false, load_depth false) and the SCALE-FREE trajectory head (Δ̃ = Δ/z0; projection anchor = GT first pixel at depth 1). Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md")
make("config/epic_v1_smoke.yaml", "config/epic_v1_rgb_scalefree.yaml", "20260909_epic_v1_rgb_scalefree",
     "# EPIC v1 `rgb_scalefree` (2026-09-09): the HOI4D rgb_scalefree recipe on the EPIC/VISOR 2D LMDB (359 records, no depth). Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md")
make("config/arctic_v1_smoke.yaml", "config/arctic_v1_rgb_scalefree.yaml", "20260909_arctic_v1_rgb_scalefree",
     "# ARCTIC v1 `rgb_scalefree` (2026-09-09): the HOI4D rgb_scalefree recipe on the ARCTIC 2D LMDB (2,559 strokes; the object-only depth is NOT used). Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md")
EOF
```

Then verify each file with `grep -n "use_depth\|trajectory_scale_free\|depth_anchor_weight\|trajectory_proj_anchor\|trajectory_scale_source\|radius_floor\|test_trajectory_scale\|load_depth\|experiments/" config/hoi4d_v2_rgb_scalefree.yaml` — every key must appear exactly once (the `test_pred_threshold` line does not exist in these configs, so `test_trajectory_scale` lands after `manual_seed` under `config:`; if it appears twice, delete one). Confirm the `config:` block indentation (4 spaces) matches its neighbours.

- [ ] **Step 2: Generate the SF3D configs**

```bash
python3 - <<'EOF'
import re, pathlib
src = pathlib.Path("config/sf3d_train_runpod_g19_dct_ft_hoi4d_tf.yaml").read_text().splitlines()
src[0] = "# SF3D g19_dct `rgb_scalefree` SCRATCH arm (2026-09-09): the g19_dct recipe with NO depth (use_depth false, load_depth false) and the scale-free trajectory head — metres = GT z0 * Δ̃ at train/val (trajectory_scale_source gt_z0), z_p * Δ̃ at test. The RGB-only counterpart of the 25.98-MA scratch row in STATE.md. Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md"
t = "\n".join(src) + "\n"
t = re.sub(r'  finetune_from_path: [^\n]*\n', '', t, count=1)          # scratch arm
subs = [
    (r'use_depth: true[^\n]*', 'use_depth: false                 # RGB-only (2026-09-09 spec)'),
    (r'(    trajectory_dct_coeffs: 6[^\n]*\n)', r'\1    trajectory_scale_free: true    # head emits Δ̃ = Δ/z0; the trainer multiplies by z0 (GT at train, z_p at test)\n'),
    (r'(    trajectory_loss_normalized: true[^\n]*\n)', r'\1    trajectory_scale_source: "gt_z0"    # SF3D: metres = GT first-point depth * Δ̃ (teacher forcing of one scalar)\n'),
    (r'(    manual_seed: 42\n)', r'\1    test_trajectory_scale: "pred_z_p" # deployable number: the model\'s own z_p sets the scale (run test again with gt_z0 for the oracle-scale MA)\n'),
    (r'(  fast_pipeline: true\n)', r'\1  load_depth: false             # RGB-only: skip the depth PNG decode\n'),
]
for pat, rep in subs:
    t, n = re.subn(pat, rep, t, count=1); assert n == 1, pat
t = re.sub(r'experiments/[0-9]{8}_[a-z0-9_]+/', 'experiments/20260909_sf3d_g19_dct_rgb_scalefree/', t)
pathlib.Path("config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml").write_text(t)
ft = t.replace("experiments/20260909_sf3d_g19_dct_rgb_scalefree/", "experiments/20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d/")
ft = ft.replace("model:\n", 'model:\n  finetune_from_path: "/workspace/SegAffordance/experiments/20260909_hoi4d_2d_v2_rgb_scalefree/checkpoints/FILL_IN_BEST.ckpt"   # HOI4D rgb_scalefree arm best (fill in after that run)\n', 1)
ft = ft.splitlines(); ft[0] = "# SF3D g19_dct `rgb_scalefree` POST-TRAINING arm (2026-09-09): the scratch rgb_scalefree recipe initialized from the HOI4D v2 rgb_scalefree arm (same model_params -> loads 1:1). RGB-only counterpart of the 31.13-MA record row. Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md"
pathlib.Path("config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml").write_text("\n".join(ft) + "\n")
print("ok")
EOF
grep -n "finetune_from_path\|use_depth\|trajectory_scale\|test_trajectory_scale\|load_depth\|experiments/" config/sf3d_train_runpod_g19_dct_rgb_scalefree*.yaml
```
Expected: the scratch file has no `finetune_from_path`; the ft file has the FILL_IN placeholder; every other key appears once per file.

- [ ] **Step 3: Pod smoke — one train + one val step on each 2D dataset** (dev pod, small batch, compile off)

```bash
for c in hoi4d_v2_rgb_scalefree epic_v1_rgb_scalefree arctic_v1_rgb_scalefree; do
  bash runpod/dev.sh run "python train_SF3D_better.py fit --config config/$c.yaml --trainer.fast_dev_run true --model.model_params.compile_model false --data.batch_size_train 8 --data.batch_size_val 8 --data.num_workers_train 4 --data.num_workers_val 2 2>&1 | tail -15"
done
```
Expected per config: the run reaches the end of `fast_dev_run` without an exception, and the printed train step shows `train/L_traj_proj` with a NON-zero value (the unit anchor keeps every row; on ARCTIC/EPIC the old recipe would have logged 0). If `L_traj_proj` is missing, the projection loss returned early — check `trajectory_proj_anchor: "unit"` landed under `loss_params`.

- [ ] **Step 4: Pod smoke — SF3D scratch config, fit then test** (exercises `gt_z0` and `pred_z_p`)

```bash
bash runpod/dev.sh run "python train_SF3D_better.py fit --config config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml --trainer.fast_dev_run true --model.model_params.compile_model false --data.batch_size_train 8 --data.batch_size_val 8 --data.num_workers_train 4 --data.num_workers_val 2 2>&1 | tail -15"
bash runpod/dev.sh run "python train_SF3D_better.py test --config config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml --trainer.limit_test_batches 2 --model.model_params.compile_model false --data.batch_size_val 8 --data.num_workers_val 2 2>&1 | grep -E 'test/(traj|proj2d|point3d|MA|mean_acc)|Error|error' | head -20"
```
Expected: fit prints `train/L_trajectory` (finite); test prints the trajectory metrics (finite, random-init values) and `test/proj2d_err`. Then the oracle variant must also run:
```bash
bash runpod/dev.sh run "python train_SF3D_better.py test --config config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml --trainer.limit_test_batches 2 --model.model_params.compile_model false --model.config.test_trajectory_scale gt_z0 --data.batch_size_val 8 --data.num_workers_val 2 2>&1 | grep -E 'test/(traj|proj2d)|Error' | head"
```

- [ ] **Step 5: Commit**

```bash
git add config/hoi4d_v2_rgb_scalefree.yaml config/epic_v1_rgb_scalefree.yaml config/arctic_v1_rgb_scalefree.yaml config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml
git commit -m "Configs: RGB-only scale-free arms for HOI4D, EPIC, ARCTIC and SF3D (scratch + ft-from-HOI4D); smoke-tested on the dev pod"
```

---

### Task 6: STATE.md wrap + commit

**Files:**
- Modify: `STATE.md` (new section at the top of the dated log, below the checkpoint table; update "Last update")

- [ ] **Step 1: Add the section** (above `## DONE 2026-09-08 ~21:45 local: ARCTIC 2D dataset v1 BUILT`):

```markdown
## LANDED 2026-09-09: RGB-only model + scale-free trajectory head (code + configs, no runs yet)

User decisions: NO depth anywhere (all datasets, `use_depth: false`,
`load_depth: false`); the 2D path does not learn scale. Spec
`docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md`,
explainers `docs/slides/2026-09-09_*explainer.html`. What changed:
`TrajectoryProjectionLoss(anchor_source="unit")` (GT first pixel at depth 1,
no depth map, never drops a row — the old recipe dropped 80% of ARCTIC and
100% of EPIC rows silently); `model_params.trajectory_scale_free` (head
emits Δ̃ = Δ/z0; trainers multiply by `loss_params.trajectory_scale_source`
= gt_z0 on SF3D / unit on 2D data, `config.test_trajectory_scale` = pred_z_p
at test, gt_z0 = oracle); `SF3DDataset(load_depth=False)`;
`finetune_slice_input_channels` loader rule (depth ckpt -> RGB-only model =
the "concat zeros" model). Tests `tests/test_scale_free_trajectory.py`;
fast_dev_run smoke passed on HOI4D/EPIC/ARCTIC/SF3D configs
(`config/*_rgb_scalefree*.yaml`). NEXT: the fresh chain —
`20260909_hoi4d_2d_v2_rgb_scalefree` (HOI4D, ~1 h) ->
`20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d` (fill in the ckpt path) and
the scratch arm `20260909_sf3d_g19_dct_rgb_scalefree`; compare with the
depth rows (scratch 25.98, tf init 31.13). Existing checkpoints are NOT
inits for these (heads in metres, FPN expects depth channels).
```

- [ ] **Step 2: Commit**

```bash
git add STATE.md
git commit -m "STATE: RGB-only + scale-free trajectory landed; next = fresh HOI4D -> SF3D chain"
```
