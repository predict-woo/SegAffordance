# Gen-6 Split Articulation Heads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the twist parameterization with split heads (type CE + axis direction + canonical 3D origin + direct 3D interaction point + K=1 trajectory), classical per-branch losses, an all-predicted consistency term, and flag-gated predicted-mask pooling — per `docs/superpowers/specs/2026-08-13-split-heads-gen6-design.md`.

**Architecture:** All new capability is flag-gated off `ModelParams`/`LossParams` dataclasses (`config/opd_train.py`); defaults preserve classical/OPD behaviour exactly. Two new 3D point heads share one `Point3DHead` module; the interaction-point head replaces the 2D heatmap on the SF3D arm and its output extends the condition vector consumed by the other heads. New losses live in `model/losses/split.py` (pure functions) and `model/losses/geometric.py` (the `pred_pred_art` consistency variant); the trainer `_common_step` in `train_OPDReal_better.py` wires them; SF3D eval metrics extend `train_SF3D_better.py`.

**Tech Stack:** PyTorch / PyTorch Lightning; pytest.

## Global Constraints

- Every new `ModelParams`/`LossParams` field defaults to classical behaviour (`use_origin_head=False`, `point_prediction_3d=False`, `pool_with_predicted_mask=False`, `axis_sign_agnostic=True`); OPD arms must be bit-identical with defaults.
- Tests run LOCALLY (Mac, CPU): `/private/tmp/claude-501/-Users-andyye-dev-ethz-workspace/ab498bc1-4197-41fb-9b67-6f93d9065aed/scratchpad/twistenv/bin/python -m pytest tests/<file> -q` — abbreviated below as `$PY -m pytest`.
- Mutating git ONLY from the Mac side (this repo is a mutagen mirror of the pod).
- The existing twist/WTA/screw code paths are NOT deleted or modified beyond None-tolerance; `tests/test_twist.py`, `tests/test_wta.py`, `tests/test_screw_consistency.py` must keep passing.
- All 3D quantities: camera frame, meters. `motion_type == 1` is revolute.
- Spec formulas (copy exactly): `q* = o_gt + ((p − o_gt)·d_gt) d_gt`; `L_pp = P(pris)·mean_i‖d_i×d̂‖² + P(rev)·mean_i(dist(d_i, axis line) − r̂)²` with the axis line `(q̂ − p̂, d̂)` in the trajectory's relative frame and `r̂` the anchor's (origin's) distance to that line.

---

### Task 1: `Point3DHead` module + `ModelOutputs` fields

**Files:**
- Modify: `model/layers.py` (add class after `OriginDepthHead`, ~line 453)
- Modify: `model/outputs.py`
- Test: `tests/test_split_heads.py` (create)

**Interfaces:**
- Produces: `Point3DHead(input_dim: int, hidden_dim: int = 256)`, `forward(condition: (B, D)) -> (B, 3)` — unconstrained absolute camera-frame point.
- Produces: `ModelOutputs.point_3d_pred: Optional[torch.Tensor]` (B, 3) and `ModelOutputs.origin_pred: Optional[torch.Tensor]` (B, 3), both default `None`; `ModelOutputs.point_logits` and `ModelOutputs.coords_hat` become `Optional[torch.Tensor] = None` (they are `None` when `point_prediction_3d` is on).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_split_heads.py
import torch

from model.layers import Point3DHead
from model.outputs import ModelOutputs


def test_point3d_head_shape():
    head = Point3DHead(input_dim=32, hidden_dim=16)
    out = head(torch.randn(4, 32))
    assert out.shape == (4, 3)
    # Unconstrained output: gradients reach the input.
    out.sum().backward()


def test_model_outputs_new_fields_default_none():
    o = ModelOutputs(mask_logits=torch.zeros(1, 1, 4, 4))
    assert o.point_3d_pred is None
    assert o.origin_pred is None
    assert o.point_logits is None
    assert o.coords_hat is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_split_heads.py -q`
Expected: FAIL — `ImportError: cannot import name 'Point3DHead'`.

- [ ] **Step 3: Implement**

In `model/layers.py`, after `OriginDepthHead`:

```python
class Point3DHead(nn.Module):
    """Absolute 3D point in camera coordinates (metres).

    Used twice by the gen-6 split arm: the interaction point (graspable
    element centroid, GT = trajectory_3d[0]) and the revolute joint origin
    (GT = q*, the axis point perpendicular to the interaction point).
    Unconstrained linear output — camera-frame positions have no box to
    project onto, and canonicalized targets keep the regression local.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.mlp(condition)
```

In `model/outputs.py`: change `point_logits: torch.Tensor` and `coords_hat: torch.Tensor` to `Optional[torch.Tensor] = None` (check dataclass field ordering — every field after `mask_logits` must have a default once these do; the later fields already default to `None`). Add, next to `origin_depth`:

```python
    #: (B, 3) absolute 3D interaction point, camera frame, metres — the
    #: split-arm replacement for point_logits/coords_hat
    #: (ModelParams.point_prediction_3d). None on the classical 2D path.
    point_3d_pred: Optional[torch.Tensor] = None
    #: (B, 3) absolute 3D joint origin, camera frame, metres — supervised
    #: toward q*, the GT-axis point perpendicular to the interaction point
    #: (ModelParams.use_origin_head). Meaningful for revolute only.
    origin_pred: Optional[torch.Tensor] = None
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_split_heads.py tests/test_step_targets.py -q`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add model/layers.py model/outputs.py tests/test_split_heads.py
git commit -m "gen-6: Point3DHead module + point_3d_pred/origin_pred output fields"
```

---

### Task 2: Segmenter wiring — 3D point mode, origin head, predicted-mask pooling

**Files:**
- Modify: `config/opd_train.py` (ModelParams)
- Modify: `model/segmenter.py`
- Test: `tests/test_split_heads.py` (extend)

**Interfaces:**
- Consumes: `Point3DHead` from Task 1.
- Produces: `ModelParams.use_origin_head: bool = False`, `ModelParams.point_prediction_3d: bool = False`, `ModelParams.pool_with_predicted_mask: bool = False`.
- Produces: in 3D mode, `CRIS.forward` returns `point_logits=None`, `coords_hat=None`, `point_3d_pred (B,3)`; with `use_origin_head`, `origin_pred (B,3)`. With `pool_with_predicted_mask=True`, train-mode forward works with `mask=None` and pools with the DETACHED predicted-mask sigmoid.
- Produces: attributes `CRIS.point_prediction_3d`, `CRIS.pool_with_predicted_mask` (read by trainer/tests).

- [ ] **Step 1: Add ModelParams fields**

In `config/opd_train.py`, after `use_motion_type_head` block:

```python
    # gen-6 split arm (docs/superpowers/specs/2026-08-13-split-heads-gen6-design.md).
    # Predict the interaction point DIRECTLY as an absolute 3D camera-frame
    # point (GT trajectory_3d[0]) instead of the 2D heatmap + soft-argmax:
    # the Projector emits the mask channel only, point_logits/coords_hat are
    # None, and the articulation heads' condition is extended with the
    # predicted 3D point instead of coords_hat. SF3D-only (needs 3D GT).
    point_prediction_3d: bool = False
    # Predict the revolute joint origin as an absolute 3D point, supervised
    # toward q* — the GT-axis point perpendicular to the interaction point.
    use_origin_head: bool = False
    # Remove the mask-pooling teacher forcing: train-time condition pooling
    # uses the DETACHED sigmoid of the predicted mask (what val/test always
    # did) instead of the GT mask. Detached so articulation losses cannot
    # steer the mask head through the pooling path.
    pool_with_predicted_mask: bool = False
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_split_heads.py`:

```python
import pytest

from config.opd_train import ModelParams
from model.segmenter import CRIS


def _params(**over):
    base = dict(
        clip_pretrain="",  # random-init CLIP (build_backbone tolerates missing file)
        word_len=17, depth_feat_channels=[8, 8], fpn_in=[64, 128, 128],
        fpn_out=[32, 64, 128], num_layers=1, num_head=2, dim_ffn=64,
        dropout=0.0, intermediate=False, proj_dropout=0.0,
        vae_latent_dim=8, vae_hidden_dim=32, num_motion_types=2,
        use_depth=True, use_cvae=False, use_trajectory_head=True,
        trajectory_delta_cumsum=True, use_twist_head=False,
        use_motion_head=True, use_motion_type_head=True,
    )
    base.update(over)
    return ModelParams(**base)


def _inputs(B=2, size=64):
    img = torch.randint(0, 255, (B, 3, size, size), dtype=torch.uint8)
    depth = torch.rand(B, 1, size, size)
    word = torch.randint(1, 100, (B, 17))
    mask = (torch.rand(B, 1, size, size) > 0.5).float()
    return img, depth, word, mask


@pytest.fixture(scope="module")
def split_model():
    m = CRIS(_params(point_prediction_3d=True, use_origin_head=True,
                     pool_with_predicted_mask=True))
    m.eval()
    return m


def test_3d_mode_output_shapes(split_model):
    img, depth, word, mask = _inputs()
    with torch.no_grad():
        out = split_model(img, depth, word, mask, None, None)
    assert out.point_logits is None and out.coords_hat is None
    assert out.point_3d_pred.shape == (2, 3)
    assert out.origin_pred.shape == (2, 3)
    assert out.motion_pred.shape == (2, 3)
    assert out.motion_type_logits.shape == (2, 2)
    assert out.trajectory_pred.shape == (2, 20, 3)
    assert out.mask_logits.shape[1] == 1


def test_predicted_mask_pooling_allows_mask_none(split_model):
    # With pool_with_predicted_mask, train mode must not need a GT mask.
    img, depth, word, _ = _inputs()
    split_model.train()
    try:
        out = split_model(img, depth, word, None, None, None)
    finally:
        split_model.eval()
    assert out.point_3d_pred.shape == (2, 3)


def test_pooling_detached_from_mask_head(split_model):
    # Gradient of an articulation output must NOT reach the mask projector
    # through the pooling path (detached sigmoid).
    img, depth, word, _ = _inputs()
    split_model.train()
    try:
        split_model.zero_grad(set_to_none=True)
        out = split_model(img, depth, word, None, None, None)
        out.point_3d_pred.sum().backward()
        proj_grads = [p.grad for p in split_model.proj.parameters()]
        assert all(g is None or torch.all(g == 0) for g in proj_grads)
    finally:
        split_model.eval()
        split_model.zero_grad(set_to_none=True)


def test_classical_2d_mode_unchanged():
    m = CRIS(_params())  # all new flags default off
    m.eval()
    img, depth, word, mask = _inputs()
    with torch.no_grad():
        out = m(img, depth, word, mask, None, None)
    assert out.point_logits is not None and out.coords_hat.shape == (2, 2)
    assert out.point_3d_pred is None and out.origin_pred is None
```

NOTE: if `build_backbone` refuses `clip_pretrain=""`, follow whatever
convention `tests/test_wta.py` / `tests/test_twist.py` already use to build
a small CRIS on CPU (they exercise the same constructor) and mirror it.

- [ ] **Step 3: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_split_heads.py -q`
Expected: new tests FAIL (`TypeError: unexpected keyword 'point_prediction_3d'` or attribute errors).

- [ ] **Step 4: Implement in `model/segmenter.py`**

In `__init__` (near the existing flag reads, before the Projector):

```python
        self.point_prediction_3d = getattr(model_params, "point_prediction_3d", False)
        self.use_origin_head = getattr(model_params, "use_origin_head", False)
        self.pool_with_predicted_mask = getattr(
            model_params, "pool_with_predicted_mask", False
        )
```

Change the Projector construction to emit the point channel only on the 2D path:

```python
        self.proj = Projector_Mult(
            state_dim,
            model_params.fpn_out[1] // 2,
            3,
            out_channels=1 if self.point_prediction_3d else 2,
            proj_dropout=model_params.proj_dropout,
        )
```

Condition-dimension bookkeeping (replace the `vae_condition_dim = vae_feature_dim + 2` line):

```python
        # 2D path: condition = features + coords_hat (2). 3D path: the point
        # head consumes the base features and its 3-dim output extends the
        # condition instead (spec: "the same role coords_hat played, in 3D").
        point_cond_dim = 3 if self.point_prediction_3d else 2
        vae_condition_dim = vae_feature_dim + point_cond_dim
```

After the trajectory/twist head constructions, add the two new heads (import `Point3DHead` in the existing `.layers` import list):

```python
        self.point_3d_head = (
            Point3DHead(
                input_dim=vae_condition_dim - 3, hidden_dim=model_params.vae_hidden_dim
            )
            if self.point_prediction_3d
            else None
        )
        self.origin_head = (
            Point3DHead(
                input_dim=vae_condition_dim, hidden_dim=model_params.vae_hidden_dim
            )
            if self.use_origin_head
            else None
        )
```

NOTE `vae_condition_dim - 3` is the base condition (features + type-hint
embedding if enabled) WITHOUT the point extension — the point head cannot
consume its own output. `origin_head` consumes the extended condition.

In `forward`, replace the maps/point block:

```python
        maps = self.proj(fq, state)
        mask_pred = maps[:, 0:1]
        if self.point_prediction_3d:
            point_pred = None
            coords_hat = None
        else:
            point_pred = maps[:, 1:2]
            coords_px = soft_argmax2d(point_pred)
            _, _, H_map, W_map = point_pred.shape
            coords_hat = coords_px / torch.tensor(
                [W_map, H_map], dtype=coords_px.dtype, device=coords_px.device
            )
```

Replace the pooling teacher-forcing block:

```python
        if self.training and not self.pool_with_predicted_mask:
            # Classical teacher forcing: pool under the GT mask.
            mask_for_pooling = F.interpolate(
                mask.float(), size=(fq_h, fq_w), mode="bilinear", align_corners=False
            )
        else:
            # Deployment-consistent pooling: the predicted mask, detached in
            # train mode so articulation losses cannot steer the mask head
            # through the pooling path (it learns only from DiceBCE).
            mask_prob = torch.sigmoid(mask_pred)
            if self.training:
                mask_prob = mask_prob.detach()
            mask_for_pooling = F.interpolate(
                mask_prob, size=(fq_h, fq_w), mode="bilinear", align_corners=False
            )
```

Replace the condition assembly (currently `vae_condition = torch.cat([vae_encoder_features, coords_hat], dim=1)` followed by the type-hint concat). New order — hint (if any) joins the BASE, the point extension comes last:

```python
        vae_condition = vae_encoder_features
        if self.use_motion_type_input:
            null_index = self.motion_type_embedding.num_embeddings - 1
            if motion_type_input is None:
                motion_type_input = torch.full(
                    (b,), null_index, dtype=torch.long, device=vae_condition.device
                )
            type_emb = self.motion_type_embedding(
                motion_type_input.to(vae_condition.device).long()
            )
            vae_condition = torch.cat([vae_condition, type_emb], dim=1)

        point_3d_pred = None
        if self.point_prediction_3d:
            point_3d_pred = self.point_3d_head(vae_condition)
            vae_condition = torch.cat([vae_condition, point_3d_pred], dim=1)
        else:
            vae_condition = torch.cat([vae_condition, coords_hat], dim=1)
```

CAREFUL: the 2D path previously concatenated `[features, coords_hat]` THEN
the type embedding; the hint is off in every live config
(`use_motion_type_input` is being retired), so the reorder only affects
column order inside a learned MLP input — note it in the commit message.

After the other heads run, add:

```python
        origin_pred = (
            self.origin_head(vae_condition) if self.origin_head is not None else None
        )
```

and extend the `ModelOutputs(...)` return with `point_3d_pred=point_3d_pred, origin_pred=origin_pred`.

Guard: `vae_encoder_features`/`vae_condition` also feed the CVAE; nothing else changes there.

- [ ] **Step 5: Run tests**

Run: `$PY -m pytest tests/test_split_heads.py tests/test_twist.py tests/test_wta.py tests/test_trajectory_head.py -q`
Expected: PASS (new tests + no twist-era regression).

- [ ] **Step 6: Commit**

```bash
git add config/opd_train.py model/segmenter.py tests/test_split_heads.py
git commit -m "gen-6: segmenter 3D point mode, origin head, flag-gated predicted-mask pooling"
```

---

### Task 3: Split loss functions — `q*` canonical origin loss, sign-sensitive axis loss

**Files:**
- Create: `model/losses/split.py`
- Modify: `model/losses/__init__.py` (export)
- Test: `tests/test_split_heads.py` (extend)

**Interfaces:**
- Produces: `perpendicular_foot(origin: (B,3), direction: (B,3), point: (B,3)) -> (B,3)` — `q* = o + ((p−o)·d̂)d̂`, direction normalized inside.
- Produces: `origin_canonical_loss(origin_pred: (B,3), origin_gt: (B,3), direction_gt: (B,3), point_gt: (B,3), motion_type: (B,)) -> scalar` — mean `‖q̂ − q*‖²` over revolute rows (`motion_type == 1`); a connected zero tensor when a batch has none.
- Produces: `axis_direction_loss(motion_pred: (B,3), motion_gt: (B,3), sign_agnostic: bool) -> scalar` — `1 − cos²` (classical) when `sign_agnostic`, `1 − cos` when not.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split_heads.py`:

```python
from model.losses.split import (
    axis_direction_loss,
    origin_canonical_loss,
    perpendicular_foot,
)


def test_q_star_is_perpendicular_foot_and_annotation_gauge_invariant():
    torch.manual_seed(0)
    o = torch.randn(8, 3)
    d = torch.nn.functional.normalize(torch.randn(8, 3), dim=1)
    p = torch.randn(8, 3)
    q_star = perpendicular_foot(o, d, p)
    # (q* - p) is perpendicular to the axis.
    assert torch.allclose(((q_star - p) * d).sum(-1), torch.zeros(8), atol=1e-5)
    # Sliding the annotated origin along the axis does not move q*.
    q_star_slid = perpendicular_foot(o + 3.7 * d, d, p)
    assert torch.allclose(q_star, q_star_slid, atol=1e-5)


def test_origin_loss_gauge_invariant_and_zero_at_target():
    torch.manual_seed(1)
    o = torch.randn(4, 3)
    d = torch.nn.functional.normalize(torch.randn(4, 3), dim=1)
    p = torch.randn(4, 3)
    ty = torch.ones(4, dtype=torch.long)
    q_star = perpendicular_foot(o, d, p)
    assert origin_canonical_loss(q_star, o, d, p, ty).item() < 1e-10
    pred = torch.randn(4, 3, requires_grad=True)
    l1 = origin_canonical_loss(pred, o, d, p, ty)
    l2 = origin_canonical_loss(pred, o - 2.2 * d, d, p, ty)
    assert torch.allclose(l1, l2, atol=1e-5)
    l1.backward()  # gradient flows


def test_origin_loss_prismatic_only_batch_is_zero_no_nan():
    pred = torch.randn(3, 3, requires_grad=True)
    ty = torch.zeros(3, dtype=torch.long)
    loss = origin_canonical_loss(
        pred, torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3), ty
    )
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_axis_loss_sign_sensitivity():
    a = torch.nn.functional.normalize(torch.randn(5, 3), dim=1)
    # Antiparallel: perfect under the classical 1-cos^2, ~2 under 1-cos.
    assert axis_direction_loss(-a, a, sign_agnostic=True).item() < 1e-6
    assert abs(axis_direction_loss(-a, a, sign_agnostic=False).item() - 2.0) < 1e-5
    assert axis_direction_loss(a, a, sign_agnostic=False).item() < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_split_heads.py -q`
Expected: FAIL — `ModuleNotFoundError: model.losses.split`.

- [ ] **Step 3: Implement `model/losses/split.py`**

```python
"""Per-branch losses of the gen-6 split articulation arm.

Spec: docs/superpowers/specs/2026-08-13-split-heads-gen6-design.md. The
origin target is CANONICALIZED rather than made gauge-invariant: q* is the
unique GT-axis point whose segment to the interaction point is
perpendicular to the axis, so all 3 output dimensions are constrained and
the target sits near the element (distance-to-line left the along-axis
component free to drift).
"""
import torch
import torch.nn.functional as F


def perpendicular_foot(
    origin: torch.Tensor, direction: torch.Tensor, point: torch.Tensor
) -> torch.Tensor:
    """q* = origin + ((point - origin) . d_hat) d_hat, all (B, 3).

    Invariant to sliding `origin` along the axis (annotation gauge).
    """
    d = F.normalize(direction, dim=-1, eps=1e-8)
    along = ((point - origin) * d).sum(-1, keepdim=True)
    return origin + along * d


def origin_canonical_loss(
    origin_pred: torch.Tensor,
    origin_gt: torch.Tensor,
    direction_gt: torch.Tensor,
    point_gt: torch.Tensor,
    motion_type: torch.Tensor,
) -> torch.Tensor:
    """Mean ||q_hat - q*||^2 over revolute rows; connected zero when none.

    Prismatic rows contribute nothing — a translation has no axis location,
    so the head receives no gradient from them.
    """
    q_star = perpendicular_foot(origin_gt, direction_gt, point_gt)
    sq = (origin_pred - q_star).pow(2).sum(-1)
    revolute = motion_type.to(sq.device) == 1
    if bool(revolute.any()):
        return sq[revolute].mean()
    # Zero that keeps the graph connected (same convention as the screw
    # losses' degenerate handling) so .backward() is always legal.
    return (origin_pred.sum() * 0.0)


def axis_direction_loss(
    motion_pred: torch.Tensor, motion_gt: torch.Tensor, sign_agnostic: bool
) -> torch.Tensor:
    """Classical 1 - cos^2 (antiparallel OK) or sign-sensitive 1 - cos.

    SF3D's stored axis sign is canonical (the GT trajectory is derived from
    it), so the gen-6 arm runs sign-sensitive; OPD keeps the classical form.
    """
    cos = F.cosine_similarity(motion_pred, motion_gt, dim=1, eps=1e-4)
    if sign_agnostic:
        return (1.0 - cos.pow(2)).mean()
    return (1.0 - cos).mean()
```

Export the three names from `model/losses/__init__.py` (add to the imports and `__all__`).

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_split_heads.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/losses/split.py model/losses/__init__.py tests/test_split_heads.py
git commit -m "gen-6: canonical q* origin loss + sign-sensitive axis loss"
```

---

### Task 4: `PredPredArticulationLoss` (all-predicted consistency) + factory registration

**Files:**
- Modify: `model/losses/geometric.py` (new class after `PredPredGeometricLoss`; register in `build_geometric_loss`)
- Modify: `config/opd_train.py` (LossParams field)
- Test: `tests/test_split_heads.py` (extend)

**Interfaces:**
- Consumes: `outputs.trajectory_pred (B,N,3 relative)`, `outputs.motion_pred (B,3)`, `outputs.motion_type_logits (B,2)`, `outputs.origin_pred (B,3)`, `outputs.point_3d_pred (B,3)`, `outputs.mask_logits` (zero-reference only). NO field of `targets` is read.
- Produces: `PredPredArticulationLoss(weight: float, degenerate_threshold: float = 1e-6)`, `forward(outputs, targets, depth=None) -> (total, {"L_geo_pred_pred_art": term})`; `build_geometric_loss` returns it for `geometric_loss == "pred_pred_art"`; `LossParams.pred_pred_art_weight: float = 0.5`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split_heads.py`:

```python
from model.losses.geometric import PredPredArticulationLoss, build_geometric_loss
from model.outputs import ModelOutputs
from model.targets import StepTargets


def _pp_outputs(traj, dhat, p_rev_logit, origin, anchor):
    B = traj.shape[0]
    logits = torch.stack(
        [torch.zeros(B), torch.full((B,), p_rev_logit)], dim=1
    )
    return ModelOutputs(
        mask_logits=torch.zeros(B, 1, 4, 4),
        motion_pred=dhat, motion_type_logits=logits,
        trajectory_pred=traj, origin_pred=origin, point_3d_pred=anchor,
    )


def test_pp_art_reads_no_targets_and_perfect_prismatic_scores_zero():
    B, N = 3, 20
    dhat = torch.nn.functional.normalize(torch.randn(B, 3), dim=1)
    t = torch.linspace(0, 0.4, N)[None, :, None]
    traj = dhat[:, None, :] * t                       # relative, parallel to dhat
    out = _pp_outputs(traj, dhat, p_rev_logit=-20.0,  # P(rev) ~ 0
                      origin=torch.randn(B, 3), anchor=torch.randn(B, 3))
    loss_fn = PredPredArticulationLoss(weight=0.5)
    total, terms = loss_fn(out, StepTargets())        # EMPTY targets: GT-free
    assert terms["L_geo_pred_pred_art"].item() < 1e-8


def _circle_traj(center_rel, dhat, start_rel, thetas):
    # Rotate start_rel about the axis (center_rel, dhat): relative curve.
    r0 = start_rel - center_rel
    axial = (r0 * dhat).sum() * dhat
    x = r0 - axial
    y = torch.cross(dhat, x, dim=-1)
    pts = [center_rel + axial + torch.cos(th) * x + torch.sin(th) * y
           for th in thetas]
    curve = torch.stack(pts)                          # (N, 3), absolute-rel
    return curve - curve[0:1]                         # first point pinned to 0


def test_pp_art_perfect_revolute_scores_zero_and_gauge_along_axis():
    dhat = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=1)
    anchor = torch.tensor([[0.5, 0.2, 1.0]])
    origin = torch.tensor([[0.1, 0.2, 1.0]])          # axis point near anchor
    center_rel = (origin - anchor)[0]
    traj = _circle_traj(center_rel, dhat[0], torch.zeros(3),
                        torch.linspace(0, 1.2, 20))[None]
    out = _pp_outputs(traj, dhat, p_rev_logit=20.0, origin=origin, anchor=anchor)
    loss_fn = PredPredArticulationLoss(weight=0.5)
    _, terms = loss_fn(out, StepTargets())
    assert terms["L_geo_pred_pred_art"].item() < 1e-6
    # Sliding the predicted origin ALONG the predicted axis changes nothing.
    out2 = _pp_outputs(traj, dhat, 20.0, origin + 2.5 * dhat, anchor)
    _, terms2 = loss_fn(out2, StepTargets())
    assert abs(terms2["L_geo_pred_pred_art"].item()
               - terms["L_geo_pred_pred_art"].item()) < 1e-6


def test_pp_art_soft_gate_selects_branch():
    # A straight-line trajectory: near-zero under P(rev)=0, positive under
    # P(rev)=1 (a line is not a constant-radius orbit about a nearby axis).
    dhat = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=1)
    t = torch.linspace(0, 0.4, 20)[None, :, None]
    traj = dhat[:, None, :] * t
    origin = torch.tensor([[0.0, 0.1, 0.0]])
    anchor = torch.zeros(1, 3)
    loss_fn = PredPredArticulationLoss(weight=1.0)
    _, pris = loss_fn(_pp_outputs(traj, dhat, -20.0, origin, anchor), StepTargets())
    _, rev = loss_fn(_pp_outputs(traj, dhat, 20.0, origin, anchor), StepTargets())
    assert pris["L_geo_pred_pred_art"].item() < 1e-8
    assert rev["L_geo_pred_pred_art"].item() > 1e-3


def test_pp_art_degenerate_trajectory_masked():
    dhat = torch.nn.functional.normalize(torch.randn(2, 3), dim=1)
    traj = torch.zeros(2, 20, 3, requires_grad=True)  # zero motion: no direction
    out = _pp_outputs(traj, dhat, 0.0, torch.randn(2, 3), torch.randn(2, 3))
    total, terms = PredPredArticulationLoss(weight=0.5)(out, StepTargets())
    assert terms["L_geo_pred_pred_art"].item() == 0.0
    total.backward()
    assert torch.isfinite(traj.grad).all()


def test_pp_art_missing_heads_noop_and_factory():
    out = ModelOutputs(mask_logits=torch.zeros(1, 1, 4, 4))
    total, terms = PredPredArticulationLoss(weight=0.5)(out, StepTargets())
    assert total.item() == 0.0 and terms == {}

    class LP:  # minimal LossParams stand-in
        geometric_loss = "pred_pred_art"
        pred_pred_art_weight = 0.5
    assert isinstance(build_geometric_loss(LP()), PredPredArticulationLoss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_split_heads.py -q`
Expected: FAIL — `ImportError: PredPredArticulationLoss`.

- [ ] **Step 3: Implement**

In `model/losses/geometric.py`, after `PredPredGeometricLoss`:

```python
class PredPredArticulationLoss(GeometricConsistencyLoss):
    """Gen-6 all-predicted consistency: trajectory <-> axis line <-> type.

    The classical line/circle residual forms with every input a prediction —
    possible only now that the origin is predicted (the cross-GT scheme had
    to teacher-force it). No field of ``targets`` is read: GT reaches each
    head only through its own direct loss, which is also what anchors the
    mutually-consistent-but-wrong failure mode.

    Everything lives in the trajectory's relative frame (first point = 0 =
    the predicted interaction point), so the predicted axis point is
    ``origin_pred - point_3d_pred``. The type gate is the SOFT predicted
    P(revolute) — coupling strengthens as the type head sharpens (same
    self-switching as PredPredGeometricLoss). Degenerate predicted curves
    (no displacement energy) are dropped from the batch mean: a zero curve
    satisfies any axis and would otherwise push the gate around.
    """

    def __init__(self, weight: float, degenerate_threshold: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.degenerate_threshold = degenerate_threshold

    def forward(self, outputs, targets, depth=None) -> LossTerms:
        required = (
            outputs.trajectory_pred, outputs.motion_pred,
            outputs.motion_type_logits, outputs.origin_pred,
            outputs.point_3d_pred,
        )
        if any(r is None for r in required):
            return self._zero(outputs.mask_logits), {}

        # fp32 throughout (precision-16 runs; ratios/norms under autocast).
        d = outputs.trajectory_pred.float()                       # (B, N, 3)
        dhat = F.normalize(outputs.motion_pred.float(), p=2, dim=1, eps=1e-8)
        dhat_n = dhat[:, None, :]

        # Prismatic: displacement energy perpendicular to the axis.
        l_line = torch.cross(d, dhat_n.expand_as(d), dim=-1).pow(2).sum(-1).mean(-1)

        # Revolute: constant distance to the predicted axis LINE, relative
        # frame. r_hat is the anchor's own distance (the first point sits at
        # 0 by construction).
        c = (outputs.origin_pred - outputs.point_3d_pred).float()[:, None, :]

        def _line_dist(x):
            rel = x - c
            along = (rel * dhat_n).sum(-1, keepdim=True)
            return (rel - along * dhat_n).norm(dim=-1)

        r_hat = _line_dist(torch.zeros_like(d[:, :1]))            # (B, 1)
        l_circle = (_line_dist(d) - r_hat).pow(2).mean(-1)        # (B,)

        p_rev = outputs.motion_type_logits.float().softmax(dim=-1)[:, 1]
        per_sample = (1.0 - p_rev) * l_line + p_rev * l_circle

        energy = d.pow(2).sum(dim=(-1, -2))
        valid = energy > self.degenerate_threshold
        if bool(valid.any()):
            term = per_sample[valid].mean()
        else:
            term = self._zero(outputs.mask_logits)
        return self.weight * term, {"L_geo_pred_pred_art": term}
```

In `build_geometric_loss`, before the final fallthrough:

```python
    if name == "pred_pred_art":
        return PredPredArticulationLoss(
            weight=getattr(loss_params, "pred_pred_art_weight", 0.5)
        )
```

In `config/opd_train.py` `LossParams`, next to `pred_pred_weight`:

```python
    # Read only by the "pred_pred_art" variant (gen-6): all-predicted
    # trajectory <-> axis-line <-> type consistency, soft type gate.
    pred_pred_art_weight: float = 0.5
```

Add `PredPredArticulationLoss` to `model/losses/__init__.py` imports/`__all__`, and update the module docstring's variant list at the top of `geometric.py` (`"cross_gt" | "pred_pred" | "pred_pred_art" | "projected" | "screw" | "none"`).

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_split_heads.py tests/test_geometric_losses.py tests/test_screw_consistency.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/losses/geometric.py model/losses/__init__.py config/opd_train.py tests/test_split_heads.py
git commit -m "gen-6: PredPredArticulationLoss — all-predicted consistency, soft type gate"
```

---

### Task 5: Trainer wiring — `L_point_3d`, `L_origin`, gated 2D point losses, axis-sign flag

**Files:**
- Modify: `config/opd_train.py` (LossParams fields)
- Modify: `train_OPDReal_better.py:165-220` (`_common_step`)
- Test: `tests/test_split_heads.py` (extend)

**Interfaces:**
- Consumes: `origin_canonical_loss`, `axis_direction_loss` (Task 3); segmenter 3D mode (Task 2).
- Produces: `LossParams.axis_sign_agnostic: bool = True`, `LossParams.origin_weight: float = 0.5`, `LossParams.point_3d_weight: float = 0.5`.
- Produces: log keys `{step}/L_point_3d`, `{step}/L_origin` (zero-valued when heads/GT absent — stable CSV columns); `{step}/L_point_map` and `{step}/L_coord` log zero on the 3D path.

- [ ] **Step 1: Add LossParams fields**

In `config/opd_train.py` `LossParams`, after `motion_type_weight`-adjacent fields:

```python
    # gen-6 split arm. axis_sign_agnostic True = classical 1 - cos^2
    # (antiparallel OK — OPD annotates only the axis LINE); False = 1 - cos,
    # for SF3D where the stored sign is canonical.
    axis_sign_agnostic: bool = True
    # MSE toward q* (the GT-axis point perpendicular to the interaction
    # point), revolute rows only. Consumed when the model has an origin head.
    origin_weight: float = 0.5
    # MSE of the direct 3D interaction point vs GT trajectory_3d[0].
    # Replaces point_map+coord (both zero on the 3D path) at the same
    # total weight budget.
    point_3d_weight: float = 0.5
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_split_heads.py` (synthetic 13-tuple SF3D batch through the Lightning module on CPU; mirror the batch/module construction conventions of `tests/test_SF3D_better.py` if they differ):

```python
from config.opd_train import Config, LossParams, OptimizerParams
from train_SF3D_better import SF3DTrainingModule


def _split_module():
    lp = LossParams(
        bce_weight=0.5, dice_weight=0.5, mask_weight=0.5,
        point_map_weight=0.0, coord_weight=0.0, vae_weight=0.5,
        motion_type_weight=0.5, point_sigma=8.0, vae_beta=0.01,
        trajectory_weight=0.5, geometric_loss="pred_pred_art",
        pred_pred_art_weight=0.5, axis_sign_agnostic=False,
        origin_weight=0.5, point_3d_weight=0.5,
    )
    op = OptimizerParams(lr=1e-5, weight_decay=0.0,
                         scheduler_milestones=[10], scheduler_gamma=0.1)
    cfg = Config(log_image_interval_steps=0, input_size=[64, 64],
                 enable_wandb=False, val_vis_samples=0, manual_seed=0)
    return SF3DTrainingModule(
        _params(point_prediction_3d=True, use_origin_head=True,
                pool_with_predicted_mask=True),
        lp, op, cfg,
    )


def _sf3d_batch(B=2, N=20, size=64):
    traj = torch.randn(B, N, 3).cumsum(dim=1) * 0.01 + torch.tensor([0., 0., 1.5])
    return (
        torch.randint(0, 255, (B, 3, size, size), dtype=torch.uint8),
        torch.rand(B, 1, size, size),
        ["open the door"] * B,
        (torch.rand(B, 1, size, size) > 0.5).float(),
        torch.zeros(B, 4),                                  # bbox (unused)
        torch.rand(B, 2),                                   # point_norm
        torch.nn.functional.normalize(torch.randn(B, 3), dim=1),
        torch.tensor([1, 0][:B] if B <= 2 else [1] * B),    # types: rot, trans
        torch.tensor([[64, 64]] * B),
        ["f.png"] * B,
        torch.randn(B, 3) + torch.tensor([0., 0., 1.5]),    # origin_3d
        torch.eye(3).expand(B, 3, 3),
        traj,
    )


def test_common_step_split_arm_finite_and_logs_new_terms():
    m = _split_module()
    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(
        name, float(value) if torch.is_tensor(value) else value
    )
    loss = m._common_step(_sf3d_batch(), 0, "train")
    assert torch.isfinite(loss)
    loss.backward()
    for key in ("train/L_point_3d", "train/L_origin",
                "train/L_geo_pred_pred_art", "train/L_motion_type"):
        assert key in logged, key
    assert logged["train/L_point_3d"] > 0.0
    # 2D point losses are zero-valued on the 3D path (stable CSV columns).
    assert logged["train/L_point_map"] == 0.0
    assert logged["train/L_coord"] == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `$PY -m pytest tests/test_split_heads.py::test_common_step_split_arm_finite_and_logs_new_terms -q`
Expected: FAIL (crash on `coords_hat=None` in `_common_step`, or missing log keys).

- [ ] **Step 4: Implement in `train_OPDReal_better.py` `_common_step`**

(a) Zero-reference device: replace every `torch.zeros((), device=coords_hat.device)` in `_common_step` with `torch.zeros((), device=mask_pred_logits.device)` (`coords_hat` may be `None` now).

(b) Gate the 2D point losses (replace the `point_gt_heatmap`/`L_point_map`/`L_coord` block):

```python
        zero = torch.zeros((), device=mask_pred_logits.device)
        if point_pred_logits is not None:
            point_gt_heatmap = make_gaussian_map(
                point_gt_norm, H_map, W_map,
                sigma=self.loss_params.point_sigma,
                device=point_pred_logits.device,
            )
            L_point_map = self.point_map_loss_fn(point_pred_logits, point_gt_heatmap)
            L_coord = self.coord_loss_fn(
                coords_hat, point_gt_norm.to(coords_hat.device)
            )
        else:
            # 3D point mode: the heatmap channel does not exist; the direct
            # 3D term below replaces both. Zero-valued (not absent) so the
            # CSV logger keeps stable columns.
            L_point_map, L_coord = zero, zero
```

(c) Direct 3D point + origin terms, inserted right after the `total_loss = (...)` sum (which is unchanged), extending it:

```python
        L_point_3d = zero
        if outputs.point_3d_pred is not None and targets.trajectory is not None:
            anchor_gt = targets.trajectory.to(outputs.point_3d_pred.device)[:, 0]
            L_point_3d = F.mse_loss(outputs.point_3d_pred, anchor_gt)
        L_origin = zero
        if (
            outputs.origin_pred is not None
            and targets.motion_origin_3d is not None
            and targets.trajectory is not None
        ):
            dev = outputs.origin_pred.device
            L_origin = origin_canonical_loss(
                outputs.origin_pred,
                targets.motion_origin_3d.to(dev).float(),
                motion_gt.to(dev).float(),
                targets.trajectory.to(dev)[:, 0].float(),
                motion_type_gt,
            )
        total_loss = (
            total_loss
            + self.loss_params.point_3d_weight * L_point_3d
            + self.loss_params.origin_weight * L_origin
        )
        grad_terms["point_3d"] = self.loss_params.point_3d_weight * L_point_3d
        grad_terms["origin"] = self.loss_params.origin_weight * L_origin
```

and log both with the same `self.log(f"{step_type}/L_point_3d", ...)` /
`.../L_origin` pattern as the neighbouring terms.

(d) Axis loss deviation flag — replace the `1 - cos^2` else-branch body:

```python
        else:
            L_motion = axis_direction_loss(
                motion_pred, motion_gt.to(motion_pred.device),
                sign_agnostic=getattr(self.loss_params, "axis_sign_agnostic", True),
            )
            L_vae, L_recon, L_kld = L_motion, L_motion, zero
```

(e) Imports: `from model.losses.split import axis_direction_loss, origin_canonical_loss`; use the new defaults via `getattr(self.loss_params, "point_3d_weight", 0.5)` / `"origin_weight"` if older configs are loaded from checkpoints (`save_hyperparameters` re-instantiates LossParams — dataclass defaults cover it, plain attribute access is fine).

- [ ] **Step 5: Run tests**

Run: `$PY -m pytest tests/test_split_heads.py tests/test_OPDReal_better.py tests/test_SF3D_better.py -q`
Expected: PASS (OPD/classical paths untouched: with default flags `point_pred_logits` is not None and every new term is a logged zero).

- [ ] **Step 6: Commit**

```bash
git add config/opd_train.py train_OPDReal_better.py tests/test_split_heads.py
git commit -m "gen-6: trainer wiring — L_point_3d, canonical L_origin, axis sign flag"
```

---

### Task 6: SF3D eval metrics for the split arm

**Files:**
- Modify: `train_SF3D_better.py` (`on_test_start`, `test_step`, `on_test_epoch_end`)
- Test: `tests/test_split_heads.py` (extend)

**Interfaces:**
- Consumes: `perpendicular_foot` (Task 3), `point_to_line_distance` from `model.losses` (exists), split-arm outputs (Task 2).
- Produces: test metrics `test/point_err_3d_m`, `test/origin_err_m`, `test/origin_line_err_m`, `test/radius_err_m` (means; 0.0 when the arm lacks the heads). Existing 2D `mean_point_error` and the coords_hat-based origin unprojection are skipped (not crashed) when `coords_hat is None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_split_heads.py`:

```python
def test_sf3d_test_step_split_arm_accumulates_3d_metrics():
    m = _split_module()
    m.eval()
    m.log = lambda *a, **k: None
    m.on_test_start()
    with torch.no_grad():
        m.test_step(_sf3d_batch(), 0)
    # One revolute + one prismatic sample in the batch.
    assert len(m._test_point3d_errors) == 2
    assert len(m._test_origin_err_m) == 1          # revolute row only
    assert len(m._test_origin_line_err_m) == 1
    assert len(m._test_radius_err_m) == 1
    assert all(np.isfinite(v) for v in m._test_point3d_errors)
```

(add `import numpy as np` at the top of the test file if missing).

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_split_heads.py::test_sf3d_test_step_split_arm_accumulates_3d_metrics -q`
Expected: FAIL — attribute `_test_point3d_errors` missing, or `test_step` crashes on `coords_hat[i]` with `coords_hat=None`.

- [ ] **Step 3: Implement in `train_SF3D_better.py`**

(a) `on_test_start` — add accumulators next to the twist ones:

```python
        # Split-arm (gen-6) metrics; empty when the arm lacks the heads.
        self._test_point3d_errors = []        # ||p_hat - traj_gt[0]||, all rows
        self._test_origin_err_m = []          # ||q_hat - q*||, revolute rows
        self._test_origin_line_err_m = []     # dist(q_hat, GT axis line)
        self._test_radius_err_m = []          # |r_pred - r_gt|, revolute rows
```

(b) `test_step` — guard the two `coords_hat` consumers: wrap the 2D
`point_err` append (`train_SF3D_better.py:321`) in `if coords_hat is not
None:` and the coords_hat-based origin unprojection block
(`train_SF3D_better.py:326-367`) in the same guard. Then add, inside the
per-sample loop (imports at top: `from model.losses.split import
perpendicular_foot`; `point_to_line_distance` is already imported; ensure
`import torch.nn.functional as F` exists in the module — add it if not):

```python
            if outputs.point_3d_pred is not None and trajectory_gt is not None:
                p_hat = outputs.point_3d_pred[i].detach().float()
                p_gt = trajectory_gt[i, 0].to(p_hat.device).float()
                self._test_point3d_errors.append(
                    torch.linalg.norm(p_hat - p_gt).item()
                )
            if (
                outputs.origin_pred is not None
                and motion_type_gt[i] == 1
                and motion_origin_3d_gt is not None
                and trajectory_gt is not None
            ):
                q_hat = outputs.origin_pred[i].detach().float()
                o_gt = motion_origin_3d_gt[i].to(q_hat.device).float()
                d_gt = F.normalize(motion_gt[i].to(q_hat.device).float(), dim=0)
                p_gt = trajectory_gt[i, 0].to(q_hat.device).float()
                q_star = perpendicular_foot(o_gt[None], d_gt[None], p_gt[None])[0]
                self._test_origin_err_m.append(
                    torch.linalg.norm(q_hat - q_star).item()
                )
                self._test_origin_line_err_m.append(
                    point_to_line_distance(q_hat[None], o_gt[None], d_gt[None])[0].item()
                )
                if outputs.motion_pred is not None:
                    d_hat = F.normalize(
                        outputs.motion_pred[i].detach().float(), dim=0
                    )
                    r_pred = point_to_line_distance(
                        p_gt[None], q_hat[None], d_hat[None]
                    )[0].item()
                    r_gt = point_to_line_distance(
                        p_gt[None], o_gt[None], d_gt[None]
                    )[0].item()
                    self._test_radius_err_m.append(abs(r_pred - r_gt))
```

(c) `on_test_epoch_end` — gather + log the four means alongside the existing metrics, following the exact `all_gather` pattern of `_test_origin_errors_rotational_all` (`train_SF3D_better.py:505-507`):

```python
        for name, values in (
            ("test/point_err_3d_m", self._test_point3d_errors),
            ("test/origin_err_m", self._test_origin_err_m),
            ("test/origin_line_err_m", self._test_origin_line_err_m),
            ("test/radius_err_m", self._test_radius_err_m),
        ):
            gathered = self.all_gather(torch.tensor(values, device=self.device))
            mean = float(gathered.mean().item()) if gathered.numel() > 0 else 0.0
            self.log(name, mean, on_epoch=True, logger=True, sync_dist=False)
```

(place after the existing gathers; also print them in the summary print
block the module already has, matching its formatting).

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_split_heads.py tests/test_SF3D_better.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_SF3D_better.py tests/test_split_heads.py
git commit -m "gen-6: SF3D eval — point_err_3d/origin_err/origin_line_err/radius_err"
```

---

### Task 7: Gen-6 config, full suite, spec status

**Files:**
- Create: `config/sf3d_train_runpod_split.yaml`
- Modify: `docs/superpowers/specs/2026-08-13-split-heads-gen6-design.md` (status line)

**Interfaces:**
- Consumes: every flag from Tasks 1-6.
- Produces: the launchable gen-6 config; experiment dir `experiments/20260813_sf3d_split_g6/`.

- [ ] **Step 1: Write the config**

Copy `config/sf3d_train_runpod_twist.yaml` to `config/sf3d_train_runpod_split.yaml` and change ONLY these blocks (leave trainer/data/optimizer identical except the two experiment paths):

```yaml
# header comment: gen-6 split articulation arm — spec
#   docs/superpowers/specs/2026-08-13-split-heads-gen6-design.md
model:
  # gen-6: CLIP UNFROZEN (frozen was the prime suspect for gen-5's mask
  # regression). No freeze_backbone key.
  model_params:
    # ... unchanged keys elided ...
    use_cvae: false
    use_twist_head: false
    use_motion_head: true          # axis direction head back on
    use_motion_type_head: true     # type CE head back on
    use_motion_type_input: false   # hint removed — the type head predicts
    twist_pitch_free: false
    trajectory_delta_cumsum: true
    twist_num_hypotheses: 1
    point_prediction_3d: true      # direct 3D interaction point
    use_origin_head: true          # canonical q* origin
    pool_with_predicted_mask: true # no mask teacher forcing
  loss_params:
    bce_weight: 0.5
    dice_weight: 0.5
    mask_weight: 0.5
    point_map_weight: 0.0          # no heatmap channel on this arm
    coord_weight: 0.0
    point_3d_weight: 0.5           # replaces both, same budget
    vae_weight: 0.5                # the 1-cos axis loss (MotionMLP path)
    axis_sign_agnostic: false      # SF3D sign is canonical
    motion_type_weight: 0.5
    origin_weight: 0.5
    trajectory_weight: 0.5         # classical value (gen-4's 4.0 was WTA-internal)
    geometric_loss: "pred_pred_art"
    pred_pred_art_weight: 0.5
    twist_weight: 0.0              # twist machinery off
    point_sigma: 8.0               # unused (no heatmap); kept for dataclass
    vae_beta: 0.01                 # unused (no CVAE); kept for dataclass
```

Checkpoint `dirpath`: `/workspace/SegAffordance/experiments/20260813_sf3d_split_g6/checkpoints`; logger `save_dir`: `.../20260813_sf3d_split_g6/logs`. Keep `max_epochs: 16`, lr `0.00001`, milestones `[13, 15]`, batch 128, the filtered key cache `sf3d_v2_keys_cutoff05_minrad010.pkl`, `min_revolute_radius: 0.10`, `point_source: "element"`, `fast_pipeline: true`.

- [ ] **Step 2: Config parses and builds the module**

Run: `$PY -c "import yaml; from config.opd_train import ModelParams, LossParams; cfg = yaml.safe_load(open('config/sf3d_train_runpod_split.yaml')); ModelParams(**cfg['model']['model_params']); LossParams(**cfg['model']['loss_params']); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Full local suite**

Run: `$PY -m pytest tests/ -q`
Expected: ALL PASS (109 pre-existing + the new file).

- [ ] **Step 4: Mark the spec implemented**

In `docs/superpowers/specs/2026-08-13-split-heads-gen6-design.md`, change `**Status:** DESIGNED (not implemented)` to `**Status:** IMPLEMENTED (2026-08-13)`.

- [ ] **Step 5: Commit**

```bash
git add config/sf3d_train_runpod_split.yaml docs/superpowers/specs/2026-08-13-split-heads-gen6-design.md
git commit -m "gen-6: split-arm training config (unfrozen CLIP, pred_pred_art, no teacher forcing)"
```

---

### Task 8: Dev-pod smoke run

**Files:** none (execution only; requires the dev pod — it is normally running, check first).

- [ ] **Step 1: Sync check + pod status**

Run: `mutagen sync list ethz-workspace | grep -i status` (want "Watching") and `bash runpod/dev.sh status`. Start the pod if stopped (`bash runpod/dev.sh start`).

- [ ] **Step 2: Clear phantom index + suite on the pod**

Run: `bash runpod/dev.sh run "git reset -q && python -m pytest tests/test_split_heads.py -q"`
Expected: PASS (GPU env parity).

- [ ] **Step 3: 100-step smoke**

Run (batch 48, compile off — 24 GB dev card, matches prior smoke recipe):

```bash
bash runpod/dev.sh run "python train_SF3D_better.py fit \
  --config config/sf3d_train_runpod_split.yaml \
  --model.model_params.compile_model false \
  --data.batch_size_train 48 --data.num_workers_train 8 \
  --trainer.limit_train_batches 100 --trainer.limit_val_batches 10 \
  --trainer.max_epochs 1 \
  --trainer.callbacks= --trainer.logger= 2>&1 | tail -30"
```

Expected: runs to completion; `train/loss_total` finite and decreasing-ish; `train/L_point_3d`, `train/L_origin`, `train/L_geo_pred_pred_art` all present and finite. If the CLI rejects `--trainer.callbacks=`, drop those two overrides and point `dirpath`/`save_dir` at `/tmp/smoke_g6` instead.

- [ ] **Step 4: Report**

Summarize term magnitudes at step ~100 (are the new terms within ~1 order of magnitude of L_mask? `L_pp` not exploding?). STOP — training-pod launch is a separate user decision, not part of this plan.

