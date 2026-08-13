# Gen-7 Heatmap + Depth Lifts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement docs/superpowers/specs/2026-08-14-heatmap-depth-lift-gen7-design.md — both articulation points become heatmap + scalar-depth predictions lifted with intrinsics; absolute direct trajectory; classical 2D point machinery restored; origin heatmap channel added.

**Architecture:** All new behaviour behind `ModelParams` flags defaulting False (classical/gen-6 paths bit-preserved). The lifted 3D points populate the EXISTING `ModelOutputs.point_3d_pred`/`origin_pred` fields so gen-6's trainer losses (`L_point_3d`, `origin_canonical_loss`) and eval metrics consume them unchanged. `CRIS.forward` gains an optional normalized-intrinsics input; without it the lifts are None and every 3D consumer no-ops.

**Tech Stack:** PyTorch / PyTorch Lightning; pytest.

## Global Constraints

- Tests run LOCALLY: `$PY -m pytest tests/<file> -q` where `$PY` = `/private/tmp/claude-501/-Users-andyye-dev-ethz-workspace/ab498bc1-4197-41fb-9b67-6f93d9065aed/scratchpad/twistenv/bin/python` (full trainer import chain installed). Current suite: 117 passed — must stay green.
- Mutating git ONLY from the Mac side. Commit per task, message suffix "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>".
- New `ModelParams` flags all default False; with defaults every existing config's forward pass is unchanged. Twist/WTA/screw/gen-6 paths untouched beyond gating.
- The interaction-point readout is named `point_uv` (renamed from coords_hat in `add7ad2`) — use that name everywhere.
- All 3D camera-frame meters; `motion_type == 1` revolute. `q* = o_gt + ((p − o_gt)·d̂_gt) d̂_gt`.
- Reuse, do not duplicate: `soft_argmax2d` (segmenter), `normalized_intrinsics`/`backproject_points`/`project_points` (model/losses/geometric.py), `make_gaussian_map` (utils/tools.py), `OriginDepthHead` (model/layers.py), `origin_canonical_loss`/`perpendicular_foot` (model/losses/split.py), `_StubBackbone` + helpers in tests/test_split_heads.py.
- tests/test_split_heads.py contains `_params(**over)`, `_inputs(...)`, `_split_module()`, `_sf3d_batch()` helpers and a `_stub_backbone` monkeypatch pattern — extend, don't reinvent. New gen-7 tests go in a NEW file tests/test_g7_lift.py (import helpers from test_split_heads).

---

### Task 1: q* projection helper + `origin_uv` output field + ModelParams flags

**Files:**
- Modify: `model/losses/split.py`, `model/losses/__init__.py`, `model/outputs.py`, `config/opd_train.py`
- Test: `tests/test_g7_lift.py` (create)

**Interfaces:**
- Produces: `project_q_star(origin_gt (B,3), direction_gt (B,3), point_gt (B,3), K_norm (B,3,3)) -> (uv_norm (B,2), valid (B,) bool)` in model/losses/split.py — uv normalized [0,1]; `valid = (z > 0.05) & in [0,1)^2`.
- Produces: `ModelOutputs.origin_uv: Optional[torch.Tensor] = None` ((B,2), normalized).
- Produces: `ModelParams.use_origin_heatmap: bool = False`, `ModelParams.predict_point_depth: bool = False`, `ModelParams.trajectory_absolute: bool = False`; `LossParams.origin_map_weight: float = 0.5`.

- [ ] **Step 1: failing tests** (`tests/test_g7_lift.py`):

```python
import torch
import torch.nn.functional as F

from model.losses.split import perpendicular_foot, project_q_star
from model.outputs import ModelOutputs


def _knorm(B):
    K = torch.eye(3).repeat(B, 1, 1)
    K[:, 0, 0] = K[:, 1, 1] = 1.2   # fx, fy (normalized units)
    K[:, 0, 2] = K[:, 1, 2] = 0.5   # cx, cy
    return K


def test_project_q_star_matches_foot_and_masks_offscreen():
    torch.manual_seed(0)
    B = 16
    o = torch.randn(B, 3) * 0.3 + torch.tensor([0.0, 0.0, 2.0])
    d = F.normalize(torch.randn(B, 3), dim=1)
    p = torch.randn(B, 3) * 0.3 + torch.tensor([0.0, 0.0, 2.0])
    K = _knorm(B)
    uv, valid = project_q_star(o, d, p, K)
    q_star = perpendicular_foot(o, d, p)
    # Where valid, uv is the pinhole projection of q_star.
    expect = torch.stack(
        [K[:, 0, 0] * q_star[:, 0] / q_star[:, 2] + K[:, 0, 2],
         K[:, 1, 1] * q_star[:, 1] / q_star[:, 2] + K[:, 1, 2]], dim=1)
    assert torch.allclose(uv[valid], expect[valid], atol=1e-5)
    assert valid.dtype == torch.bool
    # Behind-camera q* is invalid, never NaN.
    o2 = o.clone(); o2[:, 2] = -3.0; p2 = p.clone(); p2[:, 2] = -3.0
    uv2, valid2 = project_q_star(o2, d, p2, K)
    assert not valid2.any() and torch.isfinite(uv2).all()


def test_project_q_star_gauge_invariant():
    torch.manual_seed(1)
    o = torch.randn(4, 3) + torch.tensor([0.0, 0.0, 2.0])
    d = F.normalize(torch.randn(4, 3), dim=1)
    p = torch.randn(4, 3) + torch.tensor([0.0, 0.0, 2.0])
    K = _knorm(4)
    uv1, v1 = project_q_star(o, d, p, K)
    uv2, v2 = project_q_star(o + 2.9 * d, d, p, K)
    assert torch.allclose(uv1, uv2, atol=1e-5) and torch.equal(v1, v2)


def test_new_fields_and_flags_default_off():
    from config.opd_train import LossParams, ModelParams
    assert ModelOutputs(mask_logits=torch.zeros(1, 1, 4, 4)).origin_uv is None
    import dataclasses
    mf = {f.name: f.default for f in dataclasses.fields(ModelParams)}
    assert mf["use_origin_heatmap"] is False
    assert mf["predict_point_depth"] is False
    assert mf["trajectory_absolute"] is False
    lf = {f.name: f.default for f in dataclasses.fields(LossParams)}
    assert lf["origin_map_weight"] == 0.5
```

- [ ] **Step 2:** `$PY -m pytest tests/test_g7_lift.py -q` — FAIL (ImportError project_q_star).
- [ ] **Step 3: implement.** In split.py (module already imports torch, F):

```python
def project_q_star(origin_gt, direction_gt, point_gt, K_norm,
                   z_min: float = 0.05):
    """Normalized [0,1] projection of q* + validity mask.

    valid = q* in front of the camera AND inside the frame. Invalid rows
    still return finite uv (clamped z) so downstream masking is the only
    consumer of validity — no NaN can leak.
    """
    q_star = perpendicular_foot(origin_gt, direction_gt, point_gt)
    z = q_star[..., 2]
    z_safe = z.clamp(min=z_min)
    u = K_norm[..., 0, 0] * q_star[..., 0] / z_safe + K_norm[..., 0, 2]
    v = K_norm[..., 1, 1] * q_star[..., 1] / z_safe + K_norm[..., 1, 2]
    uv = torch.stack([u, v], dim=-1)
    valid = (z > z_min) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    return uv, valid
```

Export from `model/losses/__init__.py`. In outputs.py add after `origin_pred`:

```python
    #: (B, 2) soft-argmax of the origin heatmap channel, normalised to
    #: [0, 1] (ModelParams.use_origin_heatmap). The 2D face of origin_pred.
    origin_uv: Optional[torch.Tensor] = None
```

In config/opd_train.py ModelParams (near the gen-6 flags):

```python
    # gen-7 (docs/superpowers/specs/2026-08-14-heatmap-depth-lift-gen7-design.md).
    # Third projector channel: origin heatmap -> soft-argmax origin_uv ->
    # scalar depth z_q -> q_hat lifted with intrinsics. Supervised at the
    # projected q* (in-frame 99.6-100% on v2 — tools/diag_origin_inframe.py).
    use_origin_heatmap: bool = False
    # Scalar depth head z_p for the interaction point (input: condition +
    # grid_sample of decoded features at point_uv); p_hat lifted with
    # intrinsics. Requires the classical 2D point path (point_prediction_3d
    # False).
    predict_point_depth: bool = False
    # TrajectoryMLP emits 20 ABSOLUTE camera-frame points — no delta-cumsum,
    # no relative frame (gen-7 user decision; zigzag + absolute-regression
    # risks on record in the spec). Mutually exclusive with
    # trajectory_delta_cumsum.
    trajectory_absolute: bool = False
```

LossParams next to point_map_weight: `origin_map_weight: float = 0.5` with a one-line comment (BCE vs Gaussian at projected q*, revolute+in-frame rows).

- [ ] **Step 4:** `$PY -m pytest tests/test_g7_lift.py tests/test_split_heads.py -q` — PASS.
- [ ] **Step 5:** commit `git add -A` of the five files, message "gen-7: project_q_star helper, origin_uv output field, flags".

---

### Task 2: Segmenter — origin channel, depth heads, lifts, absolute trajectory

**Files:**
- Modify: `model/segmenter.py`, `model/layers.py` (TrajectooryMLP absolute mode — note: class is `TrajectoryMLP`)
- Test: `tests/test_g7_lift.py` (extend)

**Interfaces:**
- Consumes: Task 1 flags/fields; `OriginDepthHead(input_dim, hidden_dim, min_depth=0.1)` (exists, softplus+min); `backproject_points`, `soft_argmax2d`.
- Produces: `CRIS.forward(img, depth, word, mask, interaction_point, motion_gt=None, motion_type_input=None, intrinsics_norm=None)` — new LAST optional kwarg (B,3,3) normalized intrinsics. With `use_origin_heatmap`: projector `out_channels=3` (mask=ch0, point=ch1, origin=ch2), `origin_uv` (B,2) always emitted; with `predict_point_depth`: `z_p` head input `vae_condition_dim + fpn_out[1]` (grid_sample of the decoded fq at point_uv). When `intrinsics_norm` is not None: `point_3d_pred = backproject_points(K, point_uv, z_p)` (if predict_point_depth), `origin_pred = backproject_points(K, origin_uv, z_q)` (if use_origin_heatmap); else both None.
- Produces: `TrajectoryMLP(..., absolute=False)`: when True, forward returns the raw (B, K, num_points, 3) readout (no cumsum, no zero-pin); constructor asserts `not (absolute and delta_cumsum)`.
- Condition vector: `[..., point_uv] + [origin_uv]` appended LAST when use_origin_heatmap (after the existing point_uv append), so `vae_condition_dim += 2`. Classical order for existing pieces unchanged.

Key implementation notes (follow exactly):

```python
# __init__ (after the point_prediction_3d block):
self.use_origin_heatmap = getattr(model_params, "use_origin_heatmap", False)
self.predict_point_depth = getattr(model_params, "predict_point_depth", False)
if self.use_origin_heatmap and self.point_prediction_3d:
    raise ValueError("use_origin_heatmap needs the classical 2D point path")
if self.predict_point_depth and self.point_prediction_3d:
    raise ValueError("predict_point_depth needs the classical 2D point path")
# projector channels: 1 (3D mode) | 2 (classical) | 3 (gen-7)
out_ch = 1 if self.point_prediction_3d else (3 if self.use_origin_heatmap else 2)
# condition dim: += 2 for origin_uv
if self.use_origin_heatmap:
    vae_condition_dim += 2
# depth heads (AFTER vae_condition_dim is final):
self.point_depth_head = OriginDepthHead(
    input_dim=vae_condition_dim + model_params.fpn_out[1],
    hidden_dim=model_params.vae_hidden_dim,
) if self.predict_point_depth else None
self.origin_depth_head_g7 = OriginDepthHead(
    input_dim=vae_condition_dim, hidden_dim=model_params.vae_hidden_dim,
) if self.use_origin_heatmap else None
```

(NAME the second head `origin_depth_head_g7` — `origin_depth_head` already exists for the 2D arm's `predict_origin_depth`; do not collide.)

```python
# forward, after maps = self.proj(fq, state):
origin_uv = None
if self.use_origin_heatmap:
    origin_logits = maps[:, 2:3]
    origin_px = soft_argmax2d(origin_logits)
    origin_uv = origin_px / torch.tensor(
        [W_map, H_map], dtype=origin_px.dtype, device=origin_px.device)
# condition assembly: append origin_uv AFTER the existing point_uv append
# (2D path stays [features, point_uv, type_emb]; then + origin_uv last).
...
# after vae_condition is final:
point_3d_pred = origin_pred = None
z_p = z_q = None
if self.predict_point_depth:
    grid = point_uv.view(-1, 1, 1, 2) * 2.0 - 1.0        # [0,1] -> [-1,1]
    local = F.grid_sample(fq, grid, align_corners=False).flatten(1)  # (B, 512)
    z_p = self.point_depth_head(torch.cat([vae_condition, local], dim=1))
if self.use_origin_heatmap:
    z_q = self.origin_depth_head_g7(vae_condition)
if intrinsics_norm is not None:
    K = intrinsics_norm.to(vae_condition.dtype)
    if z_p is not None:
        point_3d_pred = backproject_points(K.float(), point_uv.float(), z_p.float())
    if z_q is not None:
        origin_pred = backproject_points(K.float(), origin_uv.float(), z_q.float())
```

Populate `ModelOutputs(..., origin_uv=origin_uv, point_3d_pred=point_3d_pred, origin_pred=origin_pred)` — on the gen-6 3D path those two fields keep coming from the old code path (mutually exclusive by the ValueErrors). `fq` here is the DECODED feature map (post-decoder, pre-projector) — grid_sample it, fp32-safe. Import `backproject_points` from `model.losses.geometric` at module top (no import cycle: losses never import segmenter).

TrajectoryMLP in layers.py: add `absolute: bool = False` param + assert; in forward, `if self.absolute: return trajectory_pred.view(-1, K, self.num_points, 3)` (i.e. the existing non-cumsum branch — absolute mode IS the direct readout, only the semantics of its target change; add a comment saying so). Segmenter passes `absolute=getattr(model_params, "trajectory_absolute", False)`.

- [ ] **Step 1: failing tests** (extend tests/test_g7_lift.py; reuse `_params`/`_inputs` and the backbone-stub fixture pattern from tests/test_split_heads.py via import):

```python
from tests.test_split_heads import _inputs, _params  # noqa: E402
# (mirror the existing _stub_backbone/monkeypatch fixture usage — import or
# replicate the fixture exactly as test_split_heads defines it)


def _g7_model():
    return CRIS(_params(use_origin_heatmap=True, predict_point_depth=True,
                        trajectory_absolute=True, trajectory_delta_cumsum=False))


def test_g7_forward_shapes_with_and_without_intrinsics(g7_model):
    img, depth, word, mask = _inputs()
    K = _knorm(2)
    with torch.no_grad():
        out = g7_model(img, depth, word, mask, None, None, None, K)
        out_nok = g7_model(img, depth, word, mask, None, None, None, None)
    assert out.point_uv.shape == (2, 2) and out.origin_uv.shape == (2, 2)
    assert out.point_3d_pred.shape == (2, 3) and out.origin_pred.shape == (2, 3)
    assert out.trajectory_pred.shape == (2, 20, 3)
    assert out.mask_logits.shape[1] == 1 and out.point_logits is not None
    # Lift only exists where K exists.
    assert out_nok.point_3d_pred is None and out_nok.origin_pred is None
    assert out_nok.origin_uv is not None


def test_g7_lift_roundtrip(g7_model):
    # project(point_3d_pred) must land exactly on point_uv (by construction).
    img, depth, word, mask = _inputs()
    K = _knorm(2)
    with torch.no_grad():
        out = g7_model(img, depth, word, mask, None, None, None, K)
    from model.losses.geometric import project_points
    assert torch.allclose(
        project_points(K, out.point_3d_pred.unsqueeze(1)).squeeze(1),
        out.point_uv, atol=1e-4)
    assert (out.point_3d_pred[:, 2] > 0).all()   # softplus depth


def test_g7_depth_head_inputs_asymmetric(g7_model):
    cond = 2564_placeholder = None  # compute from the model:
    d = g7_model.point_depth_head.mlp[0].in_features \
        - g7_model.origin_depth_head_g7.mlp[0].in_features
    assert d == 512    # z_p gets the 512-dim local sample, z_q does not


def test_classical_and_gen6_modes_unchanged():
    m2d = CRIS(_params())
    m3d = CRIS(_params(point_prediction_3d=True, use_origin_head=True))
    img, depth, word, mask = _inputs()
    with torch.no_grad():
        o2 = m2d.eval()(img, depth, word, mask, None, None)
        o3 = m3d.eval()(img, depth, word, mask, None, None)
    assert o2.origin_uv is None and o2.point_3d_pred is None
    assert o3.origin_uv is None and o3.point_3d_pred.shape == (2, 3)


def test_absolute_trajectory_head_no_zero_pin():
    from model.layers import TrajectoryMLP
    head = TrajectoryMLP(input_dim=8, hidden_dim=16, num_points=20,
                         delta_cumsum=False, absolute=True)
    out = head(torch.randn(3, 8))
    assert out.shape == (3, 1, 20, 3)
    assert not torch.allclose(out[:, :, 0], torch.zeros(3, 1, 3))
    import pytest as _pytest
    with _pytest.raises(AssertionError):
        TrajectoryMLP(input_dim=8, delta_cumsum=True, absolute=True)
```

(Fix the placeholder line in `test_g7_depth_head_inputs_asymmetric` — keep only the `d == 512` computation; and adapt `.mlp[0]` to OriginDepthHead's actual attribute name.)

- [ ] **Step 2:** run — FAIL (unexpected kwargs).
- [ ] **Step 3:** implement per the notes above.
- [ ] **Step 4:** `$PY -m pytest tests/test_g7_lift.py tests/test_split_heads.py tests/test_trajectory_head.py tests/test_twist.py tests/test_wta.py -q` — PASS.
- [ ] **Step 5:** commit "gen-7: segmenter — origin channel, asymmetric depth heads, intrinsics lifts, absolute trajectory".

---

### Task 3: `PredPredArticulationLoss` absolute-trajectory mode

**Files:** Modify `model/losses/geometric.py`; test `tests/test_g7_lift.py`.

**Interfaces:** constructor gains `trajectory_is_absolute: bool = False`. When True: `d = traj - traj[:, :1]` (in-loss relative) and the axis point in the relative frame is `c = origin_pred - traj[:, 0]` (spec decision) — `point_3d_pred` is NOT read in absolute mode (drop it from the required-fields check there; `origin_pred` still required). Everything else identical.

- [ ] **Step 1: failing tests** — replicate the two geometry tests from tests/test_split_heads.py (`test_pp_art_perfect_revolute...`, `test_pp_art_soft_gate...`) in absolute form: build the same perfect circle/line but ADD an arbitrary absolute offset to every trajectory point (and set `point_3d_pred=None`), construct the loss with `trajectory_is_absolute=True`, assert ~0 / branch behaviour / gauge along d̂ exactly as the relative versions do.
- [ ] **Step 2:** FAIL (unexpected kwarg).
- [ ] **Step 3:** implement (a ~6-line diff: store the flag; branch at the top of forward for `d` and `c`; adjust the required-fields tuple accordingly).
- [ ] **Step 4:** `$PY -m pytest tests/test_g7_lift.py tests/test_split_heads.py tests/test_geometric_losses.py -q` — PASS.
- [ ] **Step 5:** commit "gen-7: L_pp absolute-trajectory mode (d_i and axis anchor from traj[0])".

---

### Task 4: Trainer wiring — intrinsics into forward, L_origin_map, in-frame masking, absolute L_trajectory

**Files:** Modify `train_OPDReal_better.py`; test `tests/test_g7_lift.py`.

**Interfaces:**
- `_common_step` computes `K_norm = normalized_intrinsics(targets.camera_intrinsic, targets.img_size)` when `targets.camera_intrinsic is not None` (import from model.losses.geometric) and passes it as the new forward kwarg; None otherwise.
- `L_origin_map`: when `outputs.origin_uv is not None` and the projector emitted 3 channels — the origin channel logits are not in ModelOutputs; ADD `origin_logits: Optional[torch.Tensor] = None` to ModelOutputs in this task (B,1,H,W), populated by the segmenter (2-line follow-up in segmenter.py, part of this task's diff). Target: `uv_q, valid_q = project_q_star(motion_origin_3d, motion_gt, trajectory[:,0], K_norm)`; `heat = make_gaussian_map(uv_q, H_map, W_map, point_sigma, device)`; per-sample BCE (`F.binary_cross_entropy_with_logits(..., reduction="none").mean(dim=(1,2,3))`) masked by `valid_q & (motion_type_gt == 1)`, mean over surviving rows, zero-logged otherwise. Weight `origin_map_weight`.
- `L_origin_3d` (existing `origin_canonical_loss` call): extend its row mask by the same `valid_q` — build the mask OUTSIDE and pass a pre-masked motion_type (set invalid rows' type to 0 before the call — simplest, no signature change; comment why).
- `L_trajectory`: when `getattr(self.model, "trajectory_predictor", None)` has `absolute=True` (read `self.model_params.trajectory_absolute`), compare `outputs.trajectory_pred` to `targets.trajectory` DIRECTLY (no relative subtraction). The relative path stays for every other config.
- The geometric-loss construction passes `trajectory_is_absolute=getattr(self.loss_params-side model flag...)` — read `getattr(self.model_params, "trajectory_absolute", False)` when building `PredPredArticulationLoss` in `build_geometric_loss`: give the factory an optional `trajectory_is_absolute` kwarg (default False) that the trainer supplies: change `build_geometric_loss(self.loss_params)` to `build_geometric_loss(self.loss_params, trajectory_is_absolute=...)`.

- [ ] **Step 1: failing test** — a `_g7_module()` mirroring `_split_module()` (import from test_split_heads) with the gen-7 flags + `origin_map_weight=0.5`, run `_common_step` on `_sf3d_batch()` (it carries camera intrinsics = eye(3)): assert finite loss, backward OK, and logged keys include `train/L_origin_map`, `train/L_point_3d` (> 0, from the lifted point), `train/L_origin`, `train/L_trajectory`, `train/L_point_map`, `train/L_coord`; assert `train/L_origin_map` > 0 (batch has a revolute row with in-frame q* — if the random batch's q* lands out of frame, seed/construct the revolute row's origin near the element to guarantee validity).
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** implement per the interface block (including the 2-line `origin_logits` addition to outputs.py + segmenter).
- [ ] **Step 4:** `$PY -m pytest tests/test_g7_lift.py tests/test_split_heads.py -q` — PASS.
- [ ] **Step 5:** commit "gen-7: trainer — intrinsics lift path, L_origin_map, in-frame masking, absolute L_trajectory".

---

### Task 5: Eval + viz

**Files:** Modify `train_SF3D_better.py`, `tools/sf3d_vis_predictions.py`; test `tests/test_g7_lift.py`.

**Interfaces:**
- Eval additions in test_step (guarded on the fields existing): `self._test_point_traj0_gap.append(||point_3d_pred[i] − trajectory_pred[i,0]||)` when BOTH exist and the trajectory is absolute (read `self.model_params.trajectory_absolute`); gathered/logged as `test/point_traj0_gap_m` in on_test_epoch_end (same all_gather pattern; 0.0 when empty). The gen-6 metrics (point_err_3d_m, origin_err_m, origin_line_err_m, radius_err_m) need NO change — they read the lifted fields. The classical 2D `mean_point_error` works again automatically (point_uv exists).
- traj_dir metric: uses last−first — frame-agnostic, no change; VERIFY the existing trajectory-MSE-style comparisons in test_step don't assume relative (search for `trajectory_pred` there; the only consumer is traj_dir cos — confirm and note in the report).
- Viz: in tools/sf3d_vis_predictions.py's split-arm branch — when the checkpoint's config has `trajectory_absolute`, draw the predicted trajectory DIRECTLY (project the absolute points; no anchoring translation), and draw the origin marker at `origin_uv` (small red circle) in addition to the lifted-axis line (which already uses origin_pred + motion_pred). Twist/gen-6 rendering unchanged.
- [ ] **Steps:** failing test (test_step on `_g7_module` accumulates `_test_point_traj0_gap` with 2 entries and finite) → implement → `$PY -m pytest tests/test_g7_lift.py tests/test_SF3D_better.py -q` PASS → commit "gen-7: eval point_traj0_gap_m + absolute-trajectory viz".

---

### Task 6: Config, full suite, spec status

**Files:** Create `config/sf3d_train_runpod_g7.yaml`; modify the spec status line.

- [ ] Copy `config/sf3d_train_runpod_split.yaml` (gen-6). Change ONLY: header comment (gen-7, spec pointer); model level `freeze_backbone: true` (re-add); model_params: `point_prediction_3d: false`, `use_origin_head: false`, `pool_with_predicted_mask: false`, `use_origin_heatmap: true`, `predict_point_depth: true`, `trajectory_absolute: true`, `trajectory_delta_cumsum: false`; loss_params: `point_map_weight: 0.5`, `coord_weight: 0.5`, `origin_map_weight: 0.5` (keep `point_3d_weight: 0.5`, `origin_weight: 0.5`, `trajectory_weight: 0.5`, `geometric_loss: "pred_pred_art"`, `pred_pred_art_weight: 0.5`, `axis_sign_agnostic: false`); experiment paths -> `experiments/20260814_sf3d_g7/{checkpoints,logs}`. Data/trainer/optimizer otherwise identical (filtered cache, batch 128, 16 epochs, lr 1e-5, milestones [13,15]).
- [ ] Validate: `$PY -c "import yaml; from config.opd_train import ModelParams, LossParams; c=yaml.safe_load(open('config/sf3d_train_runpod_g7.yaml')); ModelParams(**c['model']['model_params']); LossParams(**c['model']['loss_params']); print('ok')"`.
- [ ] Full suite: `$PY -m pytest tests/ -q` — ALL PASS.
- [ ] Spec status -> `**Status:** IMPLEMENTED (2026-08-14)`.
- [ ] Commit "gen-7: training config (frozen CLIP, heatmap+depth lifts, absolute trajectory)".

---

### Task 7: Dev-pod smoke

No files. Pod is normally RUNNING (check `bash runpod/dev.sh status`; start if not). Never run mutating git on the pod (`git reset -q` allowed).

- [ ] `bash runpod/dev.sh run "git reset -q && python -m pytest tests/test_g7_lift.py tests/test_split_heads.py -q"` — PASS.
- [ ] 100-step smoke (batch 48, compile off; overlay callbacks/logger to /tmp/smoke_g7 exactly as the gen-6 smoke did — see experiments/20260813_sf3d_split_g6/notes.md context; the CLI rejects `--trainer.callbacks=`, use an overlay config):

```bash
bash runpod/dev.sh run "python train_SF3D_better.py fit \
  --config config/sf3d_train_runpod_g7.yaml \
  --model.model_params.compile_model false \
  --data.batch_size_train 48 --data.num_workers_train 8 \
  --trainer.limit_train_batches 100 --trainer.limit_val_batches 10 \
  --trainer.max_epochs 1 2>&1 | tail -30"
```

- [ ] Verify: loss_total finite/decreasing; `train/L_origin_map`, `train/L_point_3d`, `train/L_origin`, `train/L_trajectory`, `train/L_geo_pred_pred_art` present and finite; note magnitudes vs L_mask. Report and STOP — training-pod launch is a separate user decision.
