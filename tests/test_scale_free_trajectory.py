"""RGB-only + scale-free trajectory (spec docs/superpowers/specs/
2026-09-09-rgb-only-scale-free-trajectory-design.md).

The projection loss's "unit" anchor (GT first pixel at depth 1, no depth
map — exactly invariant to the unknown anchor depth when the head predicts
Δ̃ = Δ / z0), the scale-factor helpers that put metres back on the 3D path,
the RGB-only forward, and the depth-column slicing loader rule.
"""
import pytest
import torch

from model.losses.geometric import (
    TrajectoryProjectionLoss,
    apply_trajectory_scale,
    backproject_points,
    normalized_intrinsics,
    project_points,
    trajectory_scale_factor,
)
from model.outputs import ModelOutputs
from model.targets import StepTargets
from test_split_heads import _inputs, _make_cris  # stub-backbone CRIS (tests/ is on sys.path under pytest)


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


# ---- Task 1: the "unit" anchor ---------------------------------------------

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


# ---- Task 2: scale factor helpers + config fields --------------------------

def _scale_fixture():
    B = 3
    traj = torch.randn(B, 5, 3)
    hyps = torch.randn(B, 2, 5, 3)
    outputs = ModelOutputs(
        mask_logits=torch.zeros(B, 1, 4, 4), trajectory_pred=traj.clone(),
        trajectory_hyps=hyps.clone(),
        point_3d_pred=torch.tensor([[0.1, 0.2, 0.7], [0.0, 0.0, 1.3], [0.3, 0.1, 2.0]]),
    )
    gt = torch.zeros(B, 5, 3)
    gt[:, 0, 2] = torch.tensor([0.5, 1.0, 1.5])
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


def test_scaled_curve_projects_like_the_unit_anchor_curve():
    """The invariance the whole design rests on: z0 * (ray + Δ̃) and
    (ray + Δ̃) project to the same pixels."""
    B = 2
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]).expand(B, 3, 3)
    K_norm = normalized_intrinsics(K, torch.tensor([[100.0, 100.0]]).expand(B, 2))
    uv = torch.tensor([[0.3, 0.6], [0.7, 0.2]])
    rel = torch.randn(B, 6, 3) * 0.1
    z0 = torch.tensor([0.6, 1.9])
    unit = backproject_points(K_norm, uv, torch.ones(B)).unsqueeze(1) + rel
    metric = backproject_points(K_norm, uv, z0).unsqueeze(1) + rel * z0.view(-1, 1, 1)
    assert torch.allclose(project_points(K_norm, unit), project_points(K_norm, metric), atol=1e-6)


def test_new_config_fields_default_off():
    from config.opd_train import Config, LossParams, ModelParams
    assert ModelParams.__dataclass_fields__["trajectory_scale_free"].default is False
    assert ModelParams.__dataclass_fields__["finetune_slice_input_channels"].default is False
    assert LossParams.__dataclass_fields__["trajectory_scale_source"].default == "unit"
    assert Config.__dataclass_fields__["test_trajectory_scale"].default == "pred_z_p"


# ---- Task 4: RGB-only forward + loader slicing -----------------------------

def test_rgb_only_model_runs_with_depth_none():
    model = _make_cris(use_depth=False)
    assert model.depth_encoder is None
    img, _depth, word, mask = _inputs(B=2, size=64)
    out = model(img, None, word, mask, None, None)
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
