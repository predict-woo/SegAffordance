"""Unit tests for the se(3) twist supervision (model/losses/twist.py).

Pure geometry on synthetic tensors — no dataset, no GPU, no checkpoint. Run
from the repo root:

    python -m pytest tests/test_twist.py -q

These pin the physics the parameterisation was chosen for:
  * the moment is invariant to where along the axis the GT origin sits, so
    there is no along-axis gauge for the loss to punish (the failure mode the
    deleted plane term had);
  * prismatic is the interior point omega = 0, not a limit at infinity;
  * decode recovers the axis LINE, not the annotated origin;
  * the loss is sign-agnostic in the only way that preserves the axis line
    (omega and v flipped TOGETHER).
"""

import torch

from model.losses.twist import (
    TwistLoss,
    decode_twist,
    point_to_line_distance,
    twist_from_gt,
)
from model.outputs import ModelOutputs
from model.targets import StepTargets


def make_outputs(twist_pred):
    b = 1 if twist_pred is None else twist_pred.shape[0]
    dummy_map = torch.zeros(b, 1, 4, 4)
    return ModelOutputs(
        mask_logits=dummy_map,
        point_logits=dummy_map,
        coords_hat=torch.zeros(b, 2),
        motion_pred=torch.zeros(b, 3),
        motion_type_logits=torch.zeros(b, 2),
        twist_pred=twist_pred,
    )


def make_targets(motion, motion_type, origin):
    return StepTargets(motion=motion, motion_type=motion_type, motion_origin_3d=origin)


ROT = torch.tensor([1])
TRANS = torch.tensor([0])


def test_moment_is_invariant_to_sliding_the_origin_along_the_axis():
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    origin = torch.tensor([[0.4, -0.2, 1.3]])
    slid = origin + 0.7 * axis  # different height on the SAME hinge line
    twist_a = twist_from_gt(axis, ROT, origin)
    twist_b = twist_from_gt(axis, ROT, slid)
    assert torch.allclose(twist_a, twist_b, atol=1e-6)


def test_revolute_decode_recovers_the_axis_line():
    axis = torch.nn.functional.normalize(torch.tensor([[0.3, -0.5, 0.8]]), dim=-1)
    origin = torch.tensor([[0.4, 0.1, 1.5]])
    is_rev, direction, axis_point = decode_twist(twist_from_gt(axis, ROT, origin))
    assert bool(is_rev[0])
    assert torch.allclose(direction.abs(), axis.abs(), atol=1e-5)
    # The decoded point need not be the annotated origin — only on its line.
    assert point_to_line_distance(axis_point, origin, axis).item() < 1e-5


def test_prismatic_is_interior_not_a_limit():
    direction = torch.nn.functional.normalize(torch.tensor([[1.0, 2.0, -0.5]]), dim=-1)
    origin = torch.tensor([[0.4, 0.1, 1.5]])  # must be irrelevant
    twist = twist_from_gt(direction, TRANS, origin)
    assert torch.allclose(twist[:, :3], torch.zeros(1, 3))  # omega exactly 0
    assert torch.all(torch.isfinite(twist))
    is_rev, decoded_dir, _ = decode_twist(twist)
    assert not bool(is_rev[0])
    assert torch.allclose(decoded_dir, direction, atol=1e-6)


def test_loss_is_zero_for_exact_and_for_sign_flipped_prediction():
    axis = torch.tensor([[0.0, 1.0, 0.0]])
    origin = torch.tensor([[0.2, 0.0, 2.0]])
    gt = twist_from_gt(axis, ROT, origin)
    loss_fn = TwistLoss(weight=1.0)
    targets = make_targets(axis, ROT, origin)
    for pred in (gt, -gt):  # (omega, v) -> (-omega, -v) is the same axis line
        total, terms, _aux = loss_fn(make_outputs(pred), targets)
        assert total.item() < 1e-10
        assert "L_twist" in terms


def test_loss_penalises_a_wrong_hinge_line_but_not_a_slid_origin():
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    origin = torch.tensor([[0.5, 0.0, 1.0]])
    loss_fn = TwistLoss(weight=1.0)
    targets = make_targets(axis, ROT, origin)

    slid = twist_from_gt(axis, ROT, origin + 0.9 * axis)  # same line
    displaced = twist_from_gt(axis, ROT, origin + torch.tensor([[0.3, 0.0, 0.0]]))

    slid_loss, _, _aux = loss_fn(make_outputs(slid), targets)
    displaced_loss, _, _aux = loss_fn(make_outputs(displaced), targets)
    assert slid_loss.item() < 1e-10  # what the old plane term wrongly charged
    assert displaced_loss.item() > 1e-3  # a genuinely different hinge line


def test_loss_noops_without_twist_head_or_without_3d_origin():
    axis = torch.tensor([[0.0, 1.0, 0.0]])
    origin = torch.tensor([[0.2, 0.0, 2.0]])
    loss_fn = TwistLoss(weight=1.0)

    total, terms, _aux = loss_fn(make_outputs(None), make_targets(axis, ROT, origin))
    assert total.item() == 0.0 and terms == {}

    opd_targets = make_targets(axis, ROT, None)  # OPD batches have no origin
    total, terms, _aux = loss_fn(make_outputs(torch.randn(1, 6)), opd_targets)
    assert total.item() == 0.0 and terms == {}


# ---- direction supervision (sign_agnostic=False, SF3D) ----


def test_sign_sensitive_loss_penalises_the_flipped_prediction():
    # SF3D's stored sign is canonical (the preprocessor derives the GT
    # trajectory FROM it), so the reversed screw must be a real error.
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    origin = torch.tensor([[0.3, -0.1, 1.2]])
    gt = twist_from_gt(axis, ROT, origin)
    targets = make_targets(axis, ROT, origin)

    sensitive = TwistLoss(weight=1.0, sign_agnostic=False)
    loss_aligned, _, _aux = sensitive(make_outputs(gt.clone()), targets)
    loss_flipped, _, _aux = sensitive(make_outputs(-gt.clone()), targets)
    assert loss_aligned.item() < 1e-8
    assert loss_flipped.item() > 0.1

    # the default stays direction-blind (OPD annotates only the axis line)
    agnostic = TwistLoss(weight=1.0, sign_agnostic=True)
    loss_flip_ag, _, _aux = agnostic(make_outputs(-gt.clone()), targets)
    assert loss_flip_ag.item() < 1e-8


def test_sign_sensitive_prismatic_direction():
    direction = torch.tensor([[0.0, 1.0, 0.0]])
    gt = twist_from_gt(direction, TRANS, torch.zeros(1, 3))
    targets = make_targets(direction, TRANS, torch.zeros(1, 3))
    sensitive = TwistLoss(weight=1.0, sign_agnostic=False)
    loss_flipped, _, _aux = sensitive(make_outputs(-gt.clone()), targets)
    assert loss_flipped.item() > 0.1
