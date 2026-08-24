"""Tests for the analytic screw-trajectory decode (mechanism study, spec
2026-08-25): a differentiable mirror of the GT writer evaluated on
predicted articulation parameters.

The load-bearing property is WELL-POSEDNESS: fed the GT parameters, the
decode must reproduce the GT writer's curve exactly (relative frame), so a
perfect articulation prediction gets zero trajectory loss. Locked against
the REAL writer code via the same AST extraction as
test_gt_sign_convention.py.
"""

import math

import numpy as np
import pytest
import torch

from model.losses.geometric import analytic_screw_trajectory
from tests.test_gt_sign_convention import compute_trajectory, _random_case


def decode_one(motion_type, axis, origin, p0, num_points=20):
    mt = torch.tensor([1.0 if motion_type == "rot" else 0.0])
    ax = torch.as_tensor(axis, dtype=torch.float32).reshape(1, 3)
    return analytic_screw_trajectory(
        motion_type_gt=mt,
        axis_trans=ax,
        axis_rot=ax,
        origin_pred=torch.as_tensor(origin, dtype=torch.float32).reshape(1, 3),
        point_3d_pred=torch.as_tensor(p0, dtype=torch.float32).reshape(1, 3),
        num_points=num_points,
    )[0]


@pytest.mark.parametrize("seed", range(6))
def test_rot_decode_matches_writer_exactly(seed):
    axis, origin, pts = _random_case(seed)
    writer_traj = np.asarray(compute_trajectory("rot", origin, axis, pts))
    writer_rel = writer_traj - writer_traj[0:1]
    # Feed the decode the same parameters the writer used: p0 = the
    # writer's own start point (whose GT supervision target it is).
    dec = decode_one("rot", axis, origin, writer_traj[0], num_points=len(writer_traj))
    assert torch.allclose(
        dec, torch.as_tensor(writer_rel, dtype=torch.float32), atol=1e-5
    )


@pytest.mark.parametrize("seed", range(6))
def test_trans_decode_matches_writer_exactly(seed):
    axis, origin, pts = _random_case(seed)
    writer_traj = np.asarray(compute_trajectory("trans", origin, axis, pts))
    writer_rel = writer_traj - writer_traj[0:1]
    dec = decode_one("trans", axis, origin, writer_traj[0], num_points=len(writer_traj))
    assert torch.allclose(
        dec, torch.as_tensor(writer_rel, dtype=torch.float32), atol=1e-5
    )


def test_origin_gauge_freedom_along_axis():
    # Sliding the origin along the axis must not change the rot decode
    # (the lever is the perpendicular component) — matches the GT
    # convention that only the axis LINE is defined.
    axis, origin, pts = _random_case(0)
    writer_traj = np.asarray(compute_trajectory("rot", origin, axis, pts))
    a = decode_one("rot", axis, origin, writer_traj[0])
    b = decode_one("rot", axis, origin + 3.7 * axis, writer_traj[0])
    assert torch.allclose(a, b, atol=1e-5)


def test_sign_sensitivity_is_linear_not_saddled():
    # The mechanism hypothesis: in point space a flipped axis is maximally
    # wrong (arc swept the other way), with LARGE loss — unlike 1-cos's
    # antipodal saddle.
    axis, origin, pts = _random_case(1)
    writer_traj = np.asarray(compute_trajectory("rot", origin, axis, pts))
    gt_rel = torch.as_tensor(writer_traj - writer_traj[0:1], dtype=torch.float32)
    dec_flip = decode_one("rot", -axis, origin, writer_traj[0], num_points=len(writer_traj))
    err = (dec_flip - gt_rel).pow(2).sum(-1).mean()
    energy = gt_rel.pow(2).sum(-1).mean()
    assert (err / energy).item() > 1.0  # far above the collapsed-pred score


def test_gradients_flow_to_all_articulation_inputs():
    axis, origin, pts = _random_case(2)
    writer_traj = np.asarray(compute_trajectory("rot", origin, axis, pts))
    gt_rel = torch.as_tensor(writer_traj - writer_traj[0:1], dtype=torch.float32)

    ax = torch.tensor([[0.1, 0.9, 0.2]], requires_grad=True)
    og = torch.tensor([[0.0, 0.1, 1.8]], requires_grad=True)
    p0 = torch.tensor([[0.3, -0.2, 2.0]], requires_grad=True)
    dec = analytic_screw_trajectory(
        motion_type_gt=torch.tensor([1.0]),
        axis_trans=ax, axis_rot=ax, origin_pred=og, point_3d_pred=p0,
        num_points=gt_rel.shape[0],
    )
    loss = (dec[0] - gt_rel).pow(2).mean()
    loss.backward()
    for t in (ax, og, p0):
        assert t.grad is not None and torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0


def test_trans_rows_ignore_rot_axis_and_vice_versa():
    # GT-type routing: a trans row's decode must not depend on the rot
    # candidate or the origin; a rot row's must not depend on the trans
    # candidate.
    p0 = torch.zeros(1, 3)
    base = analytic_screw_trajectory(
        torch.tensor([0.0]),
        axis_trans=torch.tensor([[0.0, 0.0, 1.0]]),
        axis_rot=torch.tensor([[1.0, 0.0, 0.0]]),
        origin_pred=torch.tensor([[5.0, 5.0, 5.0]]),
        point_3d_pred=p0,
    )
    alt = analytic_screw_trajectory(
        torch.tensor([0.0]),
        axis_trans=torch.tensor([[0.0, 0.0, 1.0]]),
        axis_rot=torch.tensor([[0.0, -1.0, 0.0]]),
        origin_pred=torch.tensor([[-2.0, 0.0, 9.0]]),
        point_3d_pred=p0,
    )
    assert torch.allclose(base, alt, atol=1e-7)


def test_batch_mixes_types_row_wise():
    axis = [0.0, 0.0, 1.0]
    out = analytic_screw_trajectory(
        torch.tensor([0.0, 1.0]),
        axis_trans=torch.tensor([axis, axis], dtype=torch.float32),
        axis_rot=torch.tensor([axis, axis], dtype=torch.float32),
        origin_pred=torch.zeros(2, 3),
        point_3d_pred=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    # row 0 trans: straight 0.7m ray along z
    assert torch.allclose(out[0, -1], torch.tensor([0.0, 0.0, 0.7]), atol=1e-6)
    # row 1 rot: 90-deg arc of radius 1 about z through origin, from (1,0,0)
    assert torch.allclose(out[1, -1], torch.tensor([-1.0, 1.0, 0.0]), atol=1e-5)
