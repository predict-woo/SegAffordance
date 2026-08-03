"""TrajectoryProjectionLoss: the 2D-only pretraining data term."""

import torch

from model.losses.geometric import (
    TrajectoryProjectionLoss,
    backproject_points,
    normalized_intrinsics,
    project_points,
)
from model.outputs import ModelOutputs
from model.targets import StepTargets


def make_case(rel_traj, z0=1.5, coords=(0.5, 0.5)):
    """Build outputs/targets/depth where the GT 2D track IS the projection
    of the predicted curve — so a perfect prediction has zero loss."""
    B, N, _ = rel_traj.shape
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]).expand(B, 3, 3)
    img_size = torch.tensor([[100.0, 100.0]]).expand(B, 2)
    K_norm = normalized_intrinsics(K, img_size)
    coords_hat = torch.tensor([list(coords)]).expand(B, 2)
    depth = torch.full((B, 1, 8, 8), z0)
    anchor = backproject_points(K_norm, coords_hat, torch.full((B,), z0))
    traj_abs = anchor.unsqueeze(1) + rel_traj
    track = project_points(K_norm, traj_abs)

    dummy = torch.zeros(B, 1, 4, 4)
    outputs = ModelOutputs(
        mask_logits=dummy, point_logits=dummy, coords_hat=coords_hat,
        motion_pred=torch.zeros(B, 3), motion_type_logits=None,
        trajectory_pred=rel_traj,
    )
    targets = StepTargets(
        camera_intrinsic=K, img_size=img_size,
        trajectory_2d=track, trajectory_2d_valid=torch.ones(B, N, dtype=torch.bool),
    )
    return outputs, targets, depth, track


def test_projection_of_own_curve_is_zero_loss():
    rel = torch.tensor([[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.02, 0.0], [0.15, 0.05, 0.01]]])
    outputs, targets, depth, _ = make_case(rel)
    loss, terms = TrajectoryProjectionLoss(weight=1.0)(outputs, targets, depth)
    assert loss.item() < 1e-10
    assert "L_traj_proj" in terms


def test_shifted_track_is_penalised_and_direction_sensitive():
    rel = torch.tensor([[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    outputs, targets, depth, track = make_case(rel)
    # reversed track: same curve as a SET of points, opposite temporal order.
    # Index-matched comparison must see it as wrong (unordered matching
    # would not — that was the old track term's sign loophole).
    targets_rev = StepTargets(
        camera_intrinsic=targets.camera_intrinsic, img_size=targets.img_size,
        trajectory_2d=torch.flip(track, dims=[1]),
        trajectory_2d_valid=targets.trajectory_2d_valid,
    )
    loss_fwd, _ = TrajectoryProjectionLoss(weight=1.0)(outputs, targets, depth)
    loss_rev, _ = TrajectoryProjectionLoss(weight=1.0)(outputs, targets_rev, depth)
    assert loss_fwd.item() < 1e-10
    assert loss_rev.item() > 1e-4


def test_behind_camera_points_masked_and_gradients_finite():
    rel = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.05, 0.0, 0.0]]])
    rel = rel.clone().requires_grad_(True)
    outputs, targets, depth, _ = make_case(rel.detach())
    outputs.trajectory_pred = rel  # z0=1.5, so point 1 sits at z=-0.5: masked
    loss, _ = TrajectoryProjectionLoss(weight=1.0)(outputs, targets, depth)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(rel.grad).all()


def test_noop_without_track_or_weight():
    rel = torch.zeros(1, 3, 3)
    outputs, targets, depth, _ = make_case(rel)
    targets_no = StepTargets(camera_intrinsic=targets.camera_intrinsic, img_size=targets.img_size)
    loss, terms = TrajectoryProjectionLoss(weight=1.0)(outputs, targets_no, depth)
    assert loss.item() == 0.0 and terms == {}
    loss, terms = TrajectoryProjectionLoss(weight=0.0)(outputs, targets, depth)
    assert loss.item() == 0.0 and terms == {}
