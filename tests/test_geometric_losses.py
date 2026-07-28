"""Unit tests for the geometric consistency losses.

Pure geometry on synthetic tensors — no dataset, no GPU, no checkpoint. Run from
the repo root so that ``model`` is importable:

    python -m pytest tests/test_geometric_losses.py -q
"""

import math

import pytest
import torch

from config.opd_train import LossParams
from model.losses.geometric import (
    CrossGTGeometricLoss,
    NoGeometricLoss,
    PredPredGeometricLoss,
    build_geometric_loss,
)
from model.outputs import ModelOutputs
from model.targets import StepTargets


# --- helpers -------------------------------------------------------------

def make_outputs(motion_pred, trajectory_pred, p_revolute):
    """A ModelOutputs carrying only the fields the geometric losses read.

    p_revolute is turned into saturated logits so the softmax reproduces it for
    the 0/1 cases and lands exactly on 0.5 for the undecided case.
    """
    b = motion_pred.shape[0]
    logit = torch.full((b,), 20.0)
    if p_revolute == 0.5:
        type_logits = torch.zeros(b, 2)
    else:
        sign = 1.0 if p_revolute == 1.0 else -1.0
        type_logits = torch.stack([-sign * logit, sign * logit], dim=1)
    dummy_map = torch.zeros(b, 1, 4, 4)
    return ModelOutputs(
        mask_logits=dummy_map,
        point_logits=dummy_map,
        coords_hat=torch.zeros(b, 2),
        motion_pred=motion_pred,
        motion_type_logits=type_logits,
        trajectory_pred=trajectory_pred,
        mu=None,
        log_var=None,
    )


def make_targets(**kwargs):
    base = dict(
        mask=None,
        point_norm=None,
        motion=None,
        motion_type=None,
        img_size=None,
        trajectory=torch.zeros(1, 20, 3),
        motion_origin_3d=None,
        camera_intrinsic=None,
    )
    base.update(kwargs)
    return StepTargets(**base)


def line_trajectory(direction, num_points=20, scale=1.0):
    """Relative trajectory: a straight line from the origin along `direction`."""
    d = torch.as_tensor(direction, dtype=torch.float32)
    d = d / d.norm()
    t = torch.linspace(0.0, scale, num_points).unsqueeze(1)
    return (t * d).unsqueeze(0)


def arc_trajectory(axis, radius=1.0, sweep=math.pi / 2, num_points=20):
    """Relative trajectory: a circular arc in the plane normal to `axis`."""
    n = torch.as_tensor(axis, dtype=torch.float32)
    n = n / n.norm()
    # two unit vectors spanning the plane normal to n
    seed = torch.tensor([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
    e1 = torch.cross(n, seed, dim=0)
    e1 = e1 / e1.norm()
    e2 = torch.cross(n, e1, dim=0)
    theta = torch.linspace(0.0, sweep, num_points).unsqueeze(1)
    pts = radius * (torch.cos(theta) * e1 + torch.sin(theta) * e2)
    return (pts - pts[0:1]).unsqueeze(0)


def random_rotation(seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# --- PredPredGeometricLoss: correctness ----------------------------------

def test_line_along_axis_with_prismatic_type_scores_zero():
    loss = PredPredGeometricLoss(weight=1.0)
    out = make_outputs(
        motion_pred=torch.tensor([[5.0, 0.0, 0.0]]),  # unnormalized on purpose
        trajectory_pred=line_trajectory([1.0, 0.0, 0.0]),
        p_revolute=0.0,
    )
    total, terms = loss(out, make_targets())
    assert total.item() == pytest.approx(0.0, abs=1e-5)
    assert terms["L_geo_pred_pred"].item() == pytest.approx(0.0, abs=1e-5)


def test_arc_normal_to_axis_with_revolute_type_scores_zero():
    loss = PredPredGeometricLoss(weight=1.0)
    out = make_outputs(
        motion_pred=torch.tensor([[0.0, 0.0, 1.0]]),
        trajectory_pred=arc_trajectory([0.0, 0.0, 1.0]),
        p_revolute=1.0,
    )
    total, _ = loss(out, make_targets())
    assert total.item() == pytest.approx(0.0, abs=1e-5)


def test_line_along_axis_with_revolute_type_scores_one():
    """Catches an inverted motion-type class index, which would silently
    flip the whole loss."""
    loss = PredPredGeometricLoss(weight=1.0)
    out = make_outputs(
        motion_pred=torch.tensor([[1.0, 0.0, 0.0]]),
        trajectory_pred=line_trajectory([1.0, 0.0, 0.0]),
        p_revolute=1.0,
    )
    total, _ = loss(out, make_targets())
    assert total.item() == pytest.approx(1.0, abs=1e-5)


def test_arc_normal_to_axis_with_prismatic_type_scores_one():
    loss = PredPredGeometricLoss(weight=1.0)
    out = make_outputs(
        motion_pred=torch.tensor([[0.0, 0.0, 1.0]]),
        trajectory_pred=arc_trajectory([0.0, 0.0, 1.0]),
        p_revolute=0.0,
    )
    total, _ = loss(out, make_targets())
    assert total.item() == pytest.approx(1.0, abs=1e-5)


def test_undecided_type_scores_half_whatever_the_geometry():
    loss = PredPredGeometricLoss(weight=1.0)
    for traj in (line_trajectory([1.0, 0.0, 0.0]), arc_trajectory([0.0, 0.0, 1.0])):
        out = make_outputs(
            motion_pred=torch.tensor([[1.0, 0.0, 0.0]]),
            trajectory_pred=traj,
            p_revolute=0.5,
        )
        total, _ = loss(out, make_targets())
        assert total.item() == pytest.approx(0.5, abs=1e-5)


def test_undecided_type_sends_no_gradient_to_axis_or_trajectory():
    """dL/dR = 2p-1, so at p=0.5 the geometric heads must be untouched."""
    loss = PredPredGeometricLoss(weight=1.0)
    motion = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    traj = arc_trajectory([0.0, 0.0, 1.0]).clone().requires_grad_(True)
    total, _ = loss(make_outputs(motion, traj, p_revolute=0.5), make_targets())
    total.backward()
    assert motion.grad.abs().max().item() == pytest.approx(0.0, abs=1e-6)
    assert traj.grad.abs().max().item() == pytest.approx(0.0, abs=1e-6)


# --- PredPredGeometricLoss: invariances ----------------------------------

def test_invariant_to_axis_sign():
    loss = PredPredGeometricLoss(weight=1.0)
    traj = arc_trajectory([0.0, 0.0, 1.0], sweep=1.1)
    axis = torch.tensor([[0.3, -0.7, 1.0]])
    a, _ = loss(make_outputs(axis, traj, 1.0), make_targets())
    b, _ = loss(make_outputs(-axis, traj, 1.0), make_targets())
    assert a.item() == pytest.approx(b.item(), abs=1e-6)


def test_invariant_to_trajectory_scale():
    loss = PredPredGeometricLoss(weight=1.0)
    axis = torch.tensor([[0.3, -0.7, 1.0]])
    traj = arc_trajectory([0.0, 0.0, 1.0], sweep=1.1)
    a, _ = loss(make_outputs(axis, traj, 1.0), make_targets())
    b, _ = loss(make_outputs(axis, traj * 137.0, 1.0), make_targets())
    assert a.item() == pytest.approx(b.item(), abs=1e-5)


def test_invariant_to_shared_rotation_of_axis_and_trajectory():
    loss = PredPredGeometricLoss(weight=1.0)
    axis = torch.tensor([[0.3, -0.7, 1.0]])
    traj = arc_trajectory([0.0, 0.0, 1.0], sweep=1.1)
    rot = random_rotation(seed=7)
    a, _ = loss(make_outputs(axis, traj, 1.0), make_targets())
    b, _ = loss(
        make_outputs(axis @ rot.T, traj @ rot.T, 1.0), make_targets()
    )
    assert a.item() == pytest.approx(b.item(), abs=1e-5)


# --- PredPredGeometricLoss: degeneracy and gradients ---------------------

def test_collapsed_trajectory_is_excluded_from_the_batch_mean():
    """A zero trajectory has no direction, so it must not be scored at all.

    Scoring it would give R=0 => L=(1-p), a spurious 'everything is revolute'
    signal. Batch = [line+revolute (L=1), collapsed]; mean must be 1.0, not 0.5.
    """
    loss = PredPredGeometricLoss(weight=1.0)
    traj = torch.cat([line_trajectory([1.0, 0.0, 0.0]), torch.zeros(1, 20, 3)], dim=0)
    axis = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    total, _ = loss(make_outputs(axis, traj, p_revolute=1.0), make_targets())
    assert total.item() == pytest.approx(1.0, abs=1e-5)


def test_fully_collapsed_batch_returns_exactly_zero():
    loss = PredPredGeometricLoss(weight=1.0)
    out = make_outputs(
        motion_pred=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        trajectory_pred=torch.zeros(2, 20, 3),
        p_revolute=1.0,
    )
    total, _ = loss(out, make_targets())
    assert total.item() == 0.0


def test_gradient_reaches_both_the_axis_and_the_trajectory_head():
    """The point of the loss: coupling must be two-way."""
    loss = PredPredGeometricLoss(weight=1.0)
    motion = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    traj = line_trajectory([1.0, 1.0, 0.3]).clone().requires_grad_(True)
    total, _ = loss(make_outputs(motion, traj, p_revolute=0.0), make_targets())
    total.backward()
    assert motion.grad.abs().max().item() > 1e-6
    assert traj.grad.abs().max().item() > 1e-6


def test_weight_scales_the_total_but_not_the_logged_term():
    out = make_outputs(
        motion_pred=torch.tensor([[1.0, 0.0, 0.0]]),
        trajectory_pred=line_trajectory([1.0, 0.0, 0.0]),
        p_revolute=1.0,
    )
    total, terms = PredPredGeometricLoss(weight=0.25)(out, make_targets())
    assert total.item() == pytest.approx(0.25, abs=1e-5)
    assert terms["L_geo_pred_pred"].item() == pytest.approx(1.0, abs=1e-5)


def test_no_op_when_the_batch_carries_no_trajectory():
    loss = PredPredGeometricLoss(weight=1.0)
    out = make_outputs(
        motion_pred=torch.tensor([[1.0, 0.0, 0.0]]),
        trajectory_pred=line_trajectory([1.0, 0.0, 0.0]),
        p_revolute=1.0,
    )
    total, terms = loss(out, make_targets(trajectory=None))
    assert total.item() == 0.0
    assert terms == {}


# --- NoGeometricLoss and the registry ------------------------------------

def test_no_geometric_loss_contributes_nothing():
    out = make_outputs(
        motion_pred=torch.tensor([[1.0, 0.0, 0.0]]),
        trajectory_pred=line_trajectory([1.0, 0.0, 0.0]),
        p_revolute=1.0,
    )
    total, terms = NoGeometricLoss()(out, make_targets())
    assert total.item() == 0.0
    assert terms == {}


def _loss_params(**kwargs):
    base = dict(
        bce_weight=0.5, dice_weight=0.5, mask_weight=0.5, point_map_weight=0.5,
        coord_weight=0.3, vae_weight=0.2, motion_type_weight=0.5,
        point_sigma=8.0, vae_beta=0.01,
    )
    base.update(kwargs)
    return LossParams(**base)


def test_registry_defaults_to_cross_gt_so_existing_configs_are_unchanged():
    assert isinstance(build_geometric_loss(_loss_params()), CrossGTGeometricLoss)


def test_registry_selects_each_variant_by_name():
    assert isinstance(
        build_geometric_loss(_loss_params(geometric_loss="pred_pred")),
        PredPredGeometricLoss,
    )
    assert isinstance(
        build_geometric_loss(_loss_params(geometric_loss="none")), NoGeometricLoss
    )


def test_registry_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="nonsense"):
        build_geometric_loss(_loss_params(geometric_loss="nonsense"))


# --- CrossGTGeometricLoss: behaviour preservation ------------------------

def _reference_geometric_consistency_loss(
    trajectory_pred, motion_pred, motion_type_gt, motion_origin_3d, trajectory_gt_first
):
    """Oracle for CrossGTGeometricLoss.

    Originally a verbatim copy of OPDRealTrainingModule._geometric_consistency_loss
    as it stood before the refactor (train_OPDReal_better.py:781, commit a314e9e),
    frozen to prove the refactor changed no numbers.

    Updated 2026-07-28: the revolute plane term ``dot_n**2`` was removed from
    the production loss and therefore from this oracle. It required every
    trajectory point to sit at the motion origin's height along the axis, which
    is not what rotation about an axis means -- see CrossGTGeometricLoss for
    the full rationale. Everything else, including the prismatic branch and the
    per-sample loop's float associativity, is still pinned verbatim.
    """
    import torch.nn.functional as F

    B, N, _ = trajectory_pred.shape
    device = trajectory_pred.device
    motion_pred_norm = F.normalize(motion_pred, p=2, dim=1, eps=1e-8)
    total_loss = torch.zeros(B, device=device)

    for b in range(B):
        if motion_type_gt[b] == 0:
            P_0 = motion_origin_3d[b]
            v = motion_pred_norm[b]
            Q = trajectory_pred[b]
            Q_minus_P0 = Q - P_0
            cross_product = torch.cross(Q_minus_P0, v.unsqueeze(0).expand(N, -1))
            squared_distances = torch.sum(cross_product**2, dim=1)
            total_loss[b] = squared_distances.mean()
        elif motion_type_gt[b] == 1:
            C = motion_origin_3d[b]
            n = motion_pred_norm[b]
            Q = trajectory_pred[b]
            Q_first_gt = trajectory_gt_first[b]
            Q_first_minus_C = Q_first_gt - C
            proj_length = torch.dot(Q_first_minus_C, n)
            proj_perp = Q_first_minus_C - proj_length * n
            r = torch.norm(proj_perp)
            Q_minus_C = Q - C
            proj_lengths = torch.sum(Q_minus_C * n.unsqueeze(0).expand(N, -1), dim=1)
            proj_perp_vecs = Q_minus_C - proj_lengths.unsqueeze(1) * n.unsqueeze(0).expand(N, -1)
            circle_dists = torch.norm(proj_perp_vecs, dim=1)
            circle_error_sq = (circle_dists - r) ** 2
            total_loss[b] = circle_error_sq.mean()

    return total_loss.mean()


def test_revolute_loss_is_zero_for_a_correct_arc_off_the_origin_plane():
    """A correct rotation must score 0 even when the hinge is not level with it.

    This pins the 2026-07-28 correction. The revolute branch used to add
    ((Q-C).n)^2, which forced the trajectory to share the motion origin's
    height along the axis. A door handle sweeps a circle at its own height, so
    that term charged a perfectly valid arc the squared along-axis offset --
    here 0.5^2 = 0.25 -- and the optimiser could only reduce it by tilting the
    predicted axis away from the truth.
    """
    n = 20
    axis = torch.tensor([[0.0, 1.0, 0.0]])           # vertical hinge
    radius, height_offset = 0.3, 0.5
    theta = torch.linspace(0.0, 1.5707963, n)
    # Arc in the plane y = height_offset; the motion origin sits at y = 0.
    arc = torch.stack(
        [radius * torch.cos(theta),
         torch.full((n,), height_offset),
         radius * torch.sin(theta)],
        dim=1,
    ).unsqueeze(0)
    origin = torch.zeros(1, 3)

    # The loss consumes the relative frame, exactly as CrossGT does.
    arc_rel = arc - arc[:, 0:1, :]
    origin_rel = origin - arc[:, 0, :]

    loss = CrossGTGeometricLoss._consistency(
        trajectory=arc_rel,
        axis=axis,
        motion_type_gt=torch.tensor([1]),
        motion_origin_3d=origin_rel,
        trajectory_gt_first=torch.zeros(1, 3),
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-10), (
        "a correct revolute arc must cost nothing regardless of how far the "
        f"hinge sits along the axis; got {loss.item()}"
    )


def test_cross_gt_reproduces_the_pre_refactor_numbers():
    torch.manual_seed(0)
    b, n = 6, 20
    traj_pred = torch.randn(b, n, 3)
    traj_gt = torch.randn(b, n, 3)
    motion_pred = torch.randn(b, 3)
    motion_gt = torch.randn(b, 3)
    motion_type = torch.tensor([0, 1, 0, 1, 1, 0])
    origin = torch.randn(b, 3)

    traj_gt_first = traj_gt[:, 0:1, :]
    traj_gt_rel = traj_gt - traj_gt_first
    origin_rel = origin - traj_gt_first.squeeze(1)
    zeros = torch.zeros_like(origin_rel)

    expected_a = _reference_geometric_consistency_loss(
        traj_gt_rel, motion_pred, motion_type, origin_rel, zeros
    )
    expected_b = _reference_geometric_consistency_loss(
        traj_pred, motion_gt, motion_type, origin_rel, zeros
    )

    out = ModelOutputs(
        mask_logits=torch.zeros(b, 1, 4, 4),
        point_logits=torch.zeros(b, 1, 4, 4),
        coords_hat=torch.zeros(b, 2),
        motion_pred=motion_pred,
        motion_type_logits=torch.zeros(b, 2),
        trajectory_pred=traj_pred,
        mu=None,
        log_var=None,
    )
    targets = make_targets(
        motion=motion_gt,
        motion_type=motion_type,
        trajectory=traj_gt,
        motion_origin_3d=origin,
    )
    total, terms = CrossGTGeometricLoss(
        geometric_weight=0.5, trajectory_to_motion_weight=0.5
    )(out, targets)

    assert terms["L_geometric_pred_vector_gt_traj"].item() == pytest.approx(
        expected_a.item(), rel=1e-6
    )
    assert terms["L_geometric_pred_traj_gt_vector"].item() == pytest.approx(
        expected_b.item(), rel=1e-6
    )
    assert total.item() == pytest.approx(
        0.5 * expected_a.item() + 0.5 * expected_b.item(), rel=1e-6
    )


def test_cross_gt_no_ops_without_a_motion_origin():
    out = make_outputs(
        motion_pred=torch.tensor([[1.0, 0.0, 0.0]]),
        trajectory_pred=line_trajectory([1.0, 0.0, 0.0]),
        p_revolute=1.0,
    )
    targets = make_targets(
        motion=torch.tensor([[1.0, 0.0, 0.0]]),
        motion_type=torch.tensor([0]),
        trajectory=torch.zeros(1, 20, 3),
        motion_origin_3d=None,
    )
    total, terms = CrossGTGeometricLoss(0.5, 0.5)(out, targets)
    assert total.item() == 0.0
    assert terms == {}
