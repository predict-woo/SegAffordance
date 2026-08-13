"""Unit tests for the projected geometric consistency loss (2D pretraining).

Pure geometry on synthetic tensors — no dataset, no GPU, no model. Every case
builds an exact 3D motion, projects it with a known K to make the "observed"
2D track, then checks the loss scores the true articulation at ~0.

    python -m pytest tests/test_projected_geometric_loss.py -q

Design: docs/superpowers/specs/2026-07-27-2d-pretraining-projection-alignment-design.md
"""

import math

import pytest
import torch

from config.opd_train import LossParams
from model.losses.geometric import ProjectedGeometricLoss, build_geometric_loss
from model.outputs import ModelOutputs
from model.targets import StepTargets

# Synthetic camera: 512x512 original frame, 500px focal length, centred.
FX = FY = 500.0
CX = CY = 256.0
IMG_W = IMG_H = 512.0
N_POINTS = 20


def intrinsics(batch=1):
    K = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    return K.expand(batch, 3, 3).clone()


def img_size(batch=1):
    # dataset convention: [width, height]
    return torch.tensor([[IMG_W, IMG_H]]).expand(batch, 2).clone()


def project(points):
    """(..., 3) camera-frame points -> (..., 2) coords normalised to [0, 1]."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    u = (FX * x / z + CX) / IMG_W
    v = (FY * y / z + CY) / IMG_H
    return torch.stack([u, v], dim=-1)


def rodrigues(n, v, theta):
    """Rotate v about unit axis n by theta. n,v: (3,), theta: (M,) -> (M, 3)."""
    theta = theta.unsqueeze(-1)
    n_cross_v = torch.linalg.cross(n.expand_as(v.expand(theta.shape[0], 3)),
                                   v.expand(theta.shape[0], 3), dim=-1)
    n_dot_v = (n * v).sum()
    return (
        v * torch.cos(theta)
        + n_cross_v * torch.sin(theta)
        + n * n_dot_v * (1.0 - torch.cos(theta))
    )


def prismatic_track(direction=(1.0, 0.0, 0.0), start=(0.1, 0.0, 2.0), extent=0.4,
                    num=N_POINTS, s_values=None):
    """3D points sliding along `direction`, plus their projection."""
    n = torch.tensor(direction); n = n / n.norm()
    P0 = torch.tensor(start)
    s = torch.linspace(0.0, extent, num) if s_values is None else s_values
    pts = P0.unsqueeze(0) + s.unsqueeze(1) * n.unsqueeze(0)
    return pts, project(pts), n, P0


def revolute_track(axis=(0.0, 1.0, 0.0), centre=(0.0, 0.0, 2.0), radius=0.5,
                   sweep=1.0, num=N_POINTS, thetas=None):
    """3D points on a circular arc about `axis` through `centre`."""
    n = torch.tensor(axis); n = n / n.norm()
    c = torch.tensor(centre)
    # a start offset perpendicular to n, of the requested radius
    seed = torch.tensor([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else torch.tensor([0.0, 0.0, 1.0])
    perp = torch.linalg.cross(n, seed, dim=0)
    perp = perp / perp.norm() * radius
    th = torch.linspace(0.0, sweep, num) if thetas is None else thetas
    pts = c.unsqueeze(0) + rodrigues(n, perp, th)
    return pts, project(pts), n, c, pts[0]


def make(track_2d, axis, p_revolute, anchor_depth, origin_xy=None, origin_depth=None):
    """Bundle a ModelOutputs / StepTargets pair for one synthetic sample."""
    b = 1
    if p_revolute == 0.5:
        type_logits = torch.zeros(b, 2)
    else:
        s = 1.0 if p_revolute == 1.0 else -1.0
        type_logits = torch.tensor([[-s * 20.0, s * 20.0]])
    if origin_xy is None:
        origin_xy = torch.tensor([[0.5, 0.5]])
    if origin_depth is None:
        origin_depth = torch.tensor([2.0])
    dummy = torch.zeros(b, 1, 4, 4)
    outputs = ModelOutputs(
        mask_logits=dummy,
        point_logits=dummy,
        point_uv=origin_xy,
        motion_pred=axis.unsqueeze(0) if axis.dim() == 1 else axis,
        motion_type_logits=type_logits,
        trajectory_pred=torch.zeros(b, N_POINTS, 3),
        origin_depth=origin_depth,
    )
    targets = StepTargets(
        trajectory_2d=track_2d.unsqueeze(0),
        camera_intrinsic=intrinsics(b),
        img_size=img_size(b),
        anchor_depth=torch.tensor([anchor_depth]),
    )
    return outputs, targets


def loss_of(outputs, targets, radius_weight=0.0):
    fn = ProjectedGeometricLoss(weight=1.0, radius_weight=radius_weight, radius_ref=1.0)
    total, terms = fn(outputs, targets)
    return total.item(), terms


# --- exactness: the true articulation scores ~0 -------------------------

def test_true_prismatic_articulation_scores_zero():
    _, track, n, P0 = prismatic_track()
    out, tgt = make(track, n, p_revolute=0.0, anchor_depth=float(P0[2]))
    total, terms = loss_of(out, tgt)
    assert total == pytest.approx(0.0, abs=1e-4)
    assert "L_geo_projected" in terms


def test_true_revolute_articulation_scores_zero():
    _, track, n, c, P0 = revolute_track()
    out, tgt = make(track, n, p_revolute=1.0, anchor_depth=float(P0[2]),
                    origin_xy=project(c).unsqueeze(0), origin_depth=torch.tensor([float(c[2])]))
    total, _ = loss_of(out, tgt)
    assert total == pytest.approx(0.0, abs=5e-3)


def test_wrong_axis_costs_more_than_the_true_one():
    _, track, n, P0 = prismatic_track()
    good, _ = loss_of(*make(track, n, 0.0, float(P0[2])))
    wrong_axis = torch.tensor([0.0, 1.0, 0.0])
    bad, _ = loss_of(*make(track, wrong_axis, 0.0, float(P0[2])))
    assert bad > good + 0.01


# --- the invariances the design depends on ------------------------------

def test_body_point_invariance_same_axis_different_radius():
    """A hand and the element sit at different radii on the SAME joint.

    This is the property that lets hand tracks supervise an articulation whose
    other observations are element sweeps; without it the whole 2D pretraining
    scheme is unsound.
    """
    _, _, n, c, _ = revolute_track(radius=0.5)
    origin_xy = project(c).unsqueeze(0)
    origin_z = torch.tensor([float(c[2])])
    for radius in (0.2, 0.5, 0.9):
        _, track, _, _, P0 = revolute_track(radius=radius)
        total, _ = loss_of(*make(track, n, 1.0, float(P0[2]),
                                 origin_xy=origin_xy, origin_depth=origin_z))
        assert total == pytest.approx(0.0, abs=5e-3), f"radius {radius} scored {total}"


def test_phase_invariance_uneven_sampling_along_the_same_curve():
    """A hand that pauses or accelerates traces the same curve; the loss must
    not care where along it each observation falls."""
    _, even, n, P0 = prismatic_track()
    uneven_s = torch.tensor([0.0, 0.01, 0.02, 0.03, 0.05, 0.3, 0.31, 0.32, 0.33, 0.34,
                             0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.40, 0.40, 0.40, 0.40])
    _, uneven, _, _ = prismatic_track(s_values=uneven_s)
    a, _ = loss_of(*make(even, n, 0.0, float(P0[2])))
    b, _ = loss_of(*make(uneven, n, 0.0, float(P0[2])))
    assert a == pytest.approx(0.0, abs=1e-4)
    assert b == pytest.approx(0.0, abs=1e-4)


# --- the containment / model-selection fix ------------------------------

def test_revolute_can_mimic_a_line_without_the_complexity_penalty():
    """Establishes the problem: a large-radius arc fits a straight track just
    as well, so a bare residual gives the gate no reason to prefer prismatic."""
    _, track, n, P0 = prismatic_track()
    # a far-off origin makes the arc locally straight; axis perpendicular to it
    far_axis = torch.tensor([0.0, 1.0, 0.0])
    arc, _ = loss_of(*make(track, far_axis, 1.0, float(P0[2]),
                           origin_depth=torch.tensor([40.0])), radius_weight=0.0)
    line, _ = loss_of(*make(track, n, 0.0, float(P0[2])), radius_weight=0.0)
    assert arc < line + 0.02, "large-radius arc should fit a straight track"


def test_complexity_penalty_makes_the_line_cheaper_than_a_mimicking_arc():
    """The fix: with the radius penalty on, explaining a straight track with a
    distant hinge must cost MORE than calling it prismatic."""
    _, track, n, P0 = prismatic_track()
    far_axis = torch.tensor([0.0, 1.0, 0.0])
    arc, _ = loss_of(*make(track, far_axis, 1.0, float(P0[2]),
                           origin_depth=torch.tensor([40.0])), radius_weight=1.0)
    line, _ = loss_of(*make(track, n, 0.0, float(P0[2])), radius_weight=1.0)
    assert arc > line, f"arc {arc} should cost more than line {line}"


def test_penalty_does_not_punish_a_physically_plausible_radius():
    """A real door hinge is well inside radius_ref and must stay ~free."""
    _, track, n, c, P0 = revolute_track(radius=0.5)
    total, _ = loss_of(*make(track, n, 1.0, float(P0[2]),
                             origin_xy=project(c).unsqueeze(0),
                             origin_depth=torch.tensor([float(c[2])])), radius_weight=1.0)
    assert total == pytest.approx(0.0, abs=0.05)


# --- degeneracy and plumbing --------------------------------------------

def test_stationary_track_is_excluded_not_scored():
    n = torch.tensor([1.0, 0.0, 0.0])
    still = project(torch.tensor([[0.1, 0.0, 2.0]]).expand(N_POINTS, 3))
    total, _ = loss_of(*make(still, n, 0.0, 2.0))
    assert total == 0.0


def test_origin_behind_the_camera_does_not_produce_nan():
    _, track, n, P0 = prismatic_track()
    total, _ = loss_of(*make(track, n, 1.0, float(P0[2]),
                             origin_depth=torch.tensor([-3.0])))
    assert math.isfinite(total)


def test_no_op_without_a_2d_track():
    _, track, n, P0 = prismatic_track()
    out, tgt = make(track, n, 0.0, float(P0[2]))
    tgt.trajectory_2d = None
    total, terms = loss_of(out, tgt)
    assert total == 0.0 and terms == {}


def test_no_op_without_intrinsics():
    _, track, n, P0 = prismatic_track()
    out, tgt = make(track, n, 0.0, float(P0[2]))
    tgt.camera_intrinsic = None
    total, terms = loss_of(out, tgt)
    assert total == 0.0 and terms == {}


def test_gradient_reaches_axis_origin_depth_and_type():
    """All four predicted quantities must actually be trained by this loss."""
    _, track, n, P0 = prismatic_track()
    axis = torch.tensor([[0.2, 0.9, 0.1]], requires_grad=True)
    origin_xy = torch.tensor([[0.4, 0.6]], requires_grad=True)
    origin_depth = torch.tensor([2.5], requires_grad=True)
    out, tgt = make(track, axis, 0.5, float(P0[2]),
                    origin_xy=origin_xy, origin_depth=origin_depth)
    out.motion_type_logits = out.motion_type_logits.clone().requires_grad_(True)
    total, _ = ProjectedGeometricLoss(weight=1.0, radius_weight=1.0, radius_ref=1.0)(out, tgt)
    total.backward()
    assert axis.grad.abs().max() > 0
    assert origin_xy.grad.abs().max() > 0
    assert origin_depth.grad.abs().max() > 0
    assert out.motion_type_logits.grad.abs().max() > 0


def test_weight_scales_total_but_not_the_logged_term():
    _, track, n, P0 = prismatic_track()
    wrong = torch.tensor([0.0, 1.0, 0.0])
    out, tgt = make(track, wrong, 0.0, float(P0[2]))
    total, terms = ProjectedGeometricLoss(weight=0.25, radius_weight=0.0)(out, tgt)
    assert total.item() == pytest.approx(0.25 * terms["L_geo_projected"].item(), rel=1e-5)


def _loss_params(**kw):
    base = dict(bce_weight=.5, dice_weight=.5, mask_weight=.5, point_map_weight=.5,
                coord_weight=.3, vae_weight=.2, motion_type_weight=.5,
                point_sigma=8.0, vae_beta=.01)
    base.update(kw)
    return LossParams(**base)


def test_registry_selects_the_projected_variant():
    fn = build_geometric_loss(_loss_params(geometric_loss="projected"))
    assert isinstance(fn, ProjectedGeometricLoss)
