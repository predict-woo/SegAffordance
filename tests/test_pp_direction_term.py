"""Behavior + gradient tests for the midpoint screw-direction term in
PredPredArticulationLoss (sign consistency between the articulation
parameters and the trajectory's time ordering).

The locus residuals are invariant to axis -> -axis; this term is the
oriented complement: 1 - cos between trajectory chords and the screw's
velocity field evaluated at chord MIDPOINTS. Midpoint evaluation is what
makes it EXACTLY zero for a perfect arc at any discretization (a chord is
parallel to the tangent at the arc midpoint) — the endpoint version keeps
a 1 - cos(step/2) floor and a nonzero gradient at the optimum.

Pure synthetic geometry, CPU-only:

    python -m pytest tests/test_pp_direction_term.py -q
"""

import math

import pytest
import torch

from model.losses.geometric import PredPredArticulationLoss
from model.outputs import ModelOutputs


# --- fixtures ------------------------------------------------------------

def make_outputs(
    axis,
    trajectory_rel,
    p_revolute,
    origin=None,
    p0=None,
    axis_rot=None,
    axis_trans=None,
):
    """ModelOutputs carrying only what PredPredArticulationLoss reads.

    trajectory_rel is (1, N, 3) relative to its first point (= 0). p0 is the
    absolute first point (default 0), origin an absolute point on the axis
    (default 0) — the loss forms the relative axis anchor c = origin - p0.
    """
    b = 1
    logit = torch.full((b,), 20.0)
    if p_revolute == 0.5:
        type_logits = torch.zeros(b, 2)
    else:
        sign = 1.0 if p_revolute == 1.0 else -1.0
        type_logits = torch.stack([-sign * logit, sign * logit], dim=1)
    dummy_map = torch.zeros(b, 1, 4, 4)
    as_t = lambda x: None if x is None else torch.as_tensor(x, dtype=torch.float32).reshape(1, 3)
    return ModelOutputs(
        mask_logits=dummy_map,
        point_logits=dummy_map,
        point_uv=torch.zeros(b, 2),
        motion_pred=as_t(axis),
        motion_pred_rot=as_t(axis_rot),
        motion_pred_trans=as_t(axis_trans),
        motion_type_logits=type_logits,
        trajectory_pred=trajectory_rel,
        origin_pred=as_t(origin if origin is not None else [0.0, 0.0, 0.0]),
        point_3d_pred=as_t(p0 if p0 is not None else [0.0, 0.0, 0.0]),
    )


def dir_only_loss(**kw):
    """The loss with ONLY the direction term active (locus weight zero)."""
    return PredPredArticulationLoss(weight=0.0, dir_weight=1.0, **kw)


def line_trajectory(direction, num_points=20, scale=1.0):
    d = torch.as_tensor(direction, dtype=torch.float32)
    d = d / d.norm()
    t = torch.linspace(0.0, scale, num_points).unsqueeze(1)
    return (t * d).unsqueeze(0)


def arc_setup(axis, radius=1.0, sweep=math.pi / 2, num_points=20):
    """Positive (right-hand) rotation about `axis`: returns the RELATIVE
    trajectory plus the absolute first point (circle center at 0)."""
    n = torch.as_tensor(axis, dtype=torch.float32)
    n = n / n.norm()
    seed = torch.tensor([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
    e1 = torch.cross(n, seed, dim=0)
    e1 = e1 / e1.norm()
    e2 = torch.cross(n, e1, dim=0)  # n x e1 = e2  ->  d(pts)/dtheta = n x pts
    theta = torch.linspace(0.0, sweep, num_points).unsqueeze(1)
    pts = radius * (torch.cos(theta) * e1 + torch.sin(theta) * e2)
    return (pts - pts[0:1]).unsqueeze(0), pts[0]


AXES = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.6, -0.3, 0.8]]


# --- exactness at consistency --------------------------------------------

@pytest.mark.parametrize("axis", AXES)
def test_prismatic_aligned_is_zero(axis):
    out = make_outputs(axis, line_trajectory(axis), p_revolute=0.0)
    total, terms = dir_only_loss()(out, None)
    assert terms["L_geo_pp_dir"].item() == pytest.approx(0.0, abs=1e-6)
    assert total.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("num_points", [4, 7, 20])
def test_revolute_exact_zero_at_any_discretization(axis, num_points):
    # Coarse arcs: 120 deg in 3 segments = 40 deg per step. The endpoint
    # formulation would floor at 1 - cos(20 deg) ~= 0.060; midpoint must be 0.
    traj, p0 = arc_setup(axis, sweep=2 * math.pi / 3, num_points=num_points)
    out = make_outputs(axis, traj, p_revolute=1.0, p0=p0)
    _, terms = dir_only_loss()(out, None)
    assert terms["L_geo_pp_dir"].item() == pytest.approx(0.0, abs=1e-6)


def test_endpoint_evaluation_would_not_be_zero():
    # Reference check documenting WHY midpoints: the same arc scored with
    # the velocity field at segment STARTS keeps the half-step-angle floor.
    axis = torch.tensor([0.0, 0.0, 1.0])
    traj, p0 = arc_setup(axis, sweep=2 * math.pi / 3, num_points=4)
    d = traj[0]
    seg = d[1:] - d[:-1]
    c = -p0  # relative axis anchor
    v_end = torch.cross(axis.expand_as(d[:-1]), d[:-1] - c, dim=-1)
    cos_end = torch.nn.functional.cosine_similarity(seg, v_end, dim=-1)
    floor = (1.0 - cos_end).mean().item()
    assert floor == pytest.approx(1.0 - math.cos(math.radians(20.0)), abs=1e-4)
    assert floor > 0.05


def test_origin_gauge_invariance_along_axis():
    axis = [0.0, 1.0, 0.0]
    traj, p0 = arc_setup(axis, sweep=math.pi / 2)
    vals = []
    for t in (0.0, -3.0, 7.5):
        origin = torch.tensor(axis) * t
        out = make_outputs(axis, traj, p_revolute=1.0, p0=p0, origin=origin)
        _, terms = dir_only_loss()(out, None)
        vals.append(terms["L_geo_pp_dir"].item())
    assert max(vals) - min(vals) < 1e-7


# --- sign discrimination -------------------------------------------------

@pytest.mark.parametrize("axis", AXES)
def test_prismatic_flip_scores_two(axis):
    flipped = [-a for a in axis]
    out = make_outputs(flipped, line_trajectory(axis), p_revolute=0.0)
    _, terms = dir_only_loss()(out, None)
    assert terms["L_geo_pp_dir"].item() == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize("axis", AXES)
def test_revolute_flip_scores_two(axis):
    traj, p0 = arc_setup(axis, sweep=math.pi / 2)
    flipped = [-a for a in axis]
    out = make_outputs(flipped, traj, p_revolute=1.0, p0=p0)
    _, terms = dir_only_loss()(out, None)
    assert terms["L_geo_pp_dir"].item() == pytest.approx(2.0, abs=1e-5)


def test_locus_term_is_blind_where_dir_term_sees():
    # The regression that motivated the term: the LOCUS residual is identical
    # under an axis flip; the direction term separates 0 from 2.
    axis = [0.0, 0.0, 1.0]
    traj, p0 = arc_setup(axis, sweep=math.pi / 2)
    locus = PredPredArticulationLoss(weight=1.0, normalized=True)
    l_true = locus(make_outputs(axis, traj, 1.0, p0=p0), None)[0].item()
    l_flip = locus(make_outputs([0.0, 0.0, -1.0], traj, 1.0, p0=p0), None)[0].item()
    assert l_flip == pytest.approx(l_true, abs=1e-6)


# --- gradient quality ----------------------------------------------------

def _dir_term_for_axis(axis_param, traj, p_rev, p0):
    out = make_outputs([1.0, 0.0, 0.0], traj, p_rev, p0=p0)
    out.motion_pred = axis_param.reshape(1, 3)
    total, _ = dir_only_loss()(out, None)
    return total


@pytest.mark.parametrize("p_rev", [0.0, 1.0])
def test_gradient_vanishes_at_consistency(p_rev):
    axis = torch.tensor([0.0, 0.0, 1.0])
    if p_rev == 1.0:
        traj, p0 = arc_setup(axis, sweep=math.pi / 2)
    else:
        traj, p0 = line_trajectory(axis), torch.zeros(3)
    param = axis.clone().requires_grad_(True)
    _dir_term_for_axis(param, traj, p_rev, p0).backward()
    assert param.grad is not None
    assert param.grad.norm().item() < 1e-5


@pytest.mark.parametrize("p_rev", [0.0, 1.0])
@pytest.mark.parametrize("off_deg", [90.0, 170.0])
def test_gradient_descent_recovers_axis(p_rev, off_deg):
    # Init the axis badly wrong (incl. nearly antipodal — 170 deg; exactly
    # 180 is a measure-zero saddle) and check the direction term alone pulls
    # it back to the true oriented axis.
    true_axis = torch.tensor([0.0, 0.0, 1.0])
    if p_rev == 1.0:
        traj, p0 = arc_setup(true_axis, sweep=math.pi / 2)
    else:
        traj, p0 = line_trajectory(true_axis), torch.zeros(3)
    a = math.radians(off_deg)
    param = torch.tensor([math.sin(a), 0.0, math.cos(a)], requires_grad=True)
    opt = torch.optim.Adam([param], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        _dir_term_for_axis(param, traj, p_rev, p0).backward()
        opt.step()
    cos = torch.nn.functional.cosine_similarity(
        param.detach(), true_axis, dim=0
    ).item()
    assert cos > 0.99


def test_gradient_monotone_toward_flip():
    # 1 - cos gives a smooth, informative landscape: loss increases with the
    # tilt angle all the way to the flip (no plateaus except the antipode).
    true_axis = torch.tensor([0.0, 0.0, 1.0])
    traj, p0 = arc_setup(true_axis, sweep=math.pi / 2)
    losses = []
    for deg in (0, 45, 90, 135, 179):
        a = math.radians(deg)
        axis = [math.sin(a), 0.0, math.cos(a)]
        out = make_outputs(axis, traj, p_revolute=1.0, p0=p0)
        losses.append(dir_only_loss()(out, None)[0].item())
    assert all(b > a for a, b in zip(losses, losses[1:]))


# --- degeneracy and masking ----------------------------------------------

def test_stationary_trajectory_is_finite_zero():
    out = make_outputs([0.0, 0.0, 1.0], torch.zeros(1, 20, 3), p_revolute=1.0)
    out.motion_pred.requires_grad_(True)
    total, terms = dir_only_loss()(out, None)
    assert total.item() == 0.0
    total.backward()  # graph-connected zero — must not raise
    assert torch.isfinite(out.motion_pred.grad).all()


def test_near_axis_arc_is_masked_not_nan():
    # Radius below dir_lever_floor: the tangent is ill-defined, the segments
    # must be masked out rather than produce garbage direction gradients.
    axis = torch.tensor([0.0, 0.0, 1.0])
    traj, p0 = arc_setup(axis, radius=0.005, sweep=math.pi / 2)
    out = make_outputs(axis, traj, p_revolute=1.0, p0=p0)
    out.motion_pred.requires_grad_(True)
    total, terms = dir_only_loss(dir_seg_floor=1e-6)(out, None)
    assert terms["L_geo_pp_dir"].item() == 0.0
    total.backward()
    assert torch.isfinite(out.motion_pred.grad).all()


def test_default_off_changes_nothing():
    axis = [0.0, 0.0, 1.0]
    traj, p0 = arc_setup(axis, sweep=math.pi / 2)
    out = make_outputs([0.0, 0.0, -1.0], traj, p_revolute=1.0, p0=p0)
    loss = PredPredArticulationLoss(weight=1.0, normalized=True)
    total, terms = loss(out, None)
    assert "L_geo_pp_dir" not in terms  # flag off -> not even logged


def test_split_heads_pairing():
    # With split axis heads, the prismatic branch must read the TRANS
    # candidate and the revolute branch the ROT candidate.
    axis = [0.0, 0.0, 1.0]
    line = line_trajectory(axis)
    out = make_outputs(
        axis, line, p_revolute=0.0,
        axis_trans=axis, axis_rot=[0.0, 0.0, -1.0],  # rot candidate flipped
    )
    _, terms = dir_only_loss()(out, None)
    assert terms["L_geo_pp_dir"].item() == pytest.approx(0.0, abs=1e-6)

    traj, p0 = arc_setup(torch.tensor(axis), sweep=math.pi / 2)
    out = make_outputs(
        axis, traj, p_revolute=1.0, p0=p0,
        axis_rot=axis, axis_trans=[0.0, 0.0, -1.0],  # trans candidate flipped
    )
    _, terms = dir_only_loss()(out, None)
    assert terms["L_geo_pp_dir"].item() == pytest.approx(0.0, abs=1e-6)
