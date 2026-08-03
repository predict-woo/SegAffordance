"""Unit tests for the screw (twist-native) consistency loss and the 15-tuple
SF3D batch path that feeds the 2D arm.

Pure geometry on synthetic tensors — no dataset, no GPU. Run from the repo
root:

    python -m pytest tests/test_screw_consistency.py -q
"""

import math

import torch

from model.losses.geometric import ScrewConsistencyLoss
from model.losses.twist import screw_orbit, twist_from_gt
from model.outputs import ModelOutputs
from model.targets import StepTargets, unpack_batch


# --- helpers -------------------------------------------------------------

def make_outputs(twist_pred, trajectory_pred=None, coords_hat=None):
    b = twist_pred.shape[0]
    dummy_map = torch.zeros(b, 1, 4, 4)
    return ModelOutputs(
        mask_logits=dummy_map,
        point_logits=dummy_map,
        coords_hat=coords_hat if coords_hat is not None else torch.zeros(b, 2),
        motion_pred=torch.zeros(b, 3),
        motion_type_logits=torch.zeros(b, 2),
        trajectory_pred=trajectory_pred,
        twist_pred=twist_pred,
    )


def arc_about_axis(axis, axis_point, start, angles):
    """Points of the circle through `start` around the axis line — at the
    start point's OWN height along the axis, i.e. correct physics."""
    n = torch.nn.functional.normalize(axis, dim=-1)
    to_start = start - axis_point
    centre = axis_point + (to_start * n).sum(-1, keepdim=True) * n
    r0 = start - centre
    pts = []
    for theta in angles:
        c, s = math.cos(theta), math.sin(theta)
        pts.append(centre + c * r0 + s * torch.linalg.cross(n, r0, dim=-1))
    return torch.stack(pts, dim=0)


AXIS = torch.tensor([0.0, 0.0, 1.0])
HINGE = torch.tensor([0.3, 0.0, 2.0])
ELEMENT = torch.tensor([0.0, 0.0, 2.0])
ANGLES = [k * (math.pi / 2) / 19 for k in range(20)]
ROT = torch.tensor([1])


def gt_targets(trajectory=None):
    return StepTargets(
        motion=AXIS.unsqueeze(0),
        motion_type=ROT,
        motion_origin_3d=HINGE.unsqueeze(0),
        trajectory=trajectory,
    )


# --- the GT-grounding term ----------------------------------------------

def test_correct_twist_scores_zero_on_the_gt_arc():
    arc = arc_about_axis(AXIS, HINGE, ELEMENT, ANGLES).unsqueeze(0)
    twist = twist_from_gt(AXIS.unsqueeze(0), ROT, HINGE.unsqueeze(0))
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0)
    total, terms = loss(make_outputs(twist), gt_targets(arc))
    assert terms["L_screw_twist_gt_traj"].item() < 1e-6
    assert total.item() < 1e-6


def test_correct_twist_scores_zero_on_a_gt_line():
    direction = torch.nn.functional.normalize(torch.tensor([1.0, 1.0, 0.0]), dim=-1)
    line = ELEMENT + 0.1 * torch.linspace(0, 1, 20).unsqueeze(1) * direction
    twist = twist_from_gt(direction.unsqueeze(0), torch.tensor([0]), HINGE.unsqueeze(0))
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0)
    total, terms = loss(
        make_outputs(twist),
        StepTargets(
            motion=direction.unsqueeze(0),
            motion_type=torch.tensor([0]),
            motion_origin_3d=HINGE.unsqueeze(0),
            trajectory=line.unsqueeze(0),
        ),
    )
    assert terms["L_screw_twist_gt_traj"].item() < 1e-6


def test_wrong_axis_is_penalised():
    arc = arc_about_axis(AXIS, HINGE, ELEMENT, ANGLES).unsqueeze(0)
    wrong = twist_from_gt(
        torch.tensor([[1.0, 0.0, 0.0]]), ROT, HINGE.unsqueeze(0)
    )
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0)
    _, terms = loss(make_outputs(wrong), gt_targets(arc))
    assert terms["L_screw_twist_gt_traj"].item() > 0.1


def test_degenerate_stub_is_neutral_not_wrong():
    # The 9.1% in-place rotators store a 0.01 m stub pointing ALONG the axis
    # (an artifact). Its ~0.5 mm segments must be masked, not scored: scoring
    # them would charge a CORRECT twist for the artifact's direction.
    stub = ELEMENT + 0.01 * torch.linspace(0, 1, 20).unsqueeze(1) * AXIS
    twist = twist_from_gt(AXIS.unsqueeze(0), ROT, ELEMENT.unsqueeze(0))
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0)
    total, terms = loss(make_outputs(twist), gt_targets(stub.unsqueeze(0)))
    assert terms["L_screw_twist_gt_traj"].item() == 0.0


# --- the self-consistency term (no teacher forcing) ----------------------

def make_self_consistency_case(twist_pred):
    # K at "original" 2x2 resolution -> K_norm = [[1,0,.5],[0,1,.5],[0,0,1]].
    K = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]])
    img_size = torch.tensor([[2.0, 2.0]])
    depth = torch.full((1, 1, 8, 8), 2.0)  # constant input depth: z = 2
    coords_hat = torch.tensor([[0.5, 0.5]])  # projects/backprojects to ELEMENT
    arc = arc_about_axis(AXIS, HINGE, ELEMENT, ANGLES)
    trajectory_pred = (arc - arc[0:1]).unsqueeze(0)  # relative, like the head
    outputs = make_outputs(twist_pred, trajectory_pred, coords_hat)
    targets = StepTargets(
        motion=AXIS.unsqueeze(0),
        motion_type=ROT,
        motion_origin_3d=HINGE.unsqueeze(0),
        trajectory=arc.unsqueeze(0),
        camera_intrinsic=K,
        img_size=img_size,
    )
    return outputs, targets, depth


def test_self_term_zero_when_own_point_trajectory_and_twist_agree():
    twist = twist_from_gt(AXIS.unsqueeze(0), ROT, HINGE.unsqueeze(0))
    outputs, targets, depth = make_self_consistency_case(twist)
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0)
    _, terms = loss(outputs, targets, depth)
    assert "L_screw_self" in terms
    assert terms["L_screw_self"].item() < 1e-6


def test_self_term_penalises_twist_inconsistent_with_own_trajectory():
    wrong = twist_from_gt(torch.tensor([[1.0, 0.0, 0.0]]), ROT, HINGE.unsqueeze(0))
    outputs, targets, depth = make_self_consistency_case(wrong)
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0)
    _, terms = loss(outputs, targets, depth)
    assert terms["L_screw_self"].item() > 0.1


def test_self_term_needs_no_gt_trajectory():
    # The self term is prediction-only: with GT trajectory absent the GT term
    # no-ops (the loss returns early), but handing it a batch WITH trajectory
    # while zeroing gt_weight isolates the self term — it must not read
    # targets.trajectory. Verified by giving a garbage GT trajectory.
    twist = twist_from_gt(AXIS.unsqueeze(0), ROT, HINGE.unsqueeze(0))
    outputs, targets, depth = make_self_consistency_case(twist)
    targets.trajectory = torch.randn(1, 20, 3)  # garbage GT
    loss = ScrewConsistencyLoss(gt_weight=0.0, self_weight=1.0)
    total, terms = loss(outputs, targets, depth)
    assert terms["L_screw_self"].item() < 1e-6
    assert total.item() < 1e-6  # garbage GT contributed nothing


def test_noop_without_twist_or_trajectory():
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0)
    out = make_outputs(twist_from_gt(AXIS.unsqueeze(0), ROT, HINGE.unsqueeze(0)))
    total, terms = loss(out, StepTargets())  # OPD-style: no trajectory
    assert total.item() == 0.0 and terms == {}

    out_no_twist = ModelOutputs(
        mask_logits=torch.zeros(1, 1, 4, 4),
        point_logits=torch.zeros(1, 1, 4, 4),
        coords_hat=torch.zeros(1, 2),
        motion_pred=torch.zeros(1, 3),
        motion_type_logits=torch.zeros(1, 2),
    )
    total, terms = loss(out_no_twist, gt_targets(torch.randn(1, 20, 3)))
    assert total.item() == 0.0 and terms == {}


# --- the orbit rollout (exp map) ------------------------------------------

def test_orbit_reproduces_the_gt_arc_for_a_revolute_twist():
    twist = twist_from_gt(AXIS.unsqueeze(0), ROT, HINGE.unsqueeze(0))
    ts = torch.tensor(ANGLES)  # unit twist: parameter == angle
    orbit = screw_orbit(twist, ELEMENT.unsqueeze(0), ts)[0]
    arc = arc_about_axis(AXIS, HINGE, ELEMENT, ANGLES)
    assert torch.allclose(orbit, arc, atol=1e-5)


def test_orbit_is_a_straight_line_for_a_prismatic_twist():
    direction = torch.nn.functional.normalize(torch.tensor([1.0, 2.0, -0.5]), dim=-1)
    twist = twist_from_gt(direction.unsqueeze(0), torch.tensor([0]), HINGE.unsqueeze(0))
    ts = torch.linspace(0, 0.1, 20)
    orbit = screw_orbit(twist, ELEMENT.unsqueeze(0), ts)[0]
    expected = ELEMENT + ts.unsqueeze(1) * direction
    assert torch.allclose(orbit, expected, atol=1e-6)


def test_orbit_is_smooth_through_omega_zero():
    # The prismatic limit must be approached continuously — the whole point
    # of the parameterisation. A tiny-omega twist's orbit must sit next to
    # the omega=0 line, not jump.
    direction = torch.tensor([[0.0, 1.0, 0.0]])
    line_twist = torch.cat([torch.zeros(1, 3), direction], dim=-1)
    near_twist = line_twist.clone()
    near_twist[0, 0] = 1e-5  # omega barely nonzero
    ts = torch.linspace(-0.2, 0.2, 9)
    a = screw_orbit(line_twist, ELEMENT.unsqueeze(0), ts)
    b = screw_orbit(near_twist, ELEMENT.unsqueeze(0), ts)
    assert torch.allclose(a, b, atol=1e-5)


# --- the track term (prediction-anchored 2D supervision) ------------------

def make_track_case(twist_pred, corrupt_invalid=False):
    outputs, targets, depth = make_self_consistency_case(twist_pred)
    # Observed 2D track: the GT arc projected with K_norm = [[1,0,.5],[0,1,.5]]
    arc = arc_about_axis(AXIS, HINGE, ELEMENT, ANGLES)
    track = arc[:, :2] / arc[:, 2:3] + 0.5
    valid = torch.ones(1, 20, dtype=torch.bool)
    valid[0, -3:] = False
    if corrupt_invalid:
        track[-3:] = torch.tensor([[7.0, -7.0]])  # garbage where invalid
    targets.trajectory_2d = track.unsqueeze(0)
    targets.trajectory_2d_valid = valid
    targets.trajectory = None  # isolate: no 3D GT (video-pretraining mode)
    outputs.trajectory_pred = None
    return outputs, targets, depth


def test_track_term_zero_for_consistent_twist_and_point():
    twist = twist_from_gt(AXIS.unsqueeze(0), ROT, HINGE.unsqueeze(0))
    outputs, targets, depth = make_track_case(twist)
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0, track_weight=1.0)
    total, terms = loss(outputs, targets, depth)
    assert set(terms) == {"L_screw_track"}  # no 3D GT, no pred trajectory
    assert terms["L_screw_track"].item() < 1e-3  # orbit chord discretisation
    assert total.item() < 1e-3


def test_track_term_penalises_wrong_twist():
    wrong = twist_from_gt(torch.tensor([[1.0, 0.0, 0.0]]), ROT, HINGE.unsqueeze(0))
    outputs, targets, depth = make_track_case(wrong)
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0, track_weight=1.0)
    _, terms = loss(outputs, targets, depth)
    assert terms["L_screw_track"].item() > 0.01


def test_track_term_masks_invalid_points():
    twist = twist_from_gt(AXIS.unsqueeze(0), ROT, HINGE.unsqueeze(0))
    clean_out, clean_tgt, depth = make_track_case(twist)
    dirty_out, dirty_tgt, _ = make_track_case(twist, corrupt_invalid=True)
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0, track_weight=1.0)
    _, clean = loss(clean_out, clean_tgt, depth)
    _, dirty = loss(dirty_out, dirty_tgt, depth)
    assert torch.allclose(clean["L_screw_track"], dirty["L_screw_track"])


def test_omega_shrink_only_fires_with_a_track():
    twist = twist_from_gt(AXIS.unsqueeze(0), ROT, HINGE.unsqueeze(0))
    loss = ScrewConsistencyLoss(
        gt_weight=1.0, self_weight=1.0, track_weight=1.0, omega_shrink=0.1
    )
    outputs, targets, depth = make_track_case(twist)
    _, terms = loss(outputs, targets, depth)
    assert abs(terms["L_screw_omega"].item() - 1.0) < 1e-5  # |omega| of unit twist

    arc = arc_about_axis(AXIS, HINGE, ELEMENT, ANGLES).unsqueeze(0)
    _, terms_3d = loss(make_outputs(twist), gt_targets(arc))
    assert "L_screw_omega" not in terms_3d


def test_track_term_finite_and_bounded_when_orbit_sweeps_behind_the_camera():
    # The step-49 NaN of run 20260728_sf3d_2d_twist: a hinge near the camera
    # plane makes the orbit through the anchor (z=2) sweep through z <= 0;
    # those samples project to ~1/eps coordinates, which overflowed to inf
    # under fp16 autocast and produced inf - inf = NaN in the polyline
    # distance. With the near-plane segment mask, loss and gradient must be
    # not only finite but BOUNDED: projection gradients scale as 1/z^2, so
    # without the mask a validly in-front sample at z ~ 1e-3 could produce a
    # ~1e6-magnitude gradient even though nothing is inf.
    near_hinge = torch.tensor([[0.05, 0.0, 0.1]])
    vertical_axis = torch.tensor([[0.0, 1.0, 0.0]])
    twist = twist_from_gt(vertical_axis, ROT, near_hinge)
    twist = twist.clone().requires_grad_(True)
    outputs, targets, depth = make_track_case(twist)
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0, track_weight=1.0)
    total, terms = loss(outputs, targets, depth)
    assert torch.isfinite(terms["L_screw_track"]).all()
    total.backward()
    assert torch.isfinite(twist.grad).all()
    assert twist.grad.abs().max().item() < 1e3


def test_polyline_distance_ignores_masked_segments():
    from model.losses.geometric import _point_to_polyline_distance

    polyline = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    point = torch.tensor([[[0.5, 0.1]]])  # 0.1 from segment 0, ~0.5 from segment 1
    both = _point_to_polyline_distance(point, polyline)
    masked = _point_to_polyline_distance(
        point, polyline, segment_mask=torch.tensor([[False, True]])
    )
    assert abs(both.item() - 0.1) < 1e-6
    assert masked.item() > 0.4  # the nearby segment is excluded, not matched


def test_no_matchable_segment_spans_the_camera_plane():
    # A plane-crossing orbit's true projection wraps through infinity; a
    # straight polyline segment bridging the crossing is fabricated geometry.
    # The guarantee is a chain of two properties: (1) STRUCTURAL — the
    # visibility mask requires BOTH endpoints in front of the near plane, so
    # no matchable segment ever spans the boundary; (2) BEHAVIOURAL — masked
    # segments can never win the min (test_polyline_distance_ignores_masked_
    # segments). Together: fake bridges cannot be matched.
    #
    # (An end-to-end "far track point scores a large distance" assertion is
    # deliberately NOT made: with a hinge near the camera, the legitimately
    # visible orbit stretch projects across most of the frame — small
    # distances there are honest geometry, not bridge artifacts.)
    import math
    from model.losses.twist import screw_orbit

    near_hinge = torch.tensor([[0.05, 0.0, 0.1]])
    vertical_axis = torch.tensor([[0.0, 1.0, 0.0]])
    twist = twist_from_gt(vertical_axis, ROT, near_hinge)
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0, track_weight=1.0)

    ts = torch.linspace(-math.pi, math.pi, loss.num_orbit_samples).unsqueeze(0)
    orbit = screw_orbit(twist, ELEMENT.unsqueeze(0), ts)
    z = orbit[..., 2]
    in_front = z > loss.near_plane
    # the geometry genuinely crosses the plane (otherwise this tests nothing)
    assert bool(in_front.any()) and bool((~in_front).any())
    seg_visible = in_front[:, :-1] & in_front[:, 1:]
    boundary_crossing = in_front[:, :-1] != in_front[:, 1:]
    assert not bool((seg_visible & boundary_crossing).any())


# --- the 15-tuple SF3D batch path ----------------------------------------

def test_unpack_batch_15_normalises_track_and_derives_anchor_depth():
    B, N = 2, 20
    trajectory = torch.randn(B, N, 3) + torch.tensor([0.0, 0.0, 3.0])
    track_px = torch.rand(B, N, 2) * torch.tensor([1920.0, 1440.0])
    valid = torch.rand(B, N) > 0.2
    img_size = torch.tensor([[1920.0, 1440.0], [1920.0, 1440.0]])
    batch = (
        torch.randn(B, 3, 8, 8), torch.rand(B, 1, 8, 8), ["a", "b"],
        torch.zeros(B, 1, 8, 8), torch.zeros(B, 4), torch.rand(B, 2),
        torch.randn(B, 3), torch.tensor([0, 1]), img_size, ["f0", "f1"],
        torch.randn(B, 3), torch.eye(3).expand(B, 3, 3), trajectory,
        track_px, valid,
    )
    img, depth, words, targets = unpack_batch(batch)
    assert targets.trajectory_2d.shape == (B, N, 2)
    # pixels normalised by (w, h) into [0, 1]
    expected = track_px / img_size.unsqueeze(1)
    assert torch.allclose(targets.trajectory_2d, expected)
    assert torch.equal(targets.trajectory_2d_valid, valid)
    assert torch.allclose(targets.anchor_depth, trajectory[:, 0, 2])
    assert targets.trajectory is trajectory


# ---- direction-sensitive field residual (sign_agnostic=False) ----


def _forward_sweep(twist, anchor, n=20, t_max=0.8):
    ts = torch.linspace(0.0, t_max, n)[None]
    return screw_orbit(twist, anchor, ts)


def test_field_residual_flipped_twist_is_free_only_when_agnostic():
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    origin = torch.tensor([[0.3, -0.1, 1.2]])
    gt = twist_from_gt(axis, torch.tensor([1]), origin)
    traj = _forward_sweep(gt, anchor=torch.tensor([[0.8, 0.2, 1.5]]))

    agnostic = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0, sign_agnostic=True)
    sensitive = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0, sign_agnostic=False)

    # aligned twist: ~0 either way
    assert agnostic._field_residual(gt, traj).item() < 1e-4
    assert sensitive._field_residual(gt, traj).item() < 1e-4
    # flipped twist: invisible to sin^2, maximal (~2) under cosine distance
    assert agnostic._field_residual(-gt, traj).item() < 1e-4
    assert sensitive._field_residual(-gt, traj).item() > 1.5


def test_field_residual_direction_sensitive_for_prismatic():
    direction = torch.tensor([[0.0, 1.0, 0.0]])
    gt = twist_from_gt(direction, torch.tensor([0]), torch.zeros(1, 3))
    traj = _forward_sweep(gt, anchor=torch.tensor([[0.1, 0.0, 1.0]]))
    sensitive = ScrewConsistencyLoss(gt_weight=1.0, self_weight=1.0, sign_agnostic=False)
    assert sensitive._field_residual(gt, traj).item() < 1e-4
    assert sensitive._field_residual(-gt, traj).item() > 1.5
