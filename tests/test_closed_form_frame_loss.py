"""closed_form_frame_loss (2026-09-11): the loss designed from the shape of
the master formula — p = 0 (unit levers), rho = 0 (no leak), symmetric
radius term. knowledge/2026-09-11_revolute_loss_with_axis_term.md."""
import math

import pytest
import torch

from model.losses.geometric import closed_form_frame_loss


def _rot(n, o, p0, n_gt=None, o_gt=None, p0_gt=None, **kw):
    n_gt = n if n_gt is None else n_gt
    o_gt = o if o_gt is None else o_gt
    p0_gt = p0 if p0_gt is None else p0_gt
    B = n.shape[0]
    total, comps = closed_form_frame_loss(
        motion_type_gt=torch.ones(B), axis_trans=torch.zeros(B, 3), axis_rot=n,
        origin_pred=o, point_3d_pred=p0, axis_gt=n_gt, origin_gt=o_gt, traj_start_gt=p0_gt, **kw)
    return total, comps


def _random_case(seed, B=8):
    g = torch.Generator().manual_seed(seed)
    n = torch.nn.functional.normalize(torch.randn(B, 3, generator=g), dim=1)
    o = torch.randn(B, 3, generator=g)
    p0 = o + torch.randn(B, 3, generator=g)
    return n, o, p0


def test_zero_at_ground_truth():
    n, o, p0 = _random_case(0)
    total, comps = _rot(n, o, p0)
    assert total.abs().max() < 1e-5
    assert all(c.abs().max() < 1e-5 for c in comps.values())


def test_flipped_axis_costs_exactly_twice_the_axis_weight_for_any_origin_and_point():
    # rho = 0: the flip penalty cannot be bought back by moving o / p0.
    n, o, p0 = _random_case(1)
    g = torch.Generator().manual_seed(7)
    worst = 1e9
    for _ in range(200):
        o2 = o + 0.5 * torch.randn(*o.shape, generator=g)
        p2 = p0 + 0.5 * torch.randn(*p0.shape, generator=g)
        total, comps = _rot(-n, o2, p2, n_gt=n, o_gt=o, p0_gt=p0, radius_weight=0.0)
        worst = min(worst, float(total.min()))
        assert (comps["axis"] - 2.0).abs().max() < 1e-5
    assert worst >= 2 * 2.0 - 1e-4       # axis_weight 2 * (1 - (-1)) = 4, phase term >= 0


def test_axis_and_phase_terms_are_scale_free_in_the_predicted_lever():
    n, o, p0 = _random_case(2)
    n2 = torch.nn.functional.normalize(n + 0.3 * torch.randn(*n.shape, generator=torch.Generator().manual_seed(3)), dim=1)
    _, c1 = _rot(n2, o, p0, n_gt=n, o_gt=o, p0_gt=p0)
    _, c2 = _rot(n2, o, o + 3.0 * (p0 - o), n_gt=n, o_gt=o, p0_gt=p0)   # predicted lever 3x longer
    assert torch.allclose(c1["axis"], c2["axis"], atol=1e-5)
    assert torch.allclose(c1["phase"], c2["phase"], atol=1e-5)
    # with the axes different the two levers are projections onto different planes, so check the
    # exact log-ratio value with the axis exact instead
    _, c3 = _rot(n, o, o + 3.0 * (p0 - o), o_gt=o, p0_gt=p0)
    assert torch.allclose(c3["radius"], torch.full_like(c3["radius"], math.log(3.0) ** 2), atol=1e-5)


def test_radius_term_symmetric_in_log_ratio_and_sq_form_available():
    n, o, p0 = _random_case(4)
    _, half = _rot(n, o, o + 0.5 * (p0 - o), p0_gt=p0)
    _, dbl = _rot(n, o, o + 2.0 * (p0 - o), p0_gt=p0)
    assert torch.allclose(half["radius"], dbl["radius"], atol=1e-5)
    _, sq = _rot(n, o, o + 2.0 * (p0 - o), p0_gt=p0, radius_form="sq")
    assert torch.allclose(sq["radius"], torch.ones_like(sq["radius"]), atol=1e-5)


def test_phase_term_is_one_minus_cos_of_the_inplane_turn_when_axes_agree():
    # rotate the GT lever about the axis by psi: phase = (1 + 1)(1 - cos psi)
    n = torch.tensor([[0.0, 0.0, 1.0]]); o = torch.zeros(1, 3); p0 = torch.tensor([[1.0, 0.0, 0.0]])
    for psi in (0.3, 1.2, 2.5):
        p2 = torch.tensor([[math.cos(psi), math.sin(psi), 0.0]])
        _, comps = _rot(n, o, p2, p0_gt=p0)
        assert abs(float(comps["phase"]) - 2.0 * (1.0 - math.cos(psi))) < 1e-5
        assert float(comps["axis"]) < 1e-6 and float(comps["radius"]) < 1e-6


def test_collapsed_lever_is_bounded_and_keeps_the_axis_gradient():
    n, o, p0 = _random_case(5)
    n_pred = torch.nn.functional.normalize(n + 0.5 * torch.randn(*n.shape, generator=torch.Generator().manual_seed(9)), dim=1).requires_grad_(True)
    total, comps = _rot(n_pred, o, o + 1e-4 * (p0 - o), n_gt=n, o_gt=o, p0_gt=p0)   # predicted lever ~0
    assert torch.isfinite(total).all()
    total.sum().backward()
    assert n_pred.grad.norm(dim=-1).min() > 1e-3     # p = 0: axis gradient survives a collapsed lever
    assert torch.allclose(comps["radius"], torch.full_like(comps["radius"], math.log(0.1) ** 2), atol=1e-4)  # floored at 0.1


def test_trans_rows_are_the_direction_term():
    B = 4
    g = torch.Generator().manual_seed(11)
    d = torch.nn.functional.normalize(torch.randn(B, 3, generator=g), dim=1)
    d_gt = torch.nn.functional.normalize(torch.randn(B, 3, generator=g), dim=1)
    total, comps = closed_form_frame_loss(
        motion_type_gt=torch.zeros(B), axis_trans=d, axis_rot=torch.randn(B, 3, generator=g),
        origin_pred=torch.randn(B, 3, generator=g), point_3d_pred=torch.randn(B, 3, generator=g),
        axis_gt=d_gt, origin_gt=torch.randn(B, 3, generator=g), traj_start_gt=torch.randn(B, 3, generator=g),
        axis_weight=2.0)
    expected = 2.0 * (1.0 - (d * d_gt).sum(-1))
    assert torch.allclose(total, expected, atol=1e-5)
    assert comps["phase"].abs().max() == 0 and comps["radius"].abs().max() == 0


def test_config_fields_default_off():
    import dataclasses
    from config.opd_train import LossParams
    lp = {f.name: f.default for f in dataclasses.fields(LossParams)}
    assert lp["closed_form_frame_weight"] == 0.0
    assert lp["closed_form_frame_axis"] == 2.0 and lp["closed_form_frame_phase"] == 1.0
    assert lp["closed_form_frame_radius"] == 0.15 and lp["closed_form_frame_radius_form"] == "log"
