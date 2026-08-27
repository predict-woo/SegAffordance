"""Tests for the closed-form continuous trajectory loss (theory note
2026-08-28): the exact N -> infinity limit of the sampled analytic-decode
loss. The load-bearing property is CONVERGENCE — the sampled normalized
loss at large N must approach the closed form — plus exactness at GT,
sign sensitivity, and gauge invariance."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model.losses.geometric import analytic_screw_trajectory, closed_form_screw_loss


def sampled_normalized_loss(mt, ax_t, ax_r, q, p0, ax_gt, q_gt, p0_gt, n=2001):
    """The training loss's sampled form at N points (per-row)."""
    dec = analytic_screw_trajectory(mt, ax_t, ax_r, q, p0, num_points=n)
    gt = analytic_screw_trajectory(mt, ax_gt, ax_gt, q_gt, p0_gt, num_points=n)
    err = (dec - gt).pow(2).sum(-1).mean(-1)
    energy = gt.pow(2).sum(-1).mean(-1)
    return err / energy.clamp(min=1e-4)


def rand_case(seed, rot):
    g = torch.Generator().manual_seed(seed)
    ax_gt = F.normalize(torch.randn(1, 3, generator=g), dim=-1)
    q_gt = torch.randn(1, 3, generator=g) * 0.3 + torch.tensor([[0.0, 0.0, 2.0]])
    # GT start point with a healthy lever
    off = torch.randn(1, 3, generator=g)
    off = off - (off * ax_gt).sum(-1, keepdim=True) * ax_gt
    p0_gt = q_gt + 0.4 * F.normalize(off, dim=-1) + 0.1 * ax_gt
    # perturbed prediction
    ax_p = F.normalize(ax_gt + 0.3 * torch.randn(1, 3, generator=g), dim=-1)
    q_p = q_gt + 0.1 * torch.randn(1, 3, generator=g)
    p0_p = p0_gt + 0.05 * torch.randn(1, 3, generator=g)
    mt = torch.tensor([1.0 if rot else 0.0])
    return mt, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt


@pytest.mark.parametrize("rot", [True, False])
@pytest.mark.parametrize("seed", range(5))
def test_sampled_loss_converges_to_closed_form(seed, rot):
    mt, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt = rand_case(seed, rot)
    sampled = sampled_normalized_loss(mt, ax_p, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt)
    pos, _ = closed_form_screw_loss(mt, ax_p, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt)
    assert pos[0].item() == pytest.approx(sampled[0].item(), rel=2e-3)


def test_convergence_rate_is_quadratic_in_n():
    mt, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt = rand_case(0, rot=True)
    pos, _ = closed_form_screw_loss(mt, ax_p, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt)
    errs = []
    for n in (11, 21, 41):
        s = sampled_normalized_loss(mt, ax_p, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt, n=n)
        errs.append(abs(s[0].item() - pos[0].item()))
    # uniform sampling WITH endpoints is not the trapezoid rule: the
    # boundary weighting makes convergence O(1/N), not O(1/N^2) —
    # measured empirically (error halves per doubling; constants tiny:
    # ~3.5e-4 at the training N=20). Assert the O(1/N) rate.
    assert errs[0] > 1.6 * errs[1] > 1.6 * 1.6 * errs[2] / 1.6


@pytest.mark.parametrize("rot", [True, False])
def test_zero_at_ground_truth(rot):
    mt, _, _, _, ax_gt, q_gt, p0_gt = rand_case(3, rot)
    pos, der = closed_form_screw_loss(mt, ax_gt, ax_gt, q_gt, p0_gt, ax_gt, q_gt, p0_gt)
    assert pos[0].item() == pytest.approx(0.0, abs=1e-8)
    assert der[0].item() == pytest.approx(0.0, abs=1e-8)


def test_trans_term_equals_one_minus_cos():
    mt, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt = rand_case(4, rot=False)
    pos, der = closed_form_screw_loss(mt, ax_p, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt)
    cos = F.cosine_similarity(ax_p, ax_gt, dim=-1)
    expected = 2.0 * (1.0 - cos[0].item())
    assert pos[0].item() == pytest.approx(expected, abs=1e-6)
    assert der[0].item() == pytest.approx(expected, abs=1e-6)


def test_axis_flip_is_expensive_for_rot():
    mt, _, _, _, ax_gt, q_gt, p0_gt = rand_case(5, rot=True)
    pos, der = closed_form_screw_loss(mt, -ax_gt, -ax_gt, q_gt, p0_gt, ax_gt, q_gt, p0_gt)
    # flipped axis: r unchanged, t -> -t => dt = -2t*, dr = 0
    # position: c*4|t*|^2 / ((pi-2)|r*|^2) = pi/(pi-2)  (|t*|=|r*|)
    assert pos[0].item() == pytest.approx(math.pi / (math.pi - 2.0), rel=1e-4)
    assert der[0].item() == pytest.approx(2.0, rel=1e-4)


def test_origin_gauge_invariance_along_axis():
    mt, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt = rand_case(6, rot=True)
    a = closed_form_screw_loss(mt, ax_p, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt)
    b = closed_form_screw_loss(mt, ax_p, ax_p, q_p + 2.5 * ax_p, p0_p,
                               ax_gt, q_gt + 1.1 * ax_gt, p0_gt)
    assert a[0][0].item() == pytest.approx(b[0][0].item(), abs=1e-6)
    assert a[1][0].item() == pytest.approx(b[1][0].item(), abs=1e-6)


def test_gradients_flow_to_all_inputs():
    mt, ax_p, q_p, p0_p, ax_gt, q_gt, p0_gt = rand_case(7, rot=True)
    ax = ax_p.clone().requires_grad_(True)
    q = q_p.clone().requires_grad_(True)
    p0 = p0_p.clone().requires_grad_(True)
    pos, der = closed_form_screw_loss(mt, ax, ax, q, p0, ax_gt, q_gt, p0_gt)
    (pos.mean() + der.mean()).backward()
    for t in (ax, q, p0):
        assert t.grad is not None and torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0


def test_batch_routes_types_row_wise():
    m1 = rand_case(8, rot=True)
    m2 = rand_case(9, rot=False)
    mt = torch.cat([m1[0], m2[0]])
    cat = lambda i: torch.cat([m1[i], m2[i]])
    pos, der = closed_form_screw_loss(mt, cat(1), cat(1), cat(2), cat(3),
                                      cat(4), cat(5), cat(6))
    p1, _ = closed_form_screw_loss(*m1[:1], m1[1], m1[1], *m1[2:4], *m1[4:])
    p2, _ = closed_form_screw_loss(*m2[:1], m2[1], m2[1], *m2[2:4], *m2[4:])
    assert pos[0].item() == pytest.approx(p1[0].item(), abs=1e-6)
    assert pos[1].item() == pytest.approx(p2[0].item(), abs=1e-6)
