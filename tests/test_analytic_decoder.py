"""Analytic trajectory decoder (2026-09-11 joint design): render the trajectory
from the model's own type / axis / origin / point heads + a learned arc length."""
import math

import pytest
import torch

from model.losses.geometric import analytic_decode_curves, normalized_intrinsics, route_decoded_trajectory
from model.outputs import ModelOutputs
from test_split_heads import _inputs, _make_cris


def _case(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = torch.nn.functional.normalize(torch.randn(B, 3, generator=g), dim=1)
    d = torch.nn.functional.normalize(torch.randn(B, 3, generator=g), dim=1)
    o = torch.randn(B, 3, generator=g)
    p = o + torch.randn(B, 3, generator=g)
    L = torch.rand(B, generator=g) + 0.2
    return n, d, o, p, L


def test_trans_branch_is_a_straight_line_of_the_predicted_length():
    n, d, o, p, L = _case()
    _, trans, _ = analytic_decode_curves(n, d, o, p, L)
    assert trans.shape == (4, 20, 3)
    assert trans[:, 0].abs().max() < 1e-6
    assert torch.allclose(trans[:, -1], L[:, None] * d, atol=1e-5)
    seg = trans[:, 1:] - trans[:, :-1]
    assert torch.allclose(seg.norm(dim=-1).sum(-1), L, atol=1e-5)          # arc length = L


def test_rot_branch_is_an_arc_about_the_axis_line_with_angle_L_over_radius():
    n, d, o, p, L = _case(seed=1)
    rot, _, aux = analytic_decode_curves(n, d, o, p, L, max_angle=10.0)
    lever = (p - o) - ((p - o) * n).sum(-1, keepdim=True) * n
    radius = lever.norm(dim=-1)
    assert torch.allclose(aux["radius"], radius, atol=1e-6)
    assert torch.allclose(aux["theta_max"], L / radius, atol=1e-5)
    # every point stays at the same distance from the axis line and at the same height along it
    abs_pts = p[:, None, :] + rot
    rel = abs_pts - o[:, None, :]
    along = (rel * n[:, None, :]).sum(-1)
    perp = (rel - along[..., None] * n[:, None, :]).norm(dim=-1)
    assert torch.allclose(perp, radius[:, None].expand_as(perp), atol=1e-5)
    assert torch.allclose(along, along[:, :1].expand_as(along), atol=1e-5)
    # arc length of the polyline ~ L (20-point chord approximation)
    seg = (rot[:, 1:] - rot[:, :-1]).norm(dim=-1).sum(-1)
    assert torch.allclose(seg, L, rtol=0.02)


def test_angle_is_clamped_and_radius_is_floored():
    n, d, o, p, L = _case(seed=2)
    rot, _, aux = analytic_decode_curves(n, d, o, p, torch.full_like(L, 100.0), max_angle=math.pi)
    assert torch.allclose(aux["theta_max"], torch.full_like(L, math.pi))
    # collapsed lever: floored radius, finite curve
    rot2, _, aux2 = analytic_decode_curves(n, d, o, o + 1e-6 * (p - o), L, lever_floor=0.02)
    assert torch.allclose(aux2["radius"], torch.full_like(L, 0.02)) and torch.isfinite(rot2).all()


def test_decode_is_homogeneous_in_the_frame_scale():
    # scaling point, origin and length by the same factor scales both curves (scale-free frame is legal)
    n, d, o, p, L = _case(seed=3)
    rot1, trans1, _ = analytic_decode_curves(n, d, o, p, L)
    rot2, trans2, _ = analytic_decode_curves(n, d, 3.0 * o, 3.0 * p, 3.0 * L)
    assert torch.allclose(rot2, 3.0 * rot1, atol=1e-5) and torch.allclose(trans2, 3.0 * trans1, atol=1e-5)


def test_radius_is_detached_in_the_angle_but_not_in_the_arc():
    n, d, o, p, L = _case(seed=4)
    o = o.clone().requires_grad_(True)
    rot, _, aux = analytic_decode_curves(n, d, o, p, L, max_angle=10.0)
    assert not aux["theta_max"].requires_grad                  # angle path detached from the origin
    rot.pow(2).sum().backward()
    assert o.grad is not None and o.grad.abs().max() > 0       # arc shape still trains the origin


def test_route_by_gt_type():
    out = ModelOutputs(mask_logits=torch.zeros(3, 1, 4, 4), trajectory_pred=torch.zeros(3, 20, 3),
                       trajectory_pred_rot=torch.ones(3, 20, 3), trajectory_pred_trans=2 * torch.ones(3, 20, 3))
    out = route_decoded_trajectory(out, torch.tensor([1, 0, 1]))
    assert out.trajectory_pred[:, 0, 0].tolist() == [1.0, 2.0, 1.0]


def _decoder_model(**over):
    return _make_cris(use_trajectory_head=False, trajectory_decoder="analytic", split_axis_heads=True,
                      use_motion_type_head=True, predict_point_depth=True, use_origin_heatmap=True,
                      use_origin_local_feature=True, trajectory_scale_free=True, **over)


def test_model_renders_both_branches_and_selects_by_predicted_type():
    m = _decoder_model(); m.eval()
    img, depth, word, mask = _inputs(B=2, size=64)
    K = torch.tensor([[[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]]]).expand(2, 3, 3)
    K_norm = normalized_intrinsics(K, torch.tensor([[64.0, 64.0]]).expand(2, 2))
    with torch.no_grad():
        out = m(img, depth, word, mask, None, None, None, K_norm)
    assert out.trajectory_pred_rot.shape == (2, 20, 3) and out.trajectory_pred_trans.shape == (2, 20, 3)
    assert out.trajectory_length.shape == (2,) and (out.trajectory_length > 0).all()
    sel = out.motion_type_logits.argmax(-1)
    for i in range(2):
        want = out.trajectory_pred_rot[i] if sel[i] == 1 else out.trajectory_pred_trans[i]
        assert torch.allclose(out.trajectory_pred[i], want)
    assert out.trajectory_pred[:, 0].abs().max() < 1e-6
    assert "trajectory_predictor" not in dict(m.named_modules())


def test_model_without_intrinsics_skips_the_decoder():
    m = _decoder_model(); m.eval()
    img, depth, word, mask = _inputs(B=2, size=64)
    with torch.no_grad():
        out = m(img, depth, word, mask, None, None)
    assert out.trajectory_pred is None and out.trajectory_length is None


def test_decoder_rejects_a_learned_head():
    with pytest.raises(ValueError):
        _make_cris(use_trajectory_head=True, trajectory_decoder="analytic", split_axis_heads=True,
                   predict_point_depth=True, use_origin_heatmap=True)


def test_writer_length_mode_uses_the_constants():
    m = _decoder_model(trajectory_decoder_length="writer"); m.eval()
    img, depth, word, mask = _inputs(B=2, size=64)
    K = torch.tensor([[[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]]]).expand(2, 3, 3)
    K_norm = normalized_intrinsics(K, torch.tensor([[64.0, 64.0]]).expand(2, 2))
    with torch.no_grad():
        out = m(img, depth, word, mask, None, None, None, K_norm)
    sel = out.motion_type_logits.argmax(-1)
    for i in range(2):
        if sel[i] == 0:
            # trans length = 0.7 m / z_p in the scale-free frame: the metric line is 0.7 m long
            zp = out.point_3d_pred[i, 2]
            assert abs(float(out.trajectory_length[i] * zp) - 0.7) < 1e-4
        else:
            # rot: a quarter turn
            seg = (out.trajectory_pred_rot[i, 1:] - out.trajectory_pred_rot[i, :-1]).norm(dim=-1).sum()
            assert abs(float(out.trajectory_length[i]) - float(seg)) / float(seg) < 0.02
