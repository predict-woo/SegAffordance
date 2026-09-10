"""DCT readout conventions v2 (2026-09-10): pinned start, shape/scale split,
and the log path-length losses (3D and uv-space).

knowledge/2026-09-10_trajectory_head_synthesis_v2.md, section 3 (#2).
"""
import math

import torch

from model.layers import TrajectoryMLP


def _dct_matrix(N):
    n = torch.arange(N, dtype=torch.float64)
    k = torch.arange(N, dtype=torch.float64)
    m = torch.cos(math.pi * (n[None, :] + 0.5) * k[:, None] / N)
    m *= math.sqrt(2.0 / N)
    m[0] /= math.sqrt(2.0)
    return m.float()


def _path_length(pts):
    return (pts[..., 1:, :] - pts[..., :-1, :]).norm(dim=-1).sum(-1)


# ------------------------------------------------------------- head

def test_flags_off_is_the_legacy_dct_head():
    torch.manual_seed(0)
    head = TrajectoryMLP(input_dim=16, hidden_dim=8, dct_coeffs=6)
    assert not head.dct_pin_start and not head.dct_scale_split
    assert not hasattr(head, "scale_head")
    assert head.idct_m.shape == (20, 6)
    m = _dct_matrix(20)
    assert torch.allclose(head.idct_m, m[:6].T)         # rows 0..5, DC included


def test_pinned_start_is_exactly_zero_and_uses_ac_rows():
    torch.manual_seed(1)
    head = TrajectoryMLP(input_dim=16, hidden_dim=8, dct_coeffs=6, dct_pin_start=True)
    out = head(torch.randn(5, 16))
    assert out.shape == (5, 1, 20, 3)
    assert out[:, :, 0].abs().max() == 0.0
    m = _dct_matrix(20)
    assert torch.allclose(head.idct_m, m[1:7].T)        # rows 1..6, no DC
    # still a low-frequency curve: nothing above the 6th AC frequency
    # (the pin adds a constant, i.e. only DC energy)
    coeffs = torch.einsum("kn,bqnd->bqkd", m, out)
    assert coeffs[:, :, 7:].abs().max() < 1e-4


def test_pinned_head_has_no_dead_parameters():
    # Every trajectory_head output column must reach the decoded curve
    # (a DC coefficient under a pin would receive zero gradient).
    torch.manual_seed(2)
    head = TrajectoryMLP(input_dim=16, hidden_dim=8, dct_coeffs=6, dct_pin_start=True)
    out = head(torch.randn(4, 16))
    out.pow(2).sum().backward()
    g = head.trajectory_head.weight.grad
    assert g is not None and (g.abs().sum(dim=1) > 0).all()


def test_scale_split_output_is_unit_shape_times_softplus_scale():
    torch.manual_seed(3)
    head = TrajectoryMLP(input_dim=16, hidden_dim=8, dct_coeffs=6,
                         dct_pin_start=True, dct_scale_split=True)
    x = torch.randn(6, 16)
    out = head(x)
    h = head.backbone(x)
    scale = torch.nn.functional.softplus(head.scale_head(h)).view(-1)
    assert torch.allclose(_path_length(out)[:, 0], scale, atol=1e-5)
    assert out[:, :, 0].abs().max() == 0.0


def test_scale_split_init_is_a_typical_magnitude():
    head = TrajectoryMLP(input_dim=16, hidden_dim=8, dct_coeffs=6, dct_scale_split=True)
    out = head(torch.randn(8, 16))
    L = _path_length(out)[:, 0]
    # softplus(-0.7) ~ 0.403; zero weights -> identical for every input
    assert torch.allclose(L, torch.full_like(L, math.log1p(math.exp(-0.7))), atol=1e-5)


def test_scale_split_gradients_reach_both_branches():
    torch.manual_seed(4)
    head = TrajectoryMLP(input_dim=16, hidden_dim=8, dct_coeffs=6,
                         dct_pin_start=True, dct_scale_split=True)
    out = head(torch.randn(4, 16))
    (out - 0.3).pow(2).sum().backward()
    assert head.scale_head.weight.grad.abs().sum() > 0
    assert head.trajectory_head.weight.grad.abs().sum() > 0


def test_scale_split_handles_fp16_autocast_without_nan():
    if not torch.cuda.is_available():
        return
    head = TrajectoryMLP(input_dim=16, hidden_dim=8, dct_coeffs=6,
                         dct_pin_start=True, dct_scale_split=True).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        out = head(torch.randn(4, 16, device="cuda"))
    assert torch.isfinite(out).all()


def test_invalid_combinations_raise():
    import pytest
    with pytest.raises(ValueError):
        TrajectoryMLP(input_dim=16, hidden_dim=8, dct_pin_start=True)         # needs DCT
    with pytest.raises(ValueError):
        TrajectoryMLP(input_dim=16, hidden_dim=8, dct_scale_split=True)
    with pytest.raises(ValueError):
        TrajectoryMLP(input_dim=16, hidden_dim=8, num_points=6, dct_coeffs=6, dct_pin_start=True)
    with pytest.raises(ValueError):
        TrajectoryMLP(input_dim=16, hidden_dim=8, dct_coeffs=6, dct_pin_start=True, absolute=True)


def test_config_fields_default_off():
    import dataclasses
    from config.opd_train import LossParams, ModelParams
    mp = {f.name: f.default for f in dataclasses.fields(ModelParams)}
    assert mp["trajectory_dct_pin_start"] is False
    assert mp["trajectory_dct_scale_split"] is False
    lp = {f.name: f.default for f in dataclasses.fields(LossParams)}
    assert lp["trajectory_scale_log_weight"] == 0.0
    assert lp["proj_scale_log_weight"] == 0.0


# ------------------------------------------------------------- uv-space scale loss

def _proj_setup():
    from model.losses.geometric import TrajectoryProjectionLoss, backproject_points, normalized_intrinsics, project_points
    from model.targets import StepTargets
    from types import SimpleNamespace
    B, N = 3, 20
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]).expand(B, 3, 3)
    img = torch.tensor([640.0, 480.0]).expand(B, 2)
    K_norm = normalized_intrinsics(K, img)
    uv0 = torch.tensor([[0.5, 0.5], [0.4, 0.6], [0.6, 0.4]])
    anchor = backproject_points(K_norm, uv0, torch.ones(B))
    t = torch.linspace(0, 1, N)
    rel_gt = torch.stack([0.3 * t, 0.1 * t, 0.0 * t], -1).expand(B, N, 3)
    track = project_points(K_norm, anchor[:, None] + rel_gt)
    targets = StepTargets(trajectory_2d=track, trajectory_2d_valid=torch.ones(B, N, dtype=torch.bool),
                          camera_intrinsic=K, img_size=img)
    return TrajectoryProjectionLoss, SimpleNamespace, rel_gt, targets


def test_proj_scale_log_term_is_zero_for_exact_and_log_ratio_for_scaled():
    TPL, NS, rel_gt, targets = _proj_setup()
    loss = TPL(weight=0.0, anchor_source="unit", scale_log_weight=1.0)
    exact = NS(trajectory_pred=rel_gt.clone(), mask_logits=torch.zeros(3, 1, 4, 4), point_uv=None)
    total, terms = loss(exact, targets, None)
    assert terms["L_proj_scale"].abs() < 1e-4
    doubled = NS(trajectory_pred=2.0 * rel_gt, mask_logits=torch.zeros(3, 1, 4, 4), point_uv=None)
    total2, terms2 = loss(doubled, targets, None)
    # projected length roughly doubles for a small in-plane motion at depth 1
    assert 0.5 < terms2["L_proj_scale"].item() < 0.8
    assert total2.item() == terms2["L_proj_scale"].item()


def test_proj_scale_log_term_needs_no_weight_on_the_main_term():
    TPL, NS, rel_gt, targets = _proj_setup()
    loss = TPL(weight=0.0, anchor_source="unit")
    out = NS(trajectory_pred=rel_gt.clone(), mask_logits=torch.zeros(3, 1, 4, 4), point_uv=None)
    total, terms = loss(out, targets, None)
    assert total.item() == 0.0 and terms == {}
