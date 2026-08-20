"""Gen-19 smooth-trajectory arms (2026-08-21 spec).

Arm 1: truncated-DCT trajectory readout (TrajectoryMLP.dct_coeffs).
Arm 2: first-difference losses (velocity / angle / length).
"""
import os

import pytest
import torch
import torch.nn.functional as F
import yaml

from model.layers import TrajectoryMLP


# ------------------------------------------------------------- DCT head

def _dct_matrix(N):
    import math
    n = torch.arange(N, dtype=torch.float64)
    k = torch.arange(N, dtype=torch.float64)
    m = torch.cos(math.pi * (n[None, :] + 0.5) * k[:, None] / N)
    m *= math.sqrt(2.0 / N)
    m[0] /= math.sqrt(2.0)
    return m.float()


def test_dct_matrix_orthonormal():
    m = _dct_matrix(20)
    assert torch.allclose(m @ m.T, torch.eye(20), atol=1e-5)


def test_flag_off_is_legacy():
    torch.manual_seed(0)
    head = TrajectoryMLP(input_dim=16, hidden_dim=8)
    out = head(torch.randn(4, 16))
    assert out.shape == (4, 1, 20, 3)
    assert head.dct_coeffs == 0 and not hasattr(head, "idct_m")


def test_dct_output_shape_and_span():
    torch.manual_seed(1)
    head = TrajectoryMLP(input_dim=16, hidden_dim=8, dct_coeffs=6)
    out = head(torch.randn(4, 16))
    assert out.shape == (4, 1, 20, 3)
    # Output lies in the 6-dim low-frequency span: re-encoding with the
    # full DCT must show zero energy above coefficient 6.
    m = _dct_matrix(20)
    coeffs = torch.einsum("kn,bqnd->bqkd", m, out)
    assert coeffs[:, :, 6:].abs().max() < 1e-4


def test_dct_is_smoother_than_direct_by_construction():
    # Same random weights scale: decoded second differences must be far
    # smaller than a direct random readout's.
    torch.manual_seed(2)
    direct = TrajectoryMLP(input_dim=16, hidden_dim=8)
    dct = TrajectoryMLP(input_dim=16, hidden_dim=8, dct_coeffs=6)
    x = torch.randn(64, 16)
    def rough(t):
        return (t[:, :, 2:] - 2 * t[:, :, 1:-1] + t[:, :, :-2]).norm(dim=-1).mean()
    assert rough(dct(x)) < 0.5 * rough(direct(x))


def test_dct_hypotheses_shape():
    head = TrajectoryMLP(input_dim=16, hidden_dim=8, dct_coeffs=6,
                         num_hypotheses=3)
    assert head(torch.randn(2, 16)).shape == (2, 3, 20, 3)


def test_dct_vs_cumsum_raises():
    with pytest.raises(ValueError):
        TrajectoryMLP(input_dim=16, dct_coeffs=6, delta_cumsum=True)


def test_dct_too_many_coeffs_raises():
    with pytest.raises(ValueError):
        TrajectoryMLP(input_dim=16, num_points=20, dct_coeffs=21)


# ------------------------------------------------ first-difference losses

def _fdiff_terms(pred, gt):
    dp, dg = torch.diff(pred, dim=1), torch.diff(gt, dim=1)
    vel = (dp - dg).norm(dim=-1).mean()
    ok = dg.norm(dim=-1) > 1e-3
    ang = (1.0 - F.cosine_similarity(dp, dg, dim=-1, eps=1e-6)[ok]).mean()
    length = (dp.norm(dim=-1) - dg.norm(dim=-1)).abs().mean()
    return vel, ang, length


def test_velocity_zero_iff_equal_diffs():
    gt = torch.randn(2, 20, 3)
    pred = gt + torch.tensor([1.0, -2.0, 3.0])  # constant offset
    vel, ang, length = _fdiff_terms(pred, gt)
    assert vel < 1e-6 and ang < 1e-6 and length < 1e-6  # diff kills offsets


def test_angle_invariant_to_speed():
    gt = torch.cumsum(torch.randn(2, 20, 3).abs(), dim=1)
    pred = gt[:, 0:1] + (gt - gt[:, 0:1]) * 3.0  # same directions, 3x speed
    _, ang, _ = _fdiff_terms(pred, gt)
    assert ang < 1e-5


def test_length_invariant_to_direction():
    g = torch.zeros(1, 20, 3); g[0, :, 0] = torch.arange(20.0)
    p = torch.zeros(1, 20, 3); p[0, :, 1] = torch.arange(20.0)  # same speed, ⊥ dir
    _, _, length = _fdiff_terms(p, g)
    assert length < 1e-6


# ------------------------------------------------------------- configs

def _load_cfg(name):
    with open(os.path.join(os.path.dirname(__file__), "..", "config", name)) as f:
        return yaml.safe_load(f)


def test_g19_configs_match_spec():
    g17 = _load_cfg("sf3d_train_runpod_g17_splitax.yaml")
    dct = _load_cfg("sf3d_train_runpod_g19_dct.yaml")
    mp = dict(dct["model"]["model_params"])
    assert mp.pop("trajectory_dct_coeffs") == 6
    assert mp == g17["model"]["model_params"]
    assert dct["model"]["loss_params"] == g17["model"]["loss_params"]
    assert dct["data"] == g17["data"]
    assert "20260821_sf3d_g19_dct" in dct["trainer"]["callbacks"][0]["init_args"]["dirpath"]

    fd = _load_cfg("sf3d_train_runpod_g19_fdiff.yaml")
    assert fd["model"]["model_params"] == g17["model"]["model_params"]
    lp = dict(fd["model"]["loss_params"])
    assert lp.pop("trajectory_velocity_weight") == 1.0
    assert lp.pop("trajectory_angle_weight") == 0.5
    assert lp.pop("trajectory_length_weight") == 0.5
    assert lp == g17["model"]["loss_params"]
    assert fd["data"] == g17["data"]
    assert "20260821_sf3d_g19_fdiff" in fd["trainer"]["callbacks"][0]["init_args"]["dirpath"]


# --------------------------------------------------- trainer integration

def test_training_step_both_arms():
    from tests.test_origin_local_sample import _build_module, _training_step
    torch.manual_seed(3)
    m = _build_module(split_axis_heads=True, trajectory_dct_coeffs=6)
    loss = _training_step(m)
    assert torch.isfinite(loss)

    torch.manual_seed(3)
    m2 = _build_module(split_axis_heads=True)
    m2.loss_params.trajectory_velocity_weight = 1.0
    m2.loss_params.trajectory_angle_weight = 0.5
    m2.loss_params.trajectory_length_weight = 0.5
    loss2 = _training_step(m2)
    assert torch.isfinite(loss2)

    # weights-off byte-identity of the total loss
    torch.manual_seed(3)
    m3 = _build_module(split_axis_heads=True)
    loss3 = _training_step(m3)
    torch.manual_seed(3)
    m4 = _build_module(split_axis_heads=True)
    m4.loss_params.trajectory_velocity_weight = 0.0
    loss4 = _training_step(m4)
    assert torch.equal(loss3, loss4)
