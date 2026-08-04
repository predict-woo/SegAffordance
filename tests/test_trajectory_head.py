"""Delta-cumsum trajectory heads: structure the direct readout lacked."""

import torch

from model.layers import Trajectory2DMLP, TrajectoryMLP


def test_delta_cumsum_first_point_exactly_zero():
    for cls, dim in ((TrajectoryMLP, 3), (Trajectory2DMLP, 2)):
        head = cls(input_dim=32, hidden_dim=16, num_points=20, delta_cumsum=True)
        out = head(torch.randn(4, 32))
        assert out.shape == (4, 20, dim)
        assert torch.all(out[:, 0] == 0)


def test_delta_cumsum_is_integrated_path():
    head = TrajectoryMLP(input_dim=32, hidden_dim=16, num_points=20, delta_cumsum=True)
    out = head(torch.randn(2, 32))
    # consecutive differences reproduce the deltas — i.e. the output is a
    # connected path, not independent readouts
    diffs = out[:, 1:] - out[:, :-1]
    assert torch.allclose(torch.cumsum(diffs, dim=1), out[:, 1:], atol=1e-6)


def test_position_loss_reaches_every_delta():
    # An error at the LAST point must produce gradient on the weights that
    # generate every step before it — the coupling that kills zigzag.
    head = TrajectoryMLP(input_dim=8, hidden_dim=8, num_points=5, delta_cumsum=True)
    out = head(torch.randn(1, 8))
    out[0, -1].pow(2).sum().backward()
    g = head.trajectory_head.weight.grad
    per_step = g.view(4, 3, -1).abs().sum(dim=(1, 2))
    assert torch.all(per_step > 0)


def test_direct_readout_unchanged():
    head = TrajectoryMLP(input_dim=32, hidden_dim=16, num_points=20)
    assert head(torch.randn(4, 32)).shape == (4, 20, 3)


def test_motion_mlp_type_head_optional():
    from model.layers import MotionMLP

    with_head = MotionMLP(input_dim=16, hidden_dim=8, with_type_head=True)
    m, t = with_head(torch.randn(3, 16))
    assert t is not None and t.shape == (3, 2)

    without = MotionMLP(input_dim=16, hidden_dim=8, with_type_head=False)
    m, t = without(torch.randn(3, 16))
    assert t is None
    # truly parameter-free: nothing type-related in the state dict
    assert not any("type_head" in k for k in without.state_dict())


def test_motion_mlp_motion_head_optional():
    from model.layers import MotionMLP

    without = MotionMLP(input_dim=16, hidden_dim=8, with_motion_head=False)
    m, t = without(torch.randn(3, 16))
    assert m is None and t is not None
    assert not any("motion_head" in k for k in without.state_dict())


def test_twist_mlp_pitch_free_projection():
    from model.layers import TwistMLP

    head = TwistMLP(input_dim=16, hidden_dim=8, pitch_free=True)
    x = torch.randn(64, 16)
    out = head(x)
    omega, v = out[..., :3], out[..., 3:]
    dot = (omega * v).sum(-1).abs()
    norm = omega.norm(dim=-1)
    # revolute regime: essentially exact orthogonality (pitch-free)
    strong = norm > 0.5
    if strong.any():
        cos_axial = dot[strong] / (norm[strong] * v[strong].norm(dim=-1) + 1e-9)
        assert cos_axial.max().item() < 0.02
    # gradients flow through the projection
    out.sum().backward()
    assert torch.isfinite(head.twist_head.weight.grad).all()


def test_twist_mlp_pitch_free_keeps_prismatic_v():
    from model.layers import TwistMLP

    head = TwistMLP(input_dim=4, hidden_dim=4, pitch_free=True, pitch_eps=0.05)
    # force omega ~ 0 by zeroing the omega rows of the final linear layer
    with torch.no_grad():
        head.twist_head.weight[:3].zero_()
        head.twist_head.bias[:3].zero_()
        head.twist_head.bias[3:] = torch.tensor([0.0, 1.0, 0.0])
        head.twist_head.weight[3:].zero_()
    out = head(torch.randn(2, 4))
    # omega = 0 -> the projection must be the identity: v survives intact
    assert torch.allclose(out[..., 3:], torch.tensor([0.0, 1.0, 0.0]).expand(2, 3), atol=1e-6)
    assert torch.allclose(out[..., :3], torch.zeros(2, 3), atol=1e-6)
