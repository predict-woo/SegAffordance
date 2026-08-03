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
