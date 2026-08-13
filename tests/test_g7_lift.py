import torch
import torch.nn.functional as F

from model.losses.split import perpendicular_foot, project_q_star
from model.outputs import ModelOutputs


def _knorm(B):
    K = torch.eye(3).repeat(B, 1, 1)
    K[:, 0, 0] = K[:, 1, 1] = 1.2   # fx, fy (normalized units)
    K[:, 0, 2] = K[:, 1, 2] = 0.5   # cx, cy
    return K


def test_project_q_star_matches_foot_and_masks_offscreen():
    torch.manual_seed(0)
    B = 16
    o = torch.randn(B, 3) * 0.3 + torch.tensor([0.0, 0.0, 2.0])
    d = F.normalize(torch.randn(B, 3), dim=1)
    p = torch.randn(B, 3) * 0.3 + torch.tensor([0.0, 0.0, 2.0])
    K = _knorm(B)
    uv, valid = project_q_star(o, d, p, K)
    q_star = perpendicular_foot(o, d, p)
    # Where valid, uv is the pinhole projection of q_star.
    expect = torch.stack(
        [K[:, 0, 0] * q_star[:, 0] / q_star[:, 2] + K[:, 0, 2],
         K[:, 1, 1] * q_star[:, 1] / q_star[:, 2] + K[:, 1, 2]], dim=1)
    assert torch.allclose(uv[valid], expect[valid], atol=1e-5)
    assert valid.dtype == torch.bool
    # Behind-camera q* is invalid, never NaN.
    o2 = o.clone(); o2[:, 2] = -3.0; p2 = p.clone(); p2[:, 2] = -3.0
    uv2, valid2 = project_q_star(o2, d, p2, K)
    assert not valid2.any() and torch.isfinite(uv2).all()


def test_project_q_star_gauge_invariant():
    torch.manual_seed(1)
    o = torch.randn(4, 3) + torch.tensor([0.0, 0.0, 2.0])
    d = F.normalize(torch.randn(4, 3), dim=1)
    p = torch.randn(4, 3) + torch.tensor([0.0, 0.0, 2.0])
    K = _knorm(4)
    uv1, v1 = project_q_star(o, d, p, K)
    uv2, v2 = project_q_star(o + 2.9 * d, d, p, K)
    assert torch.allclose(uv1, uv2, atol=1e-5) and torch.equal(v1, v2)


def test_new_fields_and_flags_default_off():
    from config.opd_train import LossParams, ModelParams
    assert ModelOutputs(mask_logits=torch.zeros(1, 1, 4, 4)).origin_uv is None
    import dataclasses
    mf = {f.name: f.default for f in dataclasses.fields(ModelParams)}
    assert mf["use_origin_heatmap"] is False
    assert mf["predict_point_depth"] is False
    assert mf["trajectory_absolute"] is False
    lf = {f.name: f.default for f in dataclasses.fields(LossParams)}
    assert lf["origin_map_weight"] == 0.5
