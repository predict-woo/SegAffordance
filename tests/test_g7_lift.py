import pytest
import torch
import torch.nn.functional as F

from model.losses.split import perpendicular_foot, project_q_star
from model.outputs import ModelOutputs

# Reuses test_split_heads' stub-backbone CRIS builder (_make_cris patches
# model.segmenter.build_backbone with _StubBackbone) plus its param/input
# helpers — the same fixture pattern its split_model fixture uses.
from tests.test_split_heads import _inputs, _make_cris, _params  # noqa: E402


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


# ---- segmenter: origin channel, depth heads, lifts (Task 2) ----------------


def _g7_model():
    # fpn_out[1] = 512 mirrors production: the z_p head's grid_sample local
    # feature is a slice of the decoded fq, so this pins the 512-dim
    # asymmetry contract literally (test_g7_depth_head_inputs_asymmetric).
    return _make_cris(use_origin_heatmap=True, predict_point_depth=True,
                      trajectory_absolute=True, trajectory_delta_cumsum=False,
                      fpn_out=[32, 512, 128])


@pytest.fixture(scope="module")
def g7_model():
    m = _g7_model()
    m.eval()
    return m


def test_g7_forward_shapes_with_and_without_intrinsics(g7_model):
    img, depth, word, mask = _inputs()
    K = _knorm(2)
    with torch.no_grad():
        out = g7_model(img, depth, word, mask, None, None, None, K)
        out_nok = g7_model(img, depth, word, mask, None, None, None, None)
    assert out.point_uv.shape == (2, 2) and out.origin_uv.shape == (2, 2)
    assert out.point_3d_pred.shape == (2, 3) and out.origin_pred.shape == (2, 3)
    assert out.trajectory_pred.shape == (2, 20, 3)
    assert out.mask_logits.shape[1] == 1 and out.point_logits is not None
    # Lift only exists where K exists.
    assert out_nok.point_3d_pred is None and out_nok.origin_pred is None
    assert out_nok.origin_uv is not None


def test_g7_lift_roundtrip(g7_model):
    # project(point_3d_pred) must land exactly on point_uv (by construction).
    img, depth, word, mask = _inputs()
    K = _knorm(2)
    with torch.no_grad():
        out = g7_model(img, depth, word, mask, None, None, None, K)
    from model.losses.geometric import project_points
    assert torch.allclose(
        project_points(K, out.point_3d_pred.unsqueeze(1)).squeeze(1),
        out.point_uv, atol=1e-4)
    assert (out.point_3d_pred[:, 2] > 0).all()   # softplus depth


def test_g7_depth_head_inputs_asymmetric(g7_model):
    d = g7_model.point_depth_head.mlp[0].in_features \
        - g7_model.origin_depth_head_g7.mlp[0].in_features
    assert d == 512    # z_p gets the 512-dim local sample, z_q does not


def test_classical_and_gen6_modes_unchanged():
    m2d = _make_cris()
    m3d = _make_cris(point_prediction_3d=True, use_origin_head=True)
    img, depth, word, mask = _inputs()
    with torch.no_grad():
        o2 = m2d.eval()(img, depth, word, mask, None, None)
        o3 = m3d.eval()(img, depth, word, mask, None, None)
    assert o2.origin_uv is None and o2.point_3d_pred is None
    assert o3.origin_uv is None and o3.point_3d_pred.shape == (2, 3)


def test_absolute_trajectory_head_no_zero_pin():
    from model.layers import TrajectoryMLP
    head = TrajectoryMLP(input_dim=8, hidden_dim=16, num_points=20,
                         delta_cumsum=False, absolute=True)
    out = head(torch.randn(3, 8))
    assert out.shape == (3, 1, 20, 3)
    assert not torch.allclose(out[:, :, 0], torch.zeros(3, 1, 3))
    with pytest.raises(AssertionError):
        TrajectoryMLP(input_dim=8, delta_cumsum=True, absolute=True)


# ---- PredPredArticulationLoss absolute-trajectory mode (Task 3) -------------

from model.losses.geometric import PredPredArticulationLoss
from model.targets import StepTargets
from tests.test_split_heads import _circle_traj, _pp_outputs  # noqa: E402


def test_pp_art_absolute_perfect_revolute_scores_zero_and_gauge_along_axis():
    # Same perfect circle as the relative-mode test, but the trajectory is
    # ABSOLUTE: every point offset by the (arbitrary) first-point position.
    # point_3d_pred is None — absolute mode must not read it.
    dhat = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=1)
    first = torch.tensor([[0.5, 0.2, 1.0]])           # absolute first point
    origin = torch.tensor([[0.1, 0.2, 1.0]])          # axis point near it
    center_rel = (origin - first)[0]
    rel = _circle_traj(center_rel, dhat[0], torch.zeros(3),
                       torch.linspace(0, 1.2, 20))[None]
    traj = rel + first[:, None, :]                    # absolute curve
    out = _pp_outputs(traj, dhat, p_rev_logit=20.0, origin=origin, anchor=None)
    loss_fn = PredPredArticulationLoss(weight=0.5, trajectory_is_absolute=True)
    _, terms = loss_fn(out, StepTargets())
    assert terms["L_geo_pred_pred_art"].item() < 1e-6
    # Sliding the predicted origin ALONG the predicted axis changes nothing.
    out2 = _pp_outputs(traj, dhat, 20.0, origin + 2.5 * dhat, anchor=None)
    _, terms2 = loss_fn(out2, StepTargets())
    assert abs(terms2["L_geo_pred_pred_art"].item()
               - terms["L_geo_pred_pred_art"].item()) < 1e-6


def test_pp_art_absolute_soft_gate_selects_branch():
    # Straight absolute line: near-zero under P(rev)=0, positive under
    # P(rev)=1 — identical to the relative test up to a constant offset.
    dhat = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=1)
    t = torch.linspace(0, 0.4, 20)[None, :, None]
    offset = torch.tensor([[0.3, -0.7, 1.5]])         # arbitrary absolute start
    traj = dhat[:, None, :] * t + offset[:, None, :]
    origin = offset + torch.tensor([[0.0, 0.1, 0.0]])
    loss_fn = PredPredArticulationLoss(weight=1.0, trajectory_is_absolute=True)
    _, pris = loss_fn(_pp_outputs(traj, dhat, -20.0, origin, anchor=None),
                      StepTargets())
    _, rev = loss_fn(_pp_outputs(traj, dhat, 20.0, origin, anchor=None),
                     StepTargets())
    assert pris["L_geo_pred_pred_art"].item() < 1e-8
    assert rev["L_geo_pred_pred_art"].item() > 1e-3


# ---- trainer wiring (Task 4) -------------------------------------------------

from unittest import mock

from config.opd_train import Config, LossParams, OptimizerParams
from tests.test_split_heads import _sf3d_batch, _StubBackbone  # noqa: E402
from train_SF3D_better import SF3DTrainingModule


def _g7_module():
    # Mirrors test_split_heads._split_module, but with the gen-7 flags:
    # classical 2D point path (point_prediction_3d/use_origin_head off) plus
    # origin heatmap, point depth, and an absolute trajectory head.
    lp = LossParams(
        bce_weight=0.5, dice_weight=0.5, mask_weight=0.5,
        point_map_weight=0.5, coord_weight=0.5, vae_weight=0.5,
        motion_type_weight=0.5, point_sigma=8.0, vae_beta=0.01,
        trajectory_weight=0.5, geometric_loss="pred_pred_art",
        pred_pred_art_weight=0.5, axis_sign_agnostic=False,
        origin_weight=0.5, point_3d_weight=0.5, origin_map_weight=0.5,
    )
    op = OptimizerParams(lr=1e-5, weight_decay=0.0,
                         scheduler_milestones=[10], scheduler_gamma=0.1)
    cfg = Config(log_image_interval_steps=0, input_size=[64, 64],
                 enable_wandb=False, val_vis_samples=0, manual_seed=0)
    with mock.patch(
        "model.segmenter.build_backbone",
        lambda mp, fpn_in: _StubBackbone(fpn_in, word_len=mp.word_len),
    ):
        return SF3DTrainingModule(
            _params(use_origin_heatmap=True, predict_point_depth=True,
                    trajectory_absolute=True, trajectory_delta_cumsum=False,
                    pool_with_predicted_mask=True),
            lp, op, cfg,
        )


def _g7_batch():
    # _sf3d_batch's K is eye(3) at "original" 64x64 resolution, so the
    # normalized fx/fy become 1/64 with cx = cy = 0 — a random q* then
    # projects to u/v < 0 half the time (out of frame -> valid_q False).
    # Pin the revolute row's axis to pass through [0.5, 0.5, 1.5] along +z:
    # q* = [0.5, 0.5, p_z], whose projection lands inside [0, 1)^2 under
    # that K, guaranteeing valid_q (per the brief: adjust the batch helper
    # usage, never the production code).
    b = list(_sf3d_batch())
    b[6] = b[6].clone()
    b[6][0] = torch.tensor([0.0, 0.0, 1.0])       # motion axis, revolute row
    b[10] = b[10].clone()
    b[10][0] = torch.tensor([0.5, 0.5, 1.5])      # motion_origin_3d
    return tuple(b)


def test_g7_test_step_lifted_metrics_and_point_traj0_gap():
    # test_step must pass the batch intrinsics into the forward (K_norm, as
    # _common_step does) so the lifted point_3d_pred/origin_pred exist and the
    # gen-6 metric blocks run unchanged; the new gen-7 self-consistency gap
    # ||p_hat - traj_pred[0]|| accumulates for every sample (absolute
    # trajectory only).
    import math

    m = _g7_module()
    m.eval()
    m.log = lambda *a, **k: None
    m.on_test_start()
    with torch.no_grad():
        m.test_step(_g7_batch(), 0)
    assert len(m._test_point_traj0_gap) == 2
    assert all(math.isfinite(v) for v in m._test_point_traj0_gap)
    # The lifted fields reached the gen-6 metric blocks.
    assert len(m._test_point3d_errors) == 2
    assert len(m._test_origin_err_m) == 1          # revolute row only
    assert all(math.isfinite(v) for v in m._test_point3d_errors)


def test_g7_common_step_finite_and_logs_lift_terms():
    m = _g7_module()
    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(
        name, float(value) if torch.is_tensor(value) else value
    )
    loss = m._common_step(_g7_batch(), 0, "train")
    assert torch.isfinite(loss)
    loss.backward()
    for key in ("train/L_origin_map", "train/L_point_3d", "train/L_origin",
                "train/L_trajectory", "train/L_point_map", "train/L_coord"):
        assert key in logged, key
    # The lifted 3D point supervises against traj[:, 0], and the revolute
    # row's in-frame q* activates the origin heatmap BCE.
    assert logged["train/L_origin_map"] > 0.0
    assert logged["train/L_point_3d"] > 0.0
