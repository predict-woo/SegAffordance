"""FieldModel (model/field_model.py) + FieldTrainingModule (train_field_better.py)."""
import dataclasses
from unittest import mock

import torch

from config.opd_train import Config, LossParams, OptimizerParams
from model.field_model import FieldHead, FieldModel
from tests.test_g7_lift import _g7_batch
from tests.test_split_heads import _StubBackbone, _inputs, _params


def _field_model(**over):
    params = _params(use_depth=False, dense_hidden=32, trajectory_scale_free=True, **over)
    with mock.patch("model.segmenter.build_backbone", lambda mp, fpn_in: _StubBackbone(fpn_in, word_len=mp.word_len)), \
         mock.patch("model.field_model.build_backbone", lambda mp, fpn_in: _StubBackbone(fpn_in, word_len=mp.word_len)):
        return FieldModel(params)


def test_loss_param_default():
    assert {f.name: f.default for f in dataclasses.fields(LossParams)}["depth_field_weight"] == 0.0


def test_forward_outputs_all_fields_and_lifts():
    m = _field_model(); m.eval()
    img, depth, word, mask = _inputs(B=2, size=64)
    K = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    with torch.no_grad():
        out = m(img, depth, word, mask, None, None, None, K)
    assert out.mask_logits.shape[1] == 1 and out.point_logits.shape[1] == 1 and out.origin_logits.shape[1] == 1
    assert out.point_uv.shape == (2, 2) and out.origin_uv.shape == (2, 2)
    assert out.motion_pred_rot.shape == (2, 3) and out.motion_pred_trans.shape == (2, 3)
    assert out.motion_type_logits.shape == (2, 2) and out.motion_pred.shape == (2, 3)
    assert out.point_3d_pred.shape == (2, 3) and out.origin_pred.shape == (2, 3)
    assert out.trajectory_pred.shape == (2, 20, 3) and out.trajectory_length.shape == (2,)
    assert out.depth_field.shape[1] == 1 and out.origin_vote_uv.shape[1] == 2 and out.vote_weights.shape[1] == 1
    assert (out.point_3d_pred[:, 2] >= 0.1).all() and (out.origin_pred[:, 2] >= 0.1).all()
    assert out.motion_pred_rot.abs().max() < 1.0
    # initial votes = the part centroid, initial depth ~ 1.5 m
    assert torch.allclose(out.point_3d_pred[:, 2], torch.full((2,), 1.5), atol=1e-3)


def test_no_intrinsics_means_no_lifts_but_fields_still_there():
    m = _field_model(); m.eval()
    img, depth, word, mask = _inputs(B=2, size=64)
    with torch.no_grad():
        out = m(img, depth, word, mask, None, None, None, None)
    assert out.point_3d_pred is None and out.trajectory_pred is None and out.depth_field is not None


def test_trunk_detach_flag_isolates_the_fields_from_the_trunk():
    m = _field_model(); m.train()
    img, depth, word, mask = _inputs(B=2, size=64)
    K = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    trunk = next(m.neck.parameters())
    m.dense_trunk_detach = True
    out = m(img, depth, word, mask, None, None, None, K)
    # field-only objective (the point heatmap is a trunk output by design, so point_3d / trajectory are not used here)
    (out.motion_pred_rot.sum() + out.origin_uv.sum() + out.trajectory_length.sum() + out.depth_field.sum()).backward()
    assert m.dense_head.net[-1].weight.grad.abs().sum() > 0
    assert trunk.grad is None or trunk.grad.abs().sum() == 0
    m.zero_grad(set_to_none=True); m.dense_trunk_detach = False
    out = m(img, depth, word, mask, None, None, None, K)
    (out.motion_pred_rot.sum() + out.origin_uv.sum()).backward()
    assert trunk.grad is not None and trunk.grad.abs().sum() > 0


def test_empty_predicted_mask_does_not_amplify_gradients():
    """Run 1 diverged: with the predicted mask ~0 the weighted means divided by
    wsum ~ 1e-6. The uniform floor keeps wsum >= 1 and the field gradients of
    an empty-mask forward comparable to a full-mask forward."""
    torch.manual_seed(0)
    m = _field_model(); m.train()
    img, depth, word, mask = _inputs(B=2, size=64)
    K = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)

    def grad_norm(bias_shift):
        m.zero_grad(set_to_none=True)
        with torch.no_grad():
            m.proj.txt_fc[0].bias[-3] += bias_shift          # shift the mask channel's dynamic bias
        out = m(img, depth, word, mask, None, None, None, K)
        assert out.vote_weights.sum(dim=(-1, -2)).min() >= 1.0 - 1e-5
        (out.motion_pred_rot.sum() + out.origin_uv.sum()).backward()
        g = m.dense_head.net[0].weight.grad.norm().item()
        with torch.no_grad():
            m.proj.txt_fc[0].bias[-3] -= bias_shift
        assert torch.isfinite(out.origin_pred).all()
        return g

    g_empty, g_full = grad_norm(-40.0), grad_norm(+40.0)
    assert g_empty < 20.0 * max(g_full, 1e-8)


def test_writer_length_mode():
    m = _field_model(trajectory_decoder_length="writer"); m.eval()
    img, depth, word, mask = _inputs(B=2, size=64)
    K = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    with torch.no_grad():
        out = m(img, depth, word, mask, None, None, None, K)
    assert torch.isfinite(out.trajectory_pred).all()


def _field_module(depth_w):
    from train_field_better import FieldTrainingModule
    lp = LossParams(
        bce_weight=0.5, dice_weight=0.5, mask_weight=0.5, point_map_weight=0.5, coord_weight=0.5,
        vae_weight=0.5, motion_type_weight=0.5, point_sigma=8.0, vae_beta=0.01, trajectory_weight=0.0,
        geometric_loss="pred_pred_art", pred_pred_art_weight=0.0, axis_sign_agnostic=False,
        origin_weight=0.5, point_3d_weight=0.5, origin_map_weight=0.5, depth_field_weight=depth_w,
    )
    op = OptimizerParams(lr=1e-5, weight_decay=0.0, scheduler_milestones=[10], scheduler_gamma=0.1)
    cfg = Config(log_image_interval_steps=0, input_size=[64, 64], enable_wandb=False, val_vis_samples=0, manual_seed=0)
    with mock.patch("model.segmenter.build_backbone", lambda mp, fpn_in: _StubBackbone(fpn_in, word_len=mp.word_len)), \
         mock.patch("model.field_model.build_backbone", lambda mp, fpn_in: _StubBackbone(fpn_in, word_len=mp.word_len)):
        return FieldTrainingModule(
            _params(use_depth=False, use_trajectory_head=False, trajectory_decoder="analytic", split_axis_heads=True,
                    use_origin_heatmap=True, predict_point_depth=True, trajectory_scale_free=True, dense_hidden=32),
            lp, op, cfg,
        )


def test_depth_field_loss_pools_valid_cells():
    from train_field_better import FieldTrainingModule
    field = torch.zeros(1, 1, 2, 2)                       # log z = 0 -> z = 1
    d = torch.zeros(1, 1, 8, 8); d[:, :, :4, :4] = 2.0    # one valid cell at 2 m, three invalid (0)
    L, n = FieldTrainingModule.depth_field_loss(field, d)
    assert n == 1 and torch.isclose(L, torch.tensor(2.0).log())


def test_field_trainer_step_adds_and_logs_the_depth_term():
    m = _field_module(0.5)
    assert isinstance(m.model.core, FieldModel) and m.model.dense_head is m.model.core.dense_head
    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(name, float(value) if torch.is_tensor(value) else value)
    loss = m._common_step(_g7_batch(), 0, "train")
    assert torch.isfinite(loss)
    assert "train/L_depth_field" in logged and logged["train/L_depth_field"] > 0
    loss.backward()
    assert m.model.dense_head.net[-1].weight.grad[FieldHead.LOGZ].abs().sum() > 0
    assert m.model.last is None
    m.model.dense_trunk_detach = True
    assert m.model.core.dense_trunk_detach is True
    m.model.dense_trunk_detach = False
    m2 = _field_module(0.0)
    m2.log = lambda name, value, **kw: logged.__setitem__("off:" + name, value)
    assert torch.isfinite(m2._common_step(_g7_batch(), 0, "train")) and "off:train/L_depth_field" not in logged
