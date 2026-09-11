"""Dense hinge voting: the per-pixel offset loss (loss_params.dense_offset_weight)."""
import dataclasses
from unittest import mock

import torch

from config.opd_train import Config, LossParams, OptimizerParams
from tests.test_g7_lift import _g7_batch
from tests.test_split_heads import _StubBackbone, _params
from train_SF3D_better import SF3DTrainingModule


def _dense_module(weight):
    lp = LossParams(
        bce_weight=0.5, dice_weight=0.5, mask_weight=0.5,
        point_map_weight=0.5, coord_weight=0.5, vae_weight=0.5,
        motion_type_weight=0.5, point_sigma=8.0, vae_beta=0.01,
        trajectory_weight=0.5, geometric_loss="pred_pred_art",
        pred_pred_art_weight=0.5, axis_sign_agnostic=False,
        origin_weight=0.5, point_3d_weight=0.5, origin_map_weight=0.5,
        dense_offset_weight=weight,
    )
    op = OptimizerParams(lr=1e-5, weight_decay=0.0, scheduler_milestones=[10], scheduler_gamma=0.1)
    cfg = Config(log_image_interval_steps=0, input_size=[64, 64], enable_wandb=False, val_vis_samples=0, manual_seed=0)
    with mock.patch("model.segmenter.build_backbone",
                    lambda mp, fpn_in: _StubBackbone(fpn_in, word_len=mp.word_len)):
        return SF3DTrainingModule(
            _params(use_origin_heatmap=True, predict_point_depth=True, trajectory_absolute=True,
                    trajectory_delta_cumsum=False, pool_with_predicted_mask=True,
                    split_axis_heads=True, articulation_readout="dense", dense_hidden=32),
            lp, op, cfg,
        )


def test_default_is_off():
    assert {f.name: f.default for f in dataclasses.fields(LossParams)}["dense_offset_weight"] == 0.0


def test_dense_offset_loss_is_logged_finite_and_trains_the_votes():
    m = _dense_module(0.5)
    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(name, float(value) if torch.is_tensor(value) else value)
    loss = m._common_step(_g7_batch(), 0, "train")
    assert torch.isfinite(loss)
    assert "train/L_dense_offset" in logged and logged["train/L_dense_offset"] > 0.0
    loss.backward()
    last = m.model.dense_head.net[-1]
    assert last.weight.grad[8:].abs().sum() > 0          # the offset channels got gradient


def test_dense_offset_loss_absent_when_off():
    m = _dense_module(0.0)
    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(name, value)
    loss = m._common_step(_g7_batch(), 0, "train")
    assert torch.isfinite(loss) and "train/L_dense_offset" not in logged
