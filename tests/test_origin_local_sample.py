"""Gen-11: flag-gated grid-sampled local feature for the origin depth head.

`use_origin_local_feature` gives z_q (origin depth) the same fpn_out[1]-dim
grid_sample of the decoded feature map that z_p (point depth) already
consumes — sampled at origin_uv instead of point_uv. Default-off keeps the
gen-7 condition-only z_q bit-identical.
"""

import os
from unittest import mock

import pytest
import torch
import torch.nn as nn
import yaml

from config.opd_train import Config, LossParams, OptimizerParams
from tests.test_g7_lift import _g7_batch, _knorm  # noqa: E402
from tests.test_split_heads import _StubBackbone, _inputs, _params  # noqa: E402
from train_SF3D_better import SF3DTrainingModule


def _build_module(**flag_overrides):
    # Mirrors test_g7_lift._g7_module (stub backbone, 64x64 inputs, cvae and
    # twist off, origin heatmap + point depth on) with flag overrides on top.
    flags = dict(
        use_origin_heatmap=True, predict_point_depth=True,
        trajectory_absolute=True, trajectory_delta_cumsum=False,
        pool_with_predicted_mask=True,
    )
    flags.update(flag_overrides)
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
        return SF3DTrainingModule(_params(**flags), lp, op, cfg)


def _first_linear_in_features(head):
    for m in head.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    raise AssertionError("no nn.Linear found in head")


def _forward_with_K(module):
    img, depth, word, mask = _inputs()
    module.eval()
    with torch.no_grad():
        out = module.model(img, depth, word, mask, None, None, None, _knorm(2))
    module.train()
    return out


def _training_step(module):
    module.log = lambda *a, **k: None
    return module._common_step(_g7_batch(), 0, "train")


def test_flag_requires_origin_heatmap():
    # use_origin_local_feature without use_origin_heatmap must raise.
    with pytest.raises(ValueError):
        _build_module(use_origin_heatmap=False, use_origin_local_feature=True)


def test_origin_head_input_dim_grows():
    m_off = _build_module(use_origin_local_feature=False)
    m_on = _build_module(use_origin_local_feature=True)
    d_off = _first_linear_in_features(m_off.model.origin_depth_head_g7)
    d_on = _first_linear_in_features(m_on.model.origin_depth_head_g7)
    assert d_on == d_off + m_on.model_params.fpn_out[1]


def test_forward_with_flag_on_lifts_origin():
    module = _build_module(use_origin_local_feature=True)
    outputs = _forward_with_K(module)
    assert outputs.origin_pred is not None
    assert torch.isfinite(outputs.origin_pred).all()
    loss = _training_step(module)
    assert torch.isfinite(loss)


def test_flag_off_path_unchanged():
    # Default-off: the z_q input dim must equal the POINT depth head's
    # input dim minus fpn_out[1] — z_p is [condition, local], gen-10's z_q
    # is [condition] — an invariant of the pre-change architecture that
    # doesn't reference this change's own code. Forward + step still run.
    module = _build_module(use_origin_local_feature=False)
    d_q = _first_linear_in_features(module.model.origin_depth_head_g7)
    d_p = _first_linear_in_features(module.model.point_depth_head)
    assert d_q == d_p - module.model_params.fpn_out[1]
    outputs = _forward_with_K(module)
    assert outputs.origin_pred is not None
    assert torch.isfinite(_training_step(module))


_CFG = os.path.join(os.path.dirname(__file__), "..", "config")


def _load_cfg(name):
    with open(os.path.join(_CFG, name)) as f:
        return yaml.safe_load(f)


def test_g11_config_matches_spec():
    base = _load_cfg("sf3d_train_runpod_g10_closeup010.yaml")
    g11 = _load_cfg("sf3d_train_runpod_g11_closeup010.yaml")

    gm = dict(g11["model"]["model_params"])
    bm = dict(base["model"]["model_params"])
    assert gm.pop("use_origin_local_feature") is True
    assert gm == bm  # nothing else in model_params changed

    gd = dict(g11["data"])
    bd = dict(base["data"])
    assert gd.pop("train_data_dir") == "/workspace/datasets/sf3d_processed_v3"
    bd.pop("train_data_dir")
    assert gd == bd  # incl. same key_cache_path (validated against v3)

    assert g11["model"]["loss_params"] == base["model"]["loss_params"]
    assert g11["model"]["optimizer_params"] == base["model"]["optimizer_params"]
    assert g11["seed_everything"] == 42
    assert g11["trainer"]["max_epochs"] == 30
    ckpt = g11["trainer"]["callbacks"][0]["init_args"]["dirpath"]
    logd = g11["trainer"]["logger"]["init_args"]["save_dir"]
    assert "20260816_sf3d_g11_closeup010" in ckpt
    assert "20260816_sf3d_g11_closeup010" in logd
