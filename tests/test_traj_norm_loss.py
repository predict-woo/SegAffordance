# tests/test_traj_norm_loss.py
#
# Gen-16: per-row GT-energy-normalized trajectory loss (flag-gated,
# relative-trajectory mode only). The pure-formula tests pin the reference
# `_norm_loss`; the module tests pin the trainer against that reference
# (flag on) and against plain F.mse_loss (flag off, bit-identical).
import os
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from config.opd_train import Config, LossParams, OptimizerParams
from tests.test_split_heads import _params, _sf3d_batch, _StubBackbone  # noqa: E402
from train_SF3D_better import SF3DTrainingModule


def _rows(B=4, N=20, scale=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.randn(B, N, 3, generator=g) * scale
    gt = gt - gt[:, 0:1]
    pred = torch.randn(B, N, 3, generator=g) * scale
    pred = pred - pred[:, 0:1]
    return pred, gt


def _norm_loss(pred, gt):
    # The spec's formula, straight: used to pin the trainer's implementation.
    per_row = (pred - gt).pow(2).sum(-1).mean(-1)
    energy = gt.pow(2).sum(-1).mean(-1)
    return (per_row / energy.clamp(min=1e-4)).mean()


def test_collapsed_prediction_scores_one():
    _, gt = _rows()
    loss = _norm_loss(torch.zeros_like(gt), gt)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


def test_scale_invariance():
    pred, gt = _rows(scale=0.3)
    l1 = _norm_loss(pred, gt)
    l2 = _norm_loss(5.0 * pred, 5.0 * gt)
    assert torch.isclose(l1, l2, rtol=1e-5)


def test_eps_floor_damps_degenerate_gt():
    # 1 cm GT stub: energy ~3e-5 < 1e-4 floor -> loss is measured against
    # the floor, not amplified by the tiny denominator.
    gt = torch.zeros(1, 20, 3)
    gt[:, :, 2] = torch.linspace(0, 0.01, 20)
    pred = torch.zeros_like(gt)
    loss = _norm_loss(pred, gt)
    energy = gt.pow(2).sum(-1).mean(-1)
    assert energy.item() < 1e-4
    assert loss.item() < 1.0            # damped below the collapsed score


# ---- trainer integration ----------------------------------------------------
#
# Same stub-backbone module machinery as tests/test_split_heads._split_module
# (relative trajectory head) and tests/test_g7_lift._g7_module (absolute).


def _build_module(trajectory_absolute=False, trajectory_loss_normalized=False):
    lp = LossParams(
        bce_weight=0.5, dice_weight=0.5, mask_weight=0.5,
        point_map_weight=0.0, coord_weight=0.0, vae_weight=0.5,
        motion_type_weight=0.5, point_sigma=8.0, vae_beta=0.01,
        trajectory_weight=0.5, geometric_loss="pred_pred_art",
        pred_pred_art_weight=0.5, axis_sign_agnostic=False,
        origin_weight=0.5, point_3d_weight=0.5,
        trajectory_loss_normalized=trajectory_loss_normalized,
    )
    op = OptimizerParams(lr=1e-5, weight_decay=0.0,
                         scheduler_milestones=[10], scheduler_gamma=0.1)
    cfg = Config(log_image_interval_steps=0, input_size=[64, 64],
                 enable_wandb=False, val_vis_samples=0, manual_seed=0)
    if trajectory_absolute:
        # Mirror _g7_module's model flags (the absolute head requires
        # trajectory_delta_cumsum off).
        mp = _params(use_origin_heatmap=True, predict_point_depth=True,
                     trajectory_absolute=True, trajectory_delta_cumsum=False,
                     pool_with_predicted_mask=True)
    else:
        mp = _params(point_prediction_3d=True, use_origin_head=True,
                     pool_with_predicted_mask=True)
    with mock.patch(
        "model.segmenter.build_backbone",
        lambda mp, fpn_in: _StubBackbone(fpn_in, word_len=mp.word_len),
    ):
        return SF3DTrainingModule(mp, lp, op, cfg)


def _run_step(m, batch):
    # Capture the logged scalars AND the forward's ModelOutputs, so every
    # assertion compares the trainer's logged loss against a reference
    # recomputed independently from the module's own outputs.
    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(
        name, float(value) if torch.is_tensor(value) else value
    )
    captured = {}
    handle = m.model.register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("out", out)
    )
    try:
        loss = m._common_step(batch, 0, "train")
    finally:
        handle.remove()
    return loss, logged, captured["out"]


def test_trainer_uses_normalized_when_flag_on():
    torch.manual_seed(0)
    m = _build_module(trajectory_loss_normalized=True)
    batch = _sf3d_batch()
    loss, logged, out = _run_step(m, batch)
    assert torch.isfinite(loss)

    traj_gt = batch[12]
    gt_rel = traj_gt - traj_gt[:, 0:1, :]
    pred = out.trajectory_pred.detach()
    ref = _norm_loss(pred, gt_rel)
    assert logged["train/L_trajectory"] == pytest.approx(ref.item(), rel=1e-6)
    # The un-normalized m² diagnostic is logged alongside.
    m2_ref = (pred - gt_rel).pow(2).sum(-1).mean(-1).mean()
    assert logged["train/L_trajectory_m2"] == pytest.approx(
        m2_ref.item(), rel=1e-6
    )


def test_flag_off_bit_identical():
    torch.manual_seed(0)
    m = _build_module()          # flag defaults off
    batch = _sf3d_batch()
    loss, logged, out = _run_step(m, batch)
    assert torch.isfinite(loss)

    traj_gt = batch[12]
    gt_rel = traj_gt - traj_gt[:, 0:1, :]
    ref = F.mse_loss(out.trajectory_pred.detach(), gt_rel)
    # EXACT equality: the off-path is the pre-change nn.MSELoss call.
    assert logged["train/L_trajectory"] == ref.item()
    assert "train/L_trajectory_m2" not in logged


def test_absolute_plus_normalized_raises():
    with pytest.raises(ValueError):
        _build_module(trajectory_absolute=True, trajectory_loss_normalized=True)


# ---- gen-16 config ----------------------------------------------------------


def _load_cfg(name):
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "..", "config", name)) as f:
        return yaml.safe_load(f)


def test_g16_config_matches_spec():
    base = _load_cfg("sf3d_train_runpod_g13_res512.yaml")
    g16 = _load_cfg("sf3d_train_runpod_g16_trajnorm.yaml")
    gl, bl = dict(g16["model"]["loss_params"]), dict(base["model"]["loss_params"])
    assert gl.pop("trajectory_loss_normalized") is True
    assert gl.pop("trajectory_weight") == 0.5 and bl.pop("trajectory_weight") == 0.15
    assert gl == bl
    assert g16["model"]["model_params"] == base["model"]["model_params"]
    assert g16["data"] == base["data"]
    assert "20260817_sf3d_g16_trajnorm" in g16["trainer"]["callbacks"][0]["init_args"]["dirpath"]
    assert "20260817_sf3d_g16_trajnorm" in g16["trainer"]["logger"]["init_args"]["save_dir"]
    assert g16["trainer"]["max_epochs"] == 30 and g16["seed_everything"] == 42
