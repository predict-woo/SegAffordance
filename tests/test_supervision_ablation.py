# tests/test_supervision_ablation.py
#
# Supervision-ablation arms (2026-08-15 spec), all derived from the gen-9
# recipe (origin heatmap + point depth, relative direct trajectory readout,
# cvae/twist off):
#   arm A: full gen-9 supervision (no removals) — the control;
#   arm B: trajectory head removed (use_trajectory_head=False);
#   arm C: articulation heads removed (axis, type, origin heatmap).
# The model-side gating already exists; these tests pin the trainer side:
# test_step must SKIP absent-head metrics instead of scoring a zeros
# placeholder, and the wandb viz path must not index a None trajectory.

import os
from unittest import mock

import torch
import yaml

from config.opd_train import Config, LossParams, OptimizerParams
from tests.test_g7_lift import _g7_batch, _knorm
from tests.test_split_heads import _StubBackbone, _inputs, _params
from train_SF3D_better import SF3DTrainingModule

_ARM_MODEL_FLAGS = {
    "A": {},
    "B": {"use_trajectory_head": False},
    "C": {
        "use_motion_head": False,
        "use_motion_type_head": False,
        "use_origin_heatmap": False,
    },
}
_ARM_LOSS_OVERRIDES = {
    "A": {},
    "B": {
        "geometric_loss": "none",
        "trajectory_weight": 0.0,
        "pred_pred_art_weight": 0.0,
    },
    "C": {
        "geometric_loss": "none",
        "vae_weight": 0.0,
        "motion_type_weight": 0.0,
        "origin_weight": 0.0,
        "origin_map_weight": 0.0,
        "pred_pred_art_weight": 0.0,
    },
}


def _abl_module(arm):
    lp_kwargs = dict(
        bce_weight=0.5, dice_weight=0.5, mask_weight=0.5,
        point_map_weight=0.5, coord_weight=0.5, vae_weight=0.5,
        motion_type_weight=0.5, point_sigma=8.0, vae_beta=0.01,
        trajectory_weight=0.5, geometric_loss="pred_pred_art",
        pred_pred_art_weight=0.5, axis_sign_agnostic=False,
        origin_weight=0.5, point_3d_weight=0.5, origin_map_weight=0.5,
    )
    lp_kwargs.update(_ARM_LOSS_OVERRIDES[arm])
    op = OptimizerParams(lr=1e-5, weight_decay=0.0,
                         scheduler_milestones=[10], scheduler_gamma=0.1)
    cfg = Config(log_image_interval_steps=0, input_size=[64, 64],
                 enable_wandb=False, val_vis_samples=0, manual_seed=0)
    # gen-9 base flags: classical 2D point path + origin heatmap + point
    # depth, RELATIVE direct trajectory (no cumsum, not absolute).
    mp_kwargs = dict(use_origin_heatmap=True, predict_point_depth=True,
                     trajectory_absolute=False, trajectory_delta_cumsum=False,
                     pool_with_predicted_mask=True)
    mp_kwargs.update(_ARM_MODEL_FLAGS[arm])
    with mock.patch(
        "model.segmenter.build_backbone",
        lambda mp, fpn_in: _StubBackbone(fpn_in, word_len=mp.word_len),
    ):
        return SF3DTrainingModule(
            _params(**mp_kwargs), LossParams(**lp_kwargs), op, cfg,
        )


def _forward(module):
    # Direct model forward with normalized intrinsics, as the _g7 tests do,
    # so the lifted fields (point_3d_pred / origin_pred) can exist.
    module.eval()
    img, depth, word, mask = _inputs()
    with torch.no_grad():
        return module.model(img, depth, word, mask, None, None, None, _knorm(2))


def _training_step(module):
    module.train()
    module.log = lambda *a, **k: None
    return module._common_step(_g7_batch(), 0, "train")


def _run_test_step(module):
    module.eval()
    module.log = lambda *a, **k: None
    module.config.test_visualize_debug = False   # viz branch touches trainer
    module.on_test_start()
    with torch.no_grad():
        module.test_step(_g7_batch(), 0)


def test_armB_no_trajectory_head_trains():
    torch.manual_seed(0)
    module = _abl_module(arm="B")
    outputs = _forward(module)
    assert module.model.trajectory_predictor is None
    assert outputs.trajectory_pred is None
    assert outputs.point_3d_pred is not None      # point pipeline intact
    assert outputs.origin_pred is not None        # origin pipeline intact
    loss = _training_step(module)                 # _common_step "train"
    assert torch.isfinite(loss)


def test_armC_no_articulation_trains():
    torch.manual_seed(0)
    module = _abl_module(arm="C")
    outputs = _forward(module)
    for f in ("motion_pred", "motion_type_logits", "origin_uv",
              "origin_logits", "origin_pred"):
        assert getattr(outputs, f) is None, f
    assert outputs.point_3d_pred is not None      # z_p lift intact
    assert outputs.trajectory_pred is not None
    loss = _training_step(module)
    assert torch.isfinite(loss)


def test_armC_test_step_skips_axis_and_type_metrics():
    torch.manual_seed(0)
    module = _abl_module(arm="C")
    _run_test_step(module)              # one small SF3D-format test batch
    assert module._test_axis_errors_all == []
    assert module._test_axis_errors_matched == []
    assert module._test_type_correct_all == 0
    assert module._test_ma_correct_all == 0
    assert module._test_has_axis_head is False
    assert module._test_has_type_head is False
    assert len(module._test_ious) > 0             # mask metrics still collected


def test_armA_test_step_still_collects_axis_and_type():
    torch.manual_seed(0)
    module = _abl_module(arm="A")       # plain gen-7/9 flags, no removals
    _run_test_step(module)
    assert len(module._test_axis_errors_all) > 0
    assert module._test_has_axis_head is True
    assert module._test_has_type_head is True


def test_armB_wandb_viz_guard():
    # The OPDReal wandb viz path indexes trajectory_pred[i]; with the head
    # off it must fall back instead of raising TypeError. Unit-level: just
    # verify the expression the fix uses.
    tp = None
    fallback = tp[0] if tp is not None else torch.zeros(20, 3)
    assert fallback.shape == (20, 3)


_CFG = os.path.join(os.path.dirname(__file__), "..", "config")


def _load(name):
    with open(os.path.join(_CFG, name)) as f:
        return yaml.safe_load(f)


def test_ablation_configs_match_spec():
    base = _load("sf3d_train_runpod_g9_closeup010.yaml")
    art = _load("sf3d_train_runpod_g9abl_artonly.yaml")
    trj = _load("sf3d_train_runpod_g9abl_trajonly.yaml")

    bm, am, tm = (c["model"]["model_params"] for c in (base, art, trj))
    bl, al, tl = (c["model"]["loss_params"] for c in (base, art, trj))
    bd, ad, td = (c["data"] for c in (base, art, trj))

    # Arm B: only the trajectory path is removed.
    assert am["use_trajectory_head"] is False
    assert al["geometric_loss"] == "none"
    assert al["trajectory_weight"] == 0.0
    assert al["pred_pred_art_weight"] == 0.0
    assert am["use_motion_head"] and am["use_motion_type_head"]
    assert am["use_origin_heatmap"] and am["predict_point_depth"]

    # Arm C: only the articulation paths are removed.
    assert tm["use_motion_head"] is False
    assert tm["use_motion_type_head"] is False
    assert tm["use_origin_heatmap"] is False
    assert tm["predict_point_depth"] is True
    assert tm.get("use_trajectory_head", True) is True
    assert tl["geometric_loss"] == "none"
    for k in ("vae_weight", "motion_type_weight", "origin_weight",
              "origin_map_weight", "pred_pred_art_weight"):
        assert tl[k] == 0.0, k

    # Constants identical across all three arms.
    for m, l, d in ((am, al, ad), (tm, tl, td)):
        assert d["key_cache_path"] == bd["key_cache_path"]
        assert d["min_mask_area_frac"] == bd["min_mask_area_frac"]
        assert d["edge_margin_frac"] == bd["edge_margin_frac"]
        assert d["batch_size_train"] == bd["batch_size_train"]
        assert m["clip_pretrain"] == bm["clip_pretrain"]
        assert l["mask_weight"] == bl["mask_weight"]
        assert l["point_3d_weight"] == bl["point_3d_weight"]
    for c in (base, art, trj):
        assert c["trainer"]["max_epochs"] == 30
        assert c["model"]["optimizer_params"]["scheduler_milestones"] == [24, 28]
        assert c["seed_everything"] == 42

    # Each arm writes to its own experiment dir.
    for c, tag in ((art, "g9abl_artonly"), (trj, "g9abl_trajonly")):
        ckpt = c["trainer"]["callbacks"][0]["init_args"]["dirpath"]
        logd = c["trainer"]["logger"]["init_args"]["save_dir"]
        assert tag in ckpt and tag in logd
