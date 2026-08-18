"""Gen-18 2D-only training pieces (2026-08-18 spec).

Normalized projection loss (per-row relative error in uv-space), the z_p
depth tether, and the g18 config chain (g17 model verbatim, 3D-GT loss
weights zeroed).
"""
import os
import types

import pytest
import torch
import torch.nn.functional as F
import yaml

from model.losses.geometric import TrajectoryProjectionLoss, normalized_intrinsics


B, N = 3, 6


def _outputs(point_uv, traj):
    return types.SimpleNamespace(
        trajectory_pred=traj, point_uv=point_uv,
        mask_logits=torch.zeros(B, 1, 4, 4),
    )


def _targets(track, valid=None):
    K = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    K[:, 0, 0] = K[:, 1, 1] = 500.0
    K[:, 0, 2] = K[:, 1, 2] = 320.0
    return types.SimpleNamespace(
        trajectory_2d=track, trajectory_2d_valid=valid,
        camera_intrinsic=K, img_size=torch.full((B, 2), 640.0),
    )


def _depth():
    return torch.full((B, 1, 8, 8), 2.0)


def _setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    point_uv = torch.rand(B, 2, generator=g) * 0.5 + 0.25
    traj = torch.randn(B, N, 3, generator=g) * 0.05
    traj[:, 0] = 0.0
    track = torch.rand(B, N, 2, generator=g) * 0.4 + 0.3
    return point_uv, traj, track


def test_flag_off_bit_identity():
    point_uv, traj, track = _setup()
    legacy = TrajectoryProjectionLoss(weight=1.0)
    total_legacy, _ = legacy(_outputs(point_uv, traj), _targets(track), _depth())
    # reconstruct the flat masked mean by hand via the normalized=False path
    again, _ = TrajectoryProjectionLoss(weight=1.0, normalized=False)(
        _outputs(point_uv, traj), _targets(track), _depth()
    )
    assert torch.equal(total_legacy, again)


def test_normalized_collapsed_prediction_scores_about_one():
    # Prediction collapsed to the anchor; anchor projects exactly onto the
    # track's first point => per-row error == per-row GT motion energy
    # relative to that point => ratio ~1 for every row.
    point_uv, _, track = _setup()
    traj = torch.zeros(B, N, 3)
    track = track.clone()
    track[:, 0] = point_uv  # anchor projects to the first track point
    loss = TrajectoryProjectionLoss(weight=1.0, normalized=True)
    total, _ = loss(_outputs(point_uv, traj), _targets(track), _depth())
    assert abs(float(total) - 1.0) < 1e-4


def test_normalized_scale_invariance_in_uv():
    # Shrinking GT motion and prediction error together leaves the
    # normalized ratio unchanged (above the eps floor).
    point_uv = torch.full((B, 2), 0.5)
    base = torch.zeros(B, N, 2)
    base[:, :, 0] = torch.linspace(0, 0.3, N)
    loss = TrajectoryProjectionLoss(weight=1.0, normalized=True)
    vals = []
    for scale in (1.0, 0.5):
        track = point_uv.unsqueeze(1) + base * scale
        traj = torch.zeros(B, N, 3)  # collapsed pred; error scales with GT
        t, _ = loss(_outputs(point_uv, traj), _targets(track), _depth())
        vals.append(float(t))
    assert abs(vals[0] - vals[1]) < 1e-4


def test_normalized_eps_floor_on_degenerate_track():
    # A GT track with ~zero motion: the eps floor keeps the ratio finite.
    point_uv = torch.full((B, 2), 0.5)
    track = point_uv.unsqueeze(1).repeat(1, N, 1)  # no motion at all
    traj = torch.randn(B, N, 3) * 0.01
    loss = TrajectoryProjectionLoss(weight=1.0, normalized=True)
    total, _ = loss(_outputs(point_uv, traj), _targets(track), _depth())
    assert torch.isfinite(total)


def test_normalized_first_valid_point_defines_relative_frame():
    # Row 0's first point invalid ([0,0] placeholder): the energy frame
    # must anchor at its first VALID point, not the placeholder.
    point_uv, traj, track = _setup(seed=1)
    track = track.clone()
    track[0, 0] = 0.0  # placeholder
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[0, 0] = False
    loss = TrajectoryProjectionLoss(weight=1.0, normalized=True)
    total, _ = loss(_outputs(point_uv, traj), _targets(track, valid), _depth())
    assert torch.isfinite(total)
    # If the placeholder were used, row 0's energy would include the huge
    # [0,0]->track jump; verify the value matches a manual recompute that
    # anchors at index 1.
    # (smoke-level check: computing with all-valid must give a DIFFERENT,
    # placeholder-poisoned value)
    total_bad, _ = loss(_outputs(point_uv, traj), _targets(track, None), _depth())
    assert not torch.allclose(total, total_bad)


# ----------------------------------------------------------- config chain

def _load_cfg(name):
    with open(os.path.join(os.path.dirname(__file__), "..", "config", name)) as f:
        return yaml.safe_load(f)


def test_g18_config_matches_spec():
    g17 = _load_cfg("sf3d_train_runpod_g17_splitax.yaml")
    g18 = _load_cfg("sf3d_train_runpod_g18_2donly.yaml")
    # model architecture byte-identical to g17
    assert g18["model"]["model_params"] == g17["model"]["model_params"]
    lp = dict(g18["model"]["loss_params"])
    for k in ("vae_weight", "motion_type_weight", "origin_weight",
              "origin_map_weight", "point_3d_weight", "trajectory_weight"):
        assert lp[k] == 0.0, k
    assert lp["trajectory_proj_weight"] == 0.5
    assert lp["trajectory_proj_normalized"] is True
    assert lp["depth_anchor_weight"] == 0.5
    for k in ("mask_weight", "point_map_weight", "coord_weight"):
        assert lp[k] == 0.5, k
    assert lp["pred_pred_art_weight"] == 0.1
    assert lp["pred_pred_art_normalized"] is True
    assert g18["data"]["return_trajectory_2d"] is True
    d17 = dict(g17["data"]); d18 = dict(g18["data"])
    d18.pop("return_trajectory_2d")
    assert d17 == d18
    assert "20260818_sf3d_g18_2donly" in g18["trainer"]["callbacks"][0]["init_args"]["dirpath"]
    assert g18["trainer"]["max_epochs"] == g17["trainer"]["max_epochs"]


# ------------------------------------------------ trainer integration

def test_training_step_2donly_no_3dgt_gradient():
    # Full _common_step with the g18 loss weights: finite loss, and the
    # split axis heads receive gradient ONLY through L_pp (turning L_pp
    # off must zero their grads — proves no 3D-GT term leaks in).
    from tests.test_origin_local_sample import _build_module, _training_step

    def build(pp_weight):
        m = _build_module(split_axis_heads=True)
        lp = m.loss_params
        for k in ("vae_weight", "motion_type_weight", "origin_weight",
                  "origin_map_weight", "point_3d_weight", "trajectory_weight"):
            setattr(lp, k, 0.0)
        lp.trajectory_proj_weight = 0.5
        lp.trajectory_proj_normalized = True
        lp.depth_anchor_weight = 0.5
        lp.pred_pred_art_weight = pp_weight
        # rebuild loss modules that captured weights at __init__
        from model.losses.geometric import TrajectoryProjectionLoss as TPL
        from model.losses import build_geometric_loss
        m.traj_projection_loss = TPL(weight=0.5, normalized=True)
        m.geometric_loss = build_geometric_loss(
            m.loss_params, trajectory_is_absolute=True
        )
        return m

    torch.manual_seed(7)
    m = build(pp_weight=0.1)
    loss = _training_step(m)
    assert torch.isfinite(loss)
    loss.backward()
    rot_g = m.model.motion_mlp.motion_head_rot[0].weight.grad
    assert rot_g is not None and rot_g.abs().sum() > 0  # L_pp teaches it

    torch.manual_seed(7)
    m0 = build(pp_weight=0.0)
    loss0 = _training_step(m0)
    loss0.backward()
    g = m0.model.motion_mlp.motion_head_rot[0].weight.grad
    assert g is None or torch.all(g == 0)  # nothing else touches the axis
