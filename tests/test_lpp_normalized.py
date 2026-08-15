import torch

from model.losses.geometric import PredPredArticulationLoss, build_geometric_loss
from model.targets import StepTargets
from tests.test_split_heads import _circle_traj, _pp_outputs

torch.manual_seed(0)


def _rand_case(B=4, N=20):
    traj = torch.randn(B, N, 3) * 0.2
    traj = traj - traj[:, :1]                       # relative frame
    dhat = torch.nn.functional.normalize(torch.randn(B, 3), dim=1)
    return _pp_outputs(traj, dhat, p_rev_logit=1.3,
                       origin=torch.randn(B, 3), anchor=torch.randn(B, 3))


def test_normalized_false_bit_identical():
    out = _rand_case()
    old = PredPredArticulationLoss(weight=0.5)
    new = PredPredArticulationLoss(weight=0.5, normalized=False, radius_floor=0.10)
    t_old, _ = old(out, StepTargets())
    t_new, _ = new(out, StepTargets())
    assert torch.equal(t_old, t_new)


def test_normalized_line_branch_at_most_one():
    # All-prismatic gate: normalized line = off-axis energy fraction <= 1.
    out = _rand_case()
    out.motion_type_logits[:, 1] = -20.0            # P(rev) ~ 0
    loss = PredPredArticulationLoss(weight=1.0, normalized=True)
    _, terms = loss(out, StepTargets())
    assert terms["L_geo_pred_pred_art"].item() <= 1.0 + 1e-6


def test_normalized_circle_scale_invariant():
    # Uniformly scaling the scene must not change the relative orbit error
    # (r_hat stays above the floor at both scales).
    dhat = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=1)
    anchor = torch.tensor([[0.5, 0.2, 1.0]])
    origin = torch.tensor([[0.2, 0.2, 1.0]])        # r_hat = 0.3 m
    center_rel = (origin - anchor)[0]
    traj = _circle_traj(center_rel, dhat[0], torch.zeros(3),
                        torch.linspace(0, 1.2, 20))[None]
    traj = traj + 0.03 * torch.randn_like(traj)     # imperfect orbit
    traj = traj - traj[:, :1]
    loss = PredPredArticulationLoss(weight=1.0, normalized=True)
    _, t1 = loss(_pp_outputs(traj, dhat, 20.0, origin, anchor), StepTargets())
    s = 3.0
    _, t2 = loss(_pp_outputs(s * traj, dhat, 20.0, s * origin, s * anchor),
                 StepTargets())
    assert abs(t1["L_geo_pred_pred_art"].item()
               - t2["L_geo_pred_pred_art"].item()) < 1e-5


def test_normalized_radius_floor_engages():
    # r_hat ~ 0 (origin == anchor): denominator is the floor squared, so
    # halving the floor quadruples the loss.
    out = _rand_case(B=1)
    out.origin_pred = out.point_3d_pred.clone()     # r_hat -> 0
    out.motion_type_logits[:, 1] = 20.0             # P(rev) ~ 1
    lo = PredPredArticulationLoss(weight=1.0, normalized=True, radius_floor=0.10)
    hi = PredPredArticulationLoss(weight=1.0, normalized=True, radius_floor=0.05)
    _, t_lo = lo(out, StepTargets())
    _, t_hi = hi(out, StepTargets())
    ratio = t_hi["L_geo_pred_pred_art"].item() / max(
        t_lo["L_geo_pred_pred_art"].item(), 1e-12)
    assert abs(ratio - 4.0) < 1e-3


def test_build_geometric_loss_forwards_new_params():
    class LP:
        geometric_loss = "pred_pred_art"
        pred_pred_art_weight = 0.1
        pred_pred_art_normalized = True
        pred_pred_art_radius_floor = 0.10
    loss = build_geometric_loss(LP())
    assert loss.normalized is True
    assert loss.radius_floor == 0.10
    assert loss.weight == 0.1


def test_lossparams_defaults():
    # LossParams has required positional fields (bce_weight, ...), so a bare
    # LossParams() can't be built — read the new fields' defaults instead.
    import dataclasses

    from config.opd_train import LossParams
    defaults = {f.name: f.default for f in dataclasses.fields(LossParams)}
    assert defaults["pred_pred_art_normalized"] is False
    assert defaults["pred_pred_art_radius_floor"] == 0.10


import os

import yaml

_CFG = os.path.join(os.path.dirname(__file__), "..", "config")


def test_g10_config_matches_spec():
    with open(os.path.join(_CFG, "sf3d_train_runpod_g9_closeup010.yaml")) as f:
        base = yaml.safe_load(f)
    with open(os.path.join(_CFG, "sf3d_train_runpod_g10_closeup010.yaml")) as f:
        g10 = yaml.safe_load(f)

    gl = g10["model"]["loss_params"]
    assert gl["pred_pred_art_weight"] == 0.1
    assert gl["pred_pred_art_normalized"] is True
    assert gl["pred_pred_art_radius_floor"] == 0.10
    assert gl["geometric_loss"] == "pred_pred_art"

    # Everything else identical to gen-9.
    bl = dict(base["model"]["loss_params"])
    gl2 = dict(gl)
    for k in ("pred_pred_art_weight", "pred_pred_art_normalized",
              "pred_pred_art_radius_floor"):
        bl.pop(k, None), gl2.pop(k, None)
    assert gl2 == bl
    assert g10["model"]["model_params"] == base["model"]["model_params"]
    assert g10["data"] == base["data"]
    assert g10["model"]["optimizer_params"] == base["model"]["optimizer_params"]
    assert g10["seed_everything"] == 42
    assert g10["trainer"]["max_epochs"] == 30
    ckpt = g10["trainer"]["callbacks"][0]["init_args"]["dirpath"]
    logd = g10["trainer"]["logger"]["init_args"]["save_dir"]
    assert "20260815_sf3d_g10_closeup010" in ckpt
    assert "20260815_sf3d_g10_closeup010" in logd
