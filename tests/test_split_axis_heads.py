"""Gen-17 split articulation axis heads (2026-08-18 spec).

Per-type axis readouts (motion_head_rot / motion_head_trans) on the shared
MotionMLP trunk: GT-routed loss at train, predicted-type row-wise selection
at inference, L_pp branches paired with their matching candidate. Flag off
must be byte-identical legacy.
"""
import os

import pytest
import torch
import torch.nn.functional as F

from model.layers import MotionMLP
from model.losses.geometric import PredPredArticulationLoss
from model.losses.split import axis_direction_loss


# ---------------------------------------------------------------- MotionMLP

def test_flag_off_is_legacy():
    torch.manual_seed(0)
    m = MotionMLP(input_dim=16, hidden_dim=8)
    keys = set(m.state_dict().keys())
    assert any(k.startswith("motion_head.") for k in keys)
    assert not any("motion_head_rot" in k or "motion_head_trans" in k for k in keys)
    out = m(torch.randn(4, 16))
    assert len(out) == 2  # (motion_pred, type_logits)


def test_split_state_dict_keys():
    m = MotionMLP(input_dim=16, hidden_dim=8, split_axis_heads=True)
    keys = set(m.state_dict().keys())
    assert any(k.startswith("motion_head_rot.") for k in keys)
    assert any(k.startswith("motion_head_trans.") for k in keys)
    assert not any(k.startswith("motion_head.") for k in keys)


def test_split_forward_shapes():
    m = MotionMLP(input_dim=16, hidden_dim=8, split_axis_heads=True)
    rot, trans, logits = m(torch.randn(4, 16))
    assert rot.shape == (4, 3) and trans.shape == (4, 3)
    assert logits.shape == (4, 2)
    assert not torch.allclose(rot, trans)  # independent readouts


def test_split_without_motion_head_raises():
    with pytest.raises(ValueError):
        MotionMLP(input_dim=16, with_motion_head=False, split_axis_heads=True)


# ------------------------------------------------------- GT-routed gradient

def _routed_loss(m, x, type_gt):
    rot, trans, _ = m(x)
    stack = torch.stack([trans, rot], dim=1)
    routed = stack[torch.arange(stack.shape[0]), type_gt]
    gt = F.normalize(torch.randn(x.shape[0], 3), dim=-1)
    return axis_direction_loss(routed, gt, sign_agnostic=False)


@pytest.mark.parametrize("pure_type,active,inactive", [
    (1, "motion_head_rot", "motion_head_trans"),
    (0, "motion_head_trans", "motion_head_rot"),
])
def test_gradient_reaches_only_gt_type_head(pure_type, active, inactive):
    torch.manual_seed(1)
    m = MotionMLP(input_dim=16, hidden_dim=8, split_axis_heads=True)
    type_gt = torch.full((6,), pure_type, dtype=torch.long)
    loss = _routed_loss(m, torch.randn(6, 16), type_gt)
    loss.backward()
    act = getattr(m, active)[0].weight.grad
    inact = getattr(m, inactive)[0].weight.grad
    assert act is not None and act.abs().sum() > 0
    assert inact is None or torch.all(inact == 0)
    # trunk learns from every row regardless of routing
    assert m.backbone[0].weight.grad.abs().sum() > 0


def test_mixed_batch_routes_per_row():
    torch.manual_seed(2)
    m = MotionMLP(input_dim=16, hidden_dim=8, split_axis_heads=True)
    type_gt = torch.tensor([0, 1, 0, 1])
    loss = _routed_loss(m, torch.randn(4, 16), type_gt)
    loss.backward()
    assert m.motion_head_rot[0].weight.grad.abs().sum() > 0
    assert m.motion_head_trans[0].weight.grad.abs().sum() > 0


# ---------------------------------------------------- predicted-type select

def test_rowwise_selection_matches_argmax():
    torch.manual_seed(3)
    B = 8
    rot = torch.randn(B, 3)
    trans = torch.randn(B, 3)
    logits = torch.randn(B, 2)
    stack = torch.stack([trans, rot], dim=1)   # index 1 = rot (GT convention)
    sel = stack[torch.arange(B), logits.argmax(dim=-1)]
    for i in range(B):
        expect = rot[i] if logits[i, 1] > logits[i, 0] else trans[i]
        assert torch.equal(sel[i], expect)


# ------------------------------------------------------------ L_pp pairing

class _Out:
    mask_logits = torch.zeros(2, 1, 4, 4)

    def __init__(self, rot=None, trans=None):
        B, N = 2, 5
        g = torch.Generator().manual_seed(4)
        self.trajectory_pred = torch.randn(B, N, 3, generator=g)
        self.motion_pred = torch.randn(B, 3, generator=g)
        self.motion_pred_rot = rot
        self.motion_pred_trans = trans
        self.motion_type_logits = torch.randn(B, 2, generator=g)
        self.origin_pred = torch.randn(B, 3, generator=g)
        self.point_3d_pred = torch.randn(B, 3, generator=g)


def test_lpp_legacy_unchanged_when_fields_none():
    loss = PredPredArticulationLoss(weight=1.0)
    o = _Out()
    total, terms = loss(o, None)
    assert torch.isfinite(total)
    # both branches read motion_pred: replacing BOTH candidates with copies
    # of motion_pred must give the identical value
    o2 = _Out(rot=o.motion_pred.clone(), trans=o.motion_pred.clone())
    for attr in ("trajectory_pred", "motion_pred", "motion_type_logits",
                 "origin_pred", "point_3d_pred"):
        setattr(o2, attr, getattr(o, attr))
    total2, _ = loss(o2, None)
    assert torch.allclose(total, total2)


def test_lpp_line_reads_trans_circle_reads_rot():
    loss = PredPredArticulationLoss(weight=1.0)
    o = _Out(rot=torch.randn(2, 3), trans=torch.randn(2, 3))
    total_split, _ = loss(o, None)

    # Manual recompute with the paired axes.
    d = o.trajectory_pred.float()
    dl = F.normalize(o.motion_pred_trans.float(), dim=1)[:, None, :]
    dc = F.normalize(o.motion_pred_rot.float(), dim=1)[:, None, :]
    l_line = torch.cross(d, dl.expand_as(d), dim=-1).pow(2).sum(-1).mean(-1)
    c = (o.origin_pred - o.point_3d_pred).float()[:, None, :]

    def line_dist(x):
        rel = x - c
        along = (rel * dc).sum(-1, keepdim=True)
        return (rel - along * dc).norm(dim=-1)

    r_hat = line_dist(torch.zeros_like(d[:, :1]))
    l_circle = ((line_dist(d) - r_hat).pow(2)
                + (d * dc).sum(-1).pow(2)).mean(-1)
    p_rev = o.motion_type_logits.float().softmax(dim=-1)[:, 1]
    expect = ((1 - p_rev) * l_line + p_rev * l_circle).mean()
    assert torch.allclose(total_split, expect, atol=1e-5)

    # And it must differ from feeding motion_pred to both (distinct axes).
    o_leg = _Out()
    for attr in ("trajectory_pred", "motion_pred", "motion_type_logits",
                 "origin_pred", "point_3d_pred"):
        setattr(o_leg, attr, getattr(o, attr))
    total_leg, _ = loss(o_leg, None)
    assert not torch.allclose(total_split, total_leg)


# ------------------------------------------- segmenter: guards + integration

from tests.test_split_heads import _inputs, _make_cris  # noqa: E402


@pytest.mark.parametrize("over", [
    dict(use_cvae=True),
    dict(use_twist_head=True),
    dict(use_motion_head=False),
    dict(use_motion_type_head=False),
])
def test_segmenter_guards(over):
    with pytest.raises(ValueError):
        _make_cris(split_axis_heads=True, **over)


def _forward(model):
    img, depth, word, mask = _inputs()
    model.eval()
    with torch.no_grad():
        return model(img, depth, word, mask, None, None)


def test_segmenter_split_outputs_and_selection():
    torch.manual_seed(5)
    model = _make_cris(split_axis_heads=True)
    out = _forward(model)
    assert out.motion_pred_rot is not None and out.motion_pred_trans is not None
    assert out.motion_pred.shape == out.motion_pred_rot.shape
    # rescale applied to both candidates: components in (-1, 1)
    for t in (out.motion_pred_rot, out.motion_pred_trans):
        assert t.min() > -1.0 and t.max() < 1.0
    # motion_pred is the row-wise predicted-type selection (index 1 = rot)
    sel = out.motion_type_logits.argmax(dim=-1)
    for i in range(sel.shape[0]):
        expect = out.motion_pred_rot[i] if sel[i] == 1 else out.motion_pred_trans[i]
        assert torch.equal(out.motion_pred[i], expect)


def test_segmenter_flag_off_fields_none():
    model = _make_cris()
    out = _forward(model)
    assert out.motion_pred_rot is None and out.motion_pred_trans is None
    assert out.motion_pred is not None


# -------------------------------------------------- trainer integration

def test_training_step_routed_loss_runs():
    # Full _common_step through the GT-routed axis-loss branch: finite loss
    # and gradients reaching both per-type readouts (the g7 batch mixes
    # types).
    from tests.test_origin_local_sample import _build_module, _training_step
    torch.manual_seed(6)
    m = _build_module(split_axis_heads=True)
    loss = _training_step(m)
    assert torch.isfinite(loss)
    loss.backward()
    mlp = m.model.motion_mlp
    assert mlp.motion_head_rot[0].weight.grad is not None
    assert mlp.motion_head_trans[0].weight.grad is not None


# ------------------------------------------------------------------- config

def _load_cfg(name):
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "..", "config", name)) as f:
        return yaml.safe_load(f)


def test_g17_config_matches_spec():
    g16 = _load_cfg("sf3d_train_runpod_g16_trajnorm.yaml")
    g17 = _load_cfg("sf3d_train_runpod_g17_splitax.yaml")
    mp17 = dict(g17["model"]["model_params"])
    assert mp17.pop("split_axis_heads") is True
    assert mp17 == g16["model"]["model_params"]
    assert g17["model"]["loss_params"] == g16["model"]["loss_params"]
    assert g17["data"] == g16["data"]
    assert "20260818_sf3d_g17_splitax" in g17["trainer"]["callbacks"][0]["init_args"]["dirpath"]
    assert "20260818_sf3d_g17_splitax" in g17["trainer"]["logger"]["init_args"]["save_dir"]
    assert g17["trainer"]["max_epochs"] == g16["trainer"]["max_epochs"]
    assert g17["seed_everything"] == 42
