# tests/test_split_heads.py
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from config.opd_train import ModelParams
from model.backbones.base import BackboneBase
from model.layers import Point3DHead
from model.outputs import ModelOutputs
from model.segmenter import CRIS


def test_point3d_head_shape():
    head = Point3DHead(input_dim=32, hidden_dim=16)
    out = head(torch.randn(4, 32))
    assert out.shape == (4, 3)
    # Unconstrained output: gradients reach the input.
    out.sum().backward()


def test_model_outputs_new_fields_default_none():
    o = ModelOutputs(mask_logits=torch.zeros(1, 1, 4, 4))
    assert o.point_3d_pred is None
    assert o.origin_pred is None
    assert o.point_logits is None
    assert o.coords_hat is None


# ---- segmenter wiring (Task 2) --------------------------------------------
#
# The brief's `clip_pretrain=""` does not survive ClipRN50Backbone
# (torch.jit.load("") raises), and no existing test builds a CRIS to mirror,
# so per the brief's fallback note we build the small CPU CRIS around a stub
# backbone: same (v2, v3, v4) pyramid contract as BackboneBase, random-init
# convs, patched in via model.segmenter.build_backbone.


class _StubBackbone(BackboneBase):
    def __init__(self, fpn_in, word_len, word_dim=64, state_dim=96):
        super().__init__()
        self.fpn_in = list(fpn_in)
        self.word_dim = word_dim
        self.state_dim = state_dim
        self.pad_token_id = 0
        self.max_context_length = word_len
        self.c2 = nn.Conv2d(3, fpn_in[0], kernel_size=8, stride=8)
        self.c3 = nn.Conv2d(fpn_in[0], fpn_in[1], kernel_size=2, stride=2)
        self.c4 = nn.Conv2d(fpn_in[1], fpn_in[2], kernel_size=2, stride=2)
        self.tok = nn.Embedding(100, word_dim)
        self.state_proj = nn.Linear(word_dim, state_dim)

    def encode_image(self, img):
        v2 = self.c2(img)
        v3 = self.c3(v2)
        v4 = self.c4(v3)
        return v2, v3, v4

    def encode_text(self, tokens):
        word = self.tok(tokens)
        state = self.state_proj(word.float().mean(dim=1))
        return word, state

    def tokenize(self, texts, context_length):
        # _common_step calls model.tokenize(); integer ids within the
        # stub embedding's range are all it needs.
        return torch.randint(1, 100, (len(texts), context_length))


def _params(**over):
    base = dict(
        clip_pretrain="",  # unused: the stub backbone is patched in
        word_len=17, depth_feat_channels=[8, 8], fpn_in=[64, 128, 128],
        fpn_out=[32, 64, 128], num_layers=1, num_head=2, dim_ffn=64,
        dropout=0.0, intermediate=False, proj_dropout=0.0,
        vae_latent_dim=8, vae_hidden_dim=32, num_motion_types=2,
        use_depth=True, use_cvae=False, use_trajectory_head=True,
        trajectory_delta_cumsum=True, use_twist_head=False,
        use_motion_head=True, use_motion_type_head=True,
    )
    base.update(over)
    return ModelParams(**base)


def _make_cris(**over):
    params = _params(**over)
    with mock.patch(
        "model.segmenter.build_backbone",
        lambda mp, fpn_in: _StubBackbone(fpn_in, word_len=mp.word_len),
    ):
        return CRIS(params)


def _inputs(B=2, size=64):
    img = torch.randint(0, 255, (B, 3, size, size), dtype=torch.uint8)
    depth = torch.rand(B, 1, size, size)
    word = torch.randint(1, 100, (B, 17))
    mask = (torch.rand(B, 1, size, size) > 0.5).float()
    return img, depth, word, mask


@pytest.fixture(scope="module")
def split_model():
    m = _make_cris(point_prediction_3d=True, use_origin_head=True,
                   pool_with_predicted_mask=True)
    m.eval()
    return m


def test_3d_mode_output_shapes(split_model):
    img, depth, word, mask = _inputs()
    with torch.no_grad():
        out = split_model(img, depth, word, mask, None, None)
    assert out.point_logits is None and out.coords_hat is None
    assert out.point_3d_pred.shape == (2, 3)
    assert out.origin_pred.shape == (2, 3)
    assert out.motion_pred.shape == (2, 3)
    assert out.motion_type_logits.shape == (2, 2)
    assert out.trajectory_pred.shape == (2, 20, 3)
    assert out.mask_logits.shape[1] == 1


def test_predicted_mask_pooling_allows_mask_none(split_model):
    # With pool_with_predicted_mask, train mode must not need a GT mask.
    img, depth, word, _ = _inputs()
    split_model.train()
    try:
        out = split_model(img, depth, word, None, None, None)
    finally:
        split_model.eval()
    assert out.point_3d_pred.shape == (2, 3)


def test_pooling_detached_from_mask_head(split_model):
    # Gradient of an articulation output must NOT reach the mask projector
    # through the pooling path (detached sigmoid).
    img, depth, word, _ = _inputs()
    split_model.train()
    try:
        split_model.zero_grad(set_to_none=True)
        out = split_model(img, depth, word, None, None, None)
        out.point_3d_pred.sum().backward()
        proj_grads = [p.grad for p in split_model.proj.parameters()]
        assert all(g is None or torch.all(g == 0) for g in proj_grads)
    finally:
        split_model.eval()
        split_model.zero_grad(set_to_none=True)


def test_pool_with_predicted_mask_flag_semantics():
    # Pins the flag both ways, in train mode with seeded weights/inputs:
    #   false (classical) -> pooling reads the GT mask, so changing the GT
    #     mask MUST change an articulation output;
    #   true  -> the GT mask is unused by pooling, so the SAME mask change
    #     leaves every articulation output bit-identical.
    def _build(pool_flag):
        torch.manual_seed(123)
        m = _make_cris(pool_with_predicted_mask=pool_flag)
        m.train()  # eval always pools with the predicted mask; train is the fork
        return m

    torch.manual_seed(7)
    img, depth, word, _ = _inputs()
    mask_a = torch.zeros(2, 1, 64, 64)
    mask_a[:, :, :32, :] = 1.0
    mask_b = torch.zeros(2, 1, 64, 64)
    mask_b[:, :, 32:, :] = 1.0

    def _art(m, mask):
        with torch.no_grad():
            out = m(img, depth, word, mask, None, None)
        return out.motion_pred, out.trajectory_pred

    m_false = _build(False)
    ma, ta = _art(m_false, mask_a)
    mb, tb = _art(m_false, mask_b)
    assert not torch.equal(ma, mb) or not torch.equal(ta, tb)

    m_true = _build(True)
    ma, ta = _art(m_true, mask_a)
    mb, tb = _art(m_true, mask_b)
    assert torch.equal(ma, mb) and torch.equal(ta, tb)


def test_classical_2d_mode_unchanged():
    m = _make_cris()  # all new flags default off
    m.eval()
    img, depth, word, mask = _inputs()
    with torch.no_grad():
        out = m(img, depth, word, mask, None, None)
    assert out.point_logits is not None and out.coords_hat.shape == (2, 2)
    assert out.point_3d_pred is None and out.origin_pred is None


def test_hint_on_2d_condition_keeps_classical_column_order():
    # Checkpoint-compatibility pin: every gen-3/4/5 twist config sets
    # use_motion_type_input, and those checkpoints were trained with the
    # condition laid out [features, coords_hat, type_emb]. Loading them
    # requires the type embedding to occupy the LAST columns, with
    # coords_hat immediately before it — this test freezes that layout.
    emb_dim = 16
    m = _make_cris(use_motion_type_input=True, motion_type_embedding_dim=emb_dim)
    m.eval()
    marker = 7.25
    with torch.no_grad():
        m.motion_type_embedding.weight.fill_(marker)

    captured = []
    hook = m.motion_mlp.register_forward_hook(
        lambda mod, inp, out: captured.append(inp[0].detach())
    )
    img, depth, word, mask = _inputs()
    try:
        with torch.no_grad():
            out = m(img, depth, word, mask, None, None)
    finally:
        hook.remove()

    cond = captured[0]
    # Type embedding fills the LAST emb_dim columns...
    assert torch.all(cond[:, -emb_dim:] == marker)
    # ...and coords_hat sits immediately before it (classical order).
    assert torch.allclose(cond[:, -emb_dim - 2:-emb_dim], out.coords_hat)


# ---- split losses (Task 3) -------------------------------------------------

from model.losses.split import (
    axis_direction_loss,
    origin_canonical_loss,
    perpendicular_foot,
)


def test_q_star_is_perpendicular_foot_and_annotation_gauge_invariant():
    torch.manual_seed(0)
    o = torch.randn(8, 3)
    d = torch.nn.functional.normalize(torch.randn(8, 3), dim=1)
    p = torch.randn(8, 3)
    q_star = perpendicular_foot(o, d, p)
    # (q* - p) is perpendicular to the axis.
    assert torch.allclose(((q_star - p) * d).sum(-1), torch.zeros(8), atol=1e-5)
    # Sliding the annotated origin along the axis does not move q*.
    q_star_slid = perpendicular_foot(o + 3.7 * d, d, p)
    assert torch.allclose(q_star, q_star_slid, atol=1e-5)


def test_origin_loss_gauge_invariant_and_zero_at_target():
    torch.manual_seed(1)
    o = torch.randn(4, 3)
    d = torch.nn.functional.normalize(torch.randn(4, 3), dim=1)
    p = torch.randn(4, 3)
    ty = torch.ones(4, dtype=torch.long)
    q_star = perpendicular_foot(o, d, p)
    assert origin_canonical_loss(q_star, o, d, p, ty).item() < 1e-10
    pred = torch.randn(4, 3, requires_grad=True)
    l1 = origin_canonical_loss(pred, o, d, p, ty)
    l2 = origin_canonical_loss(pred, o - 2.2 * d, d, p, ty)
    assert torch.allclose(l1, l2, atol=1e-5)
    l1.backward()  # gradient flows


def test_origin_loss_prismatic_only_batch_is_zero_no_nan():
    pred = torch.randn(3, 3, requires_grad=True)
    ty = torch.zeros(3, dtype=torch.long)
    loss = origin_canonical_loss(
        pred, torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3), ty
    )
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_axis_loss_sign_sensitivity():
    a = torch.nn.functional.normalize(torch.randn(5, 3), dim=1)
    # Antiparallel: perfect under the classical 1-cos^2, ~2 under 1-cos.
    assert axis_direction_loss(-a, a, sign_agnostic=True).item() < 1e-6
    assert abs(axis_direction_loss(-a, a, sign_agnostic=False).item() - 2.0) < 1e-5
    assert axis_direction_loss(a, a, sign_agnostic=False).item() < 1e-6


# ---- PredPredArticulationLoss (Task 4) --------------------------------------

from model.losses.geometric import PredPredArticulationLoss, build_geometric_loss
from model.outputs import ModelOutputs
from model.targets import StepTargets


def _pp_outputs(traj, dhat, p_rev_logit, origin, anchor):
    B = traj.shape[0]
    logits = torch.stack(
        [torch.zeros(B), torch.full((B,), p_rev_logit)], dim=1
    )
    return ModelOutputs(
        mask_logits=torch.zeros(B, 1, 4, 4),
        motion_pred=dhat, motion_type_logits=logits,
        trajectory_pred=traj, origin_pred=origin, point_3d_pred=anchor,
    )


def test_pp_art_reads_no_targets_and_perfect_prismatic_scores_zero():
    B, N = 3, 20
    dhat = torch.nn.functional.normalize(torch.randn(B, 3), dim=1)
    t = torch.linspace(0, 0.4, N)[None, :, None]
    traj = dhat[:, None, :] * t                       # relative, parallel to dhat
    out = _pp_outputs(traj, dhat, p_rev_logit=-20.0,  # P(rev) ~ 0
                      origin=torch.randn(B, 3), anchor=torch.randn(B, 3))
    loss_fn = PredPredArticulationLoss(weight=0.5)
    total, terms = loss_fn(out, StepTargets())        # EMPTY targets: GT-free
    assert terms["L_geo_pred_pred_art"].item() < 1e-8


def _circle_traj(center_rel, dhat, start_rel, thetas):
    # Rotate start_rel about the axis (center_rel, dhat): relative curve.
    r0 = start_rel - center_rel
    axial = (r0 * dhat).sum() * dhat
    x = r0 - axial
    y = torch.cross(dhat, x, dim=-1)
    pts = [center_rel + axial + torch.cos(th) * x + torch.sin(th) * y
           for th in thetas]
    curve = torch.stack(pts)                          # (N, 3), absolute-rel
    return curve - curve[0:1]                         # first point pinned to 0


def test_pp_art_perfect_revolute_scores_zero_and_gauge_along_axis():
    dhat = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=1)
    anchor = torch.tensor([[0.5, 0.2, 1.0]])
    origin = torch.tensor([[0.1, 0.2, 1.0]])          # axis point near anchor
    center_rel = (origin - anchor)[0]
    traj = _circle_traj(center_rel, dhat[0], torch.zeros(3),
                        torch.linspace(0, 1.2, 20))[None]
    out = _pp_outputs(traj, dhat, p_rev_logit=20.0, origin=origin, anchor=anchor)
    loss_fn = PredPredArticulationLoss(weight=0.5)
    _, terms = loss_fn(out, StepTargets())
    assert terms["L_geo_pred_pred_art"].item() < 1e-6
    # Sliding the predicted origin ALONG the predicted axis changes nothing.
    out2 = _pp_outputs(traj, dhat, 20.0, origin + 2.5 * dhat, anchor)
    _, terms2 = loss_fn(out2, StepTargets())
    assert abs(terms2["L_geo_pred_pred_art"].item()
               - terms["L_geo_pred_pred_art"].item()) < 1e-6


def test_pp_art_soft_gate_selects_branch():
    # A straight-line trajectory: near-zero under P(rev)=0, positive under
    # P(rev)=1 (a line is not a constant-radius orbit about a nearby axis).
    dhat = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=1)
    t = torch.linspace(0, 0.4, 20)[None, :, None]
    traj = dhat[:, None, :] * t
    origin = torch.tensor([[0.0, 0.1, 0.0]])
    anchor = torch.zeros(1, 3)
    loss_fn = PredPredArticulationLoss(weight=1.0)
    _, pris = loss_fn(_pp_outputs(traj, dhat, -20.0, origin, anchor), StepTargets())
    _, rev = loss_fn(_pp_outputs(traj, dhat, 20.0, origin, anchor), StepTargets())
    assert pris["L_geo_pred_pred_art"].item() < 1e-8
    assert rev["L_geo_pred_pred_art"].item() > 1e-3


def test_pp_art_degenerate_trajectory_masked():
    dhat = torch.nn.functional.normalize(torch.randn(2, 3), dim=1)
    traj = torch.zeros(2, 20, 3, requires_grad=True)  # zero motion: no direction
    out = _pp_outputs(traj, dhat, 0.0, torch.randn(2, 3), torch.randn(2, 3))
    total, terms = PredPredArticulationLoss(weight=0.5)(out, StepTargets())
    assert terms["L_geo_pred_pred_art"].item() == 0.0
    total.backward()
    assert torch.isfinite(traj.grad).all()


def test_pp_art_missing_heads_noop_and_factory():
    out = ModelOutputs(mask_logits=torch.zeros(1, 1, 4, 4))
    total, terms = PredPredArticulationLoss(weight=0.5)(out, StepTargets())
    assert total.item() == 0.0 and terms == {}

    class LP:  # minimal LossParams stand-in
        geometric_loss = "pred_pred_art"
        pred_pred_art_weight = 0.5
    assert isinstance(build_geometric_loss(LP()), PredPredArticulationLoss)


# ---- trainer wiring (Task 5) -------------------------------------------------

from config.opd_train import Config, LossParams, OptimizerParams
from train_SF3D_better import SF3DTrainingModule


def _split_module():
    lp = LossParams(
        bce_weight=0.5, dice_weight=0.5, mask_weight=0.5,
        point_map_weight=0.0, coord_weight=0.0, vae_weight=0.5,
        motion_type_weight=0.5, point_sigma=8.0, vae_beta=0.01,
        trajectory_weight=0.5, geometric_loss="pred_pred_art",
        pred_pred_art_weight=0.5, axis_sign_agnostic=False,
        origin_weight=0.5, point_3d_weight=0.5,
    )
    op = OptimizerParams(lr=1e-5, weight_decay=0.0,
                         scheduler_milestones=[10], scheduler_gamma=0.1)
    cfg = Config(log_image_interval_steps=0, input_size=[64, 64],
                 enable_wandb=False, val_vis_samples=0, manual_seed=0)
    # Same stub-backbone patch as _make_cris: the module constructor builds a
    # real CRIS, and clip_pretrain="" cannot load.
    with mock.patch(
        "model.segmenter.build_backbone",
        lambda mp, fpn_in: _StubBackbone(fpn_in, word_len=mp.word_len),
    ):
        return SF3DTrainingModule(
            _params(point_prediction_3d=True, use_origin_head=True,
                    pool_with_predicted_mask=True),
            lp, op, cfg,
        )


def _sf3d_batch(B=2, N=20, size=64):
    traj = torch.randn(B, N, 3).cumsum(dim=1) * 0.01 + torch.tensor([0., 0., 1.5])
    return (
        torch.randint(0, 255, (B, 3, size, size), dtype=torch.uint8),
        torch.rand(B, 1, size, size),
        ["open the door"] * B,
        (torch.rand(B, 1, size, size) > 0.5).float(),
        torch.zeros(B, 4),                                  # bbox (unused)
        torch.rand(B, 2),                                   # point_norm
        torch.nn.functional.normalize(torch.randn(B, 3), dim=1),
        torch.tensor([1, 0][:B] if B <= 2 else [1] * B),    # types: rot, trans
        torch.tensor([[64, 64]] * B),
        ["f.png"] * B,
        torch.randn(B, 3) + torch.tensor([0., 0., 1.5]),    # origin_3d
        torch.eye(3).expand(B, 3, 3),
        traj,
    )


def test_common_step_split_arm_finite_and_logs_new_terms():
    m = _split_module()
    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(
        name, float(value) if torch.is_tensor(value) else value
    )
    loss = m._common_step(_sf3d_batch(), 0, "train")
    assert torch.isfinite(loss)
    loss.backward()
    for key in ("train/L_point_3d", "train/L_origin",
                "train/L_geo_pred_pred_art", "train/L_motion_type"):
        assert key in logged, key
    assert logged["train/L_point_3d"] > 0.0
    # 2D point losses are zero-valued on the 3D path (stable CSV columns).
    assert logged["train/L_point_map"] == 0.0
    assert logged["train/L_coord"] == 0.0


def test_sf3d_test_step_split_arm_accumulates_3d_metrics():
    m = _split_module()
    m.eval()
    m.log = lambda *a, **k: None
    m.on_test_start()
    with torch.no_grad():
        m.test_step(_sf3d_batch(), 0)
    # One revolute + one prismatic sample in the batch.
    assert len(m._test_point3d_errors) == 2
    assert len(m._test_origin_err_m) == 1          # revolute row only
    assert len(m._test_origin_line_err_m) == 1
    assert len(m._test_radius_err_m) == 1
    assert all(np.isfinite(v) for v in m._test_point3d_errors)
