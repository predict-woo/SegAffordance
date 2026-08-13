# tests/test_split_heads.py
from unittest import mock

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


def test_classical_2d_mode_unchanged():
    m = _make_cris()  # all new flags default off
    m.eval()
    img, depth, word, mask = _inputs()
    with torch.no_grad():
        out = m(img, depth, word, mask, None, None)
    assert out.point_logits is not None and out.coords_hat.shape == (2, 2)
    assert out.point_3d_pred is None and out.origin_pred is None
