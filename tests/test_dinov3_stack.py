from unittest import mock

import pytest
import torch

from config.opd_train import Config, LossParams, OptimizerParams
from model.backbones.base import BackboneBase
from model.backbones.pyramid import MultiTapPyramid, SimpleFeaturePyramid
from tests.test_origin_local_sample import _forward_with_K
from tests.test_split_heads import _StubBackbone, _params
from train_SF3D_better import SF3DTrainingModule


def test_multitap_pyramid_shapes():
    torch.manual_seed(0)
    fpn_in = [512, 1024, 1024]
    pyr = MultiTapPyramid(in_dim=64, out_channels=fpn_in)
    taps = [torch.randn(2, 64, 16, 16) for _ in range(4)]
    v8, v16, v32 = pyr(taps)
    assert v8.shape == (2, 512, 32, 32)
    assert v16.shape == (2, 1024, 16, 16)
    assert v32.shape == (2, 1024, 8, 8)


def test_multitap_pyramid_x_deep_replaces_deep_source():
    torch.manual_seed(0)
    pyr = MultiTapPyramid(in_dim=8, out_channels=[16, 16, 16])
    taps = [torch.randn(1, 8, 4, 4) for _ in range(4)]
    x_deep = torch.randn(1, 8, 4, 4)
    _, _, a = pyr(taps)
    _, _, b = pyr(taps, x_deep=x_deep)
    assert not torch.allclose(a, b)          # deep level follows x_deep
    v8a, v16a, _ = pyr(taps)
    v8b, v16b, _ = pyr(taps, x_deep=x_deep)
    assert torch.equal(v8a, v8b) and torch.equal(v16a, v16b)  # others don't


def test_modelparams_has_taps_flag_default_false():
    import dataclasses
    from config.opd_train import ModelParams
    fields = {f.name: f for f in dataclasses.fields(ModelParams)}
    assert fields["dinov3_multilayer_taps"].default is False


# ---- Task 2: patch-text cost map + gate subspace fix (`text_cost_map`) -----
#
# Same stub-backbone construction as tests/test_split_heads.py /
# tests/test_origin_local_sample.py: a small CPU CRIS inside an
# SF3DTrainingModule, backbone patched in via model.segmenter.build_backbone.


class _CostMapStubBackbone(_StubBackbone):
    """Stub advertising the dinotxt contract: the state splits into
    [global-half; local-half] (2048-style, here 64), and encode_image_full
    returns the text-aligned /16 patch map at HALF width (here 32)."""

    def __init__(self, fpn_in, word_len):
        super().__init__(fpn_in, word_len, word_dim=64, state_dim=64)

    def encode_image_full(self, img):
        maps = self.encode_image(img)
        b, _, h, w = img.shape
        aligned = torch.randn(b, self.state_dim // 2, h // 16, w // 16)
        return maps, {"aligned_map": aligned}


def _module_pieces():
    lp = LossParams(
        bce_weight=0.5, dice_weight=0.5, mask_weight=0.5,
        point_map_weight=0.5, coord_weight=0.5, vae_weight=0.5,
        motion_type_weight=0.5, point_sigma=8.0, vae_beta=0.01,
        trajectory_weight=0.5, geometric_loss="pred_pred_art",
        pred_pred_art_weight=0.5, axis_sign_agnostic=False,
        origin_weight=0.5, point_3d_weight=0.5,
    )
    op = OptimizerParams(lr=1e-5, weight_decay=0.0,
                         scheduler_milestones=[10], scheduler_gamma=0.1)
    cfg = Config(log_image_interval_steps=0, input_size=[64, 64],
                 enable_wandb=False, val_vis_samples=0, manual_seed=0)
    return lp, op, cfg


def _build_module_flags(**flag_overrides):
    # ModelParams defaults apply (backbone="clip_rn50"); the stub stands in
    # for whatever backbone at construction time.
    lp, op, cfg = _module_pieces()
    with mock.patch(
        "model.segmenter.build_backbone",
        lambda mp, fpn_in: _StubBackbone(fpn_in, word_len=mp.word_len),
    ):
        return SF3DTrainingModule(_params(**flag_overrides), lp, op, cfg)


def _build_costmap_module():
    lp, op, cfg = _module_pieces()
    params = _params(backbone="dinov3", text_source="dinotxt",
                     text_cost_map=True)
    with mock.patch(
        "model.segmenter.build_backbone",
        lambda mp, fpn_in: _CostMapStubBackbone(fpn_in, word_len=mp.word_len),
    ):
        return SF3DTrainingModule(params, lp, op, cfg)


def test_encode_image_full_default_empty_extras():
    # Any backbone without an override returns (maps, {}).
    class _B(BackboneBase):
        def tokenize(self, texts, l): ...
        def encode_text(self, tokens): ...
        def encode_image(self, img):
            return "MAPS"
    maps, extras = _B().encode_image_full("img")
    assert maps == "MAPS" and extras == {}


def test_text_cost_map_requires_dinotxt():
    with pytest.raises(ValueError):
        _build_module_flags(text_cost_map=True)   # default backbone=clip_rn50


def test_text_cost_map_forward_shapes_and_gate_half():
    # Stub backbone advertising the dinotxt contract: state_dim 64,
    # aligned_map channels 32 (= half), extras wired through.
    module = _build_costmap_module()
    outputs = _forward_with_K(module)
    assert outputs.mask_logits is not None        # end-to-end runs
    # FPN was built with +2 in_channels and half text_dim:
    neck = module.model.neck
    assert neck.txt_proj[0].in_features == 32     # half of the stub's 64


def test_flag_off_fpn_dims_unchanged():
    module = _build_module_flags()                # all flags off, stub clip-style
    neck = module.model.neck
    assert neck.txt_proj[0].in_features == module.model.backbone.state_dim
