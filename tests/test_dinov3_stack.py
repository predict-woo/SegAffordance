import torch

from model.backbones.pyramid import MultiTapPyramid, SimpleFeaturePyramid


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
