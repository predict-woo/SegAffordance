"""Articulation readout (2026-09-12): learned-query attention over the decoded
map in place of mask-mean pooling (attnpool) or with one query per head (query)."""
import dataclasses

import pytest
import torch

from config.opd_train import ModelParams
from model.layers import ArticulationReadout, sine_positions_2d
from test_analytic_decoder import _decoder_model
from test_split_heads import _inputs


def test_defaults_are_the_classical_readout():
    d = {f.name: f.default for f in dataclasses.fields(ModelParams)}
    assert d["articulation_readout"] == "mlp"
    assert d["readout_queries"] == 4 and d["readout_layers"] == 2
    assert d["readout_mask_eps"] == 0.01


def test_sine_positions_shape_and_distinctness():
    pos = sine_positions_2d(4, 6, 32, "cpu", torch.float32)
    assert pos.shape == (24, 32)
    assert torch.unique(pos, dim=0).shape[0] == 24         # every cell gets its own code


def test_attnpool_ignores_tokens_outside_the_mask_at_a_tiny_eps():
    torch.manual_seed(0)
    ro = ArticulationReadout(d_model=32, mode="attnpool", nhead=4, mask_eps=1e-9).eval()
    fq = torch.randn(2, 32, 5, 5)
    mask = torch.zeros(2, 1, 5, 5)
    mask[:, :, 1:3, 1:3] = 1.0
    out = ro(fq, mask)
    assert out.shape == (2, 1, 32)
    fq2 = fq.clone()
    fq2[:, :, 3:, :] = torch.randn_like(fq2[:, :, 3:, :]) * 10   # perturb only outside the part
    fq2[:, :, :, 3:] = torch.randn_like(fq2[:, :, :, 3:]) * 10
    assert torch.allclose(out, ro(fq2, mask), atol=1e-4)
    # a soft mask (eps 0.5) lets the outside through
    ro_soft = ArticulationReadout(d_model=32, mode="attnpool", nhead=4, mask_eps=0.5).eval()
    ro_soft.load_state_dict(ro.state_dict())
    assert not torch.allclose(ro_soft(fq, mask), ro_soft(fq2, mask), atol=1e-3)


def test_query_mode_shapes_and_gradients():
    ro = ArticulationReadout(d_model=32, mode="query", num_queries=4, num_layers=2, nhead=4, dim_ffn=64)
    fq = torch.randn(3, 32, 6, 4, requires_grad=True)
    mask = (torch.rand(3, 1, 6, 4) > 0.5).float()
    out = ro(fq, mask)
    assert out.shape == (3, 4, 32)
    out.sum().backward()
    assert ro.queries.grad is not None and ro.queries.grad.abs().sum() > 0
    assert fq.grad is not None and fq.grad.abs().sum() > 0
    # the queries differ from each other (no collapse at init)
    assert not torch.allclose(out[:, 0], out[:, 1])


def test_attnpool_is_a_single_query():
    ro = ArticulationReadout(d_model=16, mode="attnpool", num_queries=7, num_layers=3, nhead=2)
    assert ro.num_queries == 1 and not hasattr(ro, "layers")


@pytest.mark.parametrize("mode", ["attnpool", "query"])
def test_cris_runs_with_the_readout_in_train_and_eval(mode):
    m = _decoder_model(articulation_readout=mode, readout_queries=4, readout_layers=1, readout_dim_ffn=32)
    assert m.readout is not None
    img, depth, word, mask = _inputs(B=2, size=64)
    K = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    m.train()
    out = m(img, depth, word, mask, None, None, None, K)
    assert out.motion_pred_rot.shape == (2, 3) and out.motion_type_logits.shape == (2, 2)
    assert out.trajectory_length.shape == (2,) and out.trajectory_pred.shape == (2, 20, 3)
    loss = out.mask_logits.mean() + out.motion_pred_rot.sum() + out.trajectory_length.sum() + out.origin_pred.sum()
    loss.backward()
    assert m.readout.queries.grad is not None
    m.eval()
    with torch.no_grad():
        out = m(img, depth, word, mask, None, None, None, K)
    assert torch.isfinite(out.trajectory_pred).all()


def test_query_mode_feeds_each_head_its_own_query():
    m = _decoder_model(articulation_readout="query", readout_queries=4, readout_layers=1, readout_dim_ffn=32)
    m.eval()
    img, depth, word, mask = _inputs(B=2, size=64)
    K = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    seen = {}
    for name, mod in [("axis", m.motion_mlp), ("len", m.trajectory_length_head), ("zq", m.origin_depth_head_g7)]:
        orig = mod.forward

        def spy(cond, _o=orig, _n=name):
            seen[_n] = cond.detach().clone()
            return _o(cond)

        mod.forward = spy
    with torch.no_grad():
        m(img, depth, word, mask, None, None, None, K)
    D = m.readout_dim
    n = seen["axis"].shape[1]
    # pooled slot differs per head (queries 0 / 3 / 2) ...
    assert not torch.allclose(seen["axis"][:, :D], seen["len"][:, :D])
    assert not torch.allclose(seen["axis"][:, :D], seen["zq"][:, :D])
    # ... while the shared globals (global token, text state, uv's) are identical
    assert torch.allclose(seen["axis"][:, D:], seen["len"][:, D:])
    assert torch.allclose(seen["axis"][:, D:], seen["zq"][:, D:n])


def test_dense_head_votes_are_the_masked_means():
    from model.layers import DenseArticulationHead
    torch.manual_seed(0)
    head = DenseArticulationHead(in_dim=32, hidden=32).eval()
    with torch.no_grad():
        head.net[-1].weight[8:].normal_(); head.net[-1].bias[8:].normal_()   # non-zero offsets
    fq = torch.randn(2, 32, 4, 6)
    mask = torch.zeros(2, 1, 4, 6)
    mask[0, 0, 1, 2] = 1.0                       # one pixel -> the vote IS that pixel's
    mask[1, 0, :, :] = 1.0                       # all pixels -> plain mean
    out = head(fq, mask)
    assert out["rot"].shape == (2, 3) and out["trans"].shape == (2, 3)
    assert out["type_logits"].shape == (2, 2) and out["origin_uv"].shape == (2, 2)
    assert out["rot"].abs().max() < 1.0
    u, v = (2 + 0.5) / 6, (1 + 0.5) / 4
    expect = torch.tensor([u, v]) + out["offset_field"][0, :, 1, 2]
    assert torch.allclose(out["origin_uv"][0], expect, atol=1e-5)
    assert torch.allclose(out["origin_uv"][1], out["vote_uv"][1].mean(dim=(-1, -2)), atol=1e-5)


def test_dense_mode_in_cris_votes_the_origin_and_axes():
    m = _decoder_model(articulation_readout="dense", dense_hidden=32)
    assert m.dense_head is not None and m.motion_mlp is None and m.readout is None
    img, depth, word, mask = _inputs(B=2, size=64)
    K = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    m.train()
    out = m(img, depth, word, mask, None, None, None, K)
    assert out.motion_pred_rot.shape == (2, 3) and out.motion_type_logits.shape == (2, 2)
    assert out.origin_uv.shape == (2, 2) and out.origin_logits is not None      # heatmap kept as aux
    assert out.trajectory_pred.shape == (2, 20, 3)
    (out.origin_pred.sum() + out.motion_pred_rot.sum() + out.trajectory_pred.sum()).backward()
    assert m.dense_head.net[-1].weight.grad.abs().sum() > 0
    m.eval()
    with torch.no_grad():
        out2 = m(img, depth, word, mask, None, None, None, K)
    assert torch.isfinite(out2.origin_pred).all()


def test_query_pos_starts_identical_and_then_depends_on_location():
    from model.layers import sine_embed_uv
    torch.manual_seed(0)
    base = ArticulationReadout(d_model=32, mode="query", num_queries=4, num_layers=1, nhead=4, dim_ffn=64).eval()
    pos = ArticulationReadout(d_model=32, mode="query", num_queries=4, num_layers=1, nhead=4, dim_ffn=64, query_pos=True).eval()
    pos.load_state_dict(base.state_dict(), strict=False)
    fq = torch.randn(2, 32, 6, 8)
    mask = torch.ones(2, 1, 6, 8)
    uv = torch.rand(2, 4, 2)
    # zero-initialised projection: identical to the unconditioned readout at init
    assert torch.allclose(pos(fq, mask, uv), base(fq, mask), atol=1e-6)
    with torch.no_grad():
        pos.pos_proj.weight.normal_(); pos.pos_proj.bias.normal_()
    assert not torch.allclose(pos(fq, mask, uv), pos(fq, mask, torch.rand(2, 4, 2)), atol=1e-4)
    # the query code at a cell centre equals that cell's key position code
    from model.layers import sine_positions_2d
    code = sine_embed_uv(torch.tensor([[[(3 + 0.5) / 8, (2 + 0.5) / 6]]]), 6, 8, 32)[0, 0]
    assert torch.allclose(code, sine_positions_2d(6, 8, 32, "cpu", torch.float32)[2 * 8 + 3], atol=1e-5)


def test_cris_query_pos_without_grid_samples():
    m = _decoder_model(articulation_readout="query", readout_queries=4, readout_layers=1, readout_dim_ffn=32,
                       readout_query_pos=True, depth_local_sample=False)
    assert m.readout.pos_proj is not None and not m.depth_local_sample and not m.use_origin_local_feature
    # depth heads read only their condition vector now
    assert m.point_depth_head.mlp[0].in_features == m.origin_depth_head_g7.mlp[0].in_features
    img, depth, word, mask = _inputs(B=2, size=64)
    K = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    m.train()
    out = m(img, depth, word, mask, None, None, None, K)
    (out.origin_pred.sum() + out.point_3d_pred.sum() + out.motion_pred_rot.sum()).backward()
    assert m.readout.queries.grad is not None
    m.eval()
    with torch.no_grad():
        out = m(img, depth, word, mask, None, None, None, K)
    assert torch.isfinite(out.trajectory_pred).all() and out.trajectory_length.shape == (2,)


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        _decoder_model(articulation_readout="voting")
    with pytest.raises(ValueError):
        ArticulationReadout(d_model=16, mode="query", mask_eps=0.0)
