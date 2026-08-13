# tests/test_split_heads.py
import torch

from model.layers import Point3DHead
from model.outputs import ModelOutputs


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
