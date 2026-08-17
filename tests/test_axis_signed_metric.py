"""Sign-aware axis metric (2026-08-18).

The legacy _axis_error_deg takes |cos| and scores a perfectly flipped axis
as ~0 deg; the signed=True mode reports the true angle so flipped opening
directions become visible in the test tables.
"""
import math

import torch

from train_OPDReal_better import OPDRealTrainingModule

err = OPDRealTrainingModule._axis_error_deg


def test_default_is_legacy_unsigned():
    a = torch.tensor([0.0, 1.0, 0.0])
    assert err(a, -a).item() < 1e-3          # flip invisible (legacy)
    assert err(a, a).item() < 1e-3


def test_signed_sees_the_flip():
    a = torch.tensor([0.0, 1.0, 0.0])
    assert abs(err(a, -a, signed=True).item() - 180.0) < 1e-3
    assert err(a, a, signed=True).item() < 1e-3


def test_signed_equals_unsigned_below_90():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 1.0, 0.0])        # 45 deg apart
    s = err(a, b, signed=True).item()
    u = err(a, b).item()
    assert abs(s - 45.0) < 1e-3 and abs(u - 45.0) < 1e-3


def test_orthogonal_is_90_both_ways():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 1.0])
    assert abs(err(a, b).item() - 90.0) < 1e-3
    assert abs(err(a, b, signed=True).item() - 90.0) < 1e-3


def test_signed_normalizes_magnitude():
    a = torch.tensor([0.0, 0.3, 0.0])        # unnormalized inputs
    b = torch.tensor([0.0, -7.0, 0.0])
    assert abs(err(a, b, signed=True).item() - 180.0) < 1e-3


def test_obtuse_angle_signed_vs_unsigned():
    a = torch.tensor([1.0, 0.0, 0.0])
    c, s = math.cos(math.radians(135)), math.sin(math.radians(135))
    b = torch.tensor([c, s, 0.0])            # 135 deg apart
    assert abs(err(a, b, signed=True).item() - 135.0) < 1e-2
    assert abs(err(a, b).item() - 45.0) < 1e-2   # legacy folds to 45
