"""Structured return type for :class:`model.segmenter.CRIS`.

The forward pass used to hand back an 8-tuple with two ``None`` slots, which
every consumer unpacked positionally — including ``tools/smoke_backbone.py``,
which kept a hand-written list of field names alongside it. Loss modules need
to pick out the subset of outputs they care about by name, so the tuple became
a dataclass.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelOutputs:
    """Everything a forward pass produces.

    Shapes assume an input of (B, 3, H, W); the map resolution is H/4 × W/4.
    """

    #: (B, 1, H/4, W/4) segmentation logits
    mask_logits: torch.Tensor
    #: (B, 1, H/4, W/4) interaction-point heatmap logits
    point_logits: torch.Tensor
    #: (B, 2) soft-argmax of point_logits, normalised to [0, 1]
    coords_hat: torch.Tensor
    #: (B, 3) motion axis, unnormalised and sign-agnostic
    motion_pred: torch.Tensor
    #: (B, num_motion_types) — class 1 is rotation/revolute
    motion_type_logits: torch.Tensor
    #: (B, num_points, 3) trajectory, relative to its own first point
    trajectory_pred: torch.Tensor
    #: (B, latent_dim) CVAE posterior mean — only when motion_gt was supplied
    mu: Optional[torch.Tensor] = None
    #: (B, latent_dim) CVAE posterior log-variance — likewise
    log_var: Optional[torch.Tensor] = None
