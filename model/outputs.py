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
    #: (B, 1, H/4, W/4) interaction-point heatmap logits. None when the 2D
    #: point pathway is replaced by the 3D one (ModelParams.point_prediction_3d).
    point_logits: Optional[torch.Tensor] = None
    #: (B, 2) soft-argmax of point_logits, normalised to [0, 1]. None when
    #: point_logits is None.
    point_uv: Optional[torch.Tensor] = None
    #: (B, 3) motion axis, unnormalised and sign-agnostic. None when the
    #: axis head is off (`use_motion_head: false`, twist arms) — consumers
    #: fall back to the twist's decoded direction.
    motion_pred: Optional[torch.Tensor] = None
    #: (B, num_motion_types) — class 1 is rotation/revolute. None when the
    #: type head is off (`use_motion_type_head: false`); consumers fall back
    #: to the twist's |omega| where a type is needed.
    motion_type_logits: Optional[torch.Tensor] = None
    #: (B, num_points, 3) trajectory, relative to its own first point.
    #: The element's swept path. None when the head is disabled
    #: (`use_trajectory_head: false`), as in 2D pretraining where no
    #: element-sweep ground truth exists.
    trajectory_pred: Optional[torch.Tensor] = None
    #: (B, num_points, 2) 2D track in normalised [0, 1] coords, relative to its
    #: own first point. The hand/contact path — a different quantity from
    #: `trajectory_pred`. None unless `use_2d_trajectory_head`.
    trajectory_2d_pred: Optional[torch.Tensor] = None
    #: (B,) metric depth of the 3D joint origin. Combined with `point_uv` and
    #: the intrinsics this gives the 3D origin the model otherwise never
    #: predicts. None unless `predict_origin_depth`.
    origin_depth: Optional[torch.Tensor] = None
    #: (B, 3) absolute 3D interaction point, camera frame, metres — the
    #: split-arm replacement for point_logits/point_uv
    #: (ModelParams.point_prediction_3d). None on the classical 2D path.
    point_3d_pred: Optional[torch.Tensor] = None
    #: (B, 3) absolute 3D joint origin, camera frame, metres — supervised
    #: toward q*, the GT-axis point perpendicular to the interaction point
    #: (ModelParams.use_origin_head). Meaningful for revolute only.
    origin_pred: Optional[torch.Tensor] = None
    #: (B, 2) soft-argmax of the origin heatmap channel, normalised to
    #: [0, 1] (ModelParams.use_origin_heatmap). The 2D face of origin_pred.
    origin_uv: Optional[torch.Tensor] = None
    #: (B, 6) se(3) twist (omega, v) in the camera frame — one construct for
    #: both motion types (revolute: |omega|=1 and v encodes the axis LINE;
    #: prismatic: omega=0 and v is the direction). See model/losses/twist.py.
    #: None unless `use_twist_head`.
    twist_pred: Optional[torch.Tensor] = None
    #: (B, K, 6) all K WTA articulation hypotheses' twists; None when
    #: `twist_num_hypotheses` is 1. `twist_pred` is then the argmax-logit
    #: selection out of these.
    twist_hyps: Optional[torch.Tensor] = None
    #: (B, K, num_points, 3) the bundled trajectory of each hypothesis
    #: (relative to its own first point); `trajectory_pred` is the same
    #: argmax-logit selection. None when K = 1.
    trajectory_hyps: Optional[torch.Tensor] = None
    #: (B, K) bundle-selection logits from the twist head; argmax picks the
    #: (twist, trajectory) bundle reported in `twist_pred`/`trajectory_pred`.
    #: None when K = 1.
    twist_logits: Optional[torch.Tensor] = None
    #: (B, latent_dim) CVAE posterior mean — only when motion_gt was supplied
    mu: Optional[torch.Tensor] = None
    #: (B, latent_dim) CVAE posterior log-variance — likewise
    log_var: Optional[torch.Tensor] = None
