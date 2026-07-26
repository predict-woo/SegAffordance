"""Geometric consistency losses between the motion-axis and trajectory heads.

Two schemes, selected by ``LossParams.geometric_loss``:

``cross_gt`` (default, the historical behaviour)
    Two terms, each pairing one prediction against the *other* head's ground
    truth. Branches on GT motion type: a line loss for prismatic, a circle
    loss for revolute, with the circle's centre and radius supplied by GT.
    Nothing couples the two predictions, so the heads are never made
    self-consistent at inference.

``pred_pred``
    One symmetric term coupling ``motion_pred`` directly to
    ``trajectory_pred``. See
    ``docs/superpowers/specs/2026-07-26-pred-pred-geometric-loss-design.md``.

All variants share one interface so the training step can swap them without
knowing which is active: ``forward`` returns ``(weighted total, unweighted
named terms)``. The weights live inside the module, so there is a single
source of truth, while the logged values stay comparable across weight
changes.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.outputs import ModelOutputs
from model.targets import StepTargets

LossTerms = Tuple[torch.Tensor, Dict[str, torch.Tensor]]


class GeometricConsistencyLoss(nn.Module):
    """Base class. Subclasses return (weighted total, unweighted terms)."""

    def forward(self, outputs: ModelOutputs, targets: StepTargets) -> LossTerms:
        raise NotImplementedError

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=reference.device, dtype=torch.float32)


class NoGeometricLoss(GeometricConsistencyLoss):
    """Contributes nothing. Used on OPD and as the ablation arm."""

    def forward(self, outputs: ModelOutputs, targets: StepTargets) -> LossTerms:
        return self._zero(outputs.motion_pred), {}


class PredPredGeometricLoss(GeometricConsistencyLoss):
    """Symmetric prediction-to-prediction consistency.

    ``trajectory_pred`` is relative to its own first point, so the line
    (prismatic) or plane (revolute) that the motion should lie on passes
    through the origin — no motion origin is needed, and the uncentered
    scatter matrix is exact rather than an approximation.

    With ``n̂`` the normalised axis, ``dᵢ`` the trajectory displacements and
    ``M = Σᵢ dᵢdᵢᵀ``::

        R = n̂ᵀMn̂ / tr(M)                    ∈ [0, 1]
        p = softmax(motion_type_logits)[:, 1]   = P(revolute)
        L = (1 − p)(1 − R) + p·R

    ``R`` is the fraction of the trajectory's motion energy lying along the
    axis: prismatic wants 1, revolute wants 0. Equivalently ``L = ½(1 + uv)``
    for ``u = 2p − 1`` and ``v = 2R − 1``, which makes the symmetry between
    the two heads explicit and shows why the gradient into each is
    proportional to the other's confidence — the coupling switches itself on
    only as the type head sharpens.
    """

    def __init__(self, weight: float, degenerate_threshold: float = 1e-6):
        super().__init__()
        self.weight = weight
        # A trajectory with no extent has no direction to be consistent with.
        # Scoring it anyway would give R=0 and hence L=(1-p), which is not
        # neutral: it pushes p towards 1, a spurious "everything is revolute"
        # signal. Such samples are dropped from the batch mean instead.
        self.degenerate_threshold = degenerate_threshold

    def forward(self, outputs: ModelOutputs, targets: StepTargets) -> LossTerms:
        if targets.trajectory is None:
            return self._zero(outputs.motion_pred), {}

        # fp32 throughout: runs train at precision 16, and R is a ratio of
        # sums of squares that should not be formed under autocast.
        d = outputs.trajectory_pred.float()
        axis = F.normalize(outputs.motion_pred.float(), p=2, dim=1, eps=1e-8)

        scatter = torch.einsum("bni,bnj->bij", d, d)
        along = torch.einsum("bi,bij,bj->b", axis, scatter, axis)
        total_energy = scatter.diagonal(dim1=1, dim2=2).sum(-1)

        valid = total_energy > self.degenerate_threshold
        # clamp keeps the division finite for the degenerate rows that `valid`
        # is about to discard; without it they would be 0/0 and poison backward.
        ratio = (along / total_energy.clamp(min=self.degenerate_threshold)).clamp(0.0, 1.0)

        p_revolute = outputs.motion_type_logits.float().softmax(dim=-1)[:, 1]
        per_sample = (1.0 - p_revolute) * (1.0 - ratio) + p_revolute * ratio

        if bool(valid.any()):
            term = per_sample[valid].mean()
        else:
            term = self._zero(outputs.motion_pred)

        # The key is emitted whenever the dataset has trajectories at all, so
        # the CSV logger sees a stable set of columns across steps.
        return self.weight * term, {"L_geo_pred_pred": term}


class CrossGTGeometricLoss(GeometricConsistencyLoss):
    """The historical scheme: each prediction against the other head's GT.

    Kept numerically identical to the pre-refactor implementation — the
    per-sample Python loop and its float associativity included, since
    ``tests/test_geometric_losses.py`` pins the numbers against a frozen copy
    of the original.
    """

    def __init__(self, geometric_weight: float, trajectory_to_motion_weight: float):
        super().__init__()
        self.geometric_weight = geometric_weight
        self.trajectory_to_motion_weight = trajectory_to_motion_weight

    def forward(self, outputs: ModelOutputs, targets: StepTargets) -> LossTerms:
        if targets.trajectory is None or targets.motion_origin_3d is None:
            return self._zero(outputs.motion_pred), {}

        device = outputs.trajectory_pred.device
        trajectory_gt = targets.trajectory.to(device)
        trajectory_gt_first = trajectory_gt[:, 0:1, :]
        trajectory_gt_relative = trajectory_gt - trajectory_gt_first
        motion_origin_relative = targets.motion_origin_3d.to(device) - trajectory_gt_first.squeeze(1)
        origin_of_relative_frame = torch.zeros_like(motion_origin_relative)
        motion_type = targets.motion_type.to(device)

        pred_vector_gt_traj = self._consistency(
            trajectory=trajectory_gt_relative,
            axis=outputs.motion_pred,
            motion_type_gt=motion_type,
            motion_origin_3d=motion_origin_relative,
            trajectory_gt_first=origin_of_relative_frame,
        )
        pred_traj_gt_vector = self._consistency(
            trajectory=outputs.trajectory_pred,
            axis=targets.motion.to(device),
            motion_type_gt=motion_type,
            motion_origin_3d=motion_origin_relative,
            trajectory_gt_first=origin_of_relative_frame,
        )

        total = (
            self.geometric_weight * pred_vector_gt_traj
            + self.trajectory_to_motion_weight * pred_traj_gt_vector
        )
        return total, {
            "L_geometric_pred_vector_gt_traj": pred_vector_gt_traj,
            "L_geometric_pred_traj_gt_vector": pred_traj_gt_vector,
        }

    @staticmethod
    def _consistency(
        trajectory: torch.Tensor,      # (B, N, 3)
        axis: torch.Tensor,            # (B, 3)
        motion_type_gt: torch.Tensor,  # (B,)
        motion_origin_3d: torch.Tensor,  # (B, 3)
        trajectory_gt_first: torch.Tensor,  # (B, 3)
    ) -> torch.Tensor:
        """Line loss for prismatic, circle loss for revolute.

        Moved verbatim from ``OPDRealTrainingModule._geometric_consistency_loss``.
        """
        B, N, _ = trajectory.shape
        device = trajectory.device
        axis_norm = F.normalize(axis, p=2, dim=1, eps=1e-8)

        total_loss = torch.zeros(B, device=device)

        for b in range(B):
            if motion_type_gt[b] == 0:  # translation — distance to a line
                P_0 = motion_origin_3d[b]
                v = axis_norm[b]
                Q = trajectory[b]
                cross_product = torch.cross(Q - P_0, v.unsqueeze(0).expand(N, -1), dim=-1)
                total_loss[b] = torch.sum(cross_product**2, dim=1).mean()

            elif motion_type_gt[b] == 1:  # rotation — distance to a circle
                C = motion_origin_3d[b]
                n = axis_norm[b]
                Q = trajectory[b]
                first_minus_C = trajectory_gt_first[b] - C
                proj_perp = first_minus_C - torch.dot(first_minus_C, n) * n
                r = torch.norm(proj_perp)

                Q_minus_C = Q - C
                dot_n = torch.sum(Q_minus_C * n.unsqueeze(0).expand(N, -1), dim=1)
                plane_dist_sq = dot_n**2
                proj_perp_vecs = Q_minus_C - dot_n.unsqueeze(1) * n.unsqueeze(0).expand(N, -1)
                circle_error_sq = (torch.norm(proj_perp_vecs, dim=1) - r) ** 2
                total_loss[b] = (plane_dist_sq + circle_error_sq).mean()

        return total_loss.mean()


def build_geometric_loss(loss_params) -> GeometricConsistencyLoss:
    """Select a variant from ``LossParams.geometric_loss``.

    Defaults to ``cross_gt`` so that configs written before this option
    existed keep their exact behaviour.
    """
    name = getattr(loss_params, "geometric_loss", "cross_gt")

    if name == "cross_gt":
        return CrossGTGeometricLoss(
            geometric_weight=getattr(loss_params, "geometric_weight", 1.0),
            trajectory_to_motion_weight=getattr(
                loss_params, "trajectory_to_motion_weight", 1.0
            ),
        )
    if name == "pred_pred":
        return PredPredGeometricLoss(
            weight=getattr(loss_params, "pred_pred_weight", 0.1)
        )
    if name == "none":
        return NoGeometricLoss()

    raise ValueError(
        f"unknown geometric_loss {name!r} — expected 'cross_gt', 'pred_pred' or 'none'"
    )
