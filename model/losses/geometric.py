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

import math
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


def normalized_intrinsics(K: torch.Tensor, img_size: torch.Tensor) -> torch.Tensor:
    """K at original resolution -> K projecting camera 3D into [0, 1] coords.

    The dataset stores intrinsics for the ORIGINAL frame while every 2D
    quantity in the model (``coords_hat``, the 2D tracks) is normalised to
    [0, 1]. Folding the resolution into K once keeps the loss independent of
    whatever size the images were resized to. ``img_size`` is (width, height),
    matching ``datasets/scenefun3d.py:235``.
    """
    K_norm = K.clone().float()
    width = img_size[:, 0].to(K_norm.dtype).clamp(min=1.0).unsqueeze(-1)
    height = img_size[:, 1].to(K_norm.dtype).clamp(min=1.0).unsqueeze(-1)
    K_norm[:, 0, :] = K_norm[:, 0, :] / width
    K_norm[:, 1, :] = K_norm[:, 1, :] / height
    return K_norm


def project_points(K_norm: torch.Tensor, points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(B, M, 3) camera-frame points -> (B, M, 2) normalised image coords."""
    homogeneous = torch.einsum("bij,bmj->bmi", K_norm, points)
    depth = homogeneous[..., 2:3].clamp(min=eps)  # keeps behind-camera finite
    return homogeneous[..., :2] / depth


def backproject_points(K_norm: torch.Tensor, uv: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """(B, 2) normalised coords + (B,) depth -> (B, 3) camera-frame points."""
    ones = torch.ones_like(uv[..., :1])
    rays = torch.einsum(
        "bij,bj->bi", torch.linalg.inv(K_norm), torch.cat([uv, ones], dim=-1)
    )
    return rays * depth.unsqueeze(-1)


def _point_to_polyline_distance(points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
    """(B, N, 2) points, (B, M, 2) polyline -> (B, N) distance to the polyline.

    Distance to the *segments*, not to the sampled vertices: sampling a conic
    at M points and measuring to the vertices would put a floor of half the
    vertex spacing under the residual.
    """
    start = polyline[:, :-1].unsqueeze(1)              # (B, 1, M-1, 2)
    end = polyline[:, 1:].unsqueeze(1)                 # (B, 1, M-1, 2)
    seg = end - start
    rel = points.unsqueeze(2) - start                  # (B, N, M-1, 2)
    t = (rel * seg).sum(-1) / (seg * seg).sum(-1).clamp(min=1e-12)
    closest = start + t.clamp(0.0, 1.0).unsqueeze(-1) * seg
    return (points.unsqueeze(2) - closest).norm(dim=-1).min(dim=-1).values


def _rotate_about_axis(axis: torch.Tensor, vec: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """Rodrigues. axis/vec (B, 3), thetas (M,) -> (B, M, 3)."""
    cos = torch.cos(thetas).view(1, -1, 1)
    sin = torch.sin(thetas).view(1, -1, 1)
    axis_cross_vec = torch.linalg.cross(axis, vec, dim=-1).unsqueeze(1)
    axis_dot_vec = (axis * vec).sum(-1, keepdim=True).unsqueeze(1)
    return (
        vec.unsqueeze(1) * cos
        + axis_cross_vec * sin
        + axis.unsqueeze(1) * axis_dot_vec * (1.0 - cos)
    )


class ProjectedGeometricLoss(GeometricConsistencyLoss):
    """2D-pretraining alignment: does the predicted articulation project onto
    the observed 2D track?

    The articulation ``{type, axis, origin}`` defines a 1-parameter 3D curve
    through the track's own first point. Project it and measure how far each
    observed point falls from that curve. Nothing is fitted to the data, so a
    short low-curvature arc yields a *weak* gradient rather than a wrong one.

    Anchoring on the track's own start is what makes this immune to the
    hand-vs-element mismatch: the curve automatically carries the hand's
    radius, and ``n · d = 0`` holds for every point of a rigid body, so a hand
    arc constrains the axis exactly as an element sweep would.

    The curve parameter is free per point, so phase and speed are
    unconstrained — only *lying on* the curve is required.

    Motion type gates the branch softly. A line is a circle of infinite
    radius, so the revolute branch can always match the prismatic one and a
    bare residual would collapse the gate to revolute
    (``dL/dp = L_arc - L_line <= 0`` everywhere). ``radius_weight`` makes the
    richer hypothesis pay for itself, turning the gate into a differentiable
    model-selection criterion.
    """

    def __init__(
        self,
        weight: float,
        radius_weight: float = 0.1,
        radius_ref: float = 1.0,
        num_arc_samples: int = 64,
        degenerate_threshold: float = 1e-6,
    ):
        super().__init__()
        self.weight = weight
        self.radius_weight = radius_weight
        self.radius_ref = radius_ref
        self.num_arc_samples = num_arc_samples
        # A track with no extent has no curve to lie on.
        self.degenerate_threshold = degenerate_threshold

    def forward(self, outputs: ModelOutputs, targets: StepTargets) -> LossTerms:
        track = targets.trajectory_2d
        if (
            track is None
            or targets.camera_intrinsic is None
            or targets.img_size is None
            or targets.anchor_depth is None
            or outputs.origin_depth is None
        ):
            return self._zero(outputs.motion_pred), {}

        device = outputs.motion_pred.device
        track = track.to(device).float()
        K_norm = normalized_intrinsics(
            targets.camera_intrinsic.to(device), targets.img_size.to(device)
        )
        axis = F.normalize(outputs.motion_pred.float(), p=2, dim=1, eps=1e-8)

        # Anchor: the track's own first point, lifted to metric 3D.
        start_3d = backproject_points(
            K_norm, track[:, 0], targets.anchor_depth.to(device).float()
        )

        line_residual = self._prismatic_residual(K_norm, track, start_3d, axis)
        arc_residual, radius = self._revolute_residual(
            K_norm, track, start_3d, axis, outputs.coords_hat.float(),
            outputs.origin_depth.float(),
        )
        complexity = torch.relu(radius - self.radius_ref) / self.radius_ref

        p_revolute = outputs.motion_type_logits.float().softmax(dim=-1)[:, 1]
        per_sample = (1.0 - p_revolute) * line_residual + p_revolute * (
            arc_residual + self.radius_weight * complexity
        )

        extent = (track - track[:, 0:1]).norm(dim=-1).max(dim=1).values
        valid = extent > self.degenerate_threshold
        term = (
            per_sample[valid].mean()
            if bool(valid.any())
            else self._zero(outputs.motion_pred)
        )
        return self.weight * term, {"L_geo_projected": term}

    def _prismatic_residual(self, K_norm, track, start_3d, axis):
        """Point-to-line distance: a 3D line projects to a 2D line exactly."""
        # A second point on the line, stepped proportionally to depth so the
        # projected direction is well conditioned at any distance.
        step = start_3d[:, 2:3].abs().clamp(min=1e-3) * 0.1
        pair = torch.stack([start_3d, start_3d + step * axis], dim=1)
        projected = project_points(K_norm, pair)
        origin_2d, second_2d = projected[:, 0], projected[:, 1]

        direction = second_2d - origin_2d
        length = direction.norm(dim=-1, keepdim=True)
        unit = direction / length.clamp(min=1e-9)

        rel = track - origin_2d.unsqueeze(1)
        cross = rel[..., 0] * unit[:, None, 1] - rel[..., 1] * unit[:, None, 0]
        distance = cross.abs().mean(dim=1)

        # Axis pointing along the view ray projects to a single point, so
        # there is no line to measure against — score the raw offset instead.
        degenerate = (length.squeeze(-1) < 1e-8)
        return torch.where(degenerate, rel.norm(dim=-1).mean(dim=1), distance)

    def _revolute_residual(self, K_norm, track, start_3d, axis, coords_hat, origin_depth):
        """Point-to-polyline distance against the projected circle."""
        centre = backproject_points(K_norm, coords_hat, origin_depth)
        offset = start_3d - centre
        parallel = (offset * axis).sum(-1, keepdim=True) * axis
        radius = (offset - parallel).norm(dim=-1)

        thetas = torch.linspace(
            -math.pi, math.pi, self.num_arc_samples, device=track.device
        )
        curve_3d = centre.unsqueeze(1) + _rotate_about_axis(axis, offset, thetas)
        curve_2d = project_points(K_norm, curve_3d)
        return _point_to_polyline_distance(track, curve_2d).mean(dim=1), radius


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
    if name == "projected":
        return ProjectedGeometricLoss(
            weight=getattr(loss_params, "projected_weight", 1.0),
            radius_weight=getattr(loss_params, "projected_radius_weight", 0.1),
            radius_ref=getattr(loss_params, "projected_radius_ref", 1.0),
        )
    if name == "none":
        return NoGeometricLoss()

    raise ValueError(
        f"unknown geometric_loss {name!r} — expected 'cross_gt', 'pred_pred', "
        f"'projected' or 'none'"
    )
