"""Geometric consistency losses between the motion/twist and trajectory heads.

Selected by ``LossParams.geometric_loss``: ``cross_gt`` | ``pred_pred`` |
``pred_pred_art`` | ``projected`` | ``screw`` | ``none``. The first two are
described below; ``pred_pred_art`` (gen-6 all-predicted trajectory/axis-line/
type consistency),
``projected`` (2D-track alignment) and ``screw`` (twist-native, no motion-type
branching) document themselves on their classes.

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
import typing
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.outputs import ModelOutputs
from model.targets import StepTargets

from .twist import screw_orbit

LossTerms = Tuple[torch.Tensor, Dict[str, torch.Tensor]]


class GeometricConsistencyLoss(nn.Module):
    """Base class. Subclasses return (weighted total, unweighted terms).

    ``depth`` is the model's INPUT depth map (B, 1, H, W) — an observation the
    model receives at inference anyway, so using it never constitutes teacher
    forcing. Only the screw variant reads it (to lift the predicted 2D point
    to a metric 3D anchor); the others ignore it.
    """

    def forward(
        self,
        outputs: ModelOutputs,
        targets: StepTargets,
        depth: typing.Optional[torch.Tensor] = None,
    ) -> LossTerms:
        raise NotImplementedError

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=reference.device, dtype=torch.float32)


class NoGeometricLoss(GeometricConsistencyLoss):
    """Contributes nothing. Used on OPD and as the ablation arm."""

    def forward(self, outputs, targets, depth=None) -> LossTerms:
        return self._zero(outputs.point_uv), {}


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

    def forward(self, outputs, targets, depth=None) -> LossTerms:
        if targets.trajectory is None or outputs.motion_pred is None:
            return self._zero(outputs.point_uv), {}

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
            term = self._zero(outputs.point_uv)

        # The key is emitted whenever the dataset has trajectories at all, so
        # the CSV logger sees a stable set of columns across steps.
        return self.weight * term, {"L_geo_pred_pred": term}


class PredPredArticulationLoss(GeometricConsistencyLoss):
    """Gen-6 all-predicted consistency: trajectory <-> axis line <-> type.

    The classical line/circle residual forms with every input a prediction —
    possible only now that the origin is predicted (the cross-GT scheme had
    to teacher-force it). No field of ``targets`` is read: GT reaches each
    head only through its own direct loss, which is also what anchors the
    mutually-consistent-but-wrong failure mode.

    Everything lives in the trajectory's relative frame (first point = 0 =
    the predicted interaction point), so the predicted axis point is
    ``origin_pred - point_3d_pred``. With ``trajectory_is_absolute`` the
    head emits camera-frame points instead: the relative curve is rebuilt
    in-loss as ``traj - traj[:, :1]`` and the axis point becomes
    ``origin_pred - traj[:, 0]`` — ``point_3d_pred`` is never read there.
    The type gate is the SOFT predicted P(revolute) — coupling strengthens
    as the type head sharpens (same self-switching as
    PredPredGeometricLoss). Degenerate predicted curves (no displacement
    energy) are dropped from the batch mean: a zero curve satisfies any
    axis and would otherwise push the gate around.

    With ``normalized=True`` (gen-10) both branches become dimensionless
    relative errors: the line term is the fraction of displacement energy
    perpendicular to the axis (<= 1 by construction) and the circle term
    is the orbit residual divided by the squared predicted radius, floored
    at ``radius_floor``. This keeps the soft type gate blending
    like-scaled quantities and stops large-radius elements from dominating
    the batch mean.
    """

    def __init__(
        self,
        weight: float,
        degenerate_threshold: float = 1e-6,
        trajectory_is_absolute: bool = False,
        normalized: bool = False,
        radius_floor: float = 0.10,
    ):
        super().__init__()
        self.weight = weight
        self.degenerate_threshold = degenerate_threshold
        self.trajectory_is_absolute = trajectory_is_absolute
        self.normalized = normalized
        self.radius_floor = radius_floor

    def forward(self, outputs, targets, depth=None) -> LossTerms:
        required = (
            outputs.trajectory_pred, outputs.motion_pred,
            outputs.motion_type_logits, outputs.origin_pred,
        )
        if not self.trajectory_is_absolute:
            required = required + (outputs.point_3d_pred,)
        if any(r is None for r in required):
            return self._zero(outputs.mask_logits), {}

        # fp32 throughout (precision-16 runs; ratios/norms under autocast).
        traj = outputs.trajectory_pred.float()                    # (B, N, 3)
        d = traj - traj[:, :1] if self.trajectory_is_absolute else traj
        # Gen-17 split axis heads: pair each branch with the axis that means
        # the right thing — the LINE residual reads the trans candidate
        # (slide direction), the CIRCLE residual the rot candidate (hinge
        # axis). Legacy arms feed the single motion_pred to both.
        if getattr(outputs, "motion_pred_rot", None) is not None:
            axis_line, axis_circ = outputs.motion_pred_trans, outputs.motion_pred_rot
        else:
            axis_line = axis_circ = outputs.motion_pred
        dhat_line = F.normalize(axis_line.float(), p=2, dim=1, eps=1e-8)[:, None, :]
        dhat_n = F.normalize(axis_circ.float(), p=2, dim=1, eps=1e-8)[:, None, :]

        # Prismatic: displacement energy perpendicular to the axis.
        l_line = torch.cross(d, dhat_line.expand_as(d), dim=-1).pow(2).sum(-1).mean(-1)

        # Revolute: squared distance to the predicted CIRCLE — the orbit of
        # the anchor (first point, 0 by construction) about the predicted
        # axis line. Two components: radial (distance to the axis line stays
        # at r_hat, the anchor's own distance) and axial (no drift along the
        # axis from the anchor's own plane — the element's plane, NOT the
        # hinge pin's, so this is not the plane-term bug corrected in
        # CrossGTGeometricLoss). The axial term is what separates a line
        # parallel to the axis (constant line-distance, yet not an orbit)
        # from a true circle; it never reads c, so origin gauge freedom
        # along the axis is preserved.
        if self.trajectory_is_absolute:
            c = (outputs.origin_pred.float() - traj[:, 0])[:, None, :]
        else:
            c = (outputs.origin_pred - outputs.point_3d_pred).float()[:, None, :]

        def _line_dist(x):
            rel = x - c
            along = (rel * dhat_n).sum(-1, keepdim=True)
            return (rel - along * dhat_n).norm(dim=-1)

        r_hat = _line_dist(torch.zeros_like(d[:, :1]))            # (B, 1)
        axial_drift = (d * dhat_n).sum(-1)                        # (B, N)
        l_circle = (
            (_line_dist(d) - r_hat).pow(2) + axial_drift.pow(2)
        ).mean(-1)                                                # (B,)

        if self.normalized:
            # Gen-10 (2026-08-15 spec): dimensionless relative errors, so
            # the soft gate blends consistent units and big-radius doors
            # don't dominate small ones. Line: fraction of the displacement
            # energy perpendicular to the axis (intrinsically <= 1).
            # Circle: orbit residual relative to the predicted radius,
            # floored at radius_floor (= the dataset's min_revolute_radius)
            # so an implausibly small r_hat is measured against that scale
            # instead of exploding the ratio.
            mean_sq_disp = d.pow(2).sum(-1).mean(-1)              # (B,)
            l_line = l_line / mean_sq_disp.clamp(min=1e-8)
            l_circle = l_circle / (
                r_hat.squeeze(1).clamp(min=self.radius_floor).pow(2)
            )

        p_rev = outputs.motion_type_logits.float().softmax(dim=-1)[:, 1]
        per_sample = (1.0 - p_rev) * l_line + p_rev * l_circle

        energy = d.pow(2).sum(dim=(-1, -2))
        valid = energy > self.degenerate_threshold
        if bool(valid.any()):
            term = per_sample[valid].mean()
        else:
            # Graph-connected zero (not a fresh constant): callers backward
            # through the total unconditionally, and the inputs must receive
            # (zero) gradients rather than a "does not require grad" error.
            term = per_sample.sum() * 0.0
        return self.weight * term, {"L_geo_pred_pred_art": term}


def normalized_intrinsics(K: torch.Tensor, img_size: torch.Tensor) -> torch.Tensor:
    """K at original resolution -> K projecting camera 3D into [0, 1] coords.

    The dataset stores intrinsics for the ORIGINAL frame while every 2D
    quantity in the model (``point_uv``, the 2D tracks) is normalised to
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


def _point_to_polyline_distance(
    points: torch.Tensor,
    polyline: torch.Tensor,
    segment_mask: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """(B, N, 2) points, (B, M, 2) polyline -> (B, N) distance to the polyline.

    Distance to the *segments*, not to the sampled vertices: sampling a conic
    at M points and measuring to the vertices would put a floor of half the
    vertex spacing under the residual.

    ``segment_mask`` (B, M-1) selects which segments may be matched. Excluded
    segments are scored at a large FINITE constant, so they never win the min
    and carry zero gradient (finite, not inf: an inf that meets a zero-weight
    point mask downstream would make inf * 0 = NaN). Callers use this to drop
    segments whose 3D pre-images sit behind the camera's near plane — a
    straight 2D segment bridging a camera-plane crossing is not a projection
    of the real curve (the true projection wraps through infinity), and
    matching a point to such a fake bridge would reward wrong geometry.
    """
    start = polyline[:, :-1].unsqueeze(1)              # (B, 1, M-1, 2)
    end = polyline[:, 1:].unsqueeze(1)                 # (B, 1, M-1, 2)
    seg = end - start
    rel = points.unsqueeze(2) - start                  # (B, N, M-1, 2)
    t = (rel * seg).sum(-1) / (seg * seg).sum(-1).clamp(min=1e-12)
    closest = start + t.clamp(0.0, 1.0).unsqueeze(-1) * seg
    dist = (points.unsqueeze(2) - closest).norm(dim=-1)  # (B, N, M-1)
    if segment_mask is not None:
        dist = dist.masked_fill(~segment_mask.unsqueeze(1), 1e4)
    return dist.min(dim=-1).values


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
        near_plane: float = 0.05,
    ):
        super().__init__()
        self.weight = weight
        self.radius_weight = radius_weight
        self.radius_ref = radius_ref
        self.num_arc_samples = num_arc_samples
        # A track with no extent has no curve to lie on.
        self.degenerate_threshold = degenerate_threshold
        # Arc samples nearer the camera than this are excluded from the
        # projected polyline (same reasoning as ScrewConsistencyLoss).
        self.near_plane = near_plane

    def forward(self, outputs, targets, depth=None) -> LossTerms:
        track = targets.trajectory_2d
        if (
            track is None
            or targets.camera_intrinsic is None
            or targets.img_size is None
            or targets.anchor_depth is None
            or outputs.origin_depth is None
        ):
            return self._zero(outputs.point_uv), {}

        device = outputs.point_uv.device
        # Same fp16 overflow hazard as ScrewConsistencyLoss: projected curve
        # points near the camera plane reach ~1e6 and overflow under autocast.
        with torch.autocast(device_type=device.type, enabled=False):
            return self._forward_fp32(outputs, targets, track, device)

    def _forward_fp32(self, outputs, targets, track, device) -> LossTerms:
        track = track.to(device).float()
        # SF3D marks off-frame / behind-camera points invalid ([0,0]
        # placeholders); scoring those would measure distance to fabricated
        # corners. None (2D pretraining sources) means all points observed.
        # The track's first point anchors the curve, so a sample whose first
        # point is unobserved has no anchor and is dropped entirely.
        if targets.trajectory_2d_valid is not None:
            point_mask = targets.trajectory_2d_valid.to(device).float()
        else:
            point_mask = torch.ones(track.shape[:2], device=device)
        mask_count = point_mask.sum(dim=1)
        K_norm = normalized_intrinsics(
            targets.camera_intrinsic.to(device), targets.img_size.to(device)
        )
        axis = F.normalize(outputs.motion_pred.float(), p=2, dim=1, eps=1e-8)

        # Anchor: the track's own first point, lifted to metric 3D.
        start_3d = backproject_points(
            K_norm, track[:, 0], targets.anchor_depth.to(device).float()
        )

        line_dists = self._prismatic_residual(K_norm, track, start_3d, axis)
        arc_dists, radius, arc_visible = self._revolute_residual(
            K_norm, track, start_3d, axis, outputs.point_uv.float(),
            outputs.origin_depth.float(),
        )
        denom = mask_count.clamp(min=1.0)
        line_residual = (line_dists * point_mask).sum(dim=1) / denom
        arc_residual = (arc_dists * point_mask).sum(dim=1) / denom
        complexity = torch.relu(radius - self.radius_ref) / self.radius_ref

        p_revolute = outputs.motion_type_logits.float().softmax(dim=-1)[:, 1]
        per_sample = (1.0 - p_revolute) * line_residual + p_revolute * (
            arc_residual + self.radius_weight * complexity
        )

        extent = ((track - track[:, 0:1]).norm(dim=-1) * point_mask).max(dim=1).values
        # arc_visible: a sample whose arc hypothesis is entirely behind the
        # near plane would score the masked-fill constant, swamping the batch
        # mean and the type gate — drop it instead (rare: the arc passes
        # through the track's anchored start, which is in front).
        valid = (
            (extent > self.degenerate_threshold)
            & (point_mask[:, 0] > 0)
            & (mask_count >= 2)
            & arc_visible
        )
        term = (
            per_sample[valid].mean()
            if bool(valid.any())
            else self._zero(outputs.point_uv)
        )
        return self.weight * term, {"L_geo_projected": term}

    def _prismatic_residual(self, K_norm, track, start_3d, axis):
        """Per-point distance to the projected line (a 3D line projects to a
        2D line exactly). Returns (B, N); the caller applies the point mask."""
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

        # Axis pointing along the view ray projects to a single point, so
        # there is no line to measure against — score the raw offset instead.
        degenerate = (length.squeeze(-1) < 1e-8).unsqueeze(1)
        return torch.where(degenerate, rel.norm(dim=-1), cross.abs())

    def _revolute_residual(self, K_norm, track, start_3d, axis, point_uv, origin_depth):
        """Per-point distance to the projected circle.

        Returns ((B, N) distances, (B,) radius, (B,) any-visible-segment).
        """
        centre = backproject_points(K_norm, point_uv, origin_depth)
        offset = start_3d - centre
        parallel = (offset * axis).sum(-1, keepdim=True) * axis
        radius = (offset - parallel).norm(dim=-1)

        thetas = torch.linspace(
            -math.pi, math.pi, self.num_arc_samples, device=track.device
        )
        curve_3d = centre.unsqueeze(1) + _rotate_about_axis(axis, offset, thetas)
        z = curve_3d[..., 2]
        seg_visible = (z[:, :-1] > self.near_plane) & (z[:, 1:] > self.near_plane)
        curve_2d = project_points(K_norm, curve_3d).clamp(min=-4.0, max=5.0)
        dists = _point_to_polyline_distance(track, curve_2d, seg_visible)
        return dists, radius, seg_visible.any(dim=1)


class CrossGTGeometricLoss(GeometricConsistencyLoss):
    """The historical scheme: each prediction against the other head's GT.

    The per-sample Python loop and its float associativity are kept from the
    pre-refactor implementation.

    CORRECTION 2026-07-28 — the revolute branch used to penalise two things:

      (a) circle term: every trajectory point stays the same distance from the
          rotation axis. That is what "rotation about an axis" means; correct.
      (b) plane term:  every trajectory point stays at the same height ALONG
          the axis as the motion origin. Wrong — a door handle sweeps a circle
          at its OWN height, not at the hinge pin's.

    (b) has been removed. It was not a harmless extra constraint: because the
    SF3D GT trajectories are reconstructed (tools/sf3d_process.py), they had
    been built to satisfy (b), which slid the start of every revolute arc along
    the axis until it was level with the motion origin — off the element it
    belongs to by 0.02 m and 0.21 m on two measured annotations. With arcs
    centred correctly at the element's height, (b) evaluates to the squared
    along-axis offset, which the optimiser can only reduce by tilting the
    predicted axis away from the truth. Verified: a geometrically correct arc
    whose hinge lies 0.5 m off the arc plane scored exactly 0.25 with (b) and
    0.00 without.

    Prismatic behaviour is untouched, and (a) is unchanged.
    """

    def __init__(self, geometric_weight: float, trajectory_to_motion_weight: float):
        super().__init__()
        self.geometric_weight = geometric_weight
        self.trajectory_to_motion_weight = trajectory_to_motion_weight

    def forward(self, outputs, targets, depth=None) -> LossTerms:
        if targets.trajectory is None or targets.motion_origin_3d is None:
            return self._zero(outputs.point_uv), {}

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
                proj_perp_vecs = Q_minus_C - dot_n.unsqueeze(1) * n.unsqueeze(0).expand(N, -1)
                circle_error_sq = (torch.norm(proj_perp_vecs, dim=1) - r) ** 2
                total_loss[b] = circle_error_sq.mean()

        return total_loss.mean()


class ScrewConsistencyLoss(GeometricConsistencyLoss):
    """Twist-native consistency: trajectories must follow the screw's flow.

    A constant twist (omega, v) moves every point of space along the velocity
    field ``f(p) = omega x p + v``, whose orbits are helices — with circles
    (revolute) and straight lines (prismatic, omega = 0) as interior special
    cases. So ONE residual covers both motion types: the squared sine between
    each trajectory segment and the field at the segment's midpoint. No
    line/circle branch, no soft type gate, no radius penalty — everything the
    branchy losses needed to arbitrate between two hypotheses disappears when
    there is only one hypothesis family. The residual is scale-free in the
    twist (f enters through its direction only) and sign-agnostic (sin^2), and
    leaves the speed profile along the curve unconstrained.

    Three terms, each active only when its inputs exist, so one loss serves
    SF3D 3D training, the SF3D 2D arm, and video pretraining (which has no 3D
    GT at all):

    ``L_screw_twist_gt_traj``  (needs targets.trajectory)
        Predicted twist's field against the GT trajectory (data). Grounds the
        twist where it matters — at the element's actual location.

    ``L_screw_self``  (needs trajectory_pred + intrinsics + input depth)
        Predicted twist's field against the PREDICTED trajectory, anchored at
        the PREDICTED interaction point lifted to 3D with the input depth map
        (an observation the model receives at inference — not teacher
        forcing; no GT enters this term). This is what makes the heads
        self-consistent at inference, and it gives point_uv a geometric
        job: the twist's orbit must pass through the element point it marks.

    ``L_screw_track``  (needs targets.trajectory_2d + intrinsics + depth)
        The predicted twist's orbit through the predicted anchor, projected
        into the image, against the OBSERVED 2D track (point-to-polyline,
        validity-masked). Fully prediction-anchored: unlike the legacy
        "projected" loss it needs NO anchor_depth from the data pipeline —
        and because perspective projection is invariant to a global depth
        rescaling, the depth source's scale convention cannot corrupt this
        residual (only its direction errors matter). ``omega_shrink`` adds
        the Occam prior ``|omega|`` (logged as ``L_screw_omega``) for
        settings where nothing else pins omega: a short low-curvature track
        is genuinely ambiguous between a line and a huge-radius arc, and the
        prior resolves it toward the simpler explanation. Leave it 0 where
        the direct twist L2 supervises omega (SF3D).

    The moment v is defined about the camera origin, so the field must be
    evaluated on ABSOLUTE camera-frame points — never on a re-centred
    trajectory (translating the frame changes v by -omega x t).

    Degenerate-segment masking doubles as the in-place-rotator guard: the
    9.1% of SF3D revolute records with sub-centimetre radius store a 0.01 m
    stub whose direction is an artifact (it points along the axis); their
    ~0.5 mm segments fall under segment_threshold and contribute nothing.
    """

    def __init__(
        self,
        gt_weight: float,
        self_weight: float,
        track_weight: float = 0.5,
        omega_shrink: float = 0.0,
        segment_threshold: float = 1e-3,
        field_threshold: float = 1e-4,
        num_orbit_samples: int = 64,
        near_plane: float = 0.05,
        sign_agnostic: bool = True,
    ):
        super().__init__()
        # sign_agnostic=False (SF3D, via LossParams.twist_sign_agnostic):
        # the field residual becomes cosine distance (1 - cos), so the
        # consistency terms REINFORCE the semantic sweep direction the
        # sign-sensitive twist L2 trains, instead of accepting a flipped
        # twist as perfect (sin^2 scores both signs identically).
        self.sign_agnostic = sign_agnostic
        self.gt_weight = gt_weight
        self.self_weight = self_weight
        self.track_weight = track_weight
        self.omega_shrink = omega_shrink
        self.segment_threshold = segment_threshold
        self.field_threshold = field_threshold
        self.num_orbit_samples = num_orbit_samples
        # Orbit samples closer to the camera than this (metres) are excluded
        # from the projected polyline. No real observed element sits within
        # 5 cm of the camera, so nothing legitimate is lost — while gradients
        # through the projection stay bounded (d(u)/dz ~ 1/z^2 explodes as
        # z -> 0 even for validly in-front points) and no polyline segment
        # ever bridges a camera-plane crossing.
        self.near_plane = near_plane

    def forward(self, outputs, targets, depth=None, hyp_weights=None) -> LossTerms:
        if outputs.twist_pred is None:
            return self._zero(outputs.point_uv), {}

        device = outputs.twist_pred.device
        # fp32 is load-bearing, not hygiene: orbit points near the camera
        # plane project to ~1e6 coordinates (depth clamped at 1e-6), and under
        # fp16 autocast the projection einsums overflow to inf, then
        # inf - inf = NaN in the polyline distance. The .float() casts alone
        # do NOT protect against this — autocast re-casts einsum inputs to
        # fp16 regardless. This killed run 20260728_sf3d_2d_twist at step 49.
        with torch.autocast(device_type=device.type, enabled=False):
            return self._forward_fp32(outputs, targets, depth, device, hyp_weights)

    def _forward_fp32(self, outputs, targets, depth, device, hyp_weights=None) -> LossTerms:
        # WTA hypothesis gating (2026-08-11 spec): the GT-anchored gt-term
        # follows the annealed winner weights `hyp_weights` (same
        # stopgrad(q_T) as the WTA regression — all bundles early, winner
        # as T -> 0); the GT-free self-term averages over ALL K bundles
        # unconditionally, pairing each twist with its OWN trajectory.
        # K = 1 reduces both to the previous single-twist behaviour.
        if outputs.twist_hyps is not None:
            twists = outputs.twist_hyps.float()                  # (B, K, 6)
        else:
            twists = outputs.twist_pred.float().unsqueeze(1)     # (B, 1, 6)
        B, K = twists.shape[0], twists.shape[1]
        flat = twists.reshape(B * K, 6)
        if hyp_weights is None:
            # Fallback (callers that predate the WTA wiring): uniform.
            hyp_weights = twists.new_full((B, K), 1.0 / K)
        hyp_weights = hyp_weights.detach().float()

        total = self._zero(outputs.point_uv)
        terms: Dict[str, torch.Tensor] = {}

        if targets.trajectory is not None:
            traj = targets.trajectory.to(device).float()         # (B, N, 3)
            res, valid = self._field_residual_rows(
                flat, traj.repeat_interleave(K, dim=0)
            )
            res, valid = res.view(B, K), valid.view(B, K)
            w = hyp_weights * valid.float()
            row_ok = w.sum(dim=1) > 0
            if bool(row_ok.any()):
                gt_term = (
                    (res * w).sum(dim=1)[row_ok] / w.sum(dim=1)[row_ok]
                ).mean()
            else:
                gt_term = self._zero(flat)  # all rows degenerate: neutral
            total = total + self.gt_weight * gt_term
            terms["L_screw_twist_gt_traj"] = gt_term

        # The prediction-side anchor both remaining terms share: the model's
        # own 2D point, lifted with a differentiable lookup of the INPUT depth
        # map. point_uv is (x, y) in [0, 1]; grid_sample wants [-1, 1]. A
        # hole in the depth map gives z ~ 0 and a meaningless anchor, so such
        # samples are masked out of both terms.
        anchor = anchor_ok = K_norm = None
        if (
            depth is not None
            and targets.camera_intrinsic is not None
            and targets.img_size is not None
        ):
            K_norm = normalized_intrinsics(
                targets.camera_intrinsic.to(device), targets.img_size.to(device)
            )
            coords = outputs.point_uv.float()
            grid = (coords * 2.0 - 1.0).view(-1, 1, 1, 2)
            z = F.grid_sample(
                depth.to(device).float(), grid, align_corners=False
            ).view(-1)
            anchor = backproject_points(K_norm, coords, z)
            anchor_ok = z > 1e-3

        if anchor is not None and (
            outputs.trajectory_hyps is not None or outputs.trajectory_pred is not None
        ):
            if outputs.trajectory_hyps is not None:
                traj_h = outputs.trajectory_hyps.float()          # (B, K, N, 3)
            else:
                traj_h = outputs.trajectory_pred.float().unsqueeze(1)
            trajectory_abs = anchor.view(B, 1, 1, 3) + traj_h
            res, valid = self._field_residual_rows(
                flat, trajectory_abs.reshape(B * K, *trajectory_abs.shape[2:])
            )
            valid = valid.view(B, K) & anchor_ok.unsqueeze(1)
            if bool(valid.any()):
                self_term = res.view(B, K)[valid].mean()
            else:
                self_term = self._zero(flat)
            total = total + self.self_weight * self_term
            terms["L_screw_self"] = self_term

        if (
            self.track_weight > 0.0
            and anchor is not None
            and targets.trajectory_2d is not None
        ):
            # Track term and Occam prior act on the SELECTED twist: both are
            # 2D-arm (K = 1) constructs today, and for K > 1 the selected
            # bundle is the deployed prediction.
            track_term = self._track_residual(
                outputs.twist_pred.float(), anchor, anchor_ok, K_norm, targets, device
            )
            total = total + self.track_weight * track_term
            terms["L_screw_track"] = track_term

        # Occam prior on |omega| — independent of the track term (the 2D-only
        # arm runs track_weight 0 with the projection loss instead, and still
        # needs the prior since nothing else pins |omega| there).
        if self.omega_shrink > 0.0:
            omega_term = outputs.twist_pred.float()[..., :3].norm(dim=-1).mean()
            total = total + self.omega_shrink * omega_term
            terms["L_screw_omega"] = omega_term

        return total, terms

    def _track_residual(self, twist, anchor, anchor_ok, K_norm, targets, device):
        """Observed 2D track vs the projected orbit through the own anchor."""
        track = targets.trajectory_2d.to(device).float()
        if targets.trajectory_2d_valid is not None:
            point_mask = targets.trajectory_2d_valid.to(device).float()
        else:
            point_mask = torch.ones(track.shape[:2], device=device)

        # Curve-parameter range: |t|*|omega| up to pi covers the full circle
        # for revolute twists; for near-prismatic ones (|omega| < 1 -> clamp)
        # t in [-pi, pi] sweeps +-pi*|v| metres of line, and a line is
        # interpolated EXACTLY by the polyline, so coarse sampling costs
        # nothing there.
        base = torch.linspace(
            -math.pi, math.pi, self.num_orbit_samples, device=device
        )
        omega_norm = twist[..., :3].norm(dim=-1, keepdim=True)
        ts = base.unsqueeze(0) / omega_norm.clamp(min=1.0)
        orbit_3d = screw_orbit(twist, anchor, ts)
        # Only segments fully in front of the near plane are matchable: a 2D
        # observation cannot constrain orbit geometry it cannot see, the true
        # projection of a plane-crossing arc wraps through infinity (a
        # straight bridge segment would be fake geometry), and 1/z^2
        # projection gradients stay bounded. Excluded segments contribute
        # exactly nothing — by construction, not by clamping accident.
        z = orbit_3d[..., 2]
        seg_visible = (z[:, :-1] > self.near_plane) & (z[:, 1:] > self.near_plane)
        # Numerical backstop only (the mask is the real guard): keeps any
        # projected coordinate finite in value and gradient.
        orbit_2d = project_points(K_norm, orbit_3d).clamp(min=-4.0, max=5.0)

        dists = _point_to_polyline_distance(track, orbit_2d, seg_visible)
        count = point_mask.sum(dim=1)
        per_sample = (dists * point_mask).sum(dim=1) / count.clamp(min=1.0)
        # The orbit passes through the anchor (in front of the camera), so a
        # sample with no visible segment normally means the anchor itself was
        # bad; either way there is nothing observable to score.
        row_valid = (count > 0) & anchor_ok & seg_visible.any(dim=1)
        if bool(row_valid.any()):
            return per_sample[row_valid].mean()
        return self._zero(twist)

    def _field_residual_rows(
        self,
        twist: torch.Tensor,           # (R, 6)
        trajectory_abs: torch.Tensor,  # (R, N, 3) camera frame, ABSOLUTE
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Per-row residual: (R,) mean residual over valid segments and (R,)
        bool row validity (any valid segment). Reduction over rows — and any
        hypothesis weighting — is the caller's job."""
        omega, v = twist[..., :3], twist[..., 3:]
        delta = trajectory_abs[:, 1:] - trajectory_abs[:, :-1]
        mid = 0.5 * (trajectory_abs[:, 1:] + trajectory_abs[:, :-1])
        field = torch.linalg.cross(
            omega.unsqueeze(1).expand_as(mid), mid, dim=-1
        ) + v.unsqueeze(1)

        delta_norm = delta.norm(dim=-1)
        field_norm = field.norm(dim=-1)
        # Segments with no extent have no direction (degenerate stubs);
        # midpoints ON the predicted axis have no field direction.
        seg_valid = (delta_norm > self.segment_threshold) & (
            field_norm > self.field_threshold
        )

        cos = (delta * field).sum(-1) / (delta_norm * field_norm).clamp(min=1e-12)
        if self.sign_agnostic:
            residual = 1.0 - cos.pow(2).clamp(max=1.0)  # sin^2: axis line only
        else:
            # Cosine distance, the standard direction-sensitive alignment
            # loss: aligned = 0, perpendicular = 1, flipped = 2.
            residual = 1.0 - cos.clamp(min=-1.0, max=1.0)

        seg_count = seg_valid.float().sum(dim=1)
        per_row = (residual * seg_valid.float()).sum(dim=1) / seg_count.clamp(min=1.0)
        return per_row, seg_count > 0

    def _field_residual(
        self,
        twist: torch.Tensor,          # (B, 6)
        trajectory_abs: torch.Tensor,  # (B, N, 3) camera frame, ABSOLUTE
        sample_mask: typing.Optional[torch.Tensor] = None,  # (B,) bool
    ) -> torch.Tensor:
        per_sample, row_valid = self._field_residual_rows(twist, trajectory_abs)
        if sample_mask is not None:
            row_valid = row_valid & sample_mask
        if bool(row_valid.any()):
            return per_sample[row_valid].mean()
        return self._zero(twist)


def build_geometric_loss(
    loss_params, trajectory_is_absolute: bool = False
) -> GeometricConsistencyLoss:
    """Select a variant from ``LossParams.geometric_loss``.

    Defaults to ``cross_gt`` so that configs written before this option
    existed keep their exact behaviour. ``trajectory_is_absolute`` is
    forwarded to ``PredPredArticulationLoss`` (the only variant that
    distinguishes absolute from relative curves); other variants ignore it.
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
    if name == "pred_pred_art":
        return PredPredArticulationLoss(
            weight=getattr(loss_params, "pred_pred_art_weight", 0.5),
            trajectory_is_absolute=trajectory_is_absolute,
            normalized=getattr(loss_params, "pred_pred_art_normalized", False),
            radius_floor=getattr(loss_params, "pred_pred_art_radius_floor", 0.10),
        )
    if name == "projected":
        return ProjectedGeometricLoss(
            weight=getattr(loss_params, "projected_weight", 1.0),
            radius_weight=getattr(loss_params, "projected_radius_weight", 0.1),
            radius_ref=getattr(loss_params, "projected_radius_ref", 1.0),
        )
    if name == "screw":
        return ScrewConsistencyLoss(
            gt_weight=getattr(loss_params, "screw_gt_weight", 0.5),
            self_weight=getattr(loss_params, "screw_self_weight", 0.5),
            track_weight=getattr(loss_params, "screw_track_weight", 0.5),
            omega_shrink=getattr(loss_params, "screw_omega_shrink", 0.0),
            # One dataset-level switch: direction semantics apply to the
            # twist L2 and the consistency terms together.
            sign_agnostic=getattr(loss_params, "twist_sign_agnostic", True),
        )
    if name == "none":
        return NoGeometricLoss()

    raise ValueError(
        f"unknown geometric_loss {name!r} — expected 'cross_gt', 'pred_pred', "
        f"'pred_pred_art', 'projected', 'screw' or 'none'"
    )


class TrajectoryProjectionLoss(nn.Module):
    """Predicted 3D trajectory, projected into the image, vs the observed
    2D track — the data term of 2D-only pretraining.

    The absolute predicted curve is anchor + trajectory_pred, where the
    anchor is the model's own interaction point lifted with the INPUT depth
    map (the same prediction-side anchoring as the screw self term — no GT
    enters). Projection with the normalized intrinsics gives (B, N, 2) in
    [0, 1], compared to ``targets.trajectory_2d`` INDEX-MATCHED: point i to
    point i. That correspondence is temporal, so the loss is ordered and
    direction-sensitive by construction — it supersedes the orbit/polyline
    track term, whose unordered matching let a flipped twist score
    perfectly via its backward branch.

    Per-point mask: GT validity AND predicted z > near_plane (a point the
    model places behind the camera has no meaningful projection; such
    points are replaced by a fixed dummy BEFORE projecting so no inf/nan
    ever enters the graph, then masked out of the loss). Rows with a bad
    anchor (depth hole) are dropped. Runs in an fp32 island: projections of
    near-camera points overflow fp16 (see ScrewConsistencyLoss).
    """

    def __init__(self, weight: float, near_plane: float = 0.05):
        super().__init__()
        self.weight = weight
        self.near_plane = near_plane

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=ref.device, dtype=torch.float32)

    def forward(self, outputs, targets, depth) -> LossTerms:
        if (
            self.weight == 0.0
            or outputs.trajectory_pred is None
            or targets.trajectory_2d is None
            or targets.camera_intrinsic is None
            or targets.img_size is None
            or depth is None
            # This loss lifts point_uv to 3D, so the 3D point mode
            # (point_uv None) cannot run it — no-op there as well.
            or outputs.point_uv is None
        ):
            # mask_logits, not point_uv: always present in every mode.
            return self._zero(outputs.mask_logits), {}
        device = outputs.trajectory_pred.device
        device_type = "cuda" if outputs.trajectory_pred.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            K_norm = normalized_intrinsics(
                targets.camera_intrinsic.to(device), targets.img_size.to(device)
            )
            coords = outputs.point_uv.float()
            grid = (coords * 2.0 - 1.0).view(-1, 1, 1, 2)
            z = F.grid_sample(
                depth.to(device).float(), grid, align_corners=False
            ).view(-1)
            anchor = backproject_points(K_norm, coords, z)
            anchor_ok = z > 1e-3

            traj_abs = anchor.unsqueeze(1) + outputs.trajectory_pred.float()
            in_front = traj_abs[..., 2] > self.near_plane
            dummy = torch.tensor([0.0, 0.0, 1.0], device=device)
            safe = torch.where(in_front.unsqueeze(-1), traj_abs, dummy)
            proj = project_points(K_norm, safe)

            track = targets.trajectory_2d.to(device).float()
            if targets.trajectory_2d_valid is not None:
                valid = targets.trajectory_2d_valid.to(device).bool()
            else:
                valid = torch.ones_like(in_front, dtype=torch.bool)
            mask = (valid & in_front & anchor_ok.unsqueeze(1)).float()

            count = mask.sum()
            if count < 1.0:
                return self._zero(outputs.point_uv), {}
            sq = (proj - track).pow(2).sum(-1)
            term = (sq * mask).sum() / count
        return self.weight * term, {"L_traj_proj": term}
