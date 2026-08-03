"""Unified screw-motion (se(3) twist) supervision.

A twist ``xi = (omega, v)`` in R^6 describes a rigid motion through the
velocity field ``p_dot = omega x p + v`` (camera frame). Both joint types are
interior points of the same space:

revolute
    ``omega`` = unit axis direction, ``v = -omega x q`` for ANY point ``q`` on
    the axis. The moment ``v`` is invariant to where along the axis ``q`` sits
    (``omega x (q + t*omega) = omega x q``), so the parameterisation carries
    the axis LINE and nothing else — there is no distinguished point on the
    axis for the network to be noisy over, and a plain L2 on the 6-vector is
    well-posed.

prismatic
    ``omega = 0``, ``v`` = unit direction. A translation is a rotation about a
    line at infinity; in twist coordinates that limit is the interior point
    ``omega = 0``, so nothing blows up as motions flatten out and the network
    never has to represent a far-away axis.

Motion type is therefore emergent — ``|omega|`` is 1 for revolute GT and 0 for
prismatic GT — rather than a separate classification target.

The decode formula for a point on the axis, ``q_hat = omega x v / |omega|^2``
(the axis point closest to the camera origin), is singular near ``omega = 0``
by nature. It is used at EVAL only and never differentiated; real revolute
radii in SF3D cap at ~1.1 m, so genuine rotations never approach the
``omega ~ 0`` region where prismatic lives.

GT twists are derived on the fly from fields every SF3D batch already carries
(axis, motion type, 3D origin); batches without a 3D origin (OPD) make the
loss a no-op, mirroring the geometric losses.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.outputs import ModelOutputs
from model.targets import StepTargets

LossTerms = Tuple[torch.Tensor, Dict[str, torch.Tensor]]


def twist_from_gt(
    motion: torch.Tensor,          # (B, 3) GT axis / direction
    motion_type: torch.Tensor,     # (B,) 0 = prismatic, 1 = revolute
    motion_origin_3d: torch.Tensor,  # (B, 3) point on the axis, camera frame
) -> torch.Tensor:
    """(B, 6) GT twist. Unit magnitude: |omega|=1 revolute, |v|=1 prismatic."""
    direction = F.normalize(motion.float(), p=2, dim=-1, eps=1e-8)
    revolute = (motion_type == 1).unsqueeze(-1)
    omega = torch.where(revolute, direction, torch.zeros_like(direction))
    moment = -torch.linalg.cross(direction, motion_origin_3d.float(), dim=-1)
    v = torch.where(revolute, moment, direction)
    return torch.cat([omega, v], dim=-1)


def decode_twist(
    twist: torch.Tensor, rot_threshold: float = 0.5, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B, 6) -> (is_revolute (B,) bool, direction (B, 3) unit, axis_point (B, 3)).

    ``axis_point`` is the point on the predicted axis closest to the camera
    origin — a canonical gauge choice, NOT a prediction of the annotated
    origin's height along the axis (the twist cannot express that, on
    purpose). Meaningless for prismatic rows; check ``is_revolute``.
    """
    omega, v = twist[..., :3].float(), twist[..., 3:].float()
    omega_norm = omega.norm(dim=-1)
    is_revolute = omega_norm > rot_threshold
    dir_revolute = omega / omega_norm.clamp(min=eps).unsqueeze(-1)
    dir_prismatic = F.normalize(v, p=2, dim=-1, eps=eps)
    direction = torch.where(is_revolute.unsqueeze(-1), dir_revolute, dir_prismatic)
    axis_point = torch.linalg.cross(omega, v, dim=-1) / omega_norm.pow(2).clamp(
        min=eps
    ).unsqueeze(-1)
    return is_revolute, direction, axis_point


def screw_orbit(
    twist: torch.Tensor,   # (B, 6)
    anchor: torch.Tensor,  # (B, 3) point the orbit passes through (t = 0)
    ts: torch.Tensor,      # (M,) or (B, M) curve parameter values
    eps: float = 1e-4,
) -> torch.Tensor:
    """(B, M, 3) points of the twist's orbit through ``anchor``: exp(t*xi)p0.

    The one place the se(3) exponential map is actually needed. Closed form
    with rotation vector phi = t*omega (theta = |phi|):

        p(t) = p0 + sinc(theta) (phi x p0) + c2 (phi x (phi x p0))
                  + t v + c2 (phi x v) t + c3 (phi x (phi x v)) t   [J_l(phi) tv]

    where sinc = sin(theta)/theta, c2 = (1-cos theta)/theta^2,
    c3 = (theta - sin theta)/theta^3 — all EVEN functions with removable
    singularities at theta = 0, handled by their Taylor series below, so the
    orbit (and its gradient) is smooth through omega = 0, where it degenerates
    to the straight line p0 + t v. This smoothness at the prismatic point is
    the entire reason the twist parameterisation was chosen.
    """
    omega = twist[..., :3].float()
    v = twist[..., 3:].float()
    if ts.dim() == 1:
        ts = ts.unsqueeze(0)
    t = ts.to(twist.device).float().unsqueeze(-1)          # (B|1, M, 1)
    w = omega.unsqueeze(1)                                  # (B, 1, 3)
    p0 = anchor.float().unsqueeze(1)                        # (B, 1, 3)
    vu = v.unsqueeze(1)                                     # (B, 1, 3)

    theta = t * omega.norm(dim=-1).view(-1, 1, 1)           # (B, M, 1), signed
    small = theta.abs() < eps
    th = torch.where(small, torch.ones_like(theta), theta)  # safe denominator
    sinc = torch.where(small, 1.0 - theta**2 / 6.0, torch.sin(th) / th)
    c2 = torch.where(small, 0.5 - theta**2 / 24.0, (1.0 - torch.cos(th)) / th**2)
    c3 = torch.where(small, 1.0 / 6.0 - theta**2 / 120.0, (th - torch.sin(th)) / th**3)

    w_x_p = torch.linalg.cross(w, p0, dim=-1)
    w_x_w_x_p = torch.linalg.cross(w, w_x_p, dim=-1)
    w_x_v = torch.linalg.cross(w, vu, dim=-1)
    w_x_w_x_v = torch.linalg.cross(w, w_x_v, dim=-1)

    rotated = p0 + (t * sinc) * w_x_p + (t**2 * c2) * w_x_w_x_p
    translated = t * vu + (t**2 * c2) * w_x_v + (t**3 * c3) * w_x_w_x_v
    return rotated + translated


def point_to_line_distance(
    point: torch.Tensor, line_point: torch.Tensor, line_dir_unit: torch.Tensor
) -> torch.Tensor:
    """(…, 3) each -> (…,) perpendicular distance from point to the line."""
    rel = point - line_point
    along = (rel * line_dir_unit).sum(-1, keepdim=True) * line_dir_unit
    return (rel - along).norm(dim=-1)


class TwistLoss(nn.Module):
    """L2 between the predicted and GT twist.

    ``(omega, v) -> (-omega, -v)`` is the same screw axis traversed the other
    way. Whether that flip is an error depends on the dataset:

    sign_agnostic=True (default; OPD)
        OPD annotates only the axis LINE — signs carry no meaning — so the
        loss scores the better of the two signs. Flipping omega and v
        TOGETHER is what preserves the axis line; scoring their signs
        independently would allow geometrically wrong combinations.

    sign_agnostic=False (SF3D)
        SF3D's stored sign is CANONICAL: the preprocessor generates the GT
        trajectory from it (trans marches along +motion_dir; rot sweeps
        right-handed about the signed axis — tools/sf3d_process.py,
        compute_trajectory_3d_camera_coords), so the sign IS the sweep
        sense the task defines. Plain MSE; predicting the reversed screw is
        a real error, consistent with the trajectory head's supervision.

    No-ops on batches missing a 3D origin (OPD) or when the twist head is off,
    following the geometric-loss convention.
    """

    def __init__(self, weight: float, sign_agnostic: bool = True):
        super().__init__()
        self.weight = weight
        self.sign_agnostic = sign_agnostic

    def forward(self, outputs: ModelOutputs, targets: StepTargets) -> LossTerms:
        if (
            outputs.twist_pred is None
            or targets.motion is None
            or targets.motion_type is None
            or targets.motion_origin_3d is None
        ):
            zero = torch.zeros((), device=outputs.motion_pred.device, dtype=torch.float32)
            return zero, {}

        device = outputs.twist_pred.device
        gt = twist_from_gt(
            targets.motion.to(device),
            targets.motion_type.to(device),
            targets.motion_origin_3d.to(device),
        )
        pred = outputs.twist_pred.float()
        err_pos = (pred - gt).pow(2).mean(dim=-1)
        if self.sign_agnostic:
            err_neg = (pred + gt).pow(2).mean(dim=-1)
            term = torch.minimum(err_pos, err_neg).mean()
        else:
            term = err_pos.mean()
        return self.weight * term, {"L_twist": term}
