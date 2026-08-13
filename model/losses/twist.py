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


def twist_body_distance(
    pred: torch.Tensor,   # (..., 6)
    gt: torch.Tensor,     # broadcastable to pred
    p0: torch.Tensor,     # (..., 3) broadcastable to pred[..., :3]
    rho: float,
) -> torch.Tensor:
    """(…,) squared kinetic-energy (body-frame) distance between twists.

    ``|Δω × p0 + Δv|² + ρ²|Δω|²`` — the left-invariant rigid-body metric of a
    unit mass at ``p0`` with gyration radius ``rho``. The first term is the
    velocity-field error AT THE OBJECT (the field is linear in the twist), so
    errors are priced by how wrongly they move the observed element, in m².
    Gauge-invariant: unlike the plain 6-vector MSE (which is this same metric
    with the body at the CAMERA ORIGIN and rho = 1), the choice of coordinate
    origin cancels. See docs/superpowers/specs/2026-08-11-twist-body-metric-
    design.md for the diagnosis this fixes.
    """
    d = pred - gt
    dw, dv = d[..., :3], d[..., 3:]
    df = torch.linalg.cross(dw, p0.expand_as(dw), dim=-1) + dv
    return df.pow(2).sum(-1) + rho * rho * dw.pow(2).sum(-1)


class TwistLoss(nn.Module):
    """Annealed winner-takes-all supervision of the articulation bundles.

    Each of the model's K hypotheses is a bundled (twist, trajectory) story
    about how the object moves. Per sample, the distortion of bundle k is

        d_k = twist_weight * L_body(xi_k, xi_gt)
            + trajectory_weight * L_traj(tau_k, tau_gt_rel)

    (both in m² at the object; L_body is ``twist_body_distance`` anchored at
    the GT element point ``targets.trajectory[:, 0]``, L_traj the per-point
    MSE in the standalone loss's mean-over-points-and-dims convention). The
    regression term is the softmin-annealed WTA sum ``Σ_k sg[q_T(k)] d_k``
    with ``q_T = softmax(-d/T)`` — every bundle trains at high T, only the
    winner as T -> 0 (T is set each epoch by the training loop; T = 0 means
    hard winner, used for all of val/test). A cross-entropy on the K logits
    with the winner as label trains the selector used at inference. This is
    what breaks the posterior-mean hedge: no single bundle is ever forced to
    explain two modes at once (aMCL, arXiv:2407.15580; spec
    docs/superpowers/specs/2026-08-11-twist-wta-head-design.md).

    K = 1 (no hypothesis fields on the outputs) reduces exactly to
    ``twist_weight * L_body + trajectory_weight * L_traj`` with no CE term.
    When the batch carries no 3D trajectory the twist distortion falls back
    to the legacy Euclidean 6-vector MSE and no trajectory part is handled.

    Sign handling as before: sign_agnostic=True (OPD — axis LINE only) takes
    the better of ``±xi_gt`` inside the twist distortion; sign_agnostic=False
    (SF3D) treats the stored sign as canonical (the preprocessor derives the
    GT trajectory FROM it).

    No-ops on batches missing a 3D origin (OPD) or when the twist head is
    off. Returns ``(total, terms, aux)``; ``aux['handled_traj']`` tells the
    caller the trajectory supervision is subsumed here (do NOT double-count
    the standalone term), ``aux['weights']``/``aux['winner']`` feed the
    consistency-loss gating.
    """

    def __init__(
        self,
        weight: float,
        sign_agnostic: bool = True,
        metric_rho: float = 0.25,
        trajectory_weight: float = 0.5,
        hyp_ce_weight: float = 0.1,
    ):
        super().__init__()
        self.weight = weight
        self.sign_agnostic = sign_agnostic
        self.metric_rho = metric_rho
        self.trajectory_weight = trajectory_weight
        self.hyp_ce_weight = hyp_ce_weight
        # Set by the training loop (annealing schedule); 0 = hard winner.
        self.temperature: float = 0.0

    @staticmethod
    def _no_aux() -> Dict[str, object]:
        return {"weights": None, "winner": None, "handled_traj": False}

    def forward(self, outputs: ModelOutputs, targets: StepTargets):
        if (
            outputs.twist_pred is None
            or targets.motion is None
            or targets.motion_type is None
            or targets.motion_origin_3d is None
        ):
            # mask_logits, not coords_hat: the 3D point mode has no coords_hat.
            zero = torch.zeros((), device=outputs.mask_logits.device, dtype=torch.float32)
            return zero, {}, self._no_aux()

        device = outputs.twist_pred.device
        gt = twist_from_gt(
            targets.motion.to(device),
            targets.motion_type.to(device),
            targets.motion_origin_3d.to(device),
        )  # (B, 6)

        if outputs.twist_hyps is not None:
            hyps = outputs.twist_hyps.float()                     # (B, K, 6)
            traj_hyps = (
                outputs.trajectory_hyps.float()
                if outputs.trajectory_hyps is not None else None  # (B, K, N, 3)
            )
        else:
            hyps = outputs.twist_pred.float().unsqueeze(1)        # (B, 1, 6)
            traj_hyps = (
                outputs.trajectory_pred.float().unsqueeze(1)
                if outputs.trajectory_pred is not None else None
            )
        B, K = hyps.shape[0], hyps.shape[1]

        trajectory = (
            targets.trajectory.to(device).float()
            if targets.trajectory is not None else None
        )

        # Twist distortion: body metric anchored at the GT element point,
        # legacy Euclidean fallback when the batch has no 3D trajectory.
        gt_k = gt.unsqueeze(1)                                    # (B, 1, 6)
        if trajectory is not None:
            p0 = trajectory[:, 0].unsqueeze(1)                    # (B, 1, 3)
            d_pos = twist_body_distance(hyps, gt_k, p0, self.metric_rho)
            d_neg = twist_body_distance(hyps, -gt_k, p0, self.metric_rho)
        else:
            d_pos = (hyps - gt_k).pow(2).mean(dim=-1)
            d_neg = (hyps + gt_k).pow(2).mean(dim=-1)
        d_twist = torch.minimum(d_pos, d_neg) if self.sign_agnostic else d_pos

        d = self.weight * d_twist                                  # (B, K)
        terms: Dict[str, torch.Tensor] = {}

        handled_traj = trajectory is not None and traj_hyps is not None
        if handled_traj:
            gt_rel = (trajectory - trajectory[:, 0:1]).unsqueeze(1)  # (B,1,N,3)
            d_traj = (traj_hyps - gt_rel).pow(2).mean(dim=(-1, -2))  # (B, K)
            d = d + self.trajectory_weight * d_traj

        winner = d.argmin(dim=1)                                   # (B,)
        if K == 1:
            q = d.new_ones(B, 1)
        elif self.temperature > 0.0:
            q = F.softmax(-d.detach() / self.temperature, dim=1)
        else:
            q = F.one_hot(winner, K).to(d.dtype)
        total = (q.detach() * d).sum(dim=1).mean()

        arange = torch.arange(B, device=device)
        terms["L_twist"] = d_twist[arange, winner].mean().detach()
        if handled_traj:
            terms["L_wta_traj"] = d_traj[arange, winner].mean().detach()

        if outputs.twist_logits is not None and K > 1:
            ce = F.cross_entropy(outputs.twist_logits.float(), winner)
            total = total + self.hyp_ce_weight * ce
            terms["L_hyp_ce"] = ce.detach()

        aux = {"weights": q.detach(), "winner": winner, "handled_traj": handled_traj}
        return total, terms, aux
