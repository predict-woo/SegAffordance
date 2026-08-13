"""Per-branch losses of the gen-6 split articulation arm.

Spec: docs/superpowers/specs/2026-08-13-split-heads-gen6-design.md. The
origin target is CANONICALIZED rather than made gauge-invariant: q* is the
unique GT-axis point whose segment to the interaction point is
perpendicular to the axis, so all 3 output dimensions are constrained and
the target sits near the element (distance-to-line left the along-axis
component free to drift).
"""
import torch
import torch.nn.functional as F


def perpendicular_foot(
    origin: torch.Tensor, direction: torch.Tensor, point: torch.Tensor
) -> torch.Tensor:
    """q* = origin + ((point - origin) . d_hat) d_hat, all (B, 3).

    Invariant to sliding `origin` along the axis (annotation gauge).
    """
    d = F.normalize(direction, dim=-1, eps=1e-8)
    along = ((point - origin) * d).sum(-1, keepdim=True)
    return origin + along * d


def project_q_star(origin_gt, direction_gt, point_gt, K_norm,
                   z_min: float = 0.05):
    """Normalized [0,1] projection of q* + validity mask.

    valid = q* in front of the camera AND inside the frame. Invalid rows
    still return finite uv (clamped z) so downstream masking is the only
    consumer of validity — no NaN can leak.
    """
    q_star = perpendicular_foot(origin_gt, direction_gt, point_gt)
    z = q_star[..., 2]
    z_safe = z.clamp(min=z_min)
    u = K_norm[..., 0, 0] * q_star[..., 0] / z_safe + K_norm[..., 0, 2]
    v = K_norm[..., 1, 1] * q_star[..., 1] / z_safe + K_norm[..., 1, 2]
    uv = torch.stack([u, v], dim=-1)
    valid = (z > z_min) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    return uv, valid


def origin_canonical_loss(
    origin_pred: torch.Tensor,
    origin_gt: torch.Tensor,
    direction_gt: torch.Tensor,
    point_gt: torch.Tensor,
    motion_type: torch.Tensor,
) -> torch.Tensor:
    """Mean ||q_hat - q*||^2 over revolute rows; connected zero when none.

    Prismatic rows contribute nothing — a translation has no axis location,
    so the head receives no gradient from them.
    """
    q_star = perpendicular_foot(origin_gt, direction_gt, point_gt)
    sq = (origin_pred - q_star).pow(2).sum(-1)
    revolute = motion_type.to(sq.device) == 1
    if bool(revolute.any()):
        return sq[revolute].mean()
    # Zero that keeps the graph connected (same convention as the screw
    # losses' degenerate handling) so .backward() is always legal.
    return (origin_pred.sum() * 0.0)


def axis_direction_loss(
    motion_pred: torch.Tensor, motion_gt: torch.Tensor, sign_agnostic: bool
) -> torch.Tensor:
    """Classical 1 - cos^2 (antiparallel OK) or sign-sensitive 1 - cos.

    SF3D's stored axis sign is canonical (the GT trajectory is derived from
    it), so the gen-6 arm runs sign-sensitive; OPD keeps the classical form.
    """
    cos = F.cosine_similarity(motion_pred, motion_gt, dim=1, eps=1e-4)
    if sign_agnostic:
        return (1.0 - cos.pow(2)).mean()
    return (1.0 - cos).mean()
