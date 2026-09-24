"""Dependency-free port of SegAffordance's closed-form screw loss.

Single file dropped verbatim into each external repo so every A/B uses
bit-identical loss code. Source of truth: SegAffordance
model/losses/geometric.py (analytic_screw_trajectory, closed_form_screw_loss);
the copied test suite test_screw_loss.py is the executable spec.

Conventions (load-bearing):
  * revolute rows need a GT origin q* AND a GT point p* ON the moving part;
    the loss is invariant to sliding q along the axis and blind to absolute
    placement — keep the host's native origin/point supervision alongside.
  * axis SIGN is supervised (t = n x r carries it): GT axes must follow a
    consistent right-hand convention (the host dataset's canonicalisation is
    fine as long as pred and GT share it).
  * rows with |r*| below `min_radius` are noise (on-axis parts) — mask them.
"""
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def analytic_screw_trajectory(
    motion_type_gt: torch.Tensor,
    axis_trans: torch.Tensor,
    axis_rot: torch.Tensor,
    origin_pred: torch.Tensor,
    point_3d_pred: torch.Tensor,
    num_points: int = 20,
    trans_length: float = 0.7,
    rot_sweep: float = math.pi / 2.0,
) -> torch.Tensor:
    """Relative screw trajectory (B, N, 3) of point p about/along the predicted
    joint. trans: t*L*d_hat; rot: (cos th - 1) r + sin th (n x r)."""
    device = point_3d_pred.device
    b = point_3d_pred.shape[0]
    t = torch.linspace(0.0, 1.0, num_points, device=device)
    d_hat = F.normalize(axis_trans.float(), p=2, dim=1, eps=1e-8)
    rel_trans = trans_length * t[None, :, None] * d_hat[:, None, :]
    n_hat = F.normalize(axis_rot.float(), p=2, dim=1, eps=1e-8)
    rel0 = (point_3d_pred - origin_pred).float()
    along = (rel0 * n_hat).sum(-1, keepdim=True)
    lever = rel0 - along * n_hat
    tangent = torch.cross(n_hat, lever, dim=-1)
    theta = rot_sweep * t
    rel_rot = (
        (torch.cos(theta)[None, :, None] - 1.0) * lever[:, None, :]
        + torch.sin(theta)[None, :, None] * tangent[:, None, :]
    )
    is_rot = (motion_type_gt.view(b, 1, 1) > 0.5).float()
    return is_rot * rel_rot + (1.0 - is_rot) * rel_trans


def closed_form_screw_loss(
    motion_type_gt: torch.Tensor,
    axis_trans: torch.Tensor,
    axis_rot: torch.Tensor,
    origin_pred: torch.Tensor,
    point_3d_pred: torch.Tensor,
    axis_gt: torch.Tensor,
    origin_gt: torch.Tensor,
    traj_start_gt: torch.Tensor,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The N -> infinity trajectory loss in closed form. Returns
    (position_term, derivative_term), each (B,) dimensionless per-row.

      position (L2):   (3pi/4 - 2)|dr|^2 + (pi/4)|dt|^2 - dr.dt  / ((pi-2)|r*|^2)
      derivative (H1): (pi/4)(|dr|^2 + |dt|^2)          - dr.dt  / ((pi/2)|r*|^2)
      translation rows: |d_hat - d_hat*|^2 = 2(1 - cos) for both terms.
    Rows routed by GT type (>0.5 = revolute)."""
    n_hat = F.normalize(axis_rot.float(), p=2, dim=1, eps=1e-8)
    n_gt = F.normalize(axis_gt.float(), p=2, dim=1, eps=1e-8)

    def lever_tangent(n, p0, q):
        rel = (p0 - q).float()
        r = rel - (rel * n).sum(-1, keepdim=True) * n
        return r, torch.cross(n, r, dim=-1)

    r_p, t_p = lever_tangent(n_hat, point_3d_pred.float(), origin_pred.float())
    r_g, t_g = lever_tangent(n_gt, traj_start_gt.float(), origin_gt.float())
    dr, dt = r_p - r_g, t_p - t_g

    a = 3.0 * math.pi / 4.0 - 2.0
    c = math.pi / 4.0
    dr2 = dr.pow(2).sum(-1)
    dt2 = dt.pow(2).sum(-1)
    drdt = (dr * dt).sum(-1)
    rg2 = r_g.pow(2).sum(-1)

    rot_pos = (a * dr2 + c * dt2 - drdt) / ((math.pi - 2.0) * rg2).clamp(min=eps)
    rot_der = (c * (dr2 + dt2) - drdt) / ((math.pi / 2.0) * rg2).clamp(min=eps)

    d_hat = F.normalize(axis_trans.float(), p=2, dim=1, eps=1e-8)
    d_gt = F.normalize(axis_gt.float(), p=2, dim=1, eps=1e-8)
    trans_term = (d_hat - d_gt).pow(2).sum(-1)

    is_rot = motion_type_gt.float() > 0.5
    pos = torch.where(is_rot, rot_pos, trans_term)
    der = torch.where(is_rot, rot_der, trans_term)
    return pos, der


# ----------------------------------------------------------------------------
# Convenience wrapper for host repos (single axis head, per-row masks).
# ----------------------------------------------------------------------------
def screw_terms(
    is_rot: torch.Tensor,
    axis_pred: torch.Tensor,
    origin_pred: torch.Tensor,
    point_pred: torch.Tensor,
    axis_gt: torch.Tensor,
    origin_gt: torch.Tensor,
    point_gt: torch.Tensor,
    min_radius: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Per-row terms for hosts with ONE axis vector per joint.

    Returns dict of (B,) tensors: pos, der (closed-form quadratics, trans rows
    = 2(1-cos)), anchor (sign-sensitive 1 - cos, all rows), lever_ok (bool:
    trans rows always True; rot rows True iff |r*| >= min_radius)."""
    is_rot = is_rot.bool()
    pos, der = closed_form_screw_loss(
        is_rot.float(), axis_pred, axis_pred, origin_pred, point_pred,
        axis_gt, origin_gt, point_gt,
    )
    cos = F.cosine_similarity(axis_pred.float(), axis_gt.float(), dim=-1, eps=1e-8)
    anchor = 1.0 - cos
    n_gt = F.normalize(axis_gt.float(), p=2, dim=1, eps=1e-8)
    rel = (point_gt - origin_gt).float()
    r_g = rel - (rel * n_gt).sum(-1, keepdim=True) * n_gt
    lever_ok = torch.where(is_rot, r_g.norm(dim=-1) >= min_radius,
                           torch.ones_like(is_rot))
    return {"pos": pos, "der": der, "anchor": anchor, "lever_ok": lever_ok}


def combine_terms(terms: Dict[str, torch.Tensor], valid: torch.Tensor,
                  w_h1: float = 1.0, w_pos: float = 0.0, w_anchor: float = 0.5,
                  row_weight: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Headline SF3D recipe: H1 derivative quadratic (1.0) + 1-cos anchor (0.5);
    position quadratic off by default (fallback 0.5/0.5). `valid` (B,) bool
    selects rows (non-fixed joints, valid nodes, lever_ok)."""
    m = (valid & terms["lever_ok"]).float()
    if row_weight is not None:
        m = m * row_weight.float()
    denom = m.sum().clamp(min=1.0)
    per_row = w_h1 * terms["der"] + w_pos * terms["pos"] + w_anchor * terms["anchor"]
    loss = (per_row * m).sum() / denom
    logs = {
        "screw/der": float((terms["der"] * m).sum() / denom),
        "screw/pos": float((terms["pos"] * m).sum() / denom),
        "screw/anchor": float((terms["anchor"] * m).sum() / denom),
        "screw/n_rows": float(m.gt(0).sum()),
    }
    return loss, logs


# ----------------------------------------------------------------------------
# Metrics (numpy). Signed variants are the ones the hosts never report.
# ----------------------------------------------------------------------------
def axis_metrics(n_pred: np.ndarray, n_gt: np.ndarray,
                 o_pred: np.ndarray = None, o_gt: np.ndarray = None) -> Dict[str, float]:
    """n_*: (3,) directions; o_*: (3,) origins (revolute only).
    Returns angle_unsigned_deg (min(theta, 180-theta), what the papers report),
    angle_signed_deg, flip (1 if n.n* < 0), origin_line_dist (|(o-o*) x n*|,
    point-to-GT-axis), origin_line_dist_sym (avg of both directions)."""
    n_p = np.asarray(n_pred, dtype=np.float64); n_p = n_p / (np.linalg.norm(n_p) + 1e-12)
    n_g = np.asarray(n_gt, dtype=np.float64); n_g = n_g / (np.linalg.norm(n_g) + 1e-12)
    c = float(np.clip(np.dot(n_p, n_g), -1.0, 1.0))
    signed = math.degrees(math.acos(c))
    out = {
        "angle_signed_deg": signed,
        "angle_unsigned_deg": min(signed, 180.0 - signed),
        "flip": 1.0 if c < 0 else 0.0,
    }
    if o_pred is not None and o_gt is not None:
        d = np.asarray(o_pred, dtype=np.float64) - np.asarray(o_gt, dtype=np.float64)
        out["origin_line_dist"] = float(np.linalg.norm(np.cross(d, n_g)))
        out["origin_line_dist_sym"] = 0.5 * (out["origin_line_dist"]
                                             + float(np.linalg.norm(np.cross(-d, n_p))))
    return out
