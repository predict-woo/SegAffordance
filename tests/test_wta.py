"""Spec tests for the 2026-08-11 body-metric + annealed-WTA change set.

Pure synthetic tensors, CPU. Pins:
  * gauge invariance of the body metric (the property the old MSE lacked);
  * the pricing rebalance (radius-relevant errors no longer near-free);
  * K = 1 reduction to plain body-metric + trajectory MSE, no CE;
  * annealing endpoints (uniform at high T, one-hot winner at T = 0);
  * stop-grad + winner-only gradients at T = 0;
  * bundle integrity (one winner index for twist AND trajectory);
  * consistency gating (screw_gt follows hyp weights; screw_self covers
    every bundle);
  * the E3 commitment result: a K = 4 head recovers |omega| ~ 1 on
    ambiguous rot/trans data where the K = 1 head collapses to the mean.
"""

import math

import torch
import torch.nn.functional as F

from model.losses.geometric import ScrewConsistencyLoss
from model.losses.twist import (
    TwistLoss,
    decode_twist,
    screw_orbit,
    twist_body_distance,
    twist_from_gt,
)
from model.outputs import ModelOutputs
from model.targets import StepTargets

ROT = torch.tensor([1])
TRANS = torch.tensor([0])
P0 = torch.tensor([0.25, 0.10, 1.80])
AXIS = F.normalize(torch.tensor([0.05, -1.0, 0.10]), dim=-1)


def make_outputs(twist_pred=None, trajectory_pred=None, twist_hyps=None,
                 trajectory_hyps=None, twist_logits=None, b=1):
    dummy_map = torch.zeros(b, 1, 4, 4)
    return ModelOutputs(
        mask_logits=dummy_map,
        point_logits=dummy_map,
        point_uv=torch.zeros(b, 2),
        twist_pred=twist_pred,
        trajectory_pred=trajectory_pred,
        twist_hyps=twist_hyps,
        trajectory_hyps=trajectory_hyps,
        twist_logits=twist_logits,
    )


def revolute_case(radius=0.5, extent_deg=45.0, n=20):
    """GT twist + absolute arc through P0."""
    u = F.normalize(torch.linalg.cross(AXIS, torch.tensor([0.0, 0.0, 1.0])), dim=-1)
    q = P0 - radius * u
    gt = twist_from_gt(AXIS[None], ROT, q[None])[0]
    ts = torch.linspace(0, math.radians(extent_deg), n)
    traj = screw_orbit(gt[None], P0[None], ts[None])[0]
    return gt, traj, q


def targets_for(gt_axis, gt_type, origin, trajectory=None):
    return StepTargets(
        motion=gt_axis, motion_type=gt_type, motion_origin_3d=origin,
        trajectory=trajectory,
    )


# ---- body metric ---------------------------------------------------------

def test_body_metric_zero_at_gt_and_gauge_invariant():
    gt, traj, q = revolute_case()
    assert twist_body_distance(gt, gt, P0, 0.25).item() == 0.0

    pred = gt + torch.tensor([0.1, -0.05, 0.02, 0.03, 0.0, -0.04])
    d_cam = twist_body_distance(pred, gt, P0, 0.25)
    # Shift the coordinate origin by c: p' = p - c, twists transform as
    # omega' = omega, v' = v + omega x c. The metric must not move.
    c = torch.tensor([0.7, -1.3, 0.4])
    def shift(xi):
        return torch.cat([xi[:3], xi[3:] + torch.linalg.cross(xi[:3], c)])
    d_shift = twist_body_distance(shift(pred), shift(gt), P0 - c, 0.25)
    assert torch.allclose(d_cam, d_shift, atol=1e-5)


def test_body_metric_prices_radius_errors_the_mse_missed():
    """The spec's pricing table, as relative-to-sign-flip regressions."""
    gt, traj, q = revolute_case(radius=0.25)
    u = F.normalize(q - P0, dim=-1)

    def mse6(a, b):
        return (a - b).pow(2).mean().item()

    def body(a, b):
        return twist_body_distance(a, b, P0, 0.25).item()

    flip = -gt
    far = twist_from_gt(AXIS[None], ROT, (P0 + 3 * 0.25 * u)[None])[0]  # radius 3x
    shrunk = torch.cat([0.3 * gt[:3], gt[3:]])                          # |omega| 0.3

    # old metric: both radius errors are a few % of a flip
    assert mse6(far, gt) / mse6(flip, gt) < 0.05
    assert mse6(shrunk, gt) / mse6(flip, gt) < 0.05
    # body metric: the same errors are comparable to / larger than a flip
    assert body(far, gt) / body(flip, gt) > 0.3
    assert body(shrunk, gt) / body(flip, gt) > 1.0


# ---- WTA loss ------------------------------------------------------------

def test_k1_reduces_to_body_plus_trajectory_no_ce():
    gt, traj, q = revolute_case()
    pred = gt + 0.1
    traj_rel = traj - traj[0:1]
    pred_traj = traj_rel + 0.02

    loss = TwistLoss(weight=0.5, sign_agnostic=False, metric_rho=0.25,
                     trajectory_weight=4.0)
    out = make_outputs(twist_pred=pred[None], trajectory_pred=pred_traj[None])
    tg = targets_for(AXIS[None], ROT, q[None], trajectory=traj[None])
    total, terms, aux = loss(out, tg)

    expect = (
        0.5 * twist_body_distance(pred, gt, traj[0], 0.25)
        + 4.0 * (pred_traj - traj_rel).pow(2).mean()
    )
    assert torch.allclose(total, expect, atol=1e-6)
    assert "L_hyp_ce" not in terms
    assert aux["handled_traj"] is True


def test_euclidean_fallback_without_trajectory():
    gt, traj, q = revolute_case()
    pred = gt + 0.1
    loss = TwistLoss(weight=1.0, sign_agnostic=False)
    total, terms, aux = loss(
        make_outputs(twist_pred=pred[None]), targets_for(AXIS[None], ROT, q[None])
    )
    assert torch.allclose(total, (pred - gt).pow(2).mean(), atol=1e-6)
    assert aux["handled_traj"] is False


def wta_case(K=4):
    """B=1, K bundles: bundle 1 == GT, others wrong."""
    gt, traj, q = revolute_case()
    traj_rel = traj - traj[0:1]
    hyps = torch.stack([gt + 1.0, gt, -gt, gt + torch.randn(6)])[None]  # (1,4,6)
    traj_hyps = torch.stack(
        [traj_rel + 0.5, traj_rel, -traj_rel, traj_rel + 0.3]
    )[None]
    logits = torch.zeros(1, K, requires_grad=True)
    out = make_outputs(
        twist_pred=hyps[0, 1][None], trajectory_pred=traj_hyps[0, 1][None],
        twist_hyps=hyps, trajectory_hyps=traj_hyps, twist_logits=logits,
    )
    tg = targets_for(AXIS[None], ROT, q[None], trajectory=traj[None])
    return out, tg, gt, traj


def test_annealing_endpoints():
    out, tg, gt, traj = wta_case()
    loss = TwistLoss(weight=0.5, sign_agnostic=False, trajectory_weight=4.0)

    loss.temperature = 1e6                      # -> uniform
    _, _, aux = loss(out, tg)
    assert torch.allclose(aux["weights"], torch.full((1, 4), 0.25), atol=1e-4)

    loss.temperature = 0.0                      # -> one-hot winner
    _, terms, aux = loss(out, tg)
    assert aux["winner"].item() == 1
    assert torch.allclose(aux["weights"], F.one_hot(torch.tensor([1]), 4).float())
    assert terms["L_twist"].item() < 1e-8       # winner IS the GT bundle
    assert terms["L_wta_traj"].item() < 1e-8


def test_hard_wta_gradients_reach_only_the_winner():
    out, tg, gt, traj = wta_case()
    out.twist_hyps.requires_grad_(True)
    out.trajectory_hyps.requires_grad_(True)
    loss = TwistLoss(weight=0.5, sign_agnostic=False, trajectory_weight=4.0,
                     hyp_ce_weight=0.0)
    loss.temperature = 0.0
    total, _, aux = loss(out, tg)
    g_tw, g_tr = torch.autograd.grad(
        total, [out.twist_hyps, out.trajectory_hyps], allow_unused=True
    )
    k = aux["winner"].item()
    for g in (g_tw, g_tr):
        for j in range(4):
            if j == k:
                continue
            assert torch.all(g[0, j] == 0), "non-winner received gradient"


def test_ce_trains_logits_toward_winner():
    out, tg, gt, traj = wta_case()
    loss = TwistLoss(weight=0.5, sign_agnostic=False, trajectory_weight=4.0,
                     hyp_ce_weight=1.0)
    loss.temperature = 0.0
    total, terms, aux = loss(out, tg)
    assert "L_hyp_ce" in terms
    g = torch.autograd.grad(total, out.twist_logits)[0]
    # CE gradient decreases the winner's logit's loss: negative there
    assert g[0, aux["winner"].item()] < 0


def test_bundle_integrity_single_joint_winner():
    """Twist favors bundle 0, trajectory favors bundle 1 (by a larger
    margin) — the winner must be ONE index chosen by the joint distortion,
    used for both parts."""
    gt, traj, q = revolute_case()
    traj_rel = traj - traj[0:1]
    hyps = torch.stack([gt, gt + 0.05])[None]                 # twist: 0 wins
    traj_hyps = torch.stack([traj_rel + 10.0, traj_rel])[None]  # traj: 1 wins big
    out = make_outputs(
        twist_pred=hyps[0, 0][None], trajectory_pred=traj_hyps[0, 0][None],
        twist_hyps=hyps, trajectory_hyps=traj_hyps,
        twist_logits=torch.zeros(1, 2),
    )
    tg = targets_for(AXIS[None], ROT, q[None], trajectory=traj[None])
    loss = TwistLoss(weight=0.5, sign_agnostic=False, trajectory_weight=4.0)
    loss.temperature = 0.0
    _, terms, aux = loss(out, tg)
    assert aux["winner"].item() == 1
    # the winner's TWIST part is scored even though bundle 0's twist is better
    assert terms["L_twist"].item() > 1e-6


# ---- consistency gating --------------------------------------------------

def consistency_case():
    gt, traj, q = revolute_case()
    traj_rel = traj - traj[0:1]
    # bundle 0: coherent (twist matches its own trajectory);
    # bundle 1: incoherent (reversed trajectory under the same twist)
    hyps = torch.stack([gt, gt])[None]
    traj_hyps = torch.stack([traj_rel, -traj_rel])[None]
    K_int = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]])
    tg = StepTargets(
        motion=AXIS[None], motion_type=ROT, motion_origin_3d=q[None],
        trajectory=traj[None], camera_intrinsic=K_int,
        img_size=torch.tensor([[2.0, 2.0]]),
    )
    depth = torch.full((1, 1, 8, 8), float(P0[2]))
    out = make_outputs(
        twist_pred=hyps[0, 0][None], trajectory_pred=traj_hyps[0, 0][None],
        twist_hyps=hyps, trajectory_hyps=traj_hyps,
        twist_logits=torch.zeros(1, 2),
    )
    # point_uv backprojecting to ~P0: u = x/z etc. with the toy intrinsics
    out.point_uv[0, 0] = (2.0 * P0[0] / P0[2] + 1.0) / 2.0
    out.point_uv[0, 1] = (2.0 * P0[1] / P0[2] + 1.0) / 2.0
    return out, tg, depth


def test_screw_self_covers_all_bundles():
    out, tg, depth = consistency_case()
    loss = ScrewConsistencyLoss(gt_weight=0.0, self_weight=1.0,
                                track_weight=0.0, sign_agnostic=False)
    # weights all on the COHERENT bundle — the incoherent one must still be
    # charged by the (GT-free, ungated) self term
    w = torch.tensor([[1.0, 0.0]])
    total, terms = loss(out, tg, depth, hyp_weights=w)
    assert terms["L_screw_self"].item() > 0.3


def test_screw_gt_follows_hyp_weights():
    out, tg, depth = consistency_case()
    # make bundle 1's TWIST wrong against GT (flip it)
    out.twist_hyps[0, 1] = -out.twist_hyps[0, 1]
    loss = ScrewConsistencyLoss(gt_weight=1.0, self_weight=0.0,
                                track_weight=0.0, sign_agnostic=False)
    good = loss(out, tg, depth, hyp_weights=torch.tensor([[1.0, 0.0]]))[1]
    bad = loss(out, tg, depth, hyp_weights=torch.tensor([[0.0, 1.0]]))[1]
    assert good["L_screw_twist_gt_traj"].item() < 1e-6
    assert bad["L_screw_twist_gt_traj"].item() > 1.0


# ---- the commitment result (E3 ported) -----------------------------------

def test_wta_head_commits_where_mse_head_averages():
    """Ambiguous rot/trans data: the K = 1 head converges to the mixture
    mean (|omega| ~ P(rot), inflated radius); a K = 4 WTA head's winner
    recovers |omega| ~ 1. Direct port of the session's E3 study."""
    torch.manual_seed(0)
    gt, traj, q = revolute_case()
    v_pris = F.normalize(traj[1] - traj[0], dim=-1)
    t_pris = torch.cat([torch.zeros(3), v_pris])

    N = 1500
    x = torch.rand(N, 1)
    p_rot = torch.clamp((x.squeeze() - 0.4) / 0.2, 0, 1)
    is_rot = torch.bernoulli(p_rot).bool()
    targets = torch.where(is_rot[:, None], gt[None], t_pris[None])
    p0 = P0[None].expand(N, 3)

    def train(K, steps=600):
        net = torch.nn.Sequential(
            torch.nn.Linear(1, 64), torch.nn.ReLU(), torch.nn.Linear(64, K * 6)
        )
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for step in range(steps):
            opt.zero_grad()
            hyps = net(x).view(N, K, 6)
            d = twist_body_distance(hyps, targets[:, None], p0[:, None], 0.25)
            T = max(10.0 * (0.001 / 10.0) ** (step / (0.8 * steps)), 0.0) \
                if step < 0.8 * steps else 0.0
            if K == 1:
                loss = d.mean()
            elif T > 0:
                qw = F.softmax(-d.detach() / T, dim=1)
                loss = (qw * d).sum(1).mean()
            else:
                loss = d.min(dim=1).values.mean()
            loss.backward()
            opt.step()
        return net

    x_amb = torch.tensor([[0.5]])  # P(rot) = 0.5: maximal ambiguity

    mse_net = train(K=1)
    om_mse = mse_net(x_amb).view(1, 6)[0, :3].norm().item()

    wta_net = train(K=4)
    hyps = wta_net(x_amb).view(4, 6)
    d = twist_body_distance(hyps, gt[None], P0[None], 0.25)
    om_wta = hyps[d.argmin()][:3].norm().item()

    assert om_mse < 0.75, f"MSE head should hedge, got |omega|={om_mse:.2f}"
    assert om_wta > 0.85, f"WTA winner should commit, got |omega|={om_wta:.2f}"
