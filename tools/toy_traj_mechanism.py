"""Toy demonstration of WHY trajectory (point-space) supervision teaches an
axis better than direct angular supervision, despite carrying identical
information (mechanism study, spec
docs/superpowers/specs/2026-08-25-trajectory-mechanism-design.md).

Setup: a small MLP predicts a 3D rotation axis from random features; the
GT axis is a fixed deterministic function of the features (a random
teacher), so both losses see exactly the same labels:

  loss A (angular):    1 - cos(pred_axis, gt_axis)          [the vae/axis loss]
  loss B (point-space): MSE between the 90-deg arc swept by a per-sample
                        lever about pred_axis vs about gt_axis
                        [the trajectory loss through the analytic decode]

Panels produced (all measured, not asserted):
  1. gradient_profile.png — |d loss / d axis| as a function of the angle
     error, for both losses (single-axis probe, no network). Shows where
     each loss can and cannot push.
  2. flip_recovery.png — students PRE-TRAINED TO THE ANTI-TEACHER (100%
     flipped), then finetuned with each loss: fraction of samples still
     flipped vs training step.
  3. scratch_training.png — from random init: flip rate and median angle
     error vs step for each loss.

Usage:  python tools/toy_traj_mechanism.py --out viz/<dated dir>
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

N_POINTS = 20
THETAS = torch.linspace(0.0, np.pi / 2.0, N_POINTS)


def arc_points(axis, lever):
    """(B,3) unit axis, (B,3) lever (perp not required) -> (B,N,3) arc of the
    lever's perp component about the axis, right-hand sweep (the writer's
    construction in relative frame)."""
    n = F.normalize(axis, dim=-1, eps=1e-8)
    along = (lever * n).sum(-1, keepdim=True)
    perp = lever - along * n
    tang = torch.cross(n, perp, dim=-1)
    c = torch.cos(THETAS)[None, :, None]
    s = torch.sin(THETAS)[None, :, None]
    return (c - 1.0) * perp[:, None, :] + s * tang[:, None, :]


def loss_angular(pred, gt):
    return (1.0 - F.cosine_similarity(pred, gt, dim=-1)).mean()


def loss_pointspace(pred, gt, lever):
    pa = arc_points(pred, lever)
    ga = arc_points(gt, lever)
    # normalized per-row like the gen-16 trajectory loss
    err = (pa - ga).pow(2).sum(-1).mean(-1)
    energy = ga.pow(2).sum(-1).mean(-1)
    return (err / energy.clamp(min=1e-4)).mean()


class Student(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


def make_data(n=2000, dim=32):
    x = torch.randn(n, dim)
    teacher = torch.randn(dim, 3) / np.sqrt(dim)
    gt = F.normalize(x @ teacher, dim=-1)
    lever = torch.randn(n, 3)
    # keep the lever's perp component well away from zero
    lever = lever + 0.5 * torch.sign(torch.randn(n, 1))
    return x, gt, lever


def angle_err_deg(pred, gt):
    cos = F.cosine_similarity(F.normalize(pred, dim=-1), gt, dim=-1)
    return torch.rad2deg(torch.acos(cos.clamp(-1, 1)))


def gradient_profile():
    """|d loss/d axis| vs angle error, single sample, axis probed directly."""
    gt = torch.tensor([0.0, 0.0, 1.0])
    lever = torch.tensor([1.0, 0.0, 0.0])
    angles = np.linspace(1.0, 179.0, 90)
    ga, gb = [], []
    for a in angles:
        r = np.deg2rad(a)
        ax = torch.tensor([np.sin(r), 0.0, np.cos(r)], dtype=torch.float32,
                          requires_grad=True)
        la = loss_angular(ax[None], gt[None])
        (g,) = torch.autograd.grad(la, ax)
        # project out the radial component (normalization gauge)
        g = g - (g * ax.detach()).sum() * ax.detach() / ax.detach().pow(2).sum()
        ga.append(g.norm().item())
        ax2 = torch.tensor([np.sin(r), 0.0, np.cos(r)], dtype=torch.float32,
                           requires_grad=True)
        lb = loss_pointspace(ax2[None], gt[None], lever[None])
        (g2,) = torch.autograd.grad(lb, ax2)
        g2 = g2 - (g2 * ax2.detach()).sum() * ax2.detach() / ax2.detach().pow(2).sum()
        gb.append(g2.norm().item())
    return angles, np.array(ga), np.array(gb)


def train(loss_name, model, data, steps=1500, lr=3e-3, log_every=25):
    x, gt, lever = data
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for step in range(steps + 1):
        opt.zero_grad()
        pred = model(x)
        if loss_name == "angular":
            loss = loss_angular(pred, gt)
        else:
            loss = loss_pointspace(pred, gt, lever)
        if step % log_every == 0:
            with torch.no_grad():
                errs = angle_err_deg(pred, gt)
                hist.append((step, (errs > 90.0).float().mean().item() * 100.0,
                             errs.median().item()))
        loss.backward()
        opt.step()
    return np.array(hist)


def pretrain_flipped(model, data, steps=1200):
    """Drive the student to predict the ANTI-teacher (worst-case init)."""
    x, gt, _ = data
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss_angular(model(x), -gt).backward()
        opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = make_data()

    # Panel 1: gradient profile
    angles, ga, gb = gradient_profile()
    plt.figure(figsize=(7, 4.5))
    plt.plot(angles, ga, label="angular 1-cos (the axis loss)")
    plt.plot(angles, gb, label="point-space arc MSE (the trajectory loss)")
    plt.axvline(90, ls=":", c="gray")
    plt.xlabel("axis angle error (deg)")
    plt.ylabel("|tangential gradient|")
    plt.yscale("log")
    plt.title("Where each loss can push")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "gradient_profile.png"), dpi=140)
    plt.close()

    # Panel 2: recovery from a fully flipped student
    curves = {}
    for name in ("angular", "pointspace"):
        torch.manual_seed(1)
        m = Student()
        pretrain_flipped(m, data)
        curves[name] = train(name, m, data, steps=args.steps)
    plt.figure(figsize=(7, 4.5))
    for name, h in curves.items():
        plt.plot(h[:, 0], h[:, 1], label=name)
    plt.xlabel("finetune step"); plt.ylabel("% samples flipped (>90 deg)")
    plt.title("Recovery from a 100%-flipped init (same labels, same net)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "flip_recovery.png"), dpi=140)
    plt.close()

    # Panel 3: from-scratch training
    curves2 = {}
    for name in ("angular", "pointspace"):
        torch.manual_seed(2)
        m = Student()
        curves2[name] = train(name, m, data, steps=args.steps)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, h in curves2.items():
        axes[0].plot(h[:, 0], h[:, 1], label=name)
        axes[1].plot(h[:, 0], h[:, 2], label=name)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("% flipped"); axes[0].legend()
    axes[1].set_xlabel("step"); axes[1].set_ylabel("median angle err (deg)"); axes[1].legend()
    fig.suptitle("From random init")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "scratch_training.png"), dpi=140)
    plt.close(fig)

    # Printed summary (machine-readable last lines)
    for name in ("angular", "pointspace"):
        h = curves[name]
        print(f"RECOVERY {name}: flipped% start={h[0,1]:.1f} end={h[-1,1]:.1f}")
        h2 = curves2[name]
        print(f"SCRATCH  {name}: flipped% end={h2[-1,1]:.1f} median_err end={h2[-1,2]:.2f} deg")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Extension (same session, after the first three panels REFUTED the naive
# saddle story — both losses saddle at the antipode, and point-space is
# WORSE head-to-head on a single latent): miniature of the real ablation.
# Shared trunk, axis head, and optionally (a) an auxiliary trajectory HEAD
# supervised on the GT arc (arm-D analog — tests feature shaping, H2), or
# (b) the analytic DECODE loss from the predicted axis+origin+p0 (tests
# conjunction, H1'), with finite noisy data and a held-out test set.
# ---------------------------------------------------------------------------

class TrunkModel(nn.Module):
    def __init__(self, dim=32, hidden=96, traj_head=False, detach_traj=False):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.axis_head = nn.Linear(hidden, 3)
        self.origin_head = nn.Linear(hidden, 3)
        self.p0_head = nn.Linear(hidden, 3)
        self.traj_head = nn.Linear(hidden, N_POINTS * 3) if traj_head else None
        self.detach_traj = detach_traj

    def forward(self, x):
        f = self.trunk(x)
        out = {
            "axis": self.axis_head(f),
            "origin": self.origin_head(f),
            "p0": self.p0_head(f),
        }
        if self.traj_head is not None:
            tf = f.detach() if self.detach_traj else f
            out["traj"] = self.traj_head(tf).view(-1, N_POINTS, 3)
        return out


def make_rich_data(n_train=300, n_test=1500, dim=32, feat_noise=0.6):
    g = torch.Generator().manual_seed(7)
    W_axis = torch.randn(dim, 3, generator=g) / np.sqrt(dim)
    W_org = torch.randn(dim, 3, generator=g) / np.sqrt(dim)
    W_p0 = torch.randn(dim, 3, generator=g) / np.sqrt(dim)

    def build(n, seed):
        gg = torch.Generator().manual_seed(seed)
        x_clean = torch.randn(n, dim, generator=gg)
        axis = F.normalize(x_clean @ W_axis, dim=-1)
        origin = x_clean @ W_org
        p0 = x_clean @ W_p0 + torch.tensor([[0.0, 0.0, 2.0]])
        lever0 = p0 - origin
        arc = arc_points(axis, lever0)          # GT curve from the conjunction
        x = x_clean + feat_noise * torch.randn(n, dim, generator=gg)
        return x, axis, origin, p0, arc

    return build(n_train, 11), build(n_test, 13)


def train_trunk(mode, data_tr, steps=4000, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    x, axis, origin, p0, arc = data_tr
    model = TrunkModel(traj_head=(mode in ("aux_head", "aux_head_detached")),
                       detach_traj=(mode == "aux_head_detached"))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        o = model(x)
        loss = (loss_angular(o["axis"], axis)
                + F.mse_loss(o["origin"], origin)
                + F.mse_loss(o["p0"], p0))
        if mode in ("aux_head", "aux_head_detached"):
            loss = loss + F.mse_loss(o["traj"], arc)
        elif mode == "analytic":
            dec = arc_points(o["axis"], o["p0"] - o["origin"])
            err = (dec - arc).pow(2).sum(-1).mean(-1)
            energy = arc.pow(2).sum(-1).mean(-1)
            loss = loss + (err / energy.clamp(min=1e-4)).mean()
        loss.backward()
        opt.step()
    return model


def eval_trunk(model, data_te):
    x, axis, origin, p0, _ = data_te
    with torch.no_grad():
        o = model(x)
        errs = angle_err_deg(o["axis"], axis)
        return {
            "axis_med_deg": errs.median().item(),
            "flip_pct": (errs > 90).float().mean().item() * 100,
            "origin_mse": F.mse_loss(o["origin"], origin).item(),
            "p0_mse": F.mse_loss(o["p0"], p0).item(),
        }


def run_ablation_miniature(out_dir, seeds=(0, 1, 2, 3, 4)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_tr, data_te = make_rich_data()
    modes = ["baseline", "aux_head", "aux_head_detached", "analytic"]
    results = {m: [] for m in modes}
    for m in modes:
        for s in seeds:
            model = train_trunk(m, data_tr, seed=s)
            results[m].append(eval_trunk(model, data_te))
        med = np.median([r["axis_med_deg"] for r in results[m]])
        flp = np.median([r["flip_pct"] for r in results[m]])
        org = np.median([r["origin_mse"] for r in results[m]])
        print(f"MINIATURE {m}: axis_med={med:.2f} deg  flip%={flp:.2f}  origin_mse={org:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(modes))
    meds = [np.median([r["axis_med_deg"] for r in results[m]]) for m in modes]
    spread = [np.std([r["axis_med_deg"] for r in results[m]]) for m in modes]
    ax.bar(xs, meds, yerr=spread, capsize=4,
           color=["gray", "tab:blue", "tab:cyan", "tab:orange"])
    ax.set_xticks(xs)
    ax.set_xticklabels(["no traj\n(arm B)", "traj HEAD\n(arm D)",
                        "traj head,\nDETACHED trunk", "analytic\nDECODE"])
    ax.set_ylabel("test axis error, median deg (5 seeds)")
    ax.set_title("Ablation miniature: where does the trajectory gain come from?")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_miniature.png"), dpi=140)
    plt.close(fig)
