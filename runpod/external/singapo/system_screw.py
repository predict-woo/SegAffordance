"""SINGAPO + closed-form screw loss (x0-space auxiliary term).

Registered as `sys_singapo_screw`; identical to `sys_singapo` when
screw_weight == 0. Dropped into the upstream repo as systems/system_screw.py
(no upstream file is edited). Node layout (K nodes x 5 tokens x 6 dims):
  token0 = [aabb_max(3), aabb_min(3)], token1 = joint_type x6,
  token2 = [axis_dir(3) (*0.7, sign-canonicalised), axis_origin(3)],
  token3 = range x3, token4 = label x6.
joint_ref: 1 fixed, 2 revolute, 3 prismatic, 4 screw, 5 continuous.
"""
import torch
import systems
from systems.system import SingapoSystem
from screw_loss import screw_terms, combine_terms


@systems.register("sys_singapo_screw")
class SingapoScrewSystem(SingapoSystem):

    def compute_loss(self, batch, inputs, outputs):
        loss, loss_dict = super().compute_loss(batch, inputs, outputs)
        w = float(self.hparams.get("screw_weight", 0.0))
        if w <= 0.0:
            return loss, loss_dict

        x0 = inputs["x"].float()                       # (B', K*5, 6) GT, repeated
        xt = inputs["noisy_x"].float()
        eps_hat = outputs["noise_pred"].float()
        t = inputs["timesteps"]
        abar = self.scheduler.alphas_cumprod.to(xt.device)[t].view(-1, 1, 1).float()
        x0_hat = (xt - (1.0 - abar).sqrt() * eps_hat) / abar.sqrt()

        B, N, C = x0.shape
        K = N // 5
        g = x0.view(B, K, 5, 6)
        p = x0_hat.view(B, K, 5, 6)
        valid = self.prepare_loss_mask(batch)["valid_nodes"].view(B, K, 5, 6)[:, :, 0, 0]

        jt = ((g[:, :, 1, :].mean(-1) + 0.5) * 5.0).round().clamp(1, 5)
        is_rot = (jt == 2) | (jt == 5)
        is_trans = (jt == 3) | (jt == 4)
        nonfixed = is_rot | is_trans

        n_gt, o_gt = g[:, :, 2, 0:3], g[:, :, 2, 3:6]
        c_gt = 0.5 * (g[:, :, 0, 0:3] + g[:, :, 0, 3:6])
        n_p, o_p = p[:, :, 2, 0:3], p[:, :, 2, 3:6]
        c_p = 0.5 * (p[:, :, 0, 0:3] + p[:, :, 0, 3:6])
        if bool(self.hparams.get("screw_detach_point", True)):
            c_p = c_p.detach()

        f = lambda a: a.reshape(B * K, -1)
        terms = screw_terms(
            is_rot.reshape(-1), f(n_p), f(o_p), f(c_p), f(n_gt), f(o_gt), f(c_gt),
            min_radius=float(self.hparams.get("screw_min_radius", 0.05)),
        )
        # min-SNR weighting of the x0-space term (gamma=5), normalised to (0,1]
        gamma = float(self.hparams.get("screw_snr_gamma", 5.0))
        snr = (abar / (1.0 - abar)).view(B)
        row_w = (snr.clamp(max=gamma) / gamma).view(B, 1).expand(B, K).reshape(-1)

        loss_screw, logs = combine_terms(
            terms, (valid & nonfixed).reshape(-1),
            w_h1=float(self.hparams.get("screw_w_h1", 1.0)),
            w_pos=float(self.hparams.get("screw_w_pos", 0.0)),
            w_anchor=float(self.hparams.get("screw_w_anchor", 0.5)),
            row_weight=row_w,
        )
        loss = loss + w * loss_screw
        loss_dict["train/loss_screw"] = loss_screw.detach()
        loss_dict["train/loss_total"] = loss.detach()
        for k, v in logs.items():
            loss_dict[f"train/{k}"] = torch.tensor(v, device=loss.device)
        return loss, loss_dict
