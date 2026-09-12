"""FieldModel — "one map, many fields" (2026-09-13, user request after the
readout study: docs/superpowers/specs/2026-09-13-field-model-design.md).

The trunk is CRIS's: frozen DINOv3 + multi-tap adapter -> text-gated FPN ->
transformer decoder over the word tokens -> one decoded map fq (B, C, H/16,
W/16) -> the dynamic-kernel projector's three heatmaps at H/4 (mask, point,
origin). Everything articulation-related is then a FIELD on fq and every
prediction is a geometric readout of the fields:

  field channels (FieldHead): rot axis 3 | trans direction 3 | type logits 2 |
      offset-to-hinge 2 | log-depth 1 | arc length 1
  axis / type / arc length = part-weighted means of the fields
  origin_uv               = part-weighted mean of (pixel uv + offset)  (votes)
  point_uv                = soft-argmax of the point heatmap
  z_p / z_q               = exp(log-depth field) sampled at point_uv / origin_uv
  point_3d / origin_3d    = lifts with the batch intrinsics (as gen-7)
  trajectory              = analytic decoder from the above (as 2026-09-11)

Part weights are the DETACHED predicted mask probability (no GT-mask teacher
forcing). No pooled condition vector, no scalar MLP heads, no legacy modes
(CVAE / twist / WTA / trajectory heads / type hint / direct 3D heads / depth
encoder). ModelParams is only the carrier of the trunk hyper-parameters +
`dense_hidden` (field head width), `trajectory_decoder_max_angle`,
`trajectory_decoder_lever_floor`, `trajectory_scale_free`, `channels_last`.
Returns the same ModelOutputs the SF3D trainer consumes, so the losses,
metrics, probes and viz tools apply unchanged. CRIS (model/segmenter.py) is
untouched — legacy experiments stay reproducible.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones import build_backbone
from model.losses.geometric import analytic_decode_curves, backproject_points
from model.outputs import ModelOutputs

from .layers import FPN, Projector_Mult, TransformerDecoder
from .segmenter import soft_argmax2d


class FieldHead(nn.Module):
    """Per-pixel articulation fields on the decoded map (12 channels)."""

    ROT, TRANS, TYPE, OFF, LOGZ, LEN = slice(0, 3), slice(3, 6), slice(6, 8), slice(8, 10), 10, 11
    LENGTH_BIAS_INIT = -0.7  # softplus(-0.7) ~ 0.4: the scale-free arc length the DCT/analytic heads start at

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1),
            nn.GroupNorm(32, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(32, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 12, 1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight[self.OFF]); nn.init.zeros_(last.bias[self.OFF])   # votes start at the part centroid
        nn.init.zeros_(last.weight[self.LOGZ]); nn.init.constant_(last.bias[self.LOGZ], math.log(1.5))  # ~1.5 m
        nn.init.zeros_(last.weight[self.LEN]); nn.init.constant_(last.bias[self.LEN], self.LENGTH_BIAS_INIT)

    def forward(self, fq: torch.Tensor) -> torch.Tensor:
        return self.net(fq).float()


class FieldModel(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.register_buffer("_rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.channels_last = bool(getattr(model_params, "channels_last", False))

        # ---- trunk (CRIS's, RGB only) ----
        self.backbone = build_backbone(model_params, list(model_params.fpn_in))
        state_dim = self.backbone.state_dim
        self.neck = FPN(in_channels=list(model_params.fpn_in), out_channels=list(model_params.fpn_out), text_dim=state_dim)
        d_model = model_params.fpn_out[1]
        self.word_proj = nn.Linear(self.backbone.word_dim, d_model) if self.backbone.word_dim != d_model else None
        self.decoder = TransformerDecoder(
            num_layers=model_params.num_layers, d_model=d_model, nhead=model_params.num_head,
            dim_ffn=model_params.dim_ffn, dropout=model_params.dropout, return_intermediate=False,
        )
        # mask | point | origin heatmaps at H/4 (the origin channel is an auxiliary target;
        # the origin used downstream is the vote)
        self.proj = Projector_Mult(state_dim, d_model // 2, 3, out_channels=3,
                                   proj_dropout=model_params.proj_dropout)

        # ---- fields ----
        self.dense_head = FieldHead(d_model, hidden=int(getattr(model_params, "dense_hidden", 256)))
        self.dense_trunk_detach = False       # set per batch by the trainer (loss profile)

        # ---- decoder conventions (shared with CRIS's analytic decoder) ----
        self.trajectory_decoder = "analytic"
        self.trajectory_decoder_length = getattr(model_params, "trajectory_decoder_length", "head")
        self.trajectory_decoder_max_angle = float(getattr(model_params, "trajectory_decoder_max_angle", math.pi))
        self.trajectory_decoder_lever_floor = float(getattr(model_params, "trajectory_decoder_lever_floor", 0.02))
        self.trajectory_scale_free = bool(getattr(model_params, "trajectory_scale_free", True))
        # trainer contract: the flags the SF3D trainer reads off the model with
        # getattr (CRIS's mode switches) — fixed values for the field design
        self.use_motion_type_input = False
        self.use_cvae = False
        self.split_axis_heads = True
        self.use_motion_head = True
        self.use_motion_type_head = True
        self.use_twist_head = False
        self.twist_num_hypotheses = 1
        self.use_trajectory_head = False
        self.use_2d_trajectory_head = False
        self.point_prediction_3d = False
        self.use_origin_head = False
        self.use_origin_heatmap = True
        self.use_origin_local_feature = False
        self.predict_point_depth = True
        self.predict_origin_depth = False
        self.pool_with_predicted_mask = True
        self.text_cost_map = False
        self.min_depth, self.max_depth = 0.1, 20.0

        if self.channels_last:
            self.to(memory_format=torch.channels_last)

    # trainer contract
    def tokenize(self, texts, context_length):
        return self.backbone.tokenize(texts, context_length)

    @staticmethod
    def _sample(field: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Bilinear sample of a (B, 1, h, w) field at (B, 2) normalised uv -> (B,)."""
        grid = (uv.detach().float().view(-1, 1, 1, 2) * 2.0 - 1.0)
        return F.grid_sample(field.float(), grid, align_corners=False, padding_mode="border").view(-1)

    def forward(self, img, depth=None, word=None, mask=None, interaction_point=None, motion_gt=None,
                motion_type_input=None, intrinsics_norm=None):
        if img.dtype == torch.uint8:
            img = (img.float().div_(255.0) - self._rgb_mean) / self._rgb_std
        if self.channels_last:
            img = img.to(memory_format=torch.channels_last)
        pad_mask = self.backbone.pad_mask(word)
        vis = self.backbone.encode_image(img)
        word_feat, state = self.backbone.encode_text(word)
        if self.word_proj is not None:
            word_feat = self.word_proj(word_feat)
        fq = self.neck(vis, state)                                  # (B, C, h, w)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word_feat, pad_mask).reshape(b, c, h, w)
        maps = self.proj(fq, state)                                 # (B, 3, H/4, W/4)
        mask_logits, point_logits, origin_logits = maps[:, 0:1], maps[:, 1:2], maps[:, 2:3]
        _, _, H_map, W_map = point_logits.shape
        wh = torch.tensor([W_map, H_map], dtype=point_logits.dtype, device=point_logits.device)
        point_uv = soft_argmax2d(point_logits) / wh

        # ---- fields and their readouts ----
        feat = fq.detach() if self.dense_trunk_detach else fq
        fields = self.dense_head(feat)                              # (B, 12, h, w) fp32
        # Part weights = the detached predicted mask, FLOORED: an empty or
        # near-empty prediction (early training, a miss) must not turn the
        # weighted means into 1/wsum amplifiers — run 1 diverged exactly so
        # (val L_mask 5 -> 133 from epoch 6). Mixing in a uniform floor worth
        # one cell keeps wsum >= 1 and the readout finite; with a confident
        # mask the floor is negligible (1 / (h*w) per cell).
        weights = F.interpolate(torch.sigmoid(mask_logits.detach().float()), size=(h, w),
                                mode="bilinear", align_corners=False)   # predicted part, no teacher forcing
        weights = weights + 1.0 / float(h * w)
        wsum = weights.sum(dim=(-1, -2)).clamp(min=1.0)

        def wmean(x):
            return (x * weights).sum(dim=(-1, -2)) / wsum

        motion_pred_rot = wmean(torch.tanh(fields[:, FieldHead.ROT]))
        motion_pred_trans = wmean(torch.tanh(fields[:, FieldHead.TRANS]))
        motion_type_logits = wmean(fields[:, FieldHead.TYPE])
        ys = (torch.arange(h, device=fq.device, dtype=torch.float32) + 0.5) / h
        xs = (torch.arange(w, device=fq.device, dtype=torch.float32) + 0.5) / w
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=0)          # (2, h, w): u, v
        vote_uv = grid[None] + fields[:, FieldHead.OFF]
        origin_uv = wmean(vote_uv)
        length = F.softplus(wmean(fields[:, FieldHead.LEN:FieldHead.LEN + 1]).view(-1))
        depth_field = fields[:, FieldHead.LOGZ:FieldHead.LOGZ + 1]
        z_p = self._sample(depth_field, point_uv).exp().clamp(self.min_depth, self.max_depth)
        z_q = self._sample(depth_field, origin_uv).exp().clamp(self.min_depth, self.max_depth)

        pred_stack = torch.stack([motion_pred_trans, motion_pred_rot], dim=1)
        sel = motion_type_logits.argmax(dim=-1)
        motion_pred = pred_stack[torch.arange(b, device=fq.device), sel]

        # ---- lifts + analytic decoder ----
        point_3d_pred = origin_pred = None
        trajectory_pred = trajectory_pred_rot = trajectory_pred_trans = None
        if intrinsics_norm is not None:
            K = intrinsics_norm.float()
            point_3d_pred = backproject_points(K, point_uv.float(), z_p)
            origin_pred = backproject_points(K, origin_uv.float(), z_q)
            p3, o3 = point_3d_pred, origin_pred
            if self.trajectory_scale_free:
                zdiv = z_p.view(-1, 1).clamp(min=1e-3)
                p3, o3 = p3 / zdiv, o3 / zdiv
            traj_len = length
            if self.trajectory_decoder_length == "writer":
                n_hat = F.normalize(motion_pred_rot, p=2, dim=1, eps=1e-8)
                rel = p3 - o3
                radius = (rel - (rel * n_hat).sum(-1, keepdim=True) * n_hat).norm(dim=-1)
                trans_len = 0.7 / zdiv.view(-1) if self.trajectory_scale_free else torch.full_like(radius, 0.7)
                traj_len = torch.where(sel == 1, (math.pi / 2.0) * radius, trans_len).detach()
            trajectory_pred_rot, trajectory_pred_trans, _ = analytic_decode_curves(
                motion_pred_rot, motion_pred_trans, o3, p3, traj_len, num_points=20,
                max_angle=self.trajectory_decoder_max_angle, lever_floor=self.trajectory_decoder_lever_floor,
            )
            is_rot = (sel == 1)[:, None, None]
            trajectory_pred = torch.where(is_rot, trajectory_pred_rot, trajectory_pred_trans)

        dt = mask_logits.dtype
        return ModelOutputs(
            mask_logits=mask_logits,
            point_logits=point_logits,
            point_uv=point_uv,
            motion_pred=motion_pred.to(dt),
            motion_pred_rot=motion_pred_rot.to(dt),
            motion_pred_trans=motion_pred_trans.to(dt),
            motion_type_logits=motion_type_logits.to(dt),
            trajectory_pred=trajectory_pred,
            trajectory_pred_rot=trajectory_pred_rot,
            trajectory_pred_trans=trajectory_pred_trans,
            trajectory_length=length,
            origin_depth=z_q,
            point_3d_pred=point_3d_pred,
            origin_pred=origin_pred,
            origin_uv=origin_uv.to(dt),
            origin_logits=origin_logits,
            origin_vote_uv=vote_uv,
            vote_weights=weights,
            depth_field=depth_field,
        )
