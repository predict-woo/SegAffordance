import torch
import torch.nn as nn
import torch.nn.functional as F


from model.backbones import build_backbone
from model.losses.geometric import backproject_points
from model.outputs import ModelOutputs

from .layers import (
    FPN,
    Projector,
    TransformerDecoder,
    Projector_Mult,
    MotionVAE,
    MotionMLP,
    DepthEncoder,
    TrajectoryMLP,
    Trajectory2DMLP,
    OriginDepthHead,
    TwistMLP,
    Point3DHead,
)


def soft_argmax2d(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: Bx1xHxW  - unnormalised.
    returns: Bx2  - (x, y) in pixel units.
    """
    B, _, H, W = logits.shape
    prob = logits.view(B, -1).softmax(-1)  # B × (H·W)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=logits.device),
        torch.linspace(0, W - 1, W, device=logits.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], 0)  # 2×(H·W)
    return prob @ grid.T


class CRIS(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        # The fast input pipeline (datasets/scenefun3d.py fast_pipeline)
        # ships raw uint8 RGB and normalization happens here on the GPU —
        # per-sample CPU normalization was a top cost under RunPod's cgroup
        # CPU quota. Constants match get_default_transforms (ImageNet).
        self.register_buffer(
            "_rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        # NHWC for the conv stack (see ModelParams.channels_last). Parameters
        # are converted here; forward converts the img/depth inputs to match.
        self.channels_last = getattr(model_params, "channels_last", False)
        # Gen-15 (stage C): patch-text cost channels + local-half FPN gate.
        self.text_cost_map = getattr(model_params, "text_cost_map", False)
        if self.text_cost_map and not (
            getattr(model_params, "backbone", "clip_rn50") == "dinov3"
            and getattr(model_params, "text_source", "dinotxt") == "dinotxt"
        ):
            raise ValueError(
                "text_cost_map needs the dinov3 backbone with dino.txt text "
                "(the cost map is cosine vs the dino.txt-aligned patch tokens)"
            )
        ## Vision & Text Encoder
        # encode_image (B, 3, H, W) -> v2: (B, fpn_in[0], H/8, W/8), v3: (B, fpn_in[1], H/16, W/16), v4: (B, fpn_in[2], H/32, W/32)
        # encode_text (B, L: word_len) -> word_features: (B, L, backbone.word_dim), state: (B, backbone.state_dim)
        self.backbone = build_backbone(model_params, list(model_params.fpn_in))
        state_dim = self.backbone.state_dim

        # Optional depth usage
        self.use_depth = getattr(model_params, "use_depth", True)
        if self.use_depth:
            self.depth_encoder = DepthEncoder(
                in_channels=1, out_channels=model_params.depth_feat_channels
            )
        else:
            self.depth_encoder = None

        ## Multi-Modal FPN
        # The first two channel counts are increased by the depth feature channels only if depth is enabled
        if self.use_depth:
            fpn_in_channels = [
                model_params.fpn_in[0] + model_params.depth_feat_channels[0],
                model_params.fpn_in[1] + model_params.depth_feat_channels[1],
                model_params.fpn_in[2],
            ]
        else:
            fpn_in_channels = [
                model_params.fpn_in[0],
                model_params.fpn_in[1],
                model_params.fpn_in[2],
            ]
        # forward: v2, v3, v4 -> fq: (B, fpn_out[1], H/16, W/16)
        if self.text_cost_map:
            # Two cost channels appended per level; the gate reads only the
            # patch-aligned LOCAL half of the state (the global CLS half is
            # not what the aligned patch tokens live in).
            fpn_in_channels = [c + 2 for c in fpn_in_channels]
            fpn_text_dim = state_dim // 2
        else:
            fpn_text_dim = state_dim
        self.neck = FPN(
            in_channels=fpn_in_channels,
            out_channels=model_params.fpn_out,
            text_dim=fpn_text_dim,
        )

        ## Text token projection
        # The decoder cross-attends visual queries (d_model = fpn_out[1]) against
        # word tokens. CLIP RN50's 512-d word features happen to match; wider
        # text towers (SigLIP 2, dino.txt) need projecting down.
        d_model = model_params.fpn_out[1]
        if self.backbone.word_dim != d_model:
            self.word_proj = nn.Linear(self.backbone.word_dim, d_model)
        else:
            self.word_proj = None

        ## Decoder
        # fq: (B, fpn_out[1], H/16, W/16), word: (B, L, D_text: transformer_width), pad_mask: (B, L) -> fq: (B, fpn_out[1], H/16, W/16)
        self.decoder = TransformerDecoder(
            num_layers=model_params.num_layers,
            d_model=model_params.fpn_out[1],
            nhead=model_params.num_head,
            dim_ffn=model_params.dim_ffn,
            dropout=model_params.dropout,
            return_intermediate=model_params.intermediate,
        )

        self.point_prediction_3d = getattr(model_params, "point_prediction_3d", False)
        self.use_origin_head = getattr(model_params, "use_origin_head", False)
        self.pool_with_predicted_mask = getattr(
            model_params, "pool_with_predicted_mask", False
        )

        # gen-7 (docs/superpowers/specs/2026-08-14-heatmap-depth-lift-gen7-
        # design.md): third projector channel -> soft-argmax origin_uv ->
        # condition-only depth z_q; grid_sample local feature -> point depth
        # z_p; both lifted to 3D with the batch intrinsics when provided.
        self.use_origin_heatmap = getattr(model_params, "use_origin_heatmap", False)
        self.predict_point_depth = getattr(model_params, "predict_point_depth", False)
        if self.use_origin_heatmap and self.point_prediction_3d:
            raise ValueError("use_origin_heatmap needs the classical 2D point path")
        if self.predict_point_depth and self.point_prediction_3d:
            raise ValueError("predict_point_depth needs the classical 2D point path")
        self.use_origin_local_feature = getattr(
            model_params, "use_origin_local_feature", False
        )
        if self.use_origin_local_feature and not self.use_origin_heatmap:
            raise ValueError(
                "use_origin_local_feature needs use_origin_heatmap: the "
                "sample location IS the origin heatmap's soft-argmax"
            )

        ## Projector
        # fq: (B, fpn_out[1], H/16, W/16), state: (B, fpn_in[2]) -> maps: (B, C, H/4, W/4)
        # channels: 1 (3D mode: mask only) | 2 (classical: mask, point) |
        # 3 (gen-7: mask, point, origin heatmap)
        self.proj = Projector_Mult(
            state_dim,
            model_params.fpn_out[1] // 2,
            3,
            out_channels=(
                1 if self.point_prediction_3d
                else (3 if self.use_origin_heatmap else 2)
            ),
            proj_dropout=model_params.proj_dropout,
        )

        # motion_features + global_vis_features + global_text_features
        vae_feature_dim = model_params.fpn_out[1] + model_params.fpn_in[2] + state_dim
        # 2D path: condition = features + point_uv (2). 3D path: the point
        # head consumes the base features and its 3-dim output extends the
        # condition instead (spec: "the same role point_uv played, in 3D").
        point_cond_dim = 3 if self.point_prediction_3d else 2
        vae_condition_dim = vae_feature_dim + point_cond_dim
        # gen-7: origin_uv extends the condition LAST (after the classical
        # [features, point_uv, type_emb] layout — see forward).
        if self.use_origin_heatmap:
            vae_condition_dim += 2

        # Optional auxiliary input: the articulation type, embedded and
        # concatenated into the condition vector every head consumes (LAST
        # on the 2D path — the classical checkpoint layout; before the
        # 3D-point extension on the 3D path; see forward). One extra
        # learned NULL token is the "no hint" state — used for conditioning
        # dropout during training and ALWAYS at val/test, so the model never
        # depends on the hint being available. Must be set up before the
        # heads below so vae_condition_dim includes the embedding.
        self.use_motion_type_input = getattr(model_params, "use_motion_type_input", False)
        if self.use_motion_type_input:
            type_emb_dim = getattr(model_params, "motion_type_embedding_dim", 16)
            self.motion_type_embedding = nn.Embedding(
                model_params.num_motion_types + 1, type_emb_dim
            )
            vae_condition_dim += type_emb_dim
        else:
            self.motion_type_embedding = None

        self.use_cvae = getattr(model_params, "use_cvae", True)
        self.use_motion_type_head = getattr(model_params, "use_motion_type_head", True)
        if self.use_cvae:
            self.motion_vae = MotionVAE(
                feature_dim=vae_feature_dim,
                condition_dim=vae_condition_dim,
                latent_dim=model_params.vae_latent_dim,
                hidden_dim=model_params.vae_hidden_dim,
                num_motion_types=model_params.num_motion_types,
            )
            self.motion_mlp = None
        else:
            self.motion_vae = None
            # MLP will take the same condition used by the VAE decoder.
            # With BOTH sub-heads off (twist-only arms) skip the module
            # entirely — its shared backbone would otherwise run every step
            # feeding nothing.
            _with_type = getattr(model_params, "use_motion_type_head", True)
            _with_motion = getattr(model_params, "use_motion_head", True)
            self.motion_mlp = MotionMLP(
                input_dim=vae_condition_dim,
                hidden_dim=model_params.vae_hidden_dim,
                num_motion_types=model_params.num_motion_types,
                with_type_head=_with_type,
                with_motion_head=_with_motion,
            ) if (_with_type or _with_motion) else None

        # 3D element-sweep head. Off during 2D pretraining: video data has no
        # element-sweep GT, so it would receive no gradient at all. It is a
        # leaf — nothing consumes its output and vae_condition_dim is computed
        # independently — so removing it changes nothing upstream.
        # K > 1: winner-takes-all articulation bundles — the trajectory head
        # emits one trajectory per twist hypothesis, selected jointly by the
        # twist head's logits (2026-08-11 WTA spec). Requires the twist head.
        self.twist_num_hypotheses = getattr(model_params, "twist_num_hypotheses", 1)
        self.use_trajectory_head = getattr(model_params, "use_trajectory_head", True)
        if self.use_trajectory_head:
            self.trajectory_predictor = TrajectoryMLP(
                input_dim=vae_condition_dim,
                hidden_dim=model_params.vae_hidden_dim,  # reuse this param
                num_points=20,
                delta_cumsum=getattr(model_params, "trajectory_delta_cumsum", False),
                num_hypotheses=self.twist_num_hypotheses,
                absolute=getattr(model_params, "trajectory_absolute", False),
            )
        else:
            self.trajectory_predictor = None

        # 2D hand/contact-track head, for pretraining on mined video.
        self.use_2d_trajectory_head = getattr(model_params, "use_2d_trajectory_head", False)
        if self.use_2d_trajectory_head:
            self.trajectory_2d_predictor = Trajectory2DMLP(
                input_dim=vae_condition_dim,
                hidden_dim=model_params.vae_hidden_dim,
                num_points=20,
                delta_cumsum=getattr(model_params, "trajectory_delta_cumsum", False),
            )
        else:
            self.trajectory_2d_predictor = None

        # Unified screw-motion head: one 6-vector for both motion types
        # (model/losses/twist.py). Additive — the axis/type heads keep
        # training alongside it so runs stay comparable to the baseline and
        # OPD batches (which cannot supervise a twist: no 3D origin) still
        # train every head they used to.
        self.use_twist_head = getattr(model_params, "use_twist_head", False)
        if self.use_twist_head:
            self.twist_predictor = TwistMLP(
                input_dim=vae_condition_dim,
                hidden_dim=model_params.vae_hidden_dim,
                pitch_free=getattr(model_params, "twist_pitch_free", False),
                num_hypotheses=self.twist_num_hypotheses,
            )
        else:
            self.twist_predictor = None
        if self.twist_num_hypotheses > 1 and not (
            self.use_twist_head and self.use_trajectory_head
        ):
            raise ValueError(
                "twist_num_hypotheses > 1 needs both use_twist_head and "
                "use_trajectory_head: hypotheses are (twist, trajectory) "
                "bundles selected by one set of logits."
            )

        # gen-6 split arm: direct 3D interaction-point head and revolute
        # joint-origin head. The point head consumes the BASE condition
        # (vae_condition_dim - 3 strips its own 3-dim extension — it cannot
        # consume its own output); the origin head consumes the EXTENDED
        # condition.
        self.point_3d_head = (
            Point3DHead(
                input_dim=vae_condition_dim - 3, hidden_dim=model_params.vae_hidden_dim
            )
            if self.point_prediction_3d
            else None
        )
        self.origin_head = (
            Point3DHead(
                input_dim=vae_condition_dim, hidden_dim=model_params.vae_hidden_dim
            )
            if self.use_origin_head
            else None
        )

        # Metric depth of the joint origin, completing {type, axis, origin}.
        self.predict_origin_depth = getattr(model_params, "predict_origin_depth", False)
        if self.predict_origin_depth:
            self.origin_depth_head = OriginDepthHead(
                input_dim=vae_condition_dim,
                hidden_dim=model_params.vae_hidden_dim,
            )
        else:
            self.origin_depth_head = None

        # gen-7 asymmetric depth heads (vae_condition_dim is FINAL here).
        # z_p (interaction point): condition + a fpn_out[1]-dim grid_sample
        # of the decoded fq at point_uv — depth needs local appearance.
        # z_q (origin): condition only by default — the origin is often
        # occluded or off-object, so no local sample. Gen-11
        # (use_origin_local_feature) adds the mirror grid_sample at
        # origin_uv. NOT the same module as the 2D arm's origin_depth_head
        # (predict_origin_depth) above.
        self.point_depth_head = (
            OriginDepthHead(
                input_dim=vae_condition_dim + model_params.fpn_out[1],
                hidden_dim=model_params.vae_hidden_dim,
            )
            if self.predict_point_depth
            else None
        )
        self.origin_depth_head_g7 = (
            OriginDepthHead(
                input_dim=vae_condition_dim
                + (model_params.fpn_out[1] if self.use_origin_local_feature else 0),
                hidden_dim=model_params.vae_hidden_dim,
            )
            if self.use_origin_heatmap
            else None
        )

        if self.channels_last:
            self.to(memory_format=torch.channels_last)

    def tokenize(self, texts, context_length):
        """Tokenize with whatever tokenizer the active backbone was trained on."""
        return self.backbone.tokenize(texts, context_length)

    def forward(
        self, img, depth, word, mask, interaction_point, motion_gt=None,
        motion_type_input=None, intrinsics_norm=None,
    ):
        """
        img: (B, 3, H, W)
        depth: (B, 1, H, W)
        word: (B, L)
        mask: (B, 1, H, W)
        interaction_point: (B, 2) -> value in [0,1]
        motion_gt: (B, 3) -> ground truth motion vector
        motion_type_input: (B,) long in [0, num_motion_types] — the auxiliary
            type hint (num_motion_types = the NULL token). None means NULL
            for every sample. Ignored unless use_motion_type_input.
        intrinsics_norm: (B, 3, 3) NORMALIZED camera intrinsics (pixels
            divided by image size — model/losses/geometric.py
            normalize_intrinsics). When provided, the gen-7 scalar depths
            z_p/z_q are lifted with it into point_3d_pred/origin_pred; when
            None those lifted fields stay None.
        """
        if img.dtype == torch.uint8:
            # Fast-pipeline input: normalize on device (see __init__).
            img = (img.float().div_(255.0) - self._rgb_mean) / self._rgb_std
        if self.channels_last:
            img = img.to(memory_format=torch.channels_last)
            if depth is not None:
                depth = depth.to(memory_format=torch.channels_last)

        # padding mask used in decoder

        pad_mask = self.backbone.pad_mask(word)

        # vis: x2, x3, x4
        # word: b, length, word_dim
        # state: b, state_dim
        if self.text_cost_map:
            vis, _bb_extras = self.backbone.encode_image_full(img)  # v2, v3, v4
        else:
            vis = self.backbone.encode_image(img)  # v2, v3, v4
        word, state = self.backbone.encode_text(word)
        if self.word_proj is not None:
            word = self.word_proj(word)

        # --- Depth Fusion (optional) ---
        if self.use_depth:
            depth_feat_8, depth_feat_16 = self.depth_encoder(depth)  # type: ignore[operator]
            # v2 is H/8, v3 is H/16
            v2_fused = torch.cat([vis[0], depth_feat_8], dim=1)
            v3_fused = torch.cat([vis[1], depth_feat_16], dim=1)
            vis_fused = (v2_fused, v3_fused, vis[2])
        else:
            vis_fused = vis

        # --- Patch-text cost map (gen-15, optional) ---
        if self.text_cost_map:
            aligned_raw = _bb_extras["aligned_map"]
            half = state.shape[1] // 2
            if aligned_raw.shape[1] != half:
                # A mis-paired backbone/text checkpoint should fail loudly,
                # not silently miscompute the cosine.
                raise RuntimeError(
                    f"aligned_map has {aligned_raw.shape[1]} channels but the "
                    f"text state half is {half} — dino.txt pairs the patch "
                    "tokens with the state's local half"
                )
            aligned = F.normalize(aligned_raw.float(), dim=1)
            t_glob = F.normalize(state[:, :half].float(), dim=1)
            t_loc = F.normalize(state[:, half:].float(), dim=1)
            cost_g = torch.einsum("bchw,bc->bhw", aligned, t_glob)[:, None]
            cost_l = torch.einsum("bchw,bc->bhw", aligned, t_loc)[:, None]
            costs = torch.cat([cost_g, cost_l], dim=1).to(vis_fused[0].dtype)
            vis_fused = tuple(
                torch.cat(
                    [v, F.interpolate(costs, size=v.shape[-2:], mode="bilinear",
                                      align_corners=False)],
                    dim=1,
                )
                for v in vis_fused
            )
            state_for_gate = state[:, half:]
        else:
            state_for_gate = state

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis_fused, state_for_gate)  # fq: (B, fpn_out[1], H/16, W/16)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)

        maps = self.proj(fq, state)  # B×C×H_map×W_map (where H_map=h*4, W_map=w*4)
        mask_pred = maps[:, 0:1]  # B×1×H_map×W_map
        if self.point_prediction_3d:
            point_pred = None
            point_uv = None
        else:
            point_pred = maps[:, 1:2]  # B×1×H_map×W_map, logits, not sigmoided yet
            # soft-argmax → differentiable coordinates in pixel space of the map
            # point_px will have x in [0, W_map-1] and y in [0, H_map-1]
            point_px = soft_argmax2d(point_pred)  # B×2
            _, _, H_map, W_map = point_pred.shape
            point_uv = point_px / torch.tensor(
                [W_map, H_map], dtype=point_px.dtype, device=point_px.device
            )

        # gen-7: origin heatmap channel -> soft-argmax origin_uv in [0, 1].
        origin_uv = None
        origin_logits = None
        if self.use_origin_heatmap:
            origin_logits = maps[:, 2:3]
            _, _, H_map, W_map = origin_logits.shape
            origin_px = soft_argmax2d(origin_logits)
            origin_uv = origin_px / torch.tensor(
                [W_map, H_map], dtype=origin_px.dtype, device=origin_px.device
            )

        # --- Motion VAE part ---
        fq_h, fq_w = fq.shape[-2:]

        if self.training and not self.pool_with_predicted_mask:
            # Classical teacher forcing: pool under the GT mask.
            mask_for_pooling = F.interpolate(
                mask.float(), size=(fq_h, fq_w), mode="bilinear", align_corners=False
            )
        else:
            # Deployment-consistent pooling: the predicted mask, detached in
            # train mode so articulation losses cannot steer the mask head
            # through the pooling path (it learns only from DiceBCE).
            mask_prob = torch.sigmoid(mask_pred)
            if self.training:
                mask_prob = mask_prob.detach()
            mask_for_pooling = F.interpolate(
                mask_prob, size=(fq_h, fq_w), mode="bilinear", align_corners=False
            )

        mask_sum = mask_for_pooling.sum(dim=(-1, -2), keepdim=True) + 1e-8
        pooled_features = (fq * mask_for_pooling).sum(
            dim=(-1, -2), keepdim=True
        ) / mask_sum
        motion_features = pooled_features.squeeze(-1).squeeze(-1)

        # Create rich context for VAE
        # Global visual features
        global_vis_feat = torch.mean(vis[2], dim=(-1, -2))  # (B, fpn_in[2])
        # Encoder input
        vae_encoder_features = torch.cat(
            [motion_features, global_vis_feat, state], dim=1
        )
        # The VAE is conditioned on object features, global features, and the
        # interaction point. The column ORDER is checkpoint-load-bearing:
        #   2D path (classical): [features, point_uv, type_emb] — the exact
        #     layout every twist-era (gen-3/4/5) hint-on checkpoint was
        #     trained with; do not reorder.
        #   3D path (gen-6): [features, type_emb, point_3d_pred] — the point
        #     head consumes the base condition (features + hint, i.e. it
        #     never sees its own output) and its output extends it last.
        type_emb = None
        if self.use_motion_type_input:
            null_index = self.motion_type_embedding.num_embeddings - 1
            if motion_type_input is None:
                motion_type_input = torch.full(
                    (b,), null_index, dtype=torch.long,
                    device=vae_encoder_features.device,
                )
            type_emb = self.motion_type_embedding(
                motion_type_input.to(vae_encoder_features.device).long()
            )

        vae_condition = vae_encoder_features
        point_3d_pred = None
        if self.point_prediction_3d:
            if type_emb is not None:
                vae_condition = torch.cat([vae_condition, type_emb], dim=1)
            point_3d_pred = self.point_3d_head(vae_condition)
            vae_condition = torch.cat([vae_condition, point_3d_pred], dim=1)
        else:
            vae_condition = torch.cat([vae_condition, point_uv], dim=1)
            if type_emb is not None:
                vae_condition = torch.cat([vae_condition, type_emb], dim=1)
        if self.use_origin_heatmap:
            # gen-7: origin_uv extends the condition LAST, preserving the
            # classical [features, point_uv, type_emb] column layout.
            vae_condition = torch.cat([vae_condition, origin_uv], dim=1)

        # gen-7 scalar depths, computed from the FINAL condition. z_p also
        # sees a local sample of the decoded feature map at the predicted
        # interaction point; z_q is condition-only (see __init__).
        z_p = z_q = None
        if self.point_depth_head is not None:
            grid = point_uv.view(-1, 1, 1, 2) * 2.0 - 1.0        # [0,1] -> [-1,1]
            local = F.grid_sample(
                fq, grid, align_corners=False
            ).flatten(1)                                          # (B, fpn_out[1])
            z_p = self.point_depth_head(torch.cat([vae_condition, local], dim=1))
        if self.origin_depth_head_g7 is not None:
            zq_in = vae_condition
            if self.use_origin_local_feature:
                # Gen-11: mirror of the point path's local sample — the
                # origin heatmap's argmax pixel is typically the visible
                # hinge seam, whose appearance is depth evidence.
                ogrid = origin_uv.view(-1, 1, 1, 2) * 2.0 - 1.0
                olocal = F.grid_sample(
                    fq, ogrid, align_corners=False
                ).flatten(1)                                  # (B, fpn_out[1])
                zq_in = torch.cat([vae_condition, olocal], dim=1)
            z_q = self.origin_depth_head_g7(zq_in)

        trajectory_all = (
            self.trajectory_predictor(vae_condition)          # (B, K, N, 3)
            if self.trajectory_predictor is not None
            else None
        )
        trajectory_2d_pred = (
            self.trajectory_2d_predictor(vae_condition)
            if self.trajectory_2d_predictor is not None
            else None
        )
        origin_depth = (
            self.origin_depth_head(vae_condition)
            if self.origin_depth_head is not None
            else None
        )
        origin_pred = (
            self.origin_head(vae_condition) if self.origin_head is not None else None
        )
        # gen-7 lifts: uv + scalar depth + intrinsics -> 3D, into the SAME
        # output fields the gen-6 direct heads use (the modes are mutually
        # exclusive — see the ValueErrors in __init__). No intrinsics in the
        # batch means no lift: the fields stay None.
        if intrinsics_norm is not None:
            K = intrinsics_norm.to(vae_condition.dtype)
            if z_p is not None:
                point_3d_pred = backproject_points(
                    K.float(), point_uv.float(), z_p.float()
                )
            if z_q is not None:
                origin_pred = backproject_points(
                    K.float(), origin_uv.float(), z_q.float()
                )
        twist_pred = trajectory_pred = None
        twist_hyps = trajectory_hyps = twist_logits = None
        if self.twist_predictor is not None:
            twist_all, twist_logits = self.twist_predictor(vae_condition)
            if self.twist_num_hypotheses == 1:
                twist_pred = twist_all[:, 0]
                twist_logits = None
                if trajectory_all is not None:
                    trajectory_pred = trajectory_all[:, 0]
            else:
                # Argmax-logit bundle selection: twist_pred/trajectory_pred
                # carry the SELECTED hypothesis (what eval, viz, and every
                # legacy consumer sees); the full sets ride along for the
                # WTA loss and consistency gating.
                sel = twist_logits.argmax(dim=1)              # (B,)
                idx = torch.arange(twist_all.size(0), device=twist_all.device)
                twist_pred = twist_all[idx, sel]
                trajectory_pred = trajectory_all[idx, sel]
                twist_hyps = twist_all
                trajectory_hyps = trajectory_all
        elif trajectory_all is not None:
            trajectory_pred = trajectory_all[:, 0]

        # motion_gt can be None during pure inference, but for train/val it's provided.
        mu = None
        log_var = None
        motion_pred = None
        motion_type_logits = None
        if self.use_cvae:
            if motion_gt is not None:
                motion_pred, motion_type_logits, mu, log_var = self.motion_vae(  # type: ignore[operator]
                    motion_gt, vae_encoder_features, vae_condition
                )
            else:
                # During pure inference (e.g. in a test script), sample z from prior.
                # mu/log_var do not exist in this case and stay None.
                motion_pred, motion_type_logits = self.motion_vae.inference(vae_condition)  # type: ignore[operator]
        elif self.motion_mlp is not None:
            motion_pred, motion_type_logits = self.motion_mlp(vae_condition)  # type: ignore[operator]
            if motion_pred is not None:
                # Convert sigmoid output [0,1] to axis vector in [-1,1]
                motion_pred = (motion_pred - 0.5) * 2.0
        if not self.use_motion_type_head:
            motion_type_logits = None  # also covers the CVAE path

        return ModelOutputs(
            mask_logits=mask_pred,
            point_logits=point_pred,
            point_uv=point_uv,
            motion_pred=motion_pred,
            motion_type_logits=motion_type_logits,
            trajectory_pred=trajectory_pred,
            trajectory_2d_pred=trajectory_2d_pred,
            origin_depth=origin_depth,
            twist_pred=twist_pred,
            twist_hyps=twist_hyps,
            trajectory_hyps=trajectory_hyps,
            twist_logits=twist_logits,
            mu=mu,
            log_var=log_var,
            point_3d_pred=point_3d_pred,
            origin_pred=origin_pred,
            origin_uv=origin_uv,
            origin_logits=origin_logits,
        )
