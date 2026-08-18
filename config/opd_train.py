from dataclasses import dataclass
from typing import List

@dataclass
class ModelParams:
    clip_pretrain: str
    word_len: int
    depth_feat_channels: List[int]
    fpn_in: List[int]
    fpn_out: List[int]
    num_layers: int
    num_head: int
    dim_ffn: int
    dropout: float
    intermediate: bool
    proj_dropout: float
    vae_latent_dim: int
    vae_hidden_dim: int
    num_motion_types: int
    # Optional depth usage (enabled by default)
    use_depth: bool = True
    # Optional CVAE usage for motion prediction (enabled by default)
    use_cvae: bool = True
    # 3D element-sweep trajectory head. Disable for 2D pretraining, where no
    # element-sweep ground truth exists and the head would get no gradient.
    use_trajectory_head: bool = True
    # 2D hand/contact-track head. Enable for 2D pretraining.
    use_2d_trajectory_head: bool = False
    # Trajectory heads predict num_points-1 per-step DELTAS integrated by
    # cumsum, with point 0 pinned to exactly 0 (the interaction point is the
    # absolute anchor). The loss stays in position space. Fixes the
    # jittery/unordered point clouds the direct 20-point readout produced
    # (viz/20260803_sf3d_twist_traj_points). Applies to both the 3D and 2D
    # trajectory heads.
    trajectory_delta_cumsum: bool = False
    # Predict metric depth of the 3D joint origin, giving a full
    # {type, axis, origin} articulation. Required by geometric_loss="projected".
    predict_origin_depth: bool = False
    # Unified screw-motion head: one se(3) twist (omega, v) for both motion
    # types — revolute is the axis LINE (no point on it), prismatic is a bare
    # direction with omega = 0. Additive alongside the axis/type heads; see
    # model/losses/twist.py. SF3D-only supervision (needs a 3D origin).
    use_twist_head: bool = False
    # Feed the GT articulation type to the model as an auxiliary input: an
    # embedding of {trans, rot, NULL} concatenated into the head condition
    # vector. Trained with conditioning dropout — each sample's type is
    # replaced by the learned NULL token with motion_type_input_dropout
    # probability — so the model also works with no type given. Val/test
    # ALWAYS run with NULL: metrics and checkpoint selection reflect the
    # deployment condition, and the hint can be supplied at inference when
    # available for a free accuracy boost.
    use_motion_type_input: bool = False
    # Motion-axis PREDICTION head (the legacy 3-dim axis regressor). Off in
    # twist arms: the twist head carries the axis, and eval falls back to
    # the twist's decoded direction for axis metrics.
    use_motion_head: bool = True
    # Gen-17 (2026-08-18 spec): per-type axis readouts (motion_head_rot /
    # motion_head_trans) on the shared MotionMLP trunk. Loss is routed by
    # GT type; inference selects by predicted type. Requires
    # use_motion_head + use_motion_type_head; incompatible with
    # use_cvae / use_twist_head.
    split_axis_heads: bool = False
    # Head-structural pitch removal: TwistMLP's output map ends in a
    # smoothed orthogonal projection v <- v - (v.omega)omega/(|omega|^2+eps^2),
    # so the head's output space IS the pitch-free variety {omega.v = 0}:
    # exact orthogonality in the revolute regime, smoothly the identity as
    # omega -> 0 (where pitch is vacuous and v must stay free for
    # translation). A literal 5-parameter chart cannot exist — the variety's
    # v-fiber jumps dimension at omega = 0, and any global chart
    # reintroduces the prismatic singularity the twist removes.
    twist_pitch_free: bool = False
    # Type PREDICTION head (logits + CE loss consumer). Off for 2D-only
    # pretraining, where no type labels exist — type is then emergent from
    # the twist's |omega| (the eval decodes and scores it anyway).
    use_motion_type_head: bool = True
    # gen-6 split arm (docs/superpowers/specs/2026-08-13-split-heads-gen6-design.md).
    # Predict the interaction point DIRECTLY as an absolute 3D camera-frame
    # point (GT trajectory_3d[0]) instead of the 2D heatmap + soft-argmax:
    # the Projector emits the mask channel only, point_logits/point_uv are
    # None, and the articulation heads' condition is extended with the
    # predicted 3D point instead of point_uv. SF3D-only (needs 3D GT).
    point_prediction_3d: bool = False
    # Predict the revolute joint origin as an absolute 3D point, supervised
    # toward q* — the GT-axis point perpendicular to the interaction point.
    use_origin_head: bool = False
    # Remove the mask-pooling teacher forcing: train-time condition pooling
    # uses the DETACHED sigmoid of the predicted mask (what val/test always
    # did) instead of the GT mask. Detached so articulation losses cannot
    # steer the mask head through the pooling path.
    pool_with_predicted_mask: bool = False
    # gen-7 (docs/superpowers/specs/2026-08-14-heatmap-depth-lift-gen7-design.md).
    # Third projector channel: origin heatmap -> soft-argmax origin_uv ->
    # scalar depth z_q -> q_hat lifted with intrinsics. Supervised at the
    # projected q* (in-frame 99.6-100% on v2 — tools/diag_origin_inframe.py).
    use_origin_heatmap: bool = False
    # Scalar depth head z_p for the interaction point (input: condition +
    # grid_sample of decoded features at point_uv); p_hat lifted with
    # intrinsics. Requires the classical 2D point path (point_prediction_3d
    # False).
    predict_point_depth: bool = False
    # Gen-11: ẑ_q consumes a grid-sampled local feature at origin_uv —
    # the mirror of ẑ_p's sample at point_uv (the hinge-seam pixel IS
    # depth evidence; reverses the gen-7 condition-only choice). Requires
    # use_origin_heatmap.
    use_origin_local_feature: bool = False
    # TrajectoryMLP emits 20 ABSOLUTE camera-frame points — no delta-cumsum,
    # no relative frame (gen-7 user decision; zigzag + absolute-regression
    # risks on record in the spec). Mutually exclusive with
    # trajectory_delta_cumsum.
    trajectory_absolute: bool = False
    motion_type_input_dropout: float = 0.5
    # K > 1: winner-takes-all articulation hypotheses — TwistMLP emits K
    # (twist, logit) pairs and TrajectoryMLP K matching trajectories; one
    # bundle is selected by argmax logit. Trained with the annealed WTA loss
    # (model/losses/twist.py; docs/superpowers/specs/2026-08-11-twist-wta-
    # head-design.md). Requires use_twist_head AND use_trajectory_head.
    twist_num_hypotheses: int = 1
    motion_type_embedding_dim: int = 16
    # NHWC memory format for the conv stack. Numerically equivalent; removes
    # the NCHW->NHWC conversion cuDNN otherwise inserts around every conv to
    # reach the tensor cores (~5% of GPU time as pure layout shuffling,
    # profiled 2026-08-03) and speeds the convs themselves.
    channels_last: bool = False
    # torch.compile the whole model in-place (nn.Module.compile(), which
    # keeps state_dict keys unprefixed so checkpoints stay loadable by the
    # eval/viz tools). First steps pay compilation (~minutes).
    compile_model: bool = False
    # Which vision+text encoder to use: "clip_rn50" | "siglip2" | "dinov3"
    backbone: str = "clip_rn50"
    # HF id / hub entry for the non-CLIP backbones
    backbone_id: str = ""
    # Native input resolution the backbone checkpoint expects
    backbone_image_size: int = 256
    # DINOv3 only: where the text tower comes from ("dinotxt" | "clip")
    text_source: str = "dinotxt"
    # DINOv3 + dino.txt only: gated .pth files and a local dinov3 checkout
    dinotxt_weights: str = ""
    dinov3_backbone_weights: str = ""
    dinov3_repo_dir: str = ""
    # Gen-14 (dinov3 standard stack, stage T): build the pyramid from
    # 4 intermediate-layer taps (ViT-L blocks 4/11/17 by hook + the final
    # tokens) instead of the final layer only. dinotxt path only.
    dinov3_multilayer_taps: bool = False
    # Gen-15 (stage C): explicit patch-text similarity maps (local + global
    # halves of the dino.txt embedding vs the aligned /16 tokens) injected
    # as 2 extra FPN input channels per level, and the FPN text gate reads
    # only the patch-aligned LOCAL half of the state. dinotxt only.
    text_cost_map: bool = False


@dataclass
class LossParams:
    bce_weight: float
    dice_weight: float
    mask_weight: float
    point_map_weight: float
    coord_weight: float
    vae_weight: float
    motion_type_weight: float
    point_sigma: float
    vae_beta: float
    # Optional trajectory-related weights
    trajectory_weight: float = 1.0
    # Gen-16: per-row GT-energy-normalized trajectory loss (relative-error;
    # a collapsed prediction scores exactly 1.0). Fixes the rot-sweep
    # collapse of the trajectory_weight=0.15 arms — see the 2026-08-17
    # spec. Relative-trajectory mode only.
    trajectory_loss_normalized: bool = False
    # Which geometric consistency loss to use between the motion-axis and
    # trajectory heads: "cross_gt" | "pred_pred" | "none".
    # Default "cross_gt" reproduces the behaviour of configs written before
    # this option existed.
    geometric_loss: str = "cross_gt"
    # Read only by the "cross_gt" variant.
    geometric_weight: float = 1.0
    trajectory_to_motion_weight: float = 1.0
    # Read only by the "pred_pred" variant. Kept well under motion_type_weight
    # so the type head stays anchored by its own cross-entropy.
    pred_pred_weight: float = 0.1
    # Read only by the "pred_pred_art" variant (gen-6): all-predicted
    # trajectory <-> axis-line <-> type consistency, soft type gate.
    pred_pred_art_weight: float = 0.5
    # Gen-10: dimensionless L_pp branches (relative orbit error / off-axis
    # energy fraction) and the r_hat floor, = min_revolute_radius.
    pred_pred_art_normalized: bool = False
    pred_pred_art_radius_floor: float = 0.10
    # gen-6 split arm. axis_sign_agnostic True = classical 1 - cos^2
    # (antiparallel OK — OPD annotates only the axis LINE); False = 1 - cos,
    # for SF3D where the stored sign is canonical.
    axis_sign_agnostic: bool = True
    # MSE toward q* (the GT-axis point perpendicular to the interaction
    # point), revolute rows only. Consumed when the model has an origin head.
    origin_weight: float = 0.5
    # gen-7 origin-heatmap analogue of point_map_weight: BCE vs the Gaussian
    # at the projected q*, revolute+in-frame rows only.
    origin_map_weight: float = 0.5
    # MSE of the direct 3D interaction point vs GT trajectory_3d[0].
    # Replaces point_map+coord (both zero on the 3D path) at the same
    # total weight budget.
    point_3d_weight: float = 0.5
    # Read only by the "projected" variant (2D pretraining).
    projected_weight: float = 1.0
    # Penalty on articulation radius above projected_radius_ref metres. A line
    # is a circle of infinite radius, so without this the soft type gate
    # collapses to revolute: dL/dp = L_arc - L_line <= 0 everywhere.
    projected_radius_weight: float = 0.1
    projected_radius_ref: float = 1.0
    # 2D track supervision, used alongside the projected geometric loss.
    trajectory_2d_weight: float = 1.0
    # Weight of the twist part of the WTA distortion (use_twist_head). The
    # loss is the BODY-FRAME kinetic-energy metric anchored at the GT element
    # point (m² of motion error at the object; twist_body_distance), falling
    # back to the legacy Euclidean 6-vector MSE on batches without a 3D
    # trajectory. 0.5 carries over from the old MSE: measured init scale
    # 0.603 (body) vs 0.634 (MSE) on SF3D val.
    twist_weight: float = 0.5
    # Gyration radius (metres) of the body metric — prices angular error
    # independently of how close the object sits to the axis. 0.25 = median
    # SF3D revolute radius / typical part scale.
    twist_metric_rho: float = 0.25
    # Annealed-WTA schedule (twist_num_hypotheses > 1): softmin temperature
    # decays exponentially T0 -> 0.01 over anneal_frac * max_epochs, then
    # hard winner. Val/test always score the hard winner.
    twist_wta_T0: float = 10.0
    twist_wta_anneal_frac: float = 0.8
    # Cross-entropy on the bundle-selection logits (winner = label).
    twist_hyp_ce_weight: float = 0.1
    # True (default): score the better of (omega,v)/(-omega,-v) — correct for
    # OPD, where only the axis LINE is annotated. False (SF3D): the stored
    # sign is canonical (the preprocessor derives the GT trajectory from it,
    # tools/sf3d_process.py), so plain sign-SENSITIVE MSE — the head must
    # commit to the semantic sweep direction.
    twist_sign_agnostic: bool = True
    # Read only by the "screw" variant: predicted twist's velocity field vs
    # the GT trajectory, and vs the model's OWN trajectory anchored at its
    # OWN interaction point (no teacher forcing in the self term).
    screw_gt_weight: float = 0.5
    screw_self_weight: float = 0.5
    # Observed 2D track vs the projected orbit through the model's own
    # anchor. Fully prediction-anchored — no anchor_depth from the data.
    screw_track_weight: float = 0.5
    # Predicted 3D trajectory projected into the image vs the observed 2D
    # track, index-matched (ordered + direction-sensitive by construction).
    # THE data term of 2D-only pretraining; 0 disables (3D arms).
    trajectory_proj_weight: float = 0.0
    # Gen-18 (2026-08-18 spec): per-row normalization of the projection
    # residual by the GT track's motion energy — the gen-16 rot-collapse
    # lesson applied in uv-space.
    trajectory_proj_normalized: bool = False
    # Gen-18: GT-free z_p tether — masked MSE between the lifted point's
    # depth and the INPUT depth sampled at a detached point_uv. Keeps the
    # lift calibrated when the 3D point loss is off (2D-only arms).
    depth_anchor_weight: float = 0.0
    # Occam prior |omega| for settings where nothing else pins omega (video
    # pretraining and the 2D-only arm). Leave 0 where the direct twist L2
    # supervises it (3D arms).
    screw_omega_shrink: float = 0.0
    
@dataclass
class OptimizerParams:
    lr: float
    weight_decay: float
    scheduler_milestones: List[int]
    scheduler_gamma: float
    

@dataclass
class Config:
    log_image_interval_steps: int
    input_size: List[int]
    enable_wandb: bool
    val_vis_samples: int
    manual_seed: int
    # Test/eval settings (optional with safe defaults)
    test_motion_threshold_deg: float = 10.0
    test_iou_threshold: float = 0.5
    test_pred_threshold: float = 0.5
    # Which IoU metric to use for matching: "mask" or "bbox"
    test_match_metric: str = "mask"
    # Control logging of test metrics to external loggers like W&B
    log_test_to_wandb: bool = False
    # > 0: every N train steps, log per-loss-term gradient norms into the
    # shared backbone (train/grad_norm/<term>) — the smoke-run check that the
    # small-magnitude geometric terms are not drowned by the high-floor
    # mask/point_map gradients. Costs one extra backward per term per logged
    # step; leave 0 for real runs, and run smoke with compile_model false
    # (autograd.grad inside a compiled graph is not supported). The retained
    # graph across the extra backwards needs VRAM headroom: batch 128 OOMs a
    # 24 GB dev GPU — smoke at batch <= 48 there.
    log_term_grad_norm_interval_steps: int = 0
    # Optional local visualization during test
    test_visualize_debug: bool = False
    test_vis_output_dir: str = "debug_visualizations"
    test_vis_max_images: int = 100
    test_vis_indices: List[int] = None