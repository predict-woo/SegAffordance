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
    # Type PREDICTION head (logits + CE loss consumer). Off for 2D-only
    # pretraining, where no type labels exist — type is then emergent from
    # the twist's |omega| (the eval decodes and scores it anyway).
    use_motion_type_head: bool = True
    motion_type_input_dropout: float = 0.5
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
    # Read only by the "projected" variant (2D pretraining).
    projected_weight: float = 1.0
    # Penalty on articulation radius above projected_radius_ref metres. A line
    # is a circle of infinite radius, so without this the soft type gate
    # collapses to revolute: dL/dp = L_arc - L_line <= 0 everywhere.
    projected_radius_weight: float = 0.1
    projected_radius_ref: float = 1.0
    # 2D track supervision, used alongside the projected geometric loss.
    trajectory_2d_weight: float = 1.0
    # Sign-agnostic L2 on the se(3) twist head (use_twist_head). Units are
    # comparable to the trajectory MSE: omega is unitless O(1), v is metres.
    twist_weight: float = 0.5
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
    # Optional local visualization during test
    test_visualize_debug: bool = False
    test_vis_output_dir: str = "debug_visualizations"
    test_vis_max_images: int = 100
    test_vis_indices: List[int] = None