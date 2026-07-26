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