from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ModelParams:
    clip_pretrain: str = "path/to/clip_model.pth"
    word_len: int = 17
    fpn_in: Tuple[int, int, int] = (256, 512, 1024)
    fpn_out: Tuple[int, int, int] = (256, 512, 1024)
    num_layers: int = 6
    num_head: int = 8
    dim_ffn: int = 2048
    dropout: float = 0.1
    intermediate: bool = False
    proj_dropout: float = 0.1
    use_cvae: bool = True
    vae_latent_dim: int = 32
    vae_hidden_dim: int = 256
    use_depth: bool = False  # SF3D does not use depth
    depth_feat_channels: Tuple[int, int] = (128, 256)
    # num_motion_types is not needed for SF3D, but CRIS model expects it
    num_motion_types: int = 2


@dataclass
class LossParams:
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    point_sigma: float = 5.0
    coord_weight: float = 1.0
    vae_weight: float = 1.0
    vae_beta: float = 1.0


@dataclass
class OptimizerParams:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler_milestones: List[int] = field(default_factory=lambda: [10, 20])
    scheduler_gamma: float = 0.1


@dataclass
class Config:
    # General
    exp_name: str = "sf3d_train_exp"
    output_dir: str = "outputs/sf3d"
    manual_seed: int = 42
    precision: str = "16-mixed"
    gpus: int = 1
    log_test_to_wandb: bool = False

    # Data
    train_data_dir: str = "data/scenefun3d_processed"
    val_split_ratio: float = 0.1
    input_size: Tuple[int, int] = (416, 416)
    batch_size_train: int = 8
    batch_size_val: int = 8
    num_workers_train: int = 4
    num_workers_val: int = 4

    # Logging & Visualization
    enable_wandb: bool = True
    wandb_project: str = "CRIS_SF3D"
    val_vis_samples: int = 3
    log_image_interval_steps: int = 500

    # Test-time settings (can be left as is if not testing through this script)
    test_pred_threshold: float = 0.5
    test_iou_threshold: float = 0.5
    test_motion_threshold_deg: float = 15.0
    test_visualize_debug: bool = False
    test_vis_output_dir: str = "debug_visualizations_sf3d"
    test_vis_max_images: int = 100
