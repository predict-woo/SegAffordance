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