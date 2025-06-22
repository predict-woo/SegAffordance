from typing import TypedDict, List, Union, Optional
from dataclasses import dataclass


@dataclass
class SF3DConfig:
    # Model Architecture
    clip_pretrain: str
    word_len: int
    fpn_in: List[int]
    fpn_out: List[int]
    sync_bn: bool

    # Decoder (Transformer)
    num_layers: int
    num_head: int
    dim_ffn: int
    dropout: float
    intermediate: bool

    # Data and Dataloaders
    train_data_dir: str
    val_data_dir: str
    input_size: List[int]
    batch_size_train: int
    batch_size_val: int
    num_workers_train: int
    num_workers_val: int

    # Optimizer & Scheduler
    optimizer_lr: float
    optimizer_weight_decay: float
    scheduler_milestones: List[int]
    scheduler_gamma: float

    # Loss functions
    loss_bce_weight: float
    loss_dice_weight: float
    loss_point_sigma: float
    loss_coord_weight: float

    # Trainer settings
    max_epochs: int
    gpus: int
    precision: int
    exp_name: str
    output_dir: str
    manual_seed: int
    enable_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str] = None
