import os
import argparse
import warnings
import typing # For type hinting

import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback # Import Callback for type hinting
from pytorch_lightning.loggers import WandbLogger # Optional
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config_loader # Renamed to avoid conflict with cfg variable
from datasets.scenefun3d import SF3DDataset, get_default_transforms
from model.segmenter import CRIS
from utils.dataset import tokenize # For tokenizing text descriptions
from config.type_sf3d_cfg import SF3DConfig

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false" # To suppress tokenizer warnings if any

# --- Helper Functions --- #

def make_gaussian_map(points_norm, map_h, map_w, sigma:float =2.0, device='cpu'):
    """
    Generates a batch of 2D Gaussian heatmaps.
    Args:
        points_norm (torch.Tensor): Batch of normalized points (B, 2) in [0, 1] range.
        map_h (int): Target heatmap height.
        map_w (int): Target heatmap width.
        sigma (float): Standard deviation of the Gaussian in pixels.
        device (str or torch.device): Device to create tensors on.
    Returns:
        torch.Tensor: Batch of Gaussian heatmaps (B, 1, map_h, map_w).
    """
    B = points_norm.size(0)
    if B == 0:
        return torch.empty((0, 1, map_h, map_w), device=device)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(map_h, dtype=torch.float32, device=device),
        torch.arange(map_w, dtype=torch.float32, device=device),
        indexing='ij'
    )

    x_grid = x_coords.unsqueeze(0).expand(B, -1, -1)
    y_grid = y_coords.unsqueeze(0).expand(B, -1, -1)

    center_x = (points_norm[:, 0] * (map_w - 1)).view(B, 1, 1)
    center_y = (points_norm[:, 1] * (map_h - 1)).view(B, 1, 1)

    dist_sq = (x_grid - center_x)**2 + (y_grid - center_y)**2
    heatmaps = torch.exp(-dist_sq / (2 * sigma**2))
    return heatmaps.unsqueeze(1)

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float =0.5, dice_weight: float =0.5, smooth_dice: float =1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth_dice = smooth_dice
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target_mask):
        """
        pred_logits: (B, 1, H, W)
        target_mask: (B, 1, H, W), values in [0,1]
        """
        bce_val = self.bce_loss(pred_logits, target_mask)

        pred_sigmoid = torch.sigmoid(pred_logits)
        intersection = (pred_sigmoid * target_mask).sum(dim=(1, 2, 3))
        union_pred = pred_sigmoid.sum(dim=(1, 2, 3))
        union_target = target_mask.sum(dim=(1, 2, 3))
        
        dice_score = (2. * intersection + self.smooth_dice) / (union_pred + union_target + self.smooth_dice)
        dice_val = 1. - dice_score
        
        return (self.bce_weight * bce_val) + (self.dice_weight * dice_val.mean())






# --- PyTorch Lightning Module --- #

class SF3DTrainingModule(pl.LightningModule):
    def __init__(self, cfg:SF3DConfig):
        super().__init__()
        self.save_hyperparameters(cfg) 
        self.cfg = cfg
        self.model = CRIS(self.hparams) 

        self.mask_loss_fn = DiceBCELoss(
            bce_weight=cfg.loss_bce_weight,
            dice_weight=cfg.loss_dice_weight
        )
        self.point_map_loss_fn = nn.BCEWithLogitsLoss()
        self.coord_loss_fn = nn.L1Loss()
        
    def forward(self, img, tokenized_word, mask_condition, point_condition):
        return self.model(img, tokenized_word, mask_condition, point_condition)

    def _common_step(self, batch, batch_idx, step_type='train'):
        img, word_str_list, mask_gt, point_gt_norm = batch
        tokenized_words = tokenize(word_str_list, self.cfg.word_len, truncate=True).to(self.device)
        mask_pred_logits, point_pred_logits, coords_hat = self(img, tokenized_words, mask_gt, point_gt_norm)

        H_map, W_map = mask_pred_logits.shape[-2:]
        mask_gt_float = mask_gt.float().to(mask_pred_logits.device)
        mask_gt_downsampled = F.interpolate(mask_gt_float, size=(H_map, W_map), mode='bilinear', align_corners=False)
        L_mask = self.mask_loss_fn(mask_pred_logits, mask_gt_downsampled)

        point_gt_heatmap = make_gaussian_map(point_gt_norm, H_map, W_map, 
                                             sigma=self.cfg.loss_point_sigma,
                                             device=point_pred_logits.device)
        L_point_map = self.point_map_loss_fn(point_pred_logits, point_gt_heatmap)
        L_coord = self.coord_loss_fn(coords_hat, point_gt_norm.to(coords_hat.device))
        total_loss = L_mask + L_point_map + (self.cfg.loss_coord_weight * L_coord)

        self.log(f'{step_type}/loss_total', total_loss, on_step=(step_type=='train'), on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{step_type}/L_mask', L_mask, on_step=False, on_epoch=True, logger=True)
        self.log(f'{step_type}/L_point_map', L_point_map, on_step=False, on_epoch=True, logger=True)
        self.log(f'{step_type}/L_coord', L_coord, on_step=False, on_epoch=True, logger=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(
            trainable_params, 
            lr=self.cfg.optimizer_lr, 
            weight_decay=self.cfg.optimizer_weight_decay
        )
        scheduler = MultiStepLR(
            optimizer, 
            milestones=self.cfg.scheduler_milestones, 
            gamma=self.cfg.scheduler_gamma
        )
        return [optimizer], [scheduler]

    def _get_dataloader(self, split='train'):
        is_train = split == 'train'
        data_dir = self.cfg.train_data_dir if is_train else self.cfg.val_data_dir
        batch_size = self.cfg.batch_size_train if is_train else self.cfg.batch_size_val
        num_workers = self.cfg.num_workers_train if is_train else self.cfg.num_workers_val
        shuffle = True if is_train else False

        rgb_transform, mask_transform = get_default_transforms(image_size=(self.cfg.input_size[0], self.cfg.input_size[1]))
        
        dataset = SF3DDataset(
            processed_data_root=data_dir,
            rgb_transform=rgb_transform,
            mask_transform=mask_transform,
            skip_items_without_motion=True
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=is_train
        )

    def train_dataloader(self):
        return self._get_dataloader(split='train')

    def val_dataloader(self):
        return self._get_dataloader(split='val')

# --- Main Training Script --- #

def get_training_parser():
    parser = argparse.ArgumentParser(description='Train CRIS model on SceneFun3D data with PyTorch Lightning')
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file (overrides defaults)')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER, help='Override settings in the config')
    return parser

def main():
    parser = get_training_parser()
    args = parser.parse_args()

    default_config_path = 'config/default_sf3d_train.yaml'
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"Default config file not found at {default_config_path}. Please create it.")
    cfg = config_loader.load_cfg_from_cfg_file(default_config_path)

    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"User-specified config file not found at {args.config}")
        user_cfg = config_loader.load_cfg_from_cfg_file(args.config)
        cfg.update(user_cfg) 
    
    if args.opts:
        cfg = config_loader.merge_cfg_from_list(cfg, args.opts)

    if not hasattr(cfg, 'train_data_dir') or not cfg.train_data_dir:
        raise ValueError("Configuration error: 'train_data_dir' must be specified in your config file.")
    if not hasattr(cfg, 'val_data_dir') or not cfg.val_data_dir:
        raise ValueError("Configuration error: 'val_data_dir' must be specified in your config file.")
    if not hasattr(cfg, 'clip_pretrain') or not cfg.clip_pretrain or cfg.clip_pretrain == 'path/to/clip_model.pth':
        print(f"Warning: 'clip_pretrain' is not properly set in config (current: {cfg.clip_pretrain}). Training might fail if model requires it.")
    
    base_output_folder = os.path.dirname(cfg.output_dir) 
    cfg.output_dir = os.path.join(base_output_folder, cfg.exp_name)

    pl.seed_everything(cfg.manual_seed, workers=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = SF3DTrainingModule(typing.cast(SF3DConfig, cfg))

    callbacks_list: typing.List[Callback] = [] 
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.output_dir,
        filename='{epoch}-{val/loss_total:.4f}',
        monitor='val/loss_total',
        mode='min',
        save_top_k=3
    )
    callbacks_list.append(checkpoint_callback)

    logger: typing.Union[WandbLogger, bool] = False
    if cfg.enable_wandb:
        wandb_logger = WandbLogger(
            project=cfg.wandb_project,
            name=cfg.exp_name,
            config=vars(cfg)
        )
        logger = wandb_logger

    accelerator_type = 'cpu'
    devices_val: typing.Union[typing.List[int], str, int] = 1 

    if cfg.gpus > 0:
        if not torch.cuda.is_available():
            raise SystemError("CUDA is not available, but cfg.gpus > 0. Set cfg.gpus to 0 to run on CPU.")
        accelerator_type = 'gpu'
        if cfg.gpus == 1:
            devices_val = [0]  
        else:
            devices_val = cfg.gpus 
    else: 
        accelerator_type = 'cpu'
        devices_val = 1 

    trainer = pl.Trainer(
        accelerator=accelerator_type,
        devices=devices_val,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        callbacks=callbacks_list,
        logger=logger,
        default_root_dir=cfg.output_dir,
        deterministic=False
    )

    trainer.fit(model)

if __name__ == '__main__':
    main() 