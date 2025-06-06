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
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config_loader # Renamed to avoid conflict with cfg variable
from datasets.scenefun3d import SF3DDataset, get_default_transforms
from model.segmenter import CRIS
from utils.dataset import tokenize # For tokenizing text descriptions
from config.type_sf3d_cfg import SF3DConfig
import pickle

import wandb

try:
    from yacs.config import CfgNode
except ImportError:
    CfgNode = None # type: ignore
from argparse import Namespace

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false" # To suppress tokenizer warnings if any

# --- Helper Functions --- #

def make_gaussian_map(points_norm, map_h, map_w, sigma, device):
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
    def __init__(self, cfg: typing.Union[SF3DConfig, dict]):
        super().__init__()
        self.save_hyperparameters(cfg)


        _attribute_accessible_cfg: typing.Any
        if isinstance(cfg, dict):
            # If cfg is a dict (likely from checkpoint hparams), make it attribute-accessible.
            if CfgNode:
                _attribute_accessible_cfg = CfgNode(cfg)
            else:
                print("Warning: yacs.config.CfgNode not available. Using argparse.Namespace for config.")
                _attribute_accessible_cfg = Namespace(**cfg)
        else:
            # If cfg is already an SF3DConfig object (during initial training).
            _attribute_accessible_cfg = cfg
        
        self.cfg = _attribute_accessible_cfg

        # self.hparams is a flat dictionary of config values (due to self.save_hyperparameters(cfg)).
        # CRIS model is initialized with this flat dictionary.
        self.model = CRIS(self.hparams) 

        self.mask_loss_fn = DiceBCELoss(
            bce_weight=self.cfg.loss_bce_weight, # Now uses the attribute-accessible self.cfg
            dice_weight=self.cfg.loss_dice_weight
        )
        self.point_map_loss_fn = nn.BCEWithLogitsLoss()
        self.coord_loss_fn = nn.L1Loss()

        # Datasets will be initialized in setup()
        self.train_dataset_split: typing.Optional[Dataset] = None
        self.val_dataset_split: typing.Optional[Dataset] = None
        
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
            lr=self.cfg.optimizer_lr, # Uses self.cfg
            weight_decay=self.cfg.optimizer_weight_decay # Uses self.cfg
        )
        scheduler = MultiStepLR(
            optimizer, 
            milestones=self.cfg.scheduler_milestones, # Uses self.cfg
            gamma=self.cfg.scheduler_gamma # Uses self.cfg
        )
        return [optimizer], [scheduler]

    def setup(self, stage: typing.Optional[str] = None):
        if stage == 'fit' or stage is None:
            print(f"ℹ️ Setting up datasets with train_data_dir: {self.cfg.train_data_dir} and val_split_ratio: {self.cfg.val_split_ratio}")
            rgb_transform, mask_transform = get_default_transforms(image_size=(self.cfg.input_size[0], self.cfg.input_size[1]))
            
            full_dataset = SF3DDataset(
                lmdb_data_root=self.cfg.train_data_dir, # train_data_dir now points to the full dataset
                rgb_transform=rgb_transform,
                mask_transform=mask_transform,
                image_size_for_mask_reconstruction=(self.cfg.input_size[0], self.cfg.input_size[1])
            )
            
            print(f"✅ Load complete. Full dataset length: {len(full_dataset)}")
            
            total_len = len(full_dataset)
            val_len = int(total_len * self.cfg.val_split_ratio)
            train_len = total_len - val_len

            print(f"ℹ️ Splitting dataset into train and validation sets...")
            self.train_dataset_split, self.val_dataset_split = random_split(
                full_dataset, 
                [train_len, val_len],
                generator=torch.Generator().manual_seed(self.cfg.manual_seed) # for reproducibility
            )
            
            print(f"✅ Dataset split: Train samples: {train_len}, Validation samples: {val_len}")

    def train_dataloader(self):
        if self.train_dataset_split is None:
            raise RuntimeError("Training dataset not initialized. Please ensure setup() was called.")

        return DataLoader(
            typing.cast(Dataset, self.train_dataset_split), # Cast for type checker
            batch_size=self.cfg.batch_size_train,
            shuffle=True,
            num_workers=self.cfg.num_workers_train,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        if self.val_dataset_split is None:
            raise RuntimeError("Validation dataset not initialized. Please ensure setup() was called.")
        
        return DataLoader(
            typing.cast(Dataset, self.val_dataset_split), # Cast for type checker
            batch_size=self.cfg.batch_size_val,
            shuffle=False,
            num_workers=self.cfg.num_workers_val,
            pin_memory=True,
            drop_last=False # Typically False for validation
        )

    def on_validation_epoch_end(self):
        # Ensure logger is WandbLogger and wandb is enabled
        if not self.cfg.enable_wandb or not isinstance(self.logger, WandbLogger) or not hasattr(self.logger.experiment, 'log'): # Uses self.cfg
            if self.cfg.enable_wandb and not isinstance(self.logger, WandbLogger): # Uses self.cfg
                print("W&B logging is enabled, but the logger is not a WandbLogger instance. Skipping visualization.")
            return

        # Get a sample batch from the validation dataloader
        val_loader = self.val_dataloader()
        try:
            batch = next(iter(val_loader))
        except StopIteration:
            print("Validation dataloader is empty, cannot visualize sample.")
            return

        img, word_str_list, mask_gt, point_gt_norm = batch
        
        # Move data to the correct device
        img = img.to(self.device)
        mask_gt = mask_gt.to(self.device)
        point_gt_norm = point_gt_norm.to(self.device)

        tokenized_words = tokenize(word_str_list, self.cfg.word_len, truncate=True).to(self.device)

        # Perform inference
        with torch.no_grad():
            mask_pred_logits, point_pred_logits, coords_hat = self(img, tokenized_words, mask_gt, point_gt_norm)

        # Select the first sample from the batch for visualization
        img_sample = img[0]                     # (C, H, W)
        mask_gt_sample = mask_gt[0].float()     # (1, H_orig, W_orig)
        point_gt_norm_sample = point_gt_norm[0:1] # (1, 2) keep batch dim for make_gaussian_map
        
        mask_pred_sigmoid_sample = torch.sigmoid(mask_pred_logits[0]) # (1, H_map, W_map)
        point_pred_sigmoid_sample = torch.sigmoid(point_pred_logits[0]) # (1, H_map, W_map)
        
        H_map, W_map = mask_pred_logits.shape[-2:]

        point_gt_heatmap_sample = make_gaussian_map(
            point_gt_norm_sample, H_map, W_map,
            sigma=self.cfg.loss_point_sigma, # Uses self.cfg
            device=self.device
        )[0] # (1, H_map, W_map)

        images_to_log = {
            "val_sample/input_image": img_sample,
            "val_sample/gt_mask": mask_gt_sample.squeeze(0), # Remove channel dim if 1 for grayscale
            "val_sample/pred_mask": mask_pred_sigmoid_sample.squeeze(0),
            "val_sample/gt_point_heatmap": point_gt_heatmap_sample.squeeze(0),
            "val_sample/pred_point_heatmap": point_pred_sigmoid_sample.squeeze(0)
        }
        
        # Log to wandb
        # Need to import wandb for wandb.Image
        try:
            import wandb # Conditional import
            log_data = {k: wandb.Image(v) for k, v in images_to_log.items()}
            # Ensure logger is WandbLogger before calling experiment.log
            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(log_data, step=self.global_step)
            else:
                # This case should ideally be caught by the check at the beginning of the method
                # save as pickle to local folder
                with open(f"out/val_sample_{self.global_step}.pkl", "wb") as f:
                    pickle.dump(images_to_log, f)
        except ImportError:
            print("wandb is not installed. Skipping image logging for validation samples.")
        except Exception as e:
            print(f"Error logging images to wandb: {e}")

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

    model = SF3DTrainingModule(typing.cast(typing.Union[SF3DConfig, dict], cfg))

    callbacks_list: typing.List[Callback] = [] 
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.output_dir,
        filename='{epoch}-{val/loss_total:.4f}',
        monitor='val/loss_total',
        mode='min',
        save_top_k=20
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
        deterministic=False,
        enable_progress_bar=cfg.wandb_show_loading_bar
    )

    trainer.fit(model)

if __name__ == '__main__':
    main() 