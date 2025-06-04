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
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config_loader # Renamed to avoid conflict with cfg variable
from datasets.scenefun3d import SF3DDataset, get_default_transforms
from model.segmenter import CRIS
from utils.dataset import tokenize # For tokenizing text descriptions
from config.type_sf3d_cfg import SF3DConfig
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Attempt to import CfgNode for config handling, fallback to Namespace
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
        # The `cfg` argument can be an SF3DConfig object (during training) 
        # or a dict (when loading from checkpoint via `load_from_checkpoint(cfg=hparams_dict)`)
        # PTL's `save_hyperparameters(cfg_object)` flattens cfg_object into self.hparams.
        # So, self.hparams will be a flat dictionary of the config parameters.
        self.save_hyperparameters(cfg)

        # For consistent internal attribute access (e.g., self.cfg.attribute_name),
        # ensure self.cfg is an attribute-accessible version of the input `cfg`.
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
        
        # For collecting test samples
        self.test_samples = []
        
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
        
    def test_step(self, batch, batch_idx):
        """Collect test samples for visualization"""
        img, word_str_list, mask_gt, point_gt_norm = batch
        
        # Move data to the correct device
        img = img.to(self.device)
        mask_gt = mask_gt.to(self.device)
        point_gt_norm = point_gt_norm.to(self.device)

        tokenized_words = tokenize(word_str_list, self.cfg.word_len, truncate=True).to(self.device)

        # Perform inference
        with torch.no_grad():
            mask_pred_logits, point_pred_logits, coords_hat = self(img, tokenized_words, mask_gt, point_gt_norm)

        # Store samples for visualization
        for i in range(img.size(0)):
            sample_data = {
                'img': img[i].cpu(),
                'word_str': word_str_list[i],
                'mask_gt': mask_gt[i].cpu(),
                'point_gt_norm': point_gt_norm[i].cpu(),
                'mask_pred_logits': mask_pred_logits[i].cpu(),
                'point_pred_logits': point_pred_logits[i].cpu(),
                'coords_hat': coords_hat[i].cpu(),
                'split': 'train' if batch_idx < 10 else 'val'  # First 10 are train, next 10 are val
            }
            self.test_samples.append(sample_data)
        
        # Compute loss for logging
        return self._common_step(batch, batch_idx, 'test')

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

    def _get_dataloader(self, split='train'):
        is_train = split == 'train'
        data_dir = self.cfg.train_data_dir if is_train else self.cfg.val_data_dir # Uses self.cfg
        batch_size = self.cfg.batch_size_train if is_train else self.cfg.batch_size_val # Uses self.cfg
        num_workers = self.cfg.num_workers_train if is_train else self.cfg.num_workers_val # Uses self.cfg
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
        
    def test_dataloader(self):
        """Create test dataloader with 10 random samples from train and 10 from val"""
        # Get train dataset
        rgb_transform, mask_transform = get_default_transforms(image_size=(self.cfg.input_size[0], self.cfg.input_size[1]))
        print(self.cfg.train_data_dir, self.cfg.val_data_dir)
        train_dataset = SF3DDataset(
            processed_data_root="/local/home/andrye/dev/SF3D_Proc/train",
            rgb_transform=rgb_transform,
            mask_transform=mask_transform,
            skip_items_without_motion=True
        )
        
        val_dataset = SF3DDataset(
            processed_data_root="/local/home/andrye/dev/SF3D_Proc/val",
            rgb_transform=rgb_transform,
            mask_transform=mask_transform,
            skip_items_without_motion=True
        )
        
        # Randomly select 10 samples from each dataset
        import random
        print(len(train_dataset), len(val_dataset))
        train_indices = random.sample(range(len(train_dataset)), min(10, len(train_dataset)))
        val_indices = random.sample(range(len(val_dataset)), min(10, len(val_dataset)))
        
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        
        # Combine datasets
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([train_subset, val_subset])
        
        return DataLoader(
            combined_dataset, 
            batch_size=1,  # Process one sample at a time for easier handling
            shuffle=False,
            num_workers=0,  # Use single worker for simpler debugging
            pin_memory=True
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

        # Resize GT mask to match prediction dimensions for comparison if needed for logging
        # or keep original for better GT visualization. For direct logging, original is fine if caption explains.
        # Here we create GT heatmap for points at map dimensions
        point_gt_heatmap_sample = make_gaussian_map(
            point_gt_norm_sample, H_map, W_map,
            sigma=self.cfg.loss_point_sigma, # Uses self.cfg
            device=self.device
        )[0] # (1, H_map, W_map)

        # For masks, wandb can often handle different sizes if they are logged as separate images.
        # If we want to overlay them, they need to be the same size.
        # Let's log them as they are.
        # Ensure masks are in [0, 1] range and (C, H, W) or (H, W)
        
        # Prepare images for wandb logging
        # wandb.Image expects channel first (C, H, W) or (H, W)
        # Our tensors are already in (C, H, W) mostly, need to ensure no extra batch dim for single image.
        
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
            
    def on_test_epoch_end(self):
        """Save visualizations of test samples to folder"""
        print(f"Processing {len(self.test_samples)} test samples for visualization...")
        
        # Create output directory
        output_dir = "test_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, sample_data in enumerate(self.test_samples):
            # Extract data
            img = sample_data['img']                    # (C, H, W)
            word_str = sample_data['word_str']
            mask_gt = sample_data['mask_gt'].float()    # (1, H_orig, W_orig)
            point_gt_norm = sample_data['point_gt_norm'].unsqueeze(0)  # (1, 2)
            mask_pred_logits = sample_data['mask_pred_logits']  # (1, H_map, W_map)
            point_pred_logits = sample_data['point_pred_logits']  # (1, H_map, W_map)
            coords_hat = sample_data['coords_hat']
            split = sample_data['split']
            
            # Apply sigmoid to predictions
            mask_pred_sigmoid = torch.sigmoid(mask_pred_logits)  # (1, H_map, W_map)
            point_pred_sigmoid = torch.sigmoid(point_pred_logits)  # (1, H_map, W_map)
            
            H_map, W_map = mask_pred_logits.shape[-2:]
            
            # Create GT point heatmap
            point_gt_heatmap = make_gaussian_map(
                point_gt_norm, H_map, W_map,
                sigma=self.cfg.loss_point_sigma, 
                device='cpu'
            )[0]  # (1, H_map, W_map)
            
            # Convert tensors to numpy for visualization
            img_np = img.permute(1, 2, 0).numpy()  # (H, W, C)
            img_np = (img_np * 0.5 + 0.5).clip(0, 1)  # Denormalize from [-1,1] to [0,1]
            
            mask_gt_np = mask_gt.squeeze(0).numpy()  # (H_orig, W_orig)
            mask_pred_np = mask_pred_sigmoid.squeeze(0).numpy()  # (H_map, W_map)
            
            point_gt_heatmap_np = point_gt_heatmap.squeeze(0).numpy()  # (H_map, W_map)
            point_pred_heatmap_np = point_pred_sigmoid.squeeze(0).numpy()  # (H_map, W_map)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Input image
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title(f'Input Image\n"{word_str}"')
            axes[0, 0].axis('off')
            
            # GT mask
            axes[0, 1].imshow(mask_gt_np, cmap='gray')
            axes[0, 1].set_title('Ground Truth Mask')
            axes[0, 1].axis('off')
            
            # Predicted mask
            axes[0, 2].imshow(mask_pred_np, cmap='gray')
            axes[0, 2].set_title('Predicted Mask')
            axes[0, 2].axis('off')
            
            # GT point heatmap
            axes[1, 0].imshow(point_gt_heatmap_np, cmap='hot')
            axes[1, 0].set_title('GT Point Heatmap')
            axes[1, 0].axis('off')
            
            # Predicted point heatmap
            axes[1, 1].imshow(point_pred_heatmap_np, cmap='hot')
            axes[1, 1].set_title('Predicted Point Heatmap')
            axes[1, 1].axis('off')
            
            # Overlay predicted coordinates on input image
            axes[1, 2].imshow(img_np)
            # Convert normalized coordinates to pixel coordinates
            pred_x = coords_hat[0].item() * img_np.shape[1]
            pred_y = coords_hat[1].item() * img_np.shape[0]
            gt_x = point_gt_norm[0, 0].item() * img_np.shape[1]
            gt_y = point_gt_norm[0, 1].item() * img_np.shape[0]
            
            axes[1, 2].plot(pred_x, pred_y, 'r*', markersize=15, label='Predicted')
            axes[1, 2].plot(gt_x, gt_y, 'g*', markersize=15, label='Ground Truth')
            axes[1, 2].set_title('Predicted vs GT Coordinates')
            axes[1, 2].axis('off')
            axes[1, 2].legend()
            
            plt.tight_layout()
            
            # Save the figure
            sample_filename = f"{split}_sample_{idx:03d}_{word_str.replace(' ', '_').replace('/', '_')}.png"
            save_path = os.path.join(output_dir, sample_filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualization for {split} sample {idx}: {save_path}")
        
        print(f"All visualizations saved to {output_dir}/")
        
        # Clear the samples list
        self.test_samples.clear()

# --- Main Training Script --- #

def get_training_parser():
    parser = argparse.ArgumentParser(description='Test CRIS model on SceneFun3D data with PyTorch Lightning')
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file (overrides defaults)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint to test')
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
        print(f"Warning: 'clip_pretrain' is not properly set in config (current: {cfg.clip_pretrain}). Testing might fail if model requires it.")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")
    
    base_output_folder = os.path.dirname(cfg.output_dir) 
    cfg.output_dir = os.path.join(base_output_folder, cfg.exp_name)

    pl.seed_everything(cfg.manual_seed, workers=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Set up trainer for testing
    accelerator_type = 'cpu'
    devices_val: typing.Union[typing.List[int], str, int] = 1 

    if cfg.gpus > 0:
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available, but cfg.gpus > 0. Running on CPU instead.")
            accelerator_type = 'cpu'
            devices_val = 1
        else:
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
        precision=cfg.precision,
        logger=False,  # Disable logging for testing
        default_root_dir=cfg.output_dir,
        deterministic=False
    )

    print(f"Loading model from checkpoint: {args.checkpoint}")
    print(f"Testing with 10 samples from training set and 10 samples from validation set")
    print(f"Visualizations will be saved to: test_visualizations/")
    
    # Load model from checkpoint and run test
    model = SF3DTrainingModule.load_from_checkpoint(args.checkpoint)
    trainer.test(model)

if __name__ == '__main__':
    main() 