import os
import argparse
import warnings
import typing

import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning.strategies import DDPStrategy

import util.config as config_loader
from datasets.scenefun3d import (
    SF3DDataset,
    get_default_transforms,
    split_dataset_by_scene,
)
from model.segmenter import CRIS
from util.dataset import tokenize
from config.type_sf3d_cfg import SF3DConfig
import pickle

import wandb

try:
    from yacs.config import CfgNode
except ImportError:
    CfgNode = None
from argparse import Namespace

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable Tensor Core usage for faster training on RTX 4090
# torch.set_float32_matmul_precision("medium")

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
        indexing="ij",
    )

    x_grid = x_coords.unsqueeze(0).expand(B, -1, -1)
    y_grid = y_coords.unsqueeze(0).expand(B, -1, -1)

    center_x = (points_norm[:, 0] * (map_w - 1)).view(B, 1, 1)
    center_y = (points_norm[:, 1] * (map_h - 1)).view(B, 1, 1)

    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    heatmaps = torch.exp(-dist_sq / (2 * sigma**2))
    return heatmaps.unsqueeze(1)


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth_dice: float = 1e-6,
    ):
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

        dice_score = (2.0 * intersection + self.smooth_dice) / (
            union_pred + union_target + self.smooth_dice
        )
        dice_val = 1.0 - dice_score

        return (self.bce_weight * bce_val) + (self.dice_weight * dice_val.mean())


class VAELoss(nn.Module):
    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.recon_loss_fn = nn.L1Loss(reduction="none")
        self.reduction = reduction

    def forward(self, pred_motion, target_motion, mu, log_var):
        recon_loss_unreduced = self.recon_loss_fn(pred_motion, target_motion).sum(
            dim=1
        )  # Sum over vector dims
        kld_unreduced = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

        if self.reduction == "mean":
            recon_loss = recon_loss_unreduced.mean()
            kld = kld_unreduced.mean()
        elif self.reduction == "sum":
            recon_loss = recon_loss_unreduced.sum()
            kld = kld_unreduced.sum()
        else:
            recon_loss = recon_loss_unreduced
            kld = kld_unreduced

        total_loss = recon_loss + self.beta * kld
        return total_loss, recon_loss, kld


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
                print(
                    "Warning: yacs.config.CfgNode not available. Using argparse.Namespace for config."
                )
                _attribute_accessible_cfg = Namespace(**cfg)
        else:
            # If cfg is already an SF3DConfig object (during initial training).
            _attribute_accessible_cfg = cfg

        self.cfg = _attribute_accessible_cfg

        # self.hparams is a flat dictionary of config values (due to self.save_hyperparameters(cfg)).
        # CRIS model is initialized with this flat dictionary.
        self.model = CRIS(self.hparams)

        self.mask_loss_fn = DiceBCELoss(
            bce_weight=self.cfg.loss_bce_weight,  # Now uses the attribute-accessible self.cfg
            dice_weight=self.cfg.loss_dice_weight,
        )
        self.point_map_loss_fn = nn.BCEWithLogitsLoss()
        self.coord_loss_fn = nn.L1Loss()
        self.vae_loss_fn = VAELoss(beta=self.cfg.loss_vae_beta)

        # Datasets will be initialized in setup()
        self.train_dataset_split: typing.Optional[Dataset] = None
        self.val_dataset_split: typing.Optional[Dataset] = None
        self.val_vis_samples: typing.Optional[list] = None

    def forward(self, img, tokenized_word, mask_condition, point_condition, motion_gt):
        return self.model(
            img, tokenized_word, mask_condition, point_condition, motion_gt
        )

    def _common_step(self, batch, batch_idx, step_type="train"):
        img, word_str_list, mask_gt, point_gt_norm, motion_gt = batch
        tokenized_words = tokenize(
            list(word_str_list), self.cfg.word_len, truncate=True
        ).to(self.device)
        (
            mask_pred_logits,
            point_pred_logits,
            coords_hat,
            motion_pred,
            mu,
            log_var,
        ) = self(img, tokenized_words, mask_gt, point_gt_norm, motion_gt)

        H_map, W_map = mask_pred_logits.shape[-2:]
        mask_gt_float = mask_gt.float().to(mask_pred_logits.device)
        mask_gt_downsampled = F.interpolate(
            mask_gt_float, size=(H_map, W_map), mode="bilinear", align_corners=False
        )

        # check if any of the values in mask_pred_logits are nan

        # check if any of the values in mask_gt_downsampled are nan

        L_mask = self.mask_loss_fn(mask_pred_logits, mask_gt_downsampled)

        point_gt_heatmap = make_gaussian_map(
            point_gt_norm,
            H_map,
            W_map,
            sigma=self.cfg.loss_point_sigma,
            device=point_pred_logits.device,
        )
        L_point_map = self.point_map_loss_fn(point_pred_logits, point_gt_heatmap)
        L_coord = self.coord_loss_fn(coords_hat, point_gt_norm.to(coords_hat.device))

        L_vae, L_recon, L_kld = self.vae_loss_fn(
            motion_pred, motion_gt.to(motion_pred.device), mu, log_var
        )

        total_loss = (
            L_mask
            + L_point_map
            + (self.cfg.loss_coord_weight * L_coord)
            + (self.cfg.loss_vae_weight * L_vae)
        )

        self.log(
            f"{step_type}/loss_total",
            total_loss,
            on_step=(step_type == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,  # Synchronize across GPUs
        )
        self.log(
            f"{step_type}/L_mask",
            L_mask,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,  # Synchronize across GPUs
        )
        self.log(
            f"{step_type}/L_point_map",
            L_point_map,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,  # Synchronize across GPUs
        )
        self.log(
            f"{step_type}/L_coord",
            L_coord,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,  # Synchronize across GPUs
        )
        self.log(
            f"{step_type}/L_vae_total",
            L_vae,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_motion_recon",
            L_recon,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_kld",
            L_kld,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Only run on the main process (rank 0) for multi-GPU training
        if not self.trainer.is_global_zero:
            return

        # Log visualizations periodically
        if (
            self.cfg.log_image_interval_steps > 0
            and (self.global_step + 1) % self.cfg.log_image_interval_steps == 0
        ):
            self.log_visualizations()

    # def on_after_backward(self):
    #     # For debugging purposes, print gradient norms
    #     print(f"\n--- Gradients at step {self.global_step} ---")
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad and param.grad is not None:
    #             grad_norm = param.grad.norm(2)
    #             print(f"  - {name}: {grad_norm:.4f}")
    #             if torch.isnan(grad_norm) or torch.isinf(grad_norm):
    #                 print(f"    !!!! WARNING: NaN or Inf gradient for {name} !!!!")
    #     print("-------------------------------------\n")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.cfg.optimizer_lr,  # Uses self.cfg
            weight_decay=self.cfg.optimizer_weight_decay,  # Uses self.cfg
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.cfg.scheduler_milestones,  # Uses self.cfg
            gamma=self.cfg.scheduler_gamma,  # Uses self.cfg
        )
        return [optimizer], [scheduler]

    def setup(self, stage: typing.Optional[str] = None):
        if stage == "fit" or stage is None:
            print(
                f"ℹ️ Setting up datasets with train_data_dir: {self.cfg.train_data_dir} and val_split_ratio: {self.cfg.val_split_ratio}"
            )
            rgb_transform, mask_transform = get_default_transforms(
                image_size=(self.cfg.input_size[0], self.cfg.input_size[1])
            )

            full_dataset = SF3DDataset(
                lmdb_data_root=self.cfg.train_data_dir,  # train_data_dir now points to the full dataset
                rgb_transform=rgb_transform,
                mask_transform=mask_transform,
                image_size_for_mask_reconstruction=(
                    self.cfg.input_size[0],
                    self.cfg.input_size[1],
                ),
            )

            print(f"✅ Load complete. Full dataset length: {len(full_dataset)}")

            print(f"ℹ️ Splitting dataset by scene...")
            self.train_dataset_split, self.val_dataset_split = split_dataset_by_scene(
                full_dataset,
                val_split_ratio=self.cfg.val_split_ratio,
                manual_seed=self.cfg.manual_seed,
            )
            train_len = len(self.train_dataset_split)
            val_len = len(self.val_dataset_split)

            print(
                f"✅ Dataset split: Train samples: {train_len}, Validation samples: {val_len}"
            )

            # Prepare fixed samples for visualization
            if val_len > 3:
                generator = torch.Generator().manual_seed(self.cfg.manual_seed)
                indices = torch.randperm(val_len, generator=generator)[:3]
                self.val_vis_samples = [self.val_dataset_split[i] for i in indices]
                print(f"✅ Stored 3 fixed validation samples for visualization.")
            else:
                print(
                    "⚠️ Warning: Not enough validation samples (<3) to select for visualization."
                )

    def train_dataloader(self):
        if self.train_dataset_split is None:
            raise RuntimeError(
                "Training dataset not initialized. Please ensure setup() was called."
            )

        return DataLoader(
            typing.cast(Dataset, self.train_dataset_split),  # Cast for type checker
            batch_size=self.cfg.batch_size_train,
            shuffle=True,
            num_workers=self.cfg.num_workers_train,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.cfg.num_workers_train > 0,
            prefetch_factor=16,
        )

    def val_dataloader(self):
        if self.val_dataset_split is None:
            raise RuntimeError(
                "Validation dataset not initialized. Please ensure setup() was called."
            )

        return DataLoader(
            typing.cast(Dataset, self.val_dataset_split),  # Cast for type checker
            batch_size=self.cfg.batch_size_val,
            shuffle=False,
            num_workers=self.cfg.num_workers_val,
            pin_memory=True,
            drop_last=False,  # Typically False for validation
        )

    def log_visualizations(self):
        # Ensure logger is WandbLogger and wandb is enabled
        if (
            not self.cfg.enable_wandb
            or not isinstance(self.logger, WandbLogger)
            or not hasattr(self.logger.experiment, "log")
            or self.val_vis_samples is None
        ):
            if self.cfg.enable_wandb and not isinstance(self.logger, WandbLogger):
                print(
                    "W&B logging is enabled, but the logger is not a WandbLogger instance. Skipping visualization."
                )
            return

        # Collate the samples into a batch
        (
            img_list,
            word_str_list,
            mask_gt_list,
            point_gt_norm_list,
            motion_gt_list,
        ) = zip(*self.val_vis_samples)

        img = torch.stack(img_list)
        mask_gt = torch.stack(mask_gt_list)
        point_gt_norm = torch.stack(point_gt_norm_list)
        motion_gt = torch.stack(motion_gt_list)

        # Move data to the correct device
        img = img.to(self.device)
        mask_gt = mask_gt.to(self.device)
        point_gt_norm = point_gt_norm.to(self.device)
        motion_gt = motion_gt.to(self.device)

        tokenized_words = tokenize(
            list(word_str_list), self.cfg.word_len, truncate=True
        ).to(self.device)

        # Perform inference
        with torch.no_grad():
            (
                mask_pred_logits,
                point_pred_logits,
                coords_hat,
                motion_pred,
                _,
                _,
            ) = self(img, tokenized_words, mask_gt, point_gt_norm, motion_gt)

        try:
            import wandb  # Conditional import

            log_data = {}
            for i in range(len(img)):  # Iterate through the 3 samples
                img_sample = img[i]
                mask_gt_sample = mask_gt[i].float()
                point_gt_norm_sample = point_gt_norm[i : i + 1]

                mask_pred_sigmoid_sample = torch.sigmoid(mask_pred_logits[i])
                point_pred_sigmoid_sample = torch.sigmoid(point_pred_logits[i])

                H_map, W_map = mask_pred_logits.shape[-2:]

                point_gt_heatmap_sample = make_gaussian_map(
                    point_gt_norm_sample,
                    H_map,
                    W_map,
                    sigma=self.cfg.loss_point_sigma,
                    device=self.device,
                )[0]

                images_to_log = {
                    f"val_sample_{i}/input_image": img_sample,
                    f"val_sample_{i}/gt_mask": mask_gt_sample.squeeze(0),
                    f"val_sample_{i}/pred_mask": mask_pred_sigmoid_sample.squeeze(0),
                    f"val_sample_{i}/gt_point_heatmap": point_gt_heatmap_sample.squeeze(
                        0
                    ),
                    f"val_sample_{i}/pred_point_heatmap": point_pred_sigmoid_sample.squeeze(
                        0
                    ),
                }
                log_data.update({k: wandb.Image(v) for k, v in images_to_log.items()})

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(log_data, step=self.global_step)
            else:
                os.makedirs("out", exist_ok=True)
                with open(f"out/val_sample_{self.global_step}.pkl", "wb") as f:
                    pickle.dump(images_to_log, f)

        except ImportError:
            print(
                "wandb is not installed. Skipping image logging for validation samples."
            )
        except Exception as e:
            print(f"Error logging images to wandb: {e}")

    def on_validation_epoch_end(self):
        # Only run on the main process (rank 0) for multi-GPU training
        if self.trainer.is_global_zero is False:
            return

        # Ensure logger is WandbLogger and wandb is enabled
        if (
            not self.cfg.enable_wandb
            or not isinstance(self.logger, WandbLogger)
            or not hasattr(self.logger.experiment, "log")
        ):  # Uses self.cfg
            if self.cfg.enable_wandb and not isinstance(
                self.logger, WandbLogger
            ):  # Uses self.cfg
                print(
                    "W&B logging is enabled, but the logger is not a WandbLogger instance. Skipping visualization."
                )
            return

        # The visualization logic has been moved to log_visualizations()
        # and is triggered periodically during the training loop.
        # This hook is now kept for potential future validation-specific logic
        # that doesn't involve logging the same sample images.
        pass


# --- Main Training Script --- #


def get_training_parser():
    parser = argparse.ArgumentParser(
        description="Train CRIS model on SceneFun3D data with PyTorch Lightning"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file (overrides defaults)",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Override settings in the config",
    )
    return parser


def main():
    parser = get_training_parser()
    args = parser.parse_args()

    default_config_path = "config/default_sf3d_train.yaml"
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(
            f"Default config file not found at {default_config_path}. Please create it."
        )
    cfg = config_loader.load_cfg_from_cfg_file(default_config_path)

    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(
                f"User-specified config file not found at {args.config}"
            )
        user_cfg = config_loader.load_cfg_from_cfg_file(args.config)
        cfg.update(user_cfg)

    if args.opts:
        cfg = config_loader.merge_cfg_from_list(cfg, args.opts)

    if not hasattr(cfg, "train_data_dir") or not cfg.train_data_dir:
        raise ValueError(
            "Configuration error: 'train_data_dir' must be specified in your config file."
        )
    if not hasattr(cfg, "val_split_ratio"):
        raise ValueError(
            "Configuration error: 'val_split_ratio' must be specified in your config file."
        )
    if (
        not hasattr(cfg, "clip_pretrain")
        or not cfg.clip_pretrain
        or cfg.clip_pretrain == "path/to/clip_model.pth"
    ):
        print(
            f"Warning: 'clip_pretrain' is not properly set in config (current: {cfg.clip_pretrain}). Training might fail if model requires it."
        )

    base_output_folder = os.path.dirname(cfg.output_dir)
    cfg.output_dir = os.path.join(base_output_folder, cfg.exp_name)

    pl.seed_everything(cfg.manual_seed, workers=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = SF3DTrainingModule(typing.cast(typing.Union[SF3DConfig, dict], cfg))

    # Warn about batch size when using multiple GPUs
    if cfg.gpus > 1:
        effective_batch_size = cfg.batch_size_train * cfg.gpus
        print(
            f"⚠️  Using {cfg.gpus} GPUs. Effective batch size will be: {effective_batch_size}"
        )
        print(f"   (batch_size_train={cfg.batch_size_train} × gpus={cfg.gpus})")

    callbacks_list: typing.List[Callback] = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.output_dir,
        filename="{epoch}-{val/loss_total:.4f}",
        monitor="val/loss_total",
        mode="min",
        save_top_k=20,
    )
    callbacks_list.append(checkpoint_callback)

    logger: typing.Union[WandbLogger, bool] = False
    if cfg.enable_wandb:
        wandb_logger = WandbLogger(
            project=cfg.wandb_project, name=cfg.exp_name, config=vars(cfg)
        )
        logger = wandb_logger

    accelerator_type = "cpu"
    devices_val: typing.Union[typing.List[int], str, int] = 1
    strategy_val: typing.Union[str, DDPStrategy] = "auto"  # Default strategy

    if cfg.gpus > 0:
        if not torch.cuda.is_available():
            raise SystemError(
                "CUDA is not available, but cfg.gpus > 0. Set cfg.gpus to 0 to run on CPU."
            )
        accelerator_type = "gpu"
        if cfg.gpus == 1:
            devices_val = [0]
            strategy_val = "auto"
        else:
            devices_val = cfg.gpus
            strategy_val = DDPStrategy(find_unused_parameters=True)
    else:
        accelerator_type = "cpu"
        devices_val = 1

    trainer = pl.Trainer(
        num_nodes=1,
        accelerator=accelerator_type,
        devices=devices_val,
        strategy=strategy_val,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        callbacks=callbacks_list,
        logger=logger,
        default_root_dir=cfg.output_dir,
        deterministic=False,
        enable_progress_bar=cfg.wandb_show_loading_bar,
        gradient_clip_val=1.0,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
