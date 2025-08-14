import os
import argparse
import warnings
import typing

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning.strategies import DDPStrategy

import utils.config as config_loader
from datasets.opdreal import OPDRealDataset, get_default_transforms
from model.segmenter import CRIS
from utils.dataset import tokenize
from config.type_sf3d_cfg import SF3DConfig

import wandb
import cv2
import numpy as np

# Import from our new base
from train_OPDReal import (
    OPDRealTrainingModule,
    create_composite_visualization,
)


from argparse import Namespace

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- New VAE Loss for Motion --- #

VAL_VIS_SAMPLES = 12


class OPDMultiTrainingModule(OPDRealTrainingModule):
    def __init__(
        self,
        cfg: typing.Union[SF3DConfig, dict],
        finetune_from_path: typing.Optional[str] = None,
    ):
        super().__init__(cfg, finetune_from_path=finetune_from_path)

        if not hasattr(self.cfg, "train_only_heads"):
            self.cfg.train_only_heads = False

    def configure_optimizers(self):
        if self.cfg.train_only_heads:
            print(
                "❄️ Backbone, Depth Encoder, and Neck are frozen. Training only decoder and heads."
            )
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            for param in self.model.depth_encoder.parameters():
                param.requires_grad = False
            for param in self.model.neck.parameters():
                param.requires_grad = False

            # Sanity check
            for name, param in self.model.named_parameters():
                if name.startswith(("backbone.", "depth_encoder.", "neck.")):
                    assert not param.requires_grad

            trainable_params = filter(
                lambda p: p.requires_grad, self.model.parameters()
            )
        else:
            print("🔥 Training all parameters (backbone and heads).")
            trainable_params = filter(
                lambda p: p.requires_grad, self.model.parameters()
            )

        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.cfg.optimizer_lr,
            weight_decay=self.cfg.optimizer_weight_decay,
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.cfg.scheduler_milestones,
            gamma=self.cfg.scheduler_gamma,
        )
        return [optimizer], [scheduler]

    def setup(self, stage: typing.Optional[str] = None):
        if stage == "fit" or stage is None:
            print("ℹ️ Setting up datasets for OPDMulti...")
            train_rgb_transform, mask_transform, depth_transform = (
                get_default_transforms(
                    image_size=(self.cfg.input_size[0], self.cfg.input_size[1]),
                    is_train=True,
                )
            )
            val_rgb_transform, _, _ = get_default_transforms(
                image_size=(self.cfg.input_size[0], self.cfg.input_size[1]),
                is_train=False,
            )

            print(
                f"✅ Loading train dataset: {self.cfg.train_dataset_key} from {self.cfg.data_path}"
            )
            self.train_dataset = OPDRealDataset(
                data_path=self.cfg.data_path,
                dataset_key=self.cfg.train_dataset_key,
                rgb_transform=train_rgb_transform,
                mask_transform=mask_transform,
                depth_transform=depth_transform,
                is_multi=True,
            )
            print(f"✅ Train dataset loaded. Length: {len(self.train_dataset)}")

            print(
                f"✅ Loading validation dataset: {self.cfg.val_dataset_key} from {self.cfg.data_path}"
            )
            self.val_dataset = OPDRealDataset(
                data_path=self.cfg.data_path,
                dataset_key=self.cfg.val_dataset_key,
                rgb_transform=val_rgb_transform,
                mask_transform=mask_transform,
                depth_transform=depth_transform,
                is_multi=True,
            )
            print(f"✅ Validation dataset loaded. Length: {len(self.val_dataset)}")

            val_len = len(self.val_dataset)
            if val_len > VAL_VIS_SAMPLES:
                generator = torch.Generator().manual_seed(self.cfg.manual_seed)
                indices = torch.randperm(val_len, generator=generator)[:VAL_VIS_SAMPLES]
                self.val_vis_samples = [
                    self.val_dataset[int(i.item())] for i in indices
                ]
                print(
                    f"✅ Stored {VAL_VIS_SAMPLES} fixed validation samples for visualization."
                )
            else:
                print(
                    f"⚠️ Warning: Not enough validation samples (<{VAL_VIS_SAMPLES}) to select for visualization."
                )


# --- Main Training Script --- #


def get_training_parser():
    parser = argparse.ArgumentParser(description="Train CRIS model on OPDMulti data")
    parser.add_argument(
        "--config", type=str, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--finetune_from",
        type=str,
        default=None,
        help="Path to a checkpoint to finetune from. Loads model weights but not optimizer state.",
    )
    parser.add_argument(
        "--train_only_heads",
        action="store_true",
        help="Freeze backbone, depth encoder, and FPN neck. Only train the decoder and projection/VAE heads.",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Override settings in config",
    )
    return parser


def main():
    parser = get_training_parser()
    args = parser.parse_args()

    cfg = config_loader.load_cfg_from_cfg_file(args.config)

    if not hasattr(cfg, "num_motion_types"):
        raise ValueError("Configuration error: 'num_motion_types' must be specified.")
    if not hasattr(cfg, "loss_motion_type_weight"):
        raise ValueError(
            "Configuration error: 'loss_motion_type_weight' must be specified."
        )

    if args.opts:
        cfg = config_loader.merge_cfg_from_list(cfg, args.opts)

    if args.train_only_heads:
        cfg.train_only_heads = True

    if not hasattr(cfg, "train_only_heads"):
        cfg.train_only_heads = False

    if not hasattr(cfg, "data_path") or not cfg.data_path:
        raise ValueError("Configuration error: 'data_path' must be specified.")
    if not hasattr(cfg, "train_dataset_key") or not cfg.train_dataset_key:
        raise ValueError("Configuration error: 'train_dataset_key' must be specified.")
    if not hasattr(cfg, "val_dataset_key") or not cfg.val_dataset_key:
        raise ValueError("Configuration error: 'val_dataset_key' must be specified.")

    if (
        not hasattr(cfg, "clip_pretrain")
        or not cfg.clip_pretrain
        or cfg.clip_pretrain == "path/to/clip_model.pth"
    ):
        print(
            f"Warning: 'clip_pretrain' not set. Training might fail if model requires it."
        )

    base_output_folder = os.path.dirname(cfg.output_dir)
    cfg.output_dir = os.path.join(base_output_folder, cfg.exp_name)

    pl.seed_everything(cfg.manual_seed, workers=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = OPDMultiTrainingModule(cfg, finetune_from_path=args.finetune_from)

    if cfg.gpus > 1:
        print(
            f"⚠️  Using {cfg.gpus} GPUs. Effective batch size: {cfg.batch_size_train * cfg.gpus}"
        )

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
        logger = WandbLogger(project=cfg.wandb_project, name=cfg.exp_name, config=cfg)

    accelerator_type = "cpu"
    devices_val: typing.Union[typing.List[int], str, int] = 1
    strategy_val: typing.Union[str, DDPStrategy] = "auto"

    if cfg.gpus > 0:
        if not torch.cuda.is_available():
            raise SystemError("CUDA not available, but cfg.gpus > 0.")
        accelerator_type = "gpu"
        if cfg.gpus == 1:
            devices_val = [0]
        else:
            devices_val = cfg.gpus
            strategy_val = DDPStrategy(find_unused_parameters=True)

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
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
