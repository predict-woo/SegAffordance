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
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning.strategies import DDPStrategy

import util.config as config_loader
from datasets.opdreal import OPDRealDataset, get_default_transforms
from model.segmenter import CRIS
from util.dataset import tokenize
from config.type_sf3d_cfg import SF3DConfig

import wandb
import cv2
import numpy as np

# Import helpers from the SF3D training script
from train_SF3D import make_gaussian_map, DiceBCELoss
from argparse import Namespace

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- New VAE Loss for Motion --- #

VAL_VIS_SAMPLES = 12

class MotionVAELoss(nn.Module):
    def __init__(self, beta=0.01, cosine_eps=1e-4):
        super().__init__()
        self.beta = beta
        self.cosine_eps = cosine_eps
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.cosine_eps)

    def forward(self, recon_x, x, mu, log_var):
        # Assert that all target vectors are non-zero, as expected.
        target_norm = torch.linalg.norm(x, dim=1)
        assert torch.all(
            target_norm > self.cosine_eps
        ), "Zero-magnitude ground truth motion vector found, which is not expected."

        # Cosine similarity loss: 1 - (cos_sim)^2
        # This is low when vectors are parallel/anti-parallel, high when orthogonal.
        cos_sim = self.cos_sim(recon_x, x)
        recon_loss = (1.0 - torch.pow(cos_sim, 2)).mean()

        # KLD loss, averaged over batch
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kld_loss = kld_loss.mean()

        total_loss = recon_loss + self.beta * kld_loss
        return total_loss, recon_loss, kld_loss


# --- PyTorch Lightning Module for OPDReal --- #
class OPDRealTrainingModule(pl.LightningModule):
    def __init__(
        self,
        cfg: typing.Union[SF3DConfig, dict],
        finetune_from_path: typing.Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(cfg)

        _attribute_accessible_cfg: typing.Any
        if isinstance(cfg, dict):
            _attribute_accessible_cfg = Namespace(**cfg)
        else:
            _attribute_accessible_cfg = cfg

        self.cfg = _attribute_accessible_cfg

        self.model = CRIS(self.hparams)

        self.mask_loss_fn = DiceBCELoss(
            bce_weight=self.cfg.loss_bce_weight,
            dice_weight=self.cfg.loss_dice_weight,
        )
        self.point_map_loss_fn = nn.BCEWithLogitsLoss()
        self.coord_loss_fn = nn.L1Loss()
        self.vae_loss_fn = MotionVAELoss(beta=self.cfg.loss_vae_beta)
        self.motion_type_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        if finetune_from_path:
            self.load_finetune_weights(finetune_from_path)

        self.train_dataset: typing.Optional[Dataset] = None
        self.val_dataset: typing.Optional[Dataset] = None
        self.val_vis_samples: typing.Optional[list] = None

    def forward(
        self, img, depth, tokenized_word, mask_condition, point_condition, motion_gt
    ):
        return self.model(
            img, depth, tokenized_word, mask_condition, point_condition, motion_gt
        )

    def _common_step(self, batch, batch_idx, step_type="train"):
        (
            img,
            depth,
            word_str_list,
            mask_gt,
            _bbox,
            point_gt_norm,
            motion_gt,
            motion_type_gt,
            _img_size,
        ) = batch

        tokenized_words = tokenize(
            list(word_str_list), self.cfg.word_len, truncate=True
        ).to(self.device)

        (
            mask_pred_logits,
            point_pred_logits,
            coords_hat,
            motion_pred,
            motion_type_logits,
            mu,
            log_var,
        ) = self(img, depth, tokenized_words, mask_gt, point_gt_norm, motion_gt)

        H_map, W_map = mask_pred_logits.shape[-2:]
        mask_gt_float = mask_gt.float().to(mask_pred_logits.device)
        mask_gt_downsampled = F.interpolate(
            mask_gt_float, size=(H_map, W_map), mode="bilinear", align_corners=False
        )

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

        L_motion_type = self.motion_type_loss_fn(
            motion_type_logits, motion_type_gt.to(motion_type_logits.device)
        )

        total_loss = (
            (self.cfg.loss_mask_weight * L_mask)
            + (self.cfg.loss_point_map_weight * L_point_map)
            + (self.cfg.loss_coord_weight * L_coord)
            + (self.cfg.loss_vae_weight * L_vae)
            + (self.cfg.loss_motion_type_weight * L_motion_type)
        )

        self.log(
            f"{step_type}/loss_total",
            total_loss,
            on_step=(step_type == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_mask",
            L_mask,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_point_map",
            L_point_map,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_coord",
            L_coord,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_vae_total",
            L_vae,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_motion_recon",
            L_recon,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_kld",
            L_kld,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_motion_type",
            L_motion_type,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.trainer.is_global_zero:
            return

        if (
            self.cfg.log_image_interval_steps > 0
            and (self.global_step + 1) % self.cfg.log_image_interval_steps == 0
        ):
            self.log_visualizations()

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
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
            print("ℹ️ Setting up datasets for OPDReal...")
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

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size_train,
            shuffle=True,
            num_workers=self.cfg.num_workers_train,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.cfg.num_workers_train > 0,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size_val,
            shuffle=False,
            num_workers=self.cfg.num_workers_val,
            pin_memory=True,
            drop_last=False,
        )

    def log_visualizations(self):
        if (
            not self.cfg.enable_wandb
            or not isinstance(self.logger, WandbLogger)
            or not hasattr(self.logger.experiment, "log")
            or self.val_vis_samples is None
        ):
            return

        (
            img_list,
            depth_list,
            word_str_list,
            mask_gt_list,
            _bbox_list,
            point_gt_norm_list,
            motion_gt_list,
            motion_type_gt_list,
            _img_size_list,
        ) = zip(*self.val_vis_samples)

        img = torch.stack(img_list).to(self.device)
        depth = torch.stack(depth_list).to(self.device)
        mask_gt = torch.stack(mask_gt_list).to(self.device)
        point_gt_norm = torch.stack(point_gt_norm_list).to(self.device)
        motion_gt = torch.stack(motion_gt_list).to(self.device)

        tokenized_words = tokenize(
            list(word_str_list), self.cfg.word_len, truncate=True
        ).to(self.device)

        with torch.no_grad():
            (
                mask_pred_logits,
                point_pred_logits,
                coords_hat,
                motion_pred,
                motion_type_logits,
                _,
                _,
            ) = self(img, depth, tokenized_words, mask_gt, point_gt_norm, motion_gt)

        try:
            log_data = {}
            for i in range(len(img)):
                img_sample = img[i]
                mask_gt_sample = mask_gt[i].float()
                point_gt_norm_sample = point_gt_norm[i]
                text_description = word_str_list[i]
                mask_pred_sigmoid_sample = torch.sigmoid(mask_pred_logits[i])
                point_pred_sigmoid_sample = torch.sigmoid(point_pred_logits[i])

                composite_image = create_composite_visualization(
                    image_tensor=img_sample,
                    text_description=text_description,
                    mask_pred_sigmoid=mask_pred_sigmoid_sample,
                    point_gt_norm=point_gt_norm_sample,
                    point_pred_heatmap=point_pred_sigmoid_sample,
                    motion_gt=motion_gt[i],
                    motion_pred=motion_pred[i],
                    motion_type_gt=motion_type_gt_list[i],
                    motion_type_pred_logits=motion_type_logits[i],
                )

                log_data[f"val_sample/{i}_sample"] = wandb.Image(composite_image)

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(log_data, step=self.global_step)

        except ImportError:
            print("wandb is not installed. Skipping image logging.")
        except Exception as e:
            print(f"Error logging images to wandb: {e}")

    def on_validation_epoch_end(self):
        pass

    def load_finetune_weights(self, checkpoint_path: str):
        print(f"🔩 Loading weights for finetuning from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Strip "model." prefix from keys to match the model's state_dict
        if any(key.startswith("model.") for key in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

        model_state_dict = self.model.state_dict()

        # Filter out unnecessary keys and load only matching ones
        pretrained_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_state_dict and v.size() == model_state_dict[k].size()
        }

        if not pretrained_dict:
            print("⚠️ Warning: No matching keys found in the pretrained model.")
            return

        model_state_dict.update(pretrained_dict)
        self.model.load_state_dict(model_state_dict)

        loaded_keys = pretrained_dict.keys()
        model_keys = model_state_dict.keys()

        unloaded_keys = [k for k in model_keys if k not in loaded_keys]

        if unloaded_keys:
            print(f"⚠️ Warning: The following keys were not loaded: {unloaded_keys}")
        else:
            print("✅ All model weights loaded successfully.")


# --- Main Training Script --- #


def get_training_parser():
    parser = argparse.ArgumentParser(description="Train CRIS model on OPDReal data")
    parser.add_argument(
        "--config", type=str, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Override settings in config",
    )
    return parser


def create_composite_visualization(
    image_tensor: torch.Tensor,
    text_description: str,
    mask_pred_sigmoid: torch.Tensor,
    point_gt_norm: torch.Tensor,
    point_pred_heatmap: torch.Tensor,
    motion_gt: torch.Tensor,
    motion_pred: torch.Tensor,
    motion_type_gt: torch.Tensor,
    motion_type_pred_logits: torch.Tensor,
    vector_scale: int = 50,
):
    """
    Creates a composite visualization image with mask, vectors, and points.
    """
    # 1. Convert tensor image to OpenCV format
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    # 2. Overlay predicted mask as a transparent red overlay
    mask_np = mask_pred_sigmoid.cpu().numpy().squeeze()
    mask_resized = cv2.resize(mask_np, (w, h))
    red_overlay = np.zeros_like(img_bgr)
    red_overlay[:, :, 2] = 255  # Red in BGR
    binary_mask = (mask_resized > 0.5).astype(np.uint8)
    masked_red = cv2.bitwise_and(red_overlay, red_overlay, mask=binary_mask)
    composite_img = cv2.addWeighted(img_bgr, 1.0, masked_red, 0.5, 0)

    # 3. Draw GT (green) and predicted (red) motion vectors
    origin_x = int(point_gt_norm[0] * w)
    origin_y = int(point_gt_norm[1] * h)

    # Normalize vectors for visualization to ensure they have the same length
    motion_gt_2d = motion_gt.cpu().numpy()[:2]
    motion_pred_2d = motion_pred.cpu().numpy()[:2]

    norm_gt = np.linalg.norm(motion_gt_2d)
    if norm_gt > 1e-6:
        motion_gt_2d /= norm_gt

    norm_pred = np.linalg.norm(motion_pred_2d)
    if norm_pred > 1e-6:
        motion_pred_2d /= norm_pred

    gt_target_x = origin_x + int(motion_gt_2d[0] * vector_scale)
    gt_target_y = origin_y + int(motion_gt_2d[1] * vector_scale)
    cv2.arrowedLine(
        composite_img,
        (origin_x, origin_y),
        (gt_target_x, gt_target_y),
        (0, 255, 0),
        2,
        tipLength=0.3,
    )

    pred_target_x = origin_x + int(motion_pred_2d[0] * vector_scale)
    pred_target_y = origin_y + int(motion_pred_2d[1] * vector_scale)
    cv2.arrowedLine(
        composite_img,
        (origin_x, origin_y),
        (pred_target_x, pred_target_y),
        (0, 0, 255),
        2,
        tipLength=0.3,
    )

    # 4. Plot GT (yellow) and predicted (magenta) interaction points as stars
    gt_point_px = (int(point_gt_norm[0] * w), int(point_gt_norm[1] * h))
    cv2.drawMarker(
        composite_img,
        gt_point_px,
        (0, 255, 255),  # Yellow
        markerType=cv2.MARKER_STAR,
        markerSize=20,
        thickness=2,
    )

    heatmap_np = point_pred_heatmap.cpu().numpy().squeeze()
    map_h, map_w = heatmap_np.shape
    pred_y_map, pred_x_map = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    pred_point_px = (int(pred_x_map * w / map_w), int(pred_y_map * h / map_h))
    cv2.drawMarker(
        composite_img,
        pred_point_px,
        (255, 0, 255),  # Magenta
        markerType=cv2.MARKER_STAR,
        markerSize=20,
        thickness=2,
    )

    # 5. Overlay text description
    motion_type_map = {0: "translation", 1: "rotation"}
    gt_type_str = motion_type_map.get(int(motion_type_gt.item()), "unknown")
    pred_type_idx = torch.argmax(motion_type_pred_logits).item()
    pred_type_str = motion_type_map.get(int(pred_type_idx), "unknown")
    text = f"{text_description}\nGT: {gt_type_str}, Pred: {pred_type_str}"

    y0, dy = 15, 15
    for i, line in enumerate(text.split("\n")):
        y = y0 + i * dy
        cv2.putText(
            composite_img,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)


def main():
    parser = get_training_parser()
    args = parser.parse_args()

    # Set matmul precision for Tensor Cores
    torch.set_float32_matmul_precision("high")

    cfg = config_loader.load_cfg_from_cfg_file(args.config)

    if not hasattr(cfg, "num_motion_types"):
        raise ValueError("Configuration error: 'num_motion_types' must be specified.")
    if not hasattr(cfg, "loss_motion_type_weight"):
        raise ValueError(
            "Configuration error: 'loss_motion_type_weight' must be specified."
        )

    if args.opts:
        cfg = config_loader.merge_cfg_from_list(cfg, args.opts)

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

    model = OPDRealTrainingModule(cfg)

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
