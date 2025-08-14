import os
import typing
import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset

from config.opd_train import Config, LossParams, ModelParams, OptimizerParams
from datasets.opdreal_datamodule import OPDRealDataModule
from model.segmenter import CRIS
from utils.dataset import tokenize
from utils.tools import DiceBCELoss, MotionVAELoss, create_composite_visualization, make_gaussian_map

torch.set_float32_matmul_precision("high")

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- PyTorch Lightning Module for OPDReal --- #
class OPDRealTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model_params: ModelParams,
        loss_params: LossParams,
        optimizer_params: OptimizerParams,
        config: Config,
        finetune_from_path: typing.Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_params = model_params
        self.loss_params = loss_params
        self.optimizer_params = optimizer_params
        self.config = config


        self.model = CRIS(model_params)

        # self.mask_loss_fn = DiceBCELoss(
        #     bce_weight=self.loss_params.bce_weight,
        #     dice_weight=self.loss_params.dice_weight,
        # )
        self.mask_loss_fn = nn.BCEWithLogitsLoss()
        self.point_map_loss_fn = nn.BCEWithLogitsLoss()
        self.coord_loss_fn = nn.L1Loss()
        self.vae_loss_fn = MotionVAELoss(beta=self.loss_params.vae_beta)
        self.motion_type_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        if finetune_from_path:
            self.load_finetune_weights(finetune_from_path)

        self.train_dataset: typing.Optional[Dataset] = None
        self.val_dataset: typing.Optional[Dataset] = None
        self.val_vis_samples: typing.Optional[list] = None

        # Visualization reservoir sampling config/state
        self.vis_num_samples: int = self.config.val_vis_samples
        self.base_seed: int = self.config.manual_seed
        self._vis_buffer: typing.List[typing.Tuple] = []
        self._vis_seen_count: int = 0
        self._vis_rng: typing.Optional[torch.Generator] = None

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
            list(word_str_list), self.model_params.word_len, truncate=True
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
            sigma=self.loss_params.point_sigma,
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
            (self.loss_params.mask_weight * L_mask)
            + (self.loss_params.point_map_weight * L_point_map)
            + (self.loss_params.coord_weight * L_coord)
            + (self.loss_params.vae_weight * L_vae)
            + (self.loss_params.motion_type_weight * L_motion_type)
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
            self.config.log_image_interval_steps > 0
            and (self.global_step + 1) % self.config.log_image_interval_steps == 0
        ):
            self.log_visualizations()

    def on_validation_epoch_start(self):
        # Reset reservoir and RNG each epoch for reproducibility and variety
        self._vis_buffer = []
        self._vis_seen_count = 0
        self._vis_rng = torch.Generator(device="cpu").manual_seed(
            int(self.base_seed + int(self.current_epoch))
        )

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "val")
        self._collect_vis_samples_from_batch(batch)
        return loss

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.optimizer_params.lr,
            weight_decay=self.optimizer_params.weight_decay,
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.optimizer_params.scheduler_milestones,
            gamma=self.optimizer_params.scheduler_gamma,
        )
        return [optimizer], [scheduler]

    # Dataset logic has been moved into a separate LightningDataModule

    def log_visualizations(self):
        if not isinstance(self.logger, WandbLogger):
            return
        if not self.val_vis_samples:
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
            list(word_str_list), self.model_params.word_len, truncate=True
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
        # Only log on the main process
        if not getattr(self, "trainer", None) or not self.trainer.is_global_zero:
            return
        if self._vis_buffer:
            self.val_vis_samples = self._vis_buffer
            self.log_visualizations()
        # Release buffer
        self._vis_buffer = []

    def _collect_vis_samples_from_batch(self, batch):
        try:
            (
                img,
                depth,
                word_str_list,
                mask_gt,
                bbox,
                point_gt_norm,
                motion_gt,
                motion_type_gt,
                img_size,
            ) = batch
        except Exception:
            return

        batch_size = img.size(0) if hasattr(img, "size") else 0
        for i in range(batch_size):
            sample = (
                img[i].detach().cpu(),
                depth[i].detach().cpu(),
                word_str_list[i],
                mask_gt[i].detach().cpu(),
                bbox[i].detach().cpu(),
                point_gt_norm[i].detach().cpu(),
                motion_gt[i].detach().cpu(),
                motion_type_gt[i].detach().cpu(),
                img_size[i].detach().cpu(),
            )

            if len(self._vis_buffer) < self.vis_num_samples:
                self._vis_buffer.append(sample)
            else:
                # Reservoir sampling replacement
                if self._vis_rng is None:
                    self._vis_rng = torch.Generator(device="cpu").manual_seed(self.base_seed)
                j = int(torch.randint(0, self._vis_seen_count + 1, (1,), generator=self._vis_rng).item())
                if j < self.vis_num_samples:
                    self._vis_buffer[j] = sample
            self._vis_seen_count += 1

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


if __name__ == "__main__":
    LightningCLI(OPDRealTrainingModule, OPDRealDataModule)
