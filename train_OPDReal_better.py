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

        self.mask_loss_fn = DiceBCELoss(
            bce_weight=self.loss_params.bce_weight,
            dice_weight=self.loss_params.dice_weight,
        )
        # self.mask_loss_fn = nn.BCEWithLogitsLoss()
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

        # Test accumulators
        self._test_ious: typing.List[float] = []
        self._test_point_errors: typing.List[float] = []
        self._test_axis_errors_matched: typing.List[float] = []
        self._test_num_matched: int = 0
        self._test_correct_axis_predictions: int = 0
        self._test_correct_type_in_matched: int = 0
        self._test_correct_all: int = 0
        self._test_origin_errors_matched: typing.List[float] = []
        self._test_correct_origin_predictions: int = 0
        self._test_num_rotational_matched: int = 0

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

        if getattr(self.model, "use_cvae", True):
            L_vae, L_recon, L_kld = self.vae_loss_fn(
                motion_pred, motion_gt.to(motion_pred.device), mu, log_var
            )
        else:
            # Use 1 - cos^2 for axis direction (antiparallel is OK). Normalize target to avoid zero norms.
            cos_sim = F.cosine_similarity(motion_pred, motion_gt.to(motion_pred.device), dim=1, eps=1e-4)
            L_motion = (1.0 - cos_sim.pow(2)).mean()
            L_vae, L_recon, L_kld = L_motion, L_motion, torch.zeros((), device=motion_pred.device)

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

    # --- Testing support ---
    def test_step(self, batch, batch_idx):
        (
            img,
            depth,
            word_str_list,
            mask_gt,
            bbox_gt,
            point_gt_norm,
            motion_gt,
            motion_type_gt,
            _img_size,
        ) = batch

        tokenized_words = tokenize(
            list(word_str_list), self.model_params.word_len, truncate=True
        ).to(self.device)

        with torch.no_grad():
            (
                mask_pred_logits,
                _point_pred_logits,
                coords_hat,
                motion_pred,
                motion_type_logits,
                _mu,
                _log_var,
            ) = self(img, depth, tokenized_words, None, None, None)

        mask_pred_prob = torch.sigmoid(mask_pred_logits)
        mask_pred_upsampled = F.interpolate(
            mask_pred_prob, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False
        )
        pred_types = torch.argmax(motion_type_logits, dim=1)

        batch_size = img.size(0)
        # Optional visualization settings
        do_vis = getattr(self.config, "test_visualize_debug", False)
        vis_dir = getattr(self.config, "test_vis_output_dir", "debug_visualizations")
        vis_max = int(getattr(self.config, "test_vis_max_images", 100))

        for i in range(batch_size):
            # IoU metric
            if getattr(self.config, "test_match_metric", "mask") == "bbox":
                pred_mask_binary = (mask_pred_upsampled[i].squeeze() > self.config.test_pred_threshold).float()
                pred_bbox = self._mask_to_bbox(pred_mask_binary)
                iou_val = self._bbox_iou(pred_bbox.cpu(), bbox_gt[i].cpu()).item()
            else:
                pred_mask_binary = (mask_pred_upsampled[i] > self.config.test_pred_threshold).float()
                iou_val = self._mask_iou(pred_mask_binary, mask_gt[i]).item()
            self._test_ious.append(iou_val)

            # Point error
            point_err = torch.linalg.norm(coords_hat[i] - point_gt_norm[i]).item()
            self._test_point_errors.append(point_err)

            # Match gating
            if iou_val > self.config.test_iou_threshold:
                self._test_num_matched += 1

                # Axis error (degrees, direction-agnostic)
                axis_err = self._axis_error_deg(motion_pred[i], motion_gt[i]).item()
                self._test_axis_errors_matched.append(axis_err)

                is_axis_correct = axis_err <= self.config.test_motion_threshold_deg
                if is_axis_correct:
                    self._test_correct_axis_predictions += 1

                is_type_correct = pred_types[i] == motion_type_gt[i]
                if is_type_correct:
                    self._test_correct_type_in_matched += 1

                if is_axis_correct and is_type_correct:
                    self._test_correct_all += 1
                
                # Origin Pass Rate (for rotational motions)
                # Assuming motion type 0 is rotation
                if motion_type_gt[i] == 0:
                    self._test_num_rotational_matched += 1
                    pred_origin = coords_hat[i]
                    gt_origin = point_gt_norm[i].to(pred_origin.device)
                    gt_axis = motion_gt[i].to(pred_origin.device)
                    
                    p = pred_origin - gt_origin
                    # Ensure gt_axis has non-zero norm before division
                    gt_axis_norm_val = torch.linalg.norm(gt_axis)
                    if gt_axis_norm_val > 1e-6:
                        dist = torch.linalg.norm(torch.cross(p, gt_axis)) / gt_axis_norm_val
                        
                        bbox = bbox_gt[i]
                        diagonal = torch.sqrt(bbox[2]**2 + bbox[3]**2)
                        
                        if diagonal > 1e-6:
                            norm_dist = dist / diagonal
                            self._test_origin_errors_matched.append(norm_dist.item())
                            
                            origin_threshold = getattr(self.config, "test_origin_threshold", 0.1)
                            if norm_dist <= origin_threshold:
                                self._test_correct_origin_predictions += 1

            # Local visualization (no W&B). Save side-by-side image of prob vs. thresholded w/ bbox.
            if do_vis and (len(self._test_ious) <= vis_max) and self.trainer.is_global_zero:
                try:
                    self._save_test_debug_visualization(
                        image_tensor=img[i].detach().cpu(),
                        gt_mask_tensor=mask_gt[i].detach().cpu(),
                        pred_mask_prob_tensor=mask_pred_upsampled[i].detach().cpu(),
                        gt_bbox=bbox_gt[i].detach().cpu(),
                        description=word_str_list[i],
                        pred_threshold=float(self.config.test_pred_threshold),
                        output_dir=vis_dir,
                        sample_index=(batch_idx * batch_size + i),
                        iou_value=iou_val,
                    )
                except Exception as _:
                    pass

        return {}

    def on_test_epoch_end(self):
        dm = getattr(self.trainer, "datamodule", None) if hasattr(self, "trainer") else None
        test_ds = getattr(dm, "test_dataset", None) if dm is not None else None
        total_predictions = len(test_ds) if test_ds is not None else 0
        mean_iou = float(torch.tensor(self._test_ious).mean().item()) if self._test_ious else 0.0
        mean_point_error = float(torch.tensor(self._test_point_errors).mean().item()) if self._test_point_errors else 0.0
        err_adir = float(torch.tensor(self._test_axis_errors_matched).mean().item()) if self._test_axis_errors_matched else 0.0
        mean_origin_error = float(torch.tensor(self._test_origin_errors_matched).mean().item()) if self._test_origin_errors_matched else 0.0

        if self._test_num_matched > 0:
            pass_rate_axis = 100.0 * self._test_correct_axis_predictions / self._test_num_matched
            pass_rate_type = 100.0 * self._test_correct_type_in_matched / self._test_num_matched
        else:
            pass_rate_axis = 0.0
            pass_rate_type = 0.0
        
        if self._test_num_rotational_matched > 0:
            pass_rate_origin = 100.0 * self._test_correct_origin_predictions / self._test_num_rotational_matched
        else:
            pass_rate_origin = 0.0

        if total_predictions > 0:
            p_det = 100.0 * self._test_num_matched / total_predictions
            map_type = 100.0 * self._test_correct_type_in_matched / total_predictions
            map_all = 100.0 * self._test_correct_all / total_predictions
        else:
            p_det = 0.0
            map_type = 0.0
            map_all = 0.0

        # Always log to progress bar/console; optionally to external logger
        self.log("test/mean_iou", mean_iou, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/p_det", p_det, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/mean_point_error", mean_point_error, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/err_adir_deg", err_adir, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pass_rate_axis", pass_rate_axis, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pass_rate_type", pass_rate_type, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/mean_origin_error", mean_origin_error, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pass_rate_origin", pass_rate_origin, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/map_type", map_type, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/map_all", map_all, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)

        # Print concise summary
        if self.trainer.is_global_zero:
            print("\n--- Test Results ---")
            print(f"Total Samples: {total_predictions}")
            print(f"Mean IoU: {mean_iou:.4f}")
            print(f"PDet (IoU Pass Rate @ >{self.config.test_iou_threshold:.2f} IoU): {p_det:.2f}%")
            print(f"Mean Point Error (L2): {mean_point_error:.4f}")
            print(f"ERR_ADir (Mean Axis Error): {err_adir:.2f} degrees")
            print(f"Pass Rate Axis (correct axis for matched preds): {pass_rate_axis:.2f}%")
            print(f"Pass Rate Type (correct type for matched preds): {pass_rate_type:.2f}%")
            origin_thresh_val = getattr(self.config, "test_origin_threshold", 0.1)
            print(f"Mean Origin Error (normalized, for matched rotational preds): {mean_origin_error:.4f}")
            print(f"Pass Rate Origin (correct origin for matched rotational preds @ <{origin_thresh_val:.2f} error): {pass_rate_origin:.2f}%")
            print(f"mAP_Type (IoU > {self.config.test_iou_threshold:.2f} & Correct Type): {map_type:.2f}%")
            print(f"mAP_All (IoU > {self.config.test_iou_threshold:.2f} & Correct Type & Axis): {map_all:.2f}%")

        # Reset accumulators for potential further test runs
        self._test_ious.clear()
        self._test_point_errors.clear()
        self._test_axis_errors_matched.clear()
        self._test_num_matched = 0
        self._test_correct_axis_predictions = 0
        self._test_correct_type_in_matched = 0
        self._test_correct_all = 0
        self._test_origin_errors_matched.clear()
        self._test_correct_origin_predictions = 0
        self._test_num_rotational_matched = 0

    # --- Utilities for testing ---
    @staticmethod
    def _mask_to_bbox(mask: torch.Tensor) -> torch.Tensor:
        """mask: (H,W) float or bool tensor -> (x_min,y_min,x_max,y_max)"""
        mask_bool = mask.bool()
        if mask_bool.sum() == 0:
            return torch.zeros(4, device=mask.device)
        rows = torch.any(mask_bool, dim=1)
        cols = torch.any(mask_bool, dim=0)
        y_min, y_max = torch.where(rows)[0][[0, -1]]
        x_min, x_max = torch.where(cols)[0][[0, -1]]
        return torch.tensor([x_min, y_min, x_max, y_max], device=mask.device)

    @staticmethod
    def _bbox_iou(box_pred_xyxy: torch.Tensor, box_gt_xywh: torch.Tensor) -> torch.Tensor:
        box_gt_xyxy = torch.cat([box_gt_xywh[:2], box_gt_xywh[:2] + box_gt_xywh[2:]])
        xA = torch.max(box_pred_xyxy[0], box_gt_xyxy[0])
        yA = torch.max(box_pred_xyxy[1], box_gt_xyxy[1])
        xB = torch.min(box_pred_xyxy[2], box_gt_xyxy[2])
        yB = torch.min(box_pred_xyxy[3], box_gt_xyxy[3])
        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
        boxPredArea = (box_pred_xyxy[2] - box_pred_xyxy[0]) * (box_pred_xyxy[3] - box_pred_xyxy[1])
        boxGtArea = box_gt_xywh[2] * box_gt_xywh[3]
        return interArea / (boxPredArea + boxGtArea - interArea + 1e-7)

    @staticmethod
    def _mask_iou(mask_pred: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        mask_pred_bool = mask_pred.bool()
        mask_gt_bool = mask_gt.bool()
        intersection = torch.logical_and(mask_pred_bool, mask_gt_bool).sum()
        union = torch.logical_or(mask_pred_bool, mask_gt_bool).sum()
        return (intersection.float() + 1e-7) / (union.float() + 1e-7)

    @staticmethod
    def _axis_error_deg(pred_axis: torch.Tensor, gt_axis: torch.Tensor) -> torch.Tensor:
        pred_axis = pred_axis.squeeze()
        gt_axis = gt_axis.squeeze()
        gt_axis_norm = F.normalize(gt_axis, p=2, dim=-1)
        pred_axis_norm = F.normalize(pred_axis, p=2, dim=-1)
        cos_sim = torch.dot(gt_axis_norm, pred_axis_norm)
        cos_sim = torch.abs(cos_sim)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        axis_error_rad = torch.acos(cos_sim)
        return axis_error_rad * 180.0 / torch.tensor(3.141592653589793, device=pred_axis.device)

    # --- Visualization helper ---
    @staticmethod
    def _save_test_debug_visualization(
        image_tensor: torch.Tensor,
        gt_mask_tensor: torch.Tensor,
        pred_mask_prob_tensor: torch.Tensor,
        gt_bbox: torch.Tensor,
        description: str,
        pred_threshold: float,
        output_dir: str,
        sample_index: int,
        iou_value: float,
    ) -> None:
        import numpy as _np
        import cv2 as _cv2
        os.makedirs(output_dir, exist_ok=True)

        img_np = image_tensor.numpy().transpose(1, 2, 0)
        mean = _np.array([0.485, 0.456, 0.406])
        std = _np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = _np.clip(img_np, 0, 1)
        img_bgr = _cv2.cvtColor((img_np * 255).astype(_np.uint8), _cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape

        gt_mask_np = gt_mask_tensor.numpy().squeeze()
        pred_mask_prob_np = pred_mask_prob_tensor.numpy().squeeze()

        gt_mask_resized = _cv2.resize(gt_mask_np, (w, h), interpolation=_cv2.INTER_NEAREST)
        pred_mask_prob_resized = _cv2.resize(pred_mask_prob_np, (w, h), interpolation=_cv2.INTER_LINEAR)

        gt_mask_binary = (gt_mask_resized > 0.5).astype(_np.uint8)

        # Left: probability overlay
        overlay_prob = _np.zeros_like(img_bgr, dtype=_np.uint8)
        overlay_prob[gt_mask_binary == 1, 1] = 255  # Green
        red_channel_prob = (pred_mask_prob_resized * 255).astype(_np.uint8)
        overlay_prob[:, :, 2] = red_channel_prob
        vis_prob = _cv2.addWeighted(img_bgr, 0.6, overlay_prob, 0.4, 0)
        _cv2.putText(vis_prob, "Left: Sigmoid (Red)", (10, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, _cv2.LINE_AA)

        # Right: thresholded + bboxes
        pred_mask_binary = (pred_mask_prob_resized > pred_threshold).astype(_np.uint8)
        overlay_binary = _np.zeros_like(img_bgr, dtype=_np.uint8)
        overlay_binary[gt_mask_binary == 1, 1] = 255
        overlay_binary[pred_mask_binary == 1, 2] = 255
        vis_binary = _cv2.addWeighted(img_bgr, 0.6, overlay_binary, 0.4, 0)
        _cv2.putText(vis_binary, "Right: Thresholded (Yellow=Overlap)", (10, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, _cv2.LINE_AA)

        # GT bbox
        x, y, w_box, h_box = map(int, gt_bbox.numpy())
        _cv2.rectangle(vis_binary, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        # Pred bbox from thresholded mask
        pred_bbox_xyxy = OPDRealTrainingModule._mask_to_bbox(torch.from_numpy(pred_mask_binary))
        x1, y1, x2, y2 = map(int, pred_bbox_xyxy.numpy())
        _cv2.rectangle(vis_binary, (x1, y1), (x2, y2), (0, 0, 255), 2)

        combined_vis = _np.hstack((vis_prob, vis_binary))
        _cv2.putText(combined_vis, f"IoU: {iou_value:.2f}", (10, h - 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, _cv2.LINE_AA)
        if isinstance(description, str):
            _cv2.putText(combined_vis, f"Desc: {description[:60]}", (10, h - 40), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, _cv2.LINE_AA)

        out_path = os.path.join(output_dir, f"sample_{sample_index:06d}_iou{iou_value:.2f}.png")
        _cv2.imwrite(out_path, combined_vis)

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
    LightningCLI(OPDRealTrainingModule, OPDRealDataModule, save_config_callback=None)
