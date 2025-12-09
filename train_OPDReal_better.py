import os
from torch._tensor import Tensor
import typing
from typing import Any
import warnings
import json

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
        self.trajectory_loss_fn = nn.MSELoss()

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
        self._test_correct_pdet_ma: int = 0
        self._test_correct_pdet_mao: int = 0
        self._test_origin_errors_matched: typing.List[float] = []
        self._test_correct_origin_predictions: int = 0
        self._test_num_rotational_matched: int = 0
        self._test_debug_print_count: int = 0
        self.indices_to_visualize: typing.Optional[set] = None

    def forward(
        self, img, depth, tokenized_word, mask_condition, point_condition, motion_gt
    ):
        return self.model(
            img, depth, tokenized_word, mask_condition, point_condition, motion_gt
        )

    def _common_step(self, batch, batch_idx, step_type="train"):
        trajectory_gt = None
        motion_origin_3d = None
        if len(batch) == 13:  # SF3D with trajectory
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
                _rgb_filename,
                motion_origin_3d,
                _camera_intrinsic,
                trajectory_gt,
            ) = batch
        elif len(batch) > 10:  # Other SF3D case, no trajectory
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
                *_,
            ) = batch
        else:  # OPDReal
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
            trajectory_pred,
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

        if trajectory_gt is not None:
            # Convert GT trajectory to relative coordinates (first point at origin)
            trajectory_gt_device = trajectory_gt.to(trajectory_pred.device)
            trajectory_gt_first = trajectory_gt_device[:, 0:1, :]  # (B, 1, 3)
            trajectory_gt_relative = trajectory_gt_device - trajectory_gt_first  # (B, N, 3)
            
            # Model predicts relative trajectory, so compare directly
            L_trajectory = self.trajectory_loss_fn(trajectory_pred, trajectory_gt_relative)
            trajectory_weight = getattr(self.loss_params, "trajectory_weight", 1.0)
            total_loss += trajectory_weight * L_trajectory
            self.log(
                f"{step_type}/L_trajectory",
                L_trajectory,
                on_step=(step_type == "train"),
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            
            # Add geometric consistency losses between trajectory and motion
            # Note: For relative trajectory, motion_origin is also relative to trajectory start
            if motion_origin_3d is not None:
                motion_origin_3d_device = motion_origin_3d.to(trajectory_pred.device)
                # Convert motion origin to relative coordinates (relative to trajectory first point)
                motion_origin_3d_relative = motion_origin_3d_device - trajectory_gt_first.squeeze(1)  # (B, 3)
                trajectory_gt_first_relative = torch.zeros_like(motion_origin_3d_relative)  # (B, 3) - at origin
                
                # Loss 1: pred vector <-> gt traj (motion_pred vs trajectory_gt)
                L_geometric_pred_vector_gt_traj = self._geometric_consistency_loss(
                    trajectory_pred=trajectory_gt_relative,
                    motion_pred=motion_pred,
                    motion_type_gt=motion_type_gt.to(trajectory_pred.device),
                    motion_origin_3d=motion_origin_3d_relative,
                    trajectory_gt_first=trajectory_gt_first_relative,
                )
                geometric_weight = getattr(self.loss_params, "geometric_weight", 1.0)
                total_loss += geometric_weight * L_geometric_pred_vector_gt_traj
                self.log(
                    f"{step_type}/L_geometric_pred_vector_gt_traj",
                    L_geometric_pred_vector_gt_traj,
                    on_step=(step_type == "train"),
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
                
                # Loss 2: pred traj <-> gt vector (trajectory_pred vs motion_gt)
                L_geometric_pred_traj_gt_vector = self._geometric_consistency_loss(
                    trajectory_pred=trajectory_pred,
                    motion_pred=motion_gt.to(trajectory_pred.device),
                    motion_type_gt=motion_type_gt.to(trajectory_pred.device),
                    motion_origin_3d=motion_origin_3d_relative,
                    trajectory_gt_first=trajectory_gt_first_relative,
                )
                trajectory_to_motion_weight = getattr(self.loss_params, "trajectory_to_motion_weight", 1.0)
                total_loss += trajectory_to_motion_weight * L_geometric_pred_traj_gt_vector
                self.log(
                    f"{step_type}/L_geometric_pred_traj_gt_vector",
                    L_geometric_pred_traj_gt_vector,
                    on_step=(step_type == "train"),
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
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
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
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
            original_image_size_list,
            trajectory_gt_list,
            camera_intrinsic_list,
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
                trajectory_pred,
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
                    trajectory_gt=trajectory_gt_list[i],
                    trajectory_pred=trajectory_pred[i],
                    camera_intrinsic=camera_intrinsic_list[i],
                    original_image_size=original_image_size_list[i],
                    depth_tensor=depth[i],
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

    def on_test_start(self):
        vis_indices = getattr(self.config, "test_vis_indices", None)
        if vis_indices and self.trainer.is_global_zero:
            self.indices_to_visualize = set(vis_indices)
        else:
            self.indices_to_visualize = None

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
            category_id,
            composite_key,
        ) = batch[:11]
        
        
        
        # Unpack optional camera parameters if they exist
        camera_params_in_batch = len(batch) > 11
        if camera_params_in_batch:
            intrinsic_matrix, motion_origin_3d_gt = batch[11], batch[12]
        else:
            intrinsic_matrix, motion_origin_3d_gt = None, None

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
        dm = self.trainer.datamodule
        origin_norm_diagonals = getattr(dm, "origin_norm_diagonals", None)

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

                is_origin_correct = False
                # For translational motion, origin check is passed by default
                if motion_type_gt[i] == 0:
                    is_origin_correct = True
                # For rotational motion, we must calculate the 3D error
                elif motion_type_gt[i] == 1 and camera_params_in_batch and origin_norm_diagonals is not None:
                    self._test_num_rotational_matched += 1
                    # 1. Get predicted 2D origin and depth map
                    pred_origin_norm = coords_hat[i].detach() # (x, y) in [0, 1]
                    depth_map = depth[i].squeeze() # (H, W)
                    H, W = depth_map.shape

                    # 2. Convert normalized coords to pixel coords
                    u, v = int(pred_origin_norm[0] * W), int(pred_origin_norm[1] * H)
                    
                    # 3. Sample depth value (with a small patch for robustness)
                    patch_size = 5
                    u_start, v_start = max(0, u - patch_size//2), max(0, v - patch_size//2)
                    u_end, v_end = min(W, u + patch_size//2 + 1), min(H, v + patch_size//2 + 1)
                    depth_patch = depth_map[v_start:v_end, u_start:u_end]
                    
                    # Filter out zero depth values which are often invalid
                    valid_depths = depth_patch[depth_patch > 0]
                    z_mm = valid_depths.mean().item() if valid_depths.numel() > 0 else depth_map[v, u].item()

                    # 4. Unproject 2D point + depth to 3D
                    if z_mm > 1e-6 and intrinsic_matrix is not None:
                        K = intrinsic_matrix[i].to(self.device)
                        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                        
                        x_cam_mm = (u - cx) * z_mm / fx
                        y_cam_mm = (v - cy) * z_mm / fy
                        pred_origin_3d_mm = torch.tensor([x_cam_mm, y_cam_mm, z_mm], device=self.device)
                        
                        # Convert prediction to meters
                        pred_origin_3d = pred_origin_3d_mm / 1000.0
                        
                        # 5. Calculate error against 3D GT origin
                        gt_origin_3d = motion_origin_3d_gt[i].to(self.device)
                        origin_error = torch.linalg.norm(pred_origin_3d - gt_origin_3d)

                        # Use 3D diagonal from JSON for normalization, adapting to format
                        json_format = getattr(dm, "origin_norm_json_format", "real")
                        if json_format == 'real':
                            lookup_key = category_id[i].item()
                        else: # 'multi'
                            lookup_key = composite_key[i]
                        
                        
                        diagonal_3d = origin_norm_diagonals.get(lookup_key)

                        if diagonal_3d is not None and diagonal_3d > 1e-6:
                            norm_dist = origin_error / diagonal_3d
                            self._test_origin_errors_matched.append(norm_dist.item())
                            origin_threshold = getattr(self.config, "test_origin_threshold", 0.1)
                            if norm_dist <= origin_threshold:
                                is_origin_correct = True
                                self._test_correct_origin_predictions += 1

                if is_type_correct and is_axis_correct:
                    self._test_correct_pdet_ma += 1
                
                if is_type_correct and is_axis_correct and is_origin_correct:
                    self._test_correct_pdet_mao += 1


            # Debug visualization
            if do_vis and self.trainer.is_global_zero:
                current_sample_index: Any = batch_idx * batch_size + i
                if self.indices_to_visualize is None or current_sample_index in self.indices_to_visualize:
                    vis_dir = getattr(
                        self.config, "test_vis_output_dir", "opdreal_debug_visualizations"
                    )
                    point_pred_prob = torch.sigmoid(point_pred_logits)
                    
                    print(batch[-1][i])


                    self._save_opdreal_test_debug_visualizations(
                        image_tensor=img[i].detach().cpu(),
                        point_pred_prob_tensor=point_pred_prob[i].detach().cpu(),
                        mask_pred_prob_tensor=mask_pred_prob[i].detach().cpu(),
                        motion_pred=motion_pred[i].detach().cpu(),
                        pred_motion_type=int(pred_types[i].item()),
                        gt_point_norm=point_gt_norm[i].detach().cpu(),
                        gt_mask_tensor=mask_gt[i].detach().cpu(),
                        gt_motion=motion_gt[i].detach().cpu(),
                        description=word_str_list[i],
                        output_dir=vis_dir,
                        sample_index=current_sample_index,
                    )

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
            pdet_m = 100.0 * self._test_correct_type_in_matched / total_predictions
            pdet_ma = 100.0 * self._test_correct_pdet_ma / total_predictions
            pdet_mao = 100.0 * self._test_correct_pdet_mao / total_predictions
        else:
            p_det = 0.0
            pdet_m = 0.0
            pdet_ma = 0.0
            pdet_mao = 0.0

        # Always log to progress bar/console; optionally to external logger
        self.log("test/mean_iou", mean_iou, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/p_det", p_det, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pdet_m", pdet_m, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pdet_ma", pdet_ma, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pdet_mao", pdet_mao, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/mean_point_error", mean_point_error, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/err_adir_deg", err_adir, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pass_rate_axis", pass_rate_axis, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pass_rate_type", pass_rate_type, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/mean_origin_error", mean_origin_error, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pass_rate_origin", pass_rate_origin, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        

        # Print concise summary
        if self.trainer.is_global_zero:
            print("\n--- Test Results ---")
            print(f"Total Samples: {total_predictions}")
            print(f"Mean IoU: {mean_iou:.4f}")
            print(f"PDet (IoU Pass Rate @ >{self.config.test_iou_threshold:.2f} IoU): {p_det:.2f}%")
            print(f"PDet+M (IoU + Motion Type): {pdet_m:.2f}%")
            print(f"PDet+MA (IoU + Motion Type + Axis): {pdet_ma:.2f}%")
            print(f"PDet+MAO (IoU + Motion Type + Axis + Origin): {pdet_mao:.2f}%")
            print(f"\n--- Detailed Stats ---")
            print(f"Mean Point Error (L2): {mean_point_error:.4f}")
            print(f"ERR_ADir (Mean Axis Error for matched): {err_adir:.2f} degrees")
            print(f"Pass Rate Axis (correct axis for matched): {pass_rate_axis:.2f}%")
            print(f"Pass Rate Type (correct type for matched): {pass_rate_type:.2f}%")
            origin_thresh_val = getattr(self.config, "test_origin_threshold", 0.1)
            print(f"Mean Origin Error (normalized, for matched rotational): {mean_origin_error:.4f}")
            print(f"Pass Rate Origin (correct origin for matched rotational @ <{origin_thresh_val:.2f} error): {pass_rate_origin:.2f}%")


        # Reset accumulators for potential further test runs
        self._test_ious.clear()
        self._test_point_errors.clear()
        self._test_axis_errors_matched.clear()
        self._test_num_matched = 0
        self._test_correct_axis_predictions = 0
        self._test_correct_type_in_matched = 0
        self._test_correct_pdet_ma = 0
        self._test_correct_pdet_mao = 0
        self._test_origin_errors_matched.clear()
        self._test_correct_origin_predictions = 0
        self._test_num_rotational_matched = 0
        self._test_debug_print_count = 0

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
    def _geometric_consistency_loss(
        trajectory_pred: torch.Tensor,  # (B, N, 3)
        motion_pred: torch.Tensor,  # (B, 3)
        motion_type_gt: torch.Tensor,  # (B,)
        motion_origin_3d: torch.Tensor,  # (B, 3)
        trajectory_gt_first: torch.Tensor,  # (B, 3) - first point of GT trajectory
    ) -> torch.Tensor:
        """
        Compute geometric consistency loss between predicted trajectory and motion vector.
        
        For translation (type=0): Loss measures how well trajectory points align with a line
        For rotation (type=1): Loss measures how well trajectory points lie on a circle
        
        Returns: scalar loss value averaged over batch
        """
        B, N, _ = trajectory_pred.shape
        device = trajectory_pred.device
        
        # Normalize motion vector to unit length
        motion_pred_norm = F.normalize(motion_pred, p=2, dim=1, eps=1e-8)  # (B, 3)
        
        total_loss = torch.zeros(B, device=device)
        
        for b in range(B):
            if motion_type_gt[b] == 0:  # Translation - line loss
                # L_line = (1/N) Σ ||(Q_i - P_0) × v||²
                P_0 = motion_origin_3d[b]  # (3,)
                v = motion_pred_norm[b]  # (3,)
                Q = trajectory_pred[b]  # (N, 3)
                
                # Compute cross product: (Q_i - P_0) × v
                Q_minus_P0 = Q - P_0  # (N, 3)
                cross_product = torch.cross(Q_minus_P0, v.unsqueeze(0).expand(N, -1))  # (N, 3)
                squared_distances = torch.sum(cross_product ** 2, dim=1)  # (N,)
                line_loss = squared_distances.mean()
                
                total_loss[b] = line_loss
                
            elif motion_type_gt[b] == 1:  # Rotation - circle loss
                # L_circle = (1/N) Σ [((Q_i - C) · n)² + (||proj_perp(Q_i - C)|| - r)²]
                C = motion_origin_3d[b]  # (3,) - circle center
                n = motion_pred_norm[b]  # (3,) - plane normal (rotation axis)
                Q = trajectory_pred[b]  # (N, 3)
                
                # Compute radius from first GT trajectory point
                Q_first_gt = trajectory_gt_first[b]  # (3,)
                Q_first_minus_C = Q_first_gt - C  # (3,)
                # Project onto plane perpendicular to n: Q - C - ((Q - C) · n)n
                proj_length = torch.dot(Q_first_minus_C, n)
                proj_perp = Q_first_minus_C - proj_length * n  # (3,)
                r = torch.norm(proj_perp)  # scalar - radius
                
                # Compute loss for each predicted point
                Q_minus_C = Q - C  # (N, 3)
                
                # First term: ((Q_i - C) · n)² - ensures points lie in plane perpendicular to n
                dot_n = torch.sum(Q_minus_C * n.unsqueeze(0).expand(N, -1), dim=1)  # (N,)
                plane_dist_sq = dot_n ** 2  # (N,)
                
                # Second term: (||proj_perp(Q_i - C)|| - r)² - ensures points lie on circle
                proj_lengths = torch.sum(Q_minus_C * n.unsqueeze(0).expand(N, -1), dim=1)  # (N,)
                proj_perp_vecs = Q_minus_C - proj_lengths.unsqueeze(1) * n.unsqueeze(0).expand(N, -1)  # (N, 3)
                circle_dists = torch.norm(proj_perp_vecs, dim=1)  # (N,)
                circle_error_sq = (circle_dists - r) ** 2  # (N,)
                
                circle_loss = (plane_dist_sq + circle_error_sq).mean()
                
                total_loss[b] = circle_loss
        
        return total_loss.mean()

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
    def _save_opdreal_test_debug_visualizations(
        image_tensor: torch.Tensor,
        point_pred_prob_tensor: torch.Tensor,
        mask_pred_prob_tensor: torch.Tensor,
        motion_pred: torch.Tensor,  # 3d vector
        pred_motion_type: int,
        gt_point_norm: torch.Tensor,
        gt_mask_tensor: torch.Tensor,
        gt_motion: torch.Tensor,
        description: str,
        output_dir: str,
        sample_index: int,
    ):
        import numpy as np
        import cv2
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)

        # 1. De-normalize image tensor
        img_np_norm = image_tensor.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np_norm + mean
        img_np = np.clip(img_np, 0, 1)
        img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        h, w, _ = img_bgr.shape

        # --- Common drawing components ---
        def apply_geo_annotations(vis_image, pred_px, pred_py, motion_pred_np):
            # Draw interaction point
            cv2.circle(
                vis_image,
                (pred_px, pred_py),
                radius=5,
                color=(255, 255, 255),
                thickness=-1,
            )

            # Draw motion arrow
            motion_xy = motion_pred_np[:2]
            motion_xy_norm = motion_xy / (np.linalg.norm(motion_xy) + 1e-8)
            arrow_length = 50
            arrow_end_x = pred_px + int(motion_xy_norm[0] * arrow_length)
            arrow_end_y = pred_py + int(motion_xy_norm[1] * arrow_length)
            cv2.arrowedLine(
                vis_image,
                (pred_px, pred_py),
                (arrow_end_x, arrow_end_y),
                (255, 255, 255),
                2,  # Increased thickness
            )
            return vis_image

        def apply_text_annotations(vis_image, desc, img_h, img_w):
            # Add caption with wrapping
            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            margin = 10
            text = f"Desc: {desc}"
            words = text.split(" ")
            lines = []
            current_line = words[0]
            for word in words[1:]:
                test_line = f"{current_line} {word}"
                (text_width, text_height), _ = cv2.getTextSize(
                    test_line, font, font_scale, font_thickness
                )
                if text_width > img_w - 2 * margin:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            lines.append(current_line)
            y = img_h - margin - (len(lines) - 1) * (text_height + margin)
            for i, line in enumerate(lines):
                line_y = y + i * (text_height + margin)
                (line_width, text_height), _ = cv2.getTextSize(
                    line, font, font_scale, font_thickness
                )
                cv2.rectangle(
                    vis_image,
                    (margin - 5, line_y - text_height - 5),
                    (margin + line_width + 5, line_y + 5),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    vis_image,
                    line,
                    (margin, line_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA,
                )
            return vis_image

        # --- Visualization 1: Point Heatmap ---
        point_pred_prob_np = point_pred_prob_tensor.float().numpy().squeeze()
        point_heatmap_resized = cv2.resize(
            point_pred_prob_np, (w, h), interpolation=cv2.INTER_LINEAR
        )
        point_heatmap_inverted = 1 - point_heatmap_resized
        point_heatmap_colored = cv2.applyColorMap(
            (point_heatmap_inverted * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        vis_image_point = cv2.addWeighted(img_bgr.copy(), 0.6, point_heatmap_colored, 0.4, 0)

        # Get interaction point from argmax of point heatmap
        (pred_py, pred_px) = np.unravel_index(
            np.argmax(point_heatmap_resized), point_heatmap_resized.shape
        )

        vis_image_point = apply_geo_annotations(
            vis_image_point, pred_px, pred_py, motion_pred.numpy()
        )
        out_path_point = os.path.join(
            output_dir, f"sample_{sample_index:06d}_point.png"
        )
        cv2.imwrite(out_path_point, vis_image_point)

        # --- Visualization 2: Mask Heatmap ---
        mask_prob_np = mask_pred_prob_tensor.float().numpy().squeeze()
        # Apply sigmoid sharpening
        k = 20  # Steepness factor
        sigmoid_mask = 1.0 / (1.0 + np.exp(-k * (mask_prob_np - 0.5)))
        mask_heatmap_resized = cv2.resize(
            sigmoid_mask, (w, h), interpolation=cv2.INTER_LINEAR
        )
        mask_heatmap_inverted = 1 - mask_heatmap_resized
        mask_heatmap_colored = cv2.applyColorMap(
            (mask_heatmap_inverted * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        vis_image_mask = cv2.addWeighted(img_bgr.copy(), 0.6, mask_heatmap_colored, 0.4, 0)
        vis_image_mask = apply_geo_annotations(
            vis_image_mask, pred_px, pred_py, motion_pred.numpy()
        )
        out_path_mask = os.path.join(
            output_dir, f"sample_{sample_index:06d}_mask.png"
        )
        cv2.imwrite(out_path_mask, vis_image_mask)

        # --- Visualization 3: Ground Truth ---
        vis_image_gt = img_bgr.copy()

        # Draw GT mask as an overlay
        gt_mask_np = gt_mask_tensor.float().numpy().squeeze()
        gt_mask_resized = cv2.resize(gt_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        # Create a green overlay for the mask
        gt_mask_overlay = np.zeros_like(vis_image_gt, dtype=np.uint8)
        gt_mask_overlay[gt_mask_resized > 0.5] = (0, 200, 0)  # BGR green
        vis_image_gt = cv2.addWeighted(vis_image_gt, 1.0, gt_mask_overlay, 0.4, 0)

        # Get GT interaction point in pixel coordinates
        gt_px = int(gt_point_norm[0] * w)
        gt_py = int(gt_point_norm[1] * h)

        # Use the same annotation function for the GT visualization
        vis_image_gt = apply_geo_annotations(
            vis_image_gt, gt_px, gt_py, gt_motion.numpy()
        )
        vis_image_gt = apply_text_annotations(vis_image_gt, description, h, w)
        
        out_path_gt = os.path.join(
            output_dir, f"sample_{sample_index:06d}_gt.png"
        )
        cv2.imwrite(out_path_gt, vis_image_gt)

    def _collect_vis_samples_from_batch(self, batch):
        # Match the unpacking logic in _common_step
        trajectory_gt = None
        camera_intrinsic = None
        img, depth, word_str_list, mask_gt, bbox, point_gt_norm, motion_gt, motion_type_gt, img_size = [None] * 9

        if len(batch) >= 9:
             base_items = batch[:9]
             img, depth, word_str_list, mask_gt, bbox, point_gt_norm, motion_gt, motion_type_gt, img_size = base_items

        if len(batch) == 13:  # SF3D with trajectory
             camera_intrinsic = batch[11]
             trajectory_gt = batch[12]

        batch_size = img.size(0) if hasattr(img, "size") else 0
        for i in range(batch_size):
            sample_trajectory = trajectory_gt[i].detach().cpu() if trajectory_gt is not None else None
            sample_intrinsic = camera_intrinsic[i].detach().cpu() if camera_intrinsic is not None else None

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
                sample_trajectory,
                sample_intrinsic,
            )

            # True reservoir sampling for uniform random selection
            if self._vis_rng is None:
                self._vis_rng = torch.Generator(device="cpu").manual_seed(
                    int(self.base_seed + int(self.current_epoch))
                )
            
            if len(self._vis_buffer) < self.vis_num_samples:
                # Fill buffer first
                self._vis_buffer.append(sample)
            else:
                # Reservoir sampling: replace with probability k/n
                # where k = vis_num_samples, n = _vis_seen_count + 1
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
