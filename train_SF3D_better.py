import os
import typing
import warnings

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger

from config.opd_train import Config, LossParams, ModelParams, OptimizerParams
from datasets.scenefun3d_datamodule import SF3DDataModule
from model.segmenter import CRIS
from train_OPDReal_better import OPDRealTrainingModule
from utils.dataset import tokenize
from utils.tools import create_composite_visualization, make_gaussian_map
import torch.nn as nn

torch.set_float32_matmul_precision("high")

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SF3DTrainingModule(OPDRealTrainingModule):
    def on_test_start(self):
        vis_indices = getattr(self.config, "test_vis_indices", None)
        if vis_indices and self.trainer.is_global_zero:
            self.indices_to_visualize = set(vis_indices)
        else:
            self.indices_to_visualize = None

        # Initialize accumulators for metrics across ALL samples
        self._test_axis_errors_all = []
        self._test_type_correct_all = 0
        self._test_ma_correct_all = 0
        self._test_origin_errors_rotational_all = []

    @staticmethod
    def _save_sf3d_test_debug_visualizations(
        full_image_path: str,
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

        try:
            # 1. Load full-res image
            with Image.open(full_image_path) as img:
                img_rgb = img.convert("RGB")
                img_np = np.array(img_rgb)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        except FileNotFoundError:
            print(f"Warning: Could not find image for visualization at {full_image_path}")
            return

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
            arrow_length = 200
            arrow_end_x = pred_px + int(motion_xy_norm[0] * arrow_length)
            arrow_end_y = pred_py + int(motion_xy_norm[1] * arrow_length)
            cv2.arrowedLine(
                vis_image,
                (pred_px, pred_py),
                (arrow_end_x, arrow_end_y),
                (255, 255, 255),
                12,  # Increased thickness
            )
            return vis_image

        def apply_text_annotations(vis_image, desc, img_h, img_w):
            # Add caption with wrapping
            font_scale = 3
            font_thickness = 2
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

        out_path_gt = os.path.join(output_dir, f"sample_{sample_index:06d}_gt.png")
        cv2.imwrite(out_path_gt, vis_image_gt)

    def test_step(self, batch, batch_idx):
        # Unpack batch, handling optional camera parameters
        camera_params_in_batch = len(batch) > 10
        if camera_params_in_batch:
            # This is the new format including trajectory
            if len(batch) == 13:
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
                    rgb_image_filenames,
                    motion_origin_3d_gt,
                    intrinsic_matrix,
                    trajectory_gt,
                ) = batch
            else: # Old format without trajectory
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
                    rgb_image_filenames,
                    motion_origin_3d_gt,
                    intrinsic_matrix,
                ) = batch
                trajectory_gt = None
        else:
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
                rgb_image_filenames,
            ) = batch
            motion_origin_3d_gt, intrinsic_matrix = None, None

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
                trajectory_pred,
            ) = self(img, depth, tokenized_words, None, None, None)

        mask_pred_prob = torch.sigmoid(mask_pred_logits)
        mask_pred_upsampled = F.interpolate(
            mask_pred_prob, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False
        )
        pred_types = torch.argmax(motion_type_logits, dim=1)

        batch_size = img.size(0)
        for i in range(batch_size):
            pred_mask_binary = (mask_pred_upsampled[i] > self.config.test_pred_threshold).float()
            iou_val = self._mask_iou(pred_mask_binary, mask_gt[i]).item()
            self._test_ious.append(iou_val)

            point_err = torch.linalg.norm(coords_hat[i] - point_gt_norm[i]).item()
            self._test_point_errors.append(point_err)

            # --- 3D Origin error calculation for rotational motions ---
            if camera_params_in_batch and motion_type_gt[i] == 1:  # Rotational
                pred_origin_norm = coords_hat[i].detach()
                depth_map = depth[i].squeeze()
                H_img, W_img = depth_map.shape
                u, v = int(pred_origin_norm[0] * W_img), int(
                    pred_origin_norm[1] * H_img
                )

                patch_size = 5
                u_start, v_start = max(0, u - patch_size // 2), max(
                    0, v - patch_size // 2
                )
                u_end, v_end = min(W_img, u + patch_size // 2 + 1), min(
                    H_img, v + patch_size // 2 + 1
                )
                depth_patch = depth_map[v_start:v_end, u_start:u_end]

                valid_depths = depth_patch[depth_patch > 0]
                z_m = (
                    valid_depths.mean().item()
                    if valid_depths.numel() > 0
                    else depth_map[v, u].item()
                )

                if (
                    z_m > 1e-6
                    and intrinsic_matrix is not None
                    and motion_origin_3d_gt is not None
                ):
                    K = intrinsic_matrix[i].to(self.device)
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

                    x_cam_m = (u - cx) * z_m / fx
                    y_cam_m = (v - cy) * z_m / fy
                    pred_origin_3d_m = torch.tensor(
                        [x_cam_m, y_cam_m, z_m], device=self.device
                    )

                    gt_origin_3d = motion_origin_3d_gt[i].to(self.device)
                    origin_error = torch.linalg.norm(
                        pred_origin_3d_m - gt_origin_3d
                    ).item()
                    self._test_origin_errors_rotational_all.append(origin_error)

            # --- Motion and Axis evaluation (for all samples) ---
            axis_err = self._axis_error_deg(motion_pred[i], motion_gt[i]).item()
            # print(f"Axis error: {axis_err}")
            self._test_axis_errors_all.append(axis_err)

            is_axis_correct = axis_err <= self.config.test_motion_threshold_deg
            is_type_correct = pred_types[i] == motion_type_gt[i]

            if is_type_correct:
                self._test_type_correct_all += 1

            if is_axis_correct and is_type_correct:
                self._test_ma_correct_all += 1

            # --- Original evaluation for IoU-matched samples ---
            if iou_val > self.config.test_iou_threshold:
                self._test_num_matched += 1

                self._test_axis_errors_matched.append(axis_err)

                if is_axis_correct:
                    self._test_correct_axis_predictions += 1

                if is_type_correct:
                    self._test_correct_type_in_matched += 1

                if is_axis_correct and is_type_correct:
                    self._test_correct_pdet_ma += 1

            do_vis = getattr(self.config, "test_visualize_debug", False)
            if do_vis and self.trainer.is_global_zero:
                current_sample_index = batch_idx * batch_size + i
                if self.indices_to_visualize is None or current_sample_index in self.indices_to_visualize:
                    vis_dir = getattr(
                        self.config, "test_vis_output_dir", "sf3d_debug_visualizations"
                    )
                    dm = getattr(self.trainer, "datamodule", None)
                    if dm:
                        data_root = getattr(dm, "train_data_dir", None)
                        if data_root:
                            full_image_path = os.path.join(
                                data_root, "images", rgb_image_filenames[i]
                            )
                            point_pred_prob = torch.sigmoid(point_pred_logits)
                            mask_pred_prob = torch.sigmoid(mask_pred_logits)

                            self._save_sf3d_test_debug_visualizations(
                                full_image_path=full_image_path,
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
        # --- Gather metrics from all processes ---
        # Use all_gather to collect tensors from all devices.
        # The all_gather op is asynchronous, so we need to wait for it to finish.
        # The gathered tensors will be on the device of the current process.

        all_ious = self.all_gather(torch.tensor(self._test_ious, device=self.device))
        all_point_errors = self.all_gather(
            torch.tensor(self._test_point_errors, device=self.device)
        )
        all_axis_errors_matched = self.all_gather(
            torch.tensor(self._test_axis_errors_matched, device=self.device)
        )
        all_axis_errors_all = self.all_gather(
            torch.tensor(self._test_axis_errors_all, device=self.device)
        )
        all_origin_errors_rotational = self.all_gather(
            torch.tensor(self._test_origin_errors_rotational_all, device=self.device)
        )

        # For counters, we need to gather and then sum them up.
        num_matched_tensor = torch.tensor(
            self._test_num_matched, dtype=torch.long, device=self.device
        )
        total_num_matched = self.all_gather(num_matched_tensor).sum()

        type_correct_all_tensor = torch.tensor(
            self._test_type_correct_all, dtype=torch.long, device=self.device
        )
        total_type_correct_all = self.all_gather(type_correct_all_tensor).sum()

        ma_correct_all_tensor = torch.tensor(
            self._test_ma_correct_all, dtype=torch.long, device=self.device
        )
        total_ma_correct_all = self.all_gather(ma_correct_all_tensor).sum()

        # --- Calculate metrics on aggregated results ---
        # The dataloader is wrapped in a DistributedSampler, so len(dataset) gives the full size.
        total_predictions = len(self.trainer.datamodule.test_dataloader().dataset)

        mean_iou = float(all_ious.mean().item()) if all_ious.numel() > 0 else 0.0
        mean_point_error = (
            float(all_point_errors.mean().item()) if all_point_errors.numel() > 0 else 0.0
        )
        err_adir_matched = (
            float(all_axis_errors_matched.mean().item())
            if all_axis_errors_matched.numel() > 0
            else 0.0
        )
        err_adir_all = (
            float(all_axis_errors_all.mean().item())
            if all_axis_errors_all.numel() > 0
            else 0.0
        )
        mean_origin_error_m = (
            float(all_origin_errors_rotational.mean().item())
            if all_origin_errors_rotational.numel() > 0
            else 0.0
        )

        if total_predictions > 0:
            p_det = 100.0 * total_num_matched / total_predictions
            pass_rate_m = 100.0 * total_type_correct_all / total_predictions
            pass_rate_ma = 100.0 * total_ma_correct_all / total_predictions
        else:
            p_det, pass_rate_m, pass_rate_ma = 0.0, 0.0, 0.0

        self.log("test/p_det", p_det, prog_bar=True, logger=True, sync_dist=True)
        self.log("test/pass_rate_m", pass_rate_m, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            "test/pass_rate_ma", pass_rate_ma, prog_bar=True, logger=True, sync_dist=True
        )
        self.log("test/mean_iou", mean_iou, prog_bar=False, logger=True, sync_dist=True)
        self.log(
            "test/mean_point_error",
            mean_point_error,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/err_adir_matched_deg",
            err_adir_matched,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/err_adir_all_deg",
            err_adir_all,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/mean_origin_error_m",
            mean_origin_error_m,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        if self.trainer.is_global_zero:
            print("\n--- SF3D Test Results ---")
            print(f"Total Samples: {total_predictions}")
            print(f"Mean IoU: {mean_iou:.4f}")
            print(f"PDet (IoU > {self.config.test_iou_threshold:.2f}): {p_det:.2f}%")
            print(f"M Pass Rate (Motion Type): {pass_rate_m:.2f}%")
            print(f"MA Pass Rate (Motion Type + Axis): {pass_rate_ma:.2f}%")
            print(f"\n--- Detailed Stats ---")
            print(f"Mean Point Error (L2): {mean_point_error:.4f}")
            print(f"Mean Axis Error (all): {err_adir_all:.2f} degrees")
            print(f"Mean Axis Error (matched): {err_adir_matched:.2f} degrees")
            if all_origin_errors_rotational.numel() > 0:
                print(
                    f"Mean Origin Error (m, for rotational): {mean_origin_error_m:.4f}"
                )

        # Reset accumulators
        self._test_ious.clear()
        self._test_point_errors.clear()
        self._test_axis_errors_matched.clear()
        self._test_num_matched = 0
        self._test_correct_axis_predictions = 0
        self._test_correct_type_in_matched = 0
        self._test_correct_pdet_ma = 0

        # Reset new accumulators
        self._test_axis_errors_all.clear()
        self._test_type_correct_all = 0
        self._test_ma_correct_all = 0
        self._test_origin_errors_rotational_all.clear()


if __name__ == "__main__":
    LightningCLI(SF3DTrainingModule, SF3DDataModule, save_config_callback=None)
