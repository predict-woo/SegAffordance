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
from model.losses import decode_twist, point_to_line_distance
from model.losses.geometric import (
    backproject_points,
    normalized_intrinsics,
    project_points,
)
from model.losses.split import perpendicular_foot
from model.segmenter import CRIS
from train_OPDReal_better import OPDRealTrainingModule
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
        # Sign-aware axis metrics (2026-08-18): SF3D's stored axis sign is
        # canonical and training is sign-sensitive, but the legacy metric
        # takes |cos| and cannot see a flipped opening direction — the g16
        # probe (tools/diag_axis_sign.py) found ~10% of rot rows predicting
        # near the ANTIPODE of the GT axis while scoring ~0 unsigned. The
        # unsigned columns stay untouched for cross-generation tables;
        # everything below is additive.
        self._test_axis_errors_signed_all = []   # true angle, all axis rows
        self._test_axis_errors_signed_rot = []   # rot rows only (flips there
                                                 # reverse the swing direction)
        self._test_ma_signed_correct_all = 0     # type ok AND signed <= thr
        # 2D reprojection decomposition (2026-08-20): GT 2D track vs the
        # predicted 3D curve projected with the input-depth anchor — the
        # 2D-only arm's headline metric, split into anchor placement vs
        # trajectory shape (first points aligned). Collected only when the
        # batch carries the 2D track (return_trajectory_2d configs).
        self._test_proj2d_err = []
        self._test_proj2d_anchor = []
        self._test_proj2d_shape = []
        # Gen-19 smoothness metric: mean second-difference magnitude of the
        # predicted (and GT, as the floor) 3D trajectory — the quantitative
        # form of "the trajectories are noisy".
        self._test_traj_rough_pred = []
        self._test_traj_rough_gt = []
        self._test_origin_errors_rotational_all = []
        # Ablation arms (2026-08-15 spec): set during test_step when the
        # axis/type heads actually produced predictions.
        self._test_has_axis_head = False
        self._test_has_type_head = False

        # Twist-head metrics (empty/zero when use_twist_head is off).
        # Axis-LINE distance replaces the origin-point error: the twist cannot
        # (by design) say where along the axis the annotated origin sits, so
        # the right question is how far the GT origin lies from the predicted
        # axis line.
        self._test_twist_axis_errors_all = []
        self._test_twist_type_correct_all = 0
        self._test_twist_ma_correct_all = 0
        self._test_twist_line_dist_rotational = []
        # Direction metrics (2026-08-04): the axis metrics above are
        # sign-agnostic, so they cannot show whether the sign-sensitive
        # training actually taught direction. dir_correct = the predicted
        # twist direction agrees IN SIGN with the GT (whose stored sign is
        # canonical — the preprocessor derives the trajectory from it).
        self._test_twist_dir_correct_all = 0
        self._test_traj_dir_cos = []

        # Split-arm (gen-6) metrics; empty when the arm lacks the heads.
        self._test_point3d_errors = []        # ||p_hat - traj_gt[0]||, all rows
        self._test_origin_err_m = []          # ||q_hat - q*||, revolute rows
        self._test_origin_line_err_m = []     # dist(q_hat, GT axis line)
        self._test_radius_err_m = []          # |r_pred - r_gt|, revolute rows

        # gen-7 self-consistency (absolute trajectory only): the lifted 3D
        # point and the absolute trajectory head's first point predict the
        # same quantity — their gap measures head agreement.
        self._test_point_traj0_gap = []       # ||p_hat - traj_pred[0]||, all rows

    @staticmethod
    def _save_sf3d_test_debug_visualizations(
        full_image_path: str,
        point_pred_prob_tensor: typing.Optional[torch.Tensor],
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

        # --- Visualization 1: Point Heatmap (2D-point arms only) ---
        # Split arms (point_prediction_3d) have no 2D point head: skip this
        # panel and anchor the mask panel's annotations at the mask argmax.
        pred_px = pred_py = None
        if point_pred_prob_tensor is not None:
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
        if pred_px is None:
            (pred_py, pred_px) = np.unravel_index(
                np.argmax(mask_heatmap_resized), mask_heatmap_resized.shape
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
        _trajectory_2d_extras = []  # filled only by the 15-tuple format
        if camera_params_in_batch:
            # This is the new format including trajectory (a 15-tuple appends
            # the 2D trajectory columns, unused by these metrics)
            if len(batch) >= 13:
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
                    *_trajectory_2d_extras,
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

        tokenized_words = self.model.tokenize(
            list(word_str_list), self.model_params.word_len
        ).to(self.device)

        # gen-7: fold the batch intrinsics into the normalized form exactly as
        # _common_step does, so the model's lifted fields (point_3d_pred /
        # origin_pred) exist at test time and the 3D metric blocks below run.
        # None (no camera params in the batch) means no lift — as before.
        K_norm = None
        if intrinsic_matrix is not None:
            K_norm = normalized_intrinsics(
                intrinsic_matrix.to(self.device).float(),
                _img_size.to(self.device).float(),
            )

        with torch.no_grad():
            outputs = self(
                img, depth, tokenized_words, None, None, None, None, K_norm
            )
        mask_pred_logits = outputs.mask_logits
        point_pred_logits = outputs.point_logits
        point_uv = outputs.point_uv
        motion_pred = outputs.motion_pred
        motion_type_logits = outputs.motion_type_logits
        trajectory_pred = outputs.trajectory_pred

        mask_pred_prob = torch.sigmoid(mask_pred_logits)
        mask_pred_upsampled = F.interpolate(
            mask_pred_prob, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False
        )
        if motion_type_logits is not None:
            pred_types = torch.argmax(motion_type_logits, dim=1)
        elif outputs.twist_pred is not None:
            # Type head off: type is emergent from the twist's |omega|.
            pred_types = decode_twist(outputs.twist_pred.detach().float())[0].long()
        else:
            pred_types = torch.zeros(img.size(0), dtype=torch.long, device=img.device)

        twist_decoded = (
            decode_twist(outputs.twist_pred.detach().float())
            if outputs.twist_pred is not None
            else None
        )

        batch_size = img.size(0)
        for i in range(batch_size):
            pred_mask_binary = (mask_pred_upsampled[i] > self.config.test_pred_threshold).float()
            iou_val = self._mask_iou(pred_mask_binary, mask_gt[i]).item()
            self._test_ious.append(iou_val)

            if point_uv is not None:
                point_err = torch.linalg.norm(point_uv[i] - point_gt_norm[i]).item()
                self._test_point_errors.append(point_err)

            # --- 3D Origin error calculation for rotational motions ---
            if (
                point_uv is not None
                and camera_params_in_batch
                and motion_type_gt[i] == 1
            ):  # Rotational
                pred_origin_norm = point_uv[i].detach()
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

            # --- Split-arm (gen-6) 3D metrics ---
            if outputs.point_3d_pred is not None and trajectory_gt is not None:
                p_hat = outputs.point_3d_pred[i].detach().float()
                p_gt = trajectory_gt[i, 0].to(p_hat.device).float()
                self._test_point3d_errors.append(
                    torch.linalg.norm(p_hat - p_gt).item()
                )
            if (
                outputs.origin_pred is not None
                and motion_type_gt[i] == 1
                and motion_origin_3d_gt is not None
                and trajectory_gt is not None
            ):
                q_hat = outputs.origin_pred[i].detach().float()
                o_gt = motion_origin_3d_gt[i].to(q_hat.device).float()
                d_gt = F.normalize(motion_gt[i].to(q_hat.device).float(), dim=0)
                p_gt = trajectory_gt[i, 0].to(q_hat.device).float()
                q_star = perpendicular_foot(o_gt[None], d_gt[None], p_gt[None])[0]
                self._test_origin_err_m.append(
                    torch.linalg.norm(q_hat - q_star).item()
                )
                self._test_origin_line_err_m.append(
                    point_to_line_distance(q_hat[None], o_gt[None], d_gt[None])[0].item()
                )
                if outputs.motion_pred is not None:
                    d_hat = F.normalize(
                        outputs.motion_pred[i].detach().float(), dim=0
                    )
                    r_pred = point_to_line_distance(
                        p_gt[None], q_hat[None], d_hat[None]
                    )[0].item()
                    r_gt = point_to_line_distance(
                        p_gt[None], o_gt[None], d_gt[None]
                    )[0].item()
                    self._test_radius_err_m.append(abs(r_pred - r_gt))

            # --- gen-7 self-consistency: lifted point vs absolute traj[0] ---
            # Only meaningful when the trajectory head is ABSOLUTE (relative
            # trajectories start near zero by construction — the gap would
            # just measure ||p_hat||).
            if (
                outputs.point_3d_pred is not None
                and outputs.trajectory_pred is not None
                and getattr(self.model_params, "trajectory_absolute", False)
            ):
                p_hat = outputs.point_3d_pred[i].detach().float()
                t0 = outputs.trajectory_pred[i, 0].detach().float().to(p_hat.device)
                self._test_point_traj0_gap.append(
                    torch.linalg.norm(p_hat - t0).item()
                )

            # --- Motion and Axis evaluation (for all samples) ---
            # Ablation arms (2026-08-15 spec) may lack the axis and/or type
            # heads entirely: skip collection rather than scoring a zeros
            # placeholder, so absent-head metrics stay absent instead of
            # polluting the means.
            has_axis = motion_pred is not None or twist_decoded is not None
            # A type head trained with motion_type_weight = 0 (the 2D-only
            # arms: labels are reserved for eval and the head is
            # unsupervised) has meaningless logits — reporting its
            # "accuracy" reads as a finding when there is none (user
            # decision 2026-08-21). Same convention as absent heads: the
            # type metrics stay absent, and MA (which needs a type call)
            # with them. The head itself still runs (L_pp gate, row-wise
            # axis selection).
            _type_supervised = (
                getattr(self.loss_params, "motion_type_weight", 0.5) > 0.0
            )
            has_type = (
                motion_type_logits is not None or twist_decoded is not None
            ) and _type_supervised
            self._test_has_axis_head |= has_axis
            self._test_has_type_head |= has_type
            axis_err = None
            is_axis_correct = is_type_correct = is_axis_correct_signed = False
            if has_axis:
                axis_src = (
                    motion_pred[i] if motion_pred is not None
                    else twist_decoded[1][i]
                )
                axis_err = self._axis_error_deg(axis_src, motion_gt[i]).item()
                axis_err_signed = self._axis_error_deg(
                    axis_src, motion_gt[i], signed=True
                ).item()
                self._test_axis_errors_all.append(axis_err)
                self._test_axis_errors_signed_all.append(axis_err_signed)
                if motion_type_gt[i] == 1:
                    self._test_axis_errors_signed_rot.append(axis_err_signed)
                is_axis_correct = axis_err <= self.config.test_motion_threshold_deg
                is_axis_correct_signed = (
                    axis_err_signed <= self.config.test_motion_threshold_deg
                )
            if has_type:
                is_type_correct = bool(pred_types[i] == motion_type_gt[i])
                if is_type_correct:
                    self._test_type_correct_all += 1
            if is_axis_correct and is_type_correct:
                self._test_ma_correct_all += 1
            if is_axis_correct_signed and is_type_correct:
                self._test_ma_signed_correct_all += 1

            # --- Twist-head evaluation (unified screw parameterisation) ---
            if twist_decoded is not None:
                tw_rev, tw_dir, tw_point = twist_decoded
                tw_axis_err = self._axis_error_deg(tw_dir[i], motion_gt[i]).item()
                self._test_twist_axis_errors_all.append(tw_axis_err)

                tw_type_correct = bool(tw_rev[i].item()) == bool(
                    motion_type_gt[i].item() == 1
                )
                if tw_type_correct:
                    self._test_twist_type_correct_all += 1
                if tw_type_correct and (
                    tw_axis_err <= self.config.test_motion_threshold_deg
                ):
                    self._test_twist_ma_correct_all += 1

                gt_dir = F.normalize(motion_gt[i].float(), p=2, dim=-1, eps=1e-8)
                if (tw_dir[i].to(gt_dir.device) * gt_dir).sum() > 0:
                    self._test_twist_dir_correct_all += 1

                if (
                    camera_params_in_batch
                    and motion_type_gt[i] == 1
                    and motion_origin_3d_gt is not None
                ):
                    line_dist = point_to_line_distance(
                        motion_origin_3d_gt[i].to(tw_point.device).float(),
                        tw_point[i],
                        tw_dir[i],
                    ).item()
                    self._test_twist_line_dist_rotational.append(line_dist)

            # --- Smoothness: mean second-difference magnitude (gen-19) ---
            if (
                outputs.trajectory_pred is not None
                and trajectory_gt is not None
            ):
                _tp = outputs.trajectory_pred[i].detach().float()
                _tg = trajectory_gt[i].to(_tp.device).float()
                self._test_traj_rough_pred.append(
                    float((_tp[2:] - 2 * _tp[1:-1] + _tp[:-2]).norm(dim=-1).mean())
                )
                self._test_traj_rough_gt.append(
                    float((_tg[2:] - 2 * _tg[1:-1] + _tg[:-2]).norm(dim=-1).mean())
                )

            # Net sweep direction of the predicted 3D trajectory vs GT
            # (cos > 0 = the curve heads the right way).
            if (
                outputs.trajectory_pred is not None
                and trajectory_gt is not None
            ):
                pred_net = outputs.trajectory_pred[i, -1] - outputs.trajectory_pred[i, 0]
                gt_net = (
                    trajectory_gt[i, -1] - trajectory_gt[i, 0]
                ).to(pred_net.device).float()
                denom = pred_net.norm() * gt_net.norm()
                if denom > 1e-8:
                    self._test_traj_dir_cos.append(
                        ((pred_net * gt_net).sum() / denom).item()
                    )

            # --- 2D reprojection decomposition (mirrors tools/diag_proj2d.py
            # and the TrajectoryProjectionLoss chain: point_uv lifted with
            # the INPUT depth, + relative trajectory, projected) ---
            if (
                len(_trajectory_2d_extras) >= 2
                and outputs.point_uv is not None
                and outputs.trajectory_pred is not None
                and camera_params_in_batch
                and intrinsic_matrix is not None
            ):
                _dev = outputs.point_uv.device
                _K_n = normalized_intrinsics(
                    intrinsic_matrix[i:i + 1].float().to(_dev),
                    _img_size[i:i + 1].float().to(_dev),
                )
                _uv = outputs.point_uv[i:i + 1].detach().float()
                _grid = (_uv * 2.0 - 1.0).view(1, 1, 1, 2)
                _z = F.grid_sample(
                    depth[i:i + 1].float().to(_dev), _grid, align_corners=False
                ).view(1)
                if float(_z) > 1e-3:
                    _anchor = backproject_points(_K_n, _uv, _z)
                    _curve = (
                        _anchor.unsqueeze(1)
                        + outputs.trajectory_pred[i:i + 1].detach().float()
                    )
                    _proj = project_points(_K_n, _curve)[0]
                    _track = (
                        _trajectory_2d_extras[0][i].float()
                        / _img_size[i].float().clamp(min=1.0)
                    ).to(_dev)
                    _tvalid = _trajectory_2d_extras[1][i].bool().to(_dev)
                    _v = _tvalid & (_curve[0, :, 2] > 0.05)
                    if int(_v.sum()) >= 2 and int(_tvalid.sum()) >= 1:
                        _p, _t = _proj[_v], _track[_v]
                        self._test_proj2d_err.append(
                            float((_p - _t).norm(dim=-1).mean())
                        )
                        self._test_proj2d_shape.append(
                            float(((_p - _p[0]) - (_t - _t[0])).norm(dim=-1).mean())
                        )
                        self._test_proj2d_anchor.append(
                            float((_proj[0] - _track[_tvalid][0]).norm())
                        )

            # --- Original evaluation for IoU-matched samples ---
            if iou_val > self.config.test_iou_threshold:
                self._test_num_matched += 1

                if axis_err is not None:
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
                            # Split arm (point_prediction_3d): no 2D point
                            # head — skip the heatmap panel, keep the rest.
                            point_pred_prob = (
                                torch.sigmoid(point_pred_logits)
                                if point_pred_logits is not None
                                else None
                            )
                            mask_pred_prob = torch.sigmoid(mask_pred_logits)

                            self._save_sf3d_test_debug_visualizations(
                                full_image_path=full_image_path,
                                point_pred_prob_tensor=(
                                    point_pred_prob[i].detach().cpu()
                                    if point_pred_prob is not None
                                    else None
                                ),
                                mask_pred_prob_tensor=mask_pred_prob[i].detach().cpu(),
                                motion_pred=(
                                    motion_pred[i] if motion_pred is not None
                                    else torch.zeros(3)
                                ).detach().cpu(),
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
        all_axis_errors_signed_all = self.all_gather(
            torch.tensor(self._test_axis_errors_signed_all, device=self.device)
        )
        all_axis_errors_signed_rot = self.all_gather(
            torch.tensor(self._test_axis_errors_signed_rot, device=self.device)
        )
        total_ma_signed_correct_all = self.all_gather(
            torch.tensor(
                self._test_ma_signed_correct_all, dtype=torch.long,
                device=self.device,
            )
        ).sum()
        all_origin_errors_rotational = self.all_gather(
            torch.tensor(self._test_origin_errors_rotational_all, device=self.device)
        )
        all_twist_axis_errors = self.all_gather(
            torch.tensor(self._test_twist_axis_errors_all, device=self.device)
        )
        all_twist_line_dists = self.all_gather(
            torch.tensor(self._test_twist_line_dist_rotational, device=self.device)
        )
        total_twist_type_correct = self.all_gather(
            torch.tensor(
                self._test_twist_type_correct_all, dtype=torch.long, device=self.device
            )
        ).sum()
        total_twist_ma_correct = self.all_gather(
            torch.tensor(
                self._test_twist_ma_correct_all, dtype=torch.long, device=self.device
            )
        ).sum()

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
        # Ablation arms (2026-08-15 spec): heads that are absent in this arm
        # produce no metrics at all — skip their logs and prints entirely
        # (never a 0.0 placeholder), so absent-head metric columns stay
        # absent in the CSV. Head presence is config-determined, hence
        # identical on every rank.
        has_axis = self._test_has_axis_head
        has_type = self._test_has_type_head

        p_det = (
            100.0 * total_num_matched / total_predictions
            if total_predictions > 0
            else 0.0
        )

        self.log("test/p_det", p_det, prog_bar=True, logger=True, sync_dist=True)
        self.log("test/mean_iou", mean_iou, prog_bar=False, logger=True, sync_dist=True)
        self.log(
            "test/mean_point_error",
            mean_point_error,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        if has_type:
            pass_rate_m = (
                100.0 * total_type_correct_all / total_predictions
                if total_predictions > 0
                else 0.0
            )
            self.log(
                "test/pass_rate_m", pass_rate_m, prog_bar=True, logger=True,
                sync_dist=True,
            )
        if has_axis and has_type:
            pass_rate_ma = (
                100.0 * total_ma_correct_all / total_predictions
                if total_predictions > 0
                else 0.0
            )
            self.log(
                "test/pass_rate_ma", pass_rate_ma, prog_bar=True, logger=True,
                sync_dist=True,
            )
        if has_axis:
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
            # Sign-aware axis columns (2026-08-18). A "flip" is a signed
            # error > 90 deg: the predicted axis points into the wrong
            # hemisphere, i.e. the opening direction is reversed even when
            # the unsigned line error is tiny.
            err_adir_signed_all = (
                float(all_axis_errors_signed_all.mean().item())
                if all_axis_errors_signed_all.numel() > 0
                else 0.0
            )
            flip_rate = (
                100.0 * float(
                    (all_axis_errors_signed_all > 90.0).float().mean().item()
                )
                if all_axis_errors_signed_all.numel() > 0
                else 0.0
            )
            flip_rate_rot = (
                100.0 * float(
                    (all_axis_errors_signed_rot > 90.0).float().mean().item()
                )
                if all_axis_errors_signed_rot.numel() > 0
                else 0.0
            )
            self.log(
                "test/err_adir_signed_all_deg", err_adir_signed_all,
                prog_bar=False, logger=True, sync_dist=True,
            )
            self.log(
                "test/axis_flip_rate", flip_rate,
                prog_bar=False, logger=True, sync_dist=True,
            )
            self.log(
                "test/axis_flip_rate_rot", flip_rate_rot,
                prog_bar=False, logger=True, sync_dist=True,
            )
            if has_type:
                pass_rate_ma_signed = (
                    100.0 * total_ma_signed_correct_all / total_predictions
                    if total_predictions > 0
                    else 0.0
                )
                self.log(
                    "test/pass_rate_ma_signed", pass_rate_ma_signed,
                    prog_bar=False, logger=True, sync_dist=True,
                )
        if all_origin_errors_rotational.numel() > 0:
            mean_origin_error_m = float(all_origin_errors_rotational.mean().item())
            self.log(
                "test/mean_origin_error_m",
                mean_origin_error_m,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        # --- Twist-head metrics (only when the head ran) ---
        twist_ran = all_twist_axis_errors.numel() > 0
        if twist_ran:
            twist_axis_err = float(all_twist_axis_errors.mean().item())
            twist_line_dist = (
                float(all_twist_line_dists.mean().item())
                if all_twist_line_dists.numel() > 0
                else 0.0
            )
            if total_predictions > 0:
                twist_type_acc = 100.0 * total_twist_type_correct / total_predictions
                twist_pass_rate_ma = 100.0 * total_twist_ma_correct / total_predictions
            else:
                twist_type_acc, twist_pass_rate_ma = 0.0, 0.0
            self.log("test/twist_axis_err_deg", twist_axis_err, logger=True, sync_dist=True)
            self.log("test/twist_type_acc", twist_type_acc, logger=True, sync_dist=True)
            self.log(
                "test/twist_pass_rate_ma", twist_pass_rate_ma, logger=True, sync_dist=True
            )
            self.log(
                "test/twist_axis_line_dist_m", twist_line_dist, logger=True, sync_dist=True
            )
            if total_predictions > 0:
                twist_dir_acc = (
                    100.0 * self._test_twist_dir_correct_all / total_predictions
                )
                self.log("test/twist_dir_acc", twist_dir_acc, logger=True, sync_dist=True)

        proj2d_stats = None
        if self._test_proj2d_err:
            proj2d_stats = {}
            for name, vals in (
                ("test/traj_proj2d_err", self._test_proj2d_err),
                ("test/traj_proj2d_anchor", self._test_proj2d_anchor),
                ("test/traj_proj2d_shape", self._test_proj2d_shape),
            ):
                g = self.all_gather(torch.tensor(vals, device=self.device))
                m = float(g.mean().item()) if g.numel() > 0 else 0.0
                self.log(name, m, logger=True, sync_dist=False)
                proj2d_stats[name] = m

        rough_stats = None
        if self._test_traj_rough_pred:
            rough_stats = {}
            for name, vals in (
                ("test/traj_rough_pred", self._test_traj_rough_pred),
                ("test/traj_rough_gt", self._test_traj_rough_gt),
            ):
                g = self.all_gather(torch.tensor(vals, device=self.device))
                m = float(g.mean().item()) if g.numel() > 0 else 0.0
                self.log(name, m, logger=True, sync_dist=False)
                rough_stats[name] = m

        if self._test_traj_dir_cos:
            traj_dir_cos = float(
                torch.tensor(self._test_traj_dir_cos).mean().item()
            )
            traj_dir_acc = 100.0 * float(
                (torch.tensor(self._test_traj_dir_cos) > 0).float().mean().item()
            )
            self.log("test/traj_dir_cos", traj_dir_cos, logger=True, sync_dist=True)
            self.log("test/traj_dir_acc", traj_dir_acc, logger=True, sync_dist=True)

        # --- Split-arm (gen-6) metrics (0.0 when the arm lacks the heads) ---
        split_stats = {}
        for name, values in (
            ("test/point_err_3d_m", self._test_point3d_errors),
            ("test/origin_err_m", self._test_origin_err_m),
            ("test/origin_line_err_m", self._test_origin_line_err_m),
            ("test/radius_err_m", self._test_radius_err_m),
            ("test/point_traj0_gap_m", self._test_point_traj0_gap),
        ):
            gathered = self.all_gather(torch.tensor(values, device=self.device))
            mean = float(gathered.mean().item()) if gathered.numel() > 0 else 0.0
            if gathered.numel() > 0:
                self.log(name, mean, on_epoch=True, logger=True, sync_dist=False)
            split_stats[name] = (mean, gathered.numel())

        if self.trainer.is_global_zero:
            print("\n--- SF3D Test Results ---")
            print(f"Total Samples: {total_predictions}")
            print(f"Mean IoU: {mean_iou:.4f}")
            print(f"PDet (IoU > {self.config.test_iou_threshold:.2f}): {p_det:.2f}%")
            if has_type:
                print(f"M Pass Rate (Motion Type): {pass_rate_m:.2f}%")
            if has_axis and has_type:
                print(f"MA Pass Rate (Motion Type + Axis): {pass_rate_ma:.2f}%")
            print(f"\n--- Detailed Stats ---")
            print(f"Mean Point Error (L2): {mean_point_error:.4f}")
            if has_axis:
                print(f"Mean Axis Error (all): {err_adir_all:.2f} degrees")
                print(f"Mean Axis Error (matched): {err_adir_matched:.2f} degrees")
                print(
                    f"Mean SIGNED Axis Error (all): {err_adir_signed_all:.2f} "
                    "degrees"
                )
                print(
                    f"Axis Flip Rate (signed > 90 deg): {flip_rate:.2f}% all, "
                    f"{flip_rate_rot:.2f}% rotational"
                )
                if has_type:
                    print(
                        "MA Pass Rate (Type + SIGNED Axis): "
                        f"{pass_rate_ma_signed:.2f}%"
                    )
            if all_origin_errors_rotational.numel() > 0:
                print(
                    f"Mean Origin Error (m, for rotational): {mean_origin_error_m:.4f}"
                )
            if proj2d_stats is not None:
                print(
                    "2D Reprojection (uv): total "
                    f"{proj2d_stats['test/traj_proj2d_err']:.4f}  anchor "
                    f"{proj2d_stats['test/traj_proj2d_anchor']:.4f}  shape "
                    f"{proj2d_stats['test/traj_proj2d_shape']:.4f}"
                )
            if rough_stats is not None:
                print(
                    "Trajectory Roughness (m, 2nd-diff): pred "
                    f"{rough_stats['test/traj_rough_pred']:.5f}  gt "
                    f"{rough_stats['test/traj_rough_gt']:.5f}"
                )
            if twist_ran:
                print(f"\n--- Twist Head (unified screw parameterisation) ---")
                print(f"Type Accuracy (|omega| > 0.5): {twist_type_acc:.2f}%")
                print(f"MA Pass Rate (type + axis): {twist_pass_rate_ma:.2f}%")
                print(f"Mean Axis Error: {twist_axis_err:.2f} degrees")
                if all_twist_line_dists.numel() > 0:
                    print(
                        "Mean GT-origin -> predicted-axis-line distance "
                        f"(m, rotational): {twist_line_dist:.4f}"
                    )
            if split_stats["test/point_err_3d_m"][1] > 0:
                print(f"\n--- Split Heads (gen-6) ---")
                print(
                    f"Mean 3D Point Error (m): "
                    f"{split_stats['test/point_err_3d_m'][0]:.4f}"
                )
            if split_stats["test/origin_err_m"][1] > 0:
                print(
                    f"Mean Origin Error vs q* (m, rotational): "
                    f"{split_stats['test/origin_err_m'][0]:.4f}"
                )
                print(
                    "Mean predicted-origin -> GT-axis-line distance "
                    f"(m, rotational): {split_stats['test/origin_line_err_m'][0]:.4f}"
                )
            if split_stats["test/radius_err_m"][1] > 0:
                print(
                    f"Mean Radius Error (m, rotational): "
                    f"{split_stats['test/radius_err_m'][0]:.4f}"
                )
            if split_stats["test/point_traj0_gap_m"][1] > 0:
                print(
                    "Mean lifted-point vs traj[0] gap (m, gen-7): "
                    f"{split_stats['test/point_traj0_gap_m'][0]:.4f}"
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
        self._test_axis_errors_signed_all.clear()
        self._test_axis_errors_signed_rot.clear()
        self._test_type_correct_all = 0
        self._test_ma_correct_all = 0
        self._test_ma_signed_correct_all = 0
        self._test_proj2d_err.clear()
        self._test_proj2d_anchor.clear()
        self._test_proj2d_shape.clear()
        self._test_traj_rough_pred.clear()
        self._test_traj_rough_gt.clear()
        self._test_has_axis_head = False
        self._test_has_type_head = False
        self._test_origin_errors_rotational_all.clear()
        self._test_twist_axis_errors_all.clear()
        self._test_twist_type_correct_all = 0
        self._test_twist_ma_correct_all = 0
        self._test_twist_line_dist_rotational.clear()
        self._test_point3d_errors.clear()
        self._test_origin_err_m.clear()
        self._test_origin_line_err_m.clear()
        self._test_radius_err_m.clear()
        self._test_point_traj0_gap.clear()


if __name__ == "__main__":
    LightningCLI(SF3DTrainingModule, SF3DDataModule, save_config_callback=None)
