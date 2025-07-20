import os
import argparse
import warnings
import typing

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2

import utils.config as config_loader
from datasets.opdreal import OPDRealDataset, get_default_transforms
from train_OPDReal import OPDRealTrainingModule
from utils.dataset import tokenize

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_debug_visualization(
    image_tensor: torch.Tensor,
    gt_mask_tensor: torch.Tensor,
    pred_mask_prob_tensor: torch.Tensor,
    output_path: str,
):
    """
    Creates a debug visualization comparing a GT mask and a predicted mask,
    both before and after thresholding, and saves it to a file.

    - GT mask is shown in Green.
    - Predicted mask is shown in Red.
    - Intersection (GT & Pred) is shown in Yellow.
    """
    # 1. Convert image tensor to displayable BGR format
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    # 2. Prepare masks by resizing to image dimensions
    gt_mask_np = gt_mask_tensor.cpu().numpy().squeeze()
    pred_mask_prob_np = pred_mask_prob_tensor.cpu().numpy().squeeze()

    gt_mask_resized = cv2.resize(gt_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    pred_mask_prob_resized = cv2.resize(
        pred_mask_prob_np, (w, h), interpolation=cv2.INTER_LINEAR
    )

    gt_mask_binary = (gt_mask_resized > 0.5).astype(np.uint8)

    # --- Visualization 1: Raw Sigmoid Probabilities vs. GT ---
    overlay_prob = np.zeros_like(img_bgr, dtype=np.uint8)
    overlay_prob[gt_mask_binary == 1, 1] = 255  # Green for GT
    red_channel_prob = (pred_mask_prob_resized * 255).astype(np.uint8)
    overlay_prob[:, :, 2] = red_channel_prob  # Red for prediction probability

    vis_prob = cv2.addWeighted(img_bgr, 0.6, overlay_prob, 0.4, 0)
    cv2.putText(
        vis_prob,
        "Left: Sigmoid Output (Red)",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # --- Visualization 2: Thresholded Masks (0.5) ---
    pred_mask_binary = (pred_mask_prob_resized > 0.5).astype(np.uint8)

    overlay_binary = np.zeros_like(img_bgr, dtype=np.uint8)
    overlay_binary[gt_mask_binary == 1, 1] = 255  # Green for GT
    overlay_binary[pred_mask_binary == 1, 2] = 255  # Red for predicted

    vis_binary = cv2.addWeighted(img_bgr, 0.6, overlay_binary, 0.4, 0)
    cv2.putText(
        vis_binary,
        "Right: Thresholded (Yellow=Overlap)",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # --- Combine and Save ---
    combined_vis = np.hstack((vis_prob, vis_binary))
    cv2.imwrite(output_path, combined_vis)


def calculate_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """Calculates IoU for a single prediction and ground truth mask."""

    pred_mask = (pred_mask > 0.7).float()
    gt_mask = (gt_mask > 0.5).float()

    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    return intersection / (union + 1e-7)


def calculate_axis_error(
    pred_axis: torch.Tensor, gt_axis: torch.Tensor
) -> torch.Tensor:
    """Calculates the axis error in degrees, ignoring direction."""
    # Ensure inputs are single vectors
    pred_axis = pred_axis.squeeze()
    gt_axis = gt_axis.squeeze()

    gt_axis_norm = F.normalize(gt_axis, p=2, dim=-1)
    pred_axis_norm = F.normalize(pred_axis, p=2, dim=-1)

    cos_sim = torch.dot(gt_axis_norm, pred_axis_norm)
    cos_sim = torch.abs(cos_sim)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    axis_error_rad = torch.acos(cos_sim)
    axis_error_deg = axis_error_rad * 180.0 / np.pi
    return axis_error_deg


def get_test_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for the test script."""
    parser = argparse.ArgumentParser(description="Test CRIS model on OPDReal data")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--dataset_key",
        type=str,
        required=True,
        help="Dataset key for testing (e.g., 'opd_c_real_test').",
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        default=10.0,
        help="Axis error threshold in degrees for P_ADir.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold to consider a prediction a match.",
    )
    parser.add_argument(
        "--visualize_debug",
        action="store_true",
        help="Enable to save debug visualizations of mask predictions.",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Override settings in config file.",
    )
    return parser


def main():
    """Main evaluation function."""
    parser = get_test_parser()
    args = parser.parse_args()

    # --- Config and Device Setup ---
    cfg = config_loader.load_cfg_from_cfg_file(args.config)
    if args.opts:
        cfg = config_loader.merge_cfg_from_list(cfg, args.opts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ℹ️ Using device: {device}")

    # --- Model Loading ---
    print(f"📦 Loading model from checkpoint: {args.checkpoint}")
    model = OPDRealTrainingModule.load_from_checkpoint(args.checkpoint, cfg=cfg)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")

    # --- Dataset and DataLoader Setup ---
    print(f"📚 Setting up dataset for key: {args.dataset_key}")
    rgb_transform, mask_transform, depth_transform = get_default_transforms(
        image_size=(cfg.input_size[0], cfg.input_size[1])
    )
    try:
        test_dataset = OPDRealDataset(
            data_path=cfg.data_path,
            dataset_key=args.dataset_key,
            rgb_transform=rgb_transform,
            mask_transform=mask_transform,
            depth_transform=depth_transform,
        )
    except FileNotFoundError:
        print(f"❌ Error: Annotation file for '{args.dataset_key}' not found.")
        print(
            f"Ensure that 'data_path' in your config is correct and the dataset exists."
        )
        return

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size_val,
        shuffle=False,
        num_workers=cfg.num_workers_val,
        pin_memory=True,
        drop_last=False,
    )
    print(f"✅ Dataset loaded. Found {len(test_dataset)} samples.")

    if args.visualize_debug:
        debug_output_dir = "debug_visualizations"
        os.makedirs(debug_output_dir, exist_ok=True)
        print(f"📸 Debug visualizations will be saved to: {debug_output_dir}")

    # --- Evaluation Loop ---
    axis_errors_matched = []
    point_errors = []
    ious = []
    correct_axis_predictions = 0
    num_matched_predictions = 0
    num_correct_type_in_matched = 0
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Evaluating on {args.dataset_key}"):
            (
                img,
                depth,
                word_str_list,
                mask_gt,
                point_gt_norm,
                motion_gt,
                motion_type_gt,
            ) = batch

            img, depth, mask_gt, point_gt_norm, motion_gt, motion_type_gt = (
                img.to(device),
                depth.to(device),
                mask_gt.to(device),
                point_gt_norm.to(device),
                motion_gt.to(device),
                motion_type_gt.to(device),
            )
            tokenized_words = tokenize(
                list(word_str_list), cfg.word_len, truncate=True
            ).to(device)

            # Perform inference
            (
                mask_pred_logits,
                _,
                coords_hat,
                motion_pred,
                motion_type_logits,
                _,
                _,
            ) = model(img, depth, tokenized_words, None, None, None)

            mask_pred_prob = torch.sigmoid(mask_pred_logits)
            mask_pred_upsampled = F.interpolate(
                mask_pred_prob, size=mask_gt.shape[-2:], mode="bilinear"
            )
            pred_types = torch.argmax(motion_type_logits, dim=1)

            for i in range(img.size(0)):
                # 1. Calculate and store IoU
                iou = calculate_iou(mask_pred_upsampled[i], mask_gt[i])
                ious.append(iou.item())

                # Optional: Save debug visualization
                if args.visualize_debug:
                    output_path = os.path.join(
                        debug_output_dir, f"sample_{sample_idx:04d}_iou{iou:.2f}.png"
                    )
                    create_debug_visualization(
                        img[i], mask_gt[i], mask_pred_upsampled[i], output_path
                    )

                # 2. Calculate and store point error
                point_err = torch.linalg.norm(coords_hat[i] - point_gt_norm[i])
                point_errors.append(point_err.item())

                # 3. Check for a match and calculate axis-related metrics
                if iou > args.iou_threshold:
                    num_matched_predictions += 1
                    axis_error = calculate_axis_error(motion_pred[i], motion_gt[i])
                    axis_errors_matched.append(axis_error.item())

                    if axis_error <= args.motion_threshold:
                        correct_axis_predictions += 1

                    # 4. Check motion type for matched (high IoU) predictions
                    if pred_types[i] == motion_type_gt[i]:
                        num_correct_type_in_matched += 1

                sample_idx += 1

    # --- Report Metrics ---
    total_predictions = len(test_dataset)
    mean_iou = np.mean(ious) if ious else 0
    mean_point_error = np.mean(point_errors) if point_errors else 0
    err_adir = np.mean(axis_errors_matched) if axis_errors_matched else 0

    if num_matched_predictions > 0:
        p_adir = (correct_axis_predictions / num_matched_predictions) * 100
        precision_type = (num_correct_type_in_matched / num_matched_predictions) * 100
    else:
        p_adir = 0
        precision_type = 0

    if total_predictions > 0:
        map_type = (num_correct_type_in_matched / total_predictions) * 100
    else:
        map_type = 0

    print("\n--- Evaluation Results ---")
    print(f"Total Samples: {total_predictions}")
    print(f"IoU Threshold for Match: {args.iou_threshold:.2f}")
    print(f"Motion Threshold for P_ADir: {args.motion_threshold:.2f} degrees")
    print("-" * 26)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Point Error (L2 Norm): {mean_point_error:.4f}")
    print(
        f"Matched Predictions (IoU > {args.iou_threshold:.2f}): {num_matched_predictions}"
    )
    print(f"ERR_ADir (Mean Axis Error): {err_adir:.2f} degrees")
    print(f"P_ADir (Precision for Axis): {p_adir:.2f}%")
    print(f"mAP_Type (IoU > {args.iou_threshold:.2f} & Correct Type): {map_type:.2f}%")
    print(f"Precision_Type (Correct Type for Matched): {precision_type:.2f}%")
    print("--------------------------\n")


if __name__ == "__main__":
    main()
