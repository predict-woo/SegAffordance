import os
import argparse
import warnings
import typing
import pandas as pd
import textwrap

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
    description: str,
    gt_bbox: torch.Tensor,
    original_size: tuple,
    pred_threshold: float,
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
    pred_mask_binary = (pred_mask_prob_resized > pred_threshold).astype(np.uint8)

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

    # Create a copy for bbox visualization
    vis_binary_bbox = vis_binary.copy()

    # Draw GT bbox (Green)
    gt_bbox_xywh = gt_bbox.numpy()
    x, y, w_box, h_box = map(int, gt_bbox_xywh)
    cv2.rectangle(vis_binary_bbox, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

    # Draw Predicted bbox (Red)
    pred_bbox_xyxy = mask_to_bbox(
        torch.from_numpy(pred_mask_binary)
    ).numpy()  # Use the already resized binary mask
    x1, y1, x2, y2 = map(int, pred_bbox_xyxy)
    cv2.rectangle(vis_binary_bbox, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --- Combine and Save ---
    combined_vis = np.hstack((vis_prob, vis_binary_bbox))
    # Add description text at the bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    line_type = 1
    text_margin = 10

    # Wrap text to fit the image width
    wrapped_text = textwrap.wrap(f"Desc: {description}", width=100)
    y0 = combined_vis.shape[0] - (len(wrapped_text) * 20) - text_margin

    for i, line in enumerate(wrapped_text):
        y = y0 + i * 20
        cv2.putText(
            combined_vis,
            line,
            (text_margin, y),
            font,
            font_scale,
            font_color,
            line_type,
        )

    cv2.imwrite(output_path, combined_vis)


def mask_to_bbox(mask: torch.Tensor) -> torch.Tensor:
    """Computes a bounding box from a binary mask.
    mask: (H, W) tensor
    Returns a (4,) tensor [x_min, y_min, x_max, y_max]
    """
    if mask.sum() == 0:
        return torch.zeros(4, device=mask.device)
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    y_min, y_max = torch.where(rows)[0][[0, -1]]
    x_min, x_max = torch.where(cols)[0][[0, -1]]
    return torch.tensor([x_min, y_min, x_max, y_max], device=mask.device)


def calculate_bbox_iou(
    box_pred_xyxy: torch.Tensor, box_gt_xywh: torch.Tensor
) -> torch.Tensor:
    """
    Calculates IoU for a predicted XYXY box and a GT XYWH box.
    """
    # Convert GT from XYWH to XYXY
    box_gt_xyxy = torch.cat([box_gt_xywh[:2], box_gt_xywh[:2] + box_gt_xywh[2:]])

    # Determine the coordinates of the intersection rectangle
    xA = torch.max(box_pred_xyxy[0], box_gt_xyxy[0])
    yA = torch.max(box_pred_xyxy[1], box_gt_xyxy[1])
    xB = torch.min(box_pred_xyxy[2], box_gt_xyxy[2])
    yB = torch.min(box_pred_xyxy[3], box_gt_xyxy[3])

    interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

    boxPredArea = (box_pred_xyxy[2] - box_pred_xyxy[0]) * (
        box_pred_xyxy[3] - box_pred_xyxy[1]
    )
    boxGtArea = box_gt_xywh[2] * box_gt_xywh[3]  # w*h

    iou = interArea / (boxPredArea + boxGtArea - interArea + 1e-7)
    return iou


def calculate_iou(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, pred_threshold: float = 0.5
) -> torch.Tensor:
    """Calculates IoU for a single prediction and ground truth mask."""

    pred_mask = (pred_mask > pred_threshold).float()
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
        "--pred_threshold",
        type=float,
        default=0.5,
        help="Prediction probability threshold for binarizing masks.",
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

    # --- Load Descriptions for Testing ---
    print("📝 Loading descriptions from './test-full.csv'")
    try:
        descriptions_df = pd.read_csv("./test-full.csv")
        # Ensure all descriptions are strings and drop any empty/NaN ones
        descriptions_df.dropna(subset=["description"], inplace=True)
        descriptions_df["description"] = descriptions_df["description"].astype(str)
        descriptions_map = (
            descriptions_df.groupby("image_filename")["description"]
            .apply(list)
            .to_dict()
        )
        print(
            f"✅ Loaded {len(descriptions_df)} descriptions for {len(descriptions_map)} images."
        )
    except FileNotFoundError:
        print(
            "❌ Error: './test-full.csv' not found. Please provide the descriptions file."
        )
        return

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
            return_filename=True,
        )
    except FileNotFoundError:
        print(f"❌ Error: Annotation file for '{args.dataset_key}' not found.")
        print(
            f"Ensure that 'data_path' in your config is correct and the dataset exists."
        )
        return

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time
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
    num_all_correct = 0
    sample_idx = 0

    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc=f"Evaluating on {args.dataset_key}")
        for batch in pbar:
            (
                img,
                depth,
                _,  # gt_description, which we ignore
                mask_gt,
                bbox_gt,
                point_gt_norm,
                motion_gt,
                motion_type_gt,
                original_size,
                filename_tuple,
            ) = batch

            img_filename = filename_tuple[0]
            candidate_descriptions = descriptions_map.get(img_filename)

            if not candidate_descriptions:
                warnings.warn(
                    f"No descriptions found for image {img_filename} in test-full.csv, skipping."
                )
                continue

            img = img.to(device)
            depth = depth.to(device)
            mask_gt = mask_gt.to(device)
            point_gt_norm = point_gt_norm.to(device)
            motion_gt = motion_gt.to(device)
            motion_type_gt = motion_type_gt.to(device)

            # --- Run model for all candidate descriptions ---
            num_candidates = len(candidate_descriptions)
            img_batch = img.repeat(num_candidates, 1, 1, 1)
            depth_batch = depth.repeat(num_candidates, 1, 1, 1)
            tokenized_words = tokenize(
                candidate_descriptions, cfg.word_len, truncate=True
            ).to(device)

            (
                mask_pred_logits,
                _,
                coords_hat,
                motion_pred,
                motion_type_logits,
                _,
                _,
            ) = model(img_batch, depth_batch, tokenized_words, None, None, None)

            mask_pred_prob = torch.sigmoid(mask_pred_logits)
            mask_pred_upsampled = F.interpolate(
                mask_pred_prob, size=mask_gt.shape[-2:], mode="bilinear"
            )
            pred_types = torch.argmax(motion_type_logits, dim=1)

            # --- Find best prediction based on IoU ---
            all_ious = torch.zeros(num_candidates, device=device)
            for i in range(num_candidates):
                pred_bbox = mask_to_bbox(
                    (mask_pred_upsampled[i].squeeze() > args.pred_threshold).float()
                )
                all_ious[i] = calculate_bbox_iou(pred_bbox.cpu(), bbox_gt[0].cpu())

            best_idx = torch.argmax(all_ious)
            max_iou = all_ious[best_idx].item()

            # print("max_iou", max_iou)
            # print("all_ious", all_ious)

            ious.append(max_iou)

            best_description = candidate_descriptions[best_idx]
            best_mask_pred = mask_pred_upsampled[best_idx]
            best_coords_hat = coords_hat[best_idx]
            best_motion_pred = motion_pred[best_idx]
            best_pred_type = pred_types[best_idx]

            # --- Calculate metrics for the best prediction ---

            # Optional: Save debug visualization
            if args.visualize_debug:
                output_path = os.path.join(
                    debug_output_dir,
                    f"sample_{sample_idx:04d}_iou{max_iou:.2f}.png",
                )
                create_debug_visualization(
                    img[0],
                    mask_gt[0],
                    best_mask_pred,
                    output_path,
                    best_description,
                    bbox_gt[0],
                    (original_size[0, 0].item(), original_size[0, 1].item()),
                    args.pred_threshold,
                )

            # Calculate and store point error
            point_err = torch.linalg.norm(best_coords_hat - point_gt_norm[0])
            point_errors.append(point_err.item())

            # Check for a match and calculate axis-related metrics
            if max_iou > args.iou_threshold:
                num_matched_predictions += 1
                axis_error = calculate_axis_error(best_motion_pred, motion_gt[0])
                axis_errors_matched.append(axis_error.item())

                is_axis_correct = axis_error <= args.motion_threshold
                if is_axis_correct:
                    correct_axis_predictions += 1

                # 4. Check motion type for matched (high IoU) predictions
                is_type_correct = best_pred_type == motion_type_gt[0]
                if is_type_correct:
                    num_correct_type_in_matched += 1

                if is_axis_correct and is_type_correct:
                    num_all_correct += 1

            sample_idx += 1

            if sample_idx > 0:
                pass_rate = (num_matched_predictions / sample_idx) * 100
                pbar.set_postfix_str(f"Pass Rate: {pass_rate:.2f}%")

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
        map_all = (num_all_correct / total_predictions) * 100
    else:
        map_type = 0
        map_all = 0

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
    print(
        f"mAP_All (IoU > {args.iou_threshold:.2f} & Correct Type & Axis): {map_all:.2f}%"
    )
    print(f"Precision_Type (Correct Type for Matched): {precision_type:.2f}%")
    print("--------------------------\n")


if __name__ == "__main__":
    main()
