import argparse
import os
import torch
import warnings
import numpy as np
from PIL import Image, ImageOps

# Local imports from the project
from train_SF3D_better import SF3DTrainingModule
from datasets.scenefun3d import get_default_transforms
from utils.dataset import tokenize

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


def _save_custom_debug_visualizations(
    full_image_path: str,
    point_pred_prob_tensor: torch.Tensor,
    mask_pred_prob_tensor: torch.Tensor,
    motion_pred: torch.Tensor,  # 3d vector
    description: str,
    output_dir: str,
    sample_name: str,
):
    """
    Saves visualizations for custom inference without ground truth.
    Adapted from _save_sf3d_test_debug_visualizations.
    """
    import cv2  # Imported locally to avoid dependency if not used

    os.makedirs(output_dir, exist_ok=True)

    try:
        with Image.open(full_image_path) as img_raw:
            # Apply EXIF transpose to correct orientation from camera metadata
            img = ImageOps.exif_transpose(img_raw)
            img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except FileNotFoundError:
        print(f"Warning: Could not find image for visualization at {full_image_path}")
        return

    h, w, _ = img_bgr.shape

    # --- Common drawing components ---
    def apply_geo_annotations(vis_image, pred_px, pred_py, motion_pred_np):
        cv2.circle(
            vis_image, (pred_px, pred_py), radius=5, color=(255, 255, 255), thickness=-1
        )
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
            12,
        )
        return vis_image

    def apply_text_annotations(vis_image, desc, img_h, img_w):
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
        
    # --- Get interaction point from argmax of point heatmap ---
    point_pred_prob_np = point_pred_prob_tensor.float().numpy().squeeze()
    point_heatmap_resized_for_argmax = cv2.resize(
        point_pred_prob_np, (w, h), interpolation=cv2.INTER_LINEAR
    )
    (pred_py, pred_px) = np.unravel_index(
        np.argmax(point_heatmap_resized_for_argmax),
        point_heatmap_resized_for_argmax.shape,
    )

    # --- Visualization 1: Point Heatmap ---
    point_heatmap_inverted = 1 - point_heatmap_resized_for_argmax
    point_heatmap_colored = cv2.applyColorMap(
        (point_heatmap_inverted * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    vis_image_point = cv2.addWeighted(img_bgr.copy(), 0.6, point_heatmap_colored, 0.4, 0)
    vis_image_point = apply_geo_annotations(
        vis_image_point, pred_px, pred_py, motion_pred.numpy()
    )
    vis_image_point = apply_text_annotations(vis_image_point, description, h, w)
    out_path_point = os.path.join(output_dir, f"{sample_name}_point.png")
    cv2.imwrite(out_path_point, vis_image_point)

    # --- Visualization 2: Mask Heatmap ---
    mask_prob_np = mask_pred_prob_tensor.float().numpy().squeeze()
    k = 20  # Steepness factor for sigmoid sharpening
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
    vis_image_mask = apply_text_annotations(vis_image_mask, description, h, w)
    out_path_mask = os.path.join(output_dir, f"{sample_name}_mask.png")
    cv2.imwrite(out_path_mask, vis_image_mask)


def main(args):
    """
    Runs single-sample inference on a custom image and depth map.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.ckpt_path}")
    model = SF3DTrainingModule.load_from_checkpoint(
        args.ckpt_path, map_location=device, strict=False
    )
    model.eval()
    model.to(device)

    # --- Load and process custom inputs ---
    print(f"Loading RGB image from {args.image_path}")
    try:
        rgb_image_pil_raw = Image.open(args.image_path).convert("RGB")
        # Apply EXIF transpose to correct orientation from camera metadata
        rgb_image_pil = ImageOps.exif_transpose(rgb_image_pil_raw)
    except FileNotFoundError:
        print(f"Error: RGB image not found at {args.image_path}")
        return

    print(f"Loading depth map from {args.depth_path}")
    try:
        with np.load(args.depth_path) as data:
            if "depth" not in data:
                print(f"Error: 'depth' key not found in {args.depth_path}")
                return
            depth_map_meters = data["depth"]
    except FileNotFoundError:
        print(f"Error: Depth map not found at {args.depth_path}")
        return

    # Convert numpy depth map to a PIL Image, which is what the transform expects
    # The 'F' mode is for 32-bit floating point pixels
    depth_image_pil = Image.fromarray(depth_map_meters.astype(np.float32), mode="F")

    # --- Apply the same transforms used during training ---
    input_size = (256, 256)
    rgb_transform, _, depth_transform = get_default_transforms(image_size=input_size)
    
    img_input = rgb_transform(rgb_image_pil).unsqueeze(0).to(device)
    depth_input = depth_transform(depth_image_pil).unsqueeze(0).to(device)

    text_description = args.text_description
    print(f"Using text description: '{text_description}'")

    tokenized_text = tokenize(
        [text_description], model.model_params.word_len, truncate=True
    ).to(device)

    # --- Run Inference ---
    print("Running inference...")
    with torch.no_grad():
        (
            mask_pred_logits,
            point_pred_logits,
            _coords_hat,
            motion_pred,
            _motion_type_logits,
            _mu,
            _log_var,
        ) = model(img_input, depth_input, tokenized_text, None, None, None)

    point_pred_prob = torch.sigmoid(point_pred_logits[0])
    mask_pred_prob = torch.sigmoid(mask_pred_logits[0])
    motion_pred_sample = motion_pred[0]

    # --- Save Visualizations ---
    print(f"Saving visualizations to {args.output_dir}")
    sample_name = os.path.splitext(os.path.basename(args.image_path))[0]
    _save_custom_debug_visualizations(
        full_image_path=args.image_path,
        point_pred_prob_tensor=point_pred_prob.cpu(),
        mask_pred_prob_tensor=mask_pred_prob.cpu(),
        motion_pred=motion_pred_sample.cpu(),
        description=text_description,
        output_dir=args.output_dir,
        sample_name=sample_name,
    )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a custom image using the SegAffordance model."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Full path to the custom RGB image file.",
    )
    parser.add_argument(
        "--depth_path",
        type=str,
        required=True,
        help="Path to the corresponding depth map in .npz format (must contain a 'depth' key).",
    )
    parser.add_argument(
        "--text_description",
        type=str,
        required=True,
        help="Custom text description to guide the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom_inference_output",
        help="Directory to save visualization results.",
    )
    args = parser.parse_args()
    main(args)
