import argparse
import os
import torch
import numpy as np
import cv2
from loguru import logger

import utils.config as config
from utils.dataset import RefDataset
from model import build_segmenter


def get_parser():
    parser = argparse.ArgumentParser(
        description="Test Referring Expression Segmentation Pipeline"
    )
    parser.add_argument(
        "--config", required=True, type=str, help="config file"
    )
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


def main():
    # Load configuration
    args = get_parser()
    logger.info("Configuration loaded successfully")
    
    # Create model
    logger.info("Building model...")
    model, _ = build_segmenter(args)
    model.eval()
    logger.info("Model built successfully")
    
    # Prepare single sample for testing
    logger.info("Loading test sample...")
    test_data = RefDataset(
        lmdb_dir=args.val_lmdb,
        mask_dir=args.mask_root,
        dataset=args.dataset,
        split=args.val_split,
        mode="val",
        input_size=args.input_size,
        word_length=args.word_len,
    )
    
    # Get a single sample
    if len(test_data) == 0:
        logger.error("No samples found in dataset!")
        return
    
    logger.info(f"Dataset loaded with {len(test_data)} samples")
    
    # Take the first sample
    sample_idx = 0
    img, text, param = test_data[sample_idx]
    
    # Add batch dimension
    img = img.unsqueeze(0)
    text = text.unsqueeze(0)
    
    # Move to the same device as model (assuming CUDA is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img = img.to(device)
    text = text.to(device)
    
    logger.info(f"Running inference on device: {device}")
    
    # Run inference
    with torch.no_grad():
        pred = model(img, text)
        pred = torch.sigmoid(pred)
    
    # Resize if needed
    if pred.shape[-2:] != img.shape[-2:]:
        pred = torch.nn.functional.interpolate(
            pred, size=img.shape[-2:], mode="bicubic", align_corners=True
        ).squeeze(1)
    
    # Process result
    pred = pred.cpu().squeeze().numpy()
    
    # Get original size and transformation matrix
    h, w = param["ori_size"]
    mat = param["inverse"]
    
    # Apply affine transform
    pred = cv2.warpAffine(
        pred, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0.0,)
    )
    
    # Threshold
    pred_mask = (pred > 0.35).astype(np.uint8) * 255
    
    # Load ground truth mask
    mask_path = param["mask_dir"]
    if os.path.exists(mask_path):
        gt_mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        
        # Calculate IoU
        pred_binary = pred > 0.35
        gt_binary = gt_mask > 0
        
        intersection = (pred_binary & gt_binary).sum()
        union = (pred_binary | gt_binary).sum()
        iou = intersection / (union + 1e-6)
        
        logger.info(f"IoU: {iou:.4f}")
    else:
        logger.warning(f"Ground truth mask not found: {mask_path}")
    
    # Save results
    os.makedirs("pipeline_test_results", exist_ok=True)
    cv2.imwrite("pipeline_test_results/pred_mask.png", pred_mask)
    
    # Save visualization
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:, :, 1] = pred_mask  # Green channel for prediction
    
    if os.path.exists(mask_path):
        vis[:, :, 2] = gt_mask  # Red channel for ground truth
    
    cv2.imwrite("pipeline_test_results/visualization.png", vis)
    
    # Save text query
    with open("pipeline_test_results/query.txt", "w") as f:
        f.write(f"Text query used for this test sample (index {sample_idx})")
    
    logger.info("Pipeline test completed successfully")
    logger.info("Results saved to pipeline_test_results/")


if __name__ == "__main__":
    main() 