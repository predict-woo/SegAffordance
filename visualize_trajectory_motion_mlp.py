#!/usr/bin/env python3
"""
Script to visualize the trajectory-to-motion MLP predictions.
Loads a checkpoint, extracts the MLP, and visualizes trajectories and predicted motion vectors on images.
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from datasets.scenefun3d import SF3DDataset, get_default_transforms
from utils.dataset import tokenize


class TrajectoryToMotionMLP(nn.Module):
    """Standalone MLP for trajectory -> motion prediction."""
    def __init__(self, trajectory_points=20, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(trajectory_points * 3 + 2, hidden_dim),  # +2 for motion type one-hot
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 3),  # Output motion vector
        )
    
    def forward(self, trajectory, motion_type_onehot):
        """
        Args:
            trajectory: (B, N, 3) or (B, N*3) - trajectory points
            motion_type_onehot: (B, 2) - one-hot encoded motion type
        Returns:
            motion_vector: (B, 3) - predicted motion vector
        """
        if trajectory.dim() == 3:
            trajectory = trajectory.view(trajectory.size(0), -1)  # Flatten
        mlp_input = torch.cat([trajectory, motion_type_onehot], dim=1)
        return self.mlp(mlp_input)


def load_mlp_from_checkpoint(checkpoint_path):
    """Load the trajectory_to_motion_mlp from a PyTorch Lightning checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to extract the MLP state dict
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Find all keys related to trajectory_to_motion_mlp
    mlp_keys = {k: v for k, v in state_dict.items() if 'trajectory_to_motion_mlp' in k}
    
    if not mlp_keys:
        print("❌ No trajectory_to_motion_mlp found in checkpoint!")
        print("Available keys (first 20):")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            print(f"  {key}")
        return None
    
    # Create the MLP and load weights
    mlp = TrajectoryToMotionMLP()
    
    # Strip the prefix from keys (e.g., "trajectory_to_motion_mlp.0.weight" -> "mlp.0.weight")
    mlp_state_dict = {}
    for key, value in mlp_keys.items():
        # Remove "trajectory_to_motion_mlp." prefix and add "mlp." prefix
        new_key = key.replace('trajectory_to_motion_mlp.', 'mlp.', 1)
        mlp_state_dict[new_key] = value
    
    mlp.load_state_dict(mlp_state_dict)
    mlp.eval()
    
    print(f"✅ Successfully loaded MLP with {len(mlp_keys)} parameters")
    return mlp


def project_3d_to_2d(trajectory_3d, intrinsic_matrix, origin_2d_norm, image_size, processed_img_size=(256, 256)):
    """
    Project 3D trajectory points to 2D image coordinates.
    
    Args:
        trajectory_3d: (N, 3) - 3D points in camera coordinates
        intrinsic_matrix: (3, 3) - camera intrinsics
        origin_2d_norm: (2,) - interaction point in normalized coordinates [0,1]
        image_size: (2,) - [width, height] of the original image
        processed_img_size: (width, height) of the processed image
    Returns:
        trajectory_2d: (N, 2) - 2D pixel coordinates in processed image space
    """
    intrinsic = intrinsic_matrix.numpy()
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    
    original_width, original_height = image_size[0].item(), image_size[1].item()
    processed_width, processed_height = processed_img_size
    
    # Project 3D points to 2D in original image space
    x_cam = trajectory_3d[:, 0].numpy()
    y_cam = trajectory_3d[:, 1].numpy()
    z_cam = trajectory_3d[:, 2].numpy()
    
    # Avoid division by zero
    z_cam = np.clip(z_cam, 1e-6, None)
    
    u_orig = (fx * x_cam / z_cam) + cx
    v_orig = (fy * y_cam / z_cam) + cy
    
    # Scale to processed image size
    scale_x = processed_width / original_width
    scale_y = processed_height / original_height
    
    u = u_orig * scale_x
    v = v_orig * scale_y
    
    # Convert to pixel coordinates
    coords_2d = np.stack([u, v], axis=1)
    
    return coords_2d


def visualize_sample(image_tensor, depth_tensor, description, mask_tensor, 
                     point_gt_norm, motion_gt, motion_type_gt, trajectory_gt,
                     motion_pred_from_traj, intrinsic_matrix, img_size, output_path):
    """
    Create a visualization showing the trajectory and predicted motion vector.
    """
    # Convert tensor image to numpy
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
    
    # Overlay mask
    mask_np = mask_tensor.cpu().numpy().squeeze()
    mask_resized = cv2.resize(mask_np, (w, h))
    green_overlay = np.zeros_like(img_bgr)
    green_overlay[:, :, 1] = 255  # Green in BGR
    binary_mask = (mask_resized > 0.5).astype(np.uint8)
    masked_green = cv2.bitwise_and(green_overlay, green_overlay, mask=binary_mask)
    vis_img = cv2.addWeighted(img_bgr, 1.0, masked_green, 0.4, 0)
    
    # Project trajectory to 2D
    trajectory_2d = project_3d_to_2d(trajectory_gt.cpu(), intrinsic_matrix.cpu(), 
                                    point_gt_norm.cpu(), img_size.cpu())
    
    # Draw trajectory points
    for i, (u, v) in enumerate(trajectory_2d):
        # Color code by position in trajectory
        color_ratio = i / max(len(trajectory_2d) - 1, 1)
        color = (int(255 * (1 - color_ratio)), 0, int(255 * color_ratio))  # Blue to red
        cv2.circle(vis_img, (int(u), int(v)), radius=3, color=color, thickness=-1)
    
    # Draw GT motion vector
    origin_px = int(point_gt_norm[0] * w)
    origin_py = int(point_gt_norm[1] * h)
    
    # GT motion (green)
    motion_gt_2d = motion_gt.cpu().numpy()[:2]
    motion_gt_norm = motion_gt_2d / (np.linalg.norm(motion_gt_2d) + 1e-8)
    gt_end_x = origin_px + int(motion_gt_norm[0] * 80)
    gt_end_y = origin_py + int(motion_gt_norm[1] * 80)
    cv2.arrowedLine(vis_img, (origin_px, origin_py), (gt_end_x, gt_end_y),
                    (0, 255, 0), 3, tipLength=0.3)
    
    # Predicted motion from trajectory (red)
    motion_pred_2d = motion_pred_from_traj.cpu().numpy()[:2]
    motion_pred_norm = motion_pred_2d / (np.linalg.norm(motion_pred_2d) + 1e-8)
    pred_end_x = origin_px + int(motion_pred_norm[0] * 80)
    pred_end_y = origin_py + int(motion_pred_norm[1] * 80)
    cv2.arrowedLine(vis_img, (origin_px, origin_py), (pred_end_x, pred_end_y),
                    (0, 0, 255), 3, tipLength=0.3)
    
    # Draw interaction point
    cv2.circle(vis_img, (origin_px, origin_py), radius=5, color=(255, 255, 255), thickness=-1)
    
    # Add text
    motion_type_str = "translation" if motion_type_gt == 0 else "rotation"
    text = f"{description[:50]}\nType: {motion_type_str}"
    
    # Calculate angle error
    motion_gt_3d = motion_gt.cpu().numpy()
    motion_pred_3d = motion_pred_from_traj.cpu().numpy()
    angle_error = np.arccos(np.clip(np.dot(motion_gt_3d, motion_pred_3d) / 
                                     (np.linalg.norm(motion_gt_3d) * np.linalg.norm(motion_pred_3d) + 1e-8), 
                                     -1, 1)) * 180 / np.pi
    
    text_lines = [
        text,
        f"Angle error: {angle_error:.1f}°"
    ]
    
    y0, dy = 20, 20
    for i, line in enumerate(text_lines):
        y = y0 + i * dy
        cv2.putText(vis_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Save visualization
    cv2.imwrite(output_path, vis_img)
    print(f"✅ Saved visualization to: {output_path}")

# python visualize_trajectory_motion_mlp.py --checkpoint /cluster/work/cvg/students/andrye/SegAffordanceExperiments/experiments/SF3D_TRAJ_V3/best-epoch=8-val/loss_total=0.9004.ckpt --data_dir /cluster/work/cvg/students/andrye/sf3d_processed

def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory-to-motion MLP predictions")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file or directory containing checkpoints')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to SF3D dataset directory')
    parser.add_argument('--output_dir', type=str, default='mlp_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Specific epoch to load (if checkpoint is a directory)')
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if os.path.isdir(checkpoint_path):
        # Find checkpoint files
        pattern = os.path.join(checkpoint_path, "*.ckpt")
        ckpt_files = glob.glob(pattern)
        if not ckpt_files:
            print(f"❌ No checkpoint files found in {checkpoint_path}")
            return
        
        if args.epoch is not None:
            # Find checkpoint for specific epoch
            ckpt_files = [f for f in ckpt_files if f"epoch={args.epoch}" in f]
            if not ckpt_files:
                print(f"❌ No checkpoint found for epoch {args.epoch}")
                return
        
        # Sort by modification time and take the latest
        ckpt_files.sort(key=os.path.getmtime, reverse=True)
        checkpoint_path = ckpt_files[0]
        print(f"📂 Using checkpoint: {checkpoint_path}")
    
    # Load MLP
    mlp = load_mlp_from_checkpoint(checkpoint_path)
    if mlp is None:
        return
    
    # Setup dataset
    print(f"\n📦 Loading dataset from: {args.data_dir}")
    rgb_transform, mask_transform, depth_transform = get_default_transforms(image_size=(256, 256))
    
    dataset = SF3DDataset(
        lmdb_data_root=args.data_dir,
        rgb_transform=rgb_transform,
        mask_transform=mask_transform,
        depth_transform=depth_transform,
        image_size_for_mask_reconstruction=(256, 256),
    )
    
    print(f"✅ Dataset loaded with {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize samples
    print(f"\n🎨 Visualizing {args.num_samples} samples...")
    indices = torch.randperm(len(dataset))[:args.num_samples]
    
    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]
        
        # Unpack sample
        (img, depth, description, mask, bbox, point_gt_norm, motion_gt, motion_type_gt,
         img_size, rgb_filename, motion_origin_3d, intrinsic_matrix, trajectory_gt) = sample
        
        # Prepare inputs for MLP
        trajectory_batch = trajectory_gt.unsqueeze(0)  # Add batch dimension
        motion_type_onehot = F.one_hot(motion_type_gt.unsqueeze(0), num_classes=2).float()
        
        # Predict motion from trajectory
        with torch.no_grad():
            motion_pred_from_traj = mlp(trajectory_batch, motion_type_onehot).squeeze(0)
        
        # Create visualization
        output_path = os.path.join(args.output_dir, f"sample_{i:03d}_{os.path.basename(rgb_filename)}")
        visualize_sample(
            img, depth, description, mask, point_gt_norm, motion_gt, motion_type_gt,
            trajectory_gt, motion_pred_from_traj, intrinsic_matrix, img_size, output_path
        )
    
    print(f"\n✅ Done! Visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

