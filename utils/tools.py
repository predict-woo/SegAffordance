import torch
import torch.nn as nn
import numpy as np
import cv2
import typing


class MotionVAELoss(nn.Module):
    def __init__(self, beta=0.01, cosine_eps=1e-4):
        super().__init__()
        self.beta = beta
        self.cosine_eps = cosine_eps
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.cosine_eps)

    def forward(self, recon_x, x, mu, log_var):
        # Assert that all target vectors are non-zero, as expected.
        target_norm = torch.linalg.norm(x, dim=1)
        assert torch.all(
            target_norm > self.cosine_eps
        ), "Zero-magnitude ground truth motion vector found, which is not expected."

        # Cosine similarity loss: 1 - (cos_sim)^2
        # This is low when vectors are parallel/anti-parallel, high when orthogonal.
        cos_sim = self.cos_sim(recon_x, x)
        recon_loss = (1.0 - torch.pow(cos_sim, 2)).mean()

        # KLD loss, averaged over batch
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kld_loss = kld_loss.mean()

        total_loss = recon_loss + self.beta * kld_loss
        return total_loss, recon_loss, kld_loss


def make_gaussian_map(points_norm, map_h, map_w, sigma, device):
    """
    Generates a batch of 2D Gaussian heatmaps.
    Args:
        points_norm (torch.Tensor): Batch of normalized points (B, 2) in [0, 1] range.
        map_h (int): Target heatmap height.
        map_w (int): Target heatmap width.
        sigma (float): Standard deviation of the Gaussian in pixels.
        device (str or torch.device): Device to create tensors on.
    Returns:
        torch.Tensor: Batch of Gaussian heatmaps (B, 1, map_h, map_w).
    """
    B = points_norm.size(0)
    if B == 0:
        return torch.empty((0, 1, map_h, map_w), device=device)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(map_h, dtype=torch.float32, device=device),
        torch.arange(map_w, dtype=torch.float32, device=device),
        indexing="ij",
    )

    x_grid = x_coords.unsqueeze(0).expand(B, -1, -1)
    y_grid = y_coords.unsqueeze(0).expand(B, -1, -1)

    center_x = (points_norm[:, 0] * (map_w - 1)).view(B, 1, 1)
    center_y = (points_norm[:, 1] * (map_h - 1)).view(B, 1, 1)

    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    heatmaps = torch.exp(-dist_sq / (2 * sigma**2))
    return heatmaps.unsqueeze(1)


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth_dice: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth_dice = smooth_dice
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target_mask):
        """
        pred_logits: (B, 1, H, W)
        target_mask: (B, 1, H, W), values in [0,1]
        """
        bce_val = self.bce_loss(pred_logits, target_mask)

        pred_sigmoid = torch.sigmoid(pred_logits)
        intersection = (pred_sigmoid * target_mask).sum(dim=(1, 2, 3))
        union_pred = pred_sigmoid.sum(dim=(1, 2, 3))
        union_target = target_mask.sum(dim=(1, 2, 3))

        dice_score = (2.0 * intersection + self.smooth_dice) / (
            union_pred + union_target + self.smooth_dice
        )
        dice_val = 1.0 - dice_score

        return (self.bce_weight * bce_val) + (self.dice_weight * dice_val.mean())


def create_composite_visualization(
    image_tensor: torch.Tensor,
    text_description: str,
    mask_pred_sigmoid: torch.Tensor,
    point_gt_norm: torch.Tensor,
    point_pred_heatmap: torch.Tensor,
    motion_gt: torch.Tensor,
    motion_pred: torch.Tensor,
    motion_type_gt: typing.Optional[torch.Tensor],
    motion_type_pred_logits: typing.Optional[torch.Tensor],
    trajectory_gt: typing.Optional[torch.Tensor] = None,
    trajectory_pred: typing.Optional[torch.Tensor] = None,
    camera_intrinsic: typing.Optional[torch.Tensor] = None,
    original_image_size: typing.Optional[torch.Tensor] = None,
    depth_tensor: typing.Optional[torch.Tensor] = None,
    vector_scale: int = 50,
):
    """
    Creates a composite visualization image with mask, vectors, and points.
    
    Note: trajectory_pred is in RELATIVE coordinates (first point at origin).
          To visualize it, we back-project the predicted 2D point to 3D using depth
          and camera intrinsics, then add this origin to the relative trajectory.
    """
    # 1. Convert tensor image to OpenCV format
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    # 2. Overlay predicted mask as a transparent red overlay
    mask_np = mask_pred_sigmoid.cpu().numpy().squeeze()
    mask_resized = cv2.resize(mask_np, (w, h))
    red_overlay = np.zeros_like(img_bgr)
    red_overlay[:, :, 2] = 255  # Red in BGR
    binary_mask = (mask_resized > 0.5).astype(np.uint8)
    masked_red = cv2.bitwise_and(red_overlay, red_overlay, mask=binary_mask)
    composite_img = cv2.addWeighted(img_bgr, 1.0, masked_red, 0.5, 0)

    # 3. Draw motion arrows (matching dataset visualization style)
    origin_x = int(point_gt_norm[0] * w)
    origin_y = int(point_gt_norm[1] * h)
    
    # Get motion type for proper visualization
    motion_type_map = {0: "translation", 1: "rotation"}
    gt_type_str = motion_type_map.get(int(motion_type_gt.item()), "translation") if motion_type_gt is not None else "translation"
    pred_type_idx = torch.argmax(motion_type_pred_logits).item() if motion_type_pred_logits is not None else 0
    pred_type_str = motion_type_map.get(int(pred_type_idx), "translation")
    
    # Scale for 3D motion visualization (matching dataset style)
    arrow_scale_m = 0.2
    motion_gt_3d = motion_gt.cpu().numpy()
    motion_pred_3d = motion_pred.cpu().numpy()
    
    # Project motion arrows using camera intrinsics if available
    if camera_intrinsic is not None and original_image_size is not None:
        K = camera_intrinsic.cpu().numpy()
        original_w, original_h = original_image_size.cpu().numpy()
        scale_w = w / original_w if original_w > 0 else 1.0
        scale_h = h / original_h if original_h > 0 else 1.0
        
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_w
        K_scaled[1, 1] *= scale_h
        K_scaled[0, 2] *= scale_w
        K_scaled[1, 2] *= scale_h
        
        def project_camera_to_image(points_camera, intrinsic_matrix, width, height):
            """Project 3D points in camera coordinates to 2D image coordinates."""
            points_camera = np.asarray(points_camera)
            if points_camera.ndim == 1:
                points_camera = points_camera.reshape(1, -1)
            
            points_homo = intrinsic_matrix @ points_camera.T  # (3, N)
            z = points_homo[2, :]
            valid_mask = z > 0
            u = points_homo[0, :] / (z + 1e-8)
            v = points_homo[1, :] / (z + 1e-8)
            in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height) & valid_mask
            
            result = np.zeros((points_camera.shape[0], 3))
            result[:, 0] = v  # row (y)
            result[:, 1] = u  # col (x)
            result[:, 2] = in_bounds.astype(float)
            return result
        
        # Draw GT motion arrow
        if gt_type_str == "translation":
            # Cyan arrow for translation motion direction
            end_pt_3d = motion_gt_3d * arrow_scale_m
            origin_proj = project_camera_to_image(np.array([0, 0, 0]), K_scaled, w, h)
            end_proj = project_camera_to_image(end_pt_3d, K_scaled, w, h)
            if origin_proj[0, 2] == 1 and end_proj[0, 2] == 1:
                start = (int(origin_proj[0, 1]), int(origin_proj[0, 0]))
                end = (int(end_proj[0, 1]), int(end_proj[0, 0]))
                cv2.arrowedLine(composite_img, start, end, (0, 255, 255), 3, tipLength=0.1)
        elif gt_type_str == "rotation":
            # Yellow double arrow for rotation axis direction
            end1 = motion_gt_3d * arrow_scale_m
            origin_proj = project_camera_to_image(np.array([0, 0, 0]), K_scaled, w, h)
            end1_proj = project_camera_to_image(end1, K_scaled, w, h)
            if origin_proj[0, 2] == 1 and end1_proj[0, 2] == 1:
                start = (int(origin_proj[0, 1]), int(origin_proj[0, 0]))
                end = (int(end1_proj[0, 1]), int(end1_proj[0, 0]))
                cv2.arrowedLine(composite_img, start, end, (255, 255, 0), 3, tipLength=0.1)
                cv2.arrowedLine(composite_img, end, start, (255, 255, 0), 3, tipLength=0.1)
        
        # Draw predicted motion arrow
        if pred_type_str == "translation":
            # Magenta arrow for predicted translation motion direction
            end_pt_3d = motion_pred_3d * arrow_scale_m
            origin_proj = project_camera_to_image(np.array([0, 0, 0]), K_scaled, w, h)
            end_proj = project_camera_to_image(end_pt_3d, K_scaled, w, h)
            if origin_proj[0, 2] == 1 and end_proj[0, 2] == 1:
                start = (int(origin_proj[0, 1]), int(origin_proj[0, 0]))
                end = (int(end_proj[0, 1]), int(end_proj[0, 0]))
                cv2.arrowedLine(composite_img, start, end, (255, 0, 255), 3, tipLength=0.1)
        elif pred_type_str == "rotation":
            # White double arrow for predicted rotation axis direction
            end1 = motion_pred_3d * arrow_scale_m
            origin_proj = project_camera_to_image(np.array([0, 0, 0]), K_scaled, w, h)
            end1_proj = project_camera_to_image(end1, K_scaled, w, h)
            if origin_proj[0, 2] == 1 and end1_proj[0, 2] == 1:
                start = (int(origin_proj[0, 1]), int(origin_proj[0, 0]))
                end = (int(end1_proj[0, 1]), int(end1_proj[0, 0]))
                cv2.arrowedLine(composite_img, start, end, (255, 255, 255), 3, tipLength=0.1)
                cv2.arrowedLine(composite_img, end, start, (255, 255, 255), 3, tipLength=0.1)
    else:
        # Fallback to 2D visualization if no camera intrinsics
        motion_gt_2d = motion_gt_3d[:2]
        motion_pred_2d = motion_pred_3d[:2]
        
        norm_gt = np.linalg.norm(motion_gt_2d)
        if norm_gt > 1e-6:
            motion_gt_2d /= norm_gt
        
        norm_pred = np.linalg.norm(motion_pred_2d)
        if norm_pred > 1e-6:
            motion_pred_2d /= norm_pred
        
        gt_target_x = origin_x + int(motion_gt_2d[0] * vector_scale)
        gt_target_y = origin_y + int(motion_gt_2d[1] * vector_scale)
        cv2.arrowedLine(
            composite_img,
            (origin_x, origin_y),
            (gt_target_x, gt_target_y),
            (0, 255, 0),
            2,
            tipLength=0.3,
        )
        
        pred_target_x = origin_x + int(motion_pred_2d[0] * vector_scale)
        pred_target_y = origin_y + int(motion_pred_2d[1] * vector_scale)
        cv2.arrowedLine(
            composite_img,
            (origin_x, origin_y),
            (pred_target_x, pred_target_y),
            (0, 0, 255),
            2,
            tipLength=0.3,
        )

    # 4. Plot interaction points (matching dataset visualization style)
    # GT interaction point (green dot, larger)
    gt_point_px = (int(point_gt_norm[0] * w), int(point_gt_norm[1] * h))
    cv2.circle(composite_img, gt_point_px, 5, (0, 255, 0), -1)  # Green dot
    
    # Predicted interaction point (white dot, larger)
    heatmap_np = point_pred_heatmap.cpu().numpy().squeeze()
    map_h, map_w = heatmap_np.shape
    pred_y_map, pred_x_map = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    pred_point_px = (int(pred_x_map * w / map_w), int(pred_y_map * h / map_h))
    cv2.circle(composite_img, pred_point_px, 5, (255, 255, 255), -1)  # White dot

    # 6. Draw trajectory points if available (matching dataset visualization style)
    if (
        trajectory_gt is not None
        and trajectory_pred is not None
        and camera_intrinsic is not None
        and original_image_size is not None
    ):
        K = camera_intrinsic.cpu().numpy()
        
        # Scale intrinsics to match visualization image size
        original_w, original_h = original_image_size.cpu().numpy()
        scale_w = w / original_w if original_w > 0 else 1.0
        scale_h = h / original_h if original_h > 0 else 1.0
        
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_w  # fx
        K_scaled[1, 1] *= scale_h  # fy
        K_scaled[0, 2] *= scale_w  # cx
        K_scaled[1, 2] *= scale_h  # cy
        
        def project_camera_to_image(points_camera, intrinsic_matrix, width, height):
            """Project 3D points in camera coordinates to 2D image coordinates."""
            points_camera = np.asarray(points_camera)
            if points_camera.ndim == 1:
                points_camera = points_camera.reshape(1, -1)
            
            points_homo = intrinsic_matrix @ points_camera.T  # (3, N)
            z = points_homo[2, :]
            valid_mask = z > 0
            u = points_homo[0, :] / (z + 1e-8)
            v = points_homo[1, :] / (z + 1e-8)
            in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height) & valid_mask
            
            result = np.zeros((points_camera.shape[0], 3))
            result[:, 0] = v  # row (y)
            result[:, 1] = u  # col (x)
            result[:, 2] = in_bounds.astype(float)
            return result
        
        # Project and draw GT trajectory (blue dots, matching dataset style)
        # GT trajectory is in absolute 3D camera coordinates
        traj_gt_np = trajectory_gt.cpu().numpy()
        traj_map_gt = project_camera_to_image(traj_gt_np, K_scaled, w, h)
        vis_mask_gt = traj_map_gt[:, 2] == 1
        visible_traj_points_gt = traj_map_gt[vis_mask_gt, :2]
        
        for pt in visible_traj_points_gt:
            cv2.circle(composite_img, (int(pt[1]), int(pt[0])), 2, (255, 0, 0), -1)  # Blue dots
        
        # For predicted trajectory (relative), we need to add the 3D origin back
        # Back-project predicted 2D point to 3D using depth and intrinsics
        traj_pred_np = trajectory_pred.cpu().numpy()  # Relative trajectory
        
        if depth_tensor is not None:
            # Get predicted 2D point from heatmap
            pred_u_norm = pred_x_map / map_w  # Normalized x
            pred_v_norm = pred_y_map / map_h  # Normalized y
            
            # Sample depth at predicted location
            depth_np = depth_tensor.cpu().numpy().squeeze()  # (D_h, D_w)
            D_h, D_w = depth_np.shape
            u_depth = int(np.clip(pred_u_norm * (D_w - 1), 0, D_w - 1))
            v_depth = int(np.clip(pred_v_norm * (D_h - 1), 0, D_h - 1))
            
            # Sample depth with small patch for robustness
            patch_size = 5
            half_patch = patch_size // 2
            u_start = max(0, u_depth - half_patch)
            u_end = min(D_w, u_depth + half_patch + 1)
            v_start = max(0, v_depth - half_patch)
            v_end = min(D_h, v_depth + half_patch + 1)
            depth_patch = depth_np[v_start:v_end, u_start:u_end]
            valid_depths = depth_patch[depth_patch > 0]
            z_depth = float(valid_depths.mean()) if len(valid_depths) > 0 else float(depth_np[v_depth, u_depth])
            
            if z_depth > 1e-6:
                # Back-project to 3D using original intrinsics
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                
                # Convert normalized coords to original image pixel coords
                u_orig = pred_u_norm * original_w
                v_orig = pred_v_norm * original_h
                
                # Back-project: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
                x_cam = (u_orig - cx) * z_depth / fx
                y_cam = (v_orig - cy) * z_depth / fy
                
                origin_3d = np.array([x_cam, y_cam, z_depth])
                
                # Convert relative trajectory to absolute by adding origin
                traj_pred_absolute = traj_pred_np + origin_3d
            else:
                # If depth is invalid, use GT first point as fallback
                traj_pred_absolute = traj_pred_np + traj_gt_np[0]
        else:
            # No depth available, use GT first point as origin for visualization
            traj_pred_absolute = traj_pred_np + traj_gt_np[0]
        
        # Project and draw predicted trajectory (cyan dots)
        traj_map_pred = project_camera_to_image(traj_pred_absolute, K_scaled, w, h)
        vis_mask_pred = traj_map_pred[:, 2] == 1
        visible_traj_points_pred = traj_map_pred[vis_mask_pred, :2]
        
        for pt in visible_traj_points_pred:
            cv2.circle(composite_img, (int(pt[1]), int(pt[0])), 2, (255, 255, 0), -1)  # Cyan dots

    # 5. Overlay text description and legend (matching dataset visualization style)
    motion_type_map = {0: "translation", 1: "rotation"}
    gt_type_str = motion_type_map.get(int(motion_type_gt.item()), "unknown") if motion_type_gt is not None else "unknown"
    pred_type_idx = torch.argmax(motion_type_pred_logits).item() if motion_type_pred_logits is not None else 0
    pred_type_str = motion_type_map.get(int(pred_type_idx), "unknown")
    
    # Main description
    text_lines = [f"{gt_type_str}: {text_description[:50]}"]
    text_lines.append(f"GT: {gt_type_str}, Pred: {pred_type_str}")
    
    # Add trajectory info if available
    if trajectory_gt is not None and trajectory_pred is not None:
        traj_gt_count = len(trajectory_gt) if hasattr(trajectory_gt, '__len__') else 0
        traj_pred_count = len(trajectory_pred) if hasattr(trajectory_pred, '__len__') else 0
        text_lines.append(f"Trajectory points: GT={traj_gt_count}, Pred={traj_pred_count}")
    
    # Draw main text
    y0, dy = 15, 15
    for i, line in enumerate(text_lines):
        y = y0 + i * dy
        cv2.putText(
            composite_img,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    

    return cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
