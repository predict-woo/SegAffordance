import torch
import torch.nn as nn
import numpy as np
import cv2


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
    motion_type_gt: torch.Tensor,
    motion_type_pred_logits: torch.Tensor,
    vector_scale: int = 50,
):
    """
    Creates a composite visualization image with mask, vectors, and points.
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

    # 3. Draw GT (green) and predicted (red) motion vectors
    origin_x = int(point_gt_norm[0] * w)
    origin_y = int(point_gt_norm[1] * h)

    # Normalize vectors for visualization to ensure they have the same length
    motion_gt_2d = motion_gt.cpu().numpy()[:2]
    motion_pred_2d = motion_pred.cpu().numpy()[:2]

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

    # 4. Plot GT (yellow) and predicted (magenta) interaction points as stars
    gt_point_px = (int(point_gt_norm[0] * w), int(point_gt_norm[1] * h))
    cv2.drawMarker(
        composite_img,
        gt_point_px,
        (0, 255, 255),  # Yellow
        markerType=cv2.MARKER_STAR,
        markerSize=20,
        thickness=2,
    )

    heatmap_np = point_pred_heatmap.cpu().numpy().squeeze()
    map_h, map_w = heatmap_np.shape
    pred_y_map, pred_x_map = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    pred_point_px = (int(pred_x_map * w / map_w), int(pred_y_map * h / map_h))
    cv2.drawMarker(
        composite_img,
        pred_point_px,
        (255, 0, 255),  # Magenta
        markerType=cv2.MARKER_STAR,
        markerSize=20,
        thickness=2,
    )

    # 5. Overlay text description
    motion_type_map = {0: "translation", 1: "rotation"}
    gt_type_str = motion_type_map.get(int(motion_type_gt.item()), "unknown")
    pred_type_idx = torch.argmax(motion_type_pred_logits).item()
    pred_type_str = motion_type_map.get(int(pred_type_idx), "unknown")
    text = f"{text_description}\nGT: {gt_type_str}, Pred: {pred_type_str}"

    y0, dy = 15, 15
    for i, line in enumerate(text.split("\n")):
        y = y0 + i * dy
        cv2.putText(
            composite_img,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
