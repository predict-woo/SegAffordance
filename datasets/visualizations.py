import tyro
from pathlib import Path
import cv2
import random
import textwrap
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, cast

from scenefun3d import SF3DDataset
from opdreal import OPDRealDataset

from torchvision import transforms
from typing import Callable


def get_default_transforms(
    image_size: Tuple[int, int] = (256, 256)  # (height, width)
) -> Tuple[Callable, Callable, Callable]:
    """Returns a default set of transforms for RGB images, masks, and depth maps."""
    rgb_transform = transforms.Compose(
        [
            transforms.Resize(image_size),  # (h,w) for Resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.NEAREST,  # (h,w) for Resize
            ),
            transforms.ToTensor(),
            lambda x: (x > 0.5).float(),  # Ensure binary mask {0., 1.}
        ]
    )
    depth_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ]
    )
    return rgb_transform, mask_transform, depth_transform


rgb_transform, mask_transform, depth_transform = get_default_transforms((512, 512))


def draw_overlays(image, mask_tensor, origin_2d_norm, motion_dir_3d, description):
    original_height, original_width, _ = image.shape

    # --- Draw Mask ---
    mask_np = mask_tensor.squeeze().numpy()
    contours, _ = cv2.findContours(
        (mask_np * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Green

    # --- Draw Motion Origin ---
    origin_x = int(origin_2d_norm[0].item() * original_width)
    origin_y = int(origin_2d_norm[1].item() * original_height)
    cv2.circle(image, (origin_x, origin_y), 8, (255, 0, 0), -1)  # Blue dot

    # --- Draw Motion Arrow ---
    motion_dir_3d_np = motion_dir_3d.numpy()
    v_dir_2d = np.array([motion_dir_3d_np[0], motion_dir_3d_np[1]])
    norm = np.linalg.norm(v_dir_2d)
    if norm > 1e-6:
        v_dir_2d_norm = v_dir_2d / norm
        arrow_length_pixels = 50
        target_x = origin_x + int(v_dir_2d_norm[0] * arrow_length_pixels)
        target_y = origin_y + int(v_dir_2d_norm[1] * arrow_length_pixels)
        cv2.arrowedLine(
            image,
            (origin_x, origin_y),
            (target_x, target_y),
            (0, 0, 255),  # Red
            2,
            tipLength=0.3,
        )

    # --- Add Text Labels ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (0, 255, 255)  # Yellow
    line_type = 1
    wrapped_desc = textwrap.wrap(description, width=50)

    text_x, text_y = origin_x + 15, origin_y
    for j, line in enumerate(wrapped_desc):
        y_pos = text_y + (j * 15)
        cv2.putText(
            image,
            line,
            (text_x, y_pos),
            font,
            font_scale,
            font_color,
            line_type,
        )
    return image


def visualize_dataset(
    dataset: Dataset,
    num_samples: int,
    output_dir: Path,
    dataset_name: str,
):
    """
    Loads a dataset, samples a few items, and generates debug visualizations.
    """
    output_dir.mkdir(exist_ok=True)
    print(f"Saving debug images to {output_dir}")

    if len(cast(list, dataset)) == 0:
        print("Dataset is empty. No items to visualize.")
        return

    indices = list(range(len(cast(list, dataset))))
    random.shuffle(indices)

    print(
        f"Sampling {min(num_samples, len(cast(list, dataset)))} items for visualization..."
    )
    for i in indices[:num_samples]:
        sample = dataset[i]

        # --- Process RGB Image ---
        rgb_image_tensor = sample[0]
        img_np = rgb_image_tensor.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        if img_np.shape[2] == 3:
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
        debug_image_rgb = (img_np * 255).astype(np.uint8)
        debug_image_rgb = cv2.cvtColor(debug_image_rgb, cv2.COLOR_RGB2BGR)

        # Handle different dataset return formats
        if len(sample) == 6:  # OPDReal with depth
            (
                _,
                depth_image_tensor,
                description,
                mask_tensor,
                origin_2d_norm,
                motion_dir_3d,
            ) = sample

            # --- Process Depth Image ---
            depth_np = depth_image_tensor.squeeze().numpy()
            depth_vis = cv2.normalize(
                depth_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            debug_image_depth = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # --- Draw Overlays ---
            debug_image_rgb_with_overlay = draw_overlays(
                debug_image_rgb.copy(),
                mask_tensor,
                origin_2d_norm,
                motion_dir_3d,
                description,
            )
            debug_image_depth_with_overlay = draw_overlays(
                debug_image_depth.copy(),
                mask_tensor,
                origin_2d_norm,
                motion_dir_3d,
                description,
            )

            # --- Concatenate and Save ---
            final_image = np.hstack(
                (debug_image_rgb_with_overlay, debug_image_depth_with_overlay)
            )

        else:  # SF3D with no depth
            (
                _,
                description,
                mask_tensor,
                origin_2d_norm,
                motion_dir_3d,
            ) = sample
            final_image = draw_overlays(
                debug_image_rgb.copy(),
                mask_tensor,
                origin_2d_norm,
                motion_dir_3d,
                description,
            )

        # --- Save Image ---
        debug_filename = f"debug_{dataset_name}_{i}.png"
        debug_image_path = output_dir / debug_filename
        cv2.imwrite(str(debug_image_path), final_image)

    print(f"Finished. Debug images are saved in {output_dir.resolve()}")


def main(
    dataset_name: str,
    data_path: Path,
    num_samples: int = 10,
    output_dir: Path = Path("debug_visualization"),
    sf3d_lmdb_path: Optional[Path] = None,
    opd_dataset_key: str = "MotionNet_train",
):
    """
    Generic visualization script for datasets.

    Args:
        dataset_name (str): The name of the dataset to visualize.
                            Choices: 'scenefun3d', 'opdreal'.
        data_path (Path): Path to the root of the dataset.
        num_samples (int): Number of random samples to visualize.
        output_dir (Path): Directory to save the debug images.
        sf3d_lmdb_path (Path, optional): Path to the LMDB file for SceneFun3D.
                                         Defaults to data_path / "data.lmdb".
        opd_dataset_key (str): The specific dataset key for OPDReal
                               (e.g., 'MotionNet_train', 'opd_c_real_test').
    """
    # remove all pngs in output_dir
    print(f"Removing all pngs in {output_dir}")
    for file in output_dir.glob("*.png"):
        file.unlink()

    print(f"Loading {dataset_name} dataset...")
    if dataset_name == "scenefun3d":
        lmdb_path = sf3d_lmdb_path if sf3d_lmdb_path else data_path
        dataset = SF3DDataset(
            lmdb_data_root=str(lmdb_path),
            rgb_transform=rgb_transform,
            mask_transform=mask_transform,
        )
    elif dataset_name == "opdreal":
        dataset = OPDRealDataset(
            data_path=str(data_path),
            dataset_key=opd_dataset_key,
            rgb_transform=rgb_transform,
            mask_transform=mask_transform,
            depth_transform=depth_transform,
        )
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    print(f"Visualizing {dataset_name} dataset...")
    visualize_dataset(dataset, num_samples, output_dir, dataset_name)


if __name__ == "__main__":
    tyro.cli(main)

# python datasets/visualizations.py --dataset-name opdreal --data-path /cluster/project/cvg/students/andrye/dataset/MotionDataset_h5_real --num-samples 10 --output-dir debug
# python datasets/visualizations.py --dataset-name scenefun3d --data-path /cluster/project/cvg/students/andrye/sf3d --num-samples 10 --output-dir debug
