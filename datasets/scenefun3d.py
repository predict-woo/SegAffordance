import os
import json
from pathlib import Path

# from torch._tensor import Tensor # Not directly used, can be removed if not needed elsewhere
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import Any, Optional, Callable, Dict, List, Tuple
import tqdm
import lmdb  # Added for LMDB
import pickle  # Added for deserializing LMDB data
import cv2
import shutil
import random

LMDB_DATASET_VERSION_COMPATIBLE = "1.0"  # For checking compatibility


class SF3DDataset(Dataset):
    """
    PyTorch Dataset for loading processed SceneFun3D items from an LMDB database.
    Each sample corresponds to a specific item visible in a specific frame.
    """

    def __init__(
        self,
        lmdb_data_root: str,  # Changed from processed_data_root
        rgb_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
        # skip_items_without_motion: bool = True, # This logic is now handled during LMDB creation
        image_size_for_mask_reconstruction: Tuple[int, int] = (
            224,
            224,
        ),  # Needed if original size not stored
    ):
        """
        Args:
            lmdb_data_root (str): Path to the root directory of the LMDB dataset
                                     (output of process_frames_with_items.py).
            rgb_transform (callable, optional): Optional transform to be applied on the RGB image.
            mask_transform (callable, optional): Optional transform for the reconstructed mask.
            depth_transform (callable, optional): Optional transform for the placeholder depth map.
            image_size_for_mask_reconstruction (Tuple[int, int]): The target size (height, width)
                                                                  for reconstructing the mask if original
                                                                  dimensions aren't available per sample.
                                                                  This should match the size your model expects
                                                                  if you're not resizing the mask later.
        """
        self.lmdb_data_root = Path(lmdb_data_root)
        self.rgb_transform = rgb_transform
        self.mask_transform = (
            mask_transform  # Retained for potential use after mask reconstruction
        )
        self.depth_transform = depth_transform
        self.image_size_for_mask_reconstruction = (
            image_size_for_mask_reconstruction  # (height, width)
        )

        # self.lmdb_path = self.lmdb_data_root / "data.lmdb"
        self.lmdb_path = Path("/dev/shm/data.lmdb")
        # The following line is useful for loading from shared memory for faster access,
        # but it's commented out to make the code more portable by default.
        # You can uncomment it and adjust the path if you copy your data.lmdb to /dev/shm.
        # self.lmdb_path = Path("/dev/shm/data.lmdb")

        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB database not found at {self.lmdb_path}")

        print(f"Opening LMDB database at {self.lmdb_path}")
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,  # Important for multi-process reading if not careful
            readahead=False,  # Usually not beneficial for many small random reads
            meminit=False,  # Only if you trust the DB file completely or manage memory manually
        )
        print(f"LMDB database opened")

        print(f"Getting item keys")
        self.item_keys = self._get_item_keys()
        print(f"Item keys: length {len(self.item_keys)}")

    def _get_item_keys(self) -> List[bytes]:
        keys = []
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                if key != b"__metadata__":
                    keys.append(key)
        return keys

    def __len__(self) -> int:
        return len(self.item_keys)

    def __getitem__(self, idx: int):
        item_key_bytes = self.item_keys[idx]
        with self.env.begin(write=False) as txn:
            item_data_bytes = txn.get(item_key_bytes)
            if item_data_bytes is None:
                raise IndexError(f"Key {item_key_bytes.decode()} not found in LMDB.")

        item_data = pickle.loads(item_data_bytes)

        # --- Load RGB Image ---
        # Path is relative to lmdb_data_root/images
        rgb_image_filename = item_data["rgb_image_path"]
        rgb_image_actual_path = self.lmdb_data_root / "images" / rgb_image_filename
        rgb_image_pil = Image.open(rgb_image_actual_path).convert("RGB")
        original_width, original_height = rgb_image_pil.size

        if self.rgb_transform:
            rgb_image_tensor = self.rgb_transform(rgb_image_pil)
        else:
            rgb_image_tensor = transforms.ToTensor()(
                rgb_image_pil
            )  # Default if no transform
        
        # --- Load Depth Image (or create placeholder) ---
        depth_image_filename = item_data.get("depth_image_path")
        depth_pil = None
        if depth_image_filename:
            depth_image_actual_path = self.lmdb_data_root / "depth" / depth_image_filename
            if depth_image_actual_path.exists():
                try:
                    # Load depth image. SceneFun3D depth is stored as 16-bit PNG in millimeters.
                    depth_pil_uint16 = Image.open(depth_image_actual_path)
                    depth_np_uint16 = np.array(depth_pil_uint16, dtype=np.uint16)
                    # Convert to float32 and scale to meters
                    depth_np_float32 = depth_np_uint16.astype(np.float32) / 1000.0
                    depth_pil = Image.fromarray(depth_np_float32, mode="F")
                except Exception as e:
                    print(f"Warning: could not load depth image {depth_image_actual_path}. Using zero depth. Error: {e}")
        
        if depth_pil is None:
            # Create a placeholder if depth image is not found or fails to load
            # zero_depth = np.zeros((original_height, original_width), dtype=np.float32)
            # depth_pil = Image.fromarray(zero_depth, mode="F")
            raise ValueError(f"Depth image not found at {depth_image_actual_path}")

        if self.depth_transform:
            depth_image_tensor = self.depth_transform(depth_pil)
        else:
            depth_image_tensor = transforms.ToTensor()(depth_pil)


        mask_np = np.zeros((original_height, original_width), dtype=np.uint8)
        mask_coords_yx = item_data.get("mask_coordinates_yx", [])
        if mask_coords_yx:  # Ensure there are coordinates
            rows, cols = zip(*mask_coords_yx)  # Separate y and x
            mask_np[np.array(rows), np.array(cols)] = 255  # Fill in the mask

        mask_pil = Image.fromarray(
            mask_np, mode="L"
        )  # Convert to PIL Image (Grayscale)
        
        # --- Bounding Box from Mask ---
        rows, cols = np.where(mask_np > 0)
        if rows.size > 0:
            x_min, x_max = cols.min(), cols.max()
            y_min, y_max = rows.min(), rows.max()
            bbox_tensor = torch.tensor([x_min, y_min, x_max - x_min, y_max - y_min], dtype=torch.float32)
        else:
            bbox_tensor = torch.zeros(4, dtype=torch.float32)


        if self.mask_transform:
            mask_tensor = self.mask_transform(mask_pil)
        else:
            # Default: resize to a fixed size (e.g., same as RGB transform if any) and convert to binary tensor
            # This should align with what get_default_transforms provided previously.
            # Let's use self.image_size_for_mask_reconstruction for consistency.
            default_mask_processing = transforms.Compose(
                [
                    transforms.Resize(
                        self.image_size_for_mask_reconstruction,
                        interpolation=transforms.InterpolationMode.NEAREST,
                    ),
                    transforms.ToTensor(),
                    lambda x: (x > 0.5).float(),  # Ensure binary {0., 1.}
                ]
            )
            mask_tensor = default_mask_processing(mask_pil)

        # --- Load Description ---
        description = item_data["description"]

        # --- Load Motion Info & Interaction Point ---
        motion_info = item_data["motion_info"]
        origin_2d_image_coord_norm = torch.zeros(2, dtype=torch.float32)  # Default
        motion_dir_3d_camera_coords = torch.zeros(3, dtype=torch.float32)
        motion_origin_3d_camera_coords = torch.zeros(3, dtype=torch.float32)
        motion_type = "trans" # default

        frame_specific_motion = motion_info.get("frame_specific_motion_data")
        if frame_specific_motion:
            origin_2d = frame_specific_motion.get(
                "motion_origin_2d_image_coords"
            )  # These are (x,y) from original file
            if origin_2d is not None and len(origin_2d) == 2:
                # Normalize origin_2d_image_coord (x, y)
                # The original code normalized by original_width, original_height
                # item_data["motion_info"] contains original, non-normalized coordinates.
                norm_x = origin_2d[0] / original_width if original_width > 0 else 0.0
                norm_y = origin_2d[1] / original_height if original_height > 0 else 0.0
                origin_2d_image_coord_norm = torch.tensor(
                    [norm_x, norm_y], dtype=torch.float32
                )

            motion_vec = frame_specific_motion.get("motion_dir_3d_camera_coords")
            if motion_vec is not None and len(motion_vec) == 3:
                motion_dir_3d_camera_coords = torch.tensor(
                    motion_vec, dtype=torch.float32
                )

            origin_3d = frame_specific_motion.get("motion_origin_3d_camera_coords")
            if origin_3d is not None and len(origin_3d) == 3:
                motion_origin_3d_camera_coords = torch.tensor(
                    origin_3d, dtype=torch.float32
                )
        
        if motion_info.get("original_motion_data"):
            motion_type = motion_info["original_motion_data"].get("motion_type", "trans")


        # Clamp the normalized coordinates to be within [0, 1]
        origin_2d_image_coord_norm = torch.clamp(origin_2d_image_coord_norm, 0.0, 1.0)
        
        motion_type_map = {"trans": 0, "translation": 0, "rot": 1, "rotation": 1}
        motion_type_tensor = torch.tensor(motion_type_map.get(motion_type, 0), dtype=torch.long)

        image_size_tensor = torch.tensor([original_width, original_height], dtype=torch.float32)

        # --- Load Camera Intrinsics ---
        intrinsics_list = item_data.get("camera_intrinsics")
        if intrinsics_list is None:
            raise ValueError(f"Camera intrinsics not found in LMDB item {item_key_bytes.decode()}. This indicates a data processing error.")
        camera_intrinsic_matrix = torch.tensor(intrinsics_list, dtype=torch.float32)

        # --- Load Trajectory Data ---
        trajectory_3d_camera_coords = item_data.get("trajectory_3d_camera_coords", [])
        if trajectory_3d_camera_coords:
            trajectory_tensor = torch.tensor(trajectory_3d_camera_coords, dtype=torch.float32)
            # Sample 20 points uniformly from the trajectory to match model output
            num_points = 20
            if len(trajectory_tensor) > 0:
                indices = torch.linspace(0, len(trajectory_tensor) - 1, num_points).long()
                trajectory_tensor = trajectory_tensor[indices]
            else:
                # If no trajectory, return zeros but with the correct shape
                trajectory_tensor = torch.zeros((num_points, 3), dtype=torch.float32)
        else:
            # Create empty trajectory if not available
            raise ValueError(f"Trajectory not found in LMDB item {item_key_bytes.decode()}")
            # trajectory_tensor = torch.zeros((20, 3), dtype=torch.float32)

        # Match return signature of OPDRealDataset + trajectory + additional fields
        return (
            rgb_image_tensor,           # Shape: (3, H, W) - RGB image tensor, normalized with ImageNet stats
            depth_image_tensor,         # Shape: (1, H, W) - Depth map in meters, single channel
            description,                # str - Text description of the interaction
            mask_tensor,                # Shape: (1, H, W) - Binary mask {0., 1.} for the object
            bbox_tensor,               # Shape: (4,) - Bounding box [x_min, y_min, width, height] in pixels
            origin_2d_image_coord_norm, # Shape: (2,) - Normalized interaction point [x_norm, y_norm] in [0,1]
            motion_dir_3d_camera_coords, # Shape: (3,) - Motion direction vector in camera coordinates
            motion_type_tensor,         # Shape: () - Motion type: 0=translation, 1=rotation
            image_size_tensor,          # Shape: (2,) - Original image dimensions [width, height] in pixels
            rgb_image_filename,         # str - Filename of the RGB image
            motion_origin_3d_camera_coords, # Shape: (3,) - Motion origin point in camera coordinates
            camera_intrinsic_matrix,    # Shape: (3, 3) - Camera intrinsic matrix
            trajectory_tensor,         # Shape: (20, 3) - 3D trajectory points in camera coordinates [x,y,z]
        )


    def __del__(self):
        if hasattr(self, 'env') and self.env:
            self.env.close()


def get_default_transforms(
    image_size: Tuple[int, int] = (256, 256)  # (height, width)
) -> Tuple[Callable, Callable, Callable]:
    """Returns a default set of transforms for RGB images and masks."""
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
                image_size,
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ]
    )
    return rgb_transform, mask_transform, depth_transform


def split_dataset_by_scene(
    dataset: "SF3DDataset",
    val_split_ratio: float,
    manual_seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Splits the SF3DDataset into training and validation subsets based on scene IDs.
    This ensures that all frames from a particular scene belong to only one split,
    preventing data leakage between the train and validation sets.
    Args:
        dataset (SF3DDataset): The full dataset instance to be split.
        val_split_ratio (float): The proportion of scenes to allocate to the validation set.
        manual_seed (int): A random seed to ensure reproducible splits.
    Returns:
        Tuple[Subset, Subset]: A tuple containing the training subset and validation subset.
    """
    print(
        f"Splitting dataset by scene with val_split_ratio={val_split_ratio} and seed={manual_seed}"
    )

    # 1. Group item indices by their scene ID.
    scene_to_indices: Dict[str, List[int]] = {}
    for i, key_bytes in enumerate(dataset.item_keys):
        # Key format is assumed to be 'scene_id/...'
        key_str = key_bytes.decode("utf-8")
        scene_id = key_str.split("/")[0]
        if scene_id not in scene_to_indices:
            scene_to_indices[scene_id] = []
        scene_to_indices[scene_id].append(i)

    # 2. Shuffle and split the list of unique scene IDs.
    unique_scene_ids = sorted(list(scene_to_indices.keys()))
    rng = random.Random(manual_seed)
    rng.shuffle(unique_scene_ids)

    num_val_scenes = int(round(len(unique_scene_ids) * val_split_ratio))
    val_scenes = set(unique_scene_ids[:num_val_scenes])
    train_scenes = set(unique_scene_ids[num_val_scenes:])

    print(f"Total scenes: {len(unique_scene_ids)}")
    print(f"Train scenes: {len(train_scenes)}, Validation scenes: {len(val_scenes)}")

    # 3. Create lists of indices for the training and validation sets.
    train_indices: List[int] = []
    val_indices: List[int] = []
    for scene_id, indices in scene_to_indices.items():
        if scene_id in val_scenes:
            val_indices.extend(indices)
        else:
            # If not in val_scenes, it must be in train_scenes
            train_indices.extend(indices)

    # 4. Create Subset wrappers for the splits.
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


if __name__ == "__main__":
    print("Testing SF3DDataset by loading random elements and generating debug visualizations...")
    
    # Set up paths
    lmdb_data_root = "/cluster/work/cvg/students/andrye/sf3d_processed"  # Contains data.lmdb and images/
    
    # Get default transforms
    rgb_transform, mask_transform, depth_transform = get_default_transforms(image_size=(256, 256))
    
    # Create dataset
    dataset = SF3DDataset(
        lmdb_data_root=lmdb_data_root,
        rgb_transform=rgb_transform,
        mask_transform=mask_transform,
        depth_transform=depth_transform,
    )
    
    print(f"\nDataset size: {len(dataset)} items")
    
    # Create debug output directory
    debug_dir = Path("./debug_vis_dataset")
    debug_dir.mkdir(exist_ok=True)
    print(f"Debug images will be saved to: {debug_dir}")
    
    # Generate 10 debug images from random samples
    import random
    random.seed(42)  # For reproducibility
    random_indices = random.sample(range(len(dataset)), min(10, len(dataset)))
    
    for i, idx in enumerate(random_indices):
        print(f"\nProcessing sample {i+1}/10 (index {idx})...")
        
        try:
            # Load sample
            (
                rgb_tensor,
                depth_tensor,
                description,
                mask_tensor,
                bbox_tensor,
                origin_2d_norm,
                motion_dir_3d,
                motion_type,
                image_size,
                rgb_filename,
                motion_origin_3d,
                camera_intrinsic,
                trajectory_3d,
            ) = dataset[idx]
            
            # Convert tensors back to numpy for visualization
            # RGB: (3, H, W) -> (H, W, 3) and denormalize
            rgb_np = rgb_tensor.permute(1, 2, 0).numpy()
            rgb_np = rgb_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            rgb_np = np.clip(rgb_np, 0, 1)
            rgb_np = (rgb_np * 255).astype(np.uint8)
            
            # Depth: (1, H, W) -> (H, W)
            depth_np = depth_tensor.squeeze(0).numpy()
            
            # Mask: (1, H, W) -> (H, W)
            mask_np = mask_tensor.squeeze(0).numpy()
            
            # Convert to OpenCV format (BGR)
            debug_image = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            h, w = debug_image.shape[:2]
            
            # Get original image dimensions from the dataset
            original_w, original_h = image_size.numpy()
            
            # Calculate scaling factors for coordinate transformation
            scale_x = w / original_w
            scale_y = h / original_h
            
            # Convert normalized coordinates back to pixel coordinates in resized image
            origin_2d_pixels = origin_2d_norm.numpy() * np.array([w, h])
            
            # Convert camera intrinsics to numpy and scale for resized image
            K_matrix = camera_intrinsic.numpy()
            # Scale intrinsic matrix for the resized image
            K_matrix_scaled = K_matrix.copy()
            K_matrix_scaled[0, 0] *= scale_x  # fx
            K_matrix_scaled[1, 1] *= scale_y  # fy
            K_matrix_scaled[0, 2] *= scale_x  # cx
            K_matrix_scaled[1, 2] *= scale_y  # cy
            
            # Convert trajectory to numpy
            trajectory_np = trajectory_3d.numpy()
            
            # Project trajectory points to image coordinates
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
            
            # Project and draw trajectory points (blue dots) using scaled intrinsics
            traj_map = project_camera_to_image(trajectory_np, K_matrix_scaled, w, h)
            vis_mask = traj_map[:, 2] == 1
            visible_traj_points = traj_map[vis_mask, :2]
            
            # Draw trajectory points as blue dots
            for pt in visible_traj_points:
                cv2.circle(debug_image, (int(pt[1]), int(pt[0])), 2, (255, 0, 0), -1)
            
            # Draw interaction point (green dot, larger)
            if origin_2d_pixels[0] >= 0 and origin_2d_pixels[1] >= 0:
                cv2.circle(debug_image, (int(origin_2d_pixels[0]), int(origin_2d_pixels[1])), 5, (0, 255, 0), -1)
            
            # Draw mask points (red dots, subsampled for clarity)
            mask_coords = np.where(mask_np > 0.5)
            if len(mask_coords[0]) > 0:
                # Subsample mask points for visibility
                step = max(1, len(mask_coords[0]) // 100)
                for y, x in zip(mask_coords[0][::step], mask_coords[1][::step]):
                    cv2.circle(debug_image, (int(x), int(y)), 2, (0, 0, 255), -1)
            
            # Draw motion arrow(s)
            motion_type_str = "translation" if motion_type.item() == 0 else "rotation"
            motion_origin_3d_np = motion_origin_3d.numpy()
            motion_dir_3d_np = motion_dir_3d.numpy()
            
            arrow_scale_m = 0.2
            if motion_type_str == "translation":
                # Cyan arrow for translation motion direction
                end_pt_3d = motion_origin_3d_np + motion_dir_3d_np * arrow_scale_m
                origin_proj = project_camera_to_image(motion_origin_3d_np, K_matrix_scaled, w, h)
                end_proj = project_camera_to_image(end_pt_3d, K_matrix_scaled, w, h)
                if origin_proj[0, 2] == 1 and end_proj[0, 2] == 1:
                    start = (int(origin_proj[0, 1]), int(origin_proj[0, 0]))
                    end = (int(end_proj[0, 1]), int(end_proj[0, 0]))
                    cv2.arrowedLine(debug_image, start, end, (0, 255, 255), 3, tipLength=0.1)
            elif motion_type_str == "rotation":
                # Yellow double arrow for rotation axis direction
                end1 = motion_origin_3d_np + motion_dir_3d_np * arrow_scale_m
                origin_proj = project_camera_to_image(motion_origin_3d_np, K_matrix_scaled, w, h)
                end1_proj = project_camera_to_image(end1, K_matrix_scaled, w, h)
                if origin_proj[0, 2] == 1 and end1_proj[0, 2] == 1:
                    start = (int(origin_proj[0, 1]), int(origin_proj[0, 0]))
                    end = (int(end1_proj[0, 1]), int(end1_proj[0, 0]))
                    cv2.arrowedLine(debug_image, start, end, (255, 255, 0), 3, tipLength=0.1)
                    cv2.arrowedLine(debug_image, end, start, (255, 255, 0), 3, tipLength=0.1)
            
            # Text overlays with legend information
            cv2.putText(debug_image, f"{motion_type_str}: {description[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_image, f"Trajectory points: {int(vis_mask.sum())}/{trajectory_np.shape[0]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add legend
            legend_y = h - 120
            cv2.putText(debug_image, "Blue dots: Trajectory points", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(debug_image, "Green dot: Interaction point", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(debug_image, "Red dots: Mask pixels", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if motion_type_str == "translation":
                cv2.putText(debug_image, "Cyan arrow: Translation direction", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            elif motion_type_str == "rotation":
                cv2.putText(debug_image, "Yellow arrows: Rotation axis", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Save debug image
            debug_filename = f"dataset_debug_{i+1:02d}_{rgb_filename.replace('.jpg', '')}.png"
            debug_image_path = debug_dir / debug_filename
            cv2.imwrite(str(debug_image_path), debug_image)
            
            print(f"  Saved: {debug_filename}")
            print(f"  Description: {description[:100]}...")
            print(f"  Motion type: {motion_type_str}")
            print(f"  Trajectory points: {int(vis_mask.sum())}/{trajectory_np.shape[0]}")
            
        except Exception as e:
            print(f"  ERROR processing sample {idx}: {e}")
            continue
    
    print(f"\nDebug visualization complete! Check {debug_dir} for {len(random_indices)} debug images.")
