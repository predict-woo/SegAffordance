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
        
        # --- Create Placeholder Depth Image ---
        zero_depth = np.zeros((original_height, original_width), dtype=np.float32)
        depth_pil = Image.fromarray(zero_depth, mode="F")
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


        # Match return signature of OPDRealDataset
        return (
            rgb_image_tensor,
            depth_image_tensor,
            description,
            mask_tensor,
            bbox_tensor,
            origin_2d_image_coord_norm,
            motion_dir_3d_camera_coords,
            motion_type_tensor,
            image_size_tensor,
        )

    def __del__(self):
        if self.env:
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
