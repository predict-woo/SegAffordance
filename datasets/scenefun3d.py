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
        mask_transform: Optional[
            Callable
        ] = None,  # May not be directly applicable to coordinate-based masks in the same way
        # skip_items_without_motion: bool = True, # This logic is now handled during LMDB creation
        image_size_for_mask_reconstruction: Tuple[int, int] = (
            224,
            224,
        ),  # Needed if original size not stored
    ):
        """
        Args:
            lmdb_data_root (str): Path to the root directory of the LMDB dataset
                                     (output of convert_to_lmdb.py).
            rgb_transform (callable, optional): Optional transform to be applied on the RGB image.
            mask_transform (callable, optional): Optional transform for the reconstructed mask.
                                                Note: Mask is binary {0,1} after reconstruction.
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
        self.image_size_for_mask_reconstruction = (
            image_size_for_mask_reconstruction  # (height, width)
        )

        self.lmdb_path = self.lmdb_data_root / "data.lmdb"
        self.lmdb_path = Path("/dev/shm/data.lmdb")

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
        # Path is relative to lmdb_data_root
        rgb_image_actual_path = (
            self.lmdb_data_root / item_data["rgb_image_path_relative"]
        )
        rgb_image_pil = Image.open(rgb_image_actual_path).convert("RGB")
        original_width, original_height = rgb_image_pil.size

        if self.rgb_transform:
            rgb_image_tensor = self.rgb_transform(rgb_image_pil)
        else:
            rgb_image_tensor = transforms.ToTensor()(
                rgb_image_pil
            )  # Default if no transform

        mask_np = np.zeros((original_height, original_width), dtype=np.uint8)
        mask_coords_yx = item_data.get("mask_coordinates_yx", [])
        if mask_coords_yx:  # Ensure there are coordinates
            rows, cols = zip(*mask_coords_yx)  # Separate y and x
            mask_np[np.array(rows), np.array(cols)] = 255  # Fill in the mask

        mask_pil = Image.fromarray(
            mask_np, mode="L"
        )  # Convert to PIL Image (Grayscale)

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

        # Clamp the normalized coordinates to be within [0, 1]
        origin_2d_image_coord_norm = torch.clamp(origin_2d_image_coord_norm, 0.0, 1.0)

        # Match return signature of previous version
        # img, word, mask, interaction_point
        return rgb_image_tensor, description, mask_tensor, origin_2d_image_coord_norm

    def __del__(self):
        if self.env:
            self.env.close()


def get_default_transforms(
    image_size: Tuple[int, int] = (256, 256)  # (height, width)
) -> Tuple[Callable, Callable]:
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
    return rgb_transform, mask_transform


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
    # --- Configuration --- #
    # !!! IMPORTANT: Set this to the root directory of your LMDB-converted data !!!
    LMDB_DATASET_ROOT = "/cluster/scratch/andrye/SF3D_lmdb"  # EXAMPLE PATH
    # Example: LMDB_DATASET_ROOT = "output_from_convert_to_lmdb_script/train"

    BATCH_SIZE = 4
    NUM_WORKERS = 0  # Start with 0 for LMDB, can increase but requires careful LMDB handling in __getitem__ or worker_init_fn
    TARGET_IMAGE_SIZE = (224, 224)  # (height, width)

    print(f"Attempting to load LMDB data from: {LMDB_DATASET_ROOT}")

    # Get default transforms (or define your own)
    default_rgb_transform, default_mask_transform = get_default_transforms(
        image_size=TARGET_IMAGE_SIZE
    )

    # Create Dataset
    item_dataset = SF3DDataset(
        lmdb_data_root=LMDB_DATASET_ROOT,
        rgb_transform=default_rgb_transform,
        mask_transform=default_mask_transform,  # Pass the transform for reconstructed PIL mask
        image_size_for_mask_reconstruction=TARGET_IMAGE_SIZE,  # Pass this for default internal mask processing
    )

    # Create DataLoader
    item_dataloader = DataLoader(
        item_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            True if torch.cuda.is_available() and NUM_WORKERS == 0 else False
        ),  # pin_memory often needs num_workers=0 or custom handling
    )

    print(f"Successfully created DataLoader with {len(item_dataloader)} batches.")

    # --- Iterate through a few batches to test --- #
    print(
        f"Iterating through DataLoader batches (TARGET_IMAGE_SIZE: {TARGET_IMAGE_SIZE}):"
    )
    for i, batch_data in enumerate(tqdm.tqdm(item_dataloader, desc="Batches")):
        rgb_images, descriptions, masks, interaction_points = batch_data

        # Basic checks
        if i == 0:  # Check first batch details
            print(f"  Batch {i+1}:")
            print(
                f"    RGB Image batch shape: {rgb_images.shape}, type: {rgb_images.dtype}"
            )
            # save rgb_images[0] as an image using cv2
            cv2.imwrite(
                "debug_rgb_sample.png", rgb_images[0].permute(1, 2, 0).numpy() * 255
            )
            print(f"    Mask batch shape: {masks.shape}, type: {masks.dtype}")
            cv2.imwrite(
                "debug_mask_sample.png", masks[0].permute(1, 2, 0).numpy() * 255
            )
            print(f"    Descriptions (first in batch): '{descriptions[0][:70]}...'")
            print(f"    Interaction Points (first in batch): {interaction_points[0]}")
            print(f"    Interaction Points batch shape: {interaction_points.shape}")

        # Check for empty description string (already done in LMDB generation but good for sanity)
        for desc_idx, description in enumerate(descriptions):
            if description == "":
                print(
                    f"Warning: Empty description string in batch {i}, item index in batch {desc_idx}"
                )

        # Check mask values (should be 0. or 1.)
        if not ((masks == 0.0) | (masks == 1.0)).all():
            print(f"Warning: Batch {i} masks are not strictly binary (0.0 or 1.0).")
            unique_mask_vals = torch.unique(masks)
            print(f"    Unique values in mask batch: {unique_mask_vals}")

        # For more detailed debugging, you might save a sample image and mask:
        # if i == 0:
        #     from torchvision.utils import save_image
        #     save_image(rgb_images[0], "debug_rgb_sample.png")
        #     save_image(masks[0], "debug_mask_sample.png") # This will be a binary mask
        #     print("    Saved sample RGB and mask for the first item in the first batch.")

        if i >= 2 and os.environ.get("CI"):  # Stop early in CI for speed
            break

    print("\nDone with example dataloader iteration.")

    # Explicitly delete dataset to trigger __del__ for LMDB env.close() if needed,
    # though Python's GC should handle it upon script exit.
    del item_dataset
    del item_dataloader
    print("Dataset and DataLoader objects deleted.")
