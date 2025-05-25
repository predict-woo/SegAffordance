import os
import json
from pathlib import Path
from torch._tensor import Tensor
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Any, Optional, Callable, Dict, List, Tuple
import tqdm


class SF3DDataset(Dataset):
    """
    PyTorch Dataset for loading processed SceneFun3D items.
    Each sample corresponds to a specific item visible in a specific frame.
    """

    def __init__(
        self,
        processed_data_root: str,
        rgb_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        skip_items_without_motion: bool = True,
    ):
        """
        Args:
            processed_data_root (str): Path to the root directory of the processed dataset
                                     (output of process_frames_with_items.py).
            rgb_transform (callable, optional): Optional transform to be applied on the RGB image.
            mask_transform (callable, optional): Optional transform to be applied on the mask.
            skip_items_without_motion (bool): If True, items without valid frame-specific motion data
                                            in motion_info.json will be skipped.
        """
        self.processed_data_root = Path(processed_data_root)
        self.rgb_transform = rgb_transform
        self.mask_transform = mask_transform
        self.skip_items_without_motion = skip_items_without_motion
        self.item_samples = self._find_item_samples()

    def _find_item_samples(self) -> List[Dict[str, Path]]:
        samples = []
        if not self.processed_data_root.is_dir():
            print(f"Error: Processed data root {self.processed_data_root} not found.")
            return samples

        for visit_dir in self.processed_data_root.iterdir():
            if not visit_dir.is_dir():
                continue
            for video_dir in visit_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                for frame_dir in video_dir.iterdir():
                    if not frame_dir.is_dir() or not frame_dir.name.startswith(
                        "frame_"
                    ):
                        continue

                    rgb_image_path = frame_dir / "rgb.jpg"
                    if not rgb_image_path.exists():
                        continue

                    for item_dir in frame_dir.iterdir():
                        if not item_dir.is_dir() or not item_dir.name.startswith(
                            "item_"
                        ):
                            continue

                        mask_path = item_dir / "mask.png"
                        description_path = item_dir / "description.txt"
                        motion_info_path = item_dir / "motion_info.json"

                        if not (
                            mask_path.exists()
                            and description_path.exists()
                            and motion_info_path.exists()
                        ):
                            # print(f"Skipping item {item_dir} due to missing files.")
                            continue

                        # Optionally, further validate motion_info.json content here
                        if self.skip_items_without_motion:
                            try:
                                with open(motion_info_path, "r") as f_motion:
                                    motion_data = json.load(f_motion)
                                if (
                                    not motion_data.get("frame_specific_motion_data")
                                    or motion_data["frame_specific_motion_data"].get(
                                        "motion_origin_2d_image_coords"
                                    )
                                    is None
                                    or motion_data["frame_specific_motion_data"].get(
                                        "motion_dir_3d_camera_coords"
                                    )
                                    is None
                                ):
                                    # print(f"Skipping item {item_dir} due to incomplete motion data.")
                                    continue
                            except json.JSONDecodeError:
                                # print(f"Skipping item {item_dir} due to invalid motion_info.json.")
                                continue
                            except Exception as e:
                                # print(f"Error reading motion info for {item_dir}: {e}")
                                continue

                        samples.append(
                            {
                                "rgb_image_path": rgb_image_path,
                                "mask_path": mask_path,
                                "description_path": description_path,
                                "motion_info_path": motion_info_path,
                            }
                        )
        print(f"Found {len(samples)} processable item samples.")
        return samples

    def __len__(self) -> int:
        return len(self.item_samples)

    def __getitem__(self, idx: int) :
        sample_paths = self.item_samples[idx]

        # Load RGB Image to get original dimensions
        rgb_image_pil = Image.open(sample_paths["rgb_image_path"]).convert("RGB")
        original_width, original_height = rgb_image_pil.size

        # Apply transform if any
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image_pil)
        else:
            rgb_image = transforms.ToTensor()(rgb_image_pil)

        # Load Mask
        mask = Image.open(sample_paths["mask_path"]).convert("L")  # Grayscale
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:  # Default to tensor if no transform
            mask = transforms.ToTensor()(mask)
            mask = (
                mask > 0.5
            ).float()  # Ensure binary mask if not handled by transform

        # Load Description
        with open(sample_paths["description_path"], "r") as f_desc:
            description = f_desc.read().strip()

        # Load Motion Info
        origin_2d_image_coord = torch.zeros(2, dtype=torch.float32)
        motion_3d_camera_coord = torch.zeros(3, dtype=torch.float32)
        # motion_type = "unknown" # Could also return this if needed

        try:
            with open(sample_paths["motion_info_path"], "r") as f_motion:
                motion_data = json.load(f_motion)

            frame_specific_motion = motion_data.get("frame_specific_motion_data")
            if frame_specific_motion:
                origin_2d = frame_specific_motion.get("motion_origin_2d_image_coords")
                motion_3d_cam = frame_specific_motion.get("motion_dir_3d_camera_coords")
                # original_motion = motion_data.get("original_motion_data")
                # if original_motion:
                #     motion_type = original_motion.get("motion_type", "unknown")

                if origin_2d is not None:
                    origin_2d_image_coord = torch.tensor(origin_2d, dtype=torch.float32)
                if motion_3d_cam is not None:
                    motion_3d_camera_coord = torch.tensor(
                        motion_3d_cam, dtype=torch.float32
                    )
        except Exception as e:
            print(
                f"Warning: Could not load or parse motion info for {sample_paths['motion_info_path']}: {e}"
            )
            # Return zeros for coordinates as defined above

        # Normalize origin_2d_image_coord
        origin_2d_image_coord_norm = torch.zeros(2, dtype=torch.float32)
        if original_width > 0 and original_height > 0 :
            origin_2d_image_coord_norm[0] = origin_2d_image_coord[0] / original_width
            origin_2d_image_coord_norm[1] = origin_2d_image_coord[1] / original_height
        
        # Clamp the normalized coordinates to be within [0, 1]
        origin_2d_image_coord_norm = torch.clamp(origin_2d_image_coord_norm, 0.0, 1.0)
        
        img = rgb_image
        word = description
        interaction_point = origin_2d_image_coord_norm

        return img, word, mask, interaction_point

def get_default_transforms(
    image_size: Tuple[int, int] = (256, 256)
) -> Tuple[Callable, Callable]:
    """Returns a default set of transforms for RGB images and masks."""
    rgb_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
            lambda x: (x > 0.5).float(),  # Ensure binary mask {0., 1.}
        ]
    )
    return rgb_transform, mask_transform


if __name__ == "__main__":
    # --- Configuration --- #
    # !!! IMPORTANT: Set this to the root directory of your processed data !!!
    PROCESSED_DATA_DIR = "/local/home/andrye/dev/SF3D_Proc"
    # Example: PROCESSED_DATA_DIR = "output_from_process_frames_script"

    BATCH_SIZE = 4
    NUM_WORKERS = 4

    print(f"Attempting to load data from: {PROCESSED_DATA_DIR}")

    # Get default transforms (or define your own)
    default_rgb_transform, default_mask_transform = get_default_transforms(
        image_size=(224, 224)
    )

    # Create Dataset
    item_dataset = SF3DDataset(
        processed_data_root=PROCESSED_DATA_DIR,
        rgb_transform=default_rgb_transform,
        mask_transform=default_mask_transform,
    )

    if len(item_dataset) == 0:
        print(
            "No items found by the dataset. Check PROCESSED_DATA_DIR and the dataset structure."
        )
        exit()

    # Create DataLoader
    item_dataloader = DataLoader(
        item_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle for training
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(f"Successfully created DataLoader with {len(item_dataloader)} batches.")

    # --- Iterate through a few batches to test --- #
    # num_batches_to_show = min(3, len(item_dataloader))
    # print(f"\nShowing data from first {num_batches_to_show} batches:")
    for i, batch in enumerate(tqdm.tqdm(item_dataloader)):
        # print(batch[2])
        # break
        
        # if i >= num_batches_to_show:
        #     break
        # continue
        # print(f"\n--- Batch {i+1} ---")
        # print(f"RGB Image batch shape: {batch['rgb_image'].shape}, type: {batch['rgb_image'].dtype}")
        # print(f"Mask batch shape: {batch['mask'].shape}, type: {batch['mask'].dtype}")

        # print(f"Descriptions (first in batch): '{batch['description'][0][:70]}...'")
        # print(f"Origin 2D Coords (first in batch): {batch['origin_2d_image_coord'][0]}")
        # print(f"Shape of Origin 2D Coords batch: {batch['origin_2d_image_coord'].shape}")

        # print(f"Motion 3D Cam Coords (first in batch): {batch['motion_3d_camera_coord'][0]}")
        # print(f"Shape of Motion 3D Cam Coords batch: {batch['motion_3d_camera_coord'].shape}")
        # # Example of accessing the new normalized coordinate:
        # print(f"Normalized Origin 2D Coords (first in batch): {batch['origin_2d_image_coord_norm'][0]}")
        # print(f"Shape of Normalized Origin 2D Coords batch: {batch['origin_2d_image_coord_norm'].shape}")
        pass

    print("\nDone with example dataloader iteration.")
