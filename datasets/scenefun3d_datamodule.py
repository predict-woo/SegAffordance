import typing
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .scenefun3d import SF3DDataset, get_default_transforms, split_dataset_by_scene


class SF3DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        val_split_ratio: float,
        input_size: Tuple[int, int],
        batch_size_train: int,
        batch_size_val: int,
        num_workers_train: int,
        num_workers_val: int,
        manual_seed: int,
        point_source: str = "motion_origin",
        lmdb_path: Optional[str] = None,
        key_cache_path: Optional[str] = None,
        return_trajectory_2d: bool = False,
        frame_cache_path: Optional[str] = None,
        fast_pipeline: bool = False,
        min_revolute_radius: float = 0.0,
        min_mask_area_frac: float = 0.0,
        edge_margin_frac: float = 0.0,
    ) -> None:
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_split_ratio = val_split_ratio
        self.input_size = input_size
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.manual_seed = manual_seed
        # Passed straight through to SF3DDataset: what the interaction point
        # is supervised with ("motion_origin" | "element"), an optional LMDB
        # override (e.g. a /dev/shm staging copy), and an optional pickle that
        # memoises the sensor-filtered key list (297s -> 3s on a cold volume).
        self.point_source = point_source
        self.lmdb_path = lmdb_path
        self.key_cache_path = key_cache_path
        # Emit the 15-tuple with the 2D trajectory columns (the 2D arm).
        self.return_trajectory_2d = return_trajectory_2d
        # Training-sized frame bytes in one LMDB (tools/sf3d_build_frame_cache.py).
        self.frame_cache_path = frame_cache_path
        # cv2-only decode + uint8 RGB (GPU-normalized in the model) + reused
        # mask buffers: ~5x cheaper per sample. See SF3DDataset.fast_pipeline.
        self.fast_pipeline = fast_pipeline
        # Drop knob/dial-class revolute records (element-to-axis distance
        # below this, metres). See SF3DDataset.min_revolute_radius.
        self.min_revolute_radius = min_revolute_radius
        # GT mask must cover at least this fraction of the image
        # (splat pixels / W*H). See SF3DDataset.min_mask_area_frac.
        self.min_mask_area_frac = min_mask_area_frac
        # Interaction point + projected q* inside the central region
        # (this fraction of W/H from every edge). See SF3DDataset.
        self.edge_margin_frac = edge_margin_frac

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # The validation set is used for testing, so we set it up in both 'fit' and 'test' stages.
        if stage in ("fit", "test", None):
            # We only need to perform the setup once.
            if self.val_dataset is None:
                rgb_transform, mask_transform, depth_transform = get_default_transforms(
                    image_size=self.input_size
                )

                full_dataset = SF3DDataset(
                    lmdb_data_root=self.train_data_dir,
                    rgb_transform=rgb_transform,
                    mask_transform=mask_transform,
                    depth_transform=depth_transform,
                    image_size_for_mask_reconstruction=self.input_size,
                    point_source=self.point_source,
                    lmdb_path=self.lmdb_path,
                    key_cache_path=self.key_cache_path,
                    return_trajectory_2d=self.return_trajectory_2d,
                    frame_cache_path=self.frame_cache_path,
                    fast_pipeline=self.fast_pipeline,
                    min_revolute_radius=self.min_revolute_radius,
                    min_mask_area_frac=self.min_mask_area_frac,
                    edge_margin_frac=self.edge_margin_frac,
                )

                self.train_dataset, self.val_dataset = split_dataset_by_scene(
                    full_dataset,
                    val_split_ratio=self.val_split_ratio,
                    manual_seed=self.manual_seed,
                )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError(
                "Training dataset not initialized. Call setup('fit') first."
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers_train,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers_train > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup('fit') first."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            shuffle=True,  # Enable shuffling for random visualization sampling
            num_workers=self.num_workers_val,
            pin_memory=True,
            drop_last=False,
            # Without this the val workers are torn down and respawned every
            # epoch — measured ~1-2 min of dead time per epoch at 32 workers.
            persistent_workers=self.num_workers_val > 0,
            generator=torch.Generator().manual_seed(self.manual_seed),  # Deterministic shuffling
        )

    def test_dataloader(self) -> DataLoader:
        # SF3D does not have a standard test set defined in this setup
        # Re-using the val set for testing metrics if needed
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup('fit') first."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            num_workers=self.num_workers_val,
            pin_memory=True,
            drop_last=False,
        )
