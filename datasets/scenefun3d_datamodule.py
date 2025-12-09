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
