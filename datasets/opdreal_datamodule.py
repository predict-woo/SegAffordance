import typing
from typing import Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from .opdreal import OPDRealDataset, get_default_transforms


class OPDRealDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_dataset_key: str,
        val_dataset_key: str,
        test_dataset_key: str,
        input_size: Tuple[int, int],
        batch_size_train: int,
        batch_size_val: int,
        num_workers_train: int,
        num_workers_val: int,
        return_filename: bool = False,
        is_multi: bool = False,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.train_dataset_key = train_dataset_key
        self.val_dataset_key = val_dataset_key
        self.input_size = input_size
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.return_filename = return_filename
        self.is_multi = is_multi

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            train_rgb_transform, train_mask_transform, train_depth_transform = (
                get_default_transforms(image_size=self.input_size, is_train=True)
            )
            val_rgb_transform, val_mask_transform, val_depth_transform = (
                get_default_transforms(image_size=self.input_size, is_train=False)
            )

            self.train_dataset = OPDRealDataset(
                data_path=self.data_path,
                dataset_key=self.train_dataset_key,
                rgb_transform=train_rgb_transform,
                mask_transform=train_mask_transform,
                depth_transform=train_depth_transform,
                return_filename=self.return_filename,
                is_multi=self.is_multi,
            )

            self.val_dataset = OPDRealDataset(
                data_path=self.data_path,
                dataset_key=self.val_dataset_key,
                rgb_transform=val_rgb_transform,
                mask_transform=val_mask_transform,
                depth_transform=val_depth_transform,
                return_filename=self.return_filename,
                is_multi=self.is_multi,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup('fit') first.")
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
            raise RuntimeError("Validation dataset not initialized. Call setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            num_workers=self.num_workers_val,
            pin_memory=True,
            drop_last=False,
        )


