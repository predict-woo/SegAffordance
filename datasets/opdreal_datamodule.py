import typing
from typing import Optional, Tuple
import json

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
        use_depth: bool = True,
        return_camera_params: bool = False,
        origin_norm_json_path: typing.Optional[str] = None,
        origin_norm_json_format: str = 'real',
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.train_dataset_key = train_dataset_key
        self.val_dataset_key = val_dataset_key
        self.test_dataset_key = test_dataset_key
        self.input_size = input_size
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.return_filename = return_filename
        self.is_multi = is_multi
        self.use_depth = use_depth
        self.return_camera_params = return_camera_params
        self.origin_norm_json_path = origin_norm_json_path
        self.origin_norm_json_format = origin_norm_json_format
        self.origin_norm_diagonals = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.origin_norm_json_path and self.origin_norm_diagonals is None:
            print(f"📏 Loading 3D diagonal data for origin normalization from: {self.origin_norm_json_path}")
            try:
                with open(self.origin_norm_json_path, 'r') as f:
                    data = json.load(f)
                    if self.origin_norm_json_format == 'real':
                        self.origin_norm_diagonals = {int(k): v['diameter'] for k, v in data.items()}
                    elif self.origin_norm_json_format == 'multi':
                        self.origin_norm_diagonals = {k: v['diagonal'] for k, v in data.items()}
                    else:
                        raise ValueError(f"Unknown origin_norm_json_format: {self.origin_norm_json_format}")
            except Exception as e:
                print(f"⚠️ Error loading origin normalization data: {e}. Origin error will not be calculated.")
                self.origin_norm_diagonals = None

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
                use_depth=self.use_depth,
                return_camera_params=self.return_camera_params,
            )

            self.val_dataset = OPDRealDataset(
                data_path=self.data_path,
                dataset_key=self.val_dataset_key,
                rgb_transform=val_rgb_transform,
                mask_transform=val_mask_transform,
                depth_transform=val_depth_transform,
                return_filename=self.return_filename,
                is_multi=self.is_multi,
                use_depth=self.use_depth,
                return_camera_params=self.return_camera_params,
            )

        if stage in ("test", None):
            test_rgb_transform, test_mask_transform, test_depth_transform = (
                get_default_transforms(image_size=self.input_size, is_train=False)
            )

            self.test_dataset = OPDRealDataset(
                data_path=self.data_path,
                dataset_key=self.test_dataset_key,
                rgb_transform=test_rgb_transform,
                mask_transform=test_mask_transform,
                depth_transform=test_depth_transform,
                return_filename=self.return_filename,
                is_multi=self.is_multi,
                use_depth=self.use_depth,
                return_camera_params=self.return_camera_params,
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

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            num_workers=self.num_workers_val,
            pin_memory=True,
            drop_last=False,
        )


