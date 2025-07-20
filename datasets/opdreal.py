import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Any, Optional, Callable, Dict, List, Tuple, Union
import h5py
from pycocotools import mask as coco_mask
import cv2
import random
import textwrap

from .OPDReal.motion_data import load_motion_json


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


class OPDRealDataset(Dataset):
    """
    PyTorch Dataset for loading OPDReal items.
    Each sample corresponds to an individual articulating part in a specific frame.
    """

    def __init__(
        self,
        data_path: str,
        dataset_key: str,  # e.g., 'opd_c_real_train'
        rgb_transform: Callable,
        mask_transform: Callable,
        depth_transform: Callable,
        return_filename: bool = False,
    ):
        self.data_path = data_path
        self.dataset_key = dataset_key
        self.rgb_transform = rgb_transform
        self.mask_transform = mask_transform
        self.depth_transform = depth_transform
        self.return_filename = return_filename

        # Load annotations
        json_path = os.path.join(
            self.data_path, "annotations_wdf", f"{self.dataset_key}.json"
        )
        image_root = os.path.join(self.data_path, self.dataset_key.split("_")[-1])
        dataset_dicts = load_motion_json(json_path, image_root, self.dataset_key)

        # Create a flat list of items, where each item is (image_dict, annotation)
        self.items = []
        for d in dataset_dicts:
            for anno in d.get("annotations", []):
                self.items.append((d, anno))

        # H5 file handling
        self.h5_file: Optional[h5py.File] = None
        self.images_dset: Optional[h5py.Dataset] = None
        self.filenames_map: Optional[Dict[str, int]] = None
        self._init_h5()

        # Depth H5 file handling
        self.depth_h5_file: Optional[h5py.File] = None
        self.depth_images_dset: Optional[h5py.Dataset] = None
        self.depth_filenames_map: Optional[Dict[str, int]] = None
        self._init_depth_h5()

    def _init_h5(self):
        split_name = self.dataset_key.split("_")[-1]
        h5_path = os.path.join(self.data_path, f"{split_name}.h5")
        self.h5_file = h5py.File(h5_path, "r")
        images_dset = self.h5_file[f"{split_name}_images"]
        assert isinstance(images_dset, h5py.Dataset)
        self.images_dset = images_dset
        filenames_dset = self.h5_file[f"{split_name}_filenames"]
        assert isinstance(filenames_dset, h5py.Dataset)
        self.filenames_map = {
            name.decode("utf-8"): i for i, name in enumerate(list(filenames_dset))
        }

    def _init_depth_h5(self):
        depth_h5_path = os.path.join(self.data_path, "depth.h5")
        self.depth_h5_file = h5py.File(depth_h5_path, "r")
        depth_images_dset = self.depth_h5_file["depth_images"]
        assert isinstance(depth_images_dset, h5py.Dataset)
        self.depth_images_dset = depth_images_dset
        depth_filenames_dset = self.depth_h5_file["depth_filenames"]
        assert isinstance(depth_filenames_dset, h5py.Dataset)
        self.depth_filenames_map = {
            name.decode("utf-8"): i for i, name in enumerate(list(depth_filenames_dset))
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Union[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            str,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            str,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            str,
        ],
    ]:
        image_dict, anno = self.items[idx]

        assert self.filenames_map is not None
        assert self.images_dset is not None
        assert self.depth_filenames_map is not None
        assert self.depth_images_dset is not None

        # 1. rgb_image_tensor
        img_filename = os.path.basename(image_dict["file_name"])
        img_index = self.filenames_map[img_filename]
        img_array = self.images_dset[img_index][:, :, :3]
        rgb_image_pil = Image.fromarray(img_array)
        rgb_image_tensor = self.rgb_transform(rgb_image_pil)
        original_width, original_height = rgb_image_pil.size

        # 2. depth_image_tensor
        depth_filename = image_dict["depth_file_name"]
        depth_img_index = self.depth_filenames_map[depth_filename]
        depth_array = self.depth_images_dset[depth_img_index]
        if isinstance(depth_array, np.ndarray):
            if depth_array.ndim == 3 and depth_array.shape[2] == 1:
                depth_array = depth_array.squeeze(axis=2)
        else:
            # Handle cases where it might not be a numpy array as expected
            depth_array = np.array(depth_array)
            if depth_array.ndim == 3 and depth_array.shape[2] == 1:
                depth_array = depth_array.squeeze(axis=2)

        depth_pil = Image.fromarray(depth_array, mode="F")
        depth_image_tensor = self.depth_transform(depth_pil)

        # 3. description
        description = anno["description"]

        # 4. mask_tensor (individual articulating part)
        segm = anno["segmentation"]
        if isinstance(segm, list):  # polygon
            rles = coco_mask.frPyObjects(
                segm, image_dict["height"], image_dict["width"]
            )
            merged_rle = coco_mask.merge(rles)
            part_mask = coco_mask.decode(merged_rle)
        elif isinstance(segm, dict) and "counts" in segm and "size" in segm:  # RLE
            part_mask = coco_mask.decode(segm)  # type: ignore
        else:
            part_mask = np.zeros(
                (image_dict["height"], image_dict["width"]), dtype=np.uint8
            )
        mask_pil = Image.fromarray(part_mask.astype(np.uint8) * 255, mode="L")
        mask_tensor = self.mask_transform(mask_pil)

        # 5. Bbox
        bbox_tensor = torch.tensor(anno["bbox"], dtype=torch.float32)  # XYWH

        # 6. origin_2d_image_coord_norm & 7. motion_dir_3d_camera_coords
        is_multi = "_m_" in self.dataset_key
        if is_multi:
            intrinsic_matrix = np.reshape(
                image_dict["camera"]["intrinsic"], (3, 3), order="F"
            )
        else:
            intrinsic_matrix = np.reshape(
                image_dict["camera"]["intrinsic"]["matrix"], (3, 3), order="F"
            )

        motion = anno["motion"]
        if "current_origin" in motion:
            motion_origin_3d = motion["current_origin"]
            motion_dir_3d = motion["current_axis"]
        else:
            motion_origin_3d = motion["origin"]
            motion_dir_3d = motion["axis"]

        motion_dir_3d_camera_coords = torch.tensor(motion_dir_3d, dtype=torch.float32)

        motion_type = motion["type"]
        motion_type_map = {"translation": 0, "rotation": 1}
        motion_type_tensor = torch.tensor(
            motion_type_map[motion_type], dtype=torch.long
        )

        point_camera = np.array(motion_origin_3d)
        point_2d_homo = np.dot(intrinsic_matrix, point_camera[:3])
        origin_x = point_2d_homo[0] / point_2d_homo[2]
        origin_y = point_2d_homo[1] / point_2d_homo[2]

        norm_x = origin_x / original_width
        norm_y = origin_y / original_height
        origin_2d_image_coord_norm = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        origin_2d_image_coord_norm = torch.clamp(origin_2d_image_coord_norm, 0.0, 1.0)

        data_tuple = (
            rgb_image_tensor,
            depth_image_tensor,
            description,
            mask_tensor,
            bbox_tensor,
            origin_2d_image_coord_norm,
            motion_dir_3d_camera_coords,
            motion_type_tensor,
            torch.tensor([original_width, original_height], dtype=torch.float32),
        )

        if self.return_filename:
            return data_tuple + (img_filename,)
        else:
            return data_tuple

    def __del__(self):
        if self.h5_file:
            self.h5_file.close()
        if self.depth_h5_file:
            self.depth_h5_file.close()
