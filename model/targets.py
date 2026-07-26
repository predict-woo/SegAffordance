"""Named view over a training batch.

The datamodules emit plain tuples whose length encodes which dataset produced
them (9 for OPD, 13 for SF3D with trajectories). Loss modules cannot branch on
that, so the training step packs the batch into this bundle once and hands it
around. Fields absent for a given dataset are ``None``.

This lives beside :mod:`model.outputs` rather than in the training module so
that ``model.losses`` can name it without importing the trainer that imports
``model.losses``.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch


@dataclass
class StepTargets:
    #: (B, 1, H, W) binary segmentation target
    mask: Optional[torch.Tensor] = None
    #: (B, 2) interaction point, normalised to [0, 1]
    point_norm: Optional[torch.Tensor] = None
    #: (B, 3) ground-truth motion axis
    motion: Optional[torch.Tensor] = None
    #: (B,) motion type — 1 is rotation/revolute
    motion_type: Optional[torch.Tensor] = None
    #: (B, 2) original image size
    img_size: Optional[torch.Tensor] = None
    #: (B, num_points, 3) trajectory in absolute coordinates — SF3D only
    trajectory: Optional[torch.Tensor] = None
    #: (B, 3) 3D motion origin in absolute coordinates — SF3D only
    motion_origin_3d: Optional[torch.Tensor] = None
    #: (B, 3, 3) camera intrinsics — SF3D only
    camera_intrinsic: Optional[torch.Tensor] = None


def unpack_batch(batch) -> Tuple[Any, Any, Any, StepTargets]:
    """Split a training batch into model inputs and a named target bundle.

    The datasets emit tuples whose length says which produced them: 13 for
    SF3D with trajectories, 9 for OPD. Dispatching on that here means nothing
    downstream — least of all the loss modules — has to.

    Returns ``(img, depth, word_str_list, targets)``.
    """
    if len(batch) == 13:  # SF3D with trajectory
        (
            img, depth, word_str_list, mask_gt, _bbox, point_gt_norm, motion_gt,
            motion_type_gt, img_size, _rgb_filename, motion_origin_3d,
            camera_intrinsic, trajectory_gt,
        ) = batch
    elif len(batch) > 10:  # Other SF3D case, no trajectory
        (
            img, depth, word_str_list, mask_gt, _bbox, point_gt_norm, motion_gt,
            motion_type_gt, img_size, *_,
        ) = batch
        motion_origin_3d = camera_intrinsic = trajectory_gt = None
    else:  # OPDReal / OPDMulti
        (
            img, depth, word_str_list, mask_gt, _bbox, point_gt_norm, motion_gt,
            motion_type_gt, img_size,
        ) = batch
        motion_origin_3d = camera_intrinsic = trajectory_gt = None

    targets = StepTargets(
        mask=mask_gt,
        point_norm=point_gt_norm,
        motion=motion_gt,
        motion_type=motion_type_gt,
        img_size=img_size,
        trajectory=trajectory_gt,
        motion_origin_3d=motion_origin_3d,
        camera_intrinsic=camera_intrinsic,
    )
    return img, depth, word_str_list, targets
