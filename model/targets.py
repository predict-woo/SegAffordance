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
    #: (B, 3, 3) camera intrinsics, at ORIGINAL image resolution — SF3D only
    camera_intrinsic: Optional[torch.Tensor] = None
    #: (B, N, 2) observed 2D track, normalised to [0, 1]. In 2D pretraining
    #: the hand/contact path mined from video; from SF3D (15-tuple batches)
    #: the element sweep projected into the frame — same curve `trajectory`
    #: holds, which is exactly what makes SF3D a testbed for the 2D arm.
    trajectory_2d: Optional[torch.Tensor] = None
    #: (B, N) bool — which 2D track points are real observations. SF3D marks
    #: points behind the camera or outside the frame invalid and stores [0, 0]
    #: placeholders for them; every 2D loss must mask on this rather than
    #: regress the placeholders. None means "all points valid" (2D pretraining
    #: sources that only emit observed points).
    trajectory_2d_valid: Optional[torch.Tensor] = None
    #: (B,) metric depth of the 2D track's first point. Supplied rather than
    #: read from the depth map here, so depth-map scaling conventions (and
    #: pseudo-depth) stay out of the geometry.
    anchor_depth: Optional[torch.Tensor] = None


def unpack_batch(batch) -> Tuple[Any, Any, Any, StepTargets]:
    """Split a training batch into model inputs and a named target bundle.

    The datasets emit tuples whose length says which produced them: 15 for
    SF3D with the 2D trajectory columns, 13 for SF3D with trajectories, 9 for
    OPD. Dispatching on that here means nothing downstream — least of all the
    loss modules — has to.

    Returns ``(img, depth, word_str_list, targets)``.

    A batch may also be a **dict**, which is how a source with a different set
    of fields (2D pretraining: a hand track and intrinsics, no element sweep)
    supplies them without a positional layout having to be invented for it.
    Missing keys become ``None`` — never zeros, since a zero-filled trajectory
    would train a head to predict zeros rather than leave it untouched.
    """
    if isinstance(batch, dict):
        targets = StepTargets(
            mask=batch.get("mask"),
            point_norm=batch.get("point_norm"),
            motion=batch.get("motion"),
            motion_type=batch.get("motion_type"),
            img_size=batch.get("img_size"),
            trajectory=batch.get("trajectory"),
            motion_origin_3d=batch.get("motion_origin_3d"),
            camera_intrinsic=batch.get("camera_intrinsic"),
            trajectory_2d=batch.get("trajectory_2d"),
            trajectory_2d_valid=batch.get("trajectory_2d_valid"),
            anchor_depth=batch.get("anchor_depth"),
        )
        return batch.get("img"), batch.get("depth"), batch.get("word"), targets

    if len(batch) == 15:  # SF3D with 2D trajectory (return_trajectory_2d=True)
        (
            img, depth, word_str_list, mask_gt, _bbox, point_gt_norm, motion_gt,
            motion_type_gt, img_size, _rgb_filename, motion_origin_3d,
            camera_intrinsic, trajectory_gt, trajectory_2d_px, trajectory_2d_valid,
        ) = batch
        # The dataset emits PIXELS (a trajectory legitimately leaves the
        # frame, so it is stored unclamped); the model-side convention for 2D
        # tracks is [0, 1], so normalise here, once. img_size is (w, h).
        scale = img_size.to(trajectory_2d_px.dtype).clamp(min=1.0).unsqueeze(1)
        trajectory_2d = trajectory_2d_px / scale
        # Metric z of the track's first point — the element centroid's depth,
        # taken from the 3D curve the 2D one is a projection of (data-side
        # lifting of an observation, not teacher forcing of a prediction).
        anchor_depth = trajectory_gt[:, 0, 2]
        targets = StepTargets(
            mask=mask_gt,
            point_norm=point_gt_norm,
            motion=motion_gt,
            motion_type=motion_type_gt,
            img_size=img_size,
            trajectory=trajectory_gt,
            motion_origin_3d=motion_origin_3d,
            camera_intrinsic=camera_intrinsic,
            trajectory_2d=trajectory_2d,
            trajectory_2d_valid=trajectory_2d_valid,
            anchor_depth=anchor_depth,
        )
        return img, depth, word_str_list, targets

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
