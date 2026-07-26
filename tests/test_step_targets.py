"""Unit tests for batch unpacking.

``unpack_batch`` maps positional slots of a dataset tuple onto named fields.
Nothing downstream can detect a wrong mapping — a swapped motion origin and
camera intrinsic would just feed the geometric loss garbage — so the indices
are pinned here with distinguishable sentinels.

    python -m pytest tests/test_step_targets.py -q
"""

from model.targets import StepTargets, unpack_batch


# Sentinels rather than tensors: unpack_batch only moves references around, and
# distinct values make a mis-indexed field obvious.
OPD_BATCH = (
    "img", "depth", "words", "mask", "bbox", "point", "motion", "motion_type", "img_size",
)

SF3D_BATCH = OPD_BATCH + ("rgb_filename", "motion_origin", "camera_intrinsic", "trajectory")

SF3D_BATCH_NO_TRAJECTORY = OPD_BATCH + ("rgb_filename", "extra_a")


def test_opd_batch_maps_every_field():
    img, depth, words, targets = unpack_batch(OPD_BATCH)
    assert (img, depth, words) == ("img", "depth", "words")
    assert targets.mask == "mask"
    assert targets.point_norm == "point"
    assert targets.motion == "motion"
    assert targets.motion_type == "motion_type"
    assert targets.img_size == "img_size"


def test_opd_batch_leaves_sf3d_only_fields_unset():
    _, _, _, targets = unpack_batch(OPD_BATCH)
    assert targets.trajectory is None
    assert targets.motion_origin_3d is None
    assert targets.camera_intrinsic is None


def test_sf3d_batch_maps_the_trailing_four_slots():
    """Slots 9-12 are rgb_filename, motion_origin, camera_intrinsic, trajectory.

    Their order is the single most swappable thing in this function: origin and
    intrinsic are adjacent, same-shaped, and only one of them feeds the loss.
    """
    _, _, _, targets = unpack_batch(SF3D_BATCH)
    assert targets.motion_origin_3d == "motion_origin"
    assert targets.camera_intrinsic == "camera_intrinsic"
    assert targets.trajectory == "trajectory"


def test_sf3d_batch_still_maps_the_shared_leading_fields():
    img, depth, words, targets = unpack_batch(SF3D_BATCH)
    assert (img, depth, words) == ("img", "depth", "words")
    assert targets.mask == "mask"
    assert targets.point_norm == "point"
    assert targets.motion == "motion"
    assert targets.motion_type == "motion_type"


def test_medium_length_batch_has_no_trajectory_targets():
    """>10 but not 13: the SF3D variant that carries no trajectory."""
    _, _, _, targets = unpack_batch(SF3D_BATCH_NO_TRAJECTORY)
    assert targets.trajectory is None
    assert targets.motion_origin_3d is None
    assert targets.mask == "mask"


def test_bbox_is_dropped_rather_than_shifting_later_fields():
    """Slot 4 is bbox, which no loss consumes; the fields after it must not
    shift up into its place."""
    _, _, _, targets = unpack_batch(OPD_BATCH)
    assert targets.point_norm != "bbox"
    assert not hasattr(targets, "bbox")


def test_targets_default_to_all_none():
    empty = StepTargets()
    assert all(getattr(empty, f) is None for f in vars(empty))
