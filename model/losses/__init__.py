"""Loss modules that consume :class:`model.outputs.ModelOutputs`."""

from .geometric import (
    CrossGTGeometricLoss,
    GeometricConsistencyLoss,
    NoGeometricLoss,
    PredPredArticulationLoss,
    PredPredGeometricLoss,
    ScrewConsistencyLoss,
    build_geometric_loss,
)
from .split import (
    axis_direction_loss,
    origin_canonical_loss,
    perpendicular_foot,
)
from .twist import (
    TwistLoss,
    decode_twist,
    point_to_line_distance,
    screw_orbit,
    twist_from_gt,
)

__all__ = [
    "GeometricConsistencyLoss",
    "NoGeometricLoss",
    "PredPredGeometricLoss",
    "PredPredArticulationLoss",
    "CrossGTGeometricLoss",
    "ScrewConsistencyLoss",
    "build_geometric_loss",
    "TwistLoss",
    "twist_from_gt",
    "decode_twist",
    "point_to_line_distance",
    "screw_orbit",
    "perpendicular_foot",
    "origin_canonical_loss",
    "axis_direction_loss",
]
