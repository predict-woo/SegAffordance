"""Loss modules that consume :class:`model.outputs.ModelOutputs`."""

from .geometric import (
    CrossGTGeometricLoss,
    GeometricConsistencyLoss,
    NoGeometricLoss,
    PredPredGeometricLoss,
    ScrewConsistencyLoss,
    build_geometric_loss,
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
    "CrossGTGeometricLoss",
    "ScrewConsistencyLoss",
    "build_geometric_loss",
    "TwistLoss",
    "twist_from_gt",
    "decode_twist",
    "point_to_line_distance",
    "screw_orbit",
]
