"""Loss modules that consume :class:`model.outputs.ModelOutputs`."""

from .geometric import (
    CrossGTGeometricLoss,
    GeometricConsistencyLoss,
    NoGeometricLoss,
    PredPredGeometricLoss,
    build_geometric_loss,
)

__all__ = [
    "GeometricConsistencyLoss",
    "NoGeometricLoss",
    "PredPredGeometricLoss",
    "CrossGTGeometricLoss",
    "build_geometric_loss",
]
