"""Spatial segmentation losses and motion-aligned temporal-consistency losses."""

from .segmentation import (
    SpatialLossOutput,
    SpatialSegmentationLoss,
    soft_dice_loss,
    soft_jaccard_loss,
)
from .temporal import (
    ControlFeatureTemporalLoss,
    PixelTemporalLoss,
    TemporalConsistencyLoss,
    TemporalLossOutput,
    charbonnier,
    robust_distance,
    soft_area,
    soft_centroid,
    temporal_weight_factor,
)

__all__ = [
    "SpatialSegmentationLoss",
    "SpatialLossOutput",
    "soft_dice_loss",
    "soft_jaccard_loss",
    "TemporalConsistencyLoss",
    "TemporalLossOutput",
    "PixelTemporalLoss",
    "ControlFeatureTemporalLoss",
    "charbonnier",
    "robust_distance",
    "soft_centroid",
    "soft_area",
    "temporal_weight_factor",
]
