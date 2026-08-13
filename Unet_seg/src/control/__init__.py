"""Control-oriented outputs: postprocessing, feature extraction, quality and validity."""

from .features import (
    FeatureExtractionConfig,
    TemporalStabilityConfig,
    binary_mask_geometry,
    border_contact_ratio,
    boundary_entropy,
    extract_control_state,
    lumen_contrast,
    segmentation_confidence_from_probability,
    warp_mask_with_flow,
)
from .postprocess import PostprocessConfig, PostprocessResult, postprocess_probability
from .quality import (
    QUALITY_COMPONENT_NAMES,
    QualityConfig,
    QualityResult,
    compute_control_quality,
)
from .state import COORDINATE_CONVENTION, BoundingBox, ControlState
from .validity import REJECTION_REASONS, ValidityConfig, ValidityResult, evaluate_validity

__all__ = [
    "ControlState",
    "BoundingBox",
    "COORDINATE_CONVENTION",
    "extract_control_state",
    "FeatureExtractionConfig",
    "TemporalStabilityConfig",
    "binary_mask_geometry",
    "segmentation_confidence_from_probability",
    "boundary_entropy",
    "border_contact_ratio",
    "lumen_contrast",
    "warp_mask_with_flow",
    "PostprocessConfig",
    "PostprocessResult",
    "postprocess_probability",
    "QualityConfig",
    "QualityResult",
    "compute_control_quality",
    "QUALITY_COMPONENT_NAMES",
    "ValidityConfig",
    "ValidityResult",
    "evaluate_validity",
    "REJECTION_REASONS",
]
