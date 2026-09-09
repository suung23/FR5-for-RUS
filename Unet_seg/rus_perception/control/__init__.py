"""Control-oriented outputs: postprocessing, feature extraction, quality and validity.

Two image-quality functions live here, deliberately kept apart:

* :func:`compute_control_quality` (``Q_seg``) scores the *segmentation*, and is
  undefined until the bladder has been found;
* :func:`compute_raw_quality` (``Q_raw``) scores the *raw B-mode frame*, and is
  therefore defined from the first frame of contact onward.

They share this package because they share the acquisition geometry, not
because they measure the same thing. Neither emits a robot command.
"""

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
    FORCE_SEARCH_WEIGHTS,
    QUALITY_COMPONENT_NAMES,
    QualityConfig,
    QualityResult,
    compute_control_quality,
)
from .roi import ROI_MODES, RoiConfig, RoiError, build_roi_mask, roi_area_px
from .raw_quality import (
    RAW_QUALITY_COMPONENT_NAMES,
    RAW_REJECTION_REASONS,
    RawQualityConfig,
    RawQualityResult,
    ScanGeometryError,
    compute_raw_quality,
    coupling_gate,
    sample_a_lines,
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
    "FORCE_SEARCH_WEIGHTS",
    "RawQualityConfig",
    "RawQualityResult",
    "compute_raw_quality",
    "coupling_gate",
    "sample_a_lines",
    "ScanGeometryError",
    "RAW_QUALITY_COMPONENT_NAMES",
    "RAW_REJECTION_REASONS",
    "RoiConfig",
    "RoiError",
    "build_roi_mask",
    "roi_area_px",
    "ROI_MODES",
    "ValidityConfig",
    "ValidityResult",
    "evaluate_validity",
    "REJECTION_REASONS",
]
