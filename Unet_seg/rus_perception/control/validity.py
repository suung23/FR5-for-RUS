"""Validity gate deciding whether a frame yields a usable control measurement.

``valid_for_control = False`` means: *do not treat this frame as a new, reliable
observation*. It is deliberately not a command. What a controller does about it
-- hold position, slow down, re-acquire, stop -- is policy that lives outside
this repository.

Every criterion is optional (``None`` disables it) and every failure produces an
explicit machine-readable reason string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["ValidityConfig", "ValidityResult", "evaluate_validity", "REJECTION_REASONS"]

#: All reason codes this module can emit, for documentation and tests.
REJECTION_REASONS: tuple[str, ...] = (
    "empty_mask",
    "mask_area_too_small",
    "mask_area_too_large",
    "low_segmentation_confidence",
    "excessive_border_contact",
    "fragmented_mask",
    "low_temporal_warped_iou",
    "excessive_centroid_jump",
    "excessive_area_change",
    "low_lumen_contrast",
    "high_boundary_entropy",
    "low_temporal_stability",
    "low_control_quality",
)


@dataclass
class ValidityConfig:
    """Thresholds of the validity gate. ``None`` disables a criterion.

    Attributes:
        min_area_ratio: Smallest acceptable mask area ratio.
        max_area_ratio: Largest acceptable mask area ratio.
        min_segmentation_confidence: Minimum mean binary certainty.
        max_border_contact_ratio: Maximum fraction of the mask perimeter that
            may lie on the image border (i.e. how cut-off the lumen may be).
        min_largest_component_ratio: Minimum retained/raw mask area ratio.
        min_temporal_warped_iou: Minimum IoU against the motion-aligned
            previous mask. Skipped when there is no previous state.
        max_centroid_jump: Maximum normalized centroid displacement per frame.
        max_relative_area_change: Maximum relative area change per frame.
        min_lumen_contrast: Minimum normalized lumen/surrounding contrast.
        max_boundary_entropy: Maximum mean normalized boundary entropy.
        min_temporal_stability_score: Minimum aggregate temporal stability.
        min_control_quality_score: Minimum aggregate control-quality score.
        require_temporal: Reject frames that have no temporal evidence at all.
            Off by default so the first frame of a sequence can be valid.
    """

    min_area_ratio: Optional[float] = 0.005
    max_area_ratio: Optional[float] = 0.80
    min_segmentation_confidence: Optional[float] = 0.50
    max_border_contact_ratio: Optional[float] = 0.50
    min_largest_component_ratio: Optional[float] = 0.60
    min_temporal_warped_iou: Optional[float] = 0.50
    max_centroid_jump: Optional[float] = 0.15
    max_relative_area_change: Optional[float] = 0.50
    min_lumen_contrast: Optional[float] = None
    max_boundary_entropy: Optional[float] = None
    min_temporal_stability_score: Optional[float] = None
    min_control_quality_score: Optional[float] = None
    require_temporal: bool = False

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "ValidityConfig":
        """Build from a config mapping, rejecting unknown keys."""
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown validity key(s): {sorted(unknown)}. Known keys: {sorted(known)}"
            )
        return cls(**data)


@dataclass
class ValidityResult:
    """Outcome of the validity gate."""

    valid: bool
    reasons: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def evaluate_validity(
    mask_area_ratio: float,
    segmentation_confidence: float,
    border_contact_ratio: float,
    largest_component_ratio: float,
    mask_area_px: int = 0,
    temporal_warped_iou: Optional[float] = None,
    normalized_centroid_jump: Optional[float] = None,
    relative_area_change: Optional[float] = None,
    lumen_surrounding_contrast: Optional[float] = None,
    mean_boundary_entropy: Optional[float] = None,
    temporal_stability_score: Optional[float] = None,
    control_quality_score: Optional[float] = None,
    config: Optional[ValidityConfig] = None,
) -> ValidityResult:
    """Apply the validity gate to one frame's extracted features.

    Criteria whose inputs are ``None`` are recorded as *skipped* rather than
    failed, so the first frame of a sequence is not rejected merely for lacking a
    predecessor -- unless ``require_temporal`` is enabled.

    Args:
        mask_area_ratio: Foreground area divided by image area.
        segmentation_confidence: Mean binary certainty in ``[0, 1]``.
        border_contact_ratio: Fraction of the mask perimeter on the image border.
        largest_component_ratio: Retained-component area over raw mask area.
        mask_area_px: Foreground pixel count; 0 triggers ``empty_mask``.
        temporal_warped_iou: IoU against the motion-aligned previous mask.
        normalized_centroid_jump: Normalized centroid displacement.
        relative_area_change: Relative area change against the previous frame.
        lumen_surrounding_contrast: Normalized lumen/surrounding contrast.
        mean_boundary_entropy: Mean normalized boundary entropy.
        temporal_stability_score: Aggregate temporal stability in ``[0, 1]``.
        control_quality_score: Aggregate control-quality score in ``[0, 1]``.
        config: Thresholds.

    Returns:
        A :class:`ValidityResult` with explicit rejection reasons.
    """
    config = config or ValidityConfig()
    reasons: list[str] = []
    checked: list[str] = []
    skipped: list[str] = []

    if mask_area_px <= 0:
        reasons.append("empty_mask")
    checked.append("empty_mask")

    def check(
        name: str,
        value: Optional[float],
        threshold: Optional[float],
        comparison: str,
        reason: str,
    ) -> None:
        if threshold is None:
            skipped.append(f"{name}:disabled")
            return
        if value is None:
            skipped.append(f"{name}:no_value")
            return
        checked.append(name)
        failed = value < threshold if comparison == "min" else value > threshold
        if failed:
            reasons.append(reason)

    check("min_area_ratio", mask_area_ratio, config.min_area_ratio, "min", "mask_area_too_small")
    check("max_area_ratio", mask_area_ratio, config.max_area_ratio, "max", "mask_area_too_large")
    check(
        "min_segmentation_confidence",
        segmentation_confidence,
        config.min_segmentation_confidence,
        "min",
        "low_segmentation_confidence",
    )
    check(
        "max_border_contact_ratio",
        border_contact_ratio,
        config.max_border_contact_ratio,
        "max",
        "excessive_border_contact",
    )
    check(
        "min_largest_component_ratio",
        largest_component_ratio,
        config.min_largest_component_ratio,
        "min",
        "fragmented_mask",
    )
    check(
        "min_temporal_warped_iou",
        temporal_warped_iou,
        config.min_temporal_warped_iou,
        "min",
        "low_temporal_warped_iou",
    )
    check(
        "max_centroid_jump",
        normalized_centroid_jump,
        config.max_centroid_jump,
        "max",
        "excessive_centroid_jump",
    )
    check(
        "max_relative_area_change",
        None if relative_area_change is None else abs(relative_area_change),
        config.max_relative_area_change,
        "max",
        "excessive_area_change",
    )
    check(
        "min_lumen_contrast",
        lumen_surrounding_contrast,
        config.min_lumen_contrast,
        "min",
        "low_lumen_contrast",
    )
    check(
        "max_boundary_entropy",
        mean_boundary_entropy,
        config.max_boundary_entropy,
        "max",
        "high_boundary_entropy",
    )
    check(
        "min_temporal_stability_score",
        temporal_stability_score,
        config.min_temporal_stability_score,
        "min",
        "low_temporal_stability",
    )
    check(
        "min_control_quality_score",
        control_quality_score,
        config.min_control_quality_score,
        "min",
        "low_control_quality",
    )

    if config.require_temporal and temporal_warped_iou is None:
        reasons.append("low_temporal_warped_iou")
        checked.append("require_temporal")

    # Deduplicate while preserving the order in which criteria failed.
    unique_reasons = list(dict.fromkeys(reasons))
    return ValidityResult(valid=not unique_reasons, reasons=unique_reasons, checked=checked, skipped=skipped)
