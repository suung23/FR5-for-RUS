"""Transparent, configurable control-quality heuristic.

The score answers one question: *how usable is this frame's segmentation as a
new control measurement?* It is a weighted mean of normalized sub-scores, each
in ``[0, 1]``:

.. math::

    Q_{\\mathrm{arith}} = \\frac{\\sum_i w_i s_i}{\\sum_i w_i}
    \\qquad
    Q_{\\mathrm{geom}} = \\exp\\!\\left(\\frac{\\sum_i w_i \\ln s_i}{\\sum_i w_i}\\right)

Every sub-score and every weight is logged, so a value can always be traced back
to its components.

Choosing the aggregation
------------------------
The arithmetic mean lets a majority of near-1 sub-scores hide one that has
collapsed: with eight terms, a single sub-score falling from 1.0 to 0.0 moves
``Q`` by at most its weight share. That is the wrong shape for a gate, where one
broken measurement should be disqualifying regardless of how ordinary the rest
of the frame looks.

The geometric mean has the opposite behaviour -- it is dominated by its smallest
term and reaches 0 whenever any term does -- which also means it needs a floor
(``geometric_floor``) so that one legitimately-zero sub-score does not erase all
information in the others.

Warning:
    This score is a **heuristic**, not a clinically validated measure of
    anatomical correctness or image quality. The weights are experiment
    defaults chosen for plausibility, not from a validation study. Do not
    present it as a diagnostic quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "AGGREGATIONS",
    "QualityConfig",
    "QualityResult",
    "compute_control_quality",
    "QUALITY_COMPONENT_NAMES",
]

#: How the sub-scores are combined into one number.
AGGREGATIONS: tuple[str, ...] = ("arithmetic", "geometric")

#: Every sub-score name the quality heuristic can emit, in a fixed order.
#: Consumers (e.g. the CSV writer) rely on this so their column set stays stable
#: even on frames where the temporal components are unavailable.
QUALITY_COMPONENT_NAMES: tuple[str, ...] = (
    "segmentation_confidence",
    "boundary_sharpness",
    "lumen_centering",
    "mask_completeness",
    "lumen_contrast",
    "border_penalty",
    "component_quality",
    "temporal_iou",
    "centroid_stability",
    "area_stability",
)


@dataclass
class QualityConfig:
    """Weights and normalisation scales of the control-quality score.

    Attributes:
        weights: Weight per sub-score. A weight of 0 removes the component.
        target_area_ratio: Mask area ratio considered ideal for control; the
            completeness sub-score peaks there.
        area_tolerance: Half-width of the plateau around ``target_area_ratio``
            in which completeness stays near 1.
        contrast_reference: Lumen/surrounding contrast that scores 1.0.
        boundary_entropy_scale: Mean boundary entropy that scores ``exp(-1)``.
            Lower entropy means a crisper lumen edge, which is the image-side
            evidence that the boundary was found rather than guessed.
        centroid_offset_scale: Mask-to-darkness centroid offset, in mask radii,
            that scores ``exp(-1)``.
        centroid_jump_scale: Normalized centroid jump that scores ``exp(-1)``.
        area_change_scale: Relative area change that scores ``exp(-1)``.
        border_penalty_scale: Border-contact ratio that scores ``exp(-1)``.
        penalise_overfill: Whether a mask *larger* than the target band lowers
            ``mask_completeness``. The deployment target is HoLEP, where
            irrigation holds the bladder hydro-extended, so an unusually large
            lumen is the intended operating point rather than a fault; setting
            this to ``False`` penalises only under-filling. ``True`` keeps the
            original two-sided behaviour.
        aggregation: ``"arithmetic"`` (default, unchanged behaviour) or
            ``"geometric"``. See the module docstring.
        geometric_floor: Sub-scores are clamped up to this value before the
            geometric mean, so a single zero term cannot annihilate the score.
            Ignored by the arithmetic mean.
    """

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "segmentation_confidence": 1.0,
            # The two image-side terms default to 0 so that adding them does not
            # silently redefine Q for every configuration written before they
            # existed. A profile that wants them says so.
            "boundary_sharpness": 0.0,
            "lumen_centering": 0.0,
            "mask_completeness": 1.0,
            "lumen_contrast": 0.5,
            "border_penalty": 1.0,
            "component_quality": 1.0,
            "temporal_iou": 1.0,
            "centroid_stability": 0.5,
            "area_stability": 0.5,
        }
    )
    target_area_ratio: float = 0.15
    area_tolerance: float = 0.10
    contrast_reference: float = 0.35
    boundary_entropy_scale: float = 0.30
    centroid_offset_scale: float = 0.25
    centroid_jump_scale: float = 0.05
    area_change_scale: float = 0.25
    border_penalty_scale: float = 0.15
    penalise_overfill: bool = True
    aggregation: str = "arithmetic"
    geometric_floor: float = 0.05

    def __post_init__(self) -> None:
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {list(AGGREGATIONS)}, got {self.aggregation!r}."
            )
        if not 0.0 < self.geometric_floor < 1.0:
            raise ValueError(
                f"geometric_floor must be in (0, 1), got {self.geometric_floor}."
            )
        if any(value < 0 for value in self.weights.values()):
            raise ValueError(f"Quality weights must be non-negative, got {self.weights}.")
        if sum(self.weights.values()) <= 0:
            raise ValueError("At least one quality weight must be positive.")
        for name in (
            "target_area_ratio",
            "area_tolerance",
            "contrast_reference",
            "boundary_entropy_scale",
            "centroid_offset_scale",
            "centroid_jump_scale",
            "area_change_scale",
            "border_penalty_scale",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}.")

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "QualityConfig":
        """Build from a config mapping, rejecting unknown keys."""
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown quality config key(s): {sorted(unknown)}. Known keys: {sorted(known)}"
            )
        if "weights" in data and data["weights"] is not None:
            base = cls().weights
            unknown_weights = set(data["weights"]) - set(base)
            if unknown_weights:
                raise ValueError(
                    f"Unknown quality weight(s): {sorted(unknown_weights)}. "
                    f"Known: {sorted(base)}"
                )
            base.update({k: float(v) for k, v in data["weights"].items()})
            data["weights"] = base
        return cls(**data)


@dataclass
class QualityResult:
    """The aggregate score together with every sub-score that produced it."""

    score: float
    components: dict[str, float]
    weights: dict[str, float]
    aggregation: str = "arithmetic"

    def explain(self) -> str:
        """Render the weighted-mean computation as readable text."""
        lines = [f"control_quality_score = {self.score:.4f} "
                 f"(weighted {self.aggregation} mean of:)"]
        for name in sorted(self.components):
            weight = self.weights.get(name, 0.0)
            lines.append(f"  {name:<26} score={self.components[name]:.4f} weight={weight:.3f}")
        return "\n".join(lines)


def _decay(value: float, scale: float) -> float:
    """``exp(-value / scale)`` clamped to ``[0, 1]``; larger value -> lower score."""
    return float(min(1.0, max(0.0, math.exp(-max(0.0, value) / scale))))


def _plateau(value: float, target: float, tolerance: float,
             penalise_above: bool = True) -> float:
    """1 inside ``target +/- tolerance``, decaying smoothly outside it.

    With ``penalise_above=False`` the upper side is flat: anything at or above
    the band scores 1.0 and only under-filling is penalised. See
    ``QualityConfig.penalise_overfill`` for why that is the right shape for a
    hydro-extended bladder.
    """
    if not penalise_above and value >= target:
        return 1.0
    distance = abs(value - target)
    if distance <= tolerance:
        return 1.0
    return _decay(distance - tolerance, max(tolerance, 1e-6))


def compute_control_quality(
    segmentation_confidence: float,
    mask_area_ratio: float,
    largest_component_ratio: float,
    border_contact_ratio: float,
    lumen_surrounding_contrast: Optional[float] = None,
    temporal_warped_iou: Optional[float] = None,
    normalized_centroid_jump: Optional[float] = None,
    relative_area_change: Optional[float] = None,
    config: Optional[QualityConfig] = None,
    *,
    mean_boundary_entropy: Optional[float] = None,
    lumen_centroid_offset: Optional[float] = None,
) -> QualityResult:
    """Compute the control-quality score from already-extracted features.

    Components whose inputs are unavailable (for example every temporal term on
    the first frame of a sequence) are dropped from the weighted mean rather
    than silently scored as 0, so a first frame is not penalised for having no
    predecessor.

    Args:
        segmentation_confidence: Mean binary certainty in ``[0, 1]``.
        mask_area_ratio: Foreground area divided by image area.
        largest_component_ratio: Retained-component area over raw mask area.
        border_contact_ratio: Fraction of the mask perimeter on the image border.
        lumen_surrounding_contrast: Normalized lumen/surrounding contrast.
        temporal_warped_iou: IoU against the motion-aligned previous mask.
        normalized_centroid_jump: Centroid displacement in normalized units.
        relative_area_change: Relative area change against the previous frame.
        config: Weights and scales.
        mean_boundary_entropy: Mean normalized entropy on the mask boundary.
        lumen_centroid_offset: Mask-to-darkness centroid offset, in mask radii.

    Note:
        The two image-side inputs are keyword-only and appended after ``config``
        rather than grouped with the other image features. Inserting them in
        subject order would have silently re-bound every existing positional
        call -- ``compute_control_quality(0.9, 0.15, 1.0, 0.0, 0.4, 0.9, ...)``
        would have started reading its temporal arguments as boundary entropy.

    Returns:
        A :class:`QualityResult` in ``[0, 1]``.
    """
    config = config or QualityConfig()
    components: dict[str, float] = {
        "segmentation_confidence": float(min(1.0, max(0.0, segmentation_confidence))),
        "mask_completeness": _plateau(
            float(mask_area_ratio), config.target_area_ratio, config.area_tolerance,
            penalise_above=config.penalise_overfill,
        ),
        "component_quality": float(min(1.0, max(0.0, largest_component_ratio))),
        "border_penalty": _decay(float(border_contact_ratio), config.border_penalty_scale),
    }

    if mean_boundary_entropy is not None:
        components["boundary_sharpness"] = _decay(
            float(mean_boundary_entropy), config.boundary_entropy_scale
        )
    if lumen_centroid_offset is not None:
        components["lumen_centering"] = _decay(
            float(lumen_centroid_offset), config.centroid_offset_scale
        )
    if lumen_surrounding_contrast is not None:
        components["lumen_contrast"] = float(
            min(1.0, max(0.0, lumen_surrounding_contrast / config.contrast_reference))
        )
    if temporal_warped_iou is not None:
        components["temporal_iou"] = float(min(1.0, max(0.0, temporal_warped_iou)))
    if normalized_centroid_jump is not None:
        components["centroid_stability"] = _decay(
            float(normalized_centroid_jump), config.centroid_jump_scale
        )
    if relative_area_change is not None:
        components["area_stability"] = _decay(
            abs(float(relative_area_change)), config.area_change_scale
        )

    weights = {
        name: float(config.weights.get(name, 0.0))
        for name in components
        if config.weights.get(name, 0.0) > 0
    }
    if not weights:
        return QualityResult(score=0.0, components=components, weights={},
                             aggregation=config.aggregation)

    total_weight = sum(weights.values())
    if config.aggregation == "geometric":
        floor = config.geometric_floor
        score = math.exp(
            sum(math.log(max(components[name], floor)) * weight
                for name, weight in weights.items())
            / total_weight
        )
    else:
        score = sum(components[name] * weight for name, weight in weights.items()) / total_weight
    return QualityResult(
        score=float(min(1.0, max(0.0, score))), components=components, weights=weights,
        aggregation=config.aggregation,
    )
