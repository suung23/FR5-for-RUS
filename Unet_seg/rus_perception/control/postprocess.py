"""Configurable runtime postprocessing of the raw probability map.

Every operation is optional, every threshold comes from configuration, and the
raw probability map and the raw thresholded mask are always retained alongside
the cleaned mask. Nothing here modifies the model output silently: each applied
step is recorded in :attr:`PostprocessResult.decisions`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["PostprocessConfig", "PostprocessResult", "postprocess_probability"]


@dataclass
class PostprocessConfig:
    """Postprocessing switches and thresholds.

    Attributes:
        threshold: Probability above which a pixel is foreground.
        largest_component: Keep only the largest connected component. This is
            the default because a controller needs one bladder, not several
            candidates.
        min_component_area_ratio: Components smaller than this fraction of the
            image are removed before the largest component is selected.
        fill_holes: Fill interior holes of the retained mask.
        smooth_contour: Apply a morphological open/close pair to smooth the
            contour.
        smooth_kernel: Kernel size (odd, >= 3) used by ``smooth_contour``.
        connectivity: Connected-component connectivity, 4 or 8.
    """

    threshold: float = 0.5
    largest_component: bool = True
    min_component_area_ratio: float = 0.0
    fill_holes: bool = False
    smooth_contour: bool = False
    smooth_kernel: int = 3
    connectivity: int = 8

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}.")
        if not 0.0 <= self.min_component_area_ratio <= 1.0:
            raise ValueError(
                f"min_component_area_ratio must be in [0, 1], got {self.min_component_area_ratio}."
            )
        if self.connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {self.connectivity}.")
        if self.smooth_kernel < 3 or self.smooth_kernel % 2 == 0:
            raise ValueError(
                f"smooth_kernel must be an odd integer >= 3, got {self.smooth_kernel}."
            )

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "PostprocessConfig":
        """Build from a config mapping, rejecting unknown keys."""
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown postprocess key(s): {sorted(unknown)}. Known keys: {sorted(known)}"
            )
        return cls(**data)


@dataclass
class PostprocessResult:
    """Raw and cleaned outputs plus a record of what was applied.

    Attributes:
        probability_map: The untouched ``H x W`` probability map.
        raw_mask: Thresholded mask before any cleaning.
        mask: Mask after the configured cleaning steps.
        largest_component_ratio: Area of the retained component divided by the
            area of the raw mask. ``1.0`` means nothing was discarded; a low
            value means the raw prediction was fragmented.
        num_components: Number of connected components in the raw mask.
        decisions: Ordered log of the operations actually applied.
    """

    probability_map: np.ndarray
    raw_mask: np.ndarray
    mask: np.ndarray
    largest_component_ratio: float
    num_components: int
    removed_component_pixels: int = 0
    decisions: list[str] = field(default_factory=list)


def _connected_components(mask: np.ndarray, connectivity: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Return ``(count, labels, areas)`` for the foreground of ``mask``.

    ``areas[0]`` is the background area; component labels start at 1.

    Raises:
        ImportError: If OpenCV is unavailable.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Connected-component postprocessing requires OpenCV. "
            "Install it with: pip install opencv-python-headless"
        ) from exc

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=connectivity
    )
    areas = stats[:, cv2.CC_STAT_AREA]
    return int(count), labels, areas


def postprocess_probability(
    probability_map: np.ndarray, config: Optional[PostprocessConfig] = None
) -> PostprocessResult:
    """Threshold and clean a probability map.

    Args:
        probability_map: ``H x W`` array of foreground probabilities in ``[0, 1]``.
        config: Postprocessing configuration.

    Returns:
        A :class:`PostprocessResult`; the raw probability map and raw mask are
        always preserved.

    Raises:
        ValueError: If the input is not 2-D or leaves ``[0, 1]``.
    """
    config = config or PostprocessConfig()
    probability_map = np.asarray(probability_map, dtype=np.float32)
    if probability_map.ndim != 2:
        raise ValueError(
            f"probability_map must be a 2-D H x W array, got shape {probability_map.shape}."
        )
    if probability_map.size and (probability_map.min() < -1e-6 or probability_map.max() > 1 + 1e-6):
        raise ValueError(
            "probability_map must be in [0, 1]; apply sigmoid before postprocessing."
        )

    decisions: list[str] = [f"threshold={config.threshold}"]
    raw_mask = (probability_map >= config.threshold).astype(np.uint8)
    mask = raw_mask.copy()
    raw_area = int(raw_mask.sum())

    num_components = 0
    largest_ratio = 0.0
    removed = 0

    if raw_area > 0:
        count, labels, areas = _connected_components(raw_mask, config.connectivity)
        num_components = max(0, count - 1)
        foreground_areas = areas[1:] if count > 1 else np.array([], dtype=np.int64)

        if config.min_component_area_ratio > 0 and foreground_areas.size:
            min_area = config.min_component_area_ratio * probability_map.size
            small = [i + 1 for i, area in enumerate(foreground_areas) if area < min_area]
            if small:
                drop = np.isin(labels, small)
                removed += int(drop.sum())
                mask[drop] = 0
                decisions.append(
                    f"removed {len(small)} component(s) smaller than "
                    f"{config.min_component_area_ratio:.4f} of the image"
                )

        if config.largest_component and mask.sum() > 0:
            count2, labels2, areas2 = _connected_components(mask, config.connectivity)
            if count2 > 1:
                foreground2 = areas2[1:]
                best = int(np.argmax(foreground2)) + 1
                keep = labels2 == best
                removed += int(mask.sum()) - int(keep.sum())
                mask = keep.astype(np.uint8)
                decisions.append(f"kept largest of {count2 - 1} component(s)")

        largest_ratio = float(mask.sum()) / float(raw_area) if raw_area else 0.0

    if config.fill_holes and mask.sum() > 0:
        mask = _fill_holes(mask, config.connectivity)
        decisions.append("filled interior holes")

    if config.smooth_contour and mask.sum() > 0:
        mask = _smooth_contour(mask, config.smooth_kernel)
        decisions.append(f"smoothed contour (kernel={config.smooth_kernel})")

    return PostprocessResult(
        probability_map=probability_map,
        raw_mask=raw_mask.astype(np.uint8),
        mask=mask.astype(np.uint8),
        largest_component_ratio=largest_ratio,
        num_components=num_components,
        removed_component_pixels=removed,
        decisions=decisions,
    )


def _fill_holes(mask: np.ndarray, connectivity: int) -> np.ndarray:
    """Fill interior holes by flood-filling the background from the border."""
    import cv2

    inverted = (mask == 0).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(inverted, connectivity=connectivity)
    if count <= 1:
        return mask
    border_labels = set(labels[0, :].tolist()) | set(labels[-1, :].tolist())
    border_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    border_labels.discard(0)
    holes = (inverted == 1) & (~np.isin(labels, list(border_labels)))
    filled = mask.copy()
    filled[holes] = 1
    return filled


def _smooth_contour(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Morphological open followed by close, to remove spurs and pinholes."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed.astype(np.uint8)
