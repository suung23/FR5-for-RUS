"""The :class:`ControlState` record handed to a downstream controller.

Coordinate convention (used consistently across this repository)::

    (0, 0) is the top-left image corner
    x increases to the right
    y increases downward
    the image centre is (0.5, 0.5) in normalized coordinates

    center_error_x = centroid_x_normalized - 0.5
    center_error_y = centroid_y_normalized - 0.5

so a *positive* ``center_error_x`` means the bladder is to the **right** of the
image centre, and a *positive* ``center_error_y`` means it is **below** it.

This structure deliberately contains **no robot commands**: no velocities,
forces, positions, joint targets or impedance parameters. It reports what was
observed and how much that observation can be trusted; deciding what to do with
it is the controller's responsibility.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = ["ControlState", "BoundingBox", "COORDINATE_CONVENTION"]

COORDINATE_CONVENTION = (
    "origin=top-left; x increases right; y increases down; normalized image "
    "centre=(0.5, 0.5); center_error = normalized_centroid - 0.5"
)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box of the segmented lumen, in pixels."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        """Box width in pixels."""
        return max(0, self.x_max - self.x_min + 1)

    @property
    def height(self) -> int:
        """Box height in pixels."""
        return max(0, self.y_max - self.y_min + 1)

    def to_list(self) -> list[int]:
        """Return ``[x_min, y_min, x_max, y_max]``."""
        return [int(self.x_min), int(self.y_min), int(self.x_max), int(self.y_max)]


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars, arrays and non-finite floats into JSON-safe values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, BoundingBox):
        return value.to_list()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


@dataclass
class ControlState:
    """Per-frame perception output for a downstream robotic-ultrasound controller.

    Fields fall into six groups: provenance, latency, raw output, geometry,
    image-derived quality, and temporal stability. Every scalar is
    JSON-serialisable via :meth:`to_dict`.

    Note:
        ``valid_for_control=False`` means *the current observation should not be
        treated as a reliable new control measurement*. It does not say the
        previous state is still valid, and it is not a command to stop.
    """

    # -- provenance -------------------------------------------------------
    frame_id: str = ""
    timestamp: float = 0.0
    model_version: str = "unknown"
    checkpoint_id: str = "unknown"

    # -- latency (milliseconds) -------------------------------------------
    preprocessing_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    postprocessing_latency_ms: float = 0.0
    control_feature_latency_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0

    # -- raw output --------------------------------------------------------
    mask_threshold: float = 0.5
    binary_mask: Optional[np.ndarray] = None
    probability_map: Optional[np.ndarray] = None

    # -- geometry ----------------------------------------------------------
    centroid_x_px: Optional[float] = None
    centroid_y_px: Optional[float] = None
    centroid_x_normalized: Optional[float] = None
    centroid_y_normalized: Optional[float] = None
    center_error_x: Optional[float] = None
    center_error_y: Optional[float] = None
    mask_area_px: int = 0
    mask_area_ratio: float = 0.0
    bounding_box: Optional[BoundingBox] = None
    major_axis_length: Optional[float] = None
    minor_axis_length: Optional[float] = None
    orientation_degrees: Optional[float] = None

    # -- shape / image quality --------------------------------------------
    largest_component_ratio: float = 0.0
    border_contact_ratio: float = 0.0
    mean_boundary_entropy: float = 0.0
    segmentation_confidence: float = 0.0
    lumen_mean_intensity: Optional[float] = None
    surrounding_ring_mean_intensity: Optional[float] = None
    lumen_surrounding_contrast: Optional[float] = None

    # -- temporal stability ------------------------------------------------
    temporal_warped_iou: Optional[float] = None
    temporal_warped_dice: Optional[float] = None
    normalized_centroid_jump: Optional[float] = None
    relative_area_change: Optional[float] = None
    temporal_stability_score: Optional[float] = None

    # -- decision ----------------------------------------------------------
    control_quality_score: float = 0.0
    quality_components: dict[str, float] = field(default_factory=dict)
    valid_for_control: bool = False
    rejection_reasons: list[str] = field(default_factory=list)

    # -- misc --------------------------------------------------------------
    image_height: int = 0
    image_width: int = 0
    coordinate_convention: str = COORDINATE_CONVENTION
    metadata: dict[str, Any] = field(default_factory=dict)

    #: Keys never written by :meth:`to_dict` unless explicitly requested.
    HEAVY_FIELDS = ("probability_map", "binary_mask")

    def to_dict(
        self,
        include_probability_map: bool = False,
        include_binary_mask: bool = False,
    ) -> dict[str, Any]:
        """Return a JSON-compatible dictionary.

        The dense arrays are excluded by default: writing a full probability map
        per frame to a JSONL log would dominate the file and slow the real-time
        loop. Their shape and area are always reported so a consumer can tell
        that a mask existed.

        Args:
            include_probability_map: Serialise the probability map as nested lists.
            include_binary_mask: Serialise the binary mask as nested lists of 0/1.
        """
        raw = asdict(self)
        for key in self.HEAVY_FIELDS:
            raw.pop(key, None)

        payload = {key: _json_safe(value) for key, value in raw.items()}
        payload["bounding_box"] = self.bounding_box.to_list() if self.bounding_box else None
        payload["has_binary_mask"] = self.binary_mask is not None
        payload["has_probability_map"] = self.probability_map is not None

        if include_binary_mask and self.binary_mask is not None:
            payload["binary_mask"] = self.binary_mask.astype(np.uint8).tolist()
        if include_probability_map and self.probability_map is not None:
            payload["probability_map"] = np.round(
                self.probability_map.astype(np.float64), 4
            ).tolist()
        return payload

    def to_json(self, **kwargs: Any) -> str:
        """Return the state as a single-line JSON string (JSONL-friendly)."""
        return json.dumps(self.to_dict(**kwargs), separators=(",", ":"), sort_keys=False)

    def flat_record(self) -> dict[str, Any]:
        """Return a flat, CSV-friendly record with scalars only."""
        payload = self.to_dict()
        record: dict[str, Any] = {}
        for key, value in payload.items():
            if key in ("quality_components", "metadata"):
                continue
            if key == "rejection_reasons":
                record[key] = "|".join(self.rejection_reasons)
            elif key == "bounding_box":
                if value:
                    record.update(
                        {
                            "bbox_x_min": value[0],
                            "bbox_y_min": value[1],
                            "bbox_x_max": value[2],
                            "bbox_y_max": value[3],
                        }
                    )
                else:
                    record.update(
                        {"bbox_x_min": None, "bbox_y_min": None, "bbox_x_max": None, "bbox_y_max": None}
                    )
            elif isinstance(value, (list, dict)):
                continue
            else:
                record[key] = value
        # Emit every possible quality component so the column set is identical
        # on every frame, including the first one (which has no temporal terms).
        from .quality import QUALITY_COMPONENT_NAMES

        for name in QUALITY_COMPONENT_NAMES:
            record[f"quality_{name}"] = self.quality_components.get(name)
        return record
