"""Extraction of control-oriented geometric and quality features.

The entry point is :func:`extract_control_state`, which turns a probability map
(plus, optionally, the source image, the previous state and the optical flow
between them) into a fully populated :class:`~rus_perception.control.state.ControlState`.

All geometry follows the convention documented in
:data:`~rus_perception.control.state.COORDINATE_CONVENTION`: origin top-left, ``x`` right,
``y`` down, image centre ``(0.5, 0.5)`` normalized.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .postprocess import PostprocessConfig, PostprocessResult, postprocess_probability
from .quality import QualityConfig, compute_control_quality
from .roi import roi_area_px
from .state import BoundingBox, ControlState
from .validity import ValidityConfig, evaluate_validity

logger = logging.getLogger(__name__)

__all__ = [
    "TemporalStabilityConfig",
    "FeatureExtractionConfig",
    "extract_control_state",
    "binary_mask_geometry",
    "segmentation_confidence_from_probability",
    "boundary_entropy",
    "border_contact_ratio",
    "lumen_contrast",
    "warp_mask_with_flow",
]

_EPS = 1e-8


@dataclass
class TemporalStabilityConfig:
    """Weights and scales of the aggregate temporal-stability score.

    The score is a weighted mean of the warped IoU, a centroid-jump decay and an
    area-change decay, all in ``[0, 1]``.
    """

    iou_weight: float = 1.0
    centroid_weight: float = 0.5
    area_weight: float = 0.5
    centroid_jump_scale: float = 0.05
    area_change_scale: float = 0.25

    def __post_init__(self) -> None:
        if min(self.iou_weight, self.centroid_weight, self.area_weight) < 0:
            raise ValueError("Temporal stability weights must be non-negative.")
        if self.iou_weight + self.centroid_weight + self.area_weight <= 0:
            raise ValueError("At least one temporal stability weight must be positive.")
        if self.centroid_jump_scale <= 0 or self.area_change_scale <= 0:
            raise ValueError("Temporal stability scales must be > 0.")


@dataclass
class FeatureExtractionConfig:
    """Everything :func:`extract_control_state` needs beyond the raw inputs."""

    postprocess: PostprocessConfig = None  # type: ignore[assignment]
    quality: QualityConfig = None  # type: ignore[assignment]
    validity: ValidityConfig = None  # type: ignore[assignment]
    stability: TemporalStabilityConfig = None  # type: ignore[assignment]
    boundary_band_px: int = 2
    surrounding_ring_px: int = 6
    keep_probability_map: bool = True
    keep_binary_mask: bool = True

    def __post_init__(self) -> None:
        self.postprocess = self.postprocess or PostprocessConfig()
        self.quality = self.quality or QualityConfig()
        self.validity = self.validity or ValidityConfig()
        self.stability = self.stability or TemporalStabilityConfig()
        if self.boundary_band_px < 1:
            raise ValueError(f"boundary_band_px must be >= 1, got {self.boundary_band_px}.")
        if self.surrounding_ring_px < 1:
            raise ValueError(f"surrounding_ring_px must be >= 1, got {self.surrounding_ring_px}.")

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "FeatureExtractionConfig":
        """Build from a nested config mapping.

        Note:
            ``control.raw_quality`` is intentionally *not* consumed here.
            ``Q_raw`` is computed from the raw frame, never from a probability
            map, so it has no place in this pipeline; it is built separately via
            :meth:`~rus_perception.control.raw_quality.RawQualityConfig.from_dict`.
        """
        data = dict(data or {})
        return cls(
            postprocess=PostprocessConfig.from_dict(data.get("postprocess")),
            quality=QualityConfig.from_dict(data.get("quality")),
            validity=ValidityConfig.from_dict(data.get("validity")),
            stability=TemporalStabilityConfig(**(data.get("stability") or {})),
            boundary_band_px=int(data.get("boundary_band_px", 2)),
            surrounding_ring_px=int(data.get("surrounding_ring_px", 6)),
            keep_probability_map=bool(data.get("keep_probability_map", True)),
            keep_binary_mask=bool(data.get("keep_binary_mask", True)),
        )


def binary_mask_geometry(
    mask: np.ndarray, roi_mask: Optional[np.ndarray] = None
) -> dict[str, Any]:
    """Centroid, area, bounding box and equivalent-ellipse axes of a binary mask.

    The axis lengths are those of the ellipse with the same second-order central
    moments as the region. ``orientation_degrees`` is the angle of the major axis
    measured from the ``+x`` axis, increasing towards ``+y`` (downwards), in
    ``(-90, 90]``.

    Args:
        mask: ``H x W`` binary array.
        roi_mask: Optional imaged-sector mask. When given, ``mask_area_ratio`` is
            taken over the ROI rather than the whole frame, so the frame
            grabber's crop stops being an input to the score. Centroids stay in
            frame coordinates: they address image positions, not measured area.

    Returns:
        A dictionary of geometry values; every entry is ``None`` for an empty mask.
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = mask.shape
    denominator = float(roi_area_px((height, width), roi_mask))
    area = int(mask.sum())
    empty = {
        "centroid_x_px": None,
        "centroid_y_px": None,
        "centroid_x_normalized": None,
        "centroid_y_normalized": None,
        "bounding_box": None,
        "major_axis_length": None,
        "minor_axis_length": None,
        "orientation_degrees": None,
        "mask_area_px": 0,
        "mask_area_ratio": 0.0,
    }
    if area == 0:
        return empty

    ys, xs = np.nonzero(mask)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())

    dx = xs - centroid_x
    dy = ys - centroid_y
    mu20 = float((dx * dx).mean())
    mu02 = float((dy * dy).mean())
    mu11 = float((dx * dy).mean())

    common = math.sqrt(max(0.0, 4.0 * mu11 * mu11 + (mu20 - mu02) ** 2))
    lambda_major = 0.5 * (mu20 + mu02 + common)
    lambda_minor = 0.5 * (mu20 + mu02 - common)
    major = 4.0 * math.sqrt(max(lambda_major, 0.0))
    minor = 4.0 * math.sqrt(max(lambda_minor, 0.0))
    orientation = math.degrees(0.5 * math.atan2(2.0 * mu11, mu20 - mu02))

    return {
        "centroid_x_px": centroid_x,
        "centroid_y_px": centroid_y,
        # Pixel centres map to [0, 1]; matches rus_perception.losses.temporal.soft_centroid.
        "centroid_x_normalized": (centroid_x + 0.5) / width,
        "centroid_y_normalized": (centroid_y + 0.5) / height,
        "bounding_box": BoundingBox(
            x_min=int(xs.min()), y_min=int(ys.min()), x_max=int(xs.max()), y_max=int(ys.max())
        ),
        "major_axis_length": major,
        "minor_axis_length": minor,
        "orientation_degrees": orientation,
        "mask_area_px": area,
        "mask_area_ratio": area / max(denominator, 1.0),
    }


def segmentation_confidence_from_probability(
    probability_map: np.ndarray, roi_mask: Optional[np.ndarray] = None
) -> float:
    """Mean binary certainty ``mean(|2p - 1|)`` over the measured region, in ``[0, 1]``.

    A confident model pushes probabilities towards 0 or 1 everywhere; a model
    that hedges around 0.5 scores low. This is a transparent, threshold-free
    definition -- it is *not* a calibrated probability of correctness.

    Args:
        probability_map: ``H x W`` foreground probabilities.
        roi_mask: Optional imaged-sector mask. Without it the mean runs over the
            whole frame, where the dead region outside the sector -- on which any
            model is trivially confident -- dominates the average and hides
            hedging at the lumen boundary.
    """
    probability_map = np.asarray(probability_map, dtype=np.float64)
    if probability_map.size == 0:
        return 0.0
    certainty = np.abs(2.0 * probability_map - 1.0)
    if roi_mask is not None and roi_mask.shape == certainty.shape:
        selected = certainty[np.asarray(roi_mask) > 0]
        if selected.size == 0:
            return 0.0
        return float(selected.mean())
    return float(certainty.mean())


def boundary_entropy(
    probability_map: np.ndarray,
    mask: np.ndarray,
    band_px: int = 2,
    roi_mask: Optional[np.ndarray] = None,
) -> float:
    """Mean normalized binary entropy of the probabilities on the mask boundary.

    Entropy is ``-p log p - (1-p) log(1-p)`` divided by ``log 2``, so it lies in
    ``[0, 1]``. A crisp boundary scores low; a blurred, uncertain boundary scores
    high. Returns 0.0 when the mask is empty (there is no boundary to measure).
    """
    import cv2

    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.sum() == 0:
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    band = cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)
    if roi_mask is not None and roi_mask.shape == band.shape:
        band = band * (np.asarray(roi_mask) > 0).astype(band.dtype)
    selected = np.asarray(probability_map, dtype=np.float64)[band > 0]
    if selected.size == 0:
        return 0.0
    p = np.clip(selected, _EPS, 1.0 - _EPS)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)
    return float(np.clip(entropy.mean(), 0.0, 1.0))


def border_contact_ratio(
    mask: np.ndarray, roi_mask: Optional[np.ndarray] = None
) -> float:
    """Fraction of the mask's perimeter that lies on the edge of the field of view.

    A bladder that is partly outside the field of view has a large share of its
    outline on that edge, which is exactly the situation in which a centroid is a
    biased estimate of the true anatomical centre.

    Args:
        mask: ``H x W`` binary lumen mask.
        roi_mask: Optional imaged-sector mask. **This changes which edge counts.**
            Without it the edge is the image rectangle -- but a fan inscribed in
            the frame never reaches the rectangle, so the mask can be cut clean
            in half by the sector edge and still score 0.0. With it, the edge is
            the ROI boundary, which is the edge the beam actually stops at.

    Returns:
        A value in ``[0, 1]``; 0.0 for an empty mask.
    """
    import cv2

    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.sum() == 0:
        return 0.0
    # borderValue=0 is essential: OpenCV's default erosion treats the outside of
    # the image as foreground, so a mask running along the image edge would have
    # no perimeter there -- exactly the pixels this metric needs to count.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    perimeter = mask - eroded
    total = int(perimeter.sum())
    if total == 0:
        return 0.0

    if roi_mask is not None and roi_mask.shape == mask.shape:
        # Pixels the beam does not reach, eroded inward by one step: a mask pixel
        # adjacent to that region sits on the field-of-view edge.
        outside = (np.asarray(roi_mask) == 0).astype(np.uint8)
        outside_dilated = cv2.dilate(
            outside, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1
        )
        on_border = int((perimeter * outside_dilated).sum())
    else:
        on_border = int(
            perimeter[0, :].sum()
            + perimeter[-1, :].sum()
            + perimeter[:, 0].sum()
            + perimeter[:, -1].sum()
        )
    return float(min(1.0, on_border / total))


def lumen_contrast(
    image: np.ndarray,
    mask: np.ndarray,
    ring_px: int = 6,
    roi_mask: Optional[np.ndarray] = None,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Mean lumen intensity, surrounding-ring intensity and their contrast.

    The bladder lumen is anechoic, so it should be *darker* than the surrounding
    tissue. The contrast is the normalized difference
    ``(ring - lumen) / (ring + lumen)``, which is bounded in ``[-1, 1]`` and
    positive when the lumen is darker than its surroundings.

    Args:
        image: ``H x W`` grayscale image in ``[0, 1]``.
        mask: ``H x W`` binary lumen mask.
        ring_px: Width of the surrounding ring, in pixels.
        roi_mask: Optional imaged-sector mask. The ring is clipped to it: a lumen
            near the sector edge would otherwise draw part of its "surrounding
            tissue" from the dead region outside the beam, which is black, which
            inflates the contrast in exactly the frames where the measurement is
            least trustworthy.

    Returns:
        ``(lumen_mean, ring_mean, contrast)``; all ``None`` when either region is
        empty, so that a missing measurement is never reported as zero.
    """
    import cv2

    mask = (np.asarray(mask) > 0).astype(np.uint8)
    image = np.asarray(image, dtype=np.float64)
    if mask.sum() == 0 or image.shape != mask.shape:
        return None, None, None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    ring = cv2.dilate(mask, kernel) - mask
    if roi_mask is not None and roi_mask.shape == ring.shape:
        ring = ring * (np.asarray(roi_mask) > 0).astype(ring.dtype)
    if ring.sum() == 0:
        return float(image[mask > 0].mean()), None, None

    lumen_mean = float(image[mask > 0].mean())
    ring_mean = float(image[ring > 0].mean())
    denominator = ring_mean + lumen_mean
    contrast = None if abs(denominator) < _EPS else float((ring_mean - lumen_mean) / denominator)
    return lumen_mean, ring_mean, contrast


def warp_mask_with_flow(mask: np.ndarray, flow_backward: np.ndarray) -> np.ndarray:
    """Warp a previous-frame mask into the current frame with a backward flow.

    ``flow_backward`` is an ``H x W x 2`` current-grid field pointing into the
    previous frame, matching :func:`rus_perception.flow.warp.warp_backward`.
    """
    import cv2

    height, width = mask.shape
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = grid_x + flow_backward[..., 0].astype(np.float32)
    map_y = grid_y + flow_backward[..., 1].astype(np.float32)
    warped = cv2.remap(
        mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return (warped > 0.5).astype(np.uint8)


def _iou_dice(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """IoU and Dice between two binary masks; ``(1, 1)`` when both are empty."""
    a = (a > 0).astype(np.float64)
    b = (b > 0).astype(np.float64)
    intersection = float((a * b).sum())
    sum_ab = float(a.sum() + b.sum())
    if sum_ab == 0.0:
        return 1.0, 1.0
    union = sum_ab - intersection
    iou = intersection / union if union > 0 else 1.0
    dice = 2.0 * intersection / sum_ab
    return float(iou), float(dice)


def extract_control_state(
    probability_map: np.ndarray,
    image: Optional[np.ndarray] = None,
    previous_state: Optional[ControlState] = None,
    flow: Optional[np.ndarray] = None,
    metadata: Optional[dict[str, Any]] = None,
    config: Optional[FeatureExtractionConfig] = None,
    latencies: Optional[dict[str, float]] = None,
    roi_mask: Optional[np.ndarray] = None,
) -> ControlState:
    """Turn a probability map into a complete :class:`ControlState`.

    Args:
        probability_map: ``H x W`` foreground probabilities in ``[0, 1]``.
            Apply the sigmoid before calling this function.
        image: Optional ``H x W`` source frame in ``[0, 1]``, needed for the
            lumen/surrounding contrast features.
        previous_state: The previous frame's state, for temporal features. Its
            ``binary_mask`` must be retained for the warped IoU/Dice.
        flow: Optional ``H x W x 2`` **backward** flow (current grid, pointing
            into the previous frame). Without it, the previous mask is compared
            unwarped and ``metadata["temporal_alignment"]`` records that, since
            an unaligned comparison penalises genuine motion.
        metadata: Provenance merged into the state (``frame_id``, ``timestamp``,
            ``model_version``, ``checkpoint_id`` are lifted into named fields).
        config: Postprocessing, quality, validity and stability configuration.
        latencies: Measured stage latencies in milliseconds.
        roi_mask: Optional ``H x W`` imaged-sector mask, from
            :func:`rus_perception.control.roi.build_roi_mask`. It changes the
            denominator of ``mask_area_ratio``, the support of
            ``segmentation_confidence`` and ``mean_boundary_entropy``, the ring of
            ``lumen_surrounding_contrast``, and which edge ``border_contact_ratio``
            counts. ``None`` means the whole frame, which is the previous
            behaviour exactly.

    Returns:
        A fully populated :class:`ControlState`.

    Raises:
        ValueError: If ``probability_map`` is not 2-D or leaves ``[0, 1]``.
    """
    started = time.perf_counter()
    config = config or FeatureExtractionConfig()
    metadata = dict(metadata or {})
    latencies = dict(latencies or {})

    probability_map = np.asarray(probability_map, dtype=np.float32)
    if probability_map.ndim != 2:
        raise ValueError(
            f"probability_map must be 2-D H x W, got shape {probability_map.shape}."
        )

    post: PostprocessResult = postprocess_probability(probability_map, config.postprocess)
    mask = post.mask
    height, width = probability_map.shape

    if roi_mask is not None:
        roi_mask = np.asarray(roi_mask) > 0
        if roi_mask.shape != mask.shape:
            logger.warning(
                "ROI shape %s does not match the probability map %s; ignoring the ROI. "
                "Area ratios and confidence will be taken over the whole frame.",
                roi_mask.shape,
                mask.shape,
            )
            roi_mask = None

    geometry = binary_mask_geometry(mask, roi_mask)
    confidence = segmentation_confidence_from_probability(probability_map, roi_mask)
    entropy = boundary_entropy(probability_map, mask, config.boundary_band_px, roi_mask)
    border = border_contact_ratio(mask, roi_mask)

    lumen_mean = ring_mean = contrast = None
    if image is not None:
        image_2d = np.asarray(image, dtype=np.float32)
        if image_2d.ndim == 3:
            image_2d = image_2d.mean(axis=2)
        if image_2d.shape == mask.shape:
            lumen_mean, ring_mean, contrast = lumen_contrast(
                image_2d, mask, config.surrounding_ring_px, roi_mask
            )
        else:
            logger.warning(
                "Image shape %s does not match the probability map %s; skipping "
                "intensity features.",
                image_2d.shape,
                mask.shape,
            )

    # -- temporal features ------------------------------------------------
    warped_iou = warped_dice = centroid_jump = area_change = stability = None
    alignment = "none"
    if previous_state is not None and previous_state.binary_mask is not None:
        previous_mask = np.asarray(previous_state.binary_mask, dtype=np.uint8)
        if previous_mask.shape != mask.shape:
            logger.warning(
                "Previous mask shape %s differs from the current %s; skipping temporal "
                "features.",
                previous_mask.shape,
                mask.shape,
            )
        else:
            if flow is not None:
                flow_array = np.asarray(flow, dtype=np.float32)
                if flow_array.shape[:2] != mask.shape or flow_array.shape[-1] != 2:
                    raise ValueError(
                        f"flow must be H x W x 2 matching the probability map "
                        f"{mask.shape}, got {flow_array.shape}."
                    )
                reference = warp_mask_with_flow(previous_mask, flow_array)
                alignment = "backward_flow"
            else:
                reference = previous_mask
                alignment = "unwarped"
            warped_iou, warped_dice = _iou_dice(mask, reference)

            reference_geometry = binary_mask_geometry(reference)
            if (
                geometry["centroid_x_normalized"] is not None
                and reference_geometry["centroid_x_normalized"] is not None
            ):
                centroid_jump = float(
                    math.hypot(
                        geometry["centroid_x_normalized"]
                        - reference_geometry["centroid_x_normalized"],
                        geometry["centroid_y_normalized"]
                        - reference_geometry["centroid_y_normalized"],
                    )
                )
            current_area = float(geometry["mask_area_ratio"])
            reference_area = float(reference_geometry["mask_area_ratio"])
            if current_area + reference_area > 0:
                area_change = float(
                    (current_area - reference_area) / (0.5 * (current_area + reference_area))
                )

    if warped_iou is not None:
        stability_cfg = config.stability
        terms: list[tuple[float, float]] = [(warped_iou, stability_cfg.iou_weight)]
        if centroid_jump is not None:
            terms.append(
                (
                    math.exp(-centroid_jump / stability_cfg.centroid_jump_scale),
                    stability_cfg.centroid_weight,
                )
            )
        if area_change is not None:
            terms.append(
                (
                    math.exp(-abs(area_change) / stability_cfg.area_change_scale),
                    stability_cfg.area_weight,
                )
            )
        total_weight = sum(weight for _, weight in terms)
        if total_weight > 0:
            stability = float(
                min(1.0, max(0.0, sum(value * weight for value, weight in terms) / total_weight))
            )

    # -- quality and validity ---------------------------------------------
    quality = compute_control_quality(
        segmentation_confidence=confidence,
        mask_area_ratio=float(geometry["mask_area_ratio"]),
        largest_component_ratio=post.largest_component_ratio,
        border_contact_ratio=border,
        lumen_surrounding_contrast=contrast,
        temporal_warped_iou=warped_iou,
        normalized_centroid_jump=centroid_jump,
        relative_area_change=area_change,
        config=config.quality,
    )
    validity = evaluate_validity(
        mask_area_ratio=float(geometry["mask_area_ratio"]),
        segmentation_confidence=confidence,
        border_contact_ratio=border,
        largest_component_ratio=post.largest_component_ratio,
        mask_area_px=int(geometry["mask_area_px"]),
        temporal_warped_iou=warped_iou,
        normalized_centroid_jump=centroid_jump,
        relative_area_change=area_change,
        lumen_surrounding_contrast=contrast,
        mean_boundary_entropy=entropy,
        temporal_stability_score=stability,
        control_quality_score=quality.score,
        config=config.validity,
    )

    feature_latency = (time.perf_counter() - started) * 1000.0
    latencies.setdefault("control_feature_latency_ms", feature_latency)
    end_to_end = latencies.get("end_to_end_latency_ms")
    if end_to_end is None:
        end_to_end = (
            latencies.get("preprocessing_latency_ms", 0.0)
            + latencies.get("inference_latency_ms", 0.0)
            + latencies.get("postprocessing_latency_ms", 0.0)
            + latencies["control_feature_latency_ms"]
        )

    centroid_x_norm = geometry["centroid_x_normalized"]
    centroid_y_norm = geometry["centroid_y_normalized"]

    return ControlState(
        frame_id=str(metadata.pop("frame_id", "")),
        timestamp=float(metadata.pop("timestamp", 0.0)),
        model_version=str(metadata.pop("model_version", "unknown")),
        checkpoint_id=str(metadata.pop("checkpoint_id", "unknown")),
        preprocessing_latency_ms=float(latencies.get("preprocessing_latency_ms", 0.0)),
        inference_latency_ms=float(latencies.get("inference_latency_ms", 0.0)),
        postprocessing_latency_ms=float(latencies.get("postprocessing_latency_ms", 0.0)),
        control_feature_latency_ms=float(latencies["control_feature_latency_ms"]),
        end_to_end_latency_ms=float(end_to_end),
        mask_threshold=float(config.postprocess.threshold),
        binary_mask=mask if config.keep_binary_mask else None,
        probability_map=probability_map if config.keep_probability_map else None,
        centroid_x_px=geometry["centroid_x_px"],
        centroid_y_px=geometry["centroid_y_px"],
        centroid_x_normalized=centroid_x_norm,
        centroid_y_normalized=centroid_y_norm,
        center_error_x=None if centroid_x_norm is None else centroid_x_norm - 0.5,
        center_error_y=None if centroid_y_norm is None else centroid_y_norm - 0.5,
        mask_area_px=int(geometry["mask_area_px"]),
        mask_area_ratio=float(geometry["mask_area_ratio"]),
        bounding_box=geometry["bounding_box"],
        major_axis_length=geometry["major_axis_length"],
        minor_axis_length=geometry["minor_axis_length"],
        orientation_degrees=geometry["orientation_degrees"],
        largest_component_ratio=float(post.largest_component_ratio),
        border_contact_ratio=float(border),
        mean_boundary_entropy=float(entropy),
        segmentation_confidence=float(confidence),
        lumen_mean_intensity=lumen_mean,
        surrounding_ring_mean_intensity=ring_mean,
        lumen_surrounding_contrast=contrast,
        temporal_warped_iou=warped_iou,
        temporal_warped_dice=warped_dice,
        normalized_centroid_jump=centroid_jump,
        relative_area_change=area_change,
        temporal_stability_score=stability,
        control_quality_score=float(quality.score),
        quality_components=dict(quality.components),
        valid_for_control=bool(validity.valid),
        rejection_reasons=list(validity.reasons),
        image_height=int(height),
        image_width=int(width),
        roi_mode=str(metadata.pop("roi_mode", "full" if roi_mask is None else "custom")),
        roi_area_px=roi_area_px((height, width), roi_mask),
        metadata={
            **metadata,
            "postprocess_decisions": list(post.decisions),
            "num_components": int(post.num_components),
            "temporal_alignment": alignment,
            "validity_checked": list(validity.checked),
            "validity_skipped": list(validity.skipped),
        },
    )
