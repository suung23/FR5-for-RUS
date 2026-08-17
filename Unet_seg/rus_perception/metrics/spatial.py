"""Spatial segmentation metrics with per-frame, per-sequence and per-patient reports.

All metrics operate on **binary** masks, i.e. after thresholding. Losses use soft
probabilities; metrics report what a controller would actually receive.

Metrics that are undefined for a given frame (HD95 when either mask is empty,
precision when nothing was predicted) return ``None`` and are excluded from the
aggregate rather than being silently reported as zero.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "FrameMetrics",
    "confusion_counts",
    "compute_frame_metrics",
    "hausdorff_95",
    "aggregate_metrics",
    "MetricReport",
    "build_metric_report",
]

_EPS = 1e-8


@dataclass
class FrameMetrics:
    """Spatial metrics for a single frame."""

    frame_id: str
    patient_id: str
    sequence_id: str
    frame_index: int
    dice: Optional[float]
    iou: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    sensitivity: Optional[float]
    specificity: Optional[float]
    hd95: Optional[float]
    predicted_area_px: int
    target_area_px: int
    num_components: int
    component_failure: bool
    empty_prediction_on_positive: bool
    false_positive_on_empty: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "frame_id": self.frame_id,
            "patient_id": self.patient_id,
            "sequence_id": self.sequence_id,
            "frame_index": self.frame_index,
            "dice": self.dice,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "hd95": self.hd95,
            "predicted_area_px": self.predicted_area_px,
            "target_area_px": self.target_area_px,
            "num_components": self.num_components,
            "component_failure": self.component_failure,
            "empty_prediction_on_positive": self.empty_prediction_on_positive,
            "false_positive_on_empty": self.false_positive_on_empty,
        }


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` for two binary masks.

    Raises:
        ValueError: If the shapes differ.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {prediction.shape} and target {target.shape} must have the same shape."
        )
    p = np.asarray(prediction) > 0
    t = np.asarray(target) > 0
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(~p & ~t))
    return tp, fp, fn, tn


def _surface(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide inner boundary of a binary mask."""
    import cv2

    binary = (mask > 0).astype(np.uint8)
    # borderValue=0 makes the image edge count as a surface, matching the
    # convention used by rus_perception.control.features.border_contact_ratio. OpenCV's
    # default would treat the outside as foreground and omit those pixels.
    eroded = cv2.erode(
        binary,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (binary - eroded).astype(np.uint8)


def hausdorff_95(
    prediction: np.ndarray, target: np.ndarray, spacing: float = 1.0
) -> Optional[float]:
    """Symmetric 95th-percentile Hausdorff distance between two mask surfaces.

    Args:
        prediction: Binary predicted mask.
        target: Binary ground-truth mask.
        spacing: Physical size of one pixel. The default of 1.0 reports the
            distance in **pixels**; supply the real pixel spacing to obtain
            millimetres.

    Returns:
        The distance, or ``None`` when either mask is empty (HD95 is undefined
        without two surfaces to compare).
    """
    import cv2

    prediction = (np.asarray(prediction) > 0).astype(np.uint8)
    target = (np.asarray(target) > 0).astype(np.uint8)
    if prediction.sum() == 0 or target.sum() == 0:
        return None

    surface_p = _surface(prediction)
    surface_t = _surface(target)
    if surface_p.sum() == 0 or surface_t.sum() == 0:
        return None

    # distanceTransform measures the distance to the nearest zero pixel, so the
    # surface must be inverted to become the set of "sources".
    distance_to_t = cv2.distanceTransform((1 - surface_t).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_p = cv2.distanceTransform((1 - surface_p).astype(np.uint8), cv2.DIST_L2, 3)

    d_pt = distance_to_t[surface_p > 0]
    d_tp = distance_to_p[surface_t > 0]
    if d_pt.size == 0 or d_tp.size == 0:
        return None
    return float(max(np.percentile(d_pt, 95), np.percentile(d_tp, 95)) * spacing)


def compute_frame_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    frame_id: str = "",
    patient_id: str = "",
    sequence_id: str = "",
    frame_index: int = 0,
    compute_hd95: bool = True,
    spacing: float = 1.0,
    connectivity: int = 8,
) -> FrameMetrics:
    """Compute all spatial metrics for one frame.

    Args:
        prediction: Binary predicted mask ``H x W``.
        target: Binary ground-truth mask ``H x W``.
        frame_id, patient_id, sequence_id, frame_index: Provenance.
        compute_hd95: Whether to evaluate HD95 (the most expensive metric).
        spacing: Pixel spacing passed to :func:`hausdorff_95`.
        connectivity: Connected-component connectivity for the fragmentation check.

    Returns:
        A :class:`FrameMetrics`.
    """
    import cv2

    prediction = (np.asarray(prediction) > 0).astype(np.uint8)
    target = (np.asarray(target) > 0).astype(np.uint8)
    tp, fp, fn, tn = confusion_counts(prediction, target)

    predicted_area = int(prediction.sum())
    target_area = int(target.sum())

    dice = (
        (2.0 * tp) / (2.0 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else (1.0 if target_area == 0 and predicted_area == 0 else None)
    )
    union = tp + fp + fn
    iou = (
        tp / union if union > 0 else (1.0 if target_area == 0 and predicted_area == 0 else None)
    )
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None

    num_components = 0
    if predicted_area > 0:
        count, _, _, _ = cv2.connectedComponentsWithStats(prediction, connectivity=connectivity)
        num_components = max(0, count - 1)

    return FrameMetrics(
        frame_id=frame_id,
        patient_id=patient_id,
        sequence_id=sequence_id,
        frame_index=int(frame_index),
        dice=dice,
        iou=iou,
        precision=precision,
        recall=recall,
        sensitivity=recall,  # sensitivity is recall; reported under both names
        specificity=specificity,
        hd95=hausdorff_95(prediction, target, spacing) if compute_hd95 else None,
        predicted_area_px=predicted_area,
        target_area_px=target_area,
        num_components=num_components,
        # A controller expects exactly one bladder; more than one component is a
        # failure of the postprocessing contract even if Dice looks acceptable.
        component_failure=num_components > 1,
        empty_prediction_on_positive=(target_area > 0 and predicted_area == 0),
        false_positive_on_empty=(target_area == 0 and predicted_area > 0),
    )


def _summarise(values: Iterable[Optional[float]]) -> dict[str, Optional[float]]:
    """Mean, std, median and count of the defined values in ``values``."""
    defined = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not defined:
        return {"mean": None, "std": None, "median": None, "count": 0}
    array = np.asarray(defined, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "count": int(array.size),
    }


_METRIC_NAMES = ("dice", "iou", "precision", "recall", "sensitivity", "specificity", "hd95")


def aggregate_metrics(frames: Sequence[FrameMetrics]) -> dict[str, Any]:
    """Aggregate a list of frame metrics into summary statistics.

    Returns:
        A dictionary with a ``{mean, std, median, count}`` block per metric plus
        the failure rates (fragmentation, missed bladder, false positive on
        empty ground truth).
    """
    if not frames:
        return {"num_frames": 0}

    summary: dict[str, Any] = {"num_frames": len(frames)}
    for name in _METRIC_NAMES:
        summary[name] = _summarise(getattr(frame, name) for frame in frames)

    total = float(len(frames))
    positives = [f for f in frames if f.target_area_px > 0]
    empties = [f for f in frames if f.target_area_px == 0]
    summary["connected_component_failure_rate"] = (
        sum(1 for f in frames if f.component_failure) / total
    )
    summary["missed_bladder_rate"] = (
        sum(1 for f in positives if f.empty_prediction_on_positive) / len(positives)
        if positives
        else None
    )
    summary["empty_mask_false_positive_rate"] = (
        sum(1 for f in empties if f.false_positive_on_empty) / len(empties) if empties else None
    )
    summary["num_frames_with_positive_ground_truth"] = len(positives)
    summary["num_frames_with_empty_ground_truth"] = len(empties)
    return summary


@dataclass
class MetricReport:
    """Spatial metrics reported at frame, sequence, patient and dataset level."""

    overall: dict[str, Any]
    per_patient: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_sequence: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_frame: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "overall": self.overall,
            "per_patient": self.per_patient,
            "per_sequence": self.per_sequence,
            "per_frame": self.per_frame,
        }

    def to_text(self, max_patients: int = 20) -> str:
        """Render a compact text summary."""
        lines = ["=== Spatial metrics (overall) ==="]
        for name in _METRIC_NAMES:
            block = self.overall.get(name) or {}
            if block.get("mean") is None:
                lines.append(f"{name:<12}: n/a (no defined values)")
            else:
                lines.append(
                    f"{name:<12}: {block['mean']:.4f} +/- {block['std']:.4f} "
                    f"(median {block['median']:.4f}, n={block['count']})"
                )
        for key in (
            "connected_component_failure_rate",
            "missed_bladder_rate",
            "empty_mask_false_positive_rate",
        ):
            value = self.overall.get(key)
            lines.append(f"{key:<34}: {'n/a' if value is None else f'{value:.4f}'}")
        lines.append(f"frames: {self.overall.get('num_frames', 0)}")

        if self.per_patient:
            lines.append("")
            lines.append("=== Per patient (Dice mean) ===")
            for patient in sorted(self.per_patient)[:max_patients]:
                block = self.per_patient[patient].get("dice") or {}
                mean = block.get("mean")
                lines.append(
                    f"  {patient:<16}: {'n/a' if mean is None else f'{mean:.4f}'} "
                    f"(frames={self.per_patient[patient].get('num_frames', 0)})"
                )
            if len(self.per_patient) > max_patients:
                lines.append(f"  ... and {len(self.per_patient) - max_patients} more patients")
        return "\n".join(lines)


def build_metric_report(frames: Sequence[FrameMetrics]) -> MetricReport:
    """Group frame metrics by patient and sequence and aggregate each level."""
    by_patient: dict[str, list[FrameMetrics]] = defaultdict(list)
    by_sequence: dict[str, list[FrameMetrics]] = defaultdict(list)
    for frame in frames:
        by_patient[frame.patient_id].append(frame)
        by_sequence[f"{frame.patient_id}/{frame.sequence_id}"].append(frame)

    return MetricReport(
        overall=aggregate_metrics(frames),
        per_patient={key: aggregate_metrics(value) for key, value in by_patient.items()},
        per_sequence={key: aggregate_metrics(value) for key, value in by_sequence.items()},
        per_frame=[frame.to_dict() for frame in frames],
    )
