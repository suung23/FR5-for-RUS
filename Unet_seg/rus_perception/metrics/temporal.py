"""Temporal-stability metrics over chronologically ordered sequences.

These metrics must be computed on sequences **in acquisition order, without
shuffling**; shuffling destroys exactly the property being measured.

A critical caveat, repeated in the README: temporal stability is *not* evidence
of anatomical correctness. A model that confidently and consistently segments
the wrong structure scores perfectly here. Whenever ground truth is available,
:func:`classify_sequence_frames` separates

* ``stable_accurate``   -- consistent over time *and* correct,
* ``stable_inaccurate`` -- consistent over time but wrong,
* ``unstable``          -- inconsistent over time,

so the two properties are never conflated.

Temporal-consistency definition (v2, single source of truth)
------------------------------------------------------------
A *transition* is a consecutive frame pair ``(t-1, t)``. The previous mask is
always compared after motion compensation, i.e. against ``W(M_{t-1})``.
Transitions are scored as follows:

============================  =========================  ==================
current / motion-aligned prev  kind                       contributes
============================  =========================  ==================
non-empty / non-empty          ``scored``                 its actual Dice/IoU
non-empty / empty              ``onset``                  0.0
empty / non-empty              ``offset``                 0.0
empty / empty                  ``absent``                 **excluded**
============================  =========================  ==================

``warped_temporal_dice`` is the mean over every transition except ``absent``,
which is reported separately as ``absent_transition_rate``. Both-empty
transitions are excluded rather than scored 1.0 because a model that emits
nothing twice in a row is not consistent, it is silent -- crediting that as
perfect stability let a 61%-dropout sequence report 0.946 in an earlier
revision of this file. A sequence with no scorable transition reports ``None``,
never 1.0.

Note the asymmetry with :func:`~rus_perception.metrics.segmentation`'s
frame-level accuracy, where two empty masks legitimately are a perfect match:
there the reference is ground truth, here it is the model's own previous
output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "SCORED",
    "ONSET",
    "OFFSET",
    "ABSENT",
    "transition_kind",
    "TemporalFrameRecord",
    "TemporalSequenceMetrics",
    "compute_sequence_temporal_metrics",
    "classify_sequence_frames",
    "longest_true_run",
    "aggregate_temporal_metrics",
]

_EPS = 1e-8

SCORED = "scored"
ONSET = "onset"
OFFSET = "offset"
ABSENT = "absent"


def transition_kind(current: np.ndarray, warped_previous: np.ndarray) -> str:
    """Classify a transition as ``scored``, ``onset``, ``offset`` or ``absent``."""
    has_current = int(np.count_nonzero(current)) > 0
    has_previous = int(np.count_nonzero(warped_previous)) > 0
    if has_current and has_previous:
        return SCORED
    if has_current:
        return ONSET
    if has_previous:
        return OFFSET
    return ABSENT


@dataclass
class TemporalFrameRecord:
    """One frame's temporal evidence within a sequence.

    Attributes:
        frame_id: Frame identifier.
        mask: Binary predicted mask for this frame.
        probability_map: Optional probability map, for the reliability-weighted
            probability difference.
        warped_previous_mask: The previous frame's mask aligned into this
            frame's geometry. ``None`` for the first frame.
        warped_previous_probability: Motion-aligned previous probability map.
        reliability: ``H x W`` reliability map in ``[0, 1]``.
        centroid: ``(x, y)`` normalized centroid, or ``None`` for an empty mask.
        previous_centroid: The motion-aligned previous centroid.
        area_ratio: Foreground area ratio of this frame.
        previous_area_ratio: Motion-aligned previous area ratio.
        valid_for_control: Output of the validity gate.
        target: Optional binary ground-truth mask.
        motion_compensated: Whether ``warped_previous_mask`` was actually warped
            by a flow field. ``False`` means the previous mask was reused
            unchanged, and the sequence metrics say so rather than claiming a
            compensation that never happened.
    """

    frame_id: str
    mask: np.ndarray
    probability_map: Optional[np.ndarray] = None
    warped_previous_mask: Optional[np.ndarray] = None
    warped_previous_probability: Optional[np.ndarray] = None
    reliability: Optional[np.ndarray] = None
    centroid: Optional[tuple[float, float]] = None
    previous_centroid: Optional[tuple[float, float]] = None
    area_ratio: float = 0.0
    previous_area_ratio: Optional[float] = None
    valid_for_control: bool = True
    target: Optional[np.ndarray] = None
    motion_compensated: bool = True


def _iou_dice(a: np.ndarray, b: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """IoU and Dice between binary masks; ``(None, None)`` when both are empty.

    Returning ``None`` rather than ``1.0`` is the whole point of the v2
    definition documented at the top of this module: on a temporal axis two
    empty masks are an absent transition, not a perfectly consistent one.
    """
    a = (np.asarray(a) > 0).astype(np.float64)
    b = (np.asarray(b) > 0).astype(np.float64)
    intersection = float((a * b).sum())
    total = float(a.sum() + b.sum())
    if total == 0.0:
        return None, None
    union = total - intersection
    return (
        float(intersection / union) if union > 0 else 1.0,
        float(2.0 * intersection / total),
    )


def longest_true_run(flags: Sequence[bool]) -> int:
    """Length of the longest consecutive run of ``True`` values."""
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


@dataclass
class TemporalSequenceMetrics:
    """Temporal-stability metrics for one sequence."""

    sequence_id: str
    num_frames: int
    num_transitions: int
    warped_temporal_dice: Optional[float]
    warped_temporal_iou: Optional[float]
    num_scored_transitions: int
    num_absent_transitions: int
    absent_transition_rate: float
    track_break_rate: float
    motion_compensated: bool
    reliability_weighted_probability_difference: Optional[float]
    normalized_centroid_jitter: Optional[float]
    relative_area_jitter: Optional[float]
    mask_dropout_rate: float
    invalid_control_frame_rate: float
    longest_consecutive_invalid_interval: int
    false_positive_persistence: Optional[float]
    false_negative_persistence: Optional[float]
    stability_scores: list[float] = field(default_factory=list)
    frame_classification: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        data = {
            "sequence_id": self.sequence_id,
            "num_frames": self.num_frames,
            "num_transitions": self.num_transitions,
            "warped_temporal_dice": self.warped_temporal_dice,
            "warped_temporal_iou": self.warped_temporal_iou,
            "num_scored_transitions": self.num_scored_transitions,
            "num_absent_transitions": self.num_absent_transitions,
            "absent_transition_rate": self.absent_transition_rate,
            "track_break_rate": self.track_break_rate,
            "motion_compensated": self.motion_compensated,
            "reliability_weighted_probability_difference": (
                self.reliability_weighted_probability_difference
            ),
            "normalized_centroid_jitter": self.normalized_centroid_jitter,
            "relative_area_jitter": self.relative_area_jitter,
            "mask_dropout_rate": self.mask_dropout_rate,
            "invalid_control_frame_rate": self.invalid_control_frame_rate,
            "longest_consecutive_invalid_interval": self.longest_consecutive_invalid_interval,
            "false_positive_persistence": self.false_positive_persistence,
            "false_negative_persistence": self.false_negative_persistence,
            "frame_classification": dict(self.frame_classification),
        }
        if self.stability_scores:
            array = np.asarray(self.stability_scores, dtype=np.float64)
            data["stability_score_distribution"] = {
                "mean": float(array.mean()),
                "std": float(array.std(ddof=0)),
                "min": float(array.min()),
                "p05": float(np.percentile(array, 5)),
                "median": float(np.median(array)),
                "p95": float(np.percentile(array, 95)),
                "max": float(array.max()),
            }
        return data


def classify_sequence_frames(
    records: Sequence[TemporalFrameRecord],
    stability_iou_threshold: float = 0.7,
    accuracy_dice_threshold: float = 0.7,
) -> dict[str, int]:
    """Split frames into stable-accurate, stable-inaccurate and unstable.

    Frames without ground truth are counted under ``unknown_accuracy`` -- they
    are never assumed correct just because they are stable.

    Args:
        records: Chronologically ordered frame records.
        stability_iou_threshold: Warped IoU above which a frame counts as stable.
        accuracy_dice_threshold: Dice against ground truth above which a frame
            counts as accurate.

    Returns:
        Counts keyed by ``stable_accurate``, ``stable_inaccurate``, ``unstable``,
        ``unknown_accuracy`` and ``no_temporal_reference``.
    """
    counts = {
        "stable_accurate": 0,
        "stable_inaccurate": 0,
        "unstable": 0,
        "absent": 0,
        "unknown_accuracy": 0,
        "no_temporal_reference": 0,
    }
    for record in records:
        if record.warped_previous_mask is None:
            counts["no_temporal_reference"] += 1
            continue
        iou, _ = _iou_dice(record.mask, record.warped_previous_mask)
        if iou is None:
            counts["absent"] += 1     # nothing on either side: not stable, not unstable
            continue
        stable = iou >= stability_iou_threshold
        if record.target is None:
            counts["unknown_accuracy"] += 1
            continue
        _, dice = _iou_dice(record.mask, record.target)
        accurate = dice >= accuracy_dice_threshold
        if stable and accurate:
            counts["stable_accurate"] += 1
        elif stable and not accurate:
            counts["stable_inaccurate"] += 1
        else:
            counts["unstable"] += 1
    return counts


def compute_sequence_temporal_metrics(
    records: Sequence[TemporalFrameRecord],
    sequence_id: str = "",
    stability_iou_threshold: float = 0.7,
    accuracy_dice_threshold: float = 0.7,
) -> TemporalSequenceMetrics:
    """Compute temporal-stability metrics for one chronologically ordered sequence.

    Args:
        records: Frame records **in acquisition order**.
        sequence_id: Identifier used in the report.
        stability_iou_threshold: Threshold for the stable/unstable split.
        accuracy_dice_threshold: Threshold for the accurate/inaccurate split.

    Returns:
        A :class:`TemporalSequenceMetrics`.

    Raises:
        ValueError: If ``records`` is empty.
    """
    if not records:
        raise ValueError("compute_sequence_temporal_metrics requires at least one frame.")

    dices: list[float] = []
    ious: list[float] = []
    prob_diffs: list[float] = []
    centroid_jumps: list[float] = []
    area_changes: list[float] = []
    stability: list[float] = []

    absent = 0
    breaks = 0
    transitions = 0
    compensated = True

    for record in records:
        if record.warped_previous_mask is None:
            continue
        transitions += 1
        compensated = compensated and record.motion_compensated
        kind = transition_kind(record.mask, record.warped_previous_mask)
        if kind == ABSENT:
            absent += 1
            continue           # excluded from the mean, counted on its own axis
        if kind in (ONSET, OFFSET):
            breaks += 1
        iou, dice = _iou_dice(record.mask, record.warped_previous_mask)
        ious.append(float(iou))
        dices.append(float(dice))

        if record.probability_map is not None and record.warped_previous_probability is not None:
            difference = np.abs(
                np.asarray(record.probability_map, dtype=np.float64)
                - np.asarray(record.warped_previous_probability, dtype=np.float64)
            )
            if record.reliability is not None:
                weights = np.asarray(record.reliability, dtype=np.float64)
                total = float(weights.sum())
                prob_diffs.append(
                    float((difference * weights).sum() / total) if total > _EPS else 0.0
                )
            else:
                prob_diffs.append(float(difference.mean()))

        if record.centroid is not None and record.previous_centroid is not None:
            centroid_jumps.append(
                float(
                    math.hypot(
                        record.centroid[0] - record.previous_centroid[0],
                        record.centroid[1] - record.previous_centroid[1],
                    )
                )
            )

        if record.previous_area_ratio is not None:
            mean_area = 0.5 * (record.area_ratio + record.previous_area_ratio)
            if mean_area > _EPS:
                area_changes.append(
                    float(abs(record.area_ratio - record.previous_area_ratio) / mean_area)
                )

        terms = [iou]
        if centroid_jumps:
            terms.append(math.exp(-centroid_jumps[-1] / 0.05))
        if area_changes:
            terms.append(math.exp(-area_changes[-1] / 0.25))
        stability.append(float(sum(terms) / len(terms)))

    invalid_flags = [not record.valid_for_control for record in records]
    dropout_flags = [int(np.count_nonzero(record.mask)) == 0 for record in records]

    # Persistence: how long an error, once made, keeps being made. Measured as
    # the longest consecutive run, normalized by the sequence length.
    fp_flags = [
        record.target is not None
        and int(np.count_nonzero(record.target)) == 0
        and int(np.count_nonzero(record.mask)) > 0
        for record in records
    ]
    fn_flags = [
        record.target is not None
        and int(np.count_nonzero(record.target)) > 0
        and int(np.count_nonzero(record.mask)) == 0
        for record in records
    ]
    has_target = any(record.target is not None for record in records)

    return TemporalSequenceMetrics(
        sequence_id=sequence_id,
        num_frames=len(records),
        num_transitions=len(ious),
        warped_temporal_dice=float(np.mean(dices)) if dices else None,
        warped_temporal_iou=float(np.mean(ious)) if ious else None,
        num_scored_transitions=len(ious),
        num_absent_transitions=absent,
        absent_transition_rate=(absent / transitions) if transitions else 0.0,
        track_break_rate=(breaks / transitions) if transitions else 0.0,
        motion_compensated=compensated,
        reliability_weighted_probability_difference=(
            float(np.mean(prob_diffs)) if prob_diffs else None
        ),
        normalized_centroid_jitter=float(np.mean(centroid_jumps)) if centroid_jumps else None,
        relative_area_jitter=float(np.mean(area_changes)) if area_changes else None,
        mask_dropout_rate=float(np.mean(dropout_flags)) if records else 0.0,
        invalid_control_frame_rate=float(np.mean(invalid_flags)) if records else 0.0,
        longest_consecutive_invalid_interval=longest_true_run(invalid_flags),
        false_positive_persistence=(
            longest_true_run(fp_flags) / len(records) if has_target else None
        ),
        false_negative_persistence=(
            longest_true_run(fn_flags) / len(records) if has_target else None
        ),
        stability_scores=stability,
        frame_classification=classify_sequence_frames(
            records, stability_iou_threshold, accuracy_dice_threshold
        ),
    )


def aggregate_temporal_metrics(
    sequences: Sequence[TemporalSequenceMetrics],
) -> dict[str, Any]:
    """Aggregate per-sequence temporal metrics across a dataset.

    Sequences are weighted equally, so a single long sequence cannot dominate.
    """
    if not sequences:
        return {"num_sequences": 0}

    def mean_of(name: str) -> Optional[float]:
        values = [
            getattr(sequence, name)
            for sequence in sequences
            if getattr(sequence, name) is not None
        ]
        return float(np.mean(values)) if values else None

    classification: dict[str, int] = {}
    for sequence in sequences:
        for key, value in sequence.frame_classification.items():
            classification[key] = classification.get(key, 0) + value

    all_scores = [score for sequence in sequences for score in sequence.stability_scores]
    distribution: dict[str, float] = {}
    if all_scores:
        array = np.asarray(all_scores, dtype=np.float64)
        distribution = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "min": float(array.min()),
            "p05": float(np.percentile(array, 5)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(array.max()),
        }

    return {
        "num_sequences": len(sequences),
        "num_frames": int(sum(s.num_frames for s in sequences)),
        "warped_temporal_dice": mean_of("warped_temporal_dice"),
        "warped_temporal_iou": mean_of("warped_temporal_iou"),
        "num_scored_transitions": int(sum(s.num_scored_transitions for s in sequences)),
        "num_absent_transitions": int(sum(s.num_absent_transitions for s in sequences)),
        "absent_transition_rate": mean_of("absent_transition_rate"),
        "track_break_rate": mean_of("track_break_rate"),
        "motion_compensated": all(s.motion_compensated for s in sequences),
        "reliability_weighted_probability_difference": mean_of(
            "reliability_weighted_probability_difference"
        ),
        "normalized_centroid_jitter": mean_of("normalized_centroid_jitter"),
        "relative_area_jitter": mean_of("relative_area_jitter"),
        "mask_dropout_rate": mean_of("mask_dropout_rate"),
        "invalid_control_frame_rate": mean_of("invalid_control_frame_rate"),
        "longest_consecutive_invalid_interval": int(
            max(s.longest_consecutive_invalid_interval for s in sequences)
        ),
        "false_positive_persistence": mean_of("false_positive_persistence"),
        "false_negative_persistence": mean_of("false_negative_persistence"),
        "temporal_stability_score_distribution": distribution,
        "frame_classification": classification,
    }
