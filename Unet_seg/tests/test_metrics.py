"""Metric tests: spatial accuracy, temporal stability and latency accounting."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.latency import LatencyTracker, benchmark_callable, device_report
from src.metrics.spatial import (
    aggregate_metrics,
    build_metric_report,
    compute_frame_metrics,
    confusion_counts,
    hausdorff_95,
)
from src.metrics.temporal import (
    TemporalFrameRecord,
    aggregate_temporal_metrics,
    classify_sequence_frames,
    compute_sequence_temporal_metrics,
    longest_true_run,
)


def square(size: int = 32, x: int = 8, y: int = 8, width: int = 12) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    mask[y : y + width, x : x + width] = 1
    return mask


# -- spatial ---------------------------------------------------------------
def test_perfect_prediction_scores_one() -> None:
    mask = square()
    metrics = compute_frame_metrics(mask, mask)
    assert metrics.dice == pytest.approx(1.0)
    assert metrics.iou == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.hd95 == pytest.approx(0.0)


def test_disjoint_prediction_scores_zero() -> None:
    metrics = compute_frame_metrics(square(x=2, y=2, width=6), square(x=20, y=20, width=6))
    assert metrics.dice == pytest.approx(0.0)
    assert metrics.iou == pytest.approx(0.0)
    assert metrics.hd95 > 0.0


def test_confusion_counts_sum_to_the_pixel_count() -> None:
    prediction, target = square(), square(x=10, y=10)
    tp, fp, fn, tn = confusion_counts(prediction, target)
    assert tp + fp + fn + tn == prediction.size
    with pytest.raises(ValueError, match="same shape"):
        confusion_counts(np.zeros((4, 4)), np.zeros((8, 8)))


def test_both_empty_masks_count_as_a_perfect_match() -> None:
    empty = np.zeros((16, 16), np.uint8)
    metrics = compute_frame_metrics(empty, empty)
    assert metrics.dice == pytest.approx(1.0)
    assert metrics.iou == pytest.approx(1.0)
    assert not metrics.false_positive_on_empty
    assert not metrics.empty_prediction_on_positive


def test_missed_bladder_and_false_positive_flags() -> None:
    missed = compute_frame_metrics(np.zeros((16, 16), np.uint8), square(16, 4, 4, 6))
    assert missed.empty_prediction_on_positive
    assert missed.dice == pytest.approx(0.0)

    spurious = compute_frame_metrics(square(16, 4, 4, 6), np.zeros((16, 16), np.uint8))
    assert spurious.false_positive_on_empty


def test_fragmented_prediction_is_flagged() -> None:
    prediction = np.zeros((32, 32), np.uint8)
    prediction[4:10, 4:10] = 1
    prediction[20:26, 20:26] = 1
    metrics = compute_frame_metrics(prediction, square())
    assert metrics.num_components == 2
    assert metrics.component_failure


def test_hd95_is_none_when_a_mask_is_empty() -> None:
    assert hausdorff_95(np.zeros((16, 16), np.uint8), square(16, 2, 2, 4)) is None
    assert hausdorff_95(square(16, 2, 2, 4), np.zeros((16, 16), np.uint8)) is None


def test_hd95_scales_with_pixel_spacing() -> None:
    prediction, target = square(x=8), square(x=11)
    in_pixels = hausdorff_95(prediction, target, spacing=1.0)
    in_millimetres = hausdorff_95(prediction, target, spacing=0.2)
    assert in_millimetres == pytest.approx(in_pixels * 0.2)


def test_aggregate_excludes_undefined_metrics_instead_of_scoring_them_zero() -> None:
    defined = compute_frame_metrics(square(), square(), "f1", "P1", "S1", 0)
    undefined = compute_frame_metrics(
        np.zeros((32, 32), np.uint8), np.zeros((32, 32), np.uint8), "f2", "P1", "S1", 1
    )
    summary = aggregate_metrics([defined, undefined])

    assert summary["num_frames"] == 2
    assert summary["hd95"]["count"] == 1, "HD95 is undefined for the empty pair"
    assert summary["hd95"]["mean"] == pytest.approx(0.0)
    assert summary["dice"]["count"] == 2


def test_metric_report_groups_by_patient_and_sequence() -> None:
    frames = [
        compute_frame_metrics(square(), square(), "a", "P1", "S1", 0),
        compute_frame_metrics(square(), square(x=10), "b", "P1", "S1", 1),
        compute_frame_metrics(square(), square(), "c", "P2", "S1", 0),
    ]
    report = build_metric_report(frames)
    assert set(report.per_patient) == {"P1", "P2"}
    assert set(report.per_sequence) == {"P1/S1", "P2/S1"}
    assert report.per_patient["P1"]["num_frames"] == 2
    assert len(report.per_frame) == 3
    assert "Spatial metrics" in report.to_text()
    assert "overall" in report.to_dict()


def test_aggregate_of_no_frames_is_explicit() -> None:
    assert aggregate_metrics([]) == {"num_frames": 0}


# -- temporal --------------------------------------------------------------
def build_sequence(num_frames: int = 5, drift: int = 0, valid: bool = True):
    """A synthetic sequence where each frame is compared to its aligned predecessor."""
    records = []
    previous = None
    for index in range(num_frames):
        mask = square(64, 20 + index * drift, 20, 12)
        probability = mask.astype(np.float32)
        geometry_prev = None
        area_prev = None
        if previous is not None:
            geometry_prev = ((20 + (index - 1) * drift + 6.0) / 64, 26.0 / 64)
            area_prev = float(previous.sum()) / previous.size
        records.append(
            TemporalFrameRecord(
                frame_id=f"f{index}",
                mask=mask,
                probability_map=probability,
                warped_previous_mask=previous,
                warped_previous_probability=None if previous is None else previous.astype(np.float32),
                reliability=np.ones((64, 64), np.float32),
                centroid=((20 + index * drift + 6.0) / 64, 26.0 / 64),
                previous_centroid=geometry_prev,
                area_ratio=float(mask.sum()) / mask.size,
                previous_area_ratio=area_prev,
                valid_for_control=valid,
                target=mask,
            )
        )
        previous = mask
    return records


def test_a_perfectly_stable_sequence_scores_one() -> None:
    metrics = compute_sequence_temporal_metrics(build_sequence(drift=0), "P1/S1")
    assert metrics.warped_temporal_iou == pytest.approx(1.0)
    assert metrics.warped_temporal_dice == pytest.approx(1.0)
    assert metrics.normalized_centroid_jitter == pytest.approx(0.0, abs=1e-6)
    assert metrics.mask_dropout_rate == 0.0
    assert metrics.invalid_control_frame_rate == 0.0
    assert metrics.num_transitions == 4


def test_a_drifting_sequence_scores_lower() -> None:
    stable = compute_sequence_temporal_metrics(build_sequence(drift=0)).warped_temporal_iou
    drifting = compute_sequence_temporal_metrics(build_sequence(drift=6)).warped_temporal_iou
    assert drifting < stable


def test_invalid_frames_and_the_longest_invalid_run_are_reported() -> None:
    records = build_sequence(num_frames=6)
    for index in (1, 2, 3):
        records[index].valid_for_control = False
    metrics = compute_sequence_temporal_metrics(records)
    assert metrics.invalid_control_frame_rate == pytest.approx(3 / 6)
    assert metrics.longest_consecutive_invalid_interval == 3


def test_mask_dropout_rate_counts_empty_predictions() -> None:
    records = build_sequence(num_frames=4)
    records[2].mask = np.zeros((64, 64), np.uint8)
    assert compute_sequence_temporal_metrics(records).mask_dropout_rate == pytest.approx(0.25)


def test_reliability_weighted_probability_difference_respects_the_mask() -> None:
    records = build_sequence(num_frames=2)
    records[1].probability_map = np.ones((64, 64), np.float32)
    records[1].warped_previous_probability = np.zeros((64, 64), np.float32)

    records[1].reliability = np.ones((64, 64), np.float32)
    fully_reliable = compute_sequence_temporal_metrics(
        records
    ).reliability_weighted_probability_difference

    records[1].reliability = np.zeros((64, 64), np.float32)
    records[1].reliability[:8, :8] = 1.0
    partly = compute_sequence_temporal_metrics(records).reliability_weighted_probability_difference
    assert fully_reliable == pytest.approx(1.0)
    assert partly == pytest.approx(1.0)  # the reliable region also disagrees fully


def test_stable_but_inaccurate_frames_are_reported_separately() -> None:
    """The central caveat: stability must never be read as correctness."""
    wrong = square(64, 40, 40, 10)
    records = []
    previous = None
    for index in range(4):
        records.append(
            TemporalFrameRecord(
                frame_id=f"f{index}",
                mask=wrong,
                warped_previous_mask=previous,
                area_ratio=float(wrong.sum()) / wrong.size,
                valid_for_control=True,
                target=square(64, 5, 5, 10),  # ground truth is somewhere else entirely
            )
        )
        previous = wrong
    counts = classify_sequence_frames(records)
    assert counts["stable_inaccurate"] == 3
    assert counts["stable_accurate"] == 0
    assert counts["no_temporal_reference"] == 1


def test_frames_without_ground_truth_are_never_assumed_correct() -> None:
    records = build_sequence(num_frames=3)
    for item in records:
        item.target = None
    counts = classify_sequence_frames(records)
    assert counts["unknown_accuracy"] == 2
    assert counts["stable_accurate"] == 0


def test_false_positive_and_negative_persistence() -> None:
    empty = np.zeros((32, 32), np.uint8)
    spurious = square(32, 4, 4, 6)
    records = [
        TemporalFrameRecord(f"f{i}", mask=spurious, warped_previous_mask=spurious, target=empty)
        for i in range(5)
    ]
    metrics = compute_sequence_temporal_metrics(records)
    assert metrics.false_positive_persistence == pytest.approx(1.0)
    assert metrics.false_negative_persistence == pytest.approx(0.0)


def test_longest_true_run() -> None:
    assert longest_true_run([]) == 0
    assert longest_true_run([False, False]) == 0
    assert longest_true_run([True, True, False, True]) == 2
    assert longest_true_run([True] * 5) == 5


def test_temporal_metrics_require_at_least_one_frame() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        compute_sequence_temporal_metrics([])


def test_temporal_aggregation_across_sequences() -> None:
    sequences = [
        compute_sequence_temporal_metrics(build_sequence(drift=0), "P1/S1"),
        compute_sequence_temporal_metrics(build_sequence(drift=4), "P2/S1"),
    ]
    summary = aggregate_temporal_metrics(sequences)
    assert summary["num_sequences"] == 2
    assert summary["num_frames"] == 10
    assert 0.0 <= summary["warped_temporal_iou"] <= 1.0
    assert "temporal_stability_score_distribution" in summary
    assert summary["frame_classification"]["stable_accurate"] > 0
    assert aggregate_temporal_metrics([]) == {"num_sequences": 0}


def test_sequence_metrics_serialize_to_a_dict() -> None:
    import json

    metrics = compute_sequence_temporal_metrics(build_sequence(), "P1/S1")
    json.dumps(metrics.to_dict())  # must not raise
    assert metrics.to_dict()["sequence_id"] == "P1/S1"


# -- latency ---------------------------------------------------------------
def test_latency_tracker_discards_warmup_samples() -> None:
    tracker = LatencyTracker(warmup=3)
    for value in (100.0, 100.0, 100.0, 10.0, 20.0):
        tracker.record("inference", value)
    statistics = tracker.statistics("inference")
    assert statistics.count == 2
    assert statistics.mean_ms == pytest.approx(15.0)
    assert statistics.fps == pytest.approx(1000.0 / 15.0)


def test_latency_tracker_buffers_are_bounded() -> None:
    """A long monitoring session must not grow memory without limit."""
    tracker = LatencyTracker(warmup=0, max_samples=50)
    for index in range(1000):
        tracker.record("end_to_end", float(index))
    assert len(tracker.series("end_to_end")) == 50
    assert tracker.series("end_to_end")[-1] == 999.0


def test_latency_statistics_report_percentiles() -> None:
    tracker = LatencyTracker(warmup=0)
    for value in range(1, 101):
        tracker.record("stage", float(value))
    statistics = tracker.statistics("stage")
    assert statistics.median_ms == pytest.approx(50.5)
    assert statistics.p95_ms == pytest.approx(95.05, abs=0.5)
    assert statistics.min_ms == 1.0 and statistics.max_ms == 100.0
    assert "stage" in tracker.to_text()
    assert "stages" in tracker.to_dict()


def test_empty_latency_series_is_reported_as_such() -> None:
    tracker = LatencyTracker()
    statistics = tracker.statistics("never_recorded")
    assert statistics.count == 0 and statistics.mean_ms is None
    assert "no samples" in statistics.to_text()
    assert "No latency samples" in LatencyTracker().to_text()


def test_latency_tracker_validation() -> None:
    with pytest.raises(ValueError, match="warmup must be"):
        LatencyTracker(warmup=-1)
    with pytest.raises(ValueError, match="max_samples must be"):
        LatencyTracker(max_samples=0)


def test_benchmark_callable_measures_a_known_workload() -> None:
    counter = {"calls": 0}

    def workload() -> None:
        counter["calls"] += 1

    statistics = benchmark_callable(workload, iterations=5, warmup=2, name="noop")
    assert counter["calls"] == 7
    assert statistics.count == 5
    assert statistics.mean_ms is not None and statistics.mean_ms >= 0.0
    with pytest.raises(ValueError, match="iterations must be"):
        benchmark_callable(workload, iterations=0)


def test_device_report_is_informative_on_cpu() -> None:
    report = device_report("cpu")
    assert report["device"] == "cpu"
    assert "torch" in report and "platform" in report
    assert "cuda_available" in report
    if not report["cuda_available"]:
        assert "gpu_memory_note" in report
