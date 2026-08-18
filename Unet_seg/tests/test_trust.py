"""Tests for the quality-score trust metrics.

These pin the properties the control design leans on: that a perfect score is
recognised as perfect, that a useless score is recognised as useless, that
"not measured" never becomes "zero", and that the safety budget in
``operating_point`` is a hard constraint rather than a soft preference.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rus_perception.metrics.trust import (  # noqa: E402
    FrameOutcome,
    build_trust_report,
    calibration,
    dice_to_iou,
    iou_to_dice,
    component_attribution,
    detection_report,
    force_response,
    gate_confusion,
    group_by,
    operating_point,
    pearson,
    rank_agreement,
    reason_code_report,
    risk_coverage,
    spearman,
)

FLOOR = 0.70


def frames(quality, accuracy, valid=None, **kwargs) -> list[FrameOutcome]:
    valid = [q is not None and q >= 0.5 for q in quality] if valid is None else valid
    return [
        FrameOutcome(quality=q, accuracy=a, valid=v, **kwargs)
        for q, a, v in zip(quality, accuracy, valid)
    ]


# -- correlation primitives --------------------------------------------------


def test_spearman_is_one_for_any_monotone_relation() -> None:
    """Rank correlation, not linear correlation: the curve's shape is irrelevant."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 4.0, 9.0, 16.0, 25.0]
    assert spearman(x, y) == pytest.approx(1.0)
    assert pearson(x, y) < 1.0


def test_spearman_handles_ties_without_blowing_up() -> None:
    assert spearman([1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]) == pytest.approx(1.0)


def test_correlation_is_none_when_undefined() -> None:
    assert spearman([1.0], [1.0]) is None
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


# -- Dice / IoU --------------------------------------------------------------


def test_dice_iou_conversion_round_trips() -> None:
    for value in (0.0, 0.25, 0.538, 0.70, 0.85, 1.0):
        assert iou_to_dice(dice_to_iou(value)) == pytest.approx(value)
    assert dice_to_iou(0.70) == pytest.approx(0.5384615, abs=1e-6)
    assert dice_to_iou(1.0) == pytest.approx(1.0)
    assert dice_to_iou(0.0) == pytest.approx(0.0)


def test_conversion_rejects_values_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        dice_to_iou(1.2)
    with pytest.raises(ValueError):
        iou_to_dice(-0.1)


def test_every_rank_based_result_is_identical_under_dice_and_iou() -> None:
    """Dice and IoU are a monotone transform of each other, so they order frames
    identically. Everything in this module is rank-based, so swapping the metric
    -- with the floor converted -- must change nothing at all. If this ever
    fails, some statistic has started depending on the metric's *scale*."""
    rng = np.random.default_rng(0)
    dice = rng.beta(5, 2, 400)
    iou = np.array([dice_to_iou(d) for d in dice])
    quality = np.clip(0.6 * dice + 0.4 * rng.beta(4, 2, 400) + rng.normal(0, 0.08, 400), 0, 1)
    valid = quality >= 0.6

    by_dice = build_trust_report(
        frames(list(quality), list(dice), list(valid)), accuracy_floor=0.70
    )
    by_iou = build_trust_report(
        frames(list(quality), list(iou), list(valid)), accuracy_floor=dice_to_iou(0.70)
    )

    assert by_dice.agreement.spearman == pytest.approx(by_iou.agreement.spearman)
    assert by_dice.detection.auroc == pytest.approx(by_iou.detection.auroc)
    assert by_dice.risk.aurc == pytest.approx(by_iou.risk.aurc)
    assert by_dice.gate.to_dict() == by_iou.gate.to_dict() or (
        by_dice.gate.trusted_bad == by_iou.gate.trusted_bad
        and by_dice.gate.trusted_good == by_iou.gate.trusted_good
    )


def test_forgetting_to_convert_the_floor_changes_the_question() -> None:
    """The trap the CLI warns about: an unconverted floor is a far stricter
    criterion, and only the gate counts reveal it."""
    rng = np.random.default_rng(1)
    dice = rng.beta(5, 2, 400)
    iou = np.array([dice_to_iou(d) for d in dice])
    quality = np.clip(dice + rng.normal(0, 0.1, 400), 0, 1)

    correct = gate_confusion(frames(list(quality), list(iou)), dice_to_iou(0.70))
    unconverted = gate_confusion(frames(list(quality), list(iou)), 0.70)
    assert unconverted.trusted_bad > correct.trusted_bad


# -- agreement ---------------------------------------------------------------


def test_unmeasured_and_unlabeled_frames_are_counted_not_zeroed() -> None:
    """A missing score must never enter a mean as 0.0 -- it would look like the
    worst possible frame instead of an absent measurement."""
    outcomes = [
        FrameOutcome(quality=0.9, accuracy=0.9),
        FrameOutcome(quality=None, accuracy=0.9),
        FrameOutcome(quality=0.9, accuracy=None),
        FrameOutcome(quality=0.8, accuracy=0.8),
    ]
    report = rank_agreement(outcomes)
    assert report.n == 2
    assert report.n_unscored == 1
    assert report.n_unlabeled == 1


def test_empty_input_produces_a_report_rather_than_an_exception() -> None:
    report = build_trust_report([])
    assert report.agreement.n == 0
    assert report.detection.auroc is None
    assert report.operating.feasible is False


# -- the gate ----------------------------------------------------------------


def test_gate_quadrants_are_named_by_what_they_cost() -> None:
    outcomes = [
        FrameOutcome(quality=0.9, valid=True, accuracy=0.9),   # trusted_good
        FrameOutcome(quality=0.9, valid=True, accuracy=0.2),   # trusted_bad
        FrameOutcome(quality=0.1, valid=False, accuracy=0.2),  # rejected_bad
        FrameOutcome(quality=0.1, valid=False, accuracy=0.9),  # rejected_good
    ]
    gate = gate_confusion(outcomes, FLOOR)
    assert (gate.trusted_good, gate.trusted_bad, gate.rejected_bad, gate.rejected_good) == (1, 1, 1, 1)
    assert gate.coverage == pytest.approx(0.5)
    assert gate.trusted_bad_rate == pytest.approx(0.5)
    assert gate.recall == pytest.approx(0.5)
    assert gate.specificity == pytest.approx(0.5)


def test_a_perfect_gate_has_a_zero_trusted_bad_rate() -> None:
    outcomes = [
        FrameOutcome(quality=0.9, valid=True, accuracy=0.95),
        FrameOutcome(quality=0.1, valid=False, accuracy=0.10),
    ]
    gate = gate_confusion(outcomes, FLOOR)
    assert gate.trusted_bad_rate == pytest.approx(0.0)
    assert gate.rejected_good_rate == pytest.approx(0.0)


def test_gate_rates_are_none_rather_than_zero_when_a_side_is_empty() -> None:
    """Nothing accepted is not the same as nothing accepted being wrong."""
    outcomes = [FrameOutcome(quality=0.1, valid=False, accuracy=0.9)]
    gate = gate_confusion(outcomes, FLOOR)
    assert gate.trusted_bad_rate is None
    assert gate.specificity is None


# -- discrimination ----------------------------------------------------------


def test_a_perfect_score_reaches_auroc_one() -> None:
    quality = [0.1, 0.2, 0.8, 0.9]
    accuracy = [0.1, 0.2, 0.9, 0.95]
    assert detection_report(frames(quality, accuracy), FLOOR).auroc == pytest.approx(1.0)


def test_an_inverted_score_reaches_auroc_zero() -> None:
    """AUROC 0 is diagnostic, not a failure to compute: the score is right but
    the sign is flipped, which is a different bug from a useless score."""
    quality = [0.9, 0.8, 0.2, 0.1]
    accuracy = [0.1, 0.2, 0.9, 0.95]
    assert detection_report(frames(quality, accuracy), FLOOR).auroc == pytest.approx(0.0)


def test_a_useless_score_lands_near_auroc_half() -> None:
    rng = np.random.default_rng(3)
    accuracy = rng.uniform(0.0, 1.0, 600)
    quality = rng.uniform(0.0, 1.0, 600)
    auroc = detection_report(frames(list(quality), list(accuracy)), FLOOR).auroc
    assert 0.42 < auroc < 0.58


def test_auroc_is_none_when_only_one_class_is_present() -> None:
    """Reporting 0.5 here would read as a measured result instead of 'undefined'."""
    report = detection_report(frames([0.9, 0.8], [0.95, 0.9]), FLOOR)
    assert report.auroc is None
    assert report.average_precision is None


def test_auroc_is_tie_correct() -> None:
    """Every score identical: the detector is exactly uninformative, 0.5 exactly."""
    outcomes = frames([0.5] * 6, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    assert detection_report(outcomes, FLOOR).auroc == pytest.approx(0.5)


# -- risk / coverage ---------------------------------------------------------


def test_risk_coverage_sweeps_from_one_frame_to_all_frames() -> None:
    rng = np.random.default_rng(1)
    accuracy = list(rng.uniform(0, 1, 50))
    report = risk_coverage(frames(accuracy, accuracy), FLOOR)
    assert len(report.points) == 50
    assert report.points[-1].coverage == pytest.approx(1.0)
    assert report.points[-1].n_accepted == 50
    assert report.full_coverage_bad_rate == pytest.approx(
        float(np.mean(np.asarray(accuracy) < FLOOR))
    )


def test_a_perfect_score_gives_zero_risk_until_full_coverage() -> None:
    """With a perfect ordering, every accepted frame is good until the bad ones
    are forced in -- so AURC collapses toward zero."""
    accuracy = [0.95, 0.9, 0.85, 0.8, 0.2, 0.1]
    report = risk_coverage(frames(accuracy, accuracy), FLOOR)
    assert [p.bad_rate for p in report.points][:4] == [0.0, 0.0, 0.0, 0.0]
    assert report.aurc < 0.15


def test_a_perfect_score_has_lower_aurc_than_a_random_one() -> None:
    rng = np.random.default_rng(5)
    accuracy = list(rng.uniform(0, 1, 300))
    good = risk_coverage(frames(accuracy, accuracy), FLOOR).aurc
    noise = risk_coverage(frames(list(rng.uniform(0, 1, 300)), accuracy), FLOOR).aurc
    assert good < noise


# -- operating point ---------------------------------------------------------


def test_operating_point_respects_the_budget_as_a_hard_constraint() -> None:
    rng = np.random.default_rng(2)
    accuracy = rng.beta(5, 2, 500)
    quality = np.clip(accuracy + rng.normal(0, 0.08, 500), 0, 1)
    point = operating_point(
        frames(list(quality), list(accuracy)), target_bad_rate=0.05, accuracy_floor=FLOOR
    )
    assert point.feasible
    assert point.achieved_bad_rate <= 0.05
    assert 0.0 < point.coverage <= 1.0


def test_operating_point_reports_infeasible_rather_than_returning_the_least_bad() -> None:
    """No threshold meets the budget. Saying so is the answer; silently handing
    back the least-bad threshold would be a safety claim nobody made."""
    outcomes = frames([0.9, 0.8, 0.7], [0.1, 0.2, 0.3])
    point = operating_point(outcomes, target_bad_rate=0.01, accuracy_floor=FLOOR)
    assert point.feasible is False
    assert point.threshold is None


def test_operating_point_honours_a_minimum_coverage() -> None:
    accuracy = [0.95] + [0.1] * 20
    outcomes = frames(accuracy, accuracy)
    assert operating_point(outcomes, 0.0, FLOOR, min_coverage=0.0).feasible
    assert not operating_point(outcomes, 0.0, FLOOR, min_coverage=0.5).feasible


# -- calibration -------------------------------------------------------------


def test_quantile_bins_are_used_because_quality_scores_cluster() -> None:
    quality = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.10]
    report = calibration(frames(quality, quality), n_bins=5, min_bin_count=1)
    counts = [b.n for b in report.bins]
    assert len(report.bins) >= 4
    assert max(counts) - min(counts) <= 2  # roughly equal occupancy


def test_a_score_equal_to_accuracy_is_perfectly_calibrated() -> None:
    values = list(np.linspace(0.05, 0.95, 60))
    report = calibration(frames(values, values), n_bins=6, min_bin_count=1)
    assert report.ece == pytest.approx(0.0, abs=1e-9)
    assert report.monotone_fraction == pytest.approx(1.0)


def test_calibration_flags_a_non_monotone_score() -> None:
    """A mis-scaled score can be recalibrated. A non-monotone one cannot, so the
    monotone fraction matters more than the ECE."""
    quality = list(np.linspace(0.0, 1.0, 60))
    accuracy = list(np.abs(np.linspace(-1.0, 1.0, 60)))  # V shape: rises, falls
    report = calibration(frames(quality, accuracy), n_bins=6, min_bin_count=1)
    assert report.monotone_fraction < 1.0


def test_small_bins_are_dropped_from_the_aggregate() -> None:
    values = list(np.linspace(0.05, 0.95, 40))
    strict = calibration(frames(values, values), n_bins=8, min_bin_count=100)
    assert strict.ece is None
    assert strict.bins  # still reported for inspection


def test_calibration_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        calibration([], n_bins=1)
    with pytest.raises(ValueError, match="strategy"):
        calibration([], strategy="kmeans")


# -- component attribution ---------------------------------------------------


def test_attribution_identifies_the_component_that_carries_the_signal() -> None:
    rng = np.random.default_rng(7)
    outcomes = []
    for _ in range(400):
        accuracy = float(rng.uniform(0, 1))
        outcomes.append(
            FrameOutcome(
                quality=accuracy,
                accuracy=accuracy,
                components={"useful": accuracy, "noise": float(rng.uniform(0, 1))},
            )
        )
    result = {c.name: c for c in component_attribution(outcomes, {"useful": 1.0, "noise": 1.0})}
    assert result["useful"].spearman > 0.9
    assert abs(result["noise"].spearman) < 0.2


def test_a_harmful_component_has_a_negative_leave_one_out_delta() -> None:
    """delta < 0 means the aggregate improves when the component is removed --
    the weight should go to zero, not merely down."""
    rng = np.random.default_rng(11)
    outcomes = []
    for _ in range(400):
        accuracy = float(rng.uniform(0, 1))
        outcomes.append(
            FrameOutcome(
                quality=accuracy,
                accuracy=accuracy,
                components={
                    "useful": float(np.clip(accuracy + rng.normal(0, 0.15), 0, 1)),
                    "harmful": float(np.clip(1.0 - accuracy + rng.normal(0, 0.15), 0, 1)),
                },
            )
        )
    result = {c.name: c for c in component_attribution(outcomes, {"useful": 1.0, "harmful": 1.0})}
    assert result["harmful"].delta_spearman < 0.0
    assert result["useful"].delta_spearman > 0.0


def test_delta_is_none_when_the_aggregate_has_no_spread() -> None:
    """Two components that cancel exactly make the aggregate constant, and rank
    correlation is undefined for a constant. Reporting None beats reporting 0.0,
    which would read as 'measured, and the component does not matter'."""
    outcomes = [
        FrameOutcome(quality=a, accuracy=a, components={"up": a, "down": 1.0 - a})
        for a in np.linspace(0.0, 1.0, 50)
    ]
    result = {c.name: c for c in component_attribution(outcomes, {"up": 1.0, "down": 1.0})}
    assert result["up"].delta_spearman is None
    assert result["up"].spearman == pytest.approx(1.0)


def test_attribution_is_empty_without_components_or_labels() -> None:
    assert component_attribution([FrameOutcome(quality=0.5, accuracy=0.5)], {"a": 1.0}) == []


# -- reason codes ------------------------------------------------------------


def test_a_reason_that_fires_only_on_bad_frames_has_precision_one() -> None:
    outcomes = [
        FrameOutcome(quality=0.2, accuracy=0.2, reasons=["empty_mask"]),
        FrameOutcome(quality=0.3, accuracy=0.3, reasons=["empty_mask"]),
        FrameOutcome(quality=0.9, accuracy=0.9, reasons=[]),
        FrameOutcome(quality=0.9, accuracy=0.9, reasons=[]),
    ]
    stat = reason_code_report(outcomes, FLOOR)[0]
    assert stat.reason == "empty_mask"
    assert stat.precision == pytest.approx(1.0)
    assert stat.lift == pytest.approx(2.0)


def test_a_reason_firing_at_random_has_lift_near_one() -> None:
    """Lift ~1 means the reason marks frames no worse than average -- a supervisor
    state transition taken for nothing."""
    rng = np.random.default_rng(13)
    outcomes = [
        FrameOutcome(
            quality=0.5,
            accuracy=float(rng.uniform(0, 1)),
            reasons=["noise_code"] if rng.random() < 0.5 else [],
        )
        for _ in range(800)
    ]
    stat = next(s for s in reason_code_report(outcomes, FLOOR) if s.reason == "noise_code")
    assert 0.8 < stat.lift < 1.25


# -- Q_raw force response ----------------------------------------------------


def test_a_monotone_rise_is_detected_as_monotone_and_unimodal() -> None:
    samples = {f: [0.1 * i + 0.1] * 30 for i, f in enumerate([1.0, 2.0, 3.0, 4.0, 5.0])}
    report = force_response(samples)
    assert report.spearman_with_force == pytest.approx(1.0)
    assert report.monotone_fraction == pytest.approx(1.0)
    assert report.is_unimodal
    assert report.sign_changes == 0


def test_an_inverted_u_is_unimodal_with_exactly_one_turn() -> None:
    means = {1.0: 0.2, 2.0: 0.5, 3.0: 0.9, 4.0: 0.6, 5.0: 0.3}
    report = force_response({f: [m] * 30 for f, m in means.items()})
    assert report.is_unimodal
    assert report.sign_changes == 1
    assert report.argmax_force == pytest.approx(3.0)


def test_a_noisy_multi_modal_response_is_not_unimodal() -> None:
    """This is the outcome that invalidates the Stage 1 force search."""
    means = {1.0: 0.2, 2.0: 0.8, 3.0: 0.2, 4.0: 0.8, 5.0: 0.2}
    report = force_response({f: [m] * 30 for f, m in means.items()})
    assert report.is_unimodal is False
    assert report.sign_changes >= 2


def test_f_star_is_the_smallest_force_within_epsilon_not_the_argmax() -> None:
    """The Stage 1 rule: bias toward less pressure on the patient."""
    means = {1.0: 0.50, 2.0: 0.88, 3.0: 0.90, 4.0: 0.89}
    report = force_response({f: [m] * 30 for f, m in means.items()}, epsilon=0.03)
    assert report.argmax_force == pytest.approx(3.0)
    assert report.f_star == pytest.approx(2.0)


def test_a_level_with_too_few_valid_samples_is_a_measurement_failure() -> None:
    samples = {
        1.0: [0.5] * 30,
        2.0: [0.6] * 10 + [None] * 20,  # 33% valid, below the 60% floor
        3.0: [0.7] * 30,
    }
    report = force_response(samples, min_valid_fraction=0.60)
    by_force = {level.force: level for level in report.levels}
    assert by_force[2.0].mean is None
    assert by_force[2.0].valid_fraction == pytest.approx(1 / 3)
    assert by_force[1.0].mean == pytest.approx(0.5)


def test_force_response_needs_at_least_two_measured_levels() -> None:
    report = force_response({1.0: [0.5] * 30})
    assert report.spearman_with_force is None
    assert report.is_unimodal is None


def test_epsilon_sized_wobble_does_not_count_as_a_turn() -> None:
    """Without this, sampling noise on a flat response reads as multi-modal."""
    means = {1.0: 0.50, 2.0: 0.51, 3.0: 0.50, 4.0: 0.51, 5.0: 0.50}
    report = force_response({f: [m] * 30 for f, m in means.items()}, epsilon=0.03)
    assert report.sign_changes == 0
    assert report.is_unimodal


# -- aggregate ---------------------------------------------------------------


def test_grouping_splits_by_patient_and_rejects_other_keys() -> None:
    outcomes = [
        FrameOutcome(quality=0.5, accuracy=0.5, patient_id="P1"),
        FrameOutcome(quality=0.5, accuracy=0.5, patient_id="P2"),
        FrameOutcome(quality=0.5, accuracy=0.5, patient_id="P1"),
    ]
    groups = group_by(outcomes)
    assert sorted(groups) == ["P1", "P2"]
    assert len(groups["P1"]) == 2
    with pytest.raises(ValueError, match="patient_id or sequence_id"):
        group_by(outcomes, "frame_index")


def test_report_is_serialisable_and_reports_per_patient() -> None:
    import json

    rng = np.random.default_rng(17)
    outcomes = []
    for i in range(200):
        accuracy = float(rng.beta(4, 2))
        quality = float(np.clip(accuracy + rng.normal(0, 0.1), 0, 1))
        outcomes.append(
            FrameOutcome(
                quality=quality,
                valid=quality >= 0.6,
                accuracy=accuracy,
                components={"a": quality},
                reasons=[] if quality >= 0.6 else ["low_control_quality"],
                patient_id=f"P{i % 4}",
            )
        )
    report = build_trust_report(outcomes, weights={"a": 1.0})
    payload = report.to_dict()
    json.dumps(payload)  # must not raise
    assert sorted(report.per_patient) == ["P0", "P1", "P2", "P3"]
    assert all(math.isfinite(v) for v in [report.agreement.spearman, report.detection.auroc])
    assert len(report.summary_lines()) >= 4
