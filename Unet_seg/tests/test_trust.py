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
    asymmetric_margin,
    build_trust_report,
    calibration,
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
    unimodal_regression,
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


# -- Q_raw force response: 2026-09-08 revision -------------------------------


GRID = [1.0 + 0.5 * i for i in range(13)]  # the DESIGN_NOTES 7.2 force grid


def concave(force: float) -> float:
    return 0.8 - (force - 4.0) ** 2 / 40.0


def ar1(n: int, phi: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    """A strongly autocorrelated zero-mean sequence, like slow speckle drift."""
    x = np.zeros(n)
    noise = rng.normal(0.0, scale, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def test_asymmetric_margin_matches_the_design_values() -> None:
    """w-/w+ = 5 puts the setpoint 0.636 sigma inside the left edge of the tie
    set; the design ratio's neighbours are pinned too so a regression in the
    root finder cannot hide behind one lucky value."""
    assert asymmetric_margin(1.0, 5.0) == pytest.approx(0.636, abs=5e-4)
    assert asymmetric_margin(1.0, 3.0) == pytest.approx(0.436, abs=5e-4)
    assert asymmetric_margin(1.0, 10.0) == pytest.approx(0.901, abs=5e-4)
    assert asymmetric_margin(0.5, 5.0) == pytest.approx(0.318, abs=5e-4)  # scales with sigma
    assert asymmetric_margin(0.0, 5.0) == 0.0
    assert asymmetric_margin(-1.0, 5.0) == 0.0
    assert asymmetric_margin(1.0, 1.0) == 0.0  # symmetric cost: no margin


def test_unimodal_regression_recovers_a_rise_then_fall_exactly() -> None:
    means = np.array([0.2, 0.5, 0.9, 0.6, 0.3])
    fit, peak, sse = unimodal_regression(means)
    assert np.array_equal(fit, means)
    assert peak == 2
    assert sse == pytest.approx(0.0)

    # Monotone sequences are umbrellas with the peak at an end.
    fit, peak, sse = unimodal_regression(np.array([0.1, 0.2, 0.3, 0.4]))
    assert peak == 3 and sse == pytest.approx(0.0)
    fit, peak, sse = unimodal_regression(np.array([0.4, 0.3, 0.2, 0.1]))
    assert peak == 0 and sse == pytest.approx(0.0)


def test_unimodal_regression_pools_a_violation_and_respects_weights() -> None:
    fit, peak, sse = unimodal_regression(np.array([0.2, 0.8, 0.2, 0.8, 0.2]))
    assert peak in (1, 3)
    assert sse > 0.1
    # Rise then fall on both sides of the chosen peak.
    assert np.all(np.diff(fit[: peak + 1]) >= -1e-12)
    assert np.all(np.diff(fit[peak:]) <= 1e-12)

    # A heavily weighted point pulls the pooled block toward itself.
    light, _, _ = unimodal_regression(np.array([0.5, 0.4, 0.6]), np.array([1.0, 1.0, 1.0]))
    heavy, _, _ = unimodal_regression(np.array([0.5, 0.4, 0.6]), np.array([1.0, 100.0, 1.0]))
    assert abs(heavy[1] - 0.4) < abs(light[1] - 0.4)


def test_f_star_is_f_left_plus_the_margin() -> None:
    means = {1.0: 0.50, 2.0: 0.88, 3.0: 0.90, 4.0: 0.89}
    samples = {f: [m] * 30 for f, m in means.items()}
    legacy = force_response(samples, epsilon=0.03)
    assert legacy.f_left == pytest.approx(2.0)
    assert legacy.f_star == pytest.approx(legacy.f_left)  # no margin: unchanged rule
    assert legacy.margin_n == 0.0

    margin = asymmetric_margin(0.5, 5.0)
    shifted = force_response(samples, epsilon=0.03, margin_n=margin)
    assert shifted.f_left == pytest.approx(2.0)
    assert shifted.f_star == pytest.approx(2.0 + margin)
    assert shifted.margin_n == pytest.approx(margin)
    assert shifted.to_dict()["f_star"] == pytest.approx(shifted.f_star)


def test_autocorrelated_hold_windows_have_fewer_effective_samples() -> None:
    """A 1 s hold window is 30 correlated frames, not 30 independent draws."""
    rng = np.random.default_rng(23)
    n = 500
    iid = {f: list(concave(f) + rng.normal(0.0, 0.05, n)) for f in GRID}
    drifting = {f: list(concave(f) + ar1(n, 0.9, 0.02, rng)) for f in GRID}

    for level in force_response(iid).levels:
        assert abs(level.rho) < 0.15
        assert level.n_eff > 0.75 * n
        assert level.sem == pytest.approx(level.std / math.sqrt(level.n_eff))
    for level in force_response(drifting).levels:
        assert level.rho > 0.7
        assert level.n_eff < n / 2
        assert level.sem > level.std / math.sqrt(n)

    # Opting out restores n, and rho is still reported for inspection.
    for level in force_response(drifting, autocorr_correct=False).levels:
        assert level.n_eff == n
        assert level.rho > 0.7
        assert level.sem == pytest.approx(level.std / math.sqrt(n))


def test_welch_tie_rule_admits_by_standard_error_not_by_a_fixed_epsilon() -> None:
    rng = np.random.default_rng(29)
    n, spread = 200, 0.06

    def level(mean: float) -> list[float]:
        noise = rng.normal(0.0, 1.0, n)
        noise = (noise - noise.mean()) / noise.std(ddof=1) * spread  # exact mean and std
        return list(mean + noise)

    sem = spread / math.sqrt(n)
    z = 1.64
    close_gap = 0.8 * z * math.sqrt(2.0) * sem
    far_gap = 4.0 * z * math.sqrt(2.0) * sem
    samples = {2.0: level(0.90 - far_gap), 3.0: level(0.90 - close_gap), 4.0: level(0.90)}

    report = force_response(samples, z_alpha=z, autocorr_correct=False)
    assert report.tie_rule == "welch"
    assert report.argmax_force == pytest.approx(4.0)
    assert report.f_left == pytest.approx(3.0)  # admitted: inside z*sqrt(2)*sem
    assert report.f_star == pytest.approx(3.0)

    # The legacy rule with a generous epsilon would also admit 2.0; Welch does not.
    legacy = force_response(samples, epsilon=far_gap + 1e-9)
    assert legacy.tie_rule == "epsilon"
    assert legacy.f_left == pytest.approx(2.0)


def test_unimodality_is_judged_against_the_umbrella_fit_when_noise_is_known() -> None:
    rng = np.random.default_rng(31)
    n = 60
    noisy_concave = {f: list(concave(f) + rng.normal(0.0, 0.05, n)) for f in GRID}
    report = force_response(noisy_concave)
    assert report.unimodal_sse_ratio is not None
    assert report.unimodal_sse_ratio <= 2.0
    assert report.is_unimodal is True
    assert report.unimodal_peak_force == pytest.approx(4.0, abs=1.0)
    assert len(report.unimodal_fit) == len(GRID)
    assert report.to_dict()["unimodal_sse_ratio"] == report.unimodal_sse_ratio

    zigzag = {
        f: list(m + rng.normal(0.0, 0.02, 30))
        for f, m in zip([1.0, 2.0, 3.0, 4.0, 5.0], [0.2, 0.8, 0.2, 0.8, 0.2])
    }
    report = force_response(zigzag)
    assert report.unimodal_sse_ratio > 2.0
    assert report.is_unimodal is False
    assert report.sign_changes >= 2

    # Constant samples have no sem: the ratio is undefined and the legacy
    # sign-change rule decides, as the pre-revision tests above rely on.
    flat = force_response({f: [m] * 30 for f, m in zip([1.0, 2.0, 3.0], [0.2, 0.5, 0.3])})
    assert flat.unimodal_sse_ratio is None
    assert flat.is_unimodal is True


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
