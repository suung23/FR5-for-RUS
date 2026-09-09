"""Control-output tests: geometry, postprocessing, quality, validity, serialization."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from rus_perception.control.features import (
    FeatureExtractionConfig,
    binary_mask_geometry,
    border_contact_ratio,
    boundary_entropy,
    extract_control_state,
    lumen_contrast,
    segmentation_confidence_from_probability,
    warp_mask_with_flow,
)
from rus_perception.control.postprocess import PostprocessConfig, postprocess_probability
from rus_perception.control.quality import (
    FORCE_SEARCH_WEIGHTS,
    QUALITY_COMPONENT_NAMES,
    QualityConfig,
    compute_control_quality,
)
from rus_perception.control.state import COORDINATE_CONVENTION, BoundingBox, ControlState
from rus_perception.control.validity import REJECTION_REASONS, ValidityConfig, evaluate_validity


def square_mask(size: int = 64, x: int = 20, y: int = 24, width: int = 16) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    mask[y : y + width, x : x + width] = 1
    return mask


# -- geometry --------------------------------------------------------------
def test_centroid_of_a_known_square() -> None:
    mask = square_mask(size=64, x=20, y=24, width=16)
    geometry = binary_mask_geometry(mask)
    assert geometry["centroid_x_px"] == pytest.approx(27.5)
    assert geometry["centroid_y_px"] == pytest.approx(31.5)
    assert geometry["centroid_x_normalized"] == pytest.approx(28.0 / 64)
    assert geometry["centroid_y_normalized"] == pytest.approx(32.0 / 64)
    assert geometry["mask_area_px"] == 256
    assert geometry["mask_area_ratio"] == pytest.approx(256 / 4096)
    assert geometry["bounding_box"].to_list() == [20, 24, 35, 39]


def test_geometry_of_an_empty_mask_is_none_not_zero() -> None:
    geometry = binary_mask_geometry(np.zeros((32, 32), np.uint8))
    assert geometry["centroid_x_px"] is None
    assert geometry["bounding_box"] is None
    assert geometry["mask_area_px"] == 0


def test_axis_lengths_and_orientation_of_an_elongated_mask() -> None:
    mask = np.zeros((64, 64), np.uint8)
    mask[30:34, 10:50] = 1  # wide and short: major axis along +x
    geometry = binary_mask_geometry(mask)
    assert geometry["major_axis_length"] > geometry["minor_axis_length"]
    assert abs(geometry["orientation_degrees"]) < 5.0

    tall = np.zeros((64, 64), np.uint8)
    tall[10:50, 30:34] = 1  # tall and narrow: major axis along +y
    assert abs(binary_mask_geometry(tall)["orientation_degrees"]) == pytest.approx(90.0, abs=5.0)


def test_center_error_sign_convention() -> None:
    """Positive x error = right of centre; positive y error = below centre."""
    probability = np.zeros((64, 64), np.float32)
    probability[10:20, 40:55] = 1.0  # upper right quadrant
    state = extract_control_state(probability)
    assert state.center_error_x > 0, "a mask right of centre must give a positive x error"
    assert state.center_error_y < 0, "a mask above centre must give a negative y error"
    assert state.center_error_x == pytest.approx(state.centroid_x_normalized - 0.5)
    assert state.center_error_y == pytest.approx(state.centroid_y_normalized - 0.5)
    assert "top-left" in COORDINATE_CONVENTION


def test_area_ratio_matches_the_pixel_count() -> None:
    probability = np.zeros((40, 50), np.float32)
    probability[0:10, 0:10] = 1.0
    state = extract_control_state(probability)
    assert state.mask_area_px == 100
    assert state.mask_area_ratio == pytest.approx(100 / 2000)


# -- postprocessing --------------------------------------------------------
def test_largest_connected_component_is_retained() -> None:
    probability = np.zeros((64, 64), np.float32)
    probability[10:30, 10:30] = 0.9  # 400 px
    probability[50:56, 50:56] = 0.9  # 36 px speck
    result = postprocess_probability(probability, PostprocessConfig(largest_component=True))

    assert result.num_components == 2
    assert int(result.raw_mask.sum()) == 436
    assert int(result.mask.sum()) == 400
    assert result.mask[52, 52] == 0
    assert result.largest_component_ratio == pytest.approx(400 / 436)
    assert any("largest" in decision for decision in result.decisions)


def test_raw_probability_and_raw_mask_are_always_preserved() -> None:
    probability = np.zeros((32, 32), np.float32)
    probability[4:8, 4:8] = 0.9
    probability[20:22, 20:22] = 0.9
    result = postprocess_probability(probability, PostprocessConfig(largest_component=True))
    assert np.array_equal(result.probability_map, probability)
    assert int(result.raw_mask.sum()) > int(result.mask.sum())


def test_small_components_can_be_removed_by_area_ratio() -> None:
    probability = np.zeros((64, 64), np.float32)
    probability[10:30, 10:30] = 0.9
    probability[50:53, 50:53] = 0.9
    result = postprocess_probability(
        probability, PostprocessConfig(largest_component=False, min_component_area_ratio=0.01)
    )
    assert int(result.mask.sum()) == 400


def test_hole_filling_and_contour_smoothing_are_opt_in() -> None:
    probability = np.zeros((64, 64), np.float32)
    probability[10:40, 10:40] = 0.9
    probability[20:25, 20:25] = 0.0  # interior hole

    untouched = postprocess_probability(probability, PostprocessConfig(fill_holes=False))
    filled = postprocess_probability(probability, PostprocessConfig(fill_holes=True))
    assert int(filled.mask.sum()) > int(untouched.mask.sum())
    assert any("holes" in decision for decision in filled.decisions)

    smoothed = postprocess_probability(probability, PostprocessConfig(smooth_contour=True))
    assert any("smoothed" in decision for decision in smoothed.decisions)


def test_postprocess_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="threshold must be"):
        PostprocessConfig(threshold=1.5)
    with pytest.raises(ValueError, match="connectivity"):
        PostprocessConfig(connectivity=6)
    with pytest.raises(ValueError, match="2-D"):
        postprocess_probability(np.zeros((1, 8, 8), np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        postprocess_probability(np.full((8, 8), 5.0, np.float32))
    with pytest.raises(ValueError, match="Unknown postprocess key"):
        PostprocessConfig.from_dict({"nope": 1})


def test_empty_probability_map_yields_an_empty_mask() -> None:
    result = postprocess_probability(np.zeros((16, 16), np.float32))
    assert int(result.mask.sum()) == 0
    assert result.num_components == 0
    assert result.largest_component_ratio == 0.0


# -- image-derived quality features ---------------------------------------
def test_border_contact_ratio_detects_a_cut_off_mask() -> None:
    interior = square_mask(64, 20, 20, 16)
    assert border_contact_ratio(interior) == pytest.approx(0.0)

    cut_off = np.zeros((64, 64), np.uint8)
    cut_off[:20, :20] = 1  # touching the top-left corner
    assert border_contact_ratio(cut_off) > 0.3
    assert 0.0 <= border_contact_ratio(cut_off) <= 1.0
    assert border_contact_ratio(np.zeros((16, 16), np.uint8)) == 0.0


def test_segmentation_confidence_is_high_when_probabilities_are_saturated() -> None:
    confident = np.zeros((32, 32), np.float32)
    confident[8:24, 8:24] = 1.0
    hedged = np.full((32, 32), 0.5, np.float32)
    assert segmentation_confidence_from_probability(confident) == pytest.approx(1.0)
    assert segmentation_confidence_from_probability(hedged) == pytest.approx(0.0)


def test_boundary_entropy_is_low_for_a_crisp_boundary() -> None:
    crisp = np.zeros((64, 64), np.float32)
    crisp[20:40, 20:40] = 1.0
    mask = (crisp >= 0.5).astype(np.uint8)
    blurred = np.clip(crisp * 0.5 + 0.25, 0, 1)

    assert boundary_entropy(crisp, mask) < boundary_entropy(blurred, mask)
    assert 0.0 <= boundary_entropy(blurred, mask) <= 1.0
    assert boundary_entropy(crisp, np.zeros((64, 64), np.uint8)) == 0.0


def test_lumen_contrast_is_positive_for_a_dark_lumen() -> None:
    mask = square_mask(64, 24, 24, 16)
    image = np.full((64, 64), 0.7, np.float32)
    image[mask > 0] = 0.1
    lumen, ring, contrast = lumen_contrast(image, mask)
    assert lumen == pytest.approx(0.1, abs=1e-5)
    assert ring == pytest.approx(0.7, abs=1e-5)
    assert contrast > 0.5

    assert lumen_contrast(image, np.zeros((64, 64), np.uint8)) == (None, None, None)


# -- temporal features -----------------------------------------------------
def test_warped_iou_is_one_for_a_correctly_compensated_translation() -> None:
    probability = np.zeros((64, 64), np.float32)
    probability[20:36, 20:36] = 1.0
    first = extract_control_state(probability)

    shifted = np.roll(probability, 5, axis=1)
    flow = np.zeros((64, 64, 2), np.float32)
    flow[..., 0] = -5.0  # current -> previous
    second = extract_control_state(shifted, previous_state=first, flow=flow)

    assert second.temporal_warped_iou == pytest.approx(1.0)
    assert second.temporal_warped_dice == pytest.approx(1.0)
    assert second.normalized_centroid_jump == pytest.approx(0.0, abs=1e-6)
    assert second.metadata["temporal_alignment"] == "backward_flow"


def test_unwarped_comparison_penalises_genuine_motion() -> None:
    """Comparing against the unwarped previous mask is flagged, not silently done."""
    probability = np.zeros((64, 64), np.float32)
    probability[20:36, 20:36] = 1.0
    first = extract_control_state(probability)
    second = extract_control_state(np.roll(probability, 10, axis=1), previous_state=first)

    assert second.metadata["temporal_alignment"] == "unwarped"
    assert second.temporal_warped_iou < 0.6
    assert second.normalized_centroid_jump > 0.1


def test_warp_mask_with_flow_matches_a_known_shift() -> None:
    mask = square_mask(32, 8, 8, 6)
    flow = np.zeros((32, 32, 2), np.float32)
    flow[..., 0] = -4.0
    warped = warp_mask_with_flow(mask, flow)
    assert np.array_equal(warped, np.roll(mask, 4, axis=1))


def test_relative_area_change_is_reported() -> None:
    big = np.zeros((64, 64), np.float32)
    big[20:40, 20:40] = 1.0
    small = np.zeros((64, 64), np.float32)
    small[20:30, 20:40] = 1.0

    first = extract_control_state(big)
    second = extract_control_state(small, previous_state=first, flow=np.zeros((64, 64, 2), np.float32))
    assert second.relative_area_change < 0, "a shrinking mask must give a negative change"
    assert second.temporal_stability_score is not None


def test_extract_control_state_rejects_mismatched_flow() -> None:
    probability = np.zeros((32, 32), np.float32)
    probability[8:16, 8:16] = 1.0
    first = extract_control_state(probability)
    with pytest.raises(ValueError, match="flow must be H x W x 2"):
        extract_control_state(
            probability, previous_state=first, flow=np.zeros((16, 16, 2), np.float32)
        )


# -- quality ---------------------------------------------------------------
def test_quality_score_is_bounded_and_explains_itself() -> None:
    result = compute_control_quality(
        segmentation_confidence=0.9,
        mask_area_ratio=0.15,
        largest_component_ratio=1.0,
        border_contact_ratio=0.0,
        lumen_surrounding_contrast=0.4,
        temporal_warped_iou=0.9,
        normalized_centroid_jump=0.01,
        relative_area_change=0.02,
    )
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.8
    assert set(result.components).issubset(set(QUALITY_COMPONENT_NAMES))
    assert "control_quality_score" in result.explain()


def test_quality_score_drops_for_a_poor_observation() -> None:
    good = compute_control_quality(0.95, 0.15, 1.0, 0.0, 0.4, 0.95, 0.005, 0.01).score
    poor = compute_control_quality(0.30, 0.001, 0.3, 0.9, 0.02, 0.1, 0.4, 0.9).score
    assert good > poor
    assert poor < 0.5


def test_missing_temporal_inputs_are_dropped_not_scored_zero() -> None:
    """A first frame must not be punished for having no predecessor."""
    with_temporal = compute_control_quality(0.9, 0.15, 1.0, 0.0, 0.4, 0.9, 0.01, 0.01)
    without = compute_control_quality(0.9, 0.15, 1.0, 0.0, 0.4)
    assert "temporal_iou" not in without.components
    assert without.score == pytest.approx(with_temporal.score, abs=0.1)
    assert without.score > 0.7


def test_quality_config_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        QualityConfig(weights={"segmentation_confidence": -1.0})
    with pytest.raises(ValueError, match="Unknown quality weight"):
        QualityConfig.from_dict({"weights": {"not_a_component": 1.0}})
    with pytest.raises(ValueError, match="must be > 0"):
        QualityConfig(target_area_ratio=0.0)


def test_force_search_preset_ignores_mask_area_and_temporal_terms() -> None:
    """Stage 1b's objective: contact force compresses the bladder, so mask
    area is confounded with force and must not decide the optimum; changing
    the force changes the image, so stability must not reward standing still."""
    config = QualityConfig.for_force_search()
    assert config.weights == FORCE_SEARCH_WEIGHTS
    assert set(config.weights) == set(QUALITY_COMPONENT_NAMES)
    assert config.weights["mask_completeness"] == 0.0
    assert config.weights["segmentation_confidence"] < config.weights["boundary_sharpness"]

    shared = dict(
        segmentation_confidence=0.9,
        largest_component_ratio=1.0,
        border_contact_ratio=0.0,
        lumen_surrounding_contrast=0.3,
        mean_boundary_entropy=0.2,
        config=config,
    )
    small = compute_control_quality(mask_area_ratio=0.02, **shared)
    ideal = compute_control_quality(mask_area_ratio=config.target_area_ratio, **shared)
    large = compute_control_quality(mask_area_ratio=0.60, **shared)
    assert small.score == ideal.score == large.score
    assert "mask_completeness" not in small.weights  # reported, not scored
    assert set(small.weights) == {"segmentation_confidence", "boundary_sharpness", "lumen_contrast"}

    # A partial weights override merges over the preset; unknown names still fail.
    nudged = QualityConfig.for_force_search(weights={"segmentation_confidence": 0.8})
    assert nudged.weights["segmentation_confidence"] == 0.8
    assert nudged.weights["lumen_contrast"] == FORCE_SEARCH_WEIGHTS["lumen_contrast"]
    with pytest.raises(ValueError, match="Unknown quality weight"):
        QualityConfig.for_force_search(weights={"not_a_component": 1.0})


# -- validity gate ---------------------------------------------------------
def test_validity_gate_accepts_a_good_observation() -> None:
    result = evaluate_validity(
        mask_area_ratio=0.15,
        segmentation_confidence=0.9,
        border_contact_ratio=0.0,
        largest_component_ratio=1.0,
        mask_area_px=1000,
        temporal_warped_iou=0.9,
        normalized_centroid_jump=0.01,
        relative_area_change=0.05,
    )
    assert result.valid
    assert result.reasons == []


@pytest.mark.parametrize(
    "kwargs,expected_reason",
    [
        ({"mask_area_px": 0, "mask_area_ratio": 0.0}, "empty_mask"),
        ({"mask_area_ratio": 0.0001}, "mask_area_too_small"),
        ({"mask_area_ratio": 0.95}, "mask_area_too_large"),
        ({"segmentation_confidence": 0.1}, "low_segmentation_confidence"),
        ({"border_contact_ratio": 0.9}, "excessive_border_contact"),
        ({"largest_component_ratio": 0.2}, "fragmented_mask"),
        ({"temporal_warped_iou": 0.1}, "low_temporal_warped_iou"),
        ({"normalized_centroid_jump": 0.5}, "excessive_centroid_jump"),
        ({"relative_area_change": 0.9}, "excessive_area_change"),
    ],
)
def test_validity_gate_reports_the_specific_failure(kwargs, expected_reason) -> None:
    base = dict(
        mask_area_ratio=0.15,
        segmentation_confidence=0.9,
        border_contact_ratio=0.0,
        largest_component_ratio=1.0,
        mask_area_px=1000,
        temporal_warped_iou=0.9,
        normalized_centroid_jump=0.01,
        relative_area_change=0.05,
    )
    base.update(kwargs)
    result = evaluate_validity(**base)
    assert not result.valid
    assert expected_reason in result.reasons
    assert all(reason in REJECTION_REASONS for reason in result.reasons)


def test_validity_gate_can_report_several_reasons_at_once() -> None:
    result = evaluate_validity(
        mask_area_ratio=0.0001,
        segmentation_confidence=0.1,
        border_contact_ratio=0.9,
        largest_component_ratio=0.2,
        mask_area_px=5,
    )
    assert len(result.reasons) >= 4
    assert len(result.reasons) == len(set(result.reasons)), "reasons must be deduplicated"


def test_missing_temporal_evidence_is_skipped_not_failed() -> None:
    result = evaluate_validity(
        mask_area_ratio=0.15,
        segmentation_confidence=0.9,
        border_contact_ratio=0.0,
        largest_component_ratio=1.0,
        mask_area_px=1000,
    )
    assert result.valid
    assert any("no_value" in entry for entry in result.skipped)


def test_require_temporal_rejects_frames_without_a_predecessor() -> None:
    result = evaluate_validity(
        mask_area_ratio=0.15,
        segmentation_confidence=0.9,
        border_contact_ratio=0.0,
        largest_component_ratio=1.0,
        mask_area_px=1000,
        config=ValidityConfig(require_temporal=True),
    )
    assert not result.valid
    assert "low_temporal_warped_iou" in result.reasons


def test_disabled_criteria_are_not_applied() -> None:
    result = evaluate_validity(
        mask_area_ratio=0.9999,
        segmentation_confidence=0.0,
        border_contact_ratio=1.0,
        largest_component_ratio=0.0,
        mask_area_px=10,
        config=ValidityConfig(
            min_area_ratio=None,
            max_area_ratio=None,
            min_segmentation_confidence=None,
            max_border_contact_ratio=None,
            min_largest_component_ratio=None,
            min_temporal_warped_iou=None,
            max_centroid_jump=None,
            max_relative_area_change=None,
        ),
    )
    assert result.valid


def test_validity_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown validity key"):
        ValidityConfig.from_dict({"min_are_ratio": 0.1})


# -- serialization ---------------------------------------------------------
def test_control_state_serializes_to_json(disc_probability, disc_image) -> None:
    state = extract_control_state(
        disc_probability,
        disc_image,
        metadata={"frame_id": "P1/S1/000012", "timestamp": 1.25, "model_version": "slim-1"},
    )
    payload = state.to_dict()
    text = json.dumps(payload)  # must not raise
    restored = json.loads(text)

    assert restored["frame_id"] == "P1/S1/000012"
    assert restored["model_version"] == "slim-1"
    assert restored["coordinate_convention"] == COORDINATE_CONVENTION
    # Dense arrays are excluded by default but their presence is reported.
    assert "probability_map" not in restored
    assert "binary_mask" not in restored
    assert restored["has_binary_mask"] is True
    assert restored["has_probability_map"] is True
    assert isinstance(restored["bounding_box"], list)
    assert isinstance(restored["rejection_reasons"], list)


def test_control_state_can_include_dense_arrays_on_request(disc_probability) -> None:
    state = extract_control_state(disc_probability)
    payload = state.to_dict(include_probability_map=True, include_binary_mask=True)
    json.dumps(payload)
    assert len(payload["binary_mask"]) == 64
    assert len(payload["probability_map"]) == 64


def test_non_finite_values_serialize_as_null() -> None:
    state = ControlState(frame_id="x", control_quality_score=float("nan"))
    payload = state.to_dict()
    assert payload["control_quality_score"] is None
    json.dumps(payload)


def test_flat_record_has_a_stable_column_set(disc_probability, disc_image) -> None:
    """The CSV header must not depend on whether temporal features existed."""
    first = extract_control_state(disc_probability, disc_image)
    second = extract_control_state(
        disc_probability, disc_image, previous_state=first, flow=np.zeros((64, 64, 2), np.float32)
    )
    assert set(first.flat_record()) == set(second.flat_record())
    for name in QUALITY_COMPONENT_NAMES:
        assert f"quality_{name}" in first.flat_record()
    assert isinstance(first.flat_record()["rejection_reasons"], str)


def test_bounding_box_dimensions() -> None:
    box = BoundingBox(x_min=2, y_min=4, x_max=7, y_max=10)
    assert box.width == 6 and box.height == 7
    assert box.to_list() == [2, 4, 7, 10]


def test_control_state_never_contains_robot_commands(disc_probability) -> None:
    """This repository reports perception state only; it issues no commands."""
    payload = extract_control_state(disc_probability).to_dict()
    forbidden = ("velocity", "force", "joint", "impedance", "torque", "command", "setpoint")
    offending = [key for key in payload if any(word in key.lower() for word in forbidden)]
    assert offending == []


def test_feature_extraction_can_drop_dense_arrays(disc_probability) -> None:
    config = FeatureExtractionConfig(keep_probability_map=False, keep_binary_mask=False)
    state = extract_control_state(disc_probability, config=config)
    assert state.probability_map is None
    assert state.binary_mask is None
    assert state.mask_area_px > 0, "geometry must still be computed"


def test_latencies_are_recorded_and_summed(disc_probability) -> None:
    state = extract_control_state(
        disc_probability,
        latencies={
            "preprocessing_latency_ms": 1.0,
            "inference_latency_ms": 5.0,
            "postprocessing_latency_ms": 0.5,
        },
    )
    assert state.inference_latency_ms == 5.0
    assert state.control_feature_latency_ms > 0
    assert state.end_to_end_latency_ms >= 6.5
    assert math.isfinite(state.end_to_end_latency_ms)


# ---------------------------------------------------------------------------
# Quality aggregation and the one-sided completeness term
# ---------------------------------------------------------------------------


def test_geometric_aggregation_is_dominated_by_the_worst_sub_score() -> None:
    """One collapsed term must be able to move Q, which an eight-term mean cannot.

    This is the property the arithmetic mean lacks: with the shipped weights a
    single sub-score falling to 0 shifts the arithmetic mean by at most its
    weight share, so Q stayed near 0.9 on frames whose lumen contrast had
    vanished entirely.
    """
    inputs = dict(
        segmentation_confidence=1.0,
        mask_area_ratio=0.07,
        largest_component_ratio=1.0,
        border_contact_ratio=0.0,
        lumen_surrounding_contrast=0.0,  # the collapsed term
        temporal_warped_iou=1.0,
        normalized_centroid_jump=0.0,
        relative_area_change=0.0,
    )
    arithmetic = compute_control_quality(
        config=QualityConfig(target_area_ratio=0.07, aggregation="arithmetic"), **inputs
    )
    geometric = compute_control_quality(
        config=QualityConfig(target_area_ratio=0.07, aggregation="geometric"), **inputs
    )
    assert arithmetic.components["lumen_contrast"] == 0.0
    assert geometric.score < arithmetic.score
    assert arithmetic.score > 0.9   # the failure is invisible
    assert geometric.score < 0.85   # the failure moves the score
    assert geometric.aggregation == "geometric"


def test_geometric_floor_stops_one_zero_from_erasing_the_score() -> None:
    """A single zero sub-score lowers Q sharply but must not annihilate it."""
    result = compute_control_quality(
        segmentation_confidence=1.0,
        mask_area_ratio=0.07,
        largest_component_ratio=1.0,
        border_contact_ratio=0.0,
        lumen_surrounding_contrast=0.0,
        config=QualityConfig(target_area_ratio=0.07, aggregation="geometric",
                             geometric_floor=0.05),
    )
    assert 0.0 < result.score < 1.0


def test_overfill_is_not_penalised_when_the_flag_is_off() -> None:
    """A hydro-extended bladder is the deployment target, not a fault.

    With ``penalise_overfill=True`` a mask far above the target band scores near
    zero on completeness, which on the PFUS test cohort pushed the best-
    segmented patient to the bottom of the quality ranking.
    """
    shared = dict(
        segmentation_confidence=1.0,
        largest_component_ratio=1.0,
        border_contact_ratio=0.0,
    )
    config = dict(target_area_ratio=0.068, area_tolerance=0.027)
    two_sided = compute_control_quality(
        mask_area_ratio=0.22, config=QualityConfig(**config, penalise_overfill=True), **shared
    )
    one_sided = compute_control_quality(
        mask_area_ratio=0.22, config=QualityConfig(**config, penalise_overfill=False), **shared
    )
    assert two_sided.components["mask_completeness"] < 0.1
    assert one_sided.components["mask_completeness"] == pytest.approx(1.0)

    # Under-filling is still penalised either way: a collapsed lumen is out of
    # domain whichever side of the band the flag protects.
    collapsed = compute_control_quality(
        mask_area_ratio=0.014, config=QualityConfig(**config, penalise_overfill=False), **shared
    )
    assert collapsed.components["mask_completeness"] < 0.5


@pytest.mark.parametrize("kwargs", [
    {"aggregation": "harmonic"},
    {"geometric_floor": 0.0},
    {"geometric_floor": 1.0},
])
def test_quality_config_rejects_bad_aggregation_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        QualityConfig(**kwargs)


def test_relative_area_change_is_zero_for_an_unchanged_mask_under_an_roi() -> None:
    """Current and reference areas must share one denominator.

    Passing the ROI to the current frame's geometry but not the reference's
    makes ``relative_area_change`` report ``(frame area / ROI area)`` worth of
    change between two identical masks -- 0.40 for the PFUS sector. It silently
    flattens the ``area_stability`` sub-score and fires the
    ``excessive_area_change`` validity criterion on frames that did not move.
    """
    roi = np.zeros((64, 64), bool)
    roi[8:56, 8:56] = True  # a sector covering 56% of the frame
    probability = np.where(square_mask() > 0, 0.95, 0.05).astype(np.float32)
    config = FeatureExtractionConfig()

    first = extract_control_state(probability, config=config, roi_mask=roi)
    second = extract_control_state(
        probability, config=config, roi_mask=roi, previous_state=first
    )

    assert second.relative_area_change == pytest.approx(0.0, abs=1e-9)
    assert second.quality_components["area_stability"] == pytest.approx(1.0)
