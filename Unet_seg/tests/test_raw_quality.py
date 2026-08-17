"""Tests for the segmentation-independent raw B-mode quality score (``Q_raw``).

These tests pin the *structure* of ``Q_raw``: bounds, the meaning of ``None``,
the sign of each sub-score's response, gain invariance of the shadow test, the
reason codes, and the boundary that keeps this module out of robot control.

They deliberately do **not** assert that any default threshold is correct.
Every constant in :class:`RawQualityConfig` is provisional and cannot be fixed
until the ultrasound image geometry is known; asserting a value here would
freeze a guess into the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rus_perception.control.raw_quality import (  # noqa: E402
    RAW_QUALITY_COMPONENT_NAMES,
    RAW_REJECTION_REASONS,
    RawQualityConfig,
    ScanGeometryError,
    compute_raw_quality,
    sample_a_lines,
)

HEIGHT, WIDTH = 96, 64


def coupled_frame() -> np.ndarray:
    """A well-coupled B-mode frame: bright near field decaying with depth."""
    depth = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    # Bright at the probe face, attenuating with depth but never collapsing.
    frame = 0.55 * np.exp(-1.1 * depth) + 0.12
    return np.repeat(frame, WIDTH, axis=1).astype(np.float32)


def air_gap_frame(columns: slice) -> np.ndarray:
    """A coupled frame with an air gap killing a contiguous block of A-lines."""
    frame = coupled_frame()
    frame[:, columns] = 0.01
    return frame


def shadowed_frame(columns: slice) -> np.ndarray:
    """A coupled frame whose far field collapses behind some A-lines.

    The near field stays bright -- the probe is coupled -- but bowel gas or bone
    blocks the beam beyond a shallow depth. This is the failure ``shadow_penalty``
    exists to separate from a coupling failure.
    """
    frame = coupled_frame()
    frame[HEIGHT // 3 :, columns] = 0.005
    return frame


# -- structural guarantees ---------------------------------------------------


def test_score_and_every_subscore_are_bounded() -> None:
    result = compute_raw_quality(coupled_frame())
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert set(result.components) <= set(RAW_QUALITY_COMPONENT_NAMES)
    for name, value in result.components.items():
        assert 0.0 <= value <= 1.0, f"{name} left [0, 1]: {value}"


def test_a_well_coupled_frame_scores_above_an_air_gap() -> None:
    good = compute_raw_quality(coupled_frame())
    bad = compute_raw_quality(air_gap_frame(slice(0, WIDTH // 2)))
    assert good.score is not None and bad.score is not None
    assert good.score > bad.score
    assert good.dark_a_line_ratio == pytest.approx(0.0)
    assert bad.dark_a_line_ratio == pytest.approx(0.5)


def test_contact_continuity_counts_dead_a_lines_not_dim_pixels() -> None:
    """Halving the whole frame's brightness is not an air gap; killing 25% of
    the A-lines is. The two must not produce the same continuity score."""
    dimmed = compute_raw_quality(coupled_frame() * 0.5)
    gapped = compute_raw_quality(air_gap_frame(slice(0, WIDTH // 4)))
    assert dimmed.components["contact_continuity"] == pytest.approx(1.0)
    assert gapped.components["contact_continuity"] == pytest.approx(0.75)


def test_shadowing_is_detected_without_a_coupling_failure() -> None:
    result = compute_raw_quality(shadowed_frame(slice(0, WIDTH // 2)))
    assert result.shadowed_a_line_ratio == pytest.approx(0.5)
    # The near field is untouched, so coupling must still look fine.
    assert result.dark_a_line_ratio == pytest.approx(0.0)
    assert result.components["contact_continuity"] == pytest.approx(1.0)
    assert result.components["shadow_penalty"] == pytest.approx(0.5)


def test_shadow_detection_survives_a_global_gain_change() -> None:
    """The shadow test is relative to the frame's own far-field median, so a
    machine-side gain or TGC change must not be reported as new shadowing."""
    frame = shadowed_frame(slice(0, WIDTH // 4))
    full = compute_raw_quality(frame)
    dimmed = compute_raw_quality(frame * 0.4)
    assert dimmed.shadowed_a_line_ratio == pytest.approx(full.shadowed_a_line_ratio)


def test_shadowing_is_still_detected_when_most_a_lines_are_shadowed() -> None:
    """Regression: with a median reference, once past 50% the median becomes the
    shadow level itself and the test silently reports nothing -- going blind at
    exactly the point the frame is worst."""
    result = compute_raw_quality(shadowed_frame(slice(0, WIDTH * 3 // 4)))
    assert result.shadowed_a_line_ratio == pytest.approx(0.75)


def test_uniform_far_field_loss_is_caught_by_the_absolute_floor() -> None:
    """A relative test has no reference left when *every* line dies equally.
    The absolute floor is the only thing that sees this case."""
    frame = coupled_frame()
    frame[HEIGHT // 3 :, :] = 0.001
    result = compute_raw_quality(frame)
    assert result.shadowed_a_line_ratio == pytest.approx(1.0)
    assert "excessive_shadowing" in result.rejection_reasons


def test_shadow_reference_percentile_must_not_come_from_shadowed_lines() -> None:
    with pytest.raises(ValueError, match="shadow_reference_percentile"):
        RawQualityConfig(shadow_reference_percentile=25.0)


def test_unmeasurable_frame_scores_none_not_zero() -> None:
    """Too little ROI support means *not measured*. Returning 0.0 would be
    indistinguishable from a genuinely terrible frame and would drag a force
    level's mean quality down for a reason that has nothing to do with force."""
    roi = np.zeros((HEIGHT, WIDTH), dtype=bool)
    roi[:, :4] = True  # 4 A-lines, below the default min_valid_a_lines
    result = compute_raw_quality(coupled_frame(), roi_mask=roi)
    assert result.score is None
    assert result.components == {}
    assert result.rejection_reasons == []
    assert result.a_line_count == 4


def test_roi_mask_excludes_pixels_outside_the_imaged_sector() -> None:
    """A-lines outside the fan carry no echo and must not count as dark ones."""
    frame = coupled_frame()
    frame[:, : WIDTH // 2] = 0.0  # outside the fan: black, but not an air gap
    roi = np.zeros((HEIGHT, WIDTH), dtype=bool)
    roi[:, WIDTH // 2 :] = True
    masked = compute_raw_quality(frame, roi_mask=roi)
    unmasked = compute_raw_quality(frame)
    assert masked.dark_a_line_ratio == pytest.approx(0.0)
    assert unmasked.dark_a_line_ratio == pytest.approx(0.5)


# -- reason codes ------------------------------------------------------------


def test_an_empty_frame_reports_no_contact() -> None:
    result = compute_raw_quality(np.zeros((HEIGHT, WIDTH), dtype=np.float32))
    assert "no_contact" in result.rejection_reasons
    assert not result.usable_for_contact_search


def test_a_large_air_gap_reports_poor_acoustic_coupling() -> None:
    result = compute_raw_quality(air_gap_frame(slice(0, int(WIDTH * 0.6))))
    assert "poor_acoustic_coupling" in result.rejection_reasons
    assert not result.usable_for_contact_search


def test_widespread_shadowing_reports_excessive_shadowing() -> None:
    result = compute_raw_quality(shadowed_frame(slice(0, int(WIDTH * 0.8))))
    assert "excessive_shadowing" in result.rejection_reasons


def test_a_good_frame_is_usable_and_reports_nothing() -> None:
    result = compute_raw_quality(coupled_frame())
    assert result.rejection_reasons == []
    assert result.usable_for_contact_search


def test_every_emitted_reason_is_a_declared_code() -> None:
    frames = [
        coupled_frame(),
        np.zeros((HEIGHT, WIDTH), dtype=np.float32),
        air_gap_frame(slice(0, int(WIDTH * 0.6))),
        shadowed_frame(slice(0, int(WIDTH * 0.8))),
        coupled_frame() * 0.15,
    ]
    for frame in frames:
        for reason in compute_raw_quality(frame).rejection_reasons:
            assert reason in RAW_REJECTION_REASONS


def test_raw_reason_codes_do_not_collide_with_segmentation_reason_codes() -> None:
    """The supervisor consumes the union of both sets, so an ambiguous code
    would map one observation to two different recovery actions."""
    from rus_perception.control.validity import REJECTION_REASONS

    assert not set(RAW_REJECTION_REASONS) & set(REJECTION_REASONS)


# -- scan geometry -----------------------------------------------------------


def test_sector_geometry_raises_rather_than_mis_sampling() -> None:
    with pytest.raises(ScanGeometryError, match="not implemented"):
        sample_a_lines(coupled_frame(), scan_geometry="sector")
    with pytest.raises(ScanGeometryError):
        compute_raw_quality(coupled_frame(), config=RawQualityConfig(scan_geometry="sector"))


def test_linear_geometry_returns_the_image_unchanged() -> None:
    frame = coupled_frame()
    samples, support = sample_a_lines(frame)
    assert np.array_equal(samples, frame)
    assert support.all()


# -- input and configuration validation --------------------------------------


def test_unnormalized_image_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_raw_quality(coupled_frame() * 255.0)


def test_non_2d_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="2-D"):
        compute_raw_quality(np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32))


def test_mismatched_roi_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        compute_raw_quality(coupled_frame(), roi_mask=np.ones((8, 8), dtype=bool))


def test_config_rejects_unknown_keys_and_weights() -> None:
    with pytest.raises(ValueError, match="Unknown raw quality config key"):
        RawQualityConfig.from_dict({"near_field_frction": 0.2})
    with pytest.raises(ValueError, match="Unknown raw quality weight"):
        RawQualityConfig.from_dict({"weights": {"near_feild_echo": 1.0}})


def test_config_rejects_overlapping_depth_bands() -> None:
    with pytest.raises(ValueError, match="overlap"):
        RawQualityConfig(near_field_fraction=0.7, far_field_fraction=0.7)


def test_config_rejects_an_all_zero_weighting() -> None:
    with pytest.raises(ValueError, match="At least one"):
        RawQualityConfig(weights={name: 0.0 for name in RAW_QUALITY_COMPONENT_NAMES})


def test_a_component_can_be_disabled_by_zeroing_its_weight() -> None:
    config = RawQualityConfig.from_dict({"weights": {"total_echo_energy": 0.0}})
    result = compute_raw_quality(coupled_frame(), config=config)
    assert "total_echo_energy" in result.components  # still reported...
    assert "total_echo_energy" not in result.weights  # ...but not scored


def test_base_config_file_matches_the_dataclass_schema() -> None:
    """The shipped YAML must stay loadable; a renamed field has to fail here
    rather than in a running control loop."""
    yaml = pytest.importorskip("yaml")
    base = REPO_ROOT / "configs" / "_base.yaml"
    data = yaml.safe_load(base.read_text(encoding="utf-8"))
    section = data["control"]["raw_quality"]
    config = RawQualityConfig.from_dict(section)
    assert set(config.weights) == set(RAW_QUALITY_COMPONENT_NAMES)


# -- scope boundary ----------------------------------------------------------


def test_raw_quality_never_emits_a_robot_command() -> None:
    """Mirrors the same guarantee on ``ControlState``: this repository is the
    perception half of the loop and produces no actuation quantity."""
    result = compute_raw_quality(coupled_frame())
    forbidden = ("velocity", "force", "joint", "impedance", "torque", "command", "setpoint")
    fields = set(vars(result)) | set(result.components) | set(result.weights)
    for name in fields:
        assert not any(word in name.lower() for word in forbidden), name
