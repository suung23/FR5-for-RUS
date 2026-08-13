"""Optical-flow tests: warp direction, reliability bounds and file validation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.flow.backends import FarnebackFlow, IdentityFlow, build_flow_backend
from src.flow.base import FlowPair
from src.flow.precomputed import (
    is_valid_flow_file,
    load_flow_pair,
    save_flow_pair,
)
from src.flow.reliability import ReliabilityConfig, compute_reliability, soft_score
from src.flow.warp import (
    flow_to_sampling_grid,
    forward_backward_residual,
    warp_backward,
)


def make_square(size: int, x: int, y: int, width: int = 4) -> torch.Tensor:
    """A ``1 x 1 x size x size`` tensor with a filled square at ``(x, y)``."""
    tensor = torch.zeros(1, 1, size, size)
    tensor[0, 0, y : y + width, x : x + width] = 1.0
    return tensor


# -- warp direction (the most error-prone part of temporal consistency) ----
@pytest.mark.parametrize("shift_x,shift_y", [(3, 0), (0, 2), (-4, 3), (5, -2)])
def test_backward_warp_reproduces_a_known_translation(shift_x: int, shift_y: int) -> None:
    """Object at (x, y) in the previous frame, at (x+dx, y+dy) in the current one.

    The backward flow must be ``(-dx, -dy)``: it points from the current frame
    back into the previous one.
    """
    size = 32
    start_x, start_y = 12, 12
    previous = make_square(size, start_x, start_y)
    current = make_square(size, start_x + shift_x, start_y + shift_y)

    flow_backward = torch.zeros(1, 2, size, size)
    flow_backward[0, 0] = -shift_x
    flow_backward[0, 1] = -shift_y

    warped, in_bounds = warp_backward(previous, flow_backward)
    assert torch.allclose(warped, current, atol=1e-6), "backward warp landed in the wrong place"
    assert in_bounds.min() >= 0.0 and in_bounds.max() <= 1.0


def test_forward_flow_used_as_a_sampling_grid_is_wrong() -> None:
    """Guards against the classic bug of feeding forward flow to the warp."""
    size = 32
    previous = make_square(size, 12, 12)
    current = make_square(size, 16, 12)

    forward = torch.zeros(1, 2, size, size)
    forward[0, 0] = 4.0  # previous -> current
    backward = -forward  # current -> previous

    correct, _ = warp_backward(previous, backward)
    wrong, _ = warp_backward(previous, forward)
    assert torch.allclose(correct, current, atol=1e-6)
    assert not torch.allclose(wrong, current, atol=1e-6)


def test_zero_flow_is_the_identity_warp() -> None:
    source = torch.rand(1, 1, 16, 16)
    warped, in_bounds = warp_backward(source, torch.zeros(1, 2, 16, 16))
    assert torch.allclose(warped, source, atol=1e-6)
    assert float(in_bounds.mean()) == pytest.approx(1.0)


def test_out_of_bounds_sampling_is_flagged_and_zero_filled() -> None:
    size = 16
    source = torch.ones(1, 1, size, size)
    flow = torch.zeros(1, 2, size, size)
    flow[0, 0] = 100.0  # far outside the image

    warped, in_bounds = warp_backward(source, flow)
    assert float(in_bounds.max()) == 0.0
    assert float(warped.abs().max()) == pytest.approx(0.0)


def test_warp_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="B x 2 x H x W"):
        warp_backward(torch.zeros(1, 1, 8, 8), torch.zeros(1, 3, 8, 8))
    with pytest.raises(ValueError, match="NaN or infinite"):
        flow_to_sampling_grid(torch.full((1, 2, 8, 8), float("nan")))
    with pytest.raises(ValueError, match="spatial size"):
        warp_backward(torch.zeros(1, 1, 8, 8), torch.zeros(1, 2, 4, 4))
    with pytest.raises(ValueError, match="Batch mismatch"):
        warp_backward(torch.zeros(2, 1, 8, 8), torch.zeros(1, 2, 8, 8))


def test_forward_backward_residual_is_zero_for_consistent_flow() -> None:
    size = 24
    forward = torch.zeros(1, 2, size, size)
    forward[0, 0] = 3.0
    backward = -forward
    residual = forward_backward_residual(backward, forward)
    # Borders resample outside the image; the interior must be consistent.
    assert float(residual[0, 0, 5:-5, 5:-5].abs().max()) < 1e-4


def test_forward_backward_residual_detects_inconsistency() -> None:
    size = 24
    forward = torch.zeros(1, 2, size, size)
    forward[0, 0] = 3.0
    inconsistent_backward = torch.zeros(1, 2, size, size)
    inconsistent_backward[0, 0] = 3.0  # should have been -3
    residual = forward_backward_residual(inconsistent_backward, forward)
    assert float(residual[0, 0, 5:-5, 5:-5].mean()) > 1.0


# -- reliability -----------------------------------------------------------
def test_reliability_is_bounded_in_zero_one() -> None:
    size = 32
    torch.manual_seed(0)
    flow_backward = torch.randn(1, 2, size, size) * 2
    flow_forward = -flow_backward
    output = compute_reliability(
        flow_backward,
        flow_forward,
        torch.rand(1, 1, size, size),
        torch.rand(1, 1, size, size),
        ReliabilityConfig(border_exclusion_px=2),
    )
    assert float(output.reliability.min()) >= 0.0
    assert float(output.reliability.max()) <= 1.0
    assert not output.reliability.requires_grad, "reliability must be detached evidence"


def test_reliability_excludes_the_image_border() -> None:
    size = 20
    config = ReliabilityConfig(border_exclusion_px=3)
    output = compute_reliability(torch.zeros(1, 2, size, size), config=config)
    assert float(output.reliability[0, 0, :3, :].max()) == 0.0
    assert float(output.reliability[0, 0, -3:, :].max()) == 0.0
    assert float(output.reliability[0, 0, :, :3].max()) == 0.0
    assert float(output.reliability[0, 0, 10, 10]) > 0.0


def test_reliability_penalises_implausibly_large_flow() -> None:
    size = 20
    config = ReliabilityConfig(max_flow_magnitude=5.0, border_exclusion_px=0)
    small = compute_reliability(torch.ones(1, 2, size, size), config=config)
    huge = compute_reliability(torch.full((1, 2, size, size), 50.0), config=config)
    assert float(small.reliability.mean()) > float(huge.reliability.mean())
    assert float(huge.reliability.max()) == 0.0


def test_reliability_penalises_photometric_disagreement() -> None:
    size = 24
    zero_flow = torch.zeros(1, 2, size, size)
    config = ReliabilityConfig(border_exclusion_px=0, temperature=0.5)
    image = torch.rand(1, 1, size, size)

    matching = compute_reliability(zero_flow, zero_flow, image, image.clone(), config)
    mismatching = compute_reliability(zero_flow, zero_flow, image, 1.0 - image, config)
    assert float(matching.reliability.mean()) > float(mismatching.reliability.mean())


def test_reliability_components_are_reported() -> None:
    output = compute_reliability(
        torch.zeros(1, 2, 16, 16),
        torch.zeros(1, 2, 16, 16),
        torch.rand(1, 1, 16, 16),
        torch.rand(1, 1, 16, 16),
        ReliabilityConfig(use_speckle=True, speckle_window=5),
    )
    for key in ("reliability/mean", "reliability/in_bounds", "reliability/photometric"):
        assert key in output.components
    assert "reliability/speckle" in output.components


def test_soft_score_behaviour() -> None:
    residual = torch.tensor([0.0, 1.0, 10.0])
    hard = soft_score(residual, threshold=1.0, temperature=0.0)
    assert hard.tolist() == [1.0, 1.0, 0.0]

    soft = soft_score(residual, threshold=1.0, temperature=1.0)
    assert float(soft[0]) == pytest.approx(1.0)
    assert 0.0 < float(soft[1]) < 1.0
    assert float(soft[2]) < float(soft[1])
    with pytest.raises(ValueError, match="threshold must be"):
        soft_score(residual, threshold=0.0, temperature=1.0)


def test_reliability_config_validation() -> None:
    with pytest.raises(ValueError, match="temperature must be"):
        ReliabilityConfig(temperature=-1)
    with pytest.raises(ValueError, match="speckle_window"):
        ReliabilityConfig(speckle_window=4)
    with pytest.raises(ValueError, match="Unknown reliability config key"):
        ReliabilityConfig.from_dict({"not_a_key": 1})


# -- backends --------------------------------------------------------------
def test_identity_backend_returns_zero_flow() -> None:
    backend = IdentityFlow(warn=False)
    pair = backend.compute(np.zeros((16, 16), np.float32), np.zeros((16, 16), np.float32))
    assert np.abs(pair.forward).max() == 0.0
    assert np.abs(pair.backward).max() == 0.0
    assert "warning" in pair.metadata


def test_farneback_recovers_the_sign_of_a_translation() -> None:
    """A rightward shift must yield a negative backward x-displacement."""
    rng = np.random.default_rng(0)
    previous = rng.random((64, 64)).astype(np.float32)
    previous = np.clip(previous, 0, 1)
    shift = 4
    current = np.roll(previous, shift, axis=1)

    pair = FarnebackFlow(winsize=21, levels=4, iterations=5).compute(previous, current)
    interior_backward_x = pair.backward[16:-16, 16:-16, 0]
    interior_forward_x = pair.forward[16:-16, 16:-16, 0]
    assert interior_backward_x.mean() < 0, "backward flow should point left"
    assert interior_forward_x.mean() > 0, "forward flow should point right"


def test_backends_validate_frame_shapes() -> None:
    backend = IdentityFlow(warn=False)
    with pytest.raises(ValueError, match="identical shape"):
        backend.compute(np.zeros((8, 8)), np.zeros((16, 16)))
    with pytest.raises(ValueError, match="2-D grayscale"):
        backend.compute(np.zeros((8, 8, 3)), np.zeros((8, 8, 3)))


def test_build_flow_backend_dispatch_and_errors() -> None:
    assert build_flow_backend({"backend": "farneback"}).name == "farneback"
    assert build_flow_backend({"backend": "identity", "params": {"warn": False}}).name == "identity"
    with pytest.raises(KeyError, match="Unknown flow backend"):
        build_flow_backend({"backend": "nope"})


# -- persistence -----------------------------------------------------------
def test_flow_round_trip_preserves_arrays_and_metadata(tmp_path) -> None:
    forward = np.random.default_rng(0).random((12, 10, 2)).astype(np.float32)
    backward = -forward
    reliability = np.full((12, 10), 0.5, np.float32)
    pair = FlowPair(forward, backward, reliability, {"backend": "test"})

    path = save_flow_pair(tmp_path / "flow.npz", pair, frame_id="a/b/1", previous_frame_id="a/b/0")
    loaded = load_flow_pair(path, expected_shape=(12, 10))

    assert np.allclose(loaded.forward, forward)
    assert np.allclose(loaded.backward, backward)
    assert np.allclose(loaded.reliability, reliability)
    assert loaded.metadata["frame_id"] == "a/b/1"
    assert "direction_convention" in loaded.metadata
    assert is_valid_flow_file(path, (12, 10))


def test_loading_rejects_corrupted_missing_and_mismatched_flow(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_flow_pair(tmp_path / "absent.npz")

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"this is not an npz archive")
    with pytest.raises(ValueError, match="corrupted"):
        load_flow_pair(corrupt)
    assert not is_valid_flow_file(corrupt)

    good = save_flow_pair(
        tmp_path / "flow.npz",
        FlowPair(np.zeros((8, 8, 2), np.float32), np.zeros((8, 8, 2), np.float32)),
    )
    with pytest.raises(ValueError, match="spatial size"):
        load_flow_pair(good, expected_shape=(16, 16))


def test_flow_pair_validates_its_arrays() -> None:
    with pytest.raises(ValueError, match="H x W x 2"):
        FlowPair(np.zeros((8, 8)), np.zeros((8, 8)))
    with pytest.raises(ValueError, match="same shape"):
        FlowPair(np.zeros((8, 8, 2)), np.zeros((4, 4, 2)))
    with pytest.raises(ValueError, match=r"bounded in \[0, 1\]"):
        FlowPair(np.zeros((8, 8, 2)), np.zeros((8, 8, 2)), np.full((8, 8), 2.0))


def test_incomplete_flow_archive_is_rejected(tmp_path) -> None:
    path = tmp_path / "partial.npz"
    np.savez(path, forward=np.zeros((4, 4, 2), np.float32))  # no 'backward'
    with pytest.raises(ValueError, match="incomplete"):
        load_flow_pair(path)
