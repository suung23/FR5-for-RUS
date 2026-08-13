"""Loss tests: spatial components, temporal consistency and gradient flow."""

from __future__ import annotations

import pytest
import torch

from src.flow.warp import warp_backward
from src.losses.segmentation import (
    SpatialSegmentationLoss,
    soft_dice_loss,
    soft_jaccard_loss,
)
from src.losses.temporal import (
    ControlFeatureTemporalLoss,
    PixelTemporalLoss,
    TemporalConsistencyLoss,
    charbonnier,
    robust_distance,
    soft_area,
    soft_centroid,
    temporal_weight_factor,
)
from src.models.slim_unet import SlimUNet

SHAPE = (2, 1, 16, 16)


def logits_for(probability: float, shape=SHAPE) -> torch.Tensor:
    """Return constant logits whose sigmoid equals ``probability``."""
    value = torch.logit(torch.tensor(probability).clamp(1e-6, 1 - 1e-6))
    return torch.full(shape, float(value))


# -- region losses ---------------------------------------------------------
def test_dice_and_jaccard_are_zero_for_a_perfect_prediction() -> None:
    target = torch.zeros(SHAPE)
    target[:, :, 4:12, 4:12] = 1.0
    assert soft_dice_loss(target, target) < 1e-4
    assert soft_jaccard_loss(target, target) < 1e-4


def test_region_losses_handle_all_zero_masks() -> None:
    """An empty prediction against empty ground truth is a perfect result, not NaN."""
    zeros = torch.zeros(SHAPE)
    dice = soft_dice_loss(zeros, zeros)
    jaccard = soft_jaccard_loss(zeros, zeros)
    assert torch.isfinite(dice) and torch.isfinite(jaccard)
    assert dice < 1e-6 and jaccard < 1e-6


def test_region_losses_handle_all_one_masks() -> None:
    ones = torch.ones(SHAPE)
    assert soft_dice_loss(ones, ones) < 1e-6
    assert soft_jaccard_loss(ones, ones) < 1e-6


def test_region_losses_are_near_one_for_a_completely_wrong_prediction() -> None:
    prediction = torch.ones(SHAPE)
    target = torch.zeros(SHAPE)
    assert soft_dice_loss(prediction, target) > 0.99
    assert soft_jaccard_loss(prediction, target) > 0.99


def test_region_losses_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        soft_dice_loss(torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 4, 4))


# -- combined spatial loss -------------------------------------------------
def test_spatial_loss_reports_every_component() -> None:
    criterion = SpatialSegmentationLoss()
    target = torch.zeros(SHAPE)
    target[:, :, 4:12, 4:12] = 1.0
    output = criterion(logits_for(0.5), target)

    assert set(output.as_log_dict()) == {
        "loss/spatial_total",
        "loss/bce",
        "loss/dice",
        "loss/jaccard",
        "loss/num_supervised",
    }
    assert torch.isfinite(output.total)
    expected = output.bce + output.dice + output.jaccard
    assert torch.allclose(output.total, expected)


def test_spatial_loss_is_near_zero_for_a_confident_correct_prediction() -> None:
    criterion = SpatialSegmentationLoss()
    target = torch.ones(SHAPE)
    output = criterion(logits_for(1 - 1e-6), target)
    assert float(output.total) < 0.01


def test_spatial_loss_is_larger_for_a_wrong_prediction() -> None:
    criterion = SpatialSegmentationLoss()
    target = torch.ones(SHAPE)
    correct = criterion(logits_for(0.99), target).total
    wrong = criterion(logits_for(0.01), target).total
    assert wrong > correct


def test_spatial_loss_masks_unlabeled_samples() -> None:
    """A mixed batch must ignore the unlabeled frame entirely."""
    criterion = SpatialSegmentationLoss()
    target = torch.zeros(SHAPE)
    target[0] = 1.0  # sample 0 labelled all-foreground
    logits = torch.stack([logits_for(0.99, SHAPE[1:]), logits_for(0.99, SHAPE[1:])])

    weights = torch.tensor([1.0, 0.0])
    supervised_only = criterion(logits, target, weights)
    both = criterion(logits, target)

    assert supervised_only.num_supervised == 1
    assert both.num_supervised == 2
    # Sample 1's "wrong" prediction is excluded, so the masked loss is lower.
    assert float(supervised_only.total) < float(both.total)


def test_spatial_loss_returns_zero_when_no_sample_is_supervised() -> None:
    criterion = SpatialSegmentationLoss()
    output = criterion(logits_for(0.5), torch.zeros(SHAPE), torch.zeros(SHAPE[0]))
    assert float(output.total) == pytest.approx(0.0)
    assert output.num_supervised == 0


def test_spatial_loss_configuration_validation() -> None:
    with pytest.raises(ValueError, match="At least one spatial loss weight"):
        SpatialSegmentationLoss(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        SpatialSegmentationLoss(-1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SpatialSegmentationLoss()(logits_for(0.5), torch.full(SHAPE, 2.0))


def test_spatial_loss_weights_are_applied() -> None:
    target = torch.zeros(SHAPE)
    target[:, :, 4:12, 4:12] = 1.0
    logits = logits_for(0.5)
    only_bce = SpatialSegmentationLoss(1.0, 0.0, 0.0)(logits, target)
    assert torch.allclose(only_bce.total, only_bce.bce)


def test_spatial_loss_gradients_are_finite() -> None:
    logits = logits_for(0.5).requires_grad_(True)
    target = torch.zeros(SHAPE)
    target[:, :, 4:12, 4:12] = 1.0
    SpatialSegmentationLoss()(logits, target).total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# -- robust penalties and descriptors --------------------------------------
def test_charbonnier_is_exactly_zero_at_zero() -> None:
    """The -eps offset matters: aligned identical maps must score exactly 0."""
    assert float(charbonnier(torch.zeros(4)).abs().max()) == pytest.approx(0.0, abs=1e-12)
    assert float(charbonnier(torch.tensor([0.5])).item()) > 0.0


def test_robust_distance_variants_are_monotone() -> None:
    small = torch.tensor([0.1])
    large = torch.tensor([1.0])
    for kind in ("charbonnier", "l1", "smooth_l1"):
        assert float(robust_distance(large, kind)) > float(robust_distance(small, kind))
    with pytest.raises(ValueError, match="Unknown robust kind"):
        robust_distance(small, "huber")  # type: ignore[arg-type]


def test_soft_centroid_matches_a_known_blob_position() -> None:
    probability = torch.zeros(1, 1, 20, 20)
    probability[0, 0, 4:8, 12:16] = 1.0  # centred on pixel (13.5, 5.5)
    centroid, mass = soft_centroid(probability)
    assert float(mass) == pytest.approx(16.0)
    assert float(centroid[0, 0]) == pytest.approx((13.5 + 0.5) / 20, abs=1e-5)
    assert float(centroid[0, 1]) == pytest.approx((5.5 + 0.5) / 20, abs=1e-5)


def test_soft_centroid_falls_back_to_image_centre_for_an_empty_map() -> None:
    centroid, mass = soft_centroid(torch.zeros(1, 1, 8, 8))
    assert float(mass) == pytest.approx(0.0)
    assert float(centroid[0, 0]) == pytest.approx(0.5)
    assert float(centroid[0, 1]) == pytest.approx(0.5)


def test_soft_area_equals_the_mean_probability() -> None:
    probability = torch.zeros(1, 1, 10, 10)
    probability[0, 0, :5, :] = 1.0
    assert float(soft_area(probability)) == pytest.approx(0.5)


# -- temporal warm-up ------------------------------------------------------
@pytest.mark.parametrize(
    "epoch,warmup,ramp,expected",
    [
        (0, 5, 0, 0.0),
        (4, 5, 0, 0.0),
        (5, 5, 0, 1.0),
        (5, 5, 5, 0.2),
        (9, 5, 5, 1.0),
        (20, 5, 5, 1.0),
        (0, 0, 0, 1.0),
    ],
)
def test_temporal_weight_factor(epoch, warmup, ramp, expected) -> None:
    assert temporal_weight_factor(epoch, warmup, ramp) == pytest.approx(expected)


def test_temporal_weight_factor_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        temporal_weight_factor(-1)


# -- pixel temporal loss ---------------------------------------------------
def test_temporal_loss_is_zero_for_aligned_identical_predictions() -> None:
    probability = torch.rand(2, 1, 16, 16)
    reliability = torch.ones_like(probability)
    loss, stats = PixelTemporalLoss()(probability, probability.clone(), reliability)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)
    assert float(stats["reliable_ratio"]) == pytest.approx(1.0)


def test_temporal_loss_increases_with_jitter() -> None:
    torch.manual_seed(0)
    probability = torch.rand(2, 1, 16, 16)
    reliability = torch.ones_like(probability)
    criterion = PixelTemporalLoss()

    small, _ = criterion(probability, (probability + 0.02).clamp(0, 1), reliability)
    large, _ = criterion(probability, (probability + 0.30).clamp(0, 1), reliability)
    assert float(large) > float(small) > 0.0


def test_reliability_masks_out_unreliable_regions() -> None:
    """Disagreement inside a zero-reliability region must not be penalised."""
    current = torch.zeros(1, 1, 8, 8)
    previous = torch.zeros(1, 1, 8, 8)
    previous[:, :, :4, :] = 1.0  # disagreement confined to the top half

    reliability = torch.ones(1, 1, 8, 8)
    reliability[:, :, :4, :] = 0.0  # ... which is declared unreliable

    criterion = PixelTemporalLoss(min_reliable_ratio=0.0)
    masked, _ = criterion(current, previous, reliability)
    unmasked, _ = criterion(current, previous, torch.ones(1, 1, 8, 8))
    assert float(masked) == pytest.approx(0.0, abs=1e-6)
    assert float(unmasked) > 0.1


def test_pairs_below_the_reliability_floor_are_skipped() -> None:
    current = torch.zeros(1, 1, 8, 8)
    previous = torch.ones(1, 1, 8, 8)
    reliability = torch.zeros(1, 1, 8, 8)
    reliability[:, :, 0, 0] = 1.0  # 1/64 reliable pixels

    loss, stats = PixelTemporalLoss(min_reliable_ratio=0.5)(current, previous, reliability)
    assert float(loss) == pytest.approx(0.0)
    assert int(stats["skipped"]) == 1
    assert int(stats["evaluated"]) == 0


def test_invalid_pairs_are_excluded_via_pair_valid() -> None:
    current = torch.zeros(2, 1, 8, 8)
    previous = torch.ones(2, 1, 8, 8)
    reliability = torch.ones(2, 1, 8, 8)
    loss, stats = PixelTemporalLoss()(
        current, previous, reliability, pair_valid=torch.tensor([1.0, 0.0])
    )
    assert int(stats["evaluated"]) == 1
    assert float(loss) > 0.0


def test_pixel_temporal_loss_validates_inputs() -> None:
    criterion = PixelTemporalLoss()
    ones = torch.ones(1, 1, 8, 8)
    with pytest.raises(ValueError, match="equal shape"):
        criterion(ones, torch.ones(1, 1, 4, 4), ones)
    with pytest.raises(ValueError, match="Reliability map shape"):
        criterion(ones, ones, torch.ones(1, 1, 4, 4))
    with pytest.raises(ValueError, match=r"bounded in \[0, 1\]"):
        criterion(ones, ones, ones * 2.0)


# -- control-feature temporal loss ----------------------------------------
def test_control_temporal_loss_is_zero_for_identical_predictions() -> None:
    probability = torch.zeros(1, 1, 32, 32)
    probability[0, 0, 8:20, 8:20] = 1.0
    total, centroid, area = ControlFeatureTemporalLoss()(probability, probability.clone())
    assert float(centroid) < 1e-3
    assert float(area) < 1e-6
    assert float(total) < 1e-3


def test_control_temporal_loss_detects_a_centroid_shift() -> None:
    previous = torch.zeros(1, 1, 32, 32)
    previous[0, 0, 8:20, 8:20] = 1.0
    current = torch.roll(previous, shifts=6, dims=3)
    _, centroid, area = ControlFeatureTemporalLoss()(current, previous)
    assert float(centroid) == pytest.approx(6 / 32, abs=0.02)
    assert float(area) < 1e-6  # the area is unchanged by a pure translation


def test_control_temporal_loss_detects_an_area_change() -> None:
    previous = torch.zeros(1, 1, 32, 32)
    previous[0, 0, 8:20, 8:20] = 1.0
    current = torch.zeros(1, 1, 32, 32)
    current[0, 0, 8:20, 8:14] = 1.0  # half the area, same vertical extent
    _, _, area = ControlFeatureTemporalLoss()(current, previous)
    assert float(area) > 0.2


def test_control_temporal_loss_skips_degenerate_empty_predictions() -> None:
    """Predicting nothing must not be a free way to satisfy the loss."""
    empty = torch.zeros(1, 1, 32, 32)
    total, _, _ = ControlFeatureTemporalLoss(min_mass=1.0)(empty, empty)
    assert float(total) == pytest.approx(0.0)


# -- combined temporal loss and backpropagation ---------------------------
def test_temporal_loss_respects_warmup_and_reports_diagnostics() -> None:
    criterion = TemporalConsistencyLoss(warmup_epochs=2, ramp_epochs=0)
    current = torch.zeros(1, 1, 16, 16, requires_grad=True)
    previous = torch.ones(1, 1, 16, 16)
    reliability = torch.ones(1, 1, 16, 16)

    during_warmup = criterion(current, previous, reliability, epoch=0)
    assert float(during_warmup.total.detach()) == pytest.approx(0.0)
    assert during_warmup.weight_factor == 0.0
    assert during_warmup.skipped_pairs == 1

    after_warmup = criterion(current, previous, reliability, epoch=2)
    assert after_warmup.weight_factor == 1.0
    assert float(after_warmup.total.detach()) > 0.0
    assert "temporal/reliable_pixel_ratio" in after_warmup.as_log_dict()


def test_disabled_temporal_loss_is_always_zero() -> None:
    criterion = TemporalConsistencyLoss(enabled=False)
    output = criterion(
        torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8), epoch=99
    )
    assert float(output.total) == pytest.approx(0.0)
    assert criterion.weight_at(99) == 0.0


def test_temporal_loss_backpropagates_into_slim_unet() -> None:
    """The whole point of the regulariser: gradients must reach the network."""
    torch.manual_seed(0)
    model = SlimUNet(channels=(8, 16), dropout=0.0)
    criterion = TemporalConsistencyLoss(warmup_epochs=0, lambda_pixel=1.0, lambda_control=1.0)

    previous_image = torch.rand(1, 1, 32, 32)
    current_image = torch.rand(1, 1, 32, 32)
    flow = torch.zeros(1, 2, 32, 32)
    flow[:, 0] = -2.0

    probability_previous = torch.sigmoid(model(previous_image))
    probability_current = torch.sigmoid(model(current_image))
    warped, _ = warp_backward(probability_previous, flow)

    output = criterion(
        probability_current, warped, torch.ones(1, 1, 32, 32), epoch=0
    )
    output.total.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "temporal loss produced no gradients at all"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_temporal_loss_gradients_reach_both_branches_when_not_detached() -> None:
    torch.manual_seed(0)
    model = SlimUNet(channels=(8, 16), dropout=0.0)
    criterion = TemporalConsistencyLoss(
        warmup_epochs=0,
        lambda_pixel=1.0,
        lambda_control=0.0,
        pixel_kwargs={"detach_target": False},
    )
    previous = torch.sigmoid(model(torch.rand(1, 1, 32, 32)))
    current = torch.sigmoid(model(torch.rand(1, 1, 32, 32)))
    criterion(current, previous, torch.ones(1, 1, 32, 32), epoch=0).total.backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters()
    )
