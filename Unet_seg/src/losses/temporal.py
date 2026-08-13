"""Motion-aligned temporal-consistency losses.

Temporal information is used **only as a training-time regulariser**. The
runtime model stays single-frame and stateless (see README, "Temporal-consistency
strategy"), so nothing in this module is executed during inference.

Two complementary terms are provided:

``L_temp_pixel``
    Reliability-weighted robust distance between the current probability map and
    the motion-aligned previous probability map.

``L_temp_control``
    Consistency of the differentiable descriptors a downstream controller
    actually consumes: normalized soft centroid and normalized soft area.

Both terms compare ``p_cur`` against ``warp(p_prev)`` -- never against the
unwarped ``p_prev`` -- so genuine probe/anatomy motion is not penalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

__all__ = [
    "charbonnier",
    "robust_distance",
    "soft_centroid",
    "soft_area",
    "temporal_weight_factor",
    "TemporalLossOutput",
    "PixelTemporalLoss",
    "ControlFeatureTemporalLoss",
    "TemporalConsistencyLoss",
    "RobustKind",
]

RobustKind = Literal["charbonnier", "l1", "smooth_l1"]


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier penalty ``sqrt(x^2 + eps^2) - eps``.

    The ``- eps`` offset makes the penalty exactly zero at ``x = 0``, so an
    identical, perfectly aligned pair produces a loss of 0 rather than a
    constant floor.
    """
    return torch.sqrt(x * x + eps * eps) - eps


def robust_distance(x: torch.Tensor, kind: RobustKind = "charbonnier", eps: float = 1e-3) -> torch.Tensor:
    """Element-wise robust penalty.

    Args:
        x: Residual tensor.
        kind: ``charbonnier``, ``l1`` or ``smooth_l1``.
        eps: Charbonnier epsilon, or the smooth-L1 transition point.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind == "charbonnier":
        return charbonnier(x, eps)
    if kind == "l1":
        return x.abs()
    if kind == "smooth_l1":
        beta = max(eps, 1e-8)
        absx = x.abs()
        return torch.where(absx < beta, 0.5 * x * x / beta, absx - 0.5 * beta)
    raise ValueError(f"Unknown robust kind {kind!r}. Expected: charbonnier, l1, smooth_l1.")


def soft_centroid(
    prob: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable normalized centroid of a probability map.

    Coordinates follow the repository convention: origin at the top-left pixel,
    ``x`` increasing to the right, ``y`` increasing downward, and the image
    spanning ``[0, 1]`` in both axes so the centre is ``(0.5, 0.5)``.

    Args:
        prob: Probability map ``B x 1 x H x W`` (or ``B x H x W``) in ``[0, 1]``.
        weight: Optional per-pixel weight of the same shape, e.g. a reliability
            map. Pixels with weight 0 do not contribute.
        eps: Guard against an all-zero mass.

    Returns:
        ``(centroid, mass)`` where ``centroid`` is ``B x 2`` holding ``(x, y)``
        in normalized units and ``mass`` is ``B`` holding the summed weighted
        probability. When ``mass`` is ~0 the centroid falls back to the image
        centre ``(0.5, 0.5)``, which keeps gradients finite.
    """
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    if prob.dim() != 4:
        raise ValueError(f"Expected B x 1 x H x W probabilities, got {tuple(prob.shape)}.")
    b, _, h, w = prob.shape
    p = prob.sum(dim=1)  # B x H x W
    if weight is not None:
        if weight.dim() == 4:
            weight = weight.sum(dim=1)
        p = p * weight

    device, dtype = p.device, p.dtype
    # Pixel centres mapped to [0, 1]; a 1-pixel-wide image maps to 0.5.
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h

    mass = p.reshape(b, -1).sum(dim=1)
    cx = (p.sum(dim=1) * xs).sum(dim=1) / (mass + eps)
    cy = (p.sum(dim=2) * ys).sum(dim=1) / (mass + eps)

    fallback = torch.full_like(cx, 0.5)
    valid = mass > eps
    cx = torch.where(valid, cx, fallback)
    cy = torch.where(valid, cy, fallback)
    return torch.stack([cx, cy], dim=1), mass


def soft_area(
    prob: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable normalized foreground area (mean probability).

    Args:
        prob: Probability map ``B x 1 x H x W`` in ``[0, 1]``.
        weight: Optional per-pixel weight; the area is then the weighted mean,
            normalized by the weight total so it stays comparable across frames.
        eps: Guard against an all-zero weight total.

    Returns:
        ``B``-shaped tensor of area ratios in ``[0, 1]``.
    """
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    b = prob.shape[0]
    p = prob.reshape(b, -1)
    if weight is None:
        return p.mean(dim=1)
    w = weight.reshape(b, -1).to(p.dtype)
    return (p * w).sum(dim=1) / (w.sum(dim=1) + eps)


def temporal_weight_factor(epoch: int, warmup_epochs: int = 0, ramp_epochs: int = 0) -> float:
    """Scale factor in ``[0, 1]`` applied to temporal losses at a given epoch.

    Args:
        epoch: Zero-based epoch index.
        warmup_epochs: Number of leading epochs with the temporal loss disabled,
            letting the spatial objective establish a usable segmentation first.
        ramp_epochs: Number of epochs over which the weight ramps linearly from
            0 to 1 after warm-up. ``0`` switches the loss on immediately.

    Returns:
        The multiplier for ``lambda_temp_*``.

    Raises:
        ValueError: If ``epoch``, ``warmup_epochs`` or ``ramp_epochs`` is negative.
    """
    if epoch < 0 or warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError(
            f"epoch, warmup_epochs and ramp_epochs must be >= 0, got "
            f"{epoch}, {warmup_epochs}, {ramp_epochs}."
        )
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return 1.0
    progress = (epoch - warmup_epochs + 1) / float(ramp_epochs)
    return float(min(1.0, max(0.0, progress)))


@dataclass
class TemporalLossOutput:
    """Temporal loss terms plus the diagnostics required for monitoring."""

    total: torch.Tensor
    pixel: torch.Tensor
    control: torch.Tensor
    centroid: torch.Tensor
    area: torch.Tensor
    raw_pixel_error: torch.Tensor
    weighted_pixel_error: torch.Tensor
    reliable_pixel_ratio: float
    skipped_pairs: int
    evaluated_pairs: int
    weight_factor: float

    def as_log_dict(self) -> dict[str, float]:
        """Return float values suitable for structured logging."""
        return {
            "loss/temporal_total": float(self.total.detach()),
            "loss/temporal_pixel": float(self.pixel.detach()),
            "loss/temporal_control": float(self.control.detach()),
            "loss/temporal_centroid": float(self.centroid.detach()),
            "loss/temporal_area": float(self.area.detach()),
            "temporal/raw_pixel_error": float(self.raw_pixel_error.detach()),
            "temporal/weighted_pixel_error": float(self.weighted_pixel_error.detach()),
            "temporal/reliable_pixel_ratio": float(self.reliable_pixel_ratio),
            "temporal/skipped_pairs": float(self.skipped_pairs),
            "temporal/evaluated_pairs": float(self.evaluated_pairs),
            "temporal/weight_factor": float(self.weight_factor),
        }


class PixelTemporalLoss(nn.Module):
    """Reliability-weighted robust distance between aligned probability maps.

    .. math::

        L_{temp}^{pixel} =
            \\frac{\\sum r \\cdot \\rho(p_{cur} - \\mathrm{warp}(p_{prev}))}
                 {\\sum r + \\epsilon}

    Args:
        robust_kind: Robust penalty family.
        charbonnier_eps: Epsilon of the robust penalty.
        min_reliable_ratio: Pairs whose reliable-pixel ratio falls below this
            value are skipped entirely, so the loss is never enforced through
            unusable flow.
        detach_target: Treat the motion-aligned previous prediction as a fixed
            target (stop-gradient). This is the default because the warped map
            already carries flow and interpolation error; letting gradients flow
            into it as well lets the model reduce the loss by blurring both
            predictions to agree on those artefacts. Set ``False`` for a fully
            symmetric consistency term.
        reliable_pixel_threshold: Reliability above which a pixel counts as
            "reliable" for the ratio statistic. The loss itself uses the
            continuous reliability as a weight.
        eps: Normalisation guard.
    """

    def __init__(
        self,
        robust_kind: RobustKind = "charbonnier",
        charbonnier_eps: float = 1e-3,
        min_reliable_ratio: float = 0.1,
        reliable_pixel_threshold: float = 0.5,
        detach_target: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 <= min_reliable_ratio <= 1.0:
            raise ValueError(f"min_reliable_ratio must be in [0, 1], got {min_reliable_ratio}.")
        self.robust_kind = robust_kind
        self.charbonnier_eps = float(charbonnier_eps)
        self.min_reliable_ratio = float(min_reliable_ratio)
        self.reliable_pixel_threshold = float(reliable_pixel_threshold)
        self.detach_target = bool(detach_target)
        self.eps = float(eps)

    def forward(
        self,
        prob_current: torch.Tensor,
        prob_previous_warped: torch.Tensor,
        reliability: torch.Tensor,
        pair_valid: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the pixel temporal loss.

        Args:
            prob_current: ``sigmoid(model(x_cur))``, ``B x 1 x H x W``.
            prob_previous_warped: ``warp(sigmoid(model(x_prev)))`` aligned into
                the current frame. Must be detached from the flow estimator.
            reliability: Detached reliability map in ``[0, 1]``, same shape.
            pair_valid: Optional ``B``-shaped mask of pairs that have usable flow.

        Returns:
            ``(loss, stats)``. ``stats`` holds ``raw_error``,
            ``weighted_error``, ``reliable_ratio``, ``skipped`` and
            ``evaluated`` as detached tensors.

        Raises:
            ValueError: On shape mismatch or out-of-range reliability.
        """
        if prob_current.shape != prob_previous_warped.shape:
            raise ValueError(
                f"Probability maps must have equal shape, got {tuple(prob_current.shape)} "
                f"and {tuple(prob_previous_warped.shape)}."
            )
        if reliability.shape != prob_current.shape:
            raise ValueError(
                f"Reliability map shape {tuple(reliability.shape)} must match the "
                f"probability maps {tuple(prob_current.shape)}."
            )
        if float(reliability.min()) < 0.0 or float(reliability.max()) > 1.0:
            raise ValueError("Reliability must be bounded in [0, 1].")

        batch = prob_current.shape[0]
        reliability = reliability.detach().to(prob_current.dtype)
        target = prob_previous_warped.detach() if self.detach_target else prob_previous_warped

        residual = prob_current - target
        penalty = robust_distance(residual, self.robust_kind, self.charbonnier_eps)

        flat_rel = reliability.reshape(batch, -1)
        flat_pen = penalty.reshape(batch, -1)
        reliable_ratio = (flat_rel >= self.reliable_pixel_threshold).to(flat_rel.dtype).mean(dim=1)

        usable = reliable_ratio >= self.min_reliable_ratio
        if pair_valid is not None:
            usable = usable & pair_valid.reshape(-1).to(torch.bool).to(usable.device)

        weight_sum = flat_rel.sum(dim=1)
        per_pair = (flat_rel * flat_pen).sum(dim=1) / (weight_sum + self.eps)
        raw_per_pair = flat_pen.mean(dim=1)

        usable_f = usable.to(per_pair.dtype)
        num_usable = usable_f.sum()
        if float(num_usable) == 0.0:
            loss = per_pair.sum() * 0.0
            weighted_error = loss.detach()
            raw_error = (raw_per_pair * 0.0).sum().detach()
        else:
            loss = (per_pair * usable_f).sum() / num_usable
            weighted_error = loss.detach()
            raw_error = ((raw_per_pair * usable_f).sum() / num_usable).detach()

        stats = {
            "raw_error": raw_error,
            "weighted_error": weighted_error,
            "reliable_ratio": reliable_ratio.mean().detach(),
            "skipped": torch.tensor(float(batch - int(num_usable)), device=loss.device),
            "evaluated": torch.tensor(float(int(num_usable)), device=loss.device),
        }
        return loss, stats


class ControlFeatureTemporalLoss(nn.Module):
    """Temporal consistency of the differentiable descriptors used for control.

    .. math::

        L_{temp}^{control} = w_c L_{centroid} + w_a L_{area}

    ``L_centroid`` is the Euclidean distance between the normalized soft
    centroids (so it is expressed in image widths/heights) and ``L_area`` is the
    symmetric relative difference of the normalized soft areas, which is bounded
    in ``[0, 1]`` and therefore cannot dominate the objective.

    Both descriptors are computed under the same reliability weighting so the
    current and motion-aligned previous predictions are compared over the same
    support.

    Args:
        centroid_weight: Weight of the centroid term.
        area_weight: Weight of the area term.
        min_mass: Minimum weighted probability mass required from *both* maps
            before the loss is applied. This prevents the degenerate
            "predict nothing everywhere" solution from scoring perfectly.
        detach_target: Stop-gradient on the motion-aligned previous descriptors.
        eps: Numerical guard.

    Note:
        Orientation, major-axis and eccentricity terms are intentionally not
        included; they are unstable for small, partially visible or nearly
        circular bladder masks.
    """

    def __init__(
        self,
        centroid_weight: float = 1.0,
        area_weight: float = 1.0,
        min_mass: float = 1.0,
        detach_target: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if centroid_weight < 0 or area_weight < 0:
            raise ValueError("Control temporal loss weights must be non-negative.")
        self.centroid_weight = float(centroid_weight)
        self.area_weight = float(area_weight)
        self.min_mass = float(min_mass)
        self.detach_target = bool(detach_target)
        self.eps = float(eps)

    def forward(
        self,
        prob_current: torch.Tensor,
        prob_previous_warped: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
        pair_valid: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the control-feature temporal loss.

        Args:
            prob_current: Current probability map ``B x 1 x H x W``.
            prob_previous_warped: Motion-aligned previous probability map.
            reliability: Optional detached reliability weighting.
            pair_valid: Optional ``B``-shaped validity mask.

        Returns:
            ``(total, centroid_term, area_term)`` as scalars.
        """
        target = prob_previous_warped.detach() if self.detach_target else prob_previous_warped
        weight = None if reliability is None else reliability.detach().to(prob_current.dtype)

        centroid_cur, mass_cur = soft_centroid(prob_current, weight, self.eps)
        centroid_prev, mass_prev = soft_centroid(target, weight, self.eps)
        area_cur = soft_area(prob_current, weight, self.eps)
        area_prev = soft_area(target, weight, self.eps)

        usable = (mass_cur >= self.min_mass) & (mass_prev >= self.min_mass)
        if pair_valid is not None:
            usable = usable & pair_valid.reshape(-1).to(torch.bool).to(usable.device)
        usable_f = usable.to(prob_current.dtype)
        num_usable = usable_f.sum()

        centroid_dist = torch.sqrt(
            ((centroid_cur - centroid_prev) ** 2).sum(dim=1) + self.eps * self.eps
        )
        area_diff = (area_cur - area_prev).abs() / (area_cur + area_prev + self.eps)

        if float(num_usable) == 0.0:
            zero = prob_current.sum() * 0.0
            return zero, zero, zero

        centroid_term = (centroid_dist * usable_f).sum() / num_usable
        area_term = (area_diff * usable_f).sum() / num_usable
        total = self.centroid_weight * centroid_term + self.area_weight * area_term
        return total, centroid_term, area_term


class TemporalConsistencyLoss(nn.Module):
    """Combined temporal objective with warm-up and linear ramping.

    ``lambda_temp_pixel * L_temp_pixel + lambda_temp_control * L_temp_control``,
    the whole sum scaled by :func:`temporal_weight_factor`.

    Args:
        lambda_pixel: Weight of the pixel term.
        lambda_control: Weight of the control-descriptor term.
        warmup_epochs: Epochs with the temporal loss disabled.
        ramp_epochs: Epochs over which the weight ramps to full strength.
        enabled: Master switch; when ``False`` the loss is a constant zero.
        pixel_kwargs / control_kwargs: Forwarded to the sub-losses.
    """

    def __init__(
        self,
        lambda_pixel: float = 0.2,
        lambda_control: float = 0.05,
        warmup_epochs: int = 5,
        ramp_epochs: int = 0,
        enabled: bool = True,
        pixel_kwargs: Optional[dict] = None,
        control_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if lambda_pixel < 0 or lambda_control < 0:
            raise ValueError("Temporal lambdas must be non-negative.")
        self.lambda_pixel = float(lambda_pixel)
        self.lambda_control = float(lambda_control)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.enabled = bool(enabled)
        self.pixel_loss = PixelTemporalLoss(**(pixel_kwargs or {}))
        self.control_loss = ControlFeatureTemporalLoss(**(control_kwargs or {}))

    def weight_at(self, epoch: int) -> float:
        """Return the scale factor applied to both temporal terms at ``epoch``."""
        if not self.enabled:
            return 0.0
        return temporal_weight_factor(epoch, self.warmup_epochs, self.ramp_epochs)

    def forward(
        self,
        prob_current: torch.Tensor,
        prob_previous_warped: torch.Tensor,
        reliability: torch.Tensor,
        epoch: int = 0,
        pair_valid: Optional[torch.Tensor] = None,
    ) -> TemporalLossOutput:
        """Compute all temporal terms for one batch of adjacent frame pairs.

        Args:
            prob_current: ``sigmoid`` of the current-frame logits.
            prob_previous_warped: Motion-aligned ``sigmoid`` of the previous
                frame's logits.
            reliability: Detached reliability map in ``[0, 1]``.
            epoch: Zero-based epoch index driving warm-up/ramping.
            pair_valid: Optional ``B``-shaped mask of usable pairs.

        Returns:
            A :class:`TemporalLossOutput`; ``total`` is already scaled by the
            lambdas and the warm-up factor and can be added to the spatial loss.
        """
        factor = self.weight_at(epoch)
        batch = prob_current.shape[0]

        if factor == 0.0:
            zero = prob_current.sum() * 0.0
            return TemporalLossOutput(
                total=zero,
                pixel=zero,
                control=zero,
                centroid=zero,
                area=zero,
                raw_pixel_error=zero.detach(),
                weighted_pixel_error=zero.detach(),
                reliable_pixel_ratio=0.0,
                skipped_pairs=batch,
                evaluated_pairs=0,
                weight_factor=0.0,
            )

        pixel, stats = self.pixel_loss(
            prob_current, prob_previous_warped, reliability, pair_valid
        )
        control, centroid, area = self.control_loss(
            prob_current, prob_previous_warped, reliability, pair_valid
        )
        total = factor * (self.lambda_pixel * pixel + self.lambda_control * control)

        return TemporalLossOutput(
            total=total,
            pixel=pixel,
            control=control,
            centroid=centroid,
            area=area,
            raw_pixel_error=stats["raw_error"],
            weighted_pixel_error=stats["weighted_error"],
            reliable_pixel_ratio=float(stats["reliable_ratio"]),
            skipped_pairs=int(stats["skipped"]),
            evaluated_pairs=int(stats["evaluated"]),
            weight_factor=factor,
        )
