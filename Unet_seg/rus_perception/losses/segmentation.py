"""Spatial segmentation losses.

The paper-inspired spatial objective is

.. math::

    L_{spatial} = w_{bce} L_{BCE} + w_{dice} L_{Dice} + w_{iou} L_{SoftJaccard}

``L_BCE`` consumes raw logits through :class:`torch.nn.BCEWithLogitsLoss` for
numerical stability, while the region losses consume sigmoid probabilities so
they stay differentiable (no thresholding anywhere).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "soft_dice_loss",
    "soft_jaccard_loss",
    "SpatialLossOutput",
    "SpatialSegmentationLoss",
]


def _flatten_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Reshape ``B x C x H x W`` (or ``B x H x W``) to ``B x N``."""
    return x.reshape(x.shape[0], -1)


def _check_shapes(probs: torch.Tensor, targets: torch.Tensor) -> None:
    if probs.shape != targets.shape:
        raise ValueError(
            f"Prediction and target shapes must match, got {tuple(probs.shape)} "
            f"and {tuple(targets.shape)}."
        )


def soft_dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-7,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Soft Dice loss ``1 - 2|P.T| / (|P| + |T|)`` computed per sample.

    The ``smooth`` constant appears in both numerator and denominator so that an
    empty prediction against an empty ground truth scores a *perfect* loss of 0
    rather than an undefined value. This is what makes all-zero masks safe.

    Args:
        probs: Predicted foreground probabilities in ``[0, 1]``.
        targets: Binary ground truth of the same shape.
        smooth: Laplace smoothing constant added to numerator and denominator.
        eps: Guard against division by zero.
        sample_weight: Optional ``B``-shaped weights (e.g. 1 for labeled frames,
            0 for unlabeled ones). The result is a weighted mean.

    Returns:
        Scalar loss in ``[0, 1]``.
    """
    _check_shapes(probs, targets)
    p = _flatten_per_sample(probs.float())
    t = _flatten_per_sample(targets.float())
    intersection = (p * t).sum(dim=1)
    denominator = p.sum(dim=1) + t.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth + eps)
    return _reduce(1.0 - dice, sample_weight)


def soft_jaccard_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-7,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Soft Jaccard (IoU) loss ``1 - |P.T| / (|P| + |T| - |P.T|)`` per sample.

    Args mirror :func:`soft_dice_loss`.

    Returns:
        Scalar loss in ``[0, 1]``.
    """
    _check_shapes(probs, targets)
    p = _flatten_per_sample(probs.float())
    t = _flatten_per_sample(targets.float())
    intersection = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - intersection
    jaccard = (intersection + smooth) / (union + smooth + eps)
    return _reduce(1.0 - jaccard, sample_weight)


def _reduce(per_sample: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
    """Weighted mean over the batch, returning 0 when all weights are zero."""
    if sample_weight is None:
        return per_sample.mean()
    weight = sample_weight.to(per_sample.dtype).reshape(-1)
    if weight.shape[0] != per_sample.shape[0]:
        raise ValueError(
            f"sample_weight has length {weight.shape[0]} but the batch size is "
            f"{per_sample.shape[0]}."
        )
    total = weight.sum()
    if float(total) <= 0.0:
        return per_sample.sum() * 0.0
    return (per_sample * weight).sum() / total


@dataclass
class SpatialLossOutput:
    """Total spatial loss together with its individually logged components."""

    total: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor
    jaccard: torch.Tensor
    num_supervised: int

    def as_log_dict(self) -> dict[str, float]:
        """Return float values suitable for structured logging."""
        return {
            "loss/spatial_total": float(self.total.detach()),
            "loss/bce": float(self.bce.detach()),
            "loss/dice": float(self.dice.detach()),
            "loss/jaccard": float(self.jaccard.detach()),
            "loss/num_supervised": float(self.num_supervised),
        }


class SpatialSegmentationLoss(nn.Module):
    """Weighted sum of BCE, Soft Dice and Soft Jaccard losses.

    Args:
        bce_weight: Weight of the binary cross-entropy term.
        dice_weight: Weight of the Soft Dice term.
        jaccard_weight: Weight of the Soft Jaccard term.
        smooth: Smoothing constant for the region losses.
        eps: Numerical guard for the region losses.
        pos_weight: Optional positive-class weight forwarded to
            ``BCEWithLogitsLoss``, useful for very small lumens.

    Raises:
        ValueError: If every weight is zero, which would silently disable
            supervision.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        jaccard_weight: float = 1.0,
        smooth: float = 1.0,
        eps: float = 1e-7,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        if min(bce_weight, dice_weight, jaccard_weight) < 0:
            raise ValueError("Spatial loss weights must be non-negative.")
        if bce_weight == dice_weight == jaccard_weight == 0:
            raise ValueError(
                "At least one spatial loss weight must be non-zero; otherwise the "
                "model receives no segmentation supervision."
            )
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.jaccard_weight = float(jaccard_weight)
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.tensor(float(pos_weight)),
            persistent=False,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> SpatialLossOutput:
        """Compute the spatial loss.

        Args:
            logits: Raw model output ``B x 1 x H x W``.
            targets: Binary ground truth in ``{0, 1}``, same shape as ``logits``.
            sample_weight: Optional ``B``-shaped mask selecting which samples are
                supervised. Frames without annotation should get weight 0.

        Returns:
            A :class:`SpatialLossOutput` with the total and each component.

        Raises:
            ValueError: If shapes disagree or targets leave ``[0, 1]``.
        """
        _check_shapes(logits, targets)
        targets = targets.to(logits.dtype)
        if torch.is_floating_point(targets):
            t_min, t_max = float(targets.min()), float(targets.max())
            if t_min < 0.0 or t_max > 1.0:
                raise ValueError(
                    f"Targets must lie in [0, 1] for binary segmentation, got "
                    f"range [{t_min}, {t_max}]."
                )

        num_supervised = (
            int(targets.shape[0])
            if sample_weight is None
            else int((sample_weight.reshape(-1) > 0).sum())
        )

        per_pixel_bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        bce = _reduce(_flatten_per_sample(per_pixel_bce).mean(dim=1), sample_weight)

        probs = torch.sigmoid(logits)
        dice = soft_dice_loss(probs, targets, self.smooth, self.eps, sample_weight)
        jaccard = soft_jaccard_loss(probs, targets, self.smooth, self.eps, sample_weight)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.jaccard_weight * jaccard
        )
        return SpatialLossOutput(
            total=total,
            bce=bce,
            dice=dice,
            jaccard=jaccard,
            num_supervised=num_supervised,
        )

    def extra_repr(self) -> str:
        return (
            f"bce_weight={self.bce_weight}, dice_weight={self.dice_weight}, "
            f"jaccard_weight={self.jaccard_weight}, smooth={self.smooth}"
        )
