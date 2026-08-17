"""Reliability maps that gate where temporal consistency may be enforced.

Temporal agreement must not be demanded where the motion estimate is untrustworthy:
at occlusions, out-of-plane motion, shadowed regions, image borders, or where
speckle has decorrelated. This module turns the available evidence into a single
detached map bounded in ``[0, 1]``.

Signals combined (each contributes a soft score in ``[0, 1]``):

#. forward-backward flow consistency (also the occlusion proxy);
#. photometric reconstruction residual between the current frame and the
   motion-aligned previous frame;
#. in-bounds sampling mask;
#. flow-magnitude ceiling;
#. image-border exclusion;
#. optional local speckle decorrelation (windowed normalized cross-correlation).

Scores are combined multiplicatively, so the result is conservative: every
signal must agree before a pixel is considered reliable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from .warp import forward_backward_residual, warp_backward

__all__ = ["ReliabilityConfig", "soft_score", "compute_reliability", "ReliabilityOutput"]


@dataclass
class ReliabilityConfig:
    """Thresholds controlling the reliability map.

    Attributes:
        flow_consistency_threshold: Forward-backward residual, in pixels, at
            which a pixel is considered marginal.
        photometric_threshold: Absolute intensity residual (images scaled to
            ``[0, 1]``) at which photometric agreement is considered marginal.
        max_flow_magnitude: Displacement, in pixels, above which the flow is
            treated as implausible for adjacent ultrasound frames.
        border_exclusion_px: Width of the image border that is always
            unreliable, because flow is poorly constrained there.
        temperature: Softness of every score. ``0`` gives hard thresholds
            (``1`` below the threshold, ``0`` above); larger values give a
            smoother roll-off, with a residual equal to its threshold scoring
            ``exp(-1 / temperature)``.
        min_reliable_ratio: Fraction of reliable pixels a pair must reach before
            the temporal loss is applied to it at all.
        reliable_pixel_threshold: Reliability value above which a pixel counts
            towards ``min_reliable_ratio``.
        use_photometric: Enable the photometric-residual signal.
        use_speckle: Enable the local-NCC speckle-decorrelation signal.
        speckle_window: Window size (pixels) of the local NCC.
        speckle_threshold: NCC below this value is treated as decorrelated.
    """

    flow_consistency_threshold: float = 1.5
    photometric_threshold: float = 0.15
    max_flow_magnitude: float = 20.0
    border_exclusion_px: int = 4
    temperature: float = 0.5
    min_reliable_ratio: float = 0.1
    reliable_pixel_threshold: float = 0.5
    use_photometric: bool = True
    use_speckle: bool = False
    speckle_window: int = 9
    speckle_threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}.")
        if self.border_exclusion_px < 0:
            raise ValueError(
                f"border_exclusion_px must be >= 0, got {self.border_exclusion_px}."
            )
        for name in (
            "flow_consistency_threshold",
            "photometric_threshold",
            "max_flow_magnitude",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}.")
        if not 0.0 <= self.min_reliable_ratio <= 1.0:
            raise ValueError(
                f"min_reliable_ratio must be in [0, 1], got {self.min_reliable_ratio}."
            )
        if self.speckle_window < 3 or self.speckle_window % 2 == 0:
            raise ValueError(
                f"speckle_window must be an odd integer >= 3, got {self.speckle_window}."
            )

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ReliabilityConfig":
        """Build from a config mapping, rejecting unknown keys."""
        data = dict(data or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown reliability config key(s): {sorted(unknown)}. Known keys: {sorted(known)}"
            )
        return cls(**data)


@dataclass
class ReliabilityOutput:
    """Reliability map plus the individual signals, for logging and debugging."""

    reliability: torch.Tensor
    in_bounds: torch.Tensor
    flow_consistency: torch.Tensor
    photometric: Optional[torch.Tensor] = None
    speckle: Optional[torch.Tensor] = None
    components: dict[str, float] = field(default_factory=dict)

    @property
    def reliable_pixel_ratio(self) -> float:
        """Mean reliability over the batch (a convenience summary)."""
        return float(self.reliability.mean())


def soft_score(residual: torch.Tensor, threshold: float, temperature: float) -> torch.Tensor:
    """Map a non-negative residual to a score in ``[0, 1]``.

    With ``temperature == 0`` this is a hard indicator ``residual <= threshold``.
    Otherwise it is ``exp(-residual / (temperature * threshold))``, which equals
    1 at zero residual and decays smoothly; a residual equal to ``threshold``
    scores ``exp(-1 / temperature)``.

    Args:
        residual: Non-negative residual tensor.
        threshold: Positive scale at which the residual becomes marginal.
        temperature: Softness; ``0`` means a hard threshold.

    Returns:
        A tensor of the same shape bounded in ``[0, 1]``.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}.")
    if temperature <= 0:
        return (residual <= threshold).to(residual.dtype)
    return torch.exp(-residual.clamp_min(0) / (temperature * threshold))


def _border_mask(shape: tuple[int, int], width: int, device, dtype) -> torch.Tensor:
    """``1 x 1 x H x W`` mask that is 0 within ``width`` pixels of the border."""
    height, w = shape
    mask = torch.ones((1, 1, height, w), device=device, dtype=dtype)
    if width > 0:
        clamped = min(width, height // 2, w // 2)
        if clamped > 0:
            mask[:, :, :clamped, :] = 0
            mask[:, :, -clamped:, :] = 0
            mask[:, :, :, :clamped] = 0
            mask[:, :, :, -clamped:] = 0
    return mask


def _local_ncc(a: torch.Tensor, b: torch.Tensor, window: int, eps: float = 1e-5) -> torch.Tensor:
    """Windowed normalized cross-correlation of two ``B x 1 x H x W`` images.

    Low NCC between the current frame and the motion-aligned previous frame is
    evidence of speckle decorrelation (out-of-plane motion), which makes the
    temporal assumption invalid even when the flow looks self-consistent.

    Returns:
        NCC in ``[-1, 1]``, same shape as the inputs.
    """
    pad = window // 2
    kernel = (window, window)

    def box(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel, stride=1)

    mean_a, mean_b = box(a), box(b)
    var_a = box(a * a) - mean_a * mean_a
    var_b = box(b * b) - mean_b * mean_b
    cov = box(a * b) - mean_a * mean_b
    denominator = torch.sqrt(var_a.clamp_min(0) * var_b.clamp_min(0) + eps)
    return (cov / (denominator + eps)).clamp(-1.0, 1.0)


@torch.no_grad()
def compute_reliability(
    flow_backward: torch.Tensor,
    flow_forward: Optional[torch.Tensor] = None,
    image_current: Optional[torch.Tensor] = None,
    image_previous: Optional[torch.Tensor] = None,
    config: Optional[ReliabilityConfig] = None,
    align_corners: bool = False,
) -> ReliabilityOutput:
    """Build a detached reliability map in ``[0, 1]`` for a batch of frame pairs.

    Args:
        flow_backward: ``B x 2 x H x W`` current-grid backward flow.
        flow_forward: ``B x 2 x H x W`` previous-grid forward flow. Required for
            the forward-backward consistency signal; when absent that signal is
            skipped and a warning-free neutral score of 1 is used.
        image_current: ``B x 1 x H x W`` current frame in ``[0, 1]``, for the
            photometric signal.
        image_previous: ``B x 1 x H x W`` previous frame in ``[0, 1]``.
        config: Thresholds; defaults to :class:`ReliabilityConfig`.
        align_corners: ``grid_sample`` corner convention; must match the warp
            used by the temporal loss.

    Returns:
        A :class:`ReliabilityOutput`. The map is always detached -- it is
        evidence, not a differentiable quantity.

    Raises:
        ValueError: On shape mismatches.
    """
    config = config or ReliabilityConfig()
    if flow_backward.dim() != 4 or flow_backward.shape[1] != 2:
        raise ValueError(
            f"flow_backward must be B x 2 x H x W, got {tuple(flow_backward.shape)}."
        )

    flow_backward = flow_backward.detach().float()
    batch, _, height, width = flow_backward.shape
    device, dtype = flow_backward.device, flow_backward.dtype

    # (3) in-bounds sampling mask, obtained from the same grid the warp uses.
    zeros = torch.zeros((batch, 1, height, width), device=device, dtype=dtype)
    _, in_bounds = warp_backward(zeros, flow_backward, align_corners=align_corners)

    # (1) forward-backward consistency / occlusion proxy.
    if flow_forward is not None:
        residual = forward_backward_residual(
            flow_backward, flow_forward.detach().float(), align_corners
        )
        consistency = soft_score(
            residual, config.flow_consistency_threshold, config.temperature
        )
    else:
        consistency = torch.ones_like(in_bounds)

    # (4) flow-magnitude ceiling.
    magnitude = flow_backward.pow(2).sum(dim=1, keepdim=True).clamp_min(0).sqrt()
    magnitude_score = (magnitude <= config.max_flow_magnitude).to(dtype)

    # (5) border exclusion.
    border = _border_mask((height, width), config.border_exclusion_px, device, dtype)

    reliability = in_bounds * consistency * magnitude_score * border

    # (2) photometric residual on the motion-aligned previous frame.
    photometric_score: Optional[torch.Tensor] = None
    speckle_score: Optional[torch.Tensor] = None
    if (
        (config.use_photometric or config.use_speckle)
        and image_current is not None
        and image_previous is not None
    ):
        if image_current.shape != image_previous.shape:
            raise ValueError(
                f"image_current {tuple(image_current.shape)} and image_previous "
                f"{tuple(image_previous.shape)} must have the same shape."
            )
        if image_current.shape[-2:] != (height, width):
            raise ValueError(
                f"Images {tuple(image_current.shape[-2:])} must match the flow "
                f"resolution {(height, width)}."
            )
        warped_prev, _ = warp_backward(
            image_previous.detach().float(), flow_backward, align_corners=align_corners
        )
        if config.use_photometric:
            photometric_residual = (image_current.detach().float() - warped_prev).abs()
            if photometric_residual.shape[1] > 1:
                photometric_residual = photometric_residual.mean(dim=1, keepdim=True)
            photometric_score = soft_score(
                photometric_residual, config.photometric_threshold, config.temperature
            )
            reliability = reliability * photometric_score
        if config.use_speckle:
            current = image_current.detach().float()
            if current.shape[1] > 1:
                current = current.mean(dim=1, keepdim=True)
            reference = warped_prev.mean(dim=1, keepdim=True) if warped_prev.shape[1] > 1 else warped_prev
            ncc = _local_ncc(current, reference, config.speckle_window)
            # Decorrelation residual: how far NCC falls below the threshold.
            decorrelation = (config.speckle_threshold - ncc).clamp_min(0)
            speckle_score = soft_score(
                decorrelation, max(config.speckle_threshold, 1e-3), config.temperature
            )
            reliability = reliability * speckle_score

    reliability = reliability.clamp(0.0, 1.0).detach()

    components = {
        "reliability/mean": float(reliability.mean()),
        "reliability/in_bounds": float(in_bounds.mean()),
        "reliability/flow_consistency": float(consistency.mean()),
        "reliability/flow_magnitude": float(magnitude_score.mean()),
        "reliability/border": float(border.mean()),
        "reliability/reliable_pixel_ratio": float(
            (reliability >= config.reliable_pixel_threshold).float().mean()
        ),
    }
    if photometric_score is not None:
        components["reliability/photometric"] = float(photometric_score.mean())
    if speckle_score is not None:
        components["reliability/speckle"] = float(speckle_score.mean())

    return ReliabilityOutput(
        reliability=reliability,
        in_bounds=in_bounds,
        flow_consistency=consistency,
        photometric=photometric_score,
        speckle=speckle_score,
        components=components,
    )
