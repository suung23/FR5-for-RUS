"""Backward warping of dense maps with ``torch.nn.functional.grid_sample``.

Direction convention (this is the single most error-prone part of temporal
consistency, so it is stated explicitly and enforced by tests):

``flow_backward``
    A displacement field defined **on the current frame's pixel grid** that
    points **into the previous frame**::

        previous_position = current_position + flow_backward(current_position)

    It is exactly the sampling grid needed to pull the previous frame's content
    into the current frame's geometry. In OpenCV terms it is obtained by calling
    ``cv2.calcOpticalFlowFarneback(current, previous, ...)`` -- note the
    argument order.

``flow_forward``
    A displacement field defined **on the previous frame's pixel grid** pointing
    into the current frame, i.e. ``cv2.calcOpticalFlowFarneback(previous,
    current, ...)``. It is used for forward-backward consistency checking and
    must **not** be handed to :func:`warp_backward` directly.

Both fields store ``(dx, dy)`` in pixels in channel order ``[0] = dx``,
``[1] = dy``, with ``x`` increasing to the right and ``y`` increasing downward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["flow_to_sampling_grid", "warp_backward", "sample_flow", "forward_backward_residual"]


def _validate_flow(flow: torch.Tensor) -> None:
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError(
            f"Flow must be a B x 2 x H x W tensor with channel 0 = dx and channel "
            f"1 = dy, got shape {tuple(flow.shape)}."
        )
    if not torch.isfinite(flow).all():
        raise ValueError("Flow contains NaN or infinite values; reject it before warping.")


def flow_to_sampling_grid(
    flow: torch.Tensor, align_corners: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a pixel-space displacement field into a ``grid_sample`` grid.

    Args:
        flow: ``B x 2 x H x W`` displacement in pixels, defined on the grid of
            the frame being warped *into* (i.e. a backward flow).
        align_corners: Must match the value passed to ``grid_sample``.

    Returns:
        ``(grid, in_bounds)`` where ``grid`` is ``B x H x W x 2`` in normalized
        ``[-1, 1]`` coordinates and ``in_bounds`` is a ``B x 1 x H x W`` float
        mask that is 1 where the sampled source location lies inside the source
        image.

    Raises:
        ValueError: If ``flow`` has the wrong shape or is not finite.
    """
    _validate_flow(flow)
    _, _, height, width = flow.shape
    device, dtype = flow.device, flow.dtype

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    src_x = xs.unsqueeze(0) + flow[:, 0]
    src_y = ys.unsqueeze(0) + flow[:, 1]

    if align_corners:
        # Pixel centres map to [-1, 1] at the extreme *centres*.
        norm_x = 2.0 * src_x / max(width - 1, 1) - 1.0
        norm_y = 2.0 * src_y / max(height - 1, 1) - 1.0
    else:
        # Pixel centres map to [-1, 1] at the extreme *edges*.
        norm_x = (2.0 * src_x + 1.0) / width - 1.0
        norm_y = (2.0 * src_y + 1.0) / height - 1.0

    grid = torch.stack([norm_x, norm_y], dim=-1)
    in_bounds = (
        (src_x >= 0) & (src_x <= width - 1) & (src_y >= 0) & (src_y <= height - 1)
    )
    return grid, in_bounds.unsqueeze(1).to(dtype)


def warp_backward(
    source: torch.Tensor,
    flow_backward: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull ``source`` (previous frame) into the current frame's geometry.

    ``warped[b, :, y, x] = source[b, :, y + dy, x + dx]`` with
    ``(dx, dy) = flow_backward[b, :, y, x]``.

    Args:
        source: ``B x C x H x W`` map defined on the *previous* frame, e.g. the
            previous probability map or the previous image.
        flow_backward: ``B x 2 x H x W`` backward flow defined on the *current*
            frame's grid (see module docstring).
        mode: ``grid_sample`` interpolation mode.
        padding_mode: ``grid_sample`` padding mode.
        align_corners: ``grid_sample`` corner convention.

    Returns:
        ``(warped, in_bounds)``; ``in_bounds`` is a ``B x 1 x H x W`` float mask
        marking pixels whose source location was inside the image.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    if source.dim() != 4:
        raise ValueError(f"source must be B x C x H x W, got {tuple(source.shape)}.")
    if source.shape[-2:] != flow_backward.shape[-2:]:
        raise ValueError(
            f"source spatial size {tuple(source.shape[-2:])} must match the flow "
            f"{tuple(flow_backward.shape[-2:])}."
        )
    if source.shape[0] != flow_backward.shape[0]:
        raise ValueError(
            f"Batch mismatch: source has {source.shape[0]}, flow has {flow_backward.shape[0]}."
        )

    grid, in_bounds = flow_to_sampling_grid(flow_backward.to(source.dtype), align_corners)
    warped = F.grid_sample(
        source, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    return warped, in_bounds


def sample_flow(
    flow: torch.Tensor,
    at_flow: torch.Tensor,
    align_corners: bool = False,
) -> torch.Tensor:
    """Sample the field ``flow`` at the locations addressed by ``at_flow``.

    Used for forward-backward consistency: ``at_flow`` is the backward flow that
    tells where each current pixel lands in the previous frame, and ``flow`` is
    the forward flow defined on the previous frame's grid.

    Args:
        flow: ``B x 2 x H x W`` field defined on the source grid.
        at_flow: ``B x 2 x H x W`` backward flow addressing the source grid.
        align_corners: ``grid_sample`` corner convention.

    Returns:
        ``B x 2 x H x W`` sampled field.
    """
    sampled, _ = warp_backward(
        flow, at_flow, mode="bilinear", padding_mode="border", align_corners=align_corners
    )
    return sampled


def forward_backward_residual(
    flow_backward: torch.Tensor,
    flow_forward: torch.Tensor,
    align_corners: bool = False,
) -> torch.Tensor:
    """Per-pixel forward-backward consistency residual, in pixels.

    For a consistent (non-occluded, well-estimated) pixel, following the
    backward flow into the previous frame and then the forward flow back should
    return to the starting point, i.e.
    ``flow_backward(x) + flow_forward(x + flow_backward(x)) ~ 0``.

    Args:
        flow_backward: ``B x 2 x H x W`` current-grid backward flow.
        flow_forward: ``B x 2 x H x W`` previous-grid forward flow.
        align_corners: ``grid_sample`` corner convention.

    Returns:
        ``B x 1 x H x W`` residual magnitude in pixels.
    """
    _validate_flow(flow_backward)
    _validate_flow(flow_forward)
    resampled_forward = sample_flow(flow_forward, flow_backward, align_corners)
    residual = flow_backward + resampled_forward
    return residual.pow(2).sum(dim=1, keepdim=True).clamp_min(0).sqrt()
