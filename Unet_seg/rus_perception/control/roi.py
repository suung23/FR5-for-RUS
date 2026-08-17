"""The imaged-sector ROI -- one definition, consumed by both quality functions.

An ultrasound frame is a rectangle, but the *imaged* region is not. A linear
probe fills a rectangle inset from the frame edges; a curvilinear or phased
probe fills a fan. Everything outside is not "dark tissue", it is **no
measurement at all**, and averaging over it silently corrupts any ratio whose
denominator is "the frame":

* ``mask_area_ratio = area / (H*W)`` makes the frame grabber's crop an input to
  the score. Include more black margin and the ratio falls, which moves
  ``mask_completeness``, which moves the force the search settles on. A cropping
  change should not move the optimal contact force.
* ``segmentation_confidence = mean(|2p-1|)`` over the frame is dominated by the
  trivially-easy dead region. A model that hedges badly at the lumen boundary
  still scores high, and that score is a live validity gate.
* ``border_contact_ratio`` counts perimeter on the *image* edge. For a fan
  inscribed in the frame the mask can never reach that edge, so the criterion
  silently never fires -- the opposite of the intended behaviour.

So the ROI is defined once, here, and passed to both
:func:`~rus_perception.control.features.extract_control_state` and
:func:`~rus_perception.control.raw_quality.compute_raw_quality`.

Status:
    ``mode: full`` is the default and reproduces the previous whole-frame
    behaviour **exactly** -- ``build_roi_mask`` returns ``None`` and no consumer
    allocates anything. Every other mode needs the ultrasound image geometry
    (probe type, depth scale, sector angles), which is not known yet. The
    parameters below are therefore plumbing, not settings: nothing here is
    calibrated.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ROI_MODES", "RoiConfig", "build_roi_mask", "roi_area_px", "RoiError"]

#: Every supported way of describing the imaged region.
ROI_MODES: tuple[str, ...] = ("full", "rect", "fan", "file")

RoiMode = Literal["full", "rect", "fan", "file"]


class RoiError(ValueError):
    """Raised when an ROI cannot be built from its configuration."""


@dataclass
class RoiConfig:
    """How to build the imaged-sector mask for a frame.

    All spatial parameters are **fractions of the frame**, never pixels, so one
    configuration survives a resolution change. ``x`` fractions are of the width,
    ``y`` and radius fractions are of the height.

    Attributes:
        mode: ``full`` -- the whole rectangle (default, no ROI).
            ``rect`` -- an axis-aligned inset rectangle; the usual choice for a
            linear probe whose image is a rectangle inside a larger frame.
            ``fan`` -- an annular sector; curvilinear and phased probes after
            scan conversion.
            ``file`` -- load a hand-drawn mask, for a geometry none of the above
            describes.
        rect: ``(x0, y0, x1, y1)`` fractions, for ``mode="rect"``.
        apex_xy: Sector apex as ``(x, y)`` fractions, for ``mode="fan"``. ``y``
            may be negative: the virtual apex of a curvilinear array usually sits
            *above* the top of the image.
        radius_range: ``(r_min, r_max)`` fractions of the frame height, measured
            from the apex, for ``mode="fan"``. ``r_min`` is the transducer face.
        half_angle_deg: Half of the sector's opening angle, for ``mode="fan"``.
        path: Mask image or ``.npy`` file, for ``mode="file"``. Any non-zero
            pixel is inside.
        erode_px: Shrink the finished mask by this many pixels. The edge of a
            scan-converted sector is interpolated and half-valid; eroding a few
            pixels keeps those samples out of every statistic.
        min_area_ratio: Refuse to build an ROI covering less than this fraction
            of the frame. A typo in the geometry that produces a sliver would
            otherwise surface as inexplicable quality scores rather than an error.
    """

    mode: RoiMode = "full"

    rect: Optional[Sequence[float]] = None

    apex_xy: Optional[Sequence[float]] = None
    radius_range: Optional[Sequence[float]] = None
    half_angle_deg: Optional[float] = None

    path: Optional[str] = None

    erode_px: int = 0
    min_area_ratio: float = 0.02

    def __post_init__(self) -> None:
        if self.mode not in ROI_MODES:
            raise RoiError(f"roi.mode must be one of {list(ROI_MODES)}, got {self.mode!r}.")
        if self.erode_px < 0:
            raise RoiError(f"roi.erode_px must be >= 0, got {self.erode_px}.")
        if not 0.0 <= self.min_area_ratio < 1.0:
            raise RoiError(
                f"roi.min_area_ratio must be in [0, 1), got {self.min_area_ratio}."
            )

        if self.mode == "rect":
            if self.rect is None or len(tuple(self.rect)) != 4:
                raise RoiError("roi.mode='rect' requires roi.rect = [x0, y0, x1, y1].")
            x0, y0, x1, y1 = (float(v) for v in self.rect)
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise RoiError(
                    f"roi.rect must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {self.rect!r}."
                )
        elif self.mode == "fan":
            missing = [
                name
                for name, value in (
                    ("apex_xy", self.apex_xy),
                    ("radius_range", self.radius_range),
                    ("half_angle_deg", self.half_angle_deg),
                )
                if value is None
            ]
            if missing:
                raise RoiError(
                    f"roi.mode='fan' requires {missing}. These come from the ultrasound "
                    "image geometry (apex position, transducer radius, sweep angle) and "
                    "cannot be guessed."
                )
            r_min, r_max = (float(v) for v in self.radius_range)  # type: ignore[arg-type]
            if not 0.0 <= r_min < r_max:
                raise RoiError(
                    f"roi.radius_range must satisfy 0 <= r_min < r_max, got {self.radius_range!r}."
                )
            if not 0.0 < float(self.half_angle_deg) <= 90.0:  # type: ignore[arg-type]
                raise RoiError(
                    f"roi.half_angle_deg must be in (0, 90], got {self.half_angle_deg}."
                )
        elif self.mode == "file" and not self.path:
            raise RoiError("roi.mode='file' requires roi.path.")

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "RoiConfig":
        """Build from a config mapping, rejecting unknown keys."""
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise RoiError(
                f"Unknown roi config key(s): {sorted(unknown)}. Known keys: {sorted(known)}"
            )
        return cls(**data)

    @property
    def is_full_frame(self) -> bool:
        """True when this configuration means "no ROI"."""
        return self.mode == "full"


def _rect_mask(shape: tuple[int, int], config: RoiConfig) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = (float(v) for v in config.rect)  # type: ignore[arg-type]
    mask = np.zeros(shape, dtype=bool)
    mask[
        int(round(y0 * height)) : max(int(round(y1 * height)), 1),
        int(round(x0 * width)) : max(int(round(x1 * width)), 1),
    ] = True
    return mask


def _fan_mask(shape: tuple[int, int], config: RoiConfig) -> np.ndarray:
    """Annular sector measured from the apex, opening downward (+y).

    Depth increases downward in a B-mode frame, so the sector's axis is ``+y``
    and the angle is measured from it. A pixel is inside when its distance from
    the apex is within ``radius_range`` and its angle off the axis is within
    ``half_angle_deg``.
    """
    height, width = shape
    apex_x, apex_y = (float(v) for v in config.apex_xy)  # type: ignore[arg-type]
    r_min, r_max = (float(v) for v in config.radius_range)  # type: ignore[arg-type]

    cx = apex_x * width
    cy = apex_y * height
    r_min_px = r_min * height
    r_max_px = r_max * height

    ys, xs = np.mgrid[0:height, 0:width]
    dx = xs.astype(np.float64) + 0.5 - cx
    dy = ys.astype(np.float64) + 0.5 - cy

    radius = np.hypot(dx, dy)
    # atan2(dx, dy): angle away from the downward axis, signed left/right.
    angle = np.abs(np.arctan2(dx, np.maximum(dy, 1e-9)))

    inside = (radius >= r_min_px) & (radius <= r_max_px)
    inside &= angle <= math.radians(float(config.half_angle_deg))  # type: ignore[arg-type]
    # Everything at or above the apex is behind the transducer face.
    inside &= dy > 0.0
    return inside


def _file_mask(shape: tuple[int, int], config: RoiConfig) -> np.ndarray:
    from pathlib import Path

    path = Path(str(config.path))
    if not path.is_file():
        raise RoiError(f"roi.path does not exist: {path}")

    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        # Imported lazily: the control package must stay importable without
        # Pillow, which only the data-loading path needs.
        from ..data.io import load_mask

        array = load_mask(path)

    array = np.asarray(array)
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape != shape:
        raise RoiError(
            f"ROI file {path.name} has shape {array.shape}, but the frame is {shape}. "
            "The ROI is a property of the acquisition geometry; resizing it would "
            "silently change which samples are measured."
        )
    return array > 0


def build_roi_mask(
    shape: tuple[int, int], config: Optional[RoiConfig] = None
) -> Optional[np.ndarray]:
    """Build the imaged-sector mask for a frame of ``shape``.

    Args:
        shape: ``(height, width)`` of the frame the mask must match.
        config: ROI description. ``None`` or ``mode="full"`` means the whole
            frame.

    Returns:
        An ``H x W`` boolean mask, or **``None`` for the whole frame**. ``None``
        is not a failure: it lets every consumer skip the mask entirely rather
        than allocate and multiply by an all-true array on the real-time path.

    Raises:
        RoiError: If the configuration is unusable, or the resulting ROI covers
            less than ``min_area_ratio`` of the frame.
    """
    config = config or RoiConfig()
    if config.is_full_frame:
        return None

    height, width = int(shape[0]), int(shape[1])
    if height < 1 or width < 1:
        raise RoiError(f"Frame shape must be positive, got {shape!r}.")

    builders = {"rect": _rect_mask, "fan": _fan_mask, "file": _file_mask}
    mask = builders[config.mode]((height, width), config)

    if config.erode_px > 0:
        import cv2

        size = 2 * config.erode_px + 1
        mask = (
            cv2.erode(
                mask.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )

    ratio = float(mask.sum()) / float(height * width)
    if ratio < config.min_area_ratio:
        raise RoiError(
            f"ROI mode={config.mode!r} covers {ratio:.4f} of the frame, below "
            f"min_area_ratio={config.min_area_ratio}. Check the geometry parameters: "
            "a sliver ROI would produce quality scores that look wrong for no "
            "visible reason."
        )
    logger.debug("Built %s ROI covering %.1f%% of the frame.", config.mode, 100.0 * ratio)
    return mask


def roi_area_px(shape: tuple[int, int], roi_mask: Optional[np.ndarray]) -> int:
    """Number of measured pixels: the ROI's area, or the whole frame when ``None``."""
    if roi_mask is None:
        return int(shape[0]) * int(shape[1])
    return int(np.count_nonzero(roi_mask))
