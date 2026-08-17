"""Paired augmentation for temporally adjacent ultrasound frames.

Two rules make this module different from an ordinary augmentation pipeline:

#. **Geometric transforms are shared.** The previous frame, the current frame
   and their masks receive the *same* affine transform. Cropping, rotating,
   flipping or resizing the two frames independently would inject artificial
   motion that the temporal loss would then try to explain.
#. **Optical flow is transformed too.** A displacement field is not a passive
   image. Under a shared affine map ``x -> A x + t``, a displacement ``f``
   becomes ``A f`` and moves with the pixel it belongs to. Both effects are
   applied, otherwise the flow silently stops matching the augmented frames.

Photometric augmentation is configured separately and, by default, uses the
same parameters for both frames so the photometric reliability signal stays
meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = ["AugmentationConfig", "PairedAugmentation", "transform_flow_field"]


@dataclass
class AugmentationConfig:
    """Augmentation switches and ranges.

    Attributes:
        enabled: Master switch.
        horizontal_flip: Probability of a left-right flip.
        vertical_flip: Probability of an up-down flip.
        rotation_degrees: Maximum absolute rotation, sampled uniformly.
        scale_range: ``(min, max)`` isotropic scale factors.
        translate_ratio: Maximum translation as a fraction of image size.
        brightness: Maximum additive brightness offset (images in ``[0, 1]``).
        contrast: Maximum relative contrast change.
        gamma_range: ``(min, max)`` gamma exponents; ``(1, 1)`` disables it.
        gaussian_noise_std: Standard deviation of additive Gaussian noise.
        speckle_noise_std: Standard deviation of multiplicative speckle noise,
            which is the dominant noise model in B-mode ultrasound.
        independent_photometric: Sample photometric parameters separately for
            the two frames. Off by default because it weakens the photometric
            reliability signal used to gate the temporal loss.
    """

    enabled: bool = True
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    rotation_degrees: float = 10.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    translate_ratio: float = 0.05
    brightness: float = 0.1
    contrast: float = 0.1
    gamma_range: tuple[float, float] = (0.8, 1.25)
    gaussian_noise_std: float = 0.01
    speckle_noise_std: float = 0.0
    independent_photometric: bool = False

    def __post_init__(self) -> None:
        for name in ("horizontal_flip", "vertical_flip"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1], got {value}.")
        if self.rotation_degrees < 0:
            raise ValueError(f"rotation_degrees must be >= 0, got {self.rotation_degrees}.")
        if self.translate_ratio < 0:
            raise ValueError(f"translate_ratio must be >= 0, got {self.translate_ratio}.")
        self.scale_range = (float(self.scale_range[0]), float(self.scale_range[1]))
        if not 0 < self.scale_range[0] <= self.scale_range[1]:
            raise ValueError(f"scale_range must satisfy 0 < min <= max, got {self.scale_range}.")
        self.gamma_range = (float(self.gamma_range[0]), float(self.gamma_range[1]))
        if not 0 < self.gamma_range[0] <= self.gamma_range[1]:
            raise ValueError(f"gamma_range must satisfy 0 < min <= max, got {self.gamma_range}.")

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "AugmentationConfig":
        """Build from a config mapping, rejecting unknown keys."""
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown augmentation key(s): {sorted(unknown)}. Known keys: {sorted(known)}"
            )
        for key in ("scale_range", "gamma_range"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        return cls(**data)


@dataclass
class _GeometricParams:
    """Sampled affine parameters shared by every element of a frame pair."""

    matrix: np.ndarray  # 2x3 affine, maps source pixel -> destination pixel
    linear: np.ndarray  # 2x2 linear part, applied to displacement vectors
    identity: bool = True


@dataclass
class _PhotometricParams:
    brightness: float = 0.0
    contrast: float = 0.0
    gamma: float = 1.0
    noise_seed: int = 0
    gaussian_std: float = 0.0
    speckle_std: float = 0.0


@dataclass
class AugmentationOutput:
    """Augmented pair plus a mask of pixels that came from real source content."""

    image_previous: np.ndarray
    image_current: np.ndarray
    mask_previous: Optional[np.ndarray]
    mask_current: Optional[np.ndarray]
    flow_forward: Optional[np.ndarray]
    flow_backward: Optional[np.ndarray]
    geometry_valid: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)


def transform_flow_field(
    flow: np.ndarray, matrix: np.ndarray, linear: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Apply a shared affine transform to a dense displacement field.

    A displacement field changes in two ways under a shared affine map:
    its *support* moves with the image, and its *vectors* are rotated/scaled by
    the linear part of the map.

    Args:
        flow: ``H x W x 2`` field with channel 0 = ``dx``, channel 1 = ``dy``.
        matrix: ``2 x 3`` affine matrix mapping source to destination pixels.
        linear: ``2 x 2`` linear part of ``matrix``.
        size: Destination ``(height, width)``.

    Returns:
        The transformed ``H x W x 2`` field.

    Raises:
        ImportError: If OpenCV is unavailable.
    """
    import cv2

    height, width = size
    moved = cv2.warpAffine(
        flow.astype(np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0),
    )
    if moved.ndim == 2:  # OpenCV drops a trailing singleton dimension
        moved = moved[..., None]
    vectors = moved.reshape(-1, 2) @ linear.T
    return vectors.reshape(height, width, 2).astype(np.float32)


class PairedAugmentation:
    """Applies identical geometry (and, by default, identical photometry) to a pair.

    Args:
        config: Augmentation ranges and switches.
        seed: Base seed. Each call additionally mixes in a per-sample seed so
            augmentation is deterministic and reproducible.

    Note:
        Set ``config.enabled = False`` for validation and test data. Nothing in
        this class ever modifies a ground-truth mask other than resampling it
        with nearest-neighbour interpolation under the shared geometry.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None, seed: int = 0) -> None:
        self.config = config or AugmentationConfig()
        self.seed = int(seed)

    # -- parameter sampling ------------------------------------------------
    def _sample_geometry(self, rng: np.random.Generator, size: tuple[int, int]) -> _GeometricParams:
        height, width = size
        config = self.config

        angle = (
            float(rng.uniform(-config.rotation_degrees, config.rotation_degrees))
            if config.rotation_degrees > 0
            else 0.0
        )
        scale = (
            float(rng.uniform(*config.scale_range))
            if config.scale_range != (1.0, 1.0)
            else 1.0
        )
        flip_x = -1.0 if rng.random() < config.horizontal_flip else 1.0
        flip_y = -1.0 if rng.random() < config.vertical_flip else 1.0
        max_tx = config.translate_ratio * width
        max_ty = config.translate_ratio * height
        tx = float(rng.uniform(-max_tx, max_tx)) if max_tx > 0 else 0.0
        ty = float(rng.uniform(-max_ty, max_ty)) if max_ty > 0 else 0.0

        radians = math.radians(angle)
        cos, sin = math.cos(radians) * scale, math.sin(radians) * scale
        # Rotation/scale about the image centre, composed with axis flips.
        linear = np.array([[cos, -sin], [sin, cos]], dtype=np.float64) @ np.array(
            [[flip_x, 0.0], [0.0, flip_y]], dtype=np.float64
        )
        centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float64)
        translation = centre - linear @ centre + np.array([tx, ty], dtype=np.float64)
        matrix = np.concatenate([linear, translation[:, None]], axis=1).astype(np.float32)

        identity = (
            angle == 0.0
            and scale == 1.0
            and flip_x == 1.0
            and flip_y == 1.0
            and tx == 0.0
            and ty == 0.0
        )
        return _GeometricParams(matrix=matrix, linear=linear.astype(np.float32), identity=identity)

    def _sample_photometric(self, rng: np.random.Generator) -> _PhotometricParams:
        config = self.config
        return _PhotometricParams(
            brightness=float(rng.uniform(-config.brightness, config.brightness))
            if config.brightness > 0
            else 0.0,
            contrast=float(rng.uniform(-config.contrast, config.contrast))
            if config.contrast > 0
            else 0.0,
            gamma=float(rng.uniform(*config.gamma_range))
            if config.gamma_range != (1.0, 1.0)
            else 1.0,
            noise_seed=int(rng.integers(0, 2**31 - 1)),
            gaussian_std=config.gaussian_noise_std,
            speckle_std=config.speckle_noise_std,
        )

    # -- application -------------------------------------------------------
    @staticmethod
    def _warp_image(image: np.ndarray, params: _GeometricParams, size: tuple[int, int]) -> np.ndarray:
        import cv2

        height, width = size
        return cv2.warpAffine(
            image.astype(np.float32),
            params.matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).astype(np.float32)

    @staticmethod
    def _warp_mask(mask: np.ndarray, params: _GeometricParams, size: tuple[int, int]) -> np.ndarray:
        import cv2

        height, width = size
        warped = cv2.warpAffine(
            mask.astype(np.float32),
            params.matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return (warped > 0.5).astype(np.float32)

    @staticmethod
    def _apply_photometric(image: np.ndarray, params: _PhotometricParams) -> np.ndarray:
        out = image.astype(np.float32)
        if params.contrast != 0.0:
            mean = float(out.mean())
            out = (out - mean) * (1.0 + params.contrast) + mean
        if params.brightness != 0.0:
            out = out + params.brightness
        out = np.clip(out, 0.0, 1.0)
        if params.gamma != 1.0:
            out = np.power(out, params.gamma)
        if params.gaussian_std > 0 or params.speckle_std > 0:
            rng = np.random.default_rng(params.noise_seed)
            if params.speckle_std > 0:
                out = out * (1.0 + rng.normal(0.0, params.speckle_std, out.shape).astype(np.float32))
            if params.gaussian_std > 0:
                out = out + rng.normal(0.0, params.gaussian_std, out.shape).astype(np.float32)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def __call__(
        self,
        image_previous: np.ndarray,
        image_current: np.ndarray,
        mask_previous: Optional[np.ndarray] = None,
        mask_current: Optional[np.ndarray] = None,
        flow_forward: Optional[np.ndarray] = None,
        flow_backward: Optional[np.ndarray] = None,
        sample_seed: int = 0,
    ) -> AugmentationOutput:
        """Augment a temporally adjacent pair.

        Args:
            image_previous: ``H x W`` grayscale frame in ``[0, 1]``.
            image_current: ``H x W`` grayscale frame in ``[0, 1]``.
            mask_previous: Optional binary mask for the previous frame.
            mask_current: Optional binary mask for the current frame.
            flow_forward: Optional ``H x W x 2`` previous-grid forward flow.
            flow_backward: Optional ``H x W x 2`` current-grid backward flow.
            sample_seed: Per-sample seed mixed with the instance seed, so the
                same index yields the same augmentation in every epoch when the
                caller wants that.

        Returns:
            An :class:`AugmentationOutput`. ``geometry_valid`` marks pixels
            filled from real source content (0 in regions rotated in from
            outside the image); it should be multiplied into the reliability map.

        Raises:
            ValueError: If the two frames have different shapes.
        """
        if image_previous.shape != image_current.shape:
            raise ValueError(
                f"Paired frames must have identical shape, got {image_previous.shape} "
                f"and {image_current.shape}."
            )
        size = (int(image_previous.shape[0]), int(image_previous.shape[1]))

        if not self.config.enabled:
            return AugmentationOutput(
                image_previous=image_previous.astype(np.float32),
                image_current=image_current.astype(np.float32),
                mask_previous=mask_previous,
                mask_current=mask_current,
                flow_forward=flow_forward,
                flow_backward=flow_backward,
                geometry_valid=np.ones(size, dtype=np.float32),
                params={"enabled": False},
            )

        rng = np.random.default_rng((self.seed, int(sample_seed)))
        geometry = self._sample_geometry(rng, size)
        photo_a = self._sample_photometric(rng)
        photo_b = self._sample_photometric(rng) if self.config.independent_photometric else photo_a

        if geometry.identity:
            prev_img, cur_img = image_previous.astype(np.float32), image_current.astype(np.float32)
            prev_mask, cur_mask = mask_previous, mask_current
            fwd, bwd = flow_forward, flow_backward
            valid = np.ones(size, dtype=np.float32)
        else:
            prev_img = self._warp_image(image_previous, geometry, size)
            cur_img = self._warp_image(image_current, geometry, size)
            prev_mask = None if mask_previous is None else self._warp_mask(mask_previous, geometry, size)
            cur_mask = None if mask_current is None else self._warp_mask(mask_current, geometry, size)
            fwd = (
                None
                if flow_forward is None
                else transform_flow_field(flow_forward, geometry.matrix, geometry.linear, size)
            )
            bwd = (
                None
                if flow_backward is None
                else transform_flow_field(flow_backward, geometry.matrix, geometry.linear, size)
            )
            valid = self._warp_image(np.ones(size, dtype=np.float32), geometry, size)
            valid = (valid > 0.999).astype(np.float32)

        return AugmentationOutput(
            image_previous=self._apply_photometric(prev_img, photo_a),
            image_current=self._apply_photometric(cur_img, photo_b),
            mask_previous=prev_mask,
            mask_current=cur_mask,
            flow_forward=fwd,
            flow_backward=bwd,
            geometry_valid=valid,
            params={
                "enabled": True,
                "geometry_identity": geometry.identity,
                "matrix": geometry.matrix.tolist(),
                "brightness": photo_a.brightness,
                "contrast": photo_a.contrast,
                "gamma": photo_a.gamma,
            },
        )
