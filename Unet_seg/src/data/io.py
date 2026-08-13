"""Image and mask loading helpers shared by the dataset classes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

__all__ = ["load_grayscale", "load_mask", "resize_image", "resize_mask", "normalize_intensity"]


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load an image as ``float32`` grayscale in ``[0, 1]``.

    ``.npy`` files are accepted for pre-extracted frames; everything else is
    read through Pillow and converted to single-channel.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file cannot be interpreted as a 2-D image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix == ".npy":
        array = np.load(path)
    else:
        array = np.asarray(Image.open(path).convert("L"))

    array = np.asarray(array)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image at {path}, got shape {array.shape}.")

    array = array.astype(np.float32)
    if array.max() > 1.0 + 1e-6:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def load_mask(path: str | Path, threshold: float = 0.5) -> np.ndarray:
    """Load a binary annotation as ``float32`` values in ``{0.0, 1.0}``.

    Any non-zero label is treated as foreground, so 0/1, 0/255 and boolean masks
    all work. Ground truth is never modified beyond this binarisation.

    Args:
        path: Mask file.
        threshold: Applied after scaling the mask to ``[0, 1]``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the mask is not 2-D.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask not found: {path}")

    if path.suffix == ".npy":
        array = np.load(path)
    else:
        array = np.asarray(Image.open(path))

    array = np.asarray(array)
    if array.ndim == 3:
        array = array.max(axis=2)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask at {path}, got shape {array.shape}.")

    array = array.astype(np.float32)
    if array.max() > 1.0 + 1e-6:
        array = array / 255.0
    return (array > threshold).astype(np.float32)


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a grayscale image to ``(height, width)`` with bilinear sampling."""
    height, width = size
    if image.shape == (height, width):
        return image
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
    resized = pil.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to ``(height, width)`` with nearest-neighbour sampling.

    Nearest-neighbour keeps the mask binary; no label smoothing or dilation is
    introduced by resizing.
    """
    height, width = size
    if mask.shape == (height, width):
        return mask
    pil = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
    resized = pil.resize((width, height), resample=Image.NEAREST)
    return (np.asarray(resized) > 127).astype(np.float32)


def normalize_intensity(
    image: np.ndarray,
    mode: str = "zero_one",
    mean: Optional[float] = None,
    std: Optional[float] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalise pixel intensities.

    Args:
        image: Grayscale image already scaled to ``[0, 1]``.
        mode: ``zero_one`` (identity), ``mean_std`` (dataset statistics) or
            ``per_image`` (per-frame standardisation, robust to gain/TGC changes
            between machines).
        mean: Dataset mean, required for ``mean_std``.
        std: Dataset standard deviation, required for ``mean_std``.
        eps: Guard against zero variance.

    Raises:
        ValueError: If ``mode`` is unknown or required statistics are missing.
    """
    if mode == "zero_one":
        return image
    if mode == "per_image":
        return (image - float(image.mean())) / (float(image.std()) + eps)
    if mode == "mean_std":
        if mean is None or std is None:
            raise ValueError(
                "intensity_normalization 'mean_std' requires 'mean' and 'std' in the config."
            )
        return (image - float(mean)) / (float(std) + eps)
    raise ValueError(
        f"Unknown intensity normalization {mode!r}. Expected: zero_one, mean_std, per_image."
    )
