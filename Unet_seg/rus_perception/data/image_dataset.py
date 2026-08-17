"""Single-frame dataset used for spatial-only training, evaluation and inference."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import AugmentationConfig, PairedAugmentation
from .io import load_grayscale, load_mask, normalize_intensity, resize_image, resize_mask
from .manifest import Manifest, ManifestRecord

logger = logging.getLogger(__name__)

__all__ = ["UltrasoundFrameDataset"]


class UltrasoundFrameDataset(Dataset):
    """B-mode frames with optional binary lumen annotations.

    Every item is a dictionary with:

    ``image``
        ``1 x H x W`` float tensor.
    ``mask``
        ``1 x H x W`` float tensor in ``{0, 1}``. All zeros for unlabeled frames.
    ``labeled``
        ``1.0`` if the frame has an annotation, else ``0.0``. Use it as the
        ``sample_weight`` of the spatial loss so unlabeled frames contribute no
        segmentation supervision.
    ``frame_id``, ``patient_id``, ``sequence_id``, ``frame_index``, ``timestamp``
        Provenance carried through to logs and control output.

    Args:
        manifest: Source manifest, already filtered to the desired split.
        image_size: Target ``(height, width)``.
        augmentation: Augmentation config. ``None`` means **no augmentation**,
            which is the correct default for validation, test and inference;
            pass an explicit config to enable it for training.
        intensity_normalization: See :func:`rus_perception.data.io.normalize_intensity`.
        normalization_stats: ``{"mean": .., "std": ..}`` for ``mean_std``.
        labeled_only: Drop unlabeled frames.
        seed: Base augmentation seed.
    """

    def __init__(
        self,
        manifest: Manifest,
        image_size: tuple[int, int] = (128, 128),
        augmentation: Optional[AugmentationConfig] = None,
        intensity_normalization: str = "zero_one",
        normalization_stats: Optional[dict[str, float]] = None,
        labeled_only: bool = True,
        seed: int = 0,
    ) -> None:
        self.manifest = manifest.filter(labeled_only=labeled_only) if labeled_only else manifest
        if len(self.manifest) == 0:
            raise ValueError(
                "UltrasoundFrameDataset received an empty manifest. "
                + ("No labeled frames were found." if labeled_only else "")
            )
        self.image_size = (int(image_size[0]), int(image_size[1]))
        # Augmentation is opt-in: silently augmenting an evaluation set would
        # quietly corrupt every metric computed from it.
        self.augmentation = PairedAugmentation(
            augmentation if augmentation is not None else AugmentationConfig(enabled=False),
            seed=seed,
        )
        self.intensity_normalization = intensity_normalization
        self.normalization_stats = normalization_stats or {}
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.manifest)

    @property
    def records(self) -> list[ManifestRecord]:
        """The underlying manifest records, in chronological order."""
        return self.manifest.records

    def _load_frame(self, record: ManifestRecord) -> tuple[np.ndarray, Optional[np.ndarray]]:
        image_path = self.manifest.resolve(record.image_path)
        assert image_path is not None  # guaranteed by the manifest schema
        image = resize_image(load_grayscale(image_path), self.image_size)
        mask: Optional[np.ndarray] = None
        if record.is_labeled:
            mask_path = self.manifest.resolve(record.mask_path)
            assert mask_path is not None
            mask = resize_mask(load_mask(mask_path), self.image_size)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.manifest[index]
        image, mask = self._load_frame(record)

        # A single frame is augmented as a degenerate pair, which guarantees the
        # exact same geometry code path as the sequential dataset.
        augmented = self.augmentation(
            image_previous=image,
            image_current=image,
            mask_previous=mask,
            mask_current=mask,
            sample_seed=index + self.seed,
        )
        image = augmented.image_current
        mask = augmented.mask_current

        image = normalize_intensity(
            image,
            self.intensity_normalization,
            self.normalization_stats.get("mean"),
            self.normalization_stats.get("std"),
        )
        mask_array = np.zeros(self.image_size, dtype=np.float32) if mask is None else mask

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_array)).float().unsqueeze(0),
            "labeled": torch.tensor(1.0 if record.is_labeled else 0.0),
            "frame_id": record.frame_id,
            "patient_id": record.patient_id,
            "sequence_id": record.sequence_id,
            "frame_index": int(record.frame_index),
            "timestamp": float(record.timestamp) if record.timestamp is not None else -1.0,
        }
