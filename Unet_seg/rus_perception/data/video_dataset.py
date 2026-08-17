"""Sequential (paired-frame) dataset for temporal-consistency training.

Each item is a temporally adjacent pair ``(frame_previous, frame_current)``
drawn from the *same* patient and sequence, in chronological order. Pairs are
formed even when one or both frames are unlabeled: the spatial loss is masked by
the per-frame ``labeled`` flag while the temporal loss applies to any pair with
usable flow.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..flow.base import FlowBackend, FlowPair
from ..flow.precomputed import load_flow_pair
from .augment import AugmentationConfig, PairedAugmentation
from .io import load_grayscale, load_mask, normalize_intensity, resize_image, resize_mask
from .manifest import Manifest, ManifestRecord

logger = logging.getLogger(__name__)

__all__ = ["SequentialUltrasoundDataset", "resize_flow"]


def resize_flow(flow: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a displacement field, rescaling the vectors with the geometry.

    A displacement of 4 pixels in a 256-wide image is 2 pixels in a 128-wide
    image, so the vectors must be scaled by the same factor as the grid.

    Args:
        flow: ``H x W x 2`` field.
        size: Target ``(height, width)``.

    Returns:
        The resized ``height x width x 2`` field.
    """
    import cv2

    height, width = size
    src_h, src_w = flow.shape[:2]
    if (src_h, src_w) == (height, width):
        return flow.astype(np.float32)
    resized = cv2.resize(flow.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[..., None]
    resized[..., 0] *= width / float(src_w)
    resized[..., 1] *= height / float(src_h)
    return resized.astype(np.float32)


class SequentialUltrasoundDataset(Dataset):
    """Temporally adjacent frame pairs with optional optical flow.

    Every item is a dictionary with:

    ``image_previous`` / ``image_current``
        ``1 x H x W`` float tensors.
    ``mask_previous`` / ``mask_current``
        ``1 x H x W`` float tensors; all zeros where no annotation exists.
    ``labeled_previous`` / ``labeled_current``
        ``1.0`` / ``0.0`` supervision flags for the spatial loss.
    ``flow_backward``
        ``2 x H x W`` current-grid field pointing into the previous frame -- the
        field consumed directly by ``warp_backward``.
    ``flow_forward``
        ``2 x H x W`` previous-grid field, used for consistency checking.
    ``flow_valid``
        ``1.0`` when real flow was loaded or computed, ``0.0`` when the pair has
        no usable flow. Pairs with ``0.0`` must be excluded from the temporal
        loss (``pair_valid``), never silently treated as zero motion.
    ``precomputed_reliability``
        ``1 x H x W`` map when the flow file carried one, else all ones.
    ``geometry_valid``
        ``1 x H x W`` mask of pixels filled from real content after augmentation.

    Args:
        manifest: Source manifest, already filtered to the desired split.
        image_size: Target ``(height, width)``.
        interval: Positional gap between the two frames of a pair, in frames.
        augmentation: Augmentation config. ``None`` means **no augmentation**,
            which is the correct default for validation, test and inference;
            pass an explicit config to enable it for training.
        intensity_normalization: See :func:`rus_perception.data.io.normalize_intensity`.
        normalization_stats: ``{"mean": .., "std": ..}`` for ``mean_std``.
        require_labeled_current: Keep only pairs whose current frame is labeled.
            Set ``False`` to exploit unlabeled video for the temporal loss.
        flow_backend: Optional backend used when the manifest has no flow paths.
            Computing flow on the fly is slow; precomputing is recommended.
        require_flow: Raise instead of yielding ``flow_valid = 0`` pairs.
        seed: Base augmentation seed.

    Raises:
        ValueError: If ``interval < 1`` or no valid pair can be formed.
    """

    def __init__(
        self,
        manifest: Manifest,
        image_size: tuple[int, int] = (128, 128),
        interval: int = 1,
        augmentation: Optional[AugmentationConfig] = None,
        intensity_normalization: str = "zero_one",
        normalization_stats: Optional[dict[str, float]] = None,
        require_labeled_current: bool = True,
        flow_backend: Optional[FlowBackend] = None,
        require_flow: bool = False,
        seed: int = 0,
    ) -> None:
        if interval < 1:
            raise ValueError(f"interval must be >= 1 frame, got {interval}.")
        self.manifest = manifest
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.interval = int(interval)
        # Augmentation is opt-in: silently augmenting an evaluation set would
        # quietly corrupt every metric computed from it.
        self.augmentation = PairedAugmentation(
            augmentation if augmentation is not None else AugmentationConfig(enabled=False),
            seed=seed,
        )
        self.intensity_normalization = intensity_normalization
        self.normalization_stats = normalization_stats or {}
        self.require_labeled_current = bool(require_labeled_current)
        self.flow_backend = flow_backend
        self.require_flow = bool(require_flow)
        self.seed = int(seed)

        manifest.validate_ordering()
        self.pairs: list[tuple[ManifestRecord, ManifestRecord]] = self._build_pairs()
        if not self.pairs:
            raise ValueError(
                f"No temporally adjacent pairs could be formed with interval={interval}. "
                "Check that sequences contain more than one frame and that "
                "require_labeled_current is not filtering everything out."
            )
        logger.info(
            "SequentialUltrasoundDataset: %d pairs from %d sequences (interval=%d)",
            len(self.pairs),
            len(manifest.sequences()),
            self.interval,
        )

    def _build_pairs(self) -> list[tuple[ManifestRecord, ManifestRecord]]:
        """Form ``(previous, current)`` pairs within each chronological sequence."""
        pairs: list[tuple[ManifestRecord, ManifestRecord]] = []
        for (patient, sequence), records in sorted(self.manifest.sequences().items()):
            if len(records) <= self.interval:
                logger.debug(
                    "Sequence %s/%s has %d frame(s), too short for interval %d; skipped.",
                    patient,
                    sequence,
                    len(records),
                    self.interval,
                )
                continue
            for position in range(self.interval, len(records)):
                previous = records[position - self.interval]
                current = records[position]
                if self.require_labeled_current and not current.is_labeled:
                    continue
                pairs.append((previous, current))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, record: ManifestRecord) -> tuple[np.ndarray, Optional[np.ndarray]]:
        image_path = self.manifest.resolve(record.image_path)
        assert image_path is not None
        image = resize_image(load_grayscale(image_path), self.image_size)
        mask: Optional[np.ndarray] = None
        if record.is_labeled:
            mask_path = self.manifest.resolve(record.mask_path)
            assert mask_path is not None
            mask = resize_mask(load_mask(mask_path), self.image_size)
        return image, mask

    def _load_flow(
        self,
        previous: ManifestRecord,
        current: ManifestRecord,
        image_previous: np.ndarray,
        image_current: np.ndarray,
    ) -> tuple[Optional[FlowPair], str]:
        """Return a flow pair for this frame pair and how it was obtained."""
        backward_path = self.manifest.resolve(current.flow_backward_path)
        forward_path = self.manifest.resolve(current.flow_forward_path)
        if backward_path is not None and backward_path.is_file():
            pair = load_flow_pair(backward_path)
            if forward_path is not None and forward_path != backward_path and forward_path.is_file():
                other = load_flow_pair(forward_path)
                pair = FlowPair(
                    forward=other.forward,
                    backward=pair.backward,
                    reliability=pair.reliability if pair.reliability is not None else other.reliability,
                    metadata={**other.metadata, **pair.metadata},
                )
            return pair, "precomputed"

        if self.flow_backend is not None:
            return self.flow_backend.compute(image_previous, image_current), self.flow_backend.name

        if self.require_flow:
            raise FileNotFoundError(
                f"No optical flow available for pair {previous.frame_id} -> {current.frame_id}. "
                "Run scripts/precompute_flow.py, or set flow.backend to compute it on the fly, "
                "or set require_flow: false."
            )
        return None, "missing"

    def __getitem__(self, index: int) -> dict[str, Any]:
        previous, current = self.pairs[index]
        image_previous, mask_previous = self._load(previous)
        image_current, mask_current = self._load(current)

        flow_pair, flow_source = self._load_flow(
            previous, current, image_previous, image_current
        )
        forward = backward = None
        reliability = None
        if flow_pair is not None:
            forward = resize_flow(flow_pair.forward, self.image_size)
            backward = resize_flow(flow_pair.backward, self.image_size)
            if flow_pair.reliability is not None:
                import cv2

                reliability = cv2.resize(
                    flow_pair.reliability.astype(np.float32),
                    (self.image_size[1], self.image_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float32)

        augmented = self.augmentation(
            image_previous=image_previous,
            image_current=image_current,
            mask_previous=mask_previous,
            mask_current=mask_current,
            flow_forward=forward,
            flow_backward=backward,
            sample_seed=index + self.seed,
        )

        def to_image(array: np.ndarray) -> torch.Tensor:
            normalized = normalize_intensity(
                array,
                self.intensity_normalization,
                self.normalization_stats.get("mean"),
                self.normalization_stats.get("std"),
            )
            return torch.from_numpy(np.ascontiguousarray(normalized)).float().unsqueeze(0)

        def to_mask(array: Optional[np.ndarray]) -> torch.Tensor:
            if array is None:
                array = np.zeros(self.image_size, dtype=np.float32)
            return torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)

        def to_flow(array: Optional[np.ndarray]) -> torch.Tensor:
            if array is None:
                return torch.zeros(2, *self.image_size, dtype=torch.float32)
            return torch.from_numpy(np.ascontiguousarray(array)).float().permute(2, 0, 1)

        geometry_valid = torch.from_numpy(
            np.ascontiguousarray(augmented.geometry_valid)
        ).float().unsqueeze(0)
        if reliability is None:
            precomputed_reliability = torch.ones(1, *self.image_size, dtype=torch.float32)
        else:
            precomputed_reliability = torch.from_numpy(
                np.ascontiguousarray(np.clip(reliability, 0.0, 1.0))
            ).float().unsqueeze(0)

        return {
            "image_previous": to_image(augmented.image_previous),
            "image_current": to_image(augmented.image_current),
            "mask_previous": to_mask(augmented.mask_previous),
            "mask_current": to_mask(augmented.mask_current),
            "labeled_previous": torch.tensor(1.0 if previous.is_labeled else 0.0),
            "labeled_current": torch.tensor(1.0 if current.is_labeled else 0.0),
            "flow_forward": to_flow(augmented.flow_forward),
            "flow_backward": to_flow(augmented.flow_backward),
            "flow_valid": torch.tensor(1.0 if flow_pair is not None else 0.0),
            "flow_source": flow_source,
            "precomputed_reliability": precomputed_reliability,
            "geometry_valid": geometry_valid,
            "frame_id": current.frame_id,
            "previous_frame_id": previous.frame_id,
            "patient_id": current.patient_id,
            "sequence_id": current.sequence_id,
            "frame_index": int(current.frame_index),
            "previous_frame_index": int(previous.frame_index),
            "timestamp": float(current.timestamp) if current.timestamp is not None else -1.0,
        }
