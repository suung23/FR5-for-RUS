"""Synthetic ultrasound-like sequence generator.

Every test in this repository runs on synthetic data: no private clinical data is
required to exercise the training, evaluation, flow and monitoring pipelines.

The generator imitates the *structure* a bladder B-mode frame has -- a dark
(anechoic) lumen with a brighter boundary, multiplicative speckle, a depth
gain gradient and occasional acoustic shadowing -- so that shapes, intensity
ranges and temporal behaviour are realistic enough to exercise the code. It is
**not** a physical ultrasound simulator and must never be used to make claims
about clinical accuracy.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .manifest import Manifest, ManifestRecord, write_manifest

logger = logging.getLogger(__name__)

__all__ = ["SyntheticSequenceConfig", "generate_frame", "generate_dataset"]


class SyntheticSequenceConfig:
    """Parameters of a synthetic ultrasound sequence.

    Args:
        num_patients: Number of distinct subjects.
        sequences_per_patient: Sweeps per subject.
        frames_per_sequence: Frames per sweep.
        image_size: ``(height, width)`` of generated frames.
        labeled_fraction: Fraction of frames that receive an annotation. Values
            below 1 produce unlabeled frames, exercising the semi-supervised path.
        motion_px_per_frame: Lumen translation per frame, in pixels.
        speckle_std: Multiplicative speckle strength.
        shadow_probability: Probability that a frame contains a shadow band.
        frame_rate: Frames per second, used to fill the ``timestamp`` column.
    """

    def __init__(
        self,
        num_patients: int = 4,
        sequences_per_patient: int = 1,
        frames_per_sequence: int = 8,
        image_size: tuple[int, int] = (64, 64),
        labeled_fraction: float = 1.0,
        motion_px_per_frame: float = 1.5,
        speckle_std: float = 0.15,
        shadow_probability: float = 0.15,
        frame_rate: float = 20.0,
    ) -> None:
        if num_patients < 1 or sequences_per_patient < 1 or frames_per_sequence < 1:
            raise ValueError("Synthetic dataset dimensions must all be >= 1.")
        if not 0.0 <= labeled_fraction <= 1.0:
            raise ValueError(f"labeled_fraction must be in [0, 1], got {labeled_fraction}.")
        if frame_rate <= 0:
            raise ValueError(f"frame_rate must be > 0, got {frame_rate}.")
        self.num_patients = int(num_patients)
        self.sequences_per_patient = int(sequences_per_patient)
        self.frames_per_sequence = int(frames_per_sequence)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.labeled_fraction = float(labeled_fraction)
        self.motion_px_per_frame = float(motion_px_per_frame)
        self.speckle_std = float(speckle_std)
        self.shadow_probability = float(shadow_probability)
        self.frame_rate = float(frame_rate)


def generate_frame(
    size: tuple[int, int],
    centre: tuple[float, float],
    radii: tuple[float, float],
    angle_degrees: float,
    rng: np.random.Generator,
    speckle_std: float = 0.15,
    shadow: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one synthetic B-mode frame and its lumen mask.

    Args:
        size: ``(height, width)``.
        centre: Lumen centre ``(x, y)`` in pixels.
        radii: Ellipse semi-axes ``(rx, ry)`` in pixels.
        angle_degrees: Ellipse rotation.
        rng: Random generator driving speckle and noise.
        speckle_std: Multiplicative speckle strength.
        shadow: Add a vertical acoustic-shadow band.

    Returns:
        ``(image, mask)``; the image is ``float32`` in ``[0, 1]`` and the mask is
        ``float32`` in ``{0, 1}`` marking the lumen **interior** (the annotation
        policy of this repository -- no boundary dilation is applied).
    """
    height, width = size
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)

    radians = math.radians(angle_degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    dx, dy = xs - centre[0], ys - centre[1]
    u = (dx * cos + dy * sin) / max(radii[0], 1e-3)
    v = (-dx * sin + dy * cos) / max(radii[1], 1e-3)
    radial = np.sqrt(u * u + v * v)

    # Tissue background with a depth-dependent gain gradient.
    background = 0.45 + 0.25 * (1.0 - ys / max(height - 1, 1))
    image = background.astype(np.float32)

    # Anechoic lumen interior and a brighter specular boundary.
    interior = radial < 1.0
    boundary = (radial >= 1.0) & (radial < 1.18)
    image = np.where(interior, 0.05 + 0.03 * radial, image)
    image = np.where(boundary, np.minimum(image + 0.35, 1.0), image)

    if shadow:
        column = int(rng.integers(0, max(width - 1, 1)))
        half = max(2, width // 12)
        left, right = max(0, column - half), min(width, column + half)
        image[:, left:right] *= 0.35

    if speckle_std > 0:
        speckle = rng.normal(1.0, speckle_std, image.shape).astype(np.float32)
        image = image * speckle
    image = image + rng.normal(0.0, 0.01, image.shape).astype(np.float32)

    return np.clip(image, 0.0, 1.0).astype(np.float32), interior.astype(np.float32)


def generate_dataset(
    output_dir: str | Path,
    config: Optional[SyntheticSequenceConfig] = None,
    seed: int = 0,
    splits: Optional[Sequence[str]] = None,
    manifest_name: str = "manifest.csv",
) -> tuple[Path, Manifest]:
    """Generate a complete synthetic dataset with images, masks and a manifest.

    Args:
        output_dir: Directory to populate. ``images/`` and ``masks/``
            subdirectories are created.
        config: Dataset dimensions and appearance.
        seed: Determinism seed.
        splits: Optional per-patient split names, one per patient. When
            ``None`` the ``split`` column is left empty for
            :mod:`rus_perception.data.splits` to fill.
        manifest_name: File name of the manifest CSV.

    Returns:
        ``(manifest_path, manifest)``.

    Raises:
        ValueError: If ``splits`` is given but its length differs from the
            patient count.
    """
    from PIL import Image

    config = config or SyntheticSequenceConfig()
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if splits is not None and len(splits) != config.num_patients:
        raise ValueError(
            f"splits has {len(splits)} entries but there are {config.num_patients} patients."
        )

    height, width = config.image_size
    rng = np.random.default_rng(seed)
    records: list[ManifestRecord] = []

    for patient_index in range(config.num_patients):
        patient_id = f"P{patient_index:03d}"
        for sequence_index in range(config.sequences_per_patient):
            sequence_id = f"S{sequence_index:02d}"
            # Per-sequence anatomy and motion, so sequences differ from each other.
            centre_x = float(rng.uniform(0.35, 0.65)) * width
            centre_y = float(rng.uniform(0.40, 0.60)) * height
            radius_x = float(rng.uniform(0.14, 0.24)) * width
            radius_y = radius_x * float(rng.uniform(0.7, 1.2))
            direction = float(rng.uniform(0, 2 * math.pi))
            angle = float(rng.uniform(-30.0, 30.0))

            for frame_index in range(config.frames_per_sequence):
                step = config.motion_px_per_frame * frame_index
                centre = (
                    centre_x + step * math.cos(direction),
                    centre_y + step * math.sin(direction),
                )
                # Gentle pulsatile size change, as the lumen is not rigid.
                breathe = 1.0 + 0.03 * math.sin(frame_index * 0.7)
                image, mask = generate_frame(
                    (height, width),
                    centre,
                    (radius_x * breathe, radius_y * breathe),
                    angle,
                    rng,
                    config.speckle_std,
                    shadow=bool(rng.random() < config.shadow_probability),
                )

                stem = f"{patient_id}_{sequence_id}_{frame_index:04d}"
                image_path = image_dir / f"{stem}.png"
                Image.fromarray((image * 255).astype(np.uint8)).save(image_path)

                labeled = rng.random() < config.labeled_fraction
                mask_path: Optional[Path] = None
                if labeled:
                    mask_path = mask_dir / f"{stem}.png"
                    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

                records.append(
                    ManifestRecord(
                        patient_id=patient_id,
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        image_path=str(image_path.relative_to(output_dir)),
                        mask_path=(
                            str(mask_path.relative_to(output_dir)) if mask_path else None
                        ),
                        timestamp=round(frame_index / config.frame_rate, 6),
                        split=splits[patient_index] if splits is not None else None,
                    )
                )

    manifest_path = write_manifest(output_dir / manifest_name, records)
    manifest = Manifest(records, output_dir)
    logger.info(
        "Generated synthetic dataset at %s: %d frames, %d patients",
        output_dir,
        len(records),
        config.num_patients,
    )
    return manifest_path, manifest
