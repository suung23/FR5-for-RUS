"""Shared pytest fixtures.

Every fixture is built from synthetic data. No test in this suite requires
private clinical data or a network connection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rus_perception.data.manifest import Manifest, load_manifest  # noqa: E402
from rus_perception.data.splits import apply_split, split_by_patient  # noqa: E402
from rus_perception.data.synthetic import SyntheticSequenceConfig, generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Manifest]:
    """A small labelled synthetic dataset with patient-level splits."""
    directory = tmp_path_factory.mktemp("synthetic")
    config = SyntheticSequenceConfig(
        num_patients=4,
        sequences_per_patient=1,
        frames_per_sequence=6,
        image_size=(64, 64),
        labeled_fraction=1.0,
        motion_px_per_frame=1.5,
    )
    manifest_path, manifest = generate_dataset(directory, config, seed=0)
    assignment = split_by_patient(manifest.patients, (0.5, 0.25, 0.25), seed=0)
    manifest = apply_split(manifest, assignment)
    from rus_perception.data.manifest import write_manifest

    write_manifest(manifest_path, manifest.records)
    return manifest_path, manifest


@pytest.fixture(scope="session")
def partially_labeled_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Manifest]:
    """A synthetic dataset in which only some frames carry annotations."""
    directory = tmp_path_factory.mktemp("synthetic_partial")
    config = SyntheticSequenceConfig(
        num_patients=3,
        sequences_per_patient=1,
        frames_per_sequence=6,
        image_size=(64, 64),
        labeled_fraction=0.5,
    )
    manifest_path, _ = generate_dataset(directory, config, seed=7)
    return manifest_path, load_manifest(manifest_path)


@pytest.fixture
def disc_probability() -> np.ndarray:
    """A smooth 64x64 probability blob resembling a segmented lumen."""
    ys, xs = np.mgrid[0:64, 0:64]
    return np.exp(-(((xs - 32) / 11.0) ** 2 + ((ys - 30) / 9.0) ** 2) * 2).astype(np.float32)


@pytest.fixture
def disc_image(disc_probability: np.ndarray) -> np.ndarray:
    """A synthetic frame whose lumen is dark against brighter tissue."""
    image = np.full((64, 64), 0.6, dtype=np.float32)
    image[disc_probability > 0.5] = 0.05
    return image


@pytest.fixture
def slim_model():
    """A paper-preset :class:`SlimUNet` in eval mode."""
    from rus_perception.models.slim_unet import SlimUNet

    return SlimUNet.from_preset("paper").eval()
