"""The shipped split configuration, pinned.

The 7:3 train/validation split is a project decision, and three config keys have
to agree for it to work. They live in different sections of ``_base.yaml``, so
editing one and forgetting another is easy and the failure surfaces only when a
real dataset arrives -- which is the worst possible moment to discover it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rus_perception.data.splits import split_by_patient  # noqa: E402
from rus_perception.utils.config import load_config  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENT_CONFIGS = sorted(
    p for p in CONFIG_DIR.glob("*.yaml") if p.name not in ("_base.yaml", "realtime_monitor.yaml")
)


@pytest.fixture(scope="module")
def base() -> dict:
    return yaml.safe_load((CONFIG_DIR / "_base.yaml").read_text(encoding="utf-8"))


def test_split_is_seventy_thirty_with_no_test_set(base: dict) -> None:
    train, val, test = base["split"]["ratios"]
    assert (train, val) == (0.70, 0.30)
    assert test == 0.0
    assert train + val + test == pytest.approx(1.0)


def test_strategy_does_not_require_a_split_column(base: dict) -> None:
    """A manifest that has just been produced has no ``split`` column, and the
    ``manifest`` strategy raises on one. Data arriving must not stall on that."""
    assert base["split"]["strategy"] == "patient_random"


def test_evaluation_split_is_not_the_empty_one(base: dict) -> None:
    """The coupling that breaks silently: with ``test`` at 0, ``evaluate.py``
    exits with "Split 'test' is empty" and the whole validation report is lost."""
    _, _, test_ratio = base["split"]["ratios"]
    evaluation_split = base["evaluation"]["split"]
    if test_ratio == 0.0:
        assert evaluation_split != "test", (
            "split.ratios gives the test split 0, so evaluation.split must not be 'test'."
        )
    assert evaluation_split in ("train", "val", "test")


@pytest.mark.parametrize("config_path", EXPERIMENT_CONFIGS, ids=lambda p: p.name)
def test_every_experiment_config_inherits_the_split(config_path: Path) -> None:
    """No experiment config may quietly override the split away from 7:3."""
    config = load_config(config_path, validate=False)
    assert tuple(config.get("split.ratios")) == (0.70, 0.30, 0.0)
    assert config.get("split.strategy") == "patient_random"
    assert config.get("evaluation.split") == "val"


@pytest.mark.parametrize("n_patients", [2, 3, 10, 17, 100])
def test_the_ratio_partitions_patients_seven_to_three(n_patients: int) -> None:
    """The ratio applies to patients, not frames, and every patient lands in
    exactly one side."""
    patients = [f"P{i:03d}" for i in range(n_patients)]
    assignment = split_by_patient(patients, (0.70, 0.30, 0.0), seed=42)

    assert assignment.test == []
    assert len(assignment.train) + len(assignment.val) == n_patients
    assert set(assignment.train).isdisjoint(assignment.val)
    assert assignment.train and assignment.val
    # Rounding may move one patient either way; the proportion must not drift more.
    assert abs(len(assignment.train) / n_patients - 0.70) <= 1.0 / n_patients + 1e-9


def test_a_single_patient_cannot_be_split(caplog) -> None:
    """One patient in both halves would be leakage, so this must fail loudly
    rather than produce an empty validation set."""
    with pytest.raises(ValueError):
        split_by_patient(["P000"], (0.70, 0.30, 0.0), seed=42)


def test_the_assignment_is_stable_when_patients_are_added() -> None:
    """A seeded content hash, not a shuffle: adding subjects must not reshuffle
    the existing ones, or every historical result becomes unreproducible."""
    first = split_by_patient([f"P{i:03d}" for i in range(20)], (0.70, 0.30, 0.0), seed=42)
    grown = split_by_patient([f"P{i:03d}" for i in range(30)], (0.70, 0.30, 0.0), seed=42)

    moved = [
        patient
        for patient in first.train
        if patient in grown.val
    ] + [
        patient
        for patient in first.val
        if patient in grown.train
    ]
    # A few boundary patients shift as the counts change; a wholesale reshuffle
    # would move roughly half of them.
    assert len(moved) <= 6, f"{len(moved)}/20 patients changed side when the cohort grew"


def test_the_split_is_reproducible_from_the_seed() -> None:
    patients = [f"P{i:03d}" for i in range(25)]
    a = split_by_patient(patients, (0.70, 0.30, 0.0), seed=42)
    b = split_by_patient(patients, (0.70, 0.30, 0.0), seed=42)
    c = split_by_patient(patients, (0.70, 0.30, 0.0), seed=7)
    assert a.as_dict() == b.as_dict()
    assert a.as_dict() != c.as_dict()
