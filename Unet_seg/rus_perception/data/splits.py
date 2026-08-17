"""Patient-level dataset splitting and leakage auditing.

Frames from one video are highly correlated, so splitting frames at random
leaks a patient's anatomy from train into validation/test and inflates every
metric. Splitting is therefore always performed at the ``patient_id`` level, and
:func:`assert_no_patient_leakage` enforces the invariant wherever splits are
consumed.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from .manifest import Manifest, ManifestRecord

logger = logging.getLogger(__name__)

__all__ = [
    "PatientLeakageError",
    "SplitAssignment",
    "assert_no_patient_leakage",
    "split_by_patient",
    "apply_split",
    "split_report",
]

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


class PatientLeakageError(RuntimeError):
    """Raised when the same ``patient_id`` appears in more than one split."""


@dataclass
class SplitAssignment:
    """Mapping from split name to the patients it contains."""

    train: list[str]
    val: list[str]
    test: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        """Return ``{split_name: patients}``."""
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test)}

    def split_of(self, patient_id: str) -> Optional[str]:
        """Return the split a patient belongs to, or ``None``."""
        for name, patients in self.as_dict().items():
            if patient_id in patients:
                return name
        return None


def assert_no_patient_leakage(records: Iterable[ManifestRecord]) -> None:
    """Verify that no patient appears in more than one split.

    Records without a split are ignored, since they are not yet assigned.

    Raises:
        PatientLeakageError: Listing the offending patients and their splits.
    """
    seen: dict[str, set[str]] = {}
    for record in records:
        if not record.split:
            continue
        seen.setdefault(record.patient_id, set()).add(record.split)

    leaked = {patient: sorted(splits) for patient, splits in seen.items() if len(splits) > 1}
    if leaked:
        details = "; ".join(f"{patient} -> {splits}" for patient, splits in sorted(leaked.items()))
        raise PatientLeakageError(
            f"{len(leaked)} patient(s) appear in multiple splits: {details}. "
            "Splits must be disjoint at the patient level."
        )


def _stable_patient_order(patients: Sequence[str], seed: int) -> list[str]:
    """Deterministically shuffle patients using a seeded content hash.

    Using a hash of ``(seed, patient_id)`` rather than a PRNG over a list means
    the ordering is stable when unrelated patients are added or removed, which
    keeps historical splits reproducible as a dataset grows.
    """

    def rank(patient: str) -> str:
        digest = hashlib.sha256(f"{seed}:{patient}".encode("utf-8")).hexdigest()
        return digest

    return sorted(patients, key=rank)


def split_by_patient(
    patients: Sequence[str],
    ratios: Mapping[str, float] | Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> SplitAssignment:
    """Partition patients into train/val/test.

    Args:
        patients: Unique patient identifiers.
        ratios: Either ``{"train": .., "val": .., "test": ..}`` or a
            three-element sequence. Values are normalised to sum to 1.
        seed: Determinism seed.

    Returns:
        A :class:`SplitAssignment`.

    Raises:
        ValueError: If ``patients`` is empty, contains duplicates, or the ratios
            are invalid. Also raised when there are too few patients to fill
            every requested non-zero split, since a silently empty validation
            set would make model selection meaningless.
    """
    unique = list(dict.fromkeys(patients))
    if len(unique) != len(patients):
        raise ValueError("split_by_patient received duplicate patient identifiers.")
    if not unique:
        raise ValueError("Cannot split an empty patient list.")

    if isinstance(ratios, Mapping):
        values = [float(ratios.get(name, 0.0)) for name in SPLIT_NAMES]
    else:
        values = [float(v) for v in ratios]
        if len(values) != 3:
            raise ValueError(f"ratios must have three entries (train, val, test), got {values}.")
    if any(v < 0 for v in values):
        raise ValueError(f"Split ratios must be non-negative, got {values}.")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"Split ratios must sum to a positive value, got {values}.")
    values = [v / total for v in values]

    requested = [name for name, value in zip(SPLIT_NAMES, values) if value > 0]
    if len(unique) < len(requested):
        raise ValueError(
            f"{len(unique)} patient(s) cannot fill {len(requested)} non-empty splits "
            f"({requested}). Add more patients or set unused ratios to 0."
        )

    ordered = _stable_patient_order(unique, seed)
    counts = [int(round(value * len(ordered))) for value in values]

    # Repair rounding so the counts sum exactly and every requested split is
    # non-empty; the largest split absorbs the adjustment.
    while sum(counts) > len(ordered):
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < len(ordered):
        counts[counts.index(max(counts))] += 1
    for index, name in enumerate(SPLIT_NAMES):
        if values[index] > 0 and counts[index] == 0:
            donor = counts.index(max(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] += 1
            else:
                raise ValueError(
                    f"Not enough patients ({len(unique)}) to give split {name!r} at least "
                    "one patient. Adjust the ratios or add data."
                )

    boundaries = [0]
    for count in counts:
        boundaries.append(boundaries[-1] + count)
    assignment = SplitAssignment(
        train=ordered[boundaries[0] : boundaries[1]],
        val=ordered[boundaries[1] : boundaries[2]],
        test=ordered[boundaries[2] : boundaries[3]],
    )
    logger.info(
        "Patient-level split (seed=%d): train=%d val=%d test=%d patients",
        seed,
        len(assignment.train),
        len(assignment.val),
        len(assignment.test),
    )
    return assignment


def apply_split(manifest: Manifest, assignment: SplitAssignment) -> Manifest:
    """Return a copy of ``manifest`` with ``split`` set from ``assignment``.

    Raises:
        ValueError: If a patient present in the manifest is not covered by the
            assignment, which would silently drop data.
        PatientLeakageError: If the resulting assignment is not disjoint.
    """
    lookup: dict[str, str] = {}
    for name, patients in assignment.as_dict().items():
        for patient in patients:
            if patient in lookup:
                raise PatientLeakageError(
                    f"Patient {patient!r} is listed in both {lookup[patient]!r} and {name!r}."
                )
            lookup[patient] = name

    uncovered = sorted({r.patient_id for r in manifest} - set(lookup))
    if uncovered:
        raise ValueError(
            f"{len(uncovered)} patient(s) in the manifest have no split assignment: "
            f"{uncovered[:10]}{'...' if len(uncovered) > 10 else ''}"
        )

    records = [
        ManifestRecord(
            patient_id=r.patient_id,
            sequence_id=r.sequence_id,
            frame_index=r.frame_index,
            image_path=r.image_path,
            mask_path=r.mask_path,
            timestamp=r.timestamp,
            split=lookup[r.patient_id],
            flow_forward_path=r.flow_forward_path,
            flow_backward_path=r.flow_backward_path,
        )
        for r in manifest
    ]
    updated = Manifest(records, manifest.root)
    assert_no_patient_leakage(updated)
    return updated


def split_report(manifest: Manifest, compute_class_occupancy: bool = False) -> str:
    """Render a text report of split composition, after auditing for leakage."""
    assert_no_patient_leakage(manifest)
    return manifest.statistics(compute_class_occupancy).to_text()
