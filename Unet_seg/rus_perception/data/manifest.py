"""Manifest-based description of a sequential ultrasound dataset.

A manifest is a CSV file with one row per video frame. Recognised columns:

===================== ======== ==================================================
column                required description
===================== ======== ==================================================
``patient_id``        yes      Subject identifier; the unit of data splitting.
``sequence_id``       yes      Video/sweep identifier, unique within a patient.
``frame_index``       yes      Integer position within the sequence (chronological).
``image_path``        yes      Path to the B-mode frame.
``mask_path``         no       Path to the binary lumen annotation. Empty for
                               unlabeled frames.
``timestamp``         no       Acquisition time in seconds.
``split``             no       ``train`` / ``val`` / ``test``. Assigned by
                               :mod:`rus_perception.data.splits` when absent.
``flow_forward_path`` no       Precomputed previous->current flow.
``flow_backward_path``no       Precomputed current->previous flow.
===================== ======== ==================================================

Only the columns that genuinely exist for a dataset are required; unlabeled
frames and missing flow are first-class cases.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "ManifestRecord",
    "Manifest",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "load_manifest",
    "write_manifest",
]

REQUIRED_COLUMNS: tuple[str, ...] = ("patient_id", "sequence_id", "frame_index", "image_path")
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "mask_path",
    "timestamp",
    "split",
    "flow_forward_path",
    "flow_backward_path",
)
ALL_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


@dataclass(frozen=True)
class ManifestRecord:
    """A single frame of a sequential ultrasound dataset."""

    patient_id: str
    sequence_id: str
    frame_index: int
    image_path: str
    mask_path: Optional[str] = None
    timestamp: Optional[float] = None
    split: Optional[str] = None
    flow_forward_path: Optional[str] = None
    flow_backward_path: Optional[str] = None

    @property
    def is_labeled(self) -> bool:
        """Whether this frame carries a ground-truth annotation."""
        return bool(self.mask_path)

    @property
    def key(self) -> tuple[str, str, int]:
        """Unique ``(patient, sequence, frame_index)`` identifier."""
        return (self.patient_id, self.sequence_id, self.frame_index)

    @property
    def frame_id(self) -> str:
        """Human-readable frame identifier used in logs and control output."""
        return f"{self.patient_id}/{self.sequence_id}/{self.frame_index:06d}"

    def to_row(self) -> dict[str, Any]:
        """Return a CSV-writable dictionary."""
        row = asdict(self)
        return {key: ("" if value is None else value) for key, value in row.items()}


@dataclass
class DatasetStatistics:
    """Counts describing a manifest, used for reports and leakage auditing."""

    num_patients: int
    num_sequences: int
    num_frames: int
    num_labeled: int
    num_unlabeled: int
    frames_per_split: dict[str, int] = field(default_factory=dict)
    patients_per_split: dict[str, int] = field(default_factory=dict)
    sequences_per_split: dict[str, int] = field(default_factory=dict)
    labeled_per_split: dict[str, int] = field(default_factory=dict)
    frames_per_patient: dict[str, int] = field(default_factory=dict)
    class_occupancy: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    def to_text(self) -> str:
        """Render a short human-readable report."""
        lines = [
            f"patients            : {self.num_patients}",
            f"sequences           : {self.num_sequences}",
            f"frames              : {self.num_frames}",
            f"labeled frames      : {self.num_labeled}",
            f"unlabeled frames    : {self.num_unlabeled}",
        ]
        if self.frames_per_split:
            lines.append("split distribution  :")
            for split in sorted(self.frames_per_split):
                lines.append(
                    f"  {split:<6} frames={self.frames_per_split[split]:<7d} "
                    f"patients={self.patients_per_split.get(split, 0):<4d} "
                    f"sequences={self.sequences_per_split.get(split, 0):<4d} "
                    f"labeled={self.labeled_per_split.get(split, 0)}"
                )
        if self.class_occupancy:
            lines.append("class occupancy     :")
            for key in sorted(self.class_occupancy):
                lines.append(f"  {key:<28}: {self.class_occupancy[key]:.6f}")
        return "\n".join(lines)


class Manifest:
    """An ordered, validated collection of :class:`ManifestRecord` rows.

    Records are stored sorted by ``(patient_id, sequence_id, frame_index)``, so
    chronological order within a sequence is guaranteed by construction.

    Args:
        records: Frame records.
        root: Directory that relative paths are resolved against.
    """

    def __init__(self, records: Sequence[ManifestRecord], root: str | Path = ".") -> None:
        self.root = Path(root)
        self.records: list[ManifestRecord] = sorted(
            records, key=lambda r: (r.patient_id, r.sequence_id, r.frame_index)
        )
        self._validate()

    def _validate(self) -> None:
        """Reject duplicate frames and malformed ordering.

        Raises:
            ValueError: If a ``(patient, sequence, frame_index)`` key repeats or
                a frame index is negative.
        """
        seen: set[tuple[str, str, int]] = set()
        for record in self.records:
            if record.frame_index < 0:
                raise ValueError(
                    f"Negative frame_index {record.frame_index} for {record.frame_id}."
                )
            if record.key in seen:
                raise ValueError(
                    f"Duplicate manifest entry for {record.frame_id}. Each "
                    "(patient_id, sequence_id, frame_index) must be unique."
                )
            seen.add(record.key)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ManifestRecord]:
        return iter(self.records)

    def __getitem__(self, index: int) -> ManifestRecord:
        return self.records[index]

    # -- accessors ---------------------------------------------------------
    @property
    def patients(self) -> list[str]:
        """Sorted unique patient identifiers."""
        return sorted({r.patient_id for r in self.records})

    @property
    def splits(self) -> list[str]:
        """Sorted unique split names actually present."""
        return sorted({r.split for r in self.records if r.split})

    def sequences(self) -> dict[tuple[str, str], list[ManifestRecord]]:
        """Group records by ``(patient_id, sequence_id)``, chronologically ordered."""
        grouped: dict[tuple[str, str], list[ManifestRecord]] = defaultdict(list)
        for record in self.records:
            grouped[(record.patient_id, record.sequence_id)].append(record)
        return {key: sorted(value, key=lambda r: r.frame_index) for key, value in grouped.items()}

    def filter(
        self,
        split: Optional[str] = None,
        labeled_only: bool = False,
        patients: Optional[Iterable[str]] = None,
    ) -> "Manifest":
        """Return a new manifest restricted to the given criteria."""
        allowed = set(patients) if patients is not None else None
        records = [
            record
            for record in self.records
            if (split is None or record.split == split)
            and (not labeled_only or record.is_labeled)
            and (allowed is None or record.patient_id in allowed)
        ]
        return Manifest(records, self.root)

    def resolve(self, path: Optional[str]) -> Optional[Path]:
        """Resolve a manifest path against ``root``; ``None`` stays ``None``."""
        if not path:
            return None
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    # -- validation and reporting -----------------------------------------
    def validate_files(self, check_masks: bool = True, check_flow: bool = False) -> None:
        """Verify that referenced files exist.

        Args:
            check_masks: Also verify mask files of labeled frames.
            check_flow: Also verify precomputed flow files.

        Raises:
            FileNotFoundError: Listing up to 20 missing files.
        """
        missing: list[str] = []
        for record in self.records:
            image = self.resolve(record.image_path)
            if image is None or not image.is_file():
                missing.append(f"{record.frame_id}: image {record.image_path}")
            if check_masks and record.is_labeled:
                mask = self.resolve(record.mask_path)
                if mask is None or not mask.is_file():
                    missing.append(f"{record.frame_id}: mask {record.mask_path}")
            if check_flow:
                for label, value in (
                    ("flow_forward", record.flow_forward_path),
                    ("flow_backward", record.flow_backward_path),
                ):
                    if value:
                        resolved = self.resolve(value)
                        if resolved is None or not resolved.is_file():
                            missing.append(f"{record.frame_id}: {label} {value}")
        if missing:
            preview = "\n  ".join(missing[:20])
            more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
            raise FileNotFoundError(
                f"{len(missing)} file(s) referenced by the manifest are missing:\n  "
                f"{preview}{more}"
            )

    def validate_ordering(self) -> None:
        """Check that every sequence has strictly increasing frame indices.

        Raises:
            ValueError: If a sequence is empty or its indices are not increasing.
        """
        for (patient, sequence), records in self.sequences().items():
            if not records:
                raise ValueError(f"Sequence {patient}/{sequence} contains no frames.")
            indices = [r.frame_index for r in records]
            if any(b <= a for a, b in zip(indices, indices[1:])):
                raise ValueError(
                    f"Sequence {patient}/{sequence} has non-increasing frame indices: "
                    f"{indices[:10]}{'...' if len(indices) > 10 else ''}"
                )
            timestamps = [r.timestamp for r in records if r.timestamp is not None]
            if len(timestamps) == len(records) and any(
                b < a for a, b in zip(timestamps, timestamps[1:])
            ):
                raise ValueError(
                    f"Sequence {patient}/{sequence} has timestamps that decrease while "
                    "frame_index increases; the manifest ordering is inconsistent."
                )

    def statistics(self, compute_class_occupancy: bool = False) -> DatasetStatistics:
        """Summarise the manifest.

        Args:
            compute_class_occupancy: Read every labeled mask to measure the mean
                foreground ratio. This costs one pass over the annotations.
        """
        frames_per_split: Counter[str] = Counter()
        labeled_per_split: Counter[str] = Counter()
        patients_per_split: dict[str, set[str]] = defaultdict(set)
        sequences_per_split: dict[str, set[tuple[str, str]]] = defaultdict(set)
        frames_per_patient: Counter[str] = Counter()

        for record in self.records:
            split = record.split or "unassigned"
            frames_per_split[split] += 1
            frames_per_patient[record.patient_id] += 1
            patients_per_split[split].add(record.patient_id)
            sequences_per_split[split].add((record.patient_id, record.sequence_id))
            if record.is_labeled:
                labeled_per_split[split] += 1

        occupancy: dict[str, float] = {}
        if compute_class_occupancy:
            occupancy = self._class_occupancy()

        labeled = sum(1 for r in self.records if r.is_labeled)
        return DatasetStatistics(
            num_patients=len(self.patients),
            num_sequences=len(self.sequences()),
            num_frames=len(self.records),
            num_labeled=labeled,
            num_unlabeled=len(self.records) - labeled,
            frames_per_split=dict(frames_per_split),
            patients_per_split={k: len(v) for k, v in patients_per_split.items()},
            sequences_per_split={k: len(v) for k, v in sequences_per_split.items()},
            labeled_per_split=dict(labeled_per_split),
            frames_per_patient=dict(frames_per_patient),
            class_occupancy=occupancy,
        )

    def _class_occupancy(self) -> dict[str, float]:
        """Mean foreground ratio and empty-mask rate over labeled frames."""
        from .io import load_mask  # local import: keeps manifest import cheap

        ratios: list[float] = []
        empty = 0
        for record in self.records:
            if not record.is_labeled:
                continue
            path = self.resolve(record.mask_path)
            if path is None or not path.is_file():
                continue
            mask = load_mask(path)
            ratio = float(mask.mean())
            ratios.append(ratio)
            if ratio == 0.0:
                empty += 1
        if not ratios:
            return {}
        return {
            "foreground_ratio_mean": sum(ratios) / len(ratios),
            "foreground_ratio_min": min(ratios),
            "foreground_ratio_max": max(ratios),
            "empty_mask_rate": empty / len(ratios),
            "labeled_frames_scanned": float(len(ratios)),
        }


def _parse_optional_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse timestamp {value!r} as a float.") from exc


def load_manifest(path: str | Path, root: Optional[str | Path] = None) -> Manifest:
    """Read a manifest CSV.

    Args:
        path: CSV file to read.
        root: Directory that relative paths are resolved against. Defaults to
            the manifest's own parent directory.

    Returns:
        A validated :class:`Manifest`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required columns are missing or a row is malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    root = Path(root) if root is not None else path.parent

    records: list[ManifestRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest {path} is empty (no header row).")
        columns = {name.strip() for name in reader.fieldnames}
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise ValueError(
                f"Manifest {path} is missing required column(s): {missing}. "
                f"Required: {list(REQUIRED_COLUMNS)}; found: {sorted(columns)}"
            )
        unknown = columns - set(ALL_COLUMNS)
        if unknown:
            logger.info("Manifest %s has extra column(s) that will be ignored: %s", path, sorted(unknown))

        for line_number, row in enumerate(reader, start=2):
            try:
                frame_index = int(str(row["frame_index"]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: frame_index must be an integer, got "
                    f"{row.get('frame_index')!r}."
                ) from exc
            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                raise ValueError(f"{path}:{line_number}: image_path must not be empty.")
            patient_id = (row.get("patient_id") or "").strip()
            sequence_id = (row.get("sequence_id") or "").strip()
            if not patient_id or not sequence_id:
                raise ValueError(
                    f"{path}:{line_number}: patient_id and sequence_id must not be empty."
                )
            split = (row.get("split") or "").strip() or None
            records.append(
                ManifestRecord(
                    patient_id=patient_id,
                    sequence_id=sequence_id,
                    frame_index=frame_index,
                    image_path=image_path,
                    mask_path=(row.get("mask_path") or "").strip() or None,
                    timestamp=_parse_optional_float(row.get("timestamp", "")),
                    split=split,
                    flow_forward_path=(row.get("flow_forward_path") or "").strip() or None,
                    flow_backward_path=(row.get("flow_backward_path") or "").strip() or None,
                )
            )

    if not records:
        raise ValueError(f"Manifest {path} contains no data rows.")
    return Manifest(records, root)


def write_manifest(path: str | Path, records: Sequence[ManifestRecord]) -> Path:
    """Write records to a manifest CSV, creating parent directories.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ALL_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return path
