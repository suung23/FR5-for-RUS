#!/usr/bin/env python3
"""Turn the PFUS pelvic-floor ultrasound dataset into a manifest + binary masks.

The published dataset (Solis-Martin et al., Universidad de Sevilla, CC-BY-4.0)
is laid out as::

    data/readme.txt
    data/P000/frame_000.png     # B-mode midsagittal frame, RGB PNG
    data/P000/frame_000.json    # eight polygons, one per pelvic-floor structure
    ...

Each JSON is a list of eight ``{"label": str, "pol": [[x, y], ...]}`` entries
covering Pubis, Urethra, Bladder, Uterus, Vagina, Anus, Rectum and Levator ani
muscle, in the pixel coordinates of the frame next to it.

This script rasterises the polygons of the requested label(s) into the binary
masks the rest of the pipeline expects, and writes a manifest CSV with
patient-level train/val/test splits. Frames of one patient form one sequence,
ordered by the frame number in the filename.

Only the requested labels become foreground. Every other structure stays
background, so the default (``--labels Bladder``) reproduces exactly the binary
bladder-lumen task the models and metrics in this repository are built for.

Example:
    python scripts/prepare_pfus.py \
        --source ~/datasets/pfus/raw/data \
        --output-dir ~/datasets/pfus \
        --labels Bladder --qc-samples 12
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from _common import REPO_ROOT  # noqa: F401  (path setup)

from rus_perception.data.manifest import Manifest, ManifestRecord, write_manifest
from rus_perception.data.splits import apply_split, split_by_patient, split_report
from rus_perception.utils.logging_utils import setup_logging

logger = logging.getLogger("prepare_pfus")

#: Every structure annotated in the dataset, in the order used by the authors.
PFUS_LABELS: tuple[str, ...] = (
    "Pubis",
    "Urethra",
    "Bladder",
    "Uterus",
    "Vagina",
    "Anus",
    "Rectum",
    "Levator ani muscle",
)

SEQUENCE_ID = "seq0"  # one recorded sweep per patient in this dataset


@dataclass
class FrameResult:
    """Outcome of rasterising a single frame."""

    patient_id: str
    frame_index: int
    image_path: str
    mask_path: str
    foreground_pixels: int
    missing_label: bool


@dataclass
class PatientResult:
    """Outcome of rasterising one patient folder."""

    patient_id: str
    frames: list[FrameResult]
    skipped: list[str]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the PFUS dataset for training and evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Directory holding the PXXX patient folders (the dataset's 'data' directory)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write masks/ and manifest.csv into",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["Bladder"],
        metavar="LABEL",
        help=f"Structures to rasterise as foreground. Available: {list(PFUS_LABELS)}",
    )
    parser.add_argument(
        "--mask-subdir", default="masks", help="Subdirectory of --output-dir for the masks"
    )
    parser.add_argument("--manifest-name", default="manifest.csv", help="Manifest filename")
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Patient-level split ratios written into the manifest",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument(
        "--patients", type=int, default=None, help="Use only the first N patients (smoke tests)"
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="Worker processes"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-rasterise masks that already exist"
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Exclude frames whose mask is empty instead of keeping them as labeled background",
    )
    parser.add_argument(
        "--qc-samples",
        type=int,
        default=0,
        help="Write N image/mask overlay PNGs to <output-dir>/qc for visual inspection",
    )
    parser.add_argument(
        "--report-occupancy",
        action="store_true",
        help="Read every mask again to report the mean foreground ratio per split",
    )
    return parser.parse_args()


def frame_number(path: Path) -> int:
    """Extract ``YYY`` from ``frame_YYY.json`` / ``frame_YYY.png``.

    Raises:
        ValueError: If the stem does not end in an integer.
    """
    stem = path.stem
    digits = stem.rsplit("_", 1)[-1]
    if not digits.isdigit():
        raise ValueError(f"Cannot read a frame number from {path.name!r}.")
    return int(digits)


def rasterise_polygons(
    polygons: Iterable[Sequence[Sequence[float]]], size: tuple[int, int]
) -> Image.Image:
    """Fill ``polygons`` into a new ``L`` mask of ``size`` (width, height).

    Polygons are drawn at full intensity and simply overlap; the result is
    binary by construction, so no thresholding decision is hidden here.
    """
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        points = [(float(x), float(y)) for x, y in polygon]
        if len(points) < 3:
            continue  # a line or a point has no area to fill
        draw.polygon(points, fill=255)
    return mask


def _process_patient(
    patient_dir: str,
    mask_root: str,
    output_dir: str,
    labels: tuple[str, ...],
    overwrite: bool,
) -> PatientResult:
    """Rasterise every annotated frame of one patient folder."""
    patient_path = Path(patient_dir)
    patient_id = patient_path.name
    mask_dir = Path(mask_root) / patient_id
    mask_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(labels)

    frames: list[FrameResult] = []
    skipped: list[str] = []

    for json_path in sorted(patient_path.glob("frame_*.json")):
        image_path = json_path.with_suffix(".png")
        if not image_path.is_file():
            skipped.append(f"{patient_id}/{json_path.name}: no matching PNG")
            continue
        try:
            index = frame_number(json_path)
        except ValueError as exc:
            skipped.append(f"{patient_id}/{json_path.name}: {exc}")
            continue

        try:
            annotations = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            skipped.append(f"{patient_id}/{json_path.name}: unreadable JSON ({exc})")
            continue

        selected = [
            entry["pol"]
            for entry in annotations
            if isinstance(entry, dict) and entry.get("label") in wanted and entry.get("pol")
        ]
        missing_label = len(selected) < len(wanted)

        # Pillow reads the header lazily, so this costs no pixel decoding.
        with Image.open(image_path) as image:
            size = image.size

        mask_path = mask_dir / f"{json_path.stem}.png"
        if overwrite or not mask_path.is_file():
            mask = rasterise_polygons(selected, size)
            mask.save(mask_path, optimize=True)
        else:
            with Image.open(mask_path) as existing:
                mask = existing.copy()

        foreground = int(np.count_nonzero(np.asarray(mask) > 127))

        frames.append(
            FrameResult(
                patient_id=patient_id,
                frame_index=index,
                image_path=os.path.relpath(image_path, output_dir),
                mask_path=os.path.relpath(mask_path, output_dir),
                foreground_pixels=foreground,
                missing_label=missing_label,
            )
        )

    return PatientResult(patient_id=patient_id, frames=frames, skipped=skipped)


def write_qc_overlays(manifest: Manifest, output_dir: Path, count: int) -> None:
    """Blend mask contours over frames so the alignment can be eyeballed.

    Samples evenly across the manifest rather than taking the first N frames,
    which would only ever show one or two patients.
    """
    if count <= 0 or len(manifest) == 0:
        return
    qc_dir = output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(manifest) // count)
    written = 0
    for record in manifest.records[::step][:count]:
        image_path = manifest.resolve(record.image_path)
        mask_path = manifest.resolve(record.mask_path)
        if image_path is None or mask_path is None:
            continue
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").resize(image.size, Image.NEAREST)
        red = Image.new("RGB", image.size, (255, 0, 0))
        overlay = Image.composite(Image.blend(image, red, 0.45), image, mask)
        name = f"{record.patient_id}_{record.frame_index:04d}.png"
        overlay.save(qc_dir / name)
        written += 1
    logger.info("Wrote %d QC overlay(s) to %s", written, qc_dir)


def main() -> int:
    args = get_args()
    setup_logging("INFO")

    source = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"--source is not a directory: {source}")

    unknown = [label for label in args.labels if label not in PFUS_LABELS]
    if unknown:
        raise SystemExit(
            f"Unknown label(s) {unknown}. The dataset annotates: {list(PFUS_LABELS)}"
        )

    patient_dirs = sorted(p for p in source.glob("P*") if p.is_dir())
    if not patient_dirs:
        raise SystemExit(
            f"No PXXX patient folders under {source}. Point --source at the dataset's "
            "'data' directory."
        )
    if args.patients is not None:
        patient_dirs = patient_dirs[: args.patients]

    mask_root = output_dir / args.mask_subdir
    mask_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Rasterising %s from %d patient folder(s) in %s -> %s",
        ", ".join(args.labels),
        len(patient_dirs),
        source,
        mask_root,
    )

    labels = tuple(args.labels)
    results: list[PatientResult] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _process_patient,
                    str(directory),
                    str(mask_root),
                    str(output_dir),
                    labels,
                    args.overwrite,
                )
                for directory in patient_dirs
            ]
            for done, future in enumerate(futures, start=1):
                results.append(future.result())
                if done % 10 == 0 or done == len(futures):
                    logger.info("  %d/%d patients", done, len(futures))
    else:
        for done, directory in enumerate(patient_dirs, start=1):
            results.append(
                _process_patient(
                    str(directory), str(mask_root), str(output_dir), labels, args.overwrite
                )
            )
            if done % 10 == 0 or done == len(patient_dirs):
                logger.info("  %d/%d patients", done, len(patient_dirs))

    frames = [frame for result in results for frame in result.frames]
    skipped = [message for result in results for message in result.skipped]
    if not frames:
        raise SystemExit("No annotated frames were found; nothing to write.")

    empty = [frame for frame in frames if frame.foreground_pixels == 0]
    incomplete = [frame for frame in frames if frame.missing_label]
    if skipped:
        logger.warning("Skipped %d file(s); first few: %s", len(skipped), skipped[:5])
    if incomplete:
        logger.warning(
            "%d frame(s) do not annotate every requested label; their masks contain only "
            "the labels that are present.",
            len(incomplete),
        )
    if empty:
        message = "%d frame(s) rasterise to an empty mask (requested structure absent)."
        if args.drop_empty:
            logger.warning(message + " Dropping them (--drop-empty).", len(empty))
            frames = [frame for frame in frames if frame.foreground_pixels > 0]
        else:
            logger.warning(
                message + " Keeping them as labeled all-background frames; pass --drop-empty "
                "to exclude them.",
                len(empty),
            )

    records = [
        ManifestRecord(
            patient_id=frame.patient_id,
            sequence_id=SEQUENCE_ID,
            frame_index=frame.frame_index,
            image_path=frame.image_path,
            mask_path=frame.mask_path,
        )
        for frame in frames
    ]
    manifest = Manifest(records, output_dir)
    manifest.validate_ordering()

    assignment = split_by_patient(manifest.patients, tuple(args.split_ratios), seed=args.seed)
    manifest = apply_split(manifest, assignment)

    manifest_path = write_manifest(output_dir / args.manifest_name, manifest.records)
    (output_dir / "splits.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "ratios": list(args.split_ratios),
                "labels": list(args.labels),
                "assignment": assignment.as_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Manifest written to %s", manifest_path)

    write_qc_overlays(manifest, output_dir, args.qc_samples)

    print(split_report(manifest, compute_class_occupancy=args.report_occupancy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
