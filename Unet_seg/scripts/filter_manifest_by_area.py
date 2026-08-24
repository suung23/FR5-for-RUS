#!/usr/bin/env python3
"""Drop patients from a manifest by bladder volume and/or an explicit list.

Two independent filters, either of which may be disabled:

``--min-patient-area-ratio``
    Volume filter, described below. Pass ``0`` to disable it.
``--exclude-patients``
    Named patients to remove regardless of volume, for labels that a documented
    review found unusable. Automated label-quality statistics do not work on
    this dataset (see scripts/qc_label_overlay.py), so exclusions are decided by
    eye and recorded here as an explicit list rather than inferred.

Volume filter motivation
------------------------
The FR5 probing loop is aimed at HoLEP, where continuous irrigation holds the
bladder hydro-extended for the whole procedure. A collapsed or near-empty
bladder is therefore *outside* the deployment domain, and training on it spends
capacity on frames the controller will never see.

Why the filter is applied per patient and not per frame
-------------------------------------------------------
On PFUS the bladder area is effectively a *patient* property, not a frame
property: the median within-patient IQR is ~161 px (at 256 x 256) while patient
medians span 553 - 9,954 px. A static exam simply does not change bladder volume
across its own sequence. Filtering per frame would therefore cut a handful of
borderline patients' sequences in half -- breaking temporal pairs and skewing the
per-split statistics -- without removing meaningfully different frames. Dropping
the whole patient is the honest operation, so that is what this script does.

The ``split`` column is preserved verbatim: patients are removed from whichever
split they were already in, and no patient is ever moved. Re-splitting here
would silently invalidate every previously reported number.

Example:
    python scripts/filter_manifest_by_area.py \
        --manifest ~/datasets/pfus/manifest.csv \
        --output ~/datasets/pfus/manifest_qc.csv \
        --exclude-patients P021 \
        --report ~/datasets/pfus/manifest_qc_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

from rus_perception.data.io import load_mask
from rus_perception.data.manifest import load_manifest, write_manifest
from rus_perception.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# The area threshold is a *ratio* so it is independent of the native frame size,
# but the analysis and the evaluation report both speak in pixels at the
# 256 x 256 working resolution. This is the conversion between the two.
PX_AT_256 = 256 * 256


def _area_ratio(args: tuple[str, Optional[str]]) -> tuple[str, float]:
    """Return ``(frame_key, foreground_ratio)`` for one manifest row.

    An unlabeled frame yields ``nan`` so it can be excluded from the per-patient
    statistic rather than counted as an empty bladder.
    """
    key, path = args
    if not path:
        return key, float("nan")
    mask = load_mask(path)
    return key, float(mask.mean())


def compute_patient_areas(manifest, workers: int) -> tuple[dict[str, float], dict[str, float]]:
    """Compute the per-frame and per-patient foreground area ratios.

    Args:
        manifest: Loaded :class:`Manifest`.
        workers: Process-pool size for mask decoding.

    Returns:
        ``(frame_ratios, patient_medians)`` keyed by ``frame_id`` and
        ``patient_id`` respectively.

    Raises:
        ValueError: If no patient ends up with a usable statistic.
    """
    jobs = [
        (record.frame_id, str(manifest.resolve(record.mask_path)) if record.mask_path else None)
        for record in manifest
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame_ratios = dict(pool.map(_area_ratio, jobs, chunksize=200))

    by_patient: dict[str, list[float]] = defaultdict(list)
    for record in manifest:
        ratio = frame_ratios[record.frame_id]
        if not np.isnan(ratio):
            by_patient[record.patient_id].append(ratio)

    patient_medians = {
        patient: float(np.median(values)) for patient, values in by_patient.items() if values
    }
    if not patient_medians:
        raise ValueError(
            "No patient has a single labeled frame; the area filter has nothing to "
            "measure. Check that mask_path is populated."
        )
    unlabeled = set(manifest.patients) - set(patient_medians)
    if unlabeled:
        logger.warning(
            "%d patient(s) have no labeled frame and will be dropped: %s",
            len(unlabeled), sorted(unlabeled),
        )
    return frame_ratios, patient_medians


def summarise(manifest, keep: set[str]) -> dict[str, dict[str, int]]:
    """Count patients and frames per split, before and after the filter."""
    stats: dict[str, dict[str, int]] = {}
    for split in sorted({record.split or "unassigned" for record in manifest}):
        rows = [r for r in manifest if (r.split or "unassigned") == split]
        kept = [r for r in rows if r.patient_id in keep]
        stats[split] = {
            "patients_before": len({r.patient_id for r in rows}),
            "patients_after": len({r.patient_id for r in kept}),
            "frames_before": len(rows),
            "frames_after": len(kept),
        }
    return stats


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Source manifest CSV")
    parser.add_argument("--output", required=True, help="Filtered manifest CSV to write")
    parser.add_argument(
        "--min-patient-area-ratio",
        type=float,
        default=0.0,
        help="Keep a patient when its median labeled foreground ratio reaches "
             "this value. 0 disables the volume filter. 0.022888 is 1,500 px at "
             "the 256x256 working resolution.",
    )
    parser.add_argument(
        "--exclude-patients",
        default=None,
        help="Comma-separated patient ids to drop regardless of volume, e.g. "
             "'P021'. Record WHY in the run's notes: an exclusion that is not "
             "justified independently of the model's own score is circular.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON file recording every kept/dropped patient and its median area",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Processes used to decode masks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be dropped without writing the manifest",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()
    setup_logging("INFO")

    if not 0.0 <= args.min_patient_area_ratio < 1.0:
        raise ValueError(
            f"--min-patient-area-ratio must lie in [0, 1), got {args.min_patient_area_ratio}."
        )

    manifest = load_manifest(args.manifest)
    logger.info(
        "Loaded %d frames from %d patients: %s",
        len(manifest), len(manifest.patients), args.manifest,
    )

    excluded = {p.strip() for p in (args.exclude_patients or "").split(",") if p.strip()}
    unknown = excluded - set(manifest.patients)
    if unknown:
        raise ValueError(
            f"--exclude-patients names patient(s) absent from the manifest: "
            f"{sorted(unknown)}. Check for a typo rather than silently dropping nothing."
        )

    _, patient_medians = compute_patient_areas(manifest, args.workers)
    threshold = float(args.min_patient_area_ratio)
    keep = {
        p for p, median in patient_medians.items()
        if median >= threshold and p not in excluded
    }
    dropped = sorted(set(manifest.patients) - keep)

    if threshold > 0:
        logger.info(
            "Volume threshold %.6f (%.0f px at 256x256)", threshold, threshold * PX_AT_256
        )
    else:
        logger.info("Volume filter disabled (--min-patient-area-ratio 0)")
    if excluded:
        logger.info("Excluded by name: %s", sorted(excluded))
    logger.info(
        "Keeping %d/%d patients, dropping %d",
        len(keep), len(manifest.patients), len(dropped),
    )
    stats = summarise(manifest, keep)
    for split, row in stats.items():
        logger.info(
            "  %-11s patients %3d -> %3d   frames %6d -> %6d (%.1f%% kept)",
            split,
            row["patients_before"], row["patients_after"],
            row["frames_before"], row["frames_after"],
            100.0 * row["frames_after"] / max(row["frames_before"], 1),
        )

    empty = [s for s, row in stats.items() if row["patients_after"] == 0]
    if empty:
        raise ValueError(
            f"The filter empties the split(s) {empty} entirely. Lower "
            "--min-patient-area-ratio: a split with no patient cannot be trained on "
            "or evaluated."
        )

    records = [r for r in manifest if r.patient_id in keep]

    if args.report:
        report = {
            "source_manifest": str(Path(args.manifest).resolve()),
            "min_patient_area_ratio": threshold,
            "min_patient_area_px_at_256": threshold * PX_AT_256,
            "excluded_patients": sorted(excluded),
            "splits": stats,
            "kept_patients": {
                p: {"median_area_ratio": patient_medians[p],
                    "median_area_px_at_256": patient_medians[p] * PX_AT_256}
                for p in sorted(keep)
            },
            "dropped_patients": {
                p: {"median_area_ratio": patient_medians.get(p),
                    "median_area_px_at_256": (patient_medians[p] * PX_AT_256)
                                             if p in patient_medians else None}
                for p in dropped
            },
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Filter report written: %s", report_path)

    if args.dry_run:
        logger.info("--dry-run: no manifest written.")
        return 0

    write_manifest(args.output, records)
    logger.info("Filtered manifest written: %s (%d frames)", args.output, len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
