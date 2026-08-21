#!/usr/bin/env python3
"""Phase 4 -- PFUS1 label vs manual bladder-lumen label agreement.

Scores the dataset's own `Bladder` polygon against the expert re-annotation on
the audit subset. This is NOT a model evaluation: it measures how much of the
reported model error is really a disagreement about what the target is.

Runs only on samples that actually have a manual mask, and says so loudly when
the annotation is incomplete rather than quietly averaging over a subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_metrics import compare_masks, format_summary, summarise  # noqa: E402

METRICS = ("dice", "iou", "hd95", "assd", "relative_area_error")


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--audit-dir", default=str(root / "annotation_audit"))
    parser.add_argument("--output", default=str(root / "annotation_agreement.csv"))
    parser.add_argument("--summary", default=str(root / "metrics" / "annotation_agreement_summary.json"))
    parser.add_argument("--spacing", type=float, default=1.0, help="mm per pixel; 1.0 reports pixels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audit = Path(args.audit_dir)
    manifest = list(csv.DictReader((audit / "manifest.csv").open(encoding="utf-8")))

    record_path = audit / "annotation_record.csv"
    quality: dict[str, dict[str, str]] = {}
    if record_path.exists():
        key = {r["blind_id"]: r["sample_id"] for r in csv.DictReader((audit / "blind_key.csv").open(encoding="utf-8"))}
        for row in csv.DictReader(record_path.open(encoding="utf-8")):
            sample = key.get(row["blind_id"])
            if sample:
                quality[sample] = row

    rows, missing = [], []
    for record in manifest:
        manual_path = audit / record["manual_mask_path"]
        if not manual_path.exists():
            missing.append(record["sample_id"])
            continue
        pfus = load_mask(audit / record["original_mask_path"])
        manual = load_mask(manual_path)
        comparison = compare_masks(pfus, manual, spacing=args.spacing)
        note = quality.get(record["sample_id"], {})
        rows.append(
            {
                "sample_id": record["sample_id"],
                "patient_id": record["patient_id"],
                "frame_id": record["frame_id"],
                "dice": comparison.dice,
                "iou": comparison.iou,
                "hd95": comparison.hd95,
                "assd": comparison.assd,
                "pfus_area": comparison.area_a,
                "manual_area": comparison.area_b,
                "relative_area_error": comparison.relative_area_error,
                "status": comparison.status,
                "annotation_quality": note.get("annotation_quality", ""),
                "uncertain_boundary": note.get("uncertain_boundary", ""),
                "bladder_size": record["bladder_size"],
                "contrast_level": record["contrast_level"],
                "boundary_visibility": record["boundary_visibility"],
                "shadow_present": record["shadow_present"],
            }
        )

    if not rows:
        print("No manual masks found -- Phase 4 is BLOCKED on human annotation.")
        print(f"  expected: {audit / 'manual_masks'}/<sample_id>.png  ({len(manifest)} samples)")
        print("  run metrics/rasterise_manual.py after the LabelMe JSONs are filled in.")
        Path(args.summary).write_text(
            json.dumps({"status": "blocked", "reason": "no manual annotation present",
                        "expected_samples": len(manifest)}, indent=2), encoding="utf-8"
        )
        return 0

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    patients = [r["patient_id"] for r in rows]
    summary = {
        "status": "complete" if not missing else "partial",
        "n_samples": len(rows),
        "n_missing": len(missing),
        "n_patients": len(set(patients)),
        "spacing": args.spacing,
        "distance_units": "pixels" if args.spacing == 1.0 else "mm",
        "metrics": {m: summarise([r[m] for r in rows], patients, seed=args.seed) for m in METRICS},
        "status_counts": {s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}},
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"PFUS1 label vs manual lumen label -- {len(rows)} samples / {summary['n_patients']} patients")
    if missing:
        print(f"  WARNING: {len(missing)} samples have no manual mask and were skipped")
    print(f"  distance units: {summary['distance_units']}")
    for metric in METRICS:
        print("  " + format_summary(metric, summary["metrics"][metric]))
    print(f"  mask status: {summary['status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
