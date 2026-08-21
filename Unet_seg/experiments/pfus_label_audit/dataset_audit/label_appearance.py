#!/usr/bin/env python3
"""Does each patient's label sit on fluid? Measured without any model.

A urine-filled bladder lumen is anechoic: dark, and smooth inside relative to
the tissue around it. A label drawn over mid-grey speckle is not annotating
fluid, whatever it is called. This script measures, per patient,

    contrast    mean(ring around the label) - mean(label interior)   [8-bit]
    texture     std(label interior) / std(surrounding ring)
    darkness    label interior mean as a percentile of the sector

all restricted to the imaging sector so the black surround cannot skew them,
and all computed from image and label alone -- no prediction is involved, so an
exclusion based on these numbers cannot be circular.

Output: label_appearance.csv + label_appearance.json.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SIZE = 256
MANIFEST = Path("/home/rosotauser/datasets/pfus/manifest.csv")
HERE = Path(__file__).resolve().parent


def sector(image: np.ndarray) -> np.ndarray:
    """Imaging sector, excluding the black letterbox."""
    binary = cv2.morphologyEx(
        (image > 8).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count < 2:
        return np.ones(image.shape, dtype=bool)
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def frame_stats(image: np.ndarray, mask: np.ndarray) -> dict[str, float] | None:
    """Contrast, texture ratio and darkness percentile for one frame."""
    if not mask.any():
        return None
    field = sector(image)
    ring = cv2.dilate(
        mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    ).astype(bool) & field & ~mask
    inside = image[mask & field]
    if inside.size == 0 or not ring.any():
        return None
    outside = image[ring]
    sector_values = image[field]
    return {
        "contrast": float(outside.mean() - inside.mean()),
        "texture": float(inside.std() / outside.std()) if outside.std() > 1e-6 else 1.0,
        "darkness_pct": float((sector_values < inside.mean()).mean() * 100.0),
        "area_frac": float(mask.sum() / max(1, field.sum())),
    }


def main() -> int:
    """Run the appearance audit over every patient."""
    root = MANIFEST.parent
    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    with MANIFEST.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_patient[row["patient_id"]].append(row)

    rows = []
    for position, (patient, records) in enumerate(sorted(by_patient.items()), start=1):
        records.sort(key=lambda r: int(r["frame_index"]))
        collected = []
        for record in records:
            image = cv2.imread(str(root / record["image_path"]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(root / record["mask_path"]), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 127
            stats = frame_stats(image, mask)
            if stats is not None:
                collected.append(stats)

        if not collected:
            continue
        rows.append(
            {
                "patient_id": patient,
                "split": records[0]["split"],
                "num_frames": len(records),
                "contrast": round(float(np.median([c["contrast"] for c in collected])), 2),
                "texture": round(float(np.median([c["texture"] for c in collected])), 3),
                "darkness_pct": round(float(np.median([c["darkness_pct"] for c in collected])), 1),
                "area_frac": round(float(np.median([c["area_frac"] for c in collected])), 4),
            }
        )
        if position % 25 == 0 or position == len(by_patient):
            print(f"  {position}/{len(by_patient)} patients", flush=True)

    out_csv = HERE / "label_appearance.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    contrast = np.asarray([r["contrast"] for r in rows])
    summary = {
        "num_patients": len(rows),
        "contrast_percentiles": {
            str(p): float(np.percentile(contrast, p)) for p in (0, 5, 10, 25, 50, 75, 90, 100)
        },
        "lowest_15": sorted(rows, key=lambda r: r["contrast"])[:15],
    }
    (HERE / "label_appearance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nlabel contrast (surround - interior, 8-bit) over %d patients:" % len(rows))
    for p in (0, 5, 10, 25, 50, 75, 90, 100):
        print("  p%-3d %+7.1f" % (p, np.percentile(contrast, p)))
    print("\nlowest 15 (label interior no darker than its surroundings):")
    for row in summary["lowest_15"]:
        print("  %s split=%-5s contrast=%+6.1f texture=%.2f darkness_pct=%4.1f area=%.3f" % (
            row["patient_id"], row["split"], row["contrast"], row["texture"],
            row["darkness_pct"], row["area_frac"]))
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
