#!/usr/bin/env python3
"""Per-patient label-integrity audit -- computed WITHOUT any model prediction.

A label that does not move while its image does is not an annotation of that
image; it is a template left behind. This script measures exactly that, over
every patient in the dataset, so patients can be excluded on evidence about the
*label* rather than on the model's score against it -- which would be circular
and would inflate the reported performance.

Per consecutive frame pair it measures
    image displacement  -- global translation from phase correlation (px @256)
    label displacement  -- mask-centroid translation             (px @256)
and per patient reports the medians and their ratio. A frozen label shows
image displacement in the normal range and label displacement at ~0.

Output: label_integrity.csv (one row per patient) + label_integrity.json.
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


def load_pair(root: Path, image_rel: str, mask_rel: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one frame and its mask, resized to the common ``SIZE`` grid."""
    image = cv2.imread(str(root / image_rel), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(root / mask_rel), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise SystemExit(f"unreadable: {image_rel} / {mask_rel}")
    image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 127
    return image, mask


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Mask centroid in pixels, or ``None`` for an empty mask."""
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def main() -> int:
    """Run the audit over every patient in the manifest."""
    root = MANIFEST.parent
    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    with MANIFEST.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_patient[row["patient_id"]].append(row)

    rows = []
    for position, (patient, records) in enumerate(sorted(by_patient.items()), start=1):
        records.sort(key=lambda r: int(r["frame_index"]))
        image_steps, label_steps, areas = [], [], []
        previous_image = previous_centroid = None
        empty_frames = 0

        for record in records:
            image, mask = load_pair(root, record["image_path"], record["mask_path"])
            areas.append(float(mask.sum()))
            if not mask.any():
                empty_frames += 1
            current_centroid = centroid(mask)

            if previous_image is not None:
                shift, _ = cv2.phaseCorrelate(
                    previous_image.astype(np.float32), image.astype(np.float32)
                )
                image_steps.append(float(np.hypot(*shift)))
                if current_centroid is not None and previous_centroid is not None:
                    label_steps.append(
                        float(
                            np.hypot(
                                current_centroid[0] - previous_centroid[0],
                                current_centroid[1] - previous_centroid[1],
                            )
                        )
                    )
            previous_image, previous_centroid = image, current_centroid

        area = np.asarray(areas, dtype=np.float64)
        image_median = float(np.median(image_steps)) if image_steps else 0.0
        label_median = float(np.median(label_steps)) if label_steps else 0.0
        rows.append(
            {
                "patient_id": patient,
                "split": records[0]["split"],
                "num_frames": len(records),
                "empty_label_frames": empty_frames,
                "image_step_px": round(image_median, 4),
                "label_step_px": round(label_median, 4),
                "motion_ratio": round(label_median / image_median, 4) if image_median > 1e-6 else 0.0,
                "label_total_travel_px": round(float(np.sum(label_steps)), 2),
                "label_area_mean": round(float(area.mean()), 1),
                "label_area_cv": round(float(area.std() / area.mean()), 4) if area.mean() > 0 else 0.0,
            }
        )
        if position % 20 == 0 or position == len(by_patient):
            print(f"  {position}/{len(by_patient)} patients", flush=True)

    out_csv = HERE / "label_integrity.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ratios = np.asarray([r["motion_ratio"] for r in rows])
    summary = {
        "num_patients": len(rows),
        "motion_ratio": {
            "min": float(ratios.min()),
            "p05": float(np.percentile(ratios, 5)),
            "p25": float(np.percentile(ratios, 25)),
            "median": float(np.median(ratios)),
            "p75": float(np.percentile(ratios, 75)),
            "max": float(ratios.max()),
        },
        "lowest_10": sorted(rows, key=lambda r: r["motion_ratio"])[:10],
    }
    (HERE / "label_integrity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nmotion_ratio (label displacement / image displacement) over {len(rows)} patients:")
    print("  min %.3f  p05 %.3f  p25 %.3f  median %.3f  p75 %.3f  max %.3f" % (
        ratios.min(), np.percentile(ratios, 5), np.percentile(ratios, 25),
        np.median(ratios), np.percentile(ratios, 75), ratios.max()))
    print("\nlowest 10 (label barely moves while the image does):")
    for row in summary["lowest_10"]:
        print("  %s  split=%-5s ratio=%.3f  label=%.3f px  image=%.3f px  frames=%d" % (
            row["patient_id"], row["split"], row["motion_ratio"],
            row["label_step_px"], row["image_step_px"], row["num_frames"]))
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
