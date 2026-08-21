#!/usr/bin/env python3
"""Phase 12 -- zero-shot evaluation of a PFUS1 checkpoint on an external dataset.

Deliberately generic: any directory matching the contract in README.md works.
The checkpoint is used EXACTLY as trained -- no threshold sweep, no
recalibration, no test-time augmentation. A zero-shot number that has been
tuned on the target set is not a zero-shot number.

Aggregation is per SUBJECT, matching the patient-level statistics used
everywhere else in this audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REQUIRED = ("subject_id", "frame_id", "image_path", "mask_path")


def load_dataset(directory: Path) -> list[dict[str, str]]:
    metadata = directory / "metadata.csv"
    if not metadata.exists():
        raise SystemExit(
            f"{metadata} not found.\n"
            f"Expected layout is documented in {HERE / 'README.md'}."
        )
    rows = list(csv.DictReader(metadata.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{metadata} has no rows.")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise SystemExit(f"{metadata} is missing required column(s): {missing}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="directory name under external_data/")
    parser.add_argument("--checkpoint", default=str(REPO / "checkpoints" / "pfus_bladder" / "best.pt"))
    parser.add_argument("--config", default=str(REPO / "configs" / "pfus_bladder.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    directory = HERE / args.dataset
    if not directory.is_dir():
        raise SystemExit(
            f"No external dataset at {directory}.\n"
            f"This is expected: none is present on this machine yet. See {HERE / 'README.md'}\n"
            f"for the required layout, then re-run. No download is attempted automatically."
        )
    rows = load_dataset(directory)
    if args.limit:
        rows = rows[: args.limit]

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(HERE.parent / "metrics"))
    from rus_perception.control.features import FeatureExtractionConfig
    from rus_perception.data.io import load_grayscale, load_mask
    from rus_perception.inference.predictor import Predictor, PredictorConfig
    from rus_perception.utils.config import load_config
    from segmentation_metrics import compare_masks, format_summary, summarise

    config = load_config(args.config)
    size = config.get("model.input_size") or config.get("data.image_size") or [128, 128]
    predictor = Predictor.from_checkpoint(
        args.checkpoint,
        model_config=config.section("model"),
        predictor_config=PredictorConfig(
            input_size=(int(size[0]), int(size[1])),
            intensity_normalization=str(config.get("data.intensity_normalization", "zero_one")),
            normalization_stats=config.get("data.normalization_stats"),
            device=args.device,
            restore_original_size=True,
        ),
        feature_config=FeatureExtractionConfig.from_dict(
            {"postprocess": config.section("postprocess"), **config.section("control")}
        ),
    )

    spacing_column = "pixel_spacing_mm" if "pixel_spacing_mm" in rows[0] else None
    out = Path(args.output_dir or directory / "zero_shot")
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for index, row in enumerate(rows):
        image = load_grayscale(directory / row["image_path"])
        reference = load_mask(directory / row["mask_path"]) > 0
        state = predictor.predict_control_state(image)
        prediction = np.asarray(state.binary_mask).astype(bool)
        spacing = float(row[spacing_column]) if spacing_column and row.get(spacing_column) else 1.0
        comparison = compare_masks(prediction, reference, spacing=spacing)
        records.append({
            "subject_id": row["subject_id"], "frame_id": row["frame_id"],
            "view": row.get("view", ""), "spacing": spacing,
            **comparison.to_dict(),
        })
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(rows)}")

    with (out / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    subjects = [r["subject_id"] for r in records]
    metrics = ("dice", "iou", "hd95", "assd", "relative_area_error")
    summary = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "regime": "zero-shot (no fine-tuning, no threshold retuning)",
        "n_frames": len(records),
        "n_subjects": len(set(subjects)),
        "distance_units": "mm" if spacing_column else "pixels",
        "metrics": {m: summarise([r[m] for r in records], subjects) for m in metrics},
        "status_counts": {s: sum(1 for r in records if r["status"] == s)
                          for s in {r["status"] for r in records}},
    }
    (out / "zero_shot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nZero-shot: {args.checkpoint} -> {args.dataset}")
    print(f"  {len(records)} frames / {summary['n_subjects']} subjects, units: {summary['distance_units']}")
    for metric in metrics:
        print("  " + format_summary(metric, summary["metrics"][metric]))
    print(f"  mask status: {summary['status_counts']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
