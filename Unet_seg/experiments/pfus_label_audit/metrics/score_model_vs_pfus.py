#!/usr/bin/env python3
"""Phase 6 -- score the saved audit predictions against the PFUS1 label.

Reads predictions/predictions_<model>.npz (written by run_audit_inference.py)
and produces per-frame and aggregated metrics. Where a manual lumen mask exists
it also scores the model against that, so Phase 7's three-way ceiling analysis
has everything it needs in one table.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--audit-dir", default=str(root / "annotation_audit"))
    parser.add_argument("--predictions-dir", default=str(root / "predictions"))
    parser.add_argument("--output", default=str(root / "model_label_comparison.csv"))
    parser.add_argument("--summary", default=str(root / "metrics" / "model_vs_pfus_summary.json"))
    parser.add_argument("--models", nargs="+", default=["standard", "slim"])
    args = parser.parse_args()

    audit = Path(args.audit_dir)
    manifest = list(csv.DictReader((audit / "manifest.csv").open(encoding="utf-8")))
    by_sample = {r["sample_id"]: r for r in manifest}

    rows: list[dict] = []
    manual_available = 0
    for model in args.models:
        store = np.load(Path(args.predictions_dir) / f"predictions_{model}.npz")
        for sample_id, record in by_sample.items():
            pfus_native = np.asarray(Image.open(audit / record["original_mask_path"]).convert("L")) > 127
            manual_path = audit / record["manual_mask_path"]
            manual = (
                np.asarray(Image.open(manual_path).convert("L")) > 127 if manual_path.exists() else None
            )
            if manual is not None and model == args.models[0]:
                manual_available += 1

            for resolution, key, reference_pfus in (
                ("native", sample_id, pfus_native),
                ("r256", sample_id + "__r256", None),
            ):
                prediction = store[key].astype(bool)
                if reference_pfus is None:
                    reference_pfus = np.asarray(
                        Image.fromarray(pfus_native.astype(np.uint8) * 255).resize(
                            (prediction.shape[1], prediction.shape[0]), Image.NEAREST
                        )
                    ) > 127
                comparison = compare_masks(prediction, reference_pfus)
                row = {
                    "model": model,
                    "sample_id": sample_id,
                    "patient_id": record["patient_id"],
                    "frame_id": record["frame_id"],
                    "resolution": resolution,
                    "reference": "pfus",
                    "bladder_size": record["bladder_size"],
                    "contrast_level": record["contrast_level"],
                    "boundary_visibility": record["boundary_visibility"],
                    "shadow_present": record["shadow_present"],
                    "irregular_contour": record["irregular_contour"],
                    **comparison.to_dict(),
                }
                rows.append(row)

                if manual is not None and resolution == "native":
                    manual_cmp = compare_masks(prediction, manual)
                    rows.append({**row, "reference": "manual", **manual_cmp.to_dict()})

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"manual_masks_available": manual_available, "n_samples": len(by_sample), "groups": {}}
    print(f"Model vs PFUS1 label on the audit subset ({len(by_sample)} frames / "
          f"{len({r['patient_id'] for r in manifest})} patients)")
    print("distance units: pixels\n")
    for model in args.models:
        for resolution in ("native", "r256"):
            for reference in ("pfus", "manual"):
                subset = [r for r in rows if r["model"] == model and r["resolution"] == resolution
                          and r["reference"] == reference]
                if not subset:
                    continue
                patients = [r["patient_id"] for r in subset]
                group = {m: summarise([r[m] for r in subset], patients) for m in METRICS}
                group["n"] = len(subset)
                group["status_counts"] = {
                    s: sum(1 for r in subset if r["status"] == s) for s in {r["status"] for r in subset}
                }
                summary["groups"][f"{model}|{resolution}|{reference}"] = group
                print(f"[{model} @ {resolution}] vs {reference} GT  (n={len(subset)})")
                for metric in METRICS:
                    print("  " + format_summary(metric, group[metric]))
                print(f"  mask status: {group['status_counts']}\n")

    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not manual_available:
        print("NOTE: no manual lumen masks yet -- 'model vs manual GT' rows are absent and")
        print("      Phase 7's ceiling analysis cannot run until Phase 3 annotation is done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
