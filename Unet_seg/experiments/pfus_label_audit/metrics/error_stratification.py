#!/usr/bin/env python3
"""Phase 8 -- where does the error actually live?

Two tables, deliberately:

  full test (2,017 frames)  joins runs/*/eval_test/frame_metrics.csv with the
      per-frame image descriptors from Phase 2. Large N, but no ASSD and the
      metrics are at the model's 256x256 working resolution.

  audit subset (80 frames)  uses model_label_comparison.csv, which has ASSD and
      native-resolution distances.

Everything is aggregated patient-first: a patient with 200 frames must not
outweigh one with 73. Frame counts are reported so thin strata are visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_metrics import patient_bootstrap_ci  # noqa: E402

STRATA = ("bladder_size", "contrast_level", "boundary_visibility", "shadow_present",
          "irregular_contour", "deformation")
ORDER = {
    "bladder_size": ["small", "medium", "large"],
    "contrast_level": ["low", "medium", "high"],
    "boundary_visibility": ["weak", "moderate", "clear"],
    "shadow_present": ["0", "1"],
    "irregular_contour": ["0", "1"],
    "deformation": ["0", "1"],
}


def group(rows, key, metrics):
    out = {}
    buckets = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(row)
    for value in ORDER.get(key, sorted(buckets)):
        subset = buckets.get(value)
        if not subset:
            continue
        patients = [r["patient_id"] for r in subset]
        entry = {"n_frames": len(subset), "n_patients": len(set(patients))}
        for metric in metrics:
            values = [r.get(metric) for r in subset]
            values = [float(v) for v in values if v not in (None, "", "None")]
            if not values:
                entry[metric] = None
                continue
            sub_patients = [p for p, r in zip(patients, subset)
                            if r.get(metric) not in (None, "", "None")]
            ci = patient_bootstrap_ci(values, sub_patients, n_boot=4000)
            entry[metric] = {"patient_mean": ci["point"], "ci_low": ci["ci_low"],
                             "ci_high": ci["ci_high"], "frame_mean": float(np.mean(values))}
        out[value] = entry
    return out


def show(title, table, metrics):
    print(f"\n{title}")
    header = f"  {'stratum':<12} {'frames':>7} {'pat':>4}"
    for metric in metrics:
        header += f" {metric:>26}"
    print(header)
    for value, entry in table.items():
        line = f"  {value:<12} {entry['n_frames']:>7} {entry['n_patients']:>4}"
        for metric in metrics:
            cell = entry.get(metric)
            line += (f" {cell['patient_mean']:>10.4f} [{cell['ci_low']:.3f},{cell['ci_high']:.3f}]"
                     if cell else f" {'-':>26}")
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    repo = root.parents[1]
    parser.add_argument("--features", default=str(root / "dataset_audit" / "frame_features_test.csv"))
    parser.add_argument("--frame-metrics", default=str(repo / "runs" / "pfus_bladder" / "eval_test" / "frame_metrics.csv"))
    parser.add_argument("--slim-frame-metrics", default=str(repo / "runs" / "pfus_bladder_slim" / "eval_test" / "frame_metrics.csv"))
    parser.add_argument("--audit-metrics", default=str(root / "model_label_comparison.csv"))
    parser.add_argument("--output", default=str(root / "metrics" / "error_stratification.json"))
    args = parser.parse_args()

    # frame_metrics.csv keys frames as "P000/seq0/000000"; the Phase 2 feature
    # table keys them as (patient_id, frame_index). Join on the index, which both
    # carry unambiguously, rather than on either string form.
    features = {(r["patient_id"], int(r["frame_index"])): r
                for r in csv.DictReader(open(args.features, encoding="utf-8"))}
    payload = {}

    for name, path in (("standard", args.frame_metrics), ("slim", args.slim_frame_metrics)):
        rows = []
        for row in csv.DictReader(open(path, encoding="utf-8")):
            feature = features.get((row["patient_id"], int(row["frame_index"])))
            if feature:
                rows.append({**row, **{k: feature[k] for k in STRATA}})
        metrics = ["dice", "iou", "hd95"]
        payload[f"full_test|{name}"] = {k: group(rows, k, metrics) for k in STRATA}
        print(f"\n{'=' * 96}\nFULL TEST SPLIT -- {name}  ({len(rows)} frames, metrics at 256x256)\n{'=' * 96}")
        for key in STRATA:
            show(key, payload[f"full_test|{name}"][key], metrics)

    if Path(args.audit_metrics).exists():
        audit_rows = [r for r in csv.DictReader(open(args.audit_metrics, encoding="utf-8"))
                      if r["resolution"] == "native" and r["reference"] == "pfus"]
        for name in ("standard", "slim"):
            rows = [r for r in audit_rows if r["model"] == name]
            if not rows:
                continue
            metrics = ["dice", "hd95", "assd", "relative_area_error"]
            available = [k for k in STRATA if k in rows[0]]
            payload[f"audit|{name}"] = {k: group(rows, k, metrics) for k in available}
            print(f"\n{'=' * 96}\nAUDIT SUBSET -- {name}  ({len(rows)} frames, native resolution)\n{'=' * 96}")
            for key in available:
                show(key, payload[f"audit|{name}"][key], metrics)

    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
