#!/usr/bin/env python3
"""Phase 1 -- freeze the split the trained checkpoints actually used.

No leakage was found, so the existing patient-level assignment is preserved
verbatim rather than re-rolled: re-splitting would invalidate every checkpoint
in checkpoints/ and every result in runs/. This writes a self-contained record
(assignment + provenance + stratification metadata) so future experiments can
reproduce the identical split without reading the dataset directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/home/rosotauser/datasets/pfus/manifest.csv")
    parser.add_argument("--splits-json", default="/home/rosotauser/datasets/pfus/splits.json")
    parser.add_argument("--stats", default=None, help="Optional per-patient stats CSV for stratification metadata")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assignment: dict[str, list[str]] = defaultdict(list)
    frames: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    for row in rows:
        pid, split = row["patient_id"], row["split"]
        frames[pid] += 1
        if pid not in seen:
            seen.add(pid)
            assignment[split].append(pid)
    for split in assignment:
        assignment[split].sort()

    source = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))
    digest = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()

    payload = {
        "provenance": (
            "Snapshot of the patient-level split already present in manifest.csv, written by "
            "scripts/prepare_pfus.py via split_by_patient(). NOT a re-roll: checkpoints/ and "
            "runs/ were produced with exactly this assignment."
        ),
        "seed": source.get("seed"),
        "ratios": source.get("ratios"),
        "labels": source.get("labels"),
        "split_unit": "patient",
        "split_method": "sha256(f'{seed}:{patient_id}') rank order, then ratio cut",
        "manifest_path": args.manifest,
        "manifest_sha256": digest,
        "leakage_audit": "dataset_audit/leakage_audit.json (train/val/test intersections all empty)",
        "num_patients": len(seen),
        "num_frames": len(rows),
        "frames_per_patient": dict(sorted(frames.items())),
        "assignment": dict(assignment),
    }

    if args.stats and Path(args.stats).exists():
        with open(args.stats, newline="", encoding="utf-8") as handle:
            payload["stratification_metadata"] = {
                r["patient_id"]: {k: v for k, v in r.items() if k != "patient_id"}
                for r in csv.DictReader(handle)
            }

    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"  seed={payload['seed']} ratios={payload['ratios']} patients={payload['num_patients']}")
    print(f"  manifest sha256={digest[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
