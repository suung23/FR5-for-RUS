#!/usr/bin/env python3
"""Phase 1 -- patient-level leakage audit for the PFUS bladder task.

Reads the manifest the training runs actually consumed (not splits.json, which
is only a record of how the manifest was written) and verifies that no patient
and no sequence appears in more than one split. Writes patient_split_audit.csv
and a human-readable report; never modifies the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

SPLIT_ORDER = ("train", "val", "test")


def load_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit(rows: list[dict[str, str]]) -> dict:
    patient_splits: dict[str, set[str]] = defaultdict(set)
    sequence_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    frames_per_patient: Counter[str] = Counter()
    patient_split: dict[str, str] = {}

    for row in rows:
        pid, sid, split = row["patient_id"], row["sequence_id"], row["split"]
        patient_splits[pid].add(split)
        sequence_splits[(pid, sid)].add(split)
        frames_per_patient[pid] += 1
        patient_split[pid] = split

    per_split: dict[str, list[str]] = defaultdict(list)
    for pid, splits in patient_splits.items():
        for split in splits:
            per_split[split].append(pid)

    overlaps = {}
    for a, b in combinations(SPLIT_ORDER, 2):
        overlaps[f"{a}-{b}"] = sorted(set(per_split.get(a, [])) & set(per_split.get(b, [])))

    leaked_patients = sorted(p for p, s in patient_splits.items() if len(s) > 1)
    leaked_sequences = sorted(k for k, s in sequence_splits.items() if len(s) > 1)

    stats = {}
    for split in SPLIT_ORDER:
        pids = sorted(set(per_split.get(split, [])))
        counts = np.array([frames_per_patient[p] for p in pids], dtype=float)
        stats[split] = {
            "num_patients": len(pids),
            "num_frames": int(counts.sum()) if counts.size else 0,
            "frames_per_patient": {
                "min": float(counts.min()) if counts.size else None,
                "q1": float(np.percentile(counts, 25)) if counts.size else None,
                "median": float(np.median(counts)) if counts.size else None,
                "q3": float(np.percentile(counts, 75)) if counts.size else None,
                "max": float(counts.max()) if counts.size else None,
                "mean": float(counts.mean()) if counts.size else None,
                "std": float(counts.std(ddof=1)) if counts.size > 1 else None,
            },
            "patients": pids,
        }

    return {
        "num_rows": len(rows),
        "num_patients": len(patient_splits),
        "num_sequences": len(sequence_splits),
        "sequences_per_patient": dict(Counter(Counter(p for p, _ in sequence_splits).values())),
        "overlaps": overlaps,
        "leaked_patients": leaked_patients,
        "leaked_sequences": [list(k) for k in leaked_sequences],
        "leakage_detected": bool(leaked_patients or leaked_sequences),
        "splits": stats,
        "patient_split": patient_split,
        "frames_per_patient": dict(frames_per_patient),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/home/rosotauser/datasets/pfus/manifest.csv")
    parser.add_argument("--splits-json", default="/home/rosotauser/datasets/pfus/splits.json")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = audit(load_rows(Path(args.manifest)))

    # Cross-check against the recorded assignment, if present.
    sj = Path(args.splits_json)
    if sj.exists():
        assignment = json.loads(sj.read_text(encoding="utf-8"))["assignment"]
        mismatches = [
            {"patient_id": p, "splits_json": split, "manifest": result["patient_split"].get(p)}
            for split, pids in assignment.items()
            for p in pids
            if result["patient_split"].get(p) != split
        ]
        result["splits_json_mismatches"] = mismatches
        result["splits_json_consistent"] = not mismatches

    csv_path = out.parent / "patient_split_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "split", "num_frames"])
        for pid in sorted(result["patient_split"]):
            writer.writerow([pid, result["patient_split"][pid], result["frames_per_patient"][pid]])

    (out / "leakage_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["PFUS bladder -- Phase 1 patient-level leakage audit", "=" * 55, ""]
    for split in SPLIT_ORDER:
        s = result["splits"][split]
        lines.append(f"{split.capitalize()} patients ({s['num_patients']}): {', '.join(s['patients'])}")
    lines.append("")
    for key, value in result["overlaps"].items():
        lines.append(f"{key} overlap: {len(value)} {value if value else ''}".rstrip())
    lines.append("")
    lines.append(f"Leakage detected: {'YES' if result['leakage_detected'] else 'NO'}")
    lines.append("")
    lines.append(f"{'split':<6} {'patients':>9} {'frames':>8} {'min':>6} {'q1':>7} {'median':>7} {'q3':>7} {'max':>6} {'mean':>7} {'sd':>7}")
    for split in SPLIT_ORDER:
        s = result["splits"][split]
        f = s["frames_per_patient"]
        lines.append(
            f"{split:<6} {s['num_patients']:>9d} {s['num_frames']:>8d} "
            f"{f['min']:>6.0f} {f['q1']:>7.1f} {f['median']:>7.1f} {f['q3']:>7.1f} "
            f"{f['max']:>6.0f} {f['mean']:>7.1f} {f['std']:>7.1f}"
        )
    lines.append("")
    lines.append(f"Sequences: {result['num_sequences']} | sequences per patient: {result['sequences_per_patient']}")
    lines.append(f"Sequences spanning >1 split: {len(result['leaked_sequences'])}")
    if "splits_json_consistent" in result:
        lines.append(f"splits.json vs manifest consistent: {result['splits_json_consistent']}")
    report = "\n".join(lines)
    (out / "leakage_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
