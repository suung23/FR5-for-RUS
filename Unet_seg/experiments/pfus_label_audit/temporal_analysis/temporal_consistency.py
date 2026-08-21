#!/usr/bin/env python3
"""SUPERSEDED for headline numbers by metrics/final_evaluation.py, which uses
rus_perception.metrics.temporal directly. Kept for the per-transition CSV.

Phase 9 (part 2) -- temporal consistency, with and without motion compensation.

Two metrics per transition, on the model's own predictions:

  Method 1  TC_t = Dice(M_t, M_{t-1})
      Raw consecutive-mask overlap. Confounded by real anatomy motion, so it is
      a reference number, not a stability measure.

  Method 2  TC_t = Dice(M_t, W(M_{t-1}))
      Farneback optical flow between consecutive frames warps the previous
      prediction forward first, so genuine probe/anatomy motion is removed and
      what remains is segmentation instability.

IMPORTANT -- what the existing report actually measured. runs/*/eval_test was
produced with flow.backend: precomputed and a manifest whose flow columns are
empty, so scripts/evaluate.py took its `flow_backward is None` branch and set
`warped_previous = previous_state.binary_mask`. The number published there as
"Warped temporal Dice" is therefore Method 1, not Method 2. This script computes
both and reports the gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metrics"))
from segmentation_metrics import patient_bootstrap_ci  # noqa: E402


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    """Dice between two binary masks; ``None`` when both are empty.

    Both-empty is an ABSENT transition, not a perfect one. This matches the
    single definition in rus_perception.metrics.temporal (v2) -- see that
    module's docstring. An earlier revision returned 1.0 here, which credited a
    61%-dropout sequence with 0.946 stability.
    """
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return None
    return 2.0 * int(np.logical_and(a, b).sum()) / total


def warp(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp `mask` along `flow` (previous -> current) with nearest sampling."""
    height, width = mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(mask.astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", default=str(here))
    parser.add_argument("--split", default="test")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--models", nargs="+", default=["standard", "slim"])
    parser.add_argument("--output", default=str(here.parent / "temporal_consistency.csv"))
    parser.add_argument("--summary", default=str(here / "temporal_consistency_summary.json"))
    args = parser.parse_args()

    source = Path(args.input_dir)
    images = np.load(source / f"images_{args.split}_{args.size}.npz")
    patients = sorted({k.split("__")[0] for k in images.files})

    # Optical flow depends only on the images, so compute it once for all models.
    print(f"[flow] Farneback over {len(patients)} sequences")
    flows: dict[str, np.ndarray] = {}
    for patient in patients:
        stack = images[f"{patient}__image"]
        sequence = []
        for index in range(1, len(stack)):
            sequence.append(
                cv2.calcOpticalFlowFarneback(
                    stack[index], stack[index - 1], None,
                    pyr_scale=0.5, levels=3, winsize=21, iterations=3,
                    poly_n=7, poly_sigma=1.5, flags=0,
                )
            )
        flows[patient] = np.stack(sequence) if sequence else np.zeros((0, args.size, args.size, 2), np.float32)
        print(f"  {patient}: {len(sequence)} transitions")

    rows, per_patient = [], []
    for model in args.models:
        path = source / f"masks_{model}_{args.split}_{args.size}.npz"
        if not path.exists():
            print(f"[{model}] {path.name} missing, skipping")
            continue
        masks = np.load(path)
        for patient in patients:
            stack = masks[patient].astype(bool)
            raw, compensated, both_empty, dropouts = [], [], 0, 0
            for index in range(1, len(stack)):
                previous, current = stack[index - 1], stack[index]
                if not current.any():
                    dropouts += 1
                if not previous.any() and not current.any():
                    both_empty += 1
                raw_value = dice(current, previous)
                warped_value = dice(current, warp(previous, flows[patient][index - 1]))
                # None == absent transition: excluded from the mean, counted above.
                if raw_value is not None:
                    raw.append(raw_value)
                if warped_value is not None:
                    compensated.append(warped_value)
                rows.append({
                    "model": model, "patient_id": patient, "transition_index": index,
                    "tc_method1_raw": "" if raw_value is None else round(raw_value, 6),
                    "tc_method2_flow_warped": "" if warped_value is None else round(warped_value, 6),
                    "current_empty": int(not current.any()),
                    "previous_empty": int(not previous.any()),
                })
            per_patient.append({
                "model": model, "patient_id": patient, "num_frames": int(len(stack)),
                "num_transitions": int(len(stack) - 1),
                "num_scored_transitions": len(compensated),
                "tc_method1_mean": float(np.mean(raw)) if raw else None,
                "tc_method2_mean": float(np.mean(compensated)) if compensated else None,
                "tc_method2_p05": float(np.percentile(compensated, 5)) if compensated else None,
                "motion_compensation_gain": float(np.mean(compensated) - np.mean(raw)) if raw else None,
                "mask_dropout_rate": dropouts / max(len(stack), 1),
                "both_empty_transitions": both_empty,
            })

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_patient[0]))
        writer.writeheader()
        writer.writerows(per_patient)
    with open(Path(args.output).with_name("temporal_consistency_transitions.csv"), "w",
              newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"split": args.split, "resolution": args.size, "flow": "Farneback (cv2)", "models": {}}
    for model in args.models:
        subset = [r for r in per_patient if r["model"] == model]
        if not subset:
            continue
        keys = [r["patient_id"] for r in subset]
        entry = {
            "n_patients": len(subset),
            "method1_raw": patient_bootstrap_ci([r["tc_method1_mean"] for r in subset], keys, n_boot=4000),
            "method2_flow_warped": patient_bootstrap_ci([r["tc_method2_mean"] for r in subset], keys, n_boot=4000),
            "mask_dropout_rate": patient_bootstrap_ci([r["mask_dropout_rate"] for r in subset], keys, n_boot=4000),
            "worst_patients": sorted(
                [{"patient_id": r["patient_id"], "tc_method2_mean": r["tc_method2_mean"],
                  "mask_dropout_rate": r["mask_dropout_rate"]} for r in subset],
                key=lambda r: r["tc_method2_mean"],
            )[:5],
        }
        summary["models"][model] = entry
        print(f"\n[{model}]  {len(subset)} patients")
        for key in ("method1_raw", "method2_flow_warped", "mask_dropout_rate"):
            ci = entry[key]
            print(f"  {key:<22} {ci['point']:.4f}  95%CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
        print(f"  worst by motion-compensated TC: "
              f"{', '.join(f'{w['patient_id']}={w['tc_method2_mean']:.3f}' for w in entry['worst_patients'])}")

    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
