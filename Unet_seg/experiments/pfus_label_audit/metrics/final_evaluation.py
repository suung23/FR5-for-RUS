#!/usr/bin/env python3
"""Final, unbiased performance report on the PFUS1 test split.

Ground truth is the PFUS1 label, taken at face value. The only judgement made
about the data is a patient-level exclusion list, and that list is derived from
label appearance alone (dataset_audit/label_appearance.py) -- never from a
model's score, which would make the exclusion circular and the headline number
an artefact of its own selection.

Both cohorts are always reported, so the effect of the exclusion is visible
rather than assumed:

    all         every test patient
    retained    test patients minus the exclusion list

Spatial metrics are frame-level, aggregated to a patient mean, then summarised
across patients -- frames of one sequence are not independent observations.
Confidence intervals are patient-level bootstrap.

Temporal metrics use the single definition in rus_perception.metrics.temporal
(v2): motion-compensated with Farneback backward flow, both-empty transitions
excluded from the mean and reported as `absent_transition_rate`, onset/offset
scored 0. Nothing here re-implements that definition.

    python3 metrics/final_evaluation.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
REPO = AUDIT.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from segmentation_metrics import compare_masks  # noqa: E402
from rus_perception.metrics.temporal import (  # noqa: E402
    TemporalFrameRecord,
    compute_sequence_temporal_metrics,
)

MODELS = ("standard", "slim")
SPATIAL = ("dice", "iou", "hd95", "assd", "relative_area_error")


def backward_flow(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Farneback backward flow: on the current grid, pointing into the previous.

    Argument order matters and is the convention documented in
    rus_perception/flow/warp.py -- (current, previous), not the reverse.
    """
    return cv2.calcOpticalFlowFarneback(current, previous, None, 0.5, 3, 21, 3, 5, 1.2, 0)


def warp(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Pull ``mask`` along ``flow`` into the current frame's geometry."""
    height, width = mask.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    warped = cv2.remap(
        mask.astype(np.uint8),
        (grid_x + flow[..., 0]).astype(np.float32),
        (grid_y + flow[..., 1]).astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def load_exclusions(path: Path) -> tuple[list[str], str]:
    """Excluded patient ids and the human-readable rule that produced them."""
    if not path.exists():
        return [], "none (exclusion file absent)"
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("excluded", [])), str(data.get("rule", "unspecified"))


def summarise(values: list[float], patients: list[str], seed: int = 42) -> dict:
    """Patient-level mean/SD/median/IQR and a patient bootstrap 95% CI."""
    per_patient: dict[str, list[float]] = {}
    for value, patient in zip(values, patients):
        if value is not None and np.isfinite(value):
            per_patient.setdefault(patient, []).append(float(value))
    means = np.asarray([np.mean(v) for v in per_patient.values()], dtype=np.float64)
    if means.size == 0:
        return {"n_patients": 0}
    boot = _bootstrap(per_patient, seed)
    return {
        "n_patients": int(means.size),
        "mean": float(means.mean()),
        "sd": float(means.std(ddof=1)) if means.size > 1 else 0.0,
        "median": float(np.median(means)),
        "iqr": [float(np.percentile(means, 25)), float(np.percentile(means, 75))],
        "ci95": boot,
    }


def _bootstrap(per_patient: dict[str, list[float]], seed: int, draws: int = 4000) -> list[float]:
    """Resample PATIENTS with replacement; frames within a patient stay together."""
    rng = np.random.default_rng(seed)
    keys = list(per_patient)
    means = np.asarray([np.mean(per_patient[k]) for k in keys])
    if means.size < 2:
        return [float(means.mean()), float(means.mean())]
    samples = means[rng.integers(0, means.size, size=(draws, means.size))].mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def main() -> int:
    """Compute and report the final metrics."""
    temporal_dir = AUDIT / "temporal_analysis"
    images = np.load(temporal_dir / "images_test_256.npz")
    patients = sorted({key.split("__")[0] for key in images.files})
    excluded, rule = load_exclusions(AUDIT / "dataset_audit" / "excluded_patients.json")
    retained = [p for p in patients if p not in excluded]

    print(f"test patients: {len(patients)}   excluded: {len(excluded)} {excluded}")
    print(f"exclusion rule: {rule}\n")

    rows: list[dict] = []
    for model in MODELS:
        predictions = np.load(temporal_dir / f"masks_{model}_test_256.npz")
        for patient in patients:
            image_stack = images[f"{patient}__image"]
            truth_stack = images[f"{patient}__gt"] > 0
            prediction_stack = predictions[patient] > 0

            records: list[TemporalFrameRecord] = []
            patient_rows: list[dict] = []
            for index in range(len(image_stack)):
                comparison = compare_masks(prediction_stack[index], truth_stack[index])
                patient_rows.append(
                    {
                        "model": model,
                        "patient_id": patient,
                        "frame_index": index,
                        "cohort": "excluded" if patient in excluded else "retained",
                        "dice": comparison.dice,
                        "iou": comparison.iou,
                        "hd95": comparison.hd95,
                        "assd": comparison.assd,
                        "relative_area_error": comparison.relative_area_error,
                        "pred_area": comparison.area_a,
                        "gt_area": comparison.area_b,
                        "status": comparison.status,
                    }
                )
                warped_previous = None
                if index > 0:
                    flow = backward_flow(image_stack[index], image_stack[index - 1])
                    warped_previous = warp(prediction_stack[index - 1], flow)
                records.append(
                    TemporalFrameRecord(
                        frame_id=str(index),
                        mask=prediction_stack[index].astype(np.uint8),
                        warped_previous_mask=(
                            None if warped_previous is None else warped_previous.astype(np.uint8)
                        ),
                        target=truth_stack[index].astype(np.uint8),
                        motion_compensated=True,
                    )
                )

            sequence = compute_sequence_temporal_metrics(records, f"{model}/{patient}")
            for row in patient_rows:
                row["temporal_dice"] = sequence.warped_temporal_dice
                row["absent_transition_rate"] = sequence.absent_transition_rate
                row["track_break_rate"] = sequence.track_break_rate
                row["mask_dropout_rate"] = sequence.mask_dropout_rate
            rows.extend(patient_rows)
            print(f"  {model}/{patient}: {len(patient_rows)} frames", flush=True)

    out_csv = AUDIT / "final_metrics_frames.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report: dict = {"excluded_patients": excluded, "exclusion_rule": rule, "cohorts": {}}
    for cohort, members in (("all", patients), ("retained", retained)):
        report["cohorts"][cohort] = {"patients": members, "models": {}}
        for model in MODELS:
            subset = [r for r in rows if r["model"] == model and r["patient_id"] in members]
            entry = {}
            for metric in SPATIAL:
                values = [r[metric] for r in subset]
                ids = [r["patient_id"] for r in subset]
                entry[metric] = summarise(values, ids)
            per_patient = {}
            for row in subset:
                per_patient[row["patient_id"]] = row
            for metric in ("temporal_dice", "absent_transition_rate", "track_break_rate", "mask_dropout_rate"):
                values = [v[metric] for v in per_patient.values() if v[metric] is not None]
                ids = list(per_patient)
                entry[metric] = summarise(values, ids[: len(values)])
            report["cohorts"][cohort]["models"][model] = entry

    (AUDIT / "final_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def line(cohort: str, model: str, metric: str) -> str:
        s = report["cohorts"][cohort]["models"][model][metric]
        if not s.get("n_patients"):
            return "n/a"
        return "%.4f ± %.4f  [%.4f, %.4f]" % (s["mean"], s["sd"], s["ci95"][0], s["ci95"][1])

    print("\n" + "=" * 78)
    for cohort in ("all", "retained"):
        n = len(report["cohorts"][cohort]["patients"])
        print(f"\ncohort={cohort}  ({n} patients)")
        print("  %-24s %-34s %-34s" % ("metric", "standard", "slim"))
        for metric in SPATIAL + ("temporal_dice", "absent_transition_rate", "track_break_rate", "mask_dropout_rate"):
            print("  %-24s %-34s %-34s" % (metric, line(cohort, "standard", metric), line(cohort, "slim", metric)))
    print(f"\nwrote {out_csv}\nwrote {AUDIT / 'final_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
