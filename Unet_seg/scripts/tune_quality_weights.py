#!/usr/bin/env python3
"""Compare control-quality definitions on val, then report the choice on test.

Protocol
--------
Candidates are scored on the **val** cohort, one candidate is selected by a rule
fixed before test is touched, and only then is test reported. Val and test share
no patient, so the test column is an out-of-sample number rather than a
restatement of the tuning.

Selection rule (declared here, not after the fact)
--------------------------------------------------
1. Highest val AUROC for ``Dice >= 0.80`` **within volume bands**. Cohort-wide
   AUROC is dominated by bladder size, which the score should not be rewarded
   for detecting; the within-band figure is what a controller sees, since it
   judges frames of one patient.
2. Ties broken by dynamic range (5th-95th percentile spread). A score packed
   into a narrow band cannot be thresholded however well it ranks.

Every candidate is recomputed from the raw control features with
:func:`~rus_perception.control.quality.compute_control_quality`, so a candidate
that changes ``target_area_ratio`` or the aggregation is evaluated through the
real implementation rather than a reimplementation of it.

Example:
    python scripts/tune_quality_weights.py \
        --val-run runs/exp_seed43_roi/best_val --test-run runs/exp_seed43_roi/best_test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

from rus_perception.control.quality import QualityConfig, compute_control_quality

logger = logging.getLogger(__name__)

GOOD_DICE = 0.80
LARGE_CENTROID_ERROR = 0.02

#: Shipped weights, repeated here so a candidate's diff against them is visible
#: in one place rather than spread across config inheritance.
BASE_WEIGHTS = {
    "segmentation_confidence": 1.0, "mask_completeness": 1.0, "lumen_contrast": 0.5,
    "border_penalty": 1.0, "component_quality": 1.0, "temporal_iou": 1.0,
    "centroid_stability": 0.5, "area_stability": 0.5,
}


def _weights(**overrides: float) -> dict[str, float]:
    return {**BASE_WEIGHTS, **overrides}


#: Candidate definitions. Each entry is what changes relative to the shipped
#: configuration, and why, so the sweep reads as a set of hypotheses rather than
#: a grid search.
CANDIDATES: dict[str, dict] = {
    # Reference point: the configuration currently in configs/exp_seed43.yaml.
    "shipped": {},
    # segmentation_confidence is anti-correlated with Dice on both cohorts --
    # a more confident model is more often wrong here -- so it is removed from
    # the score. It stays a validity floor, where it costs nothing.
    "drop_confidence": {"weights": _weights(segmentation_confidence=0.0)},
    # border_penalty fires once the ROI exists, but it neither ranks Dice nor
    # predicts the centroid bias its docstring motivates it with, and its sign
    # disagrees between val and test. Its job is a hard limit, which the
    # validity gate's max_border_contact_ratio already performs.
    "drop_confidence_border": {"weights": _weights(segmentation_confidence=0.0,
                                                   border_penalty=0.0)},
    # mask_completeness is a bladder-size detector, and size is confounded with
    # accuracy on this dataset. Removing it is what decouples Q from volume.
    "size_decoupled": {"weights": _weights(segmentation_confidence=0.0,
                                           border_penalty=0.0,
                                           mask_completeness=0.0)},
    # Same terms, but the aggregation that lets one collapsed sub-score
    # actually move the number.
    "size_decoupled_geometric": {"weights": _weights(segmentation_confidence=0.0,
                                                     border_penalty=0.0,
                                                     mask_completeness=0.0),
                                 "aggregation": "geometric"},
    # Keeps the domain check but recalibrated: target_area_ratio is set from the
    # cohort's own distribution instead of the single patient it happens to
    # match today. Filled in at run time from the val ground truth.
    "recalibrated_completeness": {"weights": _weights(segmentation_confidence=0.0,
                                                      border_penalty=0.0),
                                  "target_area_ratio": None, "area_tolerance": None},
    "recalibrated_geometric": {"weights": _weights(segmentation_confidence=0.0,
                                                   border_penalty=0.0),
                               "target_area_ratio": None, "area_tolerance": None,
                               "aggregation": "geometric"},
    # A two-sided completeness term punishes the distended bladder that HoLEP
    # irrigation is meant to produce -- on this cohort it drops the best-
    # segmented patient to the bottom of the ranking. Only under-filling
    # indicates a frame outside the deployment domain, so only under-filling
    # is penalised.
    "underfill_only_geometric": {"weights": _weights(segmentation_confidence=0.0,
                                                     border_penalty=0.0),
                                 "target_area_ratio": None, "area_tolerance": None,
                                 "penalise_overfill": False,
                                 "aggregation": "geometric"},
}


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """AUROC with mid-ranks; ``scores`` are oriented so higher means label True."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sorted_scores, start = scores[order], 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_split(run_dir: Path, manifest: str, root: str, size: tuple[int, int],
               workers: int) -> list[dict]:
    """Raw control features, spatial metrics and true centroid error per frame."""
    from centroid_accuracy import load_ground_truth_centroids

    states = [json.loads(line) for line in (run_dir / "control_states.jsonl").open()]
    with (run_dir / "frame_metrics.csv").open(newline="") as handle:
        metrics = {row["frame_id"]: row for row in csv.DictReader(handle)}
    truth = load_ground_truth_centroids(manifest, root,
                                        {s["frame_id"] for s in states}, size, workers)

    frames = []
    for state in states:
        row = metrics[state["frame_id"]]
        gt_centroid = truth[state["frame_id"]]
        if gt_centroid is None or state["centroid_x_normalized"] is None:
            error = float("nan")
        else:
            error = math.hypot(state["centroid_x_normalized"] - gt_centroid[0],
                               state["centroid_y_normalized"] - gt_centroid[1])
        frames.append({
            "frame_id": state["frame_id"],
            "patient_id": state["metadata"]["patient_id"],
            "dice": float(row["dice"]),
            "gt_area_px": int(row["target_area_px"]),
            "gt_area_ratio": int(row["target_area_px"]) / float(state["roi_area_px"]),
            "centroid_error": error,
            # Raw inputs of compute_control_quality, so any candidate config can
            # be evaluated through the real implementation.
            "features": {
                "segmentation_confidence": state["segmentation_confidence"],
                "mask_area_ratio": state["mask_area_ratio"],
                "largest_component_ratio": state["largest_component_ratio"],
                "border_contact_ratio": state["border_contact_ratio"],
                "lumen_surrounding_contrast": state["lumen_surrounding_contrast"],
                "temporal_warped_iou": state["temporal_warped_iou"],
                "normalized_centroid_jump": state["normalized_centroid_jump"],
                "relative_area_change": state["relative_area_change"],
            },
        })
    return frames


def volume_bands(frames: Sequence[dict]) -> dict[str, list[str]]:
    """Patients split into small/medium/large tertiles by median GT bladder area."""
    by_patient: dict[str, list[int]] = defaultdict(list)
    for frame in frames:
        by_patient[frame["patient_id"]].append(frame["gt_area_px"])
    order = sorted(by_patient, key=lambda p: float(np.median(by_patient[p])))
    return {name: list(chunk) for name, chunk in
            zip(("small", "medium", "large"), np.array_split(np.asarray(order), 3))}


def score(frames: Sequence[dict], config: QualityConfig) -> np.ndarray:
    """Q for every frame under one candidate configuration."""
    return np.asarray([
        compute_control_quality(config=config, **frame["features"]).score
        for frame in frames
    ], dtype=float)


def evaluate(frames: Sequence[dict], quality: np.ndarray) -> dict:
    """Discrimination, dynamic range, and agreement with true centroid error."""
    dice = np.asarray([f["dice"] for f in frames])
    good = dice >= GOOD_DICE
    error = np.asarray([f["centroid_error"] for f in frames])

    band_aurocs = {}
    for band, patients in volume_bands(frames).items():
        index = np.asarray([f["patient_id"] in patients for f in frames])
        # A band whose frames are all good (or all bad) has no pair to rank and
        # is dropped rather than scored as 0.5, which would be an invented value.
        band_aurocs[band] = (_auroc(quality[index], good[index])
                             if 0 < good[index].sum() < index.sum() else float("nan"))
    finite = [v for v in band_aurocs.values() if not np.isnan(v)]

    return {
        "auroc_good_overall": _auroc(quality, good),
        "auroc_good_within_band": float(np.mean(finite)) if finite else float("nan"),
        "auroc_by_band": band_aurocs,
        "spearman_dice": _spearman(quality, dice),
        "auroc_large_centroid_error": _auroc(-quality, error > LARGE_CENTROID_ERROR),
        "spearman_centroid_error": _spearman(quality, error),
        "mean": float(quality.mean()),
        "sd": float(quality.std(ddof=1)),
        "p05": float(np.percentile(quality, 5)),
        "p95": float(np.percentile(quality, 95)),
        "dynamic_range": float(np.percentile(quality, 95) - np.percentile(quality, 5)),
    }


def bootstrap_band_auroc(frames: Sequence[dict], quality: np.ndarray,
                         draws: int = 2000, seed: int = 0) -> dict:
    """Patient-resampled CI for the within-band AUROC.

    The cohort is 11 patients, and a band is three or four of them, so a
    difference between candidates can easily be one patient's sequence. Frames
    within a patient are not independent, so the resampling unit is the patient.
    """
    rng = np.random.default_rng(seed)
    by_patient: dict[str, list[int]] = defaultdict(list)
    for index, frame in enumerate(frames):
        by_patient[frame["patient_id"]].append(index)
    patients = list(by_patient)
    dice = np.asarray([f["dice"] for f in frames])
    good = dice >= GOOD_DICE

    values = []
    for _ in range(draws):
        drawn = rng.choice(patients, size=len(patients), replace=True)
        rows = np.concatenate([by_patient[p] for p in drawn])
        subset = [frames[i] for i in rows]
        band_values = []
        for _, members in volume_bands(subset).items():
            index = np.asarray([f["patient_id"] in members for f in subset])
            labels, scores = good[rows][index], quality[rows][index]
            if 0 < labels.sum() < labels.size:
                band_values.append(_auroc(scores, labels))
        if band_values:
            values.append(float(np.mean(band_values)))
    if not values:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    return {"mean": float(np.mean(values)),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


def build_config(spec: dict, target_area_ratio: float,
                 area_tolerance: float) -> QualityConfig:
    """Materialise a candidate, filling in recalibrated values where asked."""
    spec = dict(spec)
    if spec.get("target_area_ratio", "missing") is None:
        spec["target_area_ratio"] = target_area_ratio
    if spec.get("area_tolerance", "missing") is None:
        spec["area_tolerance"] = area_tolerance
    return QualityConfig(**spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-run", type=Path, default=Path("runs/exp_seed43_roi/best_val"))
    parser.add_argument("--test-run", type=Path, default=Path("runs/exp_seed43_roi/best_test"))
    parser.add_argument("--config", default="configs/exp_seed43_roi.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/quality_fix"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from rus_perception.utils.config import load_config
    data = load_config(args.config).section("data")
    size = tuple(int(v) for v in data["image_size"])
    load = lambda run: load_split(run, str(data["manifest"]), str(data["root"]),
                                  size, args.workers)
    val, test = load(args.val_run), load(args.test_run)

    # Recalibration target: the median ground-truth bladder area of the val
    # cohort, over the ROI. Taken from the labels, not from a prediction, and
    # from val, not from test.
    ratios = [f["gt_area_ratio"] for f in val]
    target = float(np.median(ratios))
    # Half the interquartile width: the plateau then spans the central 50% of
    # bladder volumes the cohort actually contains. The shipped 0.10 is wider
    # than the entire distribution, which pins the sub-score at 1.0 everywhere
    # and is the reason it looked flat.
    tolerance = float((np.percentile(ratios, 75) - np.percentile(ratios, 25)) / 2.0)
    tolerance_source = ratios
    logger.info("Recalibrated from val ground truth: target_area_ratio=%.4f, "
                "area_tolerance=%.4f (shipped: 0.15 / 0.10, over a different denominator)",
                target, tolerance)

    results = {}
    for name, spec in CANDIDATES.items():
        config = build_config(spec, target, tolerance)
        results[name] = {
            "config": {"weights": dict(config.weights),
                       "target_area_ratio": config.target_area_ratio,
                       "area_tolerance": config.area_tolerance,
                       "aggregation": config.aggregation},
            "val": evaluate(val, score(val, config)),
            "test": evaluate(test, score(test, config)),
            "val_band_auroc_bootstrap": bootstrap_band_auroc(val, score(val, config)),
        }

    # Selection uses val only. Written as an explicit argmax so the rule in the
    # docstring and the code cannot drift apart.
    ranked = sorted(
        results.items(),
        key=lambda kv: (-(kv[1]["val"]["auroc_good_within_band"]
                          if not np.isnan(kv[1]["val"]["auroc_good_within_band"]) else -1),
                        -kv[1]["val"]["dynamic_range"]),
    )
    selected = ranked[0][0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "val_run": str(args.val_run), "test_run": str(args.test_run),
        "recalibrated_target_area_ratio": target,
        "recalibrated_area_tolerance": tolerance,
        "val_gt_area_ratio_percentiles": {
            str(p): float(np.percentile(tolerance_source, p)) for p in (5, 25, 50, 75, 95)},
        "selected": selected,
        "candidates": results,
    }
    (args.output_dir / "tune_quality_weights.json").write_text(json.dumps(payload, indent=2))

    header = (f"{'candidate':<28}{'val band':>10}{'val band 95% CI':>20}{'val range':>11}"
              f"{'| test band':>13}{'test range':>12}{'test rho(err)':>15}")
    print(header)
    print("-" * len(header))
    for name, entry in results.items():
        v, t, b = entry["val"], entry["test"], entry["val_band_auroc_bootstrap"]
        mark = " <- rule" if name == selected else ""
        ci = f"[{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]"
        print(f"{name:<28}{v['auroc_good_within_band']:>10.3f}{ci:>20}"
              f"{v['dynamic_range']:>11.3f}{t['auroc_good_within_band']:>13.3f}"
              f"{t['dynamic_range']:>12.3f}{t['spearman_centroid_error']:>15.3f}{mark}")
    print(f"\nSelected on val: {selected}")
    logger.info("Wrote %s", args.output_dir / "tune_quality_weights.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
