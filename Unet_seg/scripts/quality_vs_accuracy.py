#!/usr/bin/env python3
"""Score an evaluation run with the control-quality heuristic and test it.

What this does
--------------
``scripts/evaluate.py`` already emits one :class:`ControlState` per frame, so
every frame carries the control-quality score ``Q`` and its sub-scores. This
script joins that stream to the spatial metrics of the same run and asks the
question the score exists to answer: **would gating on Q have kept the frames
the segmentation actually got right?**

Q is a heuristic (see the warning in rus_perception/control/quality.py) and it
never sees the ground truth, so agreement with Dice is not guaranteed by
construction -- which is exactly why it is worth measuring.

Three things are reported:

``Discrimination``
    Rank correlation and AUROC of Q against Dice, plus a threshold sweep giving
    the retained frame fraction and the Dice of accepted vs rejected frames.
    This is the operating-point table a controller's gate is tuned from.

``Component ablation``
    Q recomputed with one sub-score removed at a time. The weighted mean is
    reconstructed from the stored components, so the ablation costs no
    inference and cannot drift from the run it describes. It separates terms
    that carry the signal from terms that ride along.

``Confounder``
    Q's ``mask_completeness`` term peaks at ``target_area_ratio`` (0.15 by
    default), so Q is partly a bladder-size detector, and segmentation accuracy
    on this dataset is itself strongly size-dependent. Correlations are
    therefore also reported *within* volume bands, where size is held roughly
    constant and only the residual agreement remains.

Example:
    python scripts/quality_vs_accuracy.py --run-dir runs/exp_seed43/best_test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

from rus_perception.control.quality import QUALITY_COMPONENT_NAMES, QualityConfig

logger = logging.getLogger(__name__)

#: Working resolution of the evaluation protocol, for reporting areas in pixels.
PX_AT_256 = 256 * 256

#: Dice above which a frame is called "segmented well enough to steer on". The
#: gate is a binary decision, so measuring Q as a detector needs a target label;
#: 0.80 is the conventional usable-segmentation line and is reported explicitly
#: rather than tuned, since tuning it on the same run would be circular.
GOOD_DICE = 0.80

#: Q thresholds swept for the operating-point table.
GATE_THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)

#: Alternative weight sets, evaluated against the shipped one. These are
#: *diagnostics*, not a proposed default: they are scored on the same run whose
#: sub-scores suggested them, so their margin over the shipped weights is
#: optimistic and would need a held-out cohort before shipping. What they do
#: establish honestly is dynamic range -- how much of [0, 1] the score actually
#: uses -- which no threshold can recover if the score is compressed.
WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "drop_uninformative": {"mask_completeness": 1.0, "lumen_contrast": 0.5,
                           "temporal_iou": 1.0, "centroid_stability": 0.5,
                           "area_stability": 0.5},
    "contrast_weighted": {"mask_completeness": 1.0, "lumen_contrast": 2.0,
                          "temporal_iou": 1.0, "centroid_stability": 0.5,
                          "area_stability": 0.5},
    "contrast_and_completeness": {"mask_completeness": 1.0, "lumen_contrast": 1.0},
    "lumen_contrast_alone": {"lumen_contrast": 1.0},
}


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation; ``nan`` when there is nothing to rank."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    rank_x = np.argsort(np.argsort(x[ok]))
    rank_y = np.argsort(np.argsort(y[ok]))
    if rank_x.std() == 0 or rank_y.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """AUROC via the Mann-Whitney statistic, ties handled by mid-ranks.

    Written out rather than imported so this script has no SciPy/sklearn
    dependency beyond what the package already needs.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Mid-rank correction for ties, without which a flat sub-score inflates AUROC.
    sorted_scores = scores[order]
    start = 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_run(run_dir: Path) -> list[dict]:
    """Join ``control_states.jsonl`` and ``frame_metrics.csv`` on ``frame_id``.

    Raises:
        ValueError: If the two files disagree on which frames were evaluated. A
            partial join would silently drop frames from every statistic below.
    """
    states_path, metrics_path = run_dir / "control_states.jsonl", run_dir / "frame_metrics.csv"
    states = {}
    with states_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            states[record["frame_id"]] = record

    with metrics_path.open(newline="") as handle:
        metrics = {row["frame_id"]: row for row in csv.DictReader(handle)}

    if set(states) != set(metrics):
        missing = len(set(states) ^ set(metrics))
        raise ValueError(
            f"{states_path.name} and {metrics_path.name} describe different frame sets "
            f"({missing} frame ids in one but not the other). Refusing to join them."
        )

    frames = []
    for frame_id, state in states.items():
        row = metrics[frame_id]
        frames.append({
            "frame_id": frame_id,
            "patient_id": row["patient_id"],
            "frame_index": int(row["frame_index"]),
            "dice": float(row["dice"]),
            "iou": float(row["iou"]),
            "hd95": float(row["hd95"]) if row["hd95"] not in ("", "nan") else float("nan"),
            "gt_area_px": int(row["target_area_px"]),
            "quality": float(state["control_quality_score"]),
            "components": dict(state["quality_components"]),
            "valid": bool(state["valid_for_control"]),
            "rejection_reasons": list(state["rejection_reasons"]),
            "mask_area_ratio": float(state["mask_area_ratio"]),
        })
    return frames


def recompute_quality(components: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean of the stored sub-scores under a different weight set.

    Mirrors the aggregation in :func:`compute_control_quality` exactly: absent
    components (the temporal terms on a first frame) are dropped from the mean
    rather than scored as zero, and zero-weight components are excluded.
    """
    active = {n: w for n, w in weights.items() if w > 0 and n in components}
    if not active:
        return float("nan")
    total = sum(active.values())
    return float(sum(components[n] * w for n, w in active.items()) / total)


def _describe(values: Sequence[float]) -> Optional[dict]:
    """mean / SD / median / IQR, or ``None`` when empty."""
    clean = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if clean.size == 0:
        return None
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "median": float(np.median(clean)),
        "iqr": [float(np.percentile(clean, 25)), float(np.percentile(clean, 75))],
    }


def volume_bands(frames: Sequence[dict]) -> dict[str, list[str]]:
    """Split patients into small/medium/large tertiles by median GT bladder area."""
    by_patient: dict[str, list[int]] = defaultdict(list)
    for frame in frames:
        by_patient[frame["patient_id"]].append(frame["gt_area_px"])
    order = sorted(by_patient, key=lambda p: float(np.median(by_patient[p])))
    return {name: list(chunk) for name, chunk in
            zip(("small", "medium", "large"), np.array_split(np.asarray(order), 3))}


def discrimination(frames: Sequence[dict]) -> dict:
    """How well Q separates accurate segmentations from inaccurate ones."""
    quality = [f["quality"] for f in frames]
    dice = [f["dice"] for f in frames]
    good = [f["dice"] >= GOOD_DICE for f in frames]

    sweep = []
    for threshold in GATE_THRESHOLDS:
        accepted = [f for f in frames if f["quality"] >= threshold]
        rejected = [f for f in frames if f["quality"] < threshold]
        sweep.append({
            "threshold": threshold,
            "retained_fraction": len(accepted) / len(frames),
            "n_accepted": len(accepted),
            "dice_accepted": _describe([f["dice"] for f in accepted]),
            "dice_rejected": _describe([f["dice"] for f in rejected]),
            # Of the frames the gate lets through, how many really were usable.
            "precision_good": (sum(f["dice"] >= GOOD_DICE for f in accepted) / len(accepted)
                               if accepted else float("nan")),
            "recall_good": (sum(f["dice"] >= GOOD_DICE for f in accepted) / sum(good)
                            if sum(good) else float("nan")),
        })

    return {
        "n_frames": len(frames),
        "good_dice_threshold": GOOD_DICE,
        "n_good": int(sum(good)),
        "quality": _describe(quality),
        "spearman_quality_dice": _spearman(quality, dice),
        "auroc_quality_good": _auroc(quality, good),
        "gate_sweep": sweep,
    }


def component_analysis(frames: Sequence[dict], config: QualityConfig) -> dict:
    """Per-component distributions, correlations, and leave-one-out ablation."""
    dice = [f["dice"] for f in frames]
    per_component, ablation = {}, {}

    for name in QUALITY_COMPONENT_NAMES:
        values = [f["components"].get(name, float("nan")) for f in frames]
        available = sum(1 for v in values if not np.isnan(v))
        per_component[name] = {
            "weight": float(config.weights.get(name, 0.0)),
            "available_on_frames": available,
            "distribution": _describe(values),
            "spearman_with_dice": _spearman(values, dice),
            "auroc_good": _auroc(
                [v if not np.isnan(v) else 0.0 for v in values],
                [f["dice"] >= GOOD_DICE for f in frames],
            ) if available else float("nan"),
        }

        if config.weights.get(name, 0.0) <= 0:
            continue
        dropped = {n: w for n, w in config.weights.items() if n != name}
        scores = [recompute_quality(f["components"], dropped) for f in frames]
        ablation[f"without_{name}"] = {
            "spearman_with_dice": _spearman(scores, dice),
            "auroc_good": _auroc(scores, [f["dice"] >= GOOD_DICE for f in frames]),
            "quality": _describe(scores),
        }

    # The full score, recomputed the same way, so the ablation deltas are
    # differences of like with like rather than against the stored value.
    full = [recompute_quality(f["components"], config.weights) for f in frames]
    ablation["full"] = {
        "spearman_with_dice": _spearman(full, dice),
        "auroc_good": _auroc(full, [f["dice"] >= GOOD_DICE for f in frames]),
        "quality": _describe(full),
    }
    return {"per_component": per_component, "leave_one_out": ablation}


def weight_presets(frames: Sequence[dict], config: QualityConfig) -> dict:
    """Score :data:`WEIGHT_PRESETS` beside the shipped weights on this run."""
    dice = [f["dice"] for f in frames]
    good = [f["dice"] >= GOOD_DICE for f in frames]
    results = {}
    for name, weights in {"shipped": dict(config.weights), **WEIGHT_PRESETS}.items():
        scores = [recompute_quality(f["components"], weights) for f in frames]
        array = np.asarray(scores, dtype=float)
        results[name] = {
            "weights": weights,
            "spearman_with_dice": _spearman(scores, dice),
            "auroc_good": _auroc(scores, good),
            "quality": _describe(scores),
            # Dynamic range: a score packed into a narrow band cannot be
            # thresholded usefully however well it correlates.
            "p05": float(np.percentile(array, 5)),
            "p95": float(np.percentile(array, 95)),
        }
    return results


def validity_summary(frames: Sequence[dict]) -> dict:
    """Gate outcome of the shipped ValidityConfig, and why frames were rejected."""
    invalid = [f for f in frames if not f["valid"]]
    reasons = Counter(reason for f in invalid for reason in f["rejection_reasons"])
    return {
        "valid_fraction": sum(f["valid"] for f in frames) / len(frames),
        "n_invalid": len(invalid),
        "rejection_reasons": dict(reasons.most_common()),
        "dice_valid": _describe([f["dice"] for f in frames if f["valid"]]),
        "dice_invalid": _describe([f["dice"] for f in invalid]),
    }


def analyse(frames: Sequence[dict], config: QualityConfig) -> dict:
    """Full report: cohort-wide, per patient, and within each volume band."""
    bands = volume_bands(frames)
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        by_patient[frame["patient_id"]].append(frame)

    report = {
        "quality_config": {
            "weights": dict(config.weights),
            "target_area_ratio": config.target_area_ratio,
            "area_tolerance": config.area_tolerance,
            "contrast_reference": config.contrast_reference,
            "border_penalty_scale": config.border_penalty_scale,
            "centroid_jump_scale": config.centroid_jump_scale,
            "area_change_scale": config.area_change_scale,
        },
        "overall": discrimination(frames),
        "components": component_analysis(frames, config),
        "weight_presets": weight_presets(frames, config),
        "validity": validity_summary(frames),
        "per_patient": {},
        "per_volume_band": {},
    }

    for patient, rows in sorted(by_patient.items()):
        report["per_patient"][patient] = {
            "n_frames": len(rows),
            "gt_area_px_median": float(np.median([r["gt_area_px"] for r in rows])),
            "quality_mean": float(np.mean([r["quality"] for r in rows])),
            "quality_sd": float(np.std([r["quality"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "dice_mean": float(np.mean([r["dice"] for r in rows])),
            "valid_fraction": sum(r["valid"] for r in rows) / len(rows),
            "components": {
                name: float(np.nanmean([r["components"].get(name, np.nan) for r in rows]))
                for name in QUALITY_COMPONENT_NAMES
            },
        }

    for band, patients in bands.items():
        rows = [f for f in frames if f["patient_id"] in patients]
        report["per_volume_band"][band] = {
            "patients": patients,
            "n_frames": len(rows),
            "gt_area_px_median": float(np.median([r["gt_area_px"] for r in rows])),
            "quality": _describe([r["quality"] for r in rows]),
            "dice": _describe([r["dice"] for r in rows]),
            # Within-band correlation: bladder size is roughly constant here, so
            # what survives is agreement Q earns beyond being a size detector.
            "spearman_quality_dice": _spearman([r["quality"] for r in rows],
                                               [r["dice"] for r in rows]),
            "auroc_good": _auroc([r["quality"] for r in rows],
                                 [r["dice"] >= GOOD_DICE for r in rows]),
        }

    # Patient level: does a patient's mean Q rank the patients by mean Dice?
    patients = sorted(by_patient)
    report["patient_level"] = {
        "spearman_quality_dice": _spearman(
            [report["per_patient"][p]["quality_mean"] for p in patients],
            [report["per_patient"][p]["dice_mean"] for p in patients],
        ),
        "n_patients": len(patients),
    }
    return report


def _fmt(stat: Optional[dict], digits: int = 3) -> str:
    if stat is None:
        return "--"
    return (f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f} "
            f"({stat['median']:.{digits}f} [{stat['iqr'][0]:.{digits}f}–{stat['iqr'][1]:.{digits}f}])")


def render_report(report: dict, frames: Sequence[dict], run_dir: Path,
                  highlight: Sequence[tuple[str, str, int]] = ()) -> str:
    """Markdown rendering of :func:`analyse`."""
    overall, config = report["overall"], report["quality_config"]
    lines = [
        "# Control-quality scoring of the segmentation run",
        "",
        f"- Run: `{run_dir}`",
        f"- Frames: {overall['n_frames']}; "
        f"{overall['n_good']} ({overall['n_good'] / overall['n_frames'] * 100:.1f}%) "
        f"have Dice ≥ {overall['good_dice_threshold']:.2f}",
        f"- Q over the cohort: {_fmt(overall['quality'])}",
        f"- `target_area_ratio` = {config['target_area_ratio']} "
        f"± {config['area_tolerance']} (plateau "
        f"{config['target_area_ratio'] - config['area_tolerance']:.2f}–"
        f"{config['target_area_ratio'] + config['area_tolerance']:.2f} of the frame)",
        "",
        "## Does Q track segmentation accuracy?",
        "",
        f"- Spearman rho (frame level, Q vs Dice): **{overall['spearman_quality_dice']:.3f}**",
        f"- Spearman rho (patient level, mean Q vs mean Dice): "
        f"**{report['patient_level']['spearman_quality_dice']:.3f}** "
        f"(n = {report['patient_level']['n_patients']})",
        f"- AUROC of Q as a detector of Dice ≥ {overall['good_dice_threshold']:.2f}: "
        f"**{overall['auroc_quality_good']:.3f}**",
        "",
        "### Gate operating points",
        "",
        "| Q ≥ | frames kept | retained | Dice of accepted | Dice of rejected | precision | recall |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for point in overall["gate_sweep"]:
        lines.append(
            f"| {point['threshold']:.2f} | {point['n_accepted']} "
            f"| {point['retained_fraction'] * 100:.1f}% "
            f"| {_fmt(point['dice_accepted'])} | {_fmt(point['dice_rejected'])} "
            f"| {point['precision_good']:.3f} | {point['recall_good']:.3f} |"
        )

    lines += ["", "## Sub-scores", "",
              "| component | weight | frames | value | rho with Dice | AUROC |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name, entry in report["components"]["per_component"].items():
        rho = entry["spearman_with_dice"]
        auroc = entry["auroc_good"]
        lines.append(
            f"| {name} | {entry['weight']:.1f} | {entry['available_on_frames']} "
            f"| {_fmt(entry['distribution'])} "
            f"| {'--' if np.isnan(rho) else f'{rho:+.3f}'} "
            f"| {'--' if np.isnan(auroc) else f'{auroc:.3f}'} |"
        )

    ablation = report["components"]["leave_one_out"]
    base = ablation["full"]
    lines += ["", "### Leave-one-out ablation", "",
              "Change in Q's agreement with Dice when one sub-score is removed. "
              "A large drop means the term carries the signal.", "",
              "| removed | rho with Dice | Δrho | AUROC | ΔAUROC |",
              "| --- | --- | --- | --- | --- |",
              f"| _(none — full Q)_ | {base['spearman_with_dice']:+.3f} | — "
              f"| {base['auroc_good']:.3f} | — |"]
    for key, entry in sorted(ablation.items(),
                             key=lambda kv: kv[1]["spearman_with_dice"]):
        if key == "full":
            continue
        lines.append(
            f"| {key.removeprefix('without_')} | {entry['spearman_with_dice']:+.3f} "
            f"| {entry['spearman_with_dice'] - base['spearman_with_dice']:+.3f} "
            f"| {entry['auroc_good']:.3f} "
            f"| {entry['auroc_good'] - base['auroc_good']:+.3f} |"
        )

    lines += ["", "## Alternative weight sets (diagnostic, scored on this same run)", "",
              "| weight set | rho with Dice | AUROC | Q | 5th–95th pct |",
              "| --- | --- | --- | --- | --- |"]
    for name, entry in report["weight_presets"].items():
        lines.append(
            f"| {name} | {entry['spearman_with_dice']:+.3f} | {entry['auroc_good']:.3f} "
            f"| {_fmt(entry['quality'])} | {entry['p05']:.3f}–{entry['p95']:.3f} |"
        )

    lines += ["", "## Within volume bands (size held roughly constant)", "",
              "| band | patients | frames | GT area px | Q | Dice | rho | AUROC |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for band, entry in report["per_volume_band"].items():
        rho, auroc = entry["spearman_quality_dice"], entry["auroc_good"]
        lines.append(
            f"| {band} | {', '.join(entry['patients'])} | {entry['n_frames']} "
            f"| {entry['gt_area_px_median']:.0f} | {_fmt(entry['quality'])} "
            f"| {_fmt(entry['dice'])} "
            f"| {'--' if np.isnan(rho) else f'{rho:+.3f}'} "
            f"| {'--' if np.isnan(auroc) else f'{auroc:.3f}'} |"
        )

    validity = report["validity"]
    lines += ["", "## Shipped validity gate", "",
              f"- `valid_for_control` on **{validity['valid_fraction'] * 100:.1f}%** of frames "
              f"({validity['n_invalid']} rejected)",
              f"- Dice of accepted: {_fmt(validity['dice_valid'])}",
              f"- Dice of rejected: {_fmt(validity['dice_invalid'])}", ""]
    if validity["rejection_reasons"]:
        lines += ["| rejection reason | frames |", "| --- | --- |"]
        lines += [f"| {reason} | {count} |"
                  for reason, count in validity["rejection_reasons"].items()]
    else:
        lines.append("_No frame was rejected._")

    lines += ["", "## Per patient", "",
              "| patient | frames | GT area px | Q | Dice | valid | completeness | contrast | temporal IoU |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for patient, entry in sorted(report["per_patient"].items(),
                                 key=lambda kv: -kv[1]["gt_area_px_median"]):
        comp = entry["components"]
        lines.append(
            f"| {patient} | {entry['n_frames']} | {entry['gt_area_px_median']:.0f} "
            f"| {entry['quality_mean']:.3f} ± {entry['quality_sd']:.3f} "
            f"| {entry['dice_mean']:.3f} | {entry['valid_fraction'] * 100:.0f}% "
            f"| {comp['mask_completeness']:.3f} | {comp['lumen_contrast']:.3f} "
            f"| {comp['temporal_iou']:.3f} |"
        )

    if highlight:
        index = {(f["patient_id"], f["frame_index"]): f for f in frames}
        lines += ["", "## Figure cases", "",
                  "| role | frame | Dice | Q | valid | sub-scores |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for role, patient, frame_index in highlight:
            frame = index.get((patient, frame_index))
            if frame is None:
                lines.append(f"| {role} | {patient} · {frame_index:03d} | _not in this run_ | | | |")
                continue
            parts = ", ".join(f"{n[:14]} {v:.2f}" for n, v in frame["components"].items())
            lines.append(
                f"| {role} | {patient} · frame {frame_index:03d} | {frame['dice']:.3f} "
                f"| **{frame['quality']:.3f}** | {'yes' if frame['valid'] else 'no'} | {parts} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/exp_seed43/best_test"))
    parser.add_argument("--config", default="configs/exp_seed43.yaml",
                        help="Config the run used; its control.quality section defines Q.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/quality_scoring"))
    parser.add_argument("--highlight", nargs="*", default=["Best:P041:1", "Median:P012:59",
                                                           "Worst:P000:70"],
                        help="role:patient:frame triples to tabulate individually.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from rus_perception.utils.config import load_config
    config = QualityConfig.from_dict(load_config(args.config).section("control").get("quality"))

    frames = load_run(args.run_dir)
    report = analyse(frames, config)

    highlight = []
    for item in args.highlight:
        role, patient, frame_index = item.split(":")
        highlight.append((role, patient, int(frame_index)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "quality_vs_accuracy.json").write_text(
        json.dumps({"run": str(args.run_dir), **report}, indent=2))
    text = render_report(report, frames, args.run_dir, highlight)
    (args.output_dir / "quality_vs_accuracy.md").write_text(text)
    print(text)
    logger.info("Wrote %s", args.output_dir / "quality_vs_accuracy.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
