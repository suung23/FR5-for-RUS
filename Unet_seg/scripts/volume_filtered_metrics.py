#!/usr/bin/env python3
"""Re-summarise a segmentation evaluation restricted to distended bladders.

Motivation
----------
Figure 2 of the manuscript shows best / median / worst cases side by side. The
best case (P041, frame 001) is a broadly distended bladder; the worst (P000,
frame 070) is a near-collapsed one. The deployment target is HoLEP, where
continuous irrigation keeps the bladder hydro-extended for the whole procedure,
so the frames that matter are the ones that look like the *top row*. This script
answers the obvious follow-up question: what do the numbers look like if only
those frames are scored?

Reference volume
----------------
The threshold is expressed as a fraction of one reference frame's ground-truth
foreground ratio (default: the frame shown in the figure's top row). Working in
a ratio rather than pixels keeps the criterion independent of the native frame
size; ``PX_AT_256`` converts back to the pixel counts the evaluation report and
scripts/filter_manifest_by_area.py both speak in.

Frame-wise and patient-wise filtering are both reported. On PFUS the bladder
area is effectively a patient property -- a static exam does not change bladder
volume across its own sequence -- so the two agree almost everywhere, and where
they disagree the patient-wise number is the honest one (see the module
docstring of scripts/filter_manifest_by_area.py). Reporting both makes that
visible instead of assuming it.

Example:
    python scripts/volume_filtered_metrics.py \
        --frame-metrics runs/exp_seed43/best_test/frame_metrics.csv \
        --output-dir experiments/volume_filter
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Working resolution of the evaluation protocol. Area columns in
#: frame_metrics.csv are pixel counts on this grid.
PX_AT_256 = 256 * 256

#: Metrics summarised for every stratum, and the direction that is "better".
METRICS = [
    ("dice", "higher"),
    ("iou", "higher"),
    ("precision", "higher"),
    ("recall", "higher"),
    ("hd95", "lower"),
]

#: Threshold sweep, as a fraction of the reference frame's bladder area. A
#: single threshold would hide how quickly the cohort shrinks; the sweep makes
#: the trade-off between "close to the reference volume" and "enough patients
#: left to say anything" explicit.
DEFAULT_FRACTIONS = (1.00, 0.75, 0.50, 0.33, 0.25, 0.00)


def load_frames(path: Path) -> list[dict]:
    """Read frame_metrics.csv, typing the columns this analysis uses."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no rows.")
    missing = {"patient_id", "frame_index", "target_area_px"} - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

    frames = []
    for row in rows:
        frame = {
            "patient_id": row["patient_id"],
            "frame_index": int(row["frame_index"]),
            "target_area_px": int(row["target_area_px"]),
            "predicted_area_px": int(row["predicted_area_px"]),
        }
        frame["area_ratio"] = frame["target_area_px"] / PX_AT_256
        for name, _ in METRICS:
            value = row.get(name, "")
            frame[name] = float(value) if value not in ("", "nan", None) else float("nan")
        frames.append(frame)
    return frames


def reference_area(frames: Sequence[dict], patient: str, index: Optional[int]) -> float:
    """Ground-truth area ratio of the reference case.

    Args:
        frames: All evaluated frames.
        patient: Reference patient id.
        index: Reference frame index, or ``None`` to use the patient's median
            frame -- the patient-level statistic used elsewhere in the project.

    Raises:
        KeyError: If the reference case is not in the evaluation.
    """
    if index is None:
        ratios = [f["area_ratio"] for f in frames if f["patient_id"] == patient]
        if not ratios:
            raise KeyError(f"Reference patient {patient} is not in this evaluation.")
        return float(np.median(ratios))
    for frame in frames:
        if frame["patient_id"] == patient and frame["frame_index"] == index:
            return frame["area_ratio"]
    raise KeyError(f"Reference frame {patient}/{index:03d} is not in this evaluation.")


def _describe(values: Sequence[float]) -> Optional[dict]:
    """mean / SD / median / IQR of one metric, or ``None`` when empty."""
    clean = [v for v in values if not np.isnan(v)]
    if not clean:
        return None
    array = np.asarray(clean, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "iqr": [float(np.percentile(array, 25)), float(np.percentile(array, 75))],
    }


def summarise(frames: Sequence[dict]) -> dict:
    """Frame-level and patient-level summary of one stratum.

    Both levels are reported because they answer different questions: the
    frame level is what a per-frame table quotes, while the patient level
    weights every subject equally and is what a cohort claim rests on. With a
    cohort this small the two can differ by more than the CI width.
    """
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        by_patient[frame["patient_id"]].append(frame)

    summary = {
        "n_frames": len(frames),
        "n_patients": len(by_patient),
        "patients": sorted(by_patient),
        "gt_area_px": _describe([f["target_area_px"] for f in frames]),
        "frame_level": {},
        "patient_level": {},
        "per_patient": {},
    }
    if not frames:
        return summary

    for name, _ in METRICS:
        summary["frame_level"][name] = _describe([f[name] for f in frames])
        patient_means = [
            float(np.nanmean([f[name] for f in rows])) for rows in by_patient.values()
        ]
        summary["patient_level"][name] = _describe(patient_means)

    for patient, rows in sorted(by_patient.items()):
        summary["per_patient"][patient] = {
            "n_frames": len(rows),
            "gt_area_px_median": float(np.median([r["target_area_px"] for r in rows])),
            "gt_area_ratio_median": float(np.median([r["area_ratio"] for r in rows])),
            **{name: float(np.nanmean([r[name] for r in rows])) for name, _ in METRICS},
        }
    return summary


def build_strata(frames: Sequence[dict], reference: float,
                 fractions: Sequence[float]) -> list[dict]:
    """Summarise the cohort at each threshold, filtered frame- and patient-wise."""
    by_patient: dict[str, list[float]] = defaultdict(list)
    for frame in frames:
        by_patient[frame["patient_id"]].append(frame["area_ratio"])
    patient_median = {p: float(np.median(v)) for p, v in by_patient.items()}

    strata = []
    for fraction in fractions:
        cut = reference * fraction
        kept_patients = {p for p, m in patient_median.items() if m >= cut}
        strata.append({
            "fraction_of_reference": fraction,
            "min_area_ratio": cut,
            "min_area_px_at_256": cut * PX_AT_256,
            "frame_wise": summarise([f for f in frames if f["area_ratio"] >= cut]),
            "patient_wise": summarise([f for f in frames if f["patient_id"] in kept_patients]),
        })
    return strata


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, without pulling in SciPy for one number."""
    if len(x) < 3:
        return float("nan")
    rank_x = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    rank_y = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def volume_trend(frames: Sequence[dict]) -> dict:
    """How accuracy tracks bladder volume across the whole cohort.

    The threshold sweep answers "how good is the model on distended bladders";
    this answers the question behind it -- whether bladder volume explains the
    spread in the first place. It is reported because on this dataset the
    reference case is an outlier in size, so a hard threshold selects almost
    nothing and the trend carries the information the filter cannot.
    """
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        by_patient[frame["patient_id"]].append(frame)

    patient_area = {p: float(np.median([f["target_area_px"] for f in rows]))
                    for p, rows in by_patient.items()}
    patient_dice = {p: float(np.nanmean([f["dice"] for f in rows]))
                    for p, rows in by_patient.items()}
    order = sorted(patient_area, key=patient_area.get)

    # Patient tertiles: every subject's frames stay together, so a tertile is a
    # cohort of patients, not a resampling of frames across them.
    tertiles = []
    for name, chunk in zip(("small", "medium", "large"),
                           np.array_split(np.asarray(order), 3)):
        members = list(chunk)
        rows = [f for p in members for f in by_patient[p]]
        tertiles.append({
            "band": name,
            "patients": members,
            "gt_area_px_range": [min(patient_area[p] for p in members),
                                 max(patient_area[p] for p in members)],
            **{k: v for k, v in summarise(rows).items()
               if k in ("n_frames", "n_patients", "frame_level", "patient_level")},
        })

    return {
        "spearman_frame_level": _spearman([f["target_area_px"] for f in frames],
                                          [f["dice"] for f in frames]),
        "spearman_patient_level": _spearman([patient_area[p] for p in order],
                                            [patient_dice[p] for p in order]),
        "tertiles": tertiles,
    }


def _fmt(stat: Optional[dict], digits: int = 3) -> str:
    """One metric cell: ``mean ± SD (median [IQR])``."""
    if stat is None:
        return "--"
    return (f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f} "
            f"({stat['median']:.{digits}f} [{stat['iqr'][0]:.{digits}f}–{stat['iqr'][1]:.{digits}f}])")


def render_report(strata: Sequence[dict], reference: float, reference_label: str,
                  source: Path, level: str = "patient_wise",
                  kwargs_trend: Optional[dict] = None) -> str:
    """Markdown report of the threshold sweep at one filtering level."""
    lines = [
        "# Volume-filtered segmentation results",
        "",
        f"- Source: `{source}`",
        f"- Reference case: **{reference_label}** — ground-truth bladder area "
        f"{reference * PX_AT_256:.0f} px at 256×256 ({reference * 100:.2f}% of the frame)",
        f"- Filtering level: **{level.replace('_', '-')}** "
        "(bladder area is a patient property on PFUS, so the patient-wise cut is the primary one)",
        "",
        "## Cohort size versus threshold",
        "",
        "| ≥ fraction of reference | min area (px @256) | frames | patients | patients kept |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stratum in strata:
        summary = stratum[level]
        kept = ", ".join(summary["patients"]) or "—"
        lines.append(
            f"| {stratum['fraction_of_reference'] * 100:.0f}% "
            f"| {stratum['min_area_px_at_256']:.0f} "
            f"| {summary['n_frames']} | {summary['n_patients']} | {kept} |"
        )

    lines += ["", "## Metrics per stratum", "",
              "Cells are `mean ± SD (median [IQR])`.", ""]
    for stratum in strata:
        summary = stratum[level]
        header = (f"### ≥ {stratum['fraction_of_reference'] * 100:.0f}% of reference volume "
                  f"({summary['n_frames']} frames, {summary['n_patients']} patients)")
        lines += [header, ""]
        if summary["n_frames"] == 0:
            lines += ["_No frame meets this threshold._", ""]
            continue
        lines += ["| metric | frame level | patient level |", "| --- | --- | --- |"]
        for name, _ in METRICS:
            digits = 2 if name == "hd95" else 3
            lines.append(f"| {name} | {_fmt(summary['frame_level'][name], digits)} "
                         f"| {_fmt(summary['patient_level'][name], digits)} |")
        lines.append("")

    trend = kwargs_trend
    if trend is not None:
        lines += [
            "## Accuracy versus bladder volume (whole cohort)",
            "",
            f"- Spearman rho (frame level, GT area vs Dice): **{trend['spearman_frame_level']:.3f}** "
            f"over {len(strata[-1][level]['patients'])} patients' frames",
            f"- Spearman rho (patient level, median GT area vs mean Dice): "
            f"**{trend['spearman_patient_level']:.3f}**",
            "",
            "| volume band | patients | GT area px (median range) | frames | dice (frame level) | hd95 (frame level) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for band in trend["tertiles"]:
            lines.append(
                f"| {band['band']} | {', '.join(band['patients'])} "
                f"| {band['gt_area_px_range'][0]:.0f}–{band['gt_area_px_range'][1]:.0f} "
                f"| {band['n_frames']} | {_fmt(band['frame_level']['dice'])} "
                f"| {_fmt(band['frame_level']['hd95'], 2)} |"
            )
        lines.append("")

    lines += ["## Per-patient means (unfiltered cohort)", "",
              "| patient | frames | GT area px (median) | % of reference | dice | iou | hd95 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    baseline = strata[-1][level]["per_patient"]
    for patient, row in sorted(baseline.items(),
                               key=lambda kv: -kv[1]["gt_area_px_median"]):
        lines.append(
            f"| {patient} | {row['n_frames']} | {row['gt_area_px_median']:.0f} "
            f"| {row['gt_area_ratio_median'] / reference * 100:.0f}% "
            f"| {row['dice']:.3f} | {row['iou']:.3f} | {row['hd95']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame-metrics", type=Path,
                        default=Path("runs/exp_seed43/best_test/frame_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/volume_filter"))
    parser.add_argument("--reference-patient", default="P041")
    parser.add_argument("--reference-frame", type=int, default=1,
                        help="Reference frame index; -1 uses the patient's median frame.")
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    frames = load_frames(args.frame_metrics)
    index = None if args.reference_frame < 0 else args.reference_frame
    reference = reference_area(frames, args.reference_patient, index)
    label = (f"{args.reference_patient} · frame {index:03d}" if index is not None
             else f"{args.reference_patient} · median frame")

    fractions = sorted(set(args.fractions), reverse=True)
    strata = build_strata(frames, reference, fractions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "source": str(args.frame_metrics),
        "reference": {"label": label, "area_ratio": reference,
                      "area_px_at_256": reference * PX_AT_256},
        "n_frames_total": len(frames),
        "strata": strata,
    }
    trend = volume_trend(frames)
    payload["volume_trend"] = trend
    json_path = args.output_dir / "volume_filtered_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2))

    report = render_report(strata, reference, label, args.frame_metrics,
                           kwargs_trend=trend)
    md_path = args.output_dir / "volume_filtered_metrics.md"
    md_path.write_text(report)

    print(report)
    logger.info("Wrote %s and %s", json_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
