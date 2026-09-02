#!/usr/bin/env python3
"""Measure centroid *accuracy*, which the control-quality score never sees.

Why this exists
---------------
``Q`` contains a ``centroid_stability`` term, so it is easy to read the score as
if it vouched for the centroid the controller servos on. It does not. The term
is ``exp(-jump / centroid_jump_scale)`` where ``jump`` is the distance between
this frame's *predicted* centroid and the previous frame's *predicted* centroid
(flow-warped when a flow field is available) -- see
rus_perception/control/features.py. Ground truth never enters it. A mask that is
wrong in the same place on every frame scores 1.0.

Dice does not close the gap either: it is an overlap measure, and two masks with
the same Dice can put their centroids in different places.

This script computes the quantity that is actually missing -- the distance
between the predicted centroid and the ground-truth centroid, in the normalized
units the controller uses -- and asks whether ``Q``, ``centroid_stability`` or
Dice predicts it.

Example:
    python scripts/centroid_accuracy.py --run-dir runs/exp_seed43/best_test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

from rus_perception.data.io import load_mask, resize_mask
from rus_perception.data.manifest import load_manifest

logger = logging.getLogger(__name__)

#: Centroid error, in normalized image widths, below which the measurement is
#: called good enough to steer on. 0.02 of a 256 px frame is ~5 px; it is
#: reported as a stated convention, not fitted, and the full distribution is
#: given alongside so the choice can be overridden.
GOOD_CENTROID_ERROR = 0.02


def _gt_centroid(args: tuple[str, str, int, int]) -> tuple[str, Optional[tuple[float, float]]]:
    """Normalized ground-truth centroid of one mask, at the evaluation resolution.

    The mask is resized to the network grid before the centroid is taken so the
    number is comparable with the prediction's, which is computed there. The
    normalization matches ``binary_mask_geometry``: ``(centroid + 0.5) / size``.
    """
    frame_id, mask_path, height, width = args
    mask = resize_mask(load_mask(mask_path), (height, width))
    ys, xs = np.nonzero(np.asarray(mask) > 0.5)
    if xs.size == 0:
        return frame_id, None
    return frame_id, ((float(xs.mean()) + 0.5) / width, (float(ys.mean()) + 0.5) / height)


def load_ground_truth_centroids(manifest_path: str, root: str, frame_ids: set[str],
                                size: tuple[int, int], workers: int) -> dict:
    """Ground-truth centroids for the evaluated frames, keyed by ``frame_id``.

    Raises:
        ValueError: If the manifest does not cover every evaluated frame; a
            partial join would quietly drop frames from the statistics.
    """
    manifest = load_manifest(manifest_path, root=root)
    jobs = [
        (record.frame_id, str(manifest.resolve(record.mask_path)), size[0], size[1])
        for record in manifest
        if record.frame_id in frame_ids and record.mask_path
    ]
    covered = {job[0] for job in jobs}
    if covered != frame_ids:
        raise ValueError(
            f"{len(frame_ids - covered)} evaluated frame(s) have no labelled mask in "
            f"{manifest_path}. Refusing to report a centroid error over a partial cohort."
        )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_gt_centroid, jobs, chunksize=64))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation; ``nan`` when there is nothing to rank."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    rank_x, rank_y = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    if rank_x.std() == 0 or rank_y.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """AUROC via the Mann-Whitney statistic, with mid-ranks for ties."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sorted_scores, start = scores[order], 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _describe(values: Sequence[float]) -> Optional[dict]:
    clean = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if clean.size == 0:
        return None
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "median": float(np.median(clean)),
        "iqr": [float(np.percentile(clean, 25)), float(np.percentile(clean, 75))],
        "p95": float(np.percentile(clean, 95)),
        "max": float(clean.max()),
    }


def build_frames(run_dir: Path, ground_truth: dict) -> list[dict]:
    """Join predicted centroids, ground-truth centroids and the quality terms."""
    import csv

    with (run_dir / "frame_metrics.csv").open(newline="") as handle:
        metrics = {row["frame_id"]: row for row in csv.DictReader(handle)}

    frames = []
    with (run_dir / "control_states.jsonl").open() as handle:
        for line in handle:
            state = json.loads(line)
            frame_id = state["frame_id"]
            truth = ground_truth.get(frame_id)
            row = metrics[frame_id]
            width = float(state["image_width"])
            predicted = (state["centroid_x_normalized"], state["centroid_y_normalized"])
            if truth is None or predicted[0] is None:
                error = float("nan")
                error_x = error_y = float("nan")
            else:
                error_x = predicted[0] - truth[0]
                error_y = predicted[1] - truth[1]
                error = math.hypot(error_x, error_y)
            frames.append({
                "frame_id": frame_id,
                "patient_id": state["metadata"]["patient_id"],
                "frame_index": int(row["frame_index"]),
                "dice": float(row["dice"]),
                "gt_area_px": int(row["target_area_px"]),
                "quality": float(state["control_quality_score"]),
                "centroid_stability": state["quality_components"].get(
                    "centroid_stability", float("nan")),
                "centroid_jump": (state["normalized_centroid_jump"]
                                  if state["normalized_centroid_jump"] is not None
                                  else float("nan")),
                "centroid_error": error,
                "centroid_error_px": error * width,
                "centroid_error_x": error_x,
                "centroid_error_y": error_y,
                "valid": bool(state["valid_for_control"]),
            })
    return frames


def analyse(frames: Sequence[dict]) -> dict:
    """Correlate every available quality signal against true centroid error."""
    error = [f["centroid_error"] for f in frames]
    bad = [f["centroid_error"] > GOOD_CENTROID_ERROR for f in frames]

    predictors = {
        "control_quality_score": [f["quality"] for f in frames],
        "centroid_stability": [f["centroid_stability"] for f in frames],
        "dice": [f["dice"] for f in frames],
    }
    # All three are "higher is better", so they are negated to score a detector
    # of *large* error; AUROC then reads the same way as everywhere else.
    discrimination = {
        name: {
            "spearman_with_error": _spearman(values, error),
            "auroc_detect_large_error": _auroc([-v for v in values], bad),
        }
        for name, values in predictors.items()
    }

    by_patient: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        by_patient[frame["patient_id"]].append(frame)

    per_patient = {}
    for patient, rows in sorted(by_patient.items()):
        per_patient[patient] = {
            "n_frames": len(rows),
            "gt_area_px_median": float(np.median([r["gt_area_px"] for r in rows])),
            "centroid_error": _describe([r["centroid_error"] for r in rows]),
            "centroid_error_px_mean": float(np.nanmean([r["centroid_error_px"] for r in rows])),
            "centroid_stability_mean": float(np.nanmean([r["centroid_stability"] for r in rows])),
            "quality_mean": float(np.mean([r["quality"] for r in rows])),
            "dice_mean": float(np.mean([r["dice"] for r in rows])),
            # Systematic offset: a bias survives averaging, jitter cancels. A
            # biased centroid is a standing servo error, not noise the loop
            # filters out, so the two are worth separating.
            "bias_x": float(np.nanmean([r["centroid_error_x"] for r in rows])),
            "bias_y": float(np.nanmean([r["centroid_error_y"] for r in rows])),
            "bias_magnitude": float(math.hypot(
                float(np.nanmean([r["centroid_error_x"] for r in rows])),
                float(np.nanmean([r["centroid_error_y"] for r in rows])))),
        }

    return {
        "n_frames": len(frames),
        "good_centroid_error": GOOD_CENTROID_ERROR,
        "n_large_error": int(sum(bad)),
        "centroid_error": _describe(error),
        "centroid_error_px": _describe([f["centroid_error_px"] for f in frames]),
        "discrimination": discrimination,
        # The claim the term's name invites: does self-consistency imply accuracy?
        "stability_vs_accuracy": {
            "spearman_jump_vs_error": _spearman([f["centroid_jump"] for f in frames], error),
            "stable_frames_fraction": float(np.mean(
                [f["centroid_stability"] >= 0.9 for f in frames if not np.isnan(f["centroid_stability"])])),
            "error_among_stable": _describe([
                f["centroid_error"] for f in frames
                if not np.isnan(f["centroid_stability"]) and f["centroid_stability"] >= 0.9]),
            "error_among_unstable": _describe([
                f["centroid_error"] for f in frames
                if not np.isnan(f["centroid_stability"]) and f["centroid_stability"] < 0.9]),
        },
        "per_patient": per_patient,
    }


def _fmt(stat: Optional[dict], digits: int = 4) -> str:
    if stat is None:
        return "--"
    return (f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f} "
            f"({stat['median']:.{digits}f} [{stat['iqr'][0]:.{digits}f}–{stat['iqr'][1]:.{digits}f}])")


def render_report(report: dict, frames: Sequence[dict], run_dir: Path,
                  highlight: Sequence[tuple[str, str, int]] = ()) -> str:
    """Markdown rendering of :func:`analyse`."""
    stability = report["stability_vs_accuracy"]
    lines = [
        "# Centroid accuracy versus the centroid term in Q",
        "",
        f"- Run: `{run_dir}`; {report['n_frames']} frames",
        "- `centroid_stability` in Q is `exp(-jump / 0.05)` where `jump` is the distance "
        "between this frame's predicted centroid and the previous frame's predicted "
        "centroid. **Ground truth never enters it.**",
        "- Measured here instead: ‖predicted centroid − ground-truth centroid‖, "
        "normalized to image width at 256×256.",
        "",
        f"- Centroid error: {_fmt(report['centroid_error'])} normalized "
        f"= {_fmt(report['centroid_error_px'], 2)} px",
        f"- {report['n_large_error']} frames "
        f"({report['n_large_error'] / report['n_frames'] * 100:.1f}%) exceed "
        f"{report['good_centroid_error']:.3f} ("
        f"{report['good_centroid_error'] * 256:.1f} px)",
        "",
        "## Does anything in the pipeline predict centroid error?",
        "",
        "| signal | rho with centroid error | AUROC (detect error > threshold) |",
        "| --- | --- | --- |",
    ]
    for name, entry in report["discrimination"].items():
        lines.append(f"| `{name}` | {entry['spearman_with_error']:+.3f} "
                     f"| {entry['auroc_detect_large_error']:.3f} |")

    lines += [
        "",
        "## Self-consistency is not accuracy",
        "",
        f"- Spearman rho (centroid *jump* vs centroid *error*): "
        f"**{stability['spearman_jump_vs_error']:+.3f}**",
        f"- {stability['stable_frames_fraction'] * 100:.1f}% of frames score "
        "`centroid_stability` ≥ 0.90",
        f"- Centroid error among those stable frames: {_fmt(stability['error_among_stable'])}",
        f"- Centroid error among unstable frames: {_fmt(stability['error_among_unstable'])}",
        "",
        "## Per patient",
        "",
        "| patient | frames | GT area px | centroid error (norm) | px | bias ‖·‖ | "
        "`centroid_stability` | Q | Dice |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for patient, entry in sorted(report["per_patient"].items(),
                                 key=lambda kv: -kv[1]["gt_area_px_median"]):
        lines.append(
            f"| {patient} | {entry['n_frames']} | {entry['gt_area_px_median']:.0f} "
            f"| {_fmt(entry['centroid_error'])} | {entry['centroid_error_px_mean']:.2f} "
            f"| {entry['bias_magnitude']:.4f} "
            f"| {entry['centroid_stability_mean']:.3f} | {entry['quality_mean']:.3f} "
            f"| {entry['dice_mean']:.3f} |"
        )

    if highlight:
        index = {(f["patient_id"], f["frame_index"]): f for f in frames}
        lines += ["", "## Figure cases", "",
                  "| role | frame | Dice | Q | `centroid_stability` | centroid error (px) |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for role, patient, frame_index in highlight:
            frame = index.get((patient, frame_index))
            if frame is None:
                lines.append(f"| {role} | {patient} · {frame_index:03d} | _not in run_ | | | |")
                continue
            stability_value = frame["centroid_stability"]
            lines.append(
                f"| {role} | {patient} · frame {frame_index:03d} | {frame['dice']:.3f} "
                f"| {frame['quality']:.3f} "
                f"| {'--' if np.isnan(stability_value) else f'{stability_value:.3f}'} "
                f"| **{frame['centroid_error_px']:.2f}** |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/exp_seed43/best_test"))
    parser.add_argument("--config", default="configs/exp_seed43.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/quality_scoring"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--highlight", nargs="*",
                        default=["Best:P041:1", "Median:P012:59", "Worst:P000:70"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from rus_perception.utils.config import load_config
    data = load_config(args.config).section("data")
    size = tuple(int(v) for v in data["image_size"])

    frame_ids = {json.loads(line)["frame_id"]
                 for line in (args.run_dir / "control_states.jsonl").open()}
    ground_truth = load_ground_truth_centroids(
        str(data["manifest"]), str(data["root"]), frame_ids, size, args.workers)

    frames = build_frames(args.run_dir, ground_truth)
    report = analyse(frames)

    highlight = [(role, patient, int(index))
                 for role, patient, index in (item.split(":") for item in args.highlight)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "centroid_accuracy.json").write_text(
        json.dumps({"run": str(args.run_dir), **report}, indent=2))
    text = render_report(report, frames, args.run_dir, highlight)
    (args.output_dir / "centroid_accuracy.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
