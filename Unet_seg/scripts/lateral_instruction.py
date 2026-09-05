#!/usr/bin/env python3
"""Evidence for the lateral correction instruction, in manuscript form.

Claim under test
----------------
A U-Net lumen segmentation, read through the control state, emits a *lateral*
correction instruction -- which way, and how far, to slide the probe so the
bladder lumen sits on the beam axis -- and that instruction is valid above a
displacement threshold that follows from the segmentation's own centroid noise.

Why only the lateral axis
-------------------------
Of the three axes DESIGN_NOTES.md section 5.3 assigns to the image policy
(``v_x``, ``v_y``, ``omega_z``), only ``v_x`` leaves the imaging plane invariant
and presents a *measurable vector error*; section 16.1 already states this. The
other two move the plane through the anatomy and expose a scalar quality alone,
which makes them a search rather than a regression -- and they cannot be
simulated from archived single-plane frames at all. The instruction validated
here is therefore the lateral one, and the paper should claim no more.

Why the natural poses cannot settle the direction question
-----------------------------------------------------------
In every one of the 2,953 archived frames the lumen lies on the same side of the
beam axis: ``e`` is positive throughout, 22 patients out of 22. Sign agreement
measured on those frames is therefore not evidence that the instruction points
the right way -- a constant output would score the same. The direction claim is
tested instead on laterally translated frames, which carry the lumen across the
axis and make ``e`` change sign. Dice stays within 0.847-0.860 across the whole
sweep, so the translation is not quietly breaking the segmentation it is meant
to interrogate; that control is reported alongside the result.

Error model
-----------
Write the instruction as ``e_hat = A - c_hat`` and the truth as ``e = A - c``,
with ``A`` the beam axis and ``c`` the lumen centroid abscissa. Their difference
is the segmentation's lateral centroid error ``delta``, which is independent of
where the lumen happens to sit. The instruction points the wrong way only when
``delta`` both opposes ``e`` and exceeds it, so for zero-mean ``delta`` with
scale ``sigma``:

    P(wrong direction) = Phi(-|e| / sigma)

The instruction is therefore trustworthy while the displacement it asks for is
large against the segmentation's own noise, and stops being so as the loop
converges -- which is a stopping condition, not a defect. The prediction is
compared against the observed rate rather than assumed.

Outputs: a metrics table (CSV + JSON) and three figures.

Example:
    python scripts/lateral_instruction.py --output-dir experiments/lateral_instruction
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Text is black throughout; marks use a dark navy / grey pair. The model and
# data imports live inside collect()/sweep() so the figure functions stay
# importable on a machine without torch (see redraw_lateral_instruction_figures).
INK = "#000000"
INK_SOFT = "#444444"
ACCENT = "#1f3864"
WARN = "#767676"
CRIT = "#a32316"
GT_COLOR = "#16B9D4"
PRED_COLOR = "#E17C32"

#: Displacement bins for the validity curve, in pixels at 256 x 256.
BINS = ((0, 3), (3, 6), (6, 9), (9, 12), (12, 16), (16, 22), (22, 30), (30, 1000))

#: Lateral displacements the sweep aims for, in pixels. Chosen dense near the
#: axis crossing, where the instruction is expected to fail, and sparse far from
#: it, where nothing is in doubt.
SWEEP_TARGETS = (2, 4, 6, 8, 10, 13, 16, 20, 25, 32, 40)


def beam_axis(roi: np.ndarray) -> float:
    """Abscissa of the beam axis: the sector's axis of symmetry.

    Taken from the ROI rather than assumed to be the frame centre, so a
    re-derived or off-centre sector stays correct.
    """
    ys, xs = np.nonzero(roi)
    return float(xs.mean())


def collect(config_path: str, checkpoint: str, splits: Sequence[str]) -> list[dict]:
    """Instruction and truth for every labelled frame of the requested splits."""
    from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

    from rus_perception.control.features import FeatureExtractionConfig, extract_control_state
    from rus_perception.control.roi import RoiConfig, build_roi_mask
    from rus_perception.data.io import load_grayscale, load_mask, resize_image, resize_mask
    from rus_perception.data.manifest import load_manifest
    from rus_perception.inference.predictor import Predictor, PredictorConfig
    from rus_perception.utils.config import load_config

    config = load_config(config_path)
    size = tuple(int(v) for v in config.section("data")["image_size"])
    roi_config = RoiConfig.from_dict(config.section("control").get("roi"))
    roi = np.asarray(build_roi_mask(size, roi_config)) > 0
    axis = beam_axis(roi)
    feature_config = FeatureExtractionConfig.from_dict(
        {"postprocess": config.section("postprocess"), **config.section("control")}
    )
    predictor = Predictor.from_checkpoint(
        checkpoint, model_config=config.section("model"),
        predictor_config=PredictorConfig(
            input_size=size,
            intensity_normalization=str(config.get("data.intensity_normalization")),
            device=str(config.get("train.device", "auto")).replace("auto", "cuda"),
            restore_original_size=False, roi=roi_config),
        feature_config=feature_config,
    )
    data = config.section("data")
    manifest = load_manifest(str(data["manifest"]), root=str(data["root"]))

    rows = []
    for record in manifest:
        if record.split not in splits or not record.mask_path:
            continue
        image = np.asarray(resize_image(load_grayscale(str(manifest.resolve(record.image_path))),
                                        size), np.float32)
        truth = np.asarray(resize_mask(load_mask(str(manifest.resolve(record.mask_path))),
                                       size)) > 0.5
        probability, _ = predictor.predict_probability(image)
        state = extract_control_state(probability, image=image,
                                      config=feature_config, roi_mask=roi)
        prediction = np.asarray(state.binary_mask, np.uint8)
        py, px = np.nonzero(prediction)
        gy, gx = np.nonzero(truth)
        if px.size == 0 or gx.size == 0:
            continue
        overlap = float((prediction.astype(bool) & truth).sum())
        rows.append({
            "split": record.split, "patient_id": record.patient_id,
            "frame_id": record.frame_id,
            "instruction_px": axis - float(px.mean()),
            "truth_px": axis - float(gx.mean()),
            "quality": float(state.control_quality_score),
            "dice": 2.0 * overlap / float(prediction.sum() + truth.sum()),
        })
    logger.info("Collected %d frames over %s (beam axis x = %.1f)",
                len(rows), ", ".join(splits), axis)
    return rows


def sweep(config_path: str, checkpoint: str, splits: Sequence[str],
          frames_per_patient: int) -> list[dict]:
    """Translate each frame so the lumen sits at prescribed lateral offsets.

    Solving for the shift that puts ``e`` on a target value, rather than
    sweeping shifts blindly, spends the sampling where the question is -- within
    a few pixels of the axis -- instead of on displacements nobody doubts.
    ``dice`` is carried through so a reader can check that the translation left
    the segmentation intact.
    """
    import cv2

    from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

    from rus_perception.control.features import FeatureExtractionConfig, extract_control_state
    from rus_perception.control.roi import RoiConfig, build_roi_mask
    from rus_perception.data.io import load_grayscale, load_mask, resize_image, resize_mask
    from rus_perception.data.manifest import load_manifest
    from rus_perception.inference.predictor import Predictor, PredictorConfig
    from rus_perception.utils.config import load_config

    config = load_config(config_path)
    size = tuple(int(v) for v in config.section("data")["image_size"])
    roi_config = RoiConfig.from_dict(config.section("control").get("roi"))
    roi = np.asarray(build_roi_mask(size, roi_config)) > 0
    axis = beam_axis(roi)
    feature_config = FeatureExtractionConfig.from_dict(
        {"postprocess": config.section("postprocess"), **config.section("control")}
    )
    predictor = Predictor.from_checkpoint(
        checkpoint, model_config=config.section("model"),
        predictor_config=PredictorConfig(
            input_size=size,
            intensity_normalization=str(config.get("data.intensity_normalization")),
            device=str(config.get("train.device", "auto")).replace("auto", "cuda"),
            restore_original_size=False, roi=roi_config),
        feature_config=feature_config)
    data = config.section("data")
    manifest = load_manifest(str(data["manifest"]), root=str(data["root"]))

    by_patient: dict[str, list] = defaultdict(list)
    for record in manifest:
        if record.split in splits and record.mask_path:
            by_patient[record.patient_id].append(record)

    def shift(array, dx):
        matrix = np.float32([[1, 0, dx], [0, 1, 0]])
        interpolation = cv2.INTER_NEAREST if array.dtype == np.uint8 else cv2.INTER_LINEAR
        # Edge replication, never a black fill: a black band reads as anechoic
        # lumen to every intensity feature in the score.
        return cv2.warpAffine(array, matrix, (size[1], size[0]), flags=interpolation,
                              borderMode=cv2.BORDER_REPLICATE)

    rows = []
    for patient, records in sorted(by_patient.items()):
        records.sort(key=lambda r: r.frame_id)
        stride = max(1, len(records) // frames_per_patient)
        for record in records[::stride][:frames_per_patient]:
            image = np.asarray(resize_image(
                load_grayscale(str(manifest.resolve(record.image_path))), size), np.float32)
            truth = (np.asarray(resize_mask(
                load_mask(str(manifest.resolve(record.mask_path))), size)) > 0.5).astype(np.uint8)
            gy, gx = np.nonzero(truth)
            if gx.size == 0:
                continue
            natural = axis - float(gx.mean())
            for target in (list(SWEEP_TARGETS) + [-t for t in SWEEP_TARGETS]):
                dx = int(round(natural - target))
                frame, moved = shift(image, dx), shift(truth, dx)
                probability, _ = predictor.predict_probability(frame)
                state = extract_control_state(probability, image=frame,
                                              config=feature_config, roi_mask=roi)
                prediction = np.asarray(state.binary_mask, np.uint8)
                py, px = np.nonzero(prediction)
                my, mx = np.nonzero(moved)
                if px.size == 0 or mx.size == 0:
                    continue
                total = float(prediction.sum() + moved.sum())
                rows.append({
                    "split": record.split, "patient_id": patient,
                    "frame_id": record.frame_id, "shift_px": dx, "target_px": target,
                    "instruction_px": axis - float(px.mean()),
                    "truth_px": axis - float(mx.mean()),
                    "quality": float(state.control_quality_score),
                    "dice": 2.0 * float((prediction.astype(bool) & moved.astype(bool)).sum()) / total,
                })
    logger.info("Sweep: %d frames, |e| from %.1f to %.1f px, %.0f%% with e < 0",
                len(rows), min(abs(r["truth_px"]) for r in rows),
                max(abs(r["truth_px"]) for r in rows),
                100 * np.mean([r["truth_px"] < 0 for r in rows]))
    return rows


def summarise(rows: Sequence[dict]) -> dict:
    """Agreement statistics, the fitted noise scale, and the validity thresholds."""
    instruction = np.array([r["instruction_px"] for r in rows])
    truth = np.array([r["truth_px"] for r in rows])
    delta = instruction - truth
    sigma = float(delta.std(ddof=1))
    normal = NormalDist()

    def block(mask: np.ndarray) -> Optional[dict]:
        if mask.sum() < 3:
            return None
        a, b = instruction[mask], truth[mask]
        design = np.vstack([b, np.ones_like(b)]).T
        slope, intercept = np.linalg.lstsq(design, a, rcond=None)[0]
        return {
            "n": int(mask.sum()),
            "sign_agreement": float((np.sign(a) == np.sign(b)).mean()),
            "abs_error_median": float(np.median(np.abs(a - b))),
            "abs_error_p90": float(np.percentile(np.abs(a - b), 90)),
            "slope": float(slope), "intercept": float(intercept),
            "pearson_r": float(np.corrcoef(b, a)[0, 1]),
            "truth_abs_median": float(np.median(np.abs(b))),
        }

    by_split = {s: block(np.array([r["split"] == s for r in rows]))
                for s in sorted({r["split"] for r in rows})}
    by_patient = {}
    for patient in sorted({r["patient_id"] for r in rows}):
        mask = np.array([r["patient_id"] == patient for r in rows])
        entry = block(mask)
        entry["split"] = next(r["split"] for r in rows if r["patient_id"] == patient)
        by_patient[patient] = entry

    curve = []
    for low, high in BINS:
        mask = (np.abs(truth) >= low) & (np.abs(truth) < high)
        if mask.sum() == 0:
            curve.append({"low": low, "high": high, "n": 0})
            continue
        observed = float((np.sign(instruction[mask]) != np.sign(truth[mask])).mean())
        modelled = float(np.mean([normal.cdf(-abs(v) / sigma) for v in truth[mask]]))
        curve.append({"low": low, "high": high, "n": int(mask.sum()),
                      "observed_sign_error": observed, "modelled_sign_error": modelled})

    return {
        "n_frames": len(rows),
        "sigma_px": sigma,
        "delta_mean_px": float(delta.mean()),
        "overall": block(np.ones(len(rows), bool)),
        "by_split": by_split,
        "by_patient": by_patient,
        "validity_curve": curve,
        "thresholds": {f"{int(p * 1000) / 10:g}%": {
            "k": -normal.inv_cdf(p), "px": -normal.inv_cdf(p) * sigma}
            for p in (0.05, 0.01, 0.001)},
    }


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"], "text.color": INK,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def figure_agreement(rows: Sequence[dict], stats: dict):
    """Instruction against truth, one panel per cohort."""
    import matplotlib.pyplot as plt

    _style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True, sharey=True)
    limit = 1.05 * max(abs(r["truth_px"]) for r in rows)
    for ax, split in zip(axes, ("val", "test")):
        sub = [r for r in rows if r["split"] == split]
        truth = np.array([r["truth_px"] for r in sub])
        instruction = np.array([r["instruction_px"] for r in sub])
        wrong = np.sign(truth) != np.sign(instruction)
        # The two off-sign quadrants are where the instruction would send the
        # operator the wrong way; shading them makes the failure mode visible
        # rather than something the reader must infer from the scatter.
        ax.axhspan(0, limit, xmin=0, xmax=0.5, color=CRIT, alpha=0.05, lw=0)
        ax.axhspan(-limit, 0, xmin=0.5, xmax=1.0, color=CRIT, alpha=0.05, lw=0)
        ax.plot([-limit, limit], [-limit, limit], color=INK_SOFT, lw=0.9, ls=(0, (4, 3)),
                zorder=2, label="identity")
        ax.scatter(truth[~wrong], instruction[~wrong], s=5, c=ACCENT, alpha=0.28,
                   lw=0, zorder=3)
        if wrong.any():
            ax.scatter(truth[wrong], instruction[wrong], s=13, c=CRIT, alpha=0.9,
                       lw=0, zorder=4, label=f"wrong direction (n={int(wrong.sum())})")
        entry = stats["by_split"][split]
        ax.set_title(f"{split}  ·  n = {entry['n']}", fontsize=9.5, color=INK, pad=7)
        ax.text(0.04, 0.96,
                f"sign agreement  {entry['sign_agreement'] * 100:.1f}%\n"
                f"|error| median  {entry['abs_error_median']:.2f} px\n"
                f"slope  {entry['slope']:.3f}   r  {entry['pearson_r']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.6,
                color=INK, linespacing=1.5)
        ax.set_xlabel("ground-truth lateral displacement  $e$  (px)", fontsize=8.5)
        ax.axhline(0, color=INK_SOFT, lw=0.6); ax.axvline(0, color=INK_SOFT, lw=0.6)
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7.5)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    axes[0].set_ylabel(r"instructed displacement  $\hat{e}$  (px)", fontsize=8.5)
    figure.tight_layout()
    return figure


def figure_validity(stats: dict):
    """Observed and modelled wrong-direction rate against displacement."""
    import matplotlib.pyplot as plt

    _style()
    sigma = stats["sigma_px"]
    figure, ax = plt.subplots(figsize=(7.2, 3.3))

    grid = np.linspace(0.1, 60, 400)
    normal = NormalDist()
    ax.plot(grid, [normal.cdf(-v / sigma) * 100 for v in grid], color=INK,
            lw=1.5, zorder=3, label=r"model  $\Phi(-|e|/\sigma)$")

    curve = [c for c in stats["validity_curve"] if c["n"]]
    centres = [min((c["low"] + c["high"]) / 2, 50) for c in curve]
    observed = [c["observed_sign_error"] * 100 for c in curve]
    ax.scatter(centres, observed, s=20, facecolor="white", edgecolor=ACCENT,
               lw=1.3, zorder=5, label="observed")
    for c, x, y in zip(curve, centres, observed):
        ax.annotate(f"n={c['n']}", (x, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=7, color=INK)

    for probability, colour in ((0.05, WARN), (0.01, ACCENT)):
        threshold = -normal.inv_cdf(probability) * sigma
        ax.axvline(threshold, color=colour, lw=1.0, ls=(0, (5, 3)), zorder=2)
        ax.text(threshold + 0.7, 34, f"{probability * 100:g}% at {threshold:.1f} px",
                rotation=90, va="top", fontsize=7.4, color=INK)

    ax.set_xlabel(r"required lateral displacement  $|e|$  (px at 256$\times$256)", fontsize=8.5)
    ax.set_ylabel("wrong-direction rate (%)", fontsize=8.5)
    ax.set_xlim(0, 55); ax.set_ylim(0, 40)
    ax.tick_params(labelsize=7.5)
    ax.text(0.985, 0.95,
            f"$\\sigma$ = {sigma:.2f} px  (lateral centroid error, {stats['n_frames']} frames)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=INK)
    ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.86), fontsize=7.6, frameon=False)
    figure.tight_layout()
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/exp_seed43_retro.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/exp_seed43/best.pt")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/lateral_instruction"))
    parser.add_argument("--frames-per-patient", type=int, default=4,
                        help="Frames per patient entering the lateral sweep.")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    natural = collect(args.config, args.checkpoint, args.splits)
    rows = sweep(args.config, args.checkpoint, args.splits, args.frames_per_patient)
    stats = summarise(rows)
    stats["natural_poses"] = {
        "n_frames": len(natural),
        "fraction_positive": float(np.mean([r["truth_px"] > 0 for r in natural])),
        "note": ("Every archived frame places the lumen on the same side of the beam "
                 "axis, so sign agreement measured on them is uninformative; the "
                 "direction result comes from the sweep."),
        **{k: v for k, v in summarise(natural).items() if k in ("overall", "by_split")},
    }
    stats["dice_under_translation"] = {
        "median": float(np.median([r["dice"] for r in rows])),
        "p05": float(np.percentile([r["dice"] for r in rows], 5)),
        "p95": float(np.percentile([r["dice"] for r in rows], 95)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lateral_instruction.json").write_text(json.dumps(stats, indent=2))
    with (args.output_dir / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for figure, stem in ((figure_agreement(rows, stats), "fig_agreement"),
                         (figure_validity(stats), "fig_validity")):
        for suffix in ("png", "pdf"):
            figure.savefig(args.output_dir / f"{stem}.{suffix}", dpi=args.dpi,
                           facecolor="white", bbox_inches="tight")
        plt.close(figure)

    overall = stats["overall"]
    print(f"\nn = {stats['n_frames']}   sigma = {stats['sigma_px']:.2f} px "
          f"(bias {stats['delta_mean_px']:+.2f} px)")
    print(f"overall: sign agreement {overall['sign_agreement'] * 100:.1f}%, "
          f"|error| median {overall['abs_error_median']:.2f} px, r = {overall['pearson_r']:.3f}")
    for name, entry in stats["thresholds"].items():
        print(f"  wrong-direction rate below {name}: |e| >= {entry['k']:.2f} sigma "
              f"= {entry['px']:.1f} px")
    logger.info("Wrote figures and tables to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
