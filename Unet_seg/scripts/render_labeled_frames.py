#!/usr/bin/env python3
"""Draw the segmentation on the frames the quality audit labelled.

The audit reports that ``Q`` admits frames the segmentation got wrong. That is a
claim about pictures, and it should be checked on the pictures. This script
takes rows from ``frame_labels.csv``, re-runs the checkpoint on exactly those
frames, and renders the prediction against the ground truth with the labels
printed beside it.

Colours follow scripts/make_paper_figures.py so a panel here and a panel in the
manuscript figure mean the same thing.

Selection is by gate outcome and metric, so the default -- the ten worst-Dice
frames the gate admitted -- is the evidence for the audit's headline rather than
a hand-picked set.

Example:
    python scripts/render_labeled_frames.py --outcome false_accept --sort dice --limit 10
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

logger = logging.getLogger(__name__)

GT_COLOR = "#16B9D4"
PRED_COLOR = "#E17C32"
TP_COLOR = "#2D6F9F"
FN_COLOR = "#4FA65B"
FP_COLOR = "#D34A4A"
OVERLAY_ALPHA = 0.40

#: Outcome labels, and how each should read in a caption.
#: Families tried, in order, for the Korean captions. matplotlib's bundled
#: DejaVu Sans has no Hangul and drops those glyphs with only a warning, so a
#: CJK family is selected explicitly and the fallback is checked, not assumed.
KOREAN_FAMILIES = ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic",
                   "Malgun Gothic", "UnDotum")

OUTCOME_TEXT = {
    "false_accept": "FALSE ACCEPT — admitted by Q, segmentation poor",
    "false_reject": "FALSE REJECT — rejected by Q, segmentation good",
    "true_accept": "TRUE ACCEPT — admitted by Q, segmentation good",
    "true_reject": "TRUE REJECT — rejected by Q, segmentation poor",
}


def use_korean_font() -> str:
    """Point matplotlib at a Hangul-capable family; warn loudly if there is none.

    Returns:
        The family that was selected, or ``""`` when the captions will render
        with missing glyphs.
    """
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for family in KOREAN_FAMILIES:
        if family in available:
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return family
    logger.warning("No Hangul font found (tried %s); Korean captions will render "
                   "as blank boxes.", ", ".join(KOREAN_FAMILIES))
    return ""


def select(labels_csv: Path, outcome: Optional[str], sort_key: str, limit: int,
           descending: bool, patients: Optional[Sequence[str]]) -> list[dict]:
    """Rows from the audit's label table, filtered and ordered."""
    with labels_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if outcome:
        rows = [r for r in rows if r["q_decision"] == outcome]
    if patients:
        rows = [r for r in rows if r["patient_id"] in set(patients)]
    if not rows:
        raise SystemExit("No row matches that selection; widen --outcome or --patients.")
    rows.sort(key=lambda r: float(r[sort_key]), reverse=descending)
    return rows[:limit]


def build_predictor(config_path: str, checkpoint: str):
    """The predictor for the run whose labels are being illustrated."""
    from rus_perception.utils.config import load_config
    from rus_perception.control.features import FeatureExtractionConfig
    from rus_perception.control.roi import RoiConfig
    from rus_perception.inference.predictor import Predictor, PredictorConfig

    config = load_config(config_path)
    size = tuple(int(v) for v in config.section("data")["image_size"])
    predictor = Predictor.from_checkpoint(
        checkpoint,
        model_config=config.section("model"),
        predictor_config=PredictorConfig(
            input_size=size,
            intensity_normalization=str(config.get("data.intensity_normalization")),
            device=str(config.get("train.device", "auto")).replace("auto", "cuda"),
            restore_original_size=False,   # panels are drawn on the evaluation grid
            roi=RoiConfig.from_dict(config.section("control").get("roi")),
        ),
        feature_config=FeatureExtractionConfig.from_dict(
            {"postprocess": config.section("postprocess"), **config.section("control")}
        ),
    )
    return predictor, config, size


def load_frame(manifest, frame_id: str, size: tuple[int, int]):
    """Image and ground-truth mask for one frame, on the evaluation grid."""
    from rus_perception.data.io import load_grayscale, load_mask, resize_image, resize_mask

    record = next(r for r in manifest if r.frame_id == frame_id)
    image = resize_image(load_grayscale(str(manifest.resolve(record.image_path))), size)
    truth = resize_mask(load_mask(str(manifest.resolve(record.mask_path))), size)
    return np.asarray(image, np.float32), (np.asarray(truth) > 0.5).astype(np.uint8)


def _contours(ax, mask: np.ndarray, color: str, linewidth: float,
              dashes: Optional[tuple] = None) -> None:
    import cv2

    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return
    found, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for contour in found:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        closed = np.vstack([points, points[:1]])
        line, = ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=linewidth,
                        solid_joinstyle="round", zorder=4)
        if dashes:
            line.set_dashes(list(dashes))


def _centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
    ys, xs = np.nonzero(mask > 0.5)
    return (float(xs.mean()), float(ys.mean())) if xs.size else None


def draw_pair(ax_overlay, ax_error, image, truth, prediction, row) -> None:
    """One frame: contours on the left, the error map and centroids on the right."""
    for ax in (ax_overlay, ax_error):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="antialiased")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#c8d2d8"); spine.set_linewidth(0.6)

    _contours(ax_overlay, truth, GT_COLOR, 1.5)
    _contours(ax_overlay, prediction, PRED_COLOR, 1.8, dashes=(3.4, 2.1))

    gt_bool, pred_bool = truth > 0.5, prediction > 0.5
    for region, color in ((np.logical_and(gt_bool, pred_bool), TP_COLOR),
                          (np.logical_and(gt_bool, ~pred_bool), FN_COLOR),
                          (np.logical_and(~gt_bool, pred_bool), FP_COLOR)):
        if region.any():
            overlay = np.zeros(region.shape + (4,), dtype=float)
            rgb = tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
            overlay[region] = (*rgb, OVERLAY_ALPHA)
            ax_error.imshow(overlay, interpolation="nearest", zorder=3)

    # The centroid is what the controller servos on, so the two centroids and
    # the vector between them are drawn explicitly rather than left to be
    # inferred from the shaded regions.
    gt_c, pred_c = _centroid(truth), _centroid(prediction)
    if gt_c and pred_c:
        ax_error.annotate("", xy=pred_c, xytext=gt_c, zorder=6,
                          arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=1.6,
                                          shrinkA=0, shrinkB=0))
        ax_error.plot(*gt_c, marker="o", ms=4.2, mfc=GT_COLOR, mec="white", mew=0.9, zorder=7)
        ax_error.plot(*pred_c, marker="o", ms=4.2, mfc=PRED_COLOR, mec="white", mew=0.9, zorder=7)


def caption(ax, row, reproduced_dice: Optional[float] = None) -> None:
    """Metrics and gate labels for one frame, set beside the panels.

    The block is drawn as a single multi-line string: stacking separate text
    calls inside a short axis lets the lines overlap when the axis is smaller
    than the type, which is exactly what a caption must never do.
    """
    ax.axis("off")
    outcome = row["q_decision"]
    colour = {"false_accept": FP_COLOR, "false_reject": "#B07A17",
              "true_accept": "#3D5A66", "true_reject": TP_COLOR}[outcome]

    ax.text(0.0, 0.90, f"{row['patient_id']} · frame {int(row['frame_index']):03d} · {row['split']}",
            transform=ax.transAxes, va="top", fontsize=11.5, fontweight="bold", color="#101a20")
    ax.text(0.0, 0.745, OUTCOME_TEXT[outcome], transform=ax.transAxes, va="top",
            fontsize=9.0, fontweight="bold", color=colour)

    warn = ""
    if reproduced_dice is not None and abs(reproduced_dice - float(row["dice"])) > 5e-4:
        warn = f"\n! reproduced Dice {reproduced_dice:.4f} disagrees with the label"
    body = (
        f"Dice  {float(row['dice']):.3f}         IoU  {float(row['iou']):.3f}\n"
        f"HD95  {float(row['hd95']):.1f} px       centroid error  "
        f"{float(row['centroid_error_px']):.2f} px\n"
        f"\n"
        f"Q     {float(row['quality']):.3f}\n"
        f"lowest term   {row['limiting_component']}  {float(row['limiting_value']):.2f}\n"
        f"GT area / ROI  {float(row['gt_area_ratio']):.4f}" + warn
    )
    ax.text(0.0, 0.58, body, transform=ax.transAxes, va="top", fontsize=9.2,
            color="#33454e", linespacing=1.7)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path,
                        default=Path("experiments/quality_retro/audit/frame_labels.csv"))
    parser.add_argument("--config", default="configs/exp_seed43_retro.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/exp_seed43/best.pt")
    parser.add_argument("--outcome", default="false_accept",
                        choices=["false_accept", "false_reject", "true_accept",
                                 "true_reject", "any"])
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--sort", default="dice")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/quality_retro/frames"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = select(args.labels, None if args.outcome == "any" else args.outcome,
                  args.sort, args.limit, args.descending, args.patients)

    predictor, config, size = build_predictor(args.config, args.checkpoint)
    from rus_perception.data.manifest import load_manifest
    data = config.section("data")
    manifest = load_manifest(str(data["manifest"]), root=str(data["root"]))

    panels, mismatched = [], 0
    for row in rows:
        image, truth = load_frame(manifest, row["frame_id"], size)
        state = predictor.predict_control_state(image)
        prediction = np.asarray(state.binary_mask, np.uint8)
        # The label table came from an earlier evaluation run. Re-deriving Dice
        # here and comparing is what keeps the picture and the numbers printed
        # on it describing the same prediction.
        overlap = float(np.logical_and(prediction > 0, truth > 0).sum())
        total = float((prediction > 0).sum() + (truth > 0).sum())
        reproduced = (2.0 * overlap / total) if total else float("nan")
        if abs(reproduced - float(row["dice"])) > 5e-4:
            mismatched += 1
            logger.warning("%s: reproduced Dice %.4f but the label says %s",
                           row["frame_id"], reproduced, row["dice"])
        panels.append((row, image, truth, prediction, reproduced))
        logger.info("%s  Dice %.4f  Q %s  %s", row["frame_id"], reproduced,
                    row["quality"], row["q_decision"])
    if mismatched:
        logger.warning("%d of %d panels disagree with the label table; the run and the "
                       "labels may be out of sync.", mismatched, len(panels))

    import matplotlib
    matplotlib.use("Agg")
    family = use_korean_font()
    if family:
        logger.info("Captions set in %s", family)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- individual panels ---
    for row, image, truth, prediction, reproduced in panels:
        figure = plt.figure(figsize=(8.6, 2.45))
        grid = GridSpec(1, 3, figure=figure, width_ratios=[1, 1, 1.72],
                        wspace=0.045, left=0.008, right=0.992, top=0.97, bottom=0.03)
        draw_pair(figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]),
                  image, truth, prediction, row)
        caption(figure.add_subplot(grid[0, 2]), row, reproduced)
        name = f"{row['split']}_{row['patient_id']}_{int(row['frame_index']):03d}.png"
        figure.savefig(args.output_dir / name, dpi=args.dpi, facecolor="white")
        plt.close(figure)

    # --- contact sheet ---
    row_h = 2.45
    figure = plt.figure(figsize=(8.6, row_h * len(panels) + 0.6))
    top = 1.0 - 0.55 / (row_h * len(panels) + 0.6)
    grid = GridSpec(len(panels), 3, figure=figure, width_ratios=[1, 1, 1.72],
                    wspace=0.045, hspace=0.09,
                    left=0.008, right=0.992, top=top, bottom=0.008)
    for index, (row, image, truth, prediction, reproduced) in enumerate(panels):
        draw_pair(figure.add_subplot(grid[index, 0]), figure.add_subplot(grid[index, 1]),
                  image, truth, prediction, row)
        caption(figure.add_subplot(grid[index, 2]), row, reproduced)

    figure.legend(
        handles=[Line2D([], [], color=GT_COLOR, lw=1.7, label="Ground truth"),
                 Line2D([], [], color=PRED_COLOR, lw=1.9, dashes=(3, 1.8), label="Prediction"),
                 Patch(facecolor=TP_COLOR, alpha=OVERLAY_ALPHA, label="True positive"),
                 Patch(facecolor=FN_COLOR, alpha=OVERLAY_ALPHA, label="False negative"),
                 Patch(facecolor=FP_COLOR, alpha=OVERLAY_ALPHA, label="False positive"),
                 Line2D([], [], color="#5a6b74", lw=1.4, marker="o", ms=4.5, mfc="white",
                        label="GT $\\rightarrow$ predicted centroid")],
        loc="upper center", ncol=6, fontsize=8.4, frameon=False,
        bbox_to_anchor=(0.5, 0.998), columnspacing=1.3, handlelength=1.8,
        handletextpad=0.5)
    sheet = args.output_dir / f"sheet_{args.outcome}_{len(panels)}.png"
    figure.savefig(sheet, dpi=args.dpi, facecolor="white")
    plt.close(figure)
    logger.info("Wrote %d panels and %s", len(panels), sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
