#!/usr/bin/env python3
"""The two drawn figures of the lateral-instruction manuscript.

Figure 1 is a schematic: which probe axes leave the imaging plane invariant, and
how the lateral instruction is read off one frame. Figure 2 is a qualitative
strip showing the same frame at three lateral displacements, including one
inside the band where the instruction is no longer trustworthy.

The quantitative figures (agreement, validity curve) are produced by
scripts/lateral_instruction.py; this file draws only what has to be drawn.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Text is black throughout; the only colours are a dark navy / grey pair for
# graphic marks, and the annotation colours the manuscript captions name.
INK = "#000000"
INK_SOFT = "#444444"
MUTED = "#8a8a8a"
ACCENT = "#1f3864"
CRIT = "#a32316"
GT_COLOR = "#16B9D4"
PRED_COLOR = "#E17C32"


def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def figure_axes_and_instruction(roi: np.ndarray):
    """(a) which axes preserve the plane, (b) how the instruction is read."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, Rectangle

    _style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.4, 2.9),
                                         gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- (a) axis decomposition --------------------------------------
    left.set_xlim(0, 10); left.set_ylim(0, 8); left.axis("off")
    left.add_patch(Rectangle((3.5, 5.6), 3.0, 1.0, facecolor="#ececec",
                             edgecolor=INK_SOFT, lw=0.9))
    left.text(5.0, 6.1, "probe", ha="center", va="center", fontsize=8.5, color=INK)
    # imaging plane, drawn as the dark screen the probe fills — the same
    # register as the B-mode panel beside it.
    left.fill([5.0, 2.6, 7.4], [5.6, 0.9, 0.9], color="#14181d", edgecolor=INK_SOFT, lw=0.8)
    left.text(5.0, 2.1, "imaging plane", ha="center", fontsize=8, color="#f2f2f2")

    left.add_patch(FancyArrow(5.0, 4.6, 2.0, 0, width=0.055, head_width=0.30,
                              head_length=0.42, color="#f2f2f2", length_includes_head=True))
    left.add_patch(FancyArrow(5.0, 4.6, -2.0, 0, width=0.055, head_width=0.30,
                              head_length=0.42, color="#f2f2f2", length_includes_head=True))
    left.text(7.2, 4.95, "$v_x$", fontsize=10, color=INK, ha="center", fontweight="bold")
    left.text(7.25, 4.28, "in plane", fontsize=7.4, color=INK, ha="center")

    left.add_patch(FancyArrow(5.0, 6.9, 1.15, 0.62, width=0.045, head_width=0.26,
                              head_length=0.36, color=MUTED, length_includes_head=True))
    left.text(6.6, 7.55, "$v_y$", fontsize=10, color=INK, ha="center")
    left.text(2.5, 7.55, "$\\omega_z$", fontsize=10, color=INK, ha="center")
    left.annotate("", xy=(3.05, 7.2), xytext=(2.0, 6.75),
                  arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3,
                                  connectionstyle="arc3,rad=0.5"))
    left.text(4.55, 7.9, "leave the plane", fontsize=7.4, color=INK, ha="center")
    left.text(0.0, 1.02, "(a)", transform=left.transAxes, fontsize=10,
              fontweight="bold", color=INK, va="bottom")

    # ---- (b) the instruction on one frame ----------------------------
    # Rendered as a synthetic B-mode: speckle-textured sector on black,
    # anechoic lumen with a bright wall and posterior enhancement, white
    # annotation — the register of an actual ultrasound display.
    from PIL import Image as _PILImage
    from PIL import ImageFilter as _PILFilter

    def gaussian_filter(a, sigma):
        im = _PILImage.fromarray((255 * (a - a.min()) / (float(np.ptp(a)) or 1)).astype(np.uint8))
        return np.asarray(im.filter(_PILFilter.GaussianBlur(radius=sigma)), float)

    rng = np.random.default_rng(7)
    speckle = gaussian_filter(rng.gamma(2.0, 1.0, size=roi.shape), 1.1)
    speckle = (speckle - speckle.min()) / (speckle.max() - speckle.min())
    yy, xx = np.mgrid[0:256, 0:256]
    depth_gain = 1.0 - 0.30 * (yy / 255.0)
    frame = np.zeros(roi.shape, float)
    frame[roi] = (0.14 + 0.34 * speckle[roi]) * depth_gain[roi]

    ys, xs = np.nonzero(roi)
    axis_x = xs.mean()
    lumen = (86.0, 108.0)
    inside = ((xx - lumen[0]) / 30.0) ** 2 + ((yy - lumen[1]) / 21.0) ** 2
    frame[(inside <= 1.0) & roi] = 0.03 + 0.03 * speckle[(inside <= 1.0) & roi]
    # posterior acoustic enhancement below the anechoic lumen — feathered in
    # x and ramped in y so it reads as physics, not as a painted rectangle
    boost = np.exp(-(((xx - lumen[0]) / 24.0) ** 2))
    ramp = 1.0 / (1.0 + np.exp(-(yy - (lumen[1] + 26)) / 7.0))
    gain = boost * ramp
    frame[roi] = np.clip(frame[roi] * (1 + 1.1 * gain[roi]) + 0.05 * gain[roi], 0, 1)
    right.imshow(frame, cmap="gray", vmin=0, vmax=1, origin="upper",
                 interpolation="antialiased")
    right.set_xlim(0, 256); right.set_ylim(256, 0); right.axis("off")
    US_INK = "#f2f2f2"
    right.axvline(axis_x, color=US_INK, lw=1.1, ls=(0, (5, 3)), zorder=4)
    right.text(axis_x + 4, 16, "beam axis $A$", fontsize=7.6, color=US_INK, va="top")

    theta = np.linspace(0, 2 * np.pi, 200)
    right.plot(lumen[0] + 30 * np.cos(theta), lumen[1] + 21 * np.sin(theta),
               color="#e9ede8", lw=1.6, zorder=5)
    right.plot(*lumen, marker="o", ms=5, mfc="white", mec=INK, mew=1.0, zorder=7)
    right.text(lumen[0], lumen[1] - 26, "lumen", fontsize=7.6, color=US_INK,
               ha="center", va="bottom", zorder=7)
    right.annotate("", xy=(axis_x, 152), xytext=(lumen[0], 152), zorder=8,
                   arrowprops=dict(arrowstyle="-|>", color=US_INK, lw=1.7))
    right.text((axis_x + lumen[0]) / 2, 146, "$e = A - c$", fontsize=9.5, color=US_INK,
               ha="center", va="bottom", zorder=8)
    right.text(0.0, 1.02, "(b)", transform=right.transAxes, fontsize=10,
               fontweight="bold", color=INK, va="bottom")

    figure.tight_layout(w_pad=1.4)
    return figure


def figure_examples(predictor, feature_config, roi, image, truth, axis, targets):
    """One frame at three lateral displacements, with the instruction drawn."""
    import cv2
    import matplotlib.pyplot as plt

    from rus_perception.control.features import extract_control_state

    _style()
    gy, gx = np.nonzero(truth)
    natural = axis - float(gx.mean())
    figure, axes = plt.subplots(1, len(targets), figsize=(7.4, 2.85))

    for ax, target in zip(axes, targets):
        dx = int(round(natural - target))
        matrix = np.float32([[1, 0, dx], [0, 1, 0]])
        frame = cv2.warpAffine(image, matrix, (256, 256), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        moved = cv2.warpAffine(truth, matrix, (256, 256), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REPLICATE)
        probability, _ = predictor.predict_probability(frame)
        state = extract_control_state(probability, image=frame,
                                      config=feature_config, roi_mask=roi)
        prediction = np.asarray(state.binary_mask, np.uint8)
        py, px = np.nonzero(prediction)
        my, mx = np.nonzero(moved)
        instruction = axis - float(px.mean())
        true = axis - float(mx.mean())

        ax.imshow(frame, cmap="gray", vmin=0, vmax=1, interpolation="antialiased")
        for mask, colour, width, dashes in ((moved, GT_COLOR, 1.4, None),
                                            (prediction, PRED_COLOR, 1.7, (3.2, 2.0))):
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                points = contour.reshape(-1, 2)
                if len(points) < 3:
                    continue
                closed = np.vstack([points, points[:1]])
                line, = ax.plot(closed[:, 0], closed[:, 1], color=colour, lw=width, zorder=4)
                if dashes:
                    line.set_dashes(list(dashes))
        ax.axvline(axis, color="white", lw=1.0, ls=(0, (5, 3)), zorder=5)
        wrong = np.sign(instruction) != np.sign(true)
        ax.annotate("", xy=(axis, 208), xytext=(float(px.mean()), 208), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", lw=2.0,
                                    color=CRIT if wrong else "#2e7d4f"))
        ax.set_title(f"$e$ = {true:+.1f} px", fontsize=9, color=INK, pad=5)
        ax.text(0.5, -0.055,
                f"instruction {instruction:+.1f} px" + ("   wrong way" if wrong else ""),
                transform=ax.transAxes, ha="center", va="top", fontsize=7.8,
                color=CRIT if wrong else INK,
                fontweight="bold" if wrong else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#b3b3b3"); spine.set_linewidth(0.6)

    figure.tight_layout(w_pad=0.9)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp_seed43_retro.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/exp_seed43/best.pt")
    parser.add_argument("--patient", default="P041")
    parser.add_argument("--targets", type=float, nargs=3, default=[40, 13, 2])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/lateral_instruction"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

    from rus_perception.control.features import FeatureExtractionConfig
    from rus_perception.control.roi import RoiConfig, build_roi_mask
    from rus_perception.data.io import (load_grayscale, load_mask, resize_image,
                                        resize_mask)
    from rus_perception.data.manifest import load_manifest
    from rus_perception.inference.predictor import Predictor, PredictorConfig
    from rus_perception.utils.config import load_config
    config = load_config(args.config)
    size = tuple(int(v) for v in config.section("data")["image_size"])
    roi_config = RoiConfig.from_dict(config.section("control").get("roi"))
    roi = np.asarray(build_roi_mask(size, roi_config)) > 0
    ys, xs = np.nonzero(roi)
    axis = float(xs.mean())
    feature_config = FeatureExtractionConfig.from_dict(
        {"postprocess": config.section("postprocess"), **config.section("control")})
    predictor = Predictor.from_checkpoint(
        args.checkpoint, model_config=config.section("model"),
        predictor_config=PredictorConfig(
            input_size=size,
            intensity_normalization=str(config.get("data.intensity_normalization")),
            device=str(config.get("train.device", "auto")).replace("auto", "cuda"),
            restore_original_size=False, roi=roi_config),
        feature_config=feature_config)

    data = config.section("data")
    manifest = load_manifest(str(data["manifest"]), root=str(data["root"]))
    records = sorted([r for r in manifest
                      if r.patient_id == args.patient and r.mask_path],
                     key=lambda r: r.frame_id)
    if not records:
        raise SystemExit(f"No labelled frame for {args.patient}.")
    record = records[len(records) // 2]
    image = np.asarray(resize_image(load_grayscale(str(manifest.resolve(record.image_path))),
                                    size), np.float32)
    truth = (np.asarray(resize_mask(load_mask(str(manifest.resolve(record.mask_path))),
                                    size)) > 0.5).astype(np.uint8)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = ((figure_axes_and_instruction(roi), "fig_axes"),
             (figure_examples(predictor, feature_config, roi, image, truth,
                              axis, args.targets), "fig_examples"))
    for figure, stem in pairs:
        for suffix in ("png", "pdf"):
            figure.savefig(args.output_dir / f"{stem}.{suffix}", dpi=args.dpi,
                           facecolor="white", bbox_inches="tight")
        plt.close(figure)
    logger.info("Wrote fig_axes and fig_examples to %s (frame %s)",
                args.output_dir, record.frame_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
