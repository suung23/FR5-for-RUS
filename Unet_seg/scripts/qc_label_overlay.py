#!/usr/bin/env python3
"""Render a raw / ground-truth / prediction montage for label review.

Automated label-quality statistics did not survive contact with this dataset:
boundary-gradient ratio, interior-to-ring contrast, anechoic-component overlap
and label-versus-image motion all rank at least one visually-correct patient
below a visually-broken one (P048 scores like P021 on the first two, P020 and
P012 score below P021 on the third, P020 ties it on the fourth). Bladder labels
here are therefore reviewed by eye, and this script produces the record of that
review so a drop decision can be re-examined later.

Each row is one frame; the columns are the raw B-mode, the raw frame with the
ground-truth contour, and one column per checkpoint with its prediction filled
in under the same contour. A reference patient with known-good labels belongs in
every montage -- a suspect label only reads as suspect next to a correct one.

Predictions are evidence only for a patient the checkpoint did NOT train on;
the header marks a patient that was in the training split so a memorised mask is
not mistaken for agreement.

Example:
    python scripts/qc_label_overlay.py --patient P000 --frames 0,40,80,120,146 \
        --reference P020 --reference-frames 20,60 \
        --checkpoint checkpoints/pfus_bladder/best.pt:configs/pfus_bladder.yaml \
        --output docs/qc_p000.png
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

from rus_perception.control.features import FeatureExtractionConfig
from rus_perception.data.io import load_grayscale, load_mask
from rus_perception.data.manifest import load_manifest
from rus_perception.inference.predictor import Predictor, PredictorConfig
from rus_perception.utils.config import load_config
from rus_perception.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

GT_COLOUR = (255, 255, 0)          # cyan contour, drawn in every column but the first
PRED_COLOURS = [(90, 150, 255), (120, 255, 140), (200, 140, 255), (140, 220, 255)]
ROW_HEIGHT = 210


def build_predictor(checkpoint: str, config_path: str, device: str) -> tuple[Predictor, set[str]]:
    """Load a checkpoint and report which patients its training split contained.

    Returns:
        ``(predictor, train_patients)``. ``train_patients`` is empty when the
        manifest cannot be read, in which case no training warning is shown.
    """
    config = load_config(config_path)
    size = config.get("model.input_size") or config.get("data.image_size") or [128, 128]
    predictor_config = PredictorConfig(
        input_size=(int(size[0]), int(size[1])),
        intensity_normalization=str(config.get("data.intensity_normalization", "zero_one")),
        device=device,
        restore_original_size=True,
    )
    feature_config = FeatureExtractionConfig.from_dict(
        {"postprocess": config.section("postprocess"), **config.section("control")}
    )
    predictor = Predictor.from_checkpoint(
        checkpoint,
        model_config=config.section("model"),
        predictor_config=predictor_config,
        feature_config=feature_config,
    )
    train_patients: set[str] = set()
    try:
        manifest = load_manifest(str(config.require("data.manifest")))
        train_patients = {r.patient_id for r in manifest if (r.split or "") == "train"}
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.warning("Could not read the training split of %s: %s", config_path, exc)
    return predictor, train_patients


def frame_paths(root: Path, patient: str, index: int) -> tuple[Path, Path]:
    return (root / "raw" / "data" / patient / f"frame_{index:03d}.png",
            root / "masks" / patient / f"frame_{index:03d}.png")


def render_row(
    root: Path, patient: str, index: int, predictors: list[tuple[str, Predictor]],
    grad_ratio: dict[tuple[str, int], float],
) -> np.ndarray:
    """Render one frame as a horizontal strip of panels.

    Raises:
        FileNotFoundError: If the frame or its mask is missing.
    """
    image_path, mask_path = frame_paths(root, patient, index)
    image = load_grayscale(image_path)
    truth = load_mask(mask_path)
    if truth.shape != image.shape:
        truth = cv2.resize(truth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    base = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        (truth > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    panels = [base.copy()]
    with_truth = base.copy()
    cv2.drawContours(with_truth, contours, -1, GT_COLOUR, 2)
    panels.append(with_truth)

    for position, (_, predictor) in enumerate(predictors):
        probability, _ = predictor.predict_probability(image, output_size=image.shape)
        mask = np.asarray(probability) > 0.5
        panel = base.copy()
        overlay = panel.copy()
        overlay[mask] = PRED_COLOURS[position % len(PRED_COLOURS)]
        panel = cv2.addWeighted(panel, 0.55, overlay, 0.45, 0)
        cv2.drawContours(panel, contours, -1, GT_COLOUR, 2)
        panels.append(panel)

    scaled = [
        cv2.resize(p, (int(p.shape[1] * ROW_HEIGHT / p.shape[0]), ROW_HEIGHT)) for p in panels
    ]
    width = min(p.shape[1] for p in scaled)
    strip = np.hstack(
        [np.hstack([p[:, :width], np.full((ROW_HEIGHT, 3, 3), 40, np.uint8)]) for p in scaled]
    )

    caption = f"{patient}  frame {index:03d}"
    ratio = grad_ratio.get((patient, index))
    if ratio is not None:
        caption += f"   grad ratio {ratio:.2f}"
    label = np.full((26, strip.shape[1], 3), 25, np.uint8)
    cv2.putText(label, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([label, strip])


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", default="/home/rosotauser/datasets/pfus")
    parser.add_argument("--patient", required=True, help="Patient under review")
    parser.add_argument("--frames", required=True, help="Comma-separated frame indices")
    parser.add_argument("--reference", default=None, help="Known-good patient for comparison")
    parser.add_argument("--reference-frames", default=None, help="Frames of the reference patient")
    parser.add_argument(
        "--checkpoint", action="append", default=None, metavar="CKPT:CONFIG",
        help="Checkpoint and its config, separated by a colon. Repeatable.",
    )
    parser.add_argument("--grad-ratio-csv", default=None,
                        help="Optional CSV with patient_id,frame_index,grad_ratio to caption rows")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, help="PNG to write")
    return parser.parse_args()


def main() -> int:
    args = get_args()
    setup_logging("INFO")
    root = Path(args.dataset_root)

    predictors: list[tuple[str, Predictor]] = []
    train_patients: list[set[str]] = []
    for spec in args.checkpoint or []:
        if ":" not in spec:
            raise ValueError(f"--checkpoint must be CKPT:CONFIG, got {spec!r}.")
        ckpt, cfg = spec.rsplit(":", 1)
        predictor, trained = build_predictor(ckpt, cfg, args.device)
        predictors.append((Path(ckpt).parent.name, predictor))
        train_patients.append(trained)

    grad_ratio: dict[tuple[str, int], float] = {}
    if args.grad_ratio_csv:
        with open(args.grad_ratio_csv, newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    grad_ratio[(row["patient_id"], int(row["frame_index"]))] = float(row["grad_ratio"])
                except (KeyError, ValueError):
                    continue

    blocks = []
    plan = [(args.patient, [int(x) for x in args.frames.split(",")])]
    if args.reference:
        ref_frames = args.reference_frames or args.frames
        plan.append((args.reference, [int(x) for x in ref_frames.split(",")]))

    for patient, indices in plan:
        rows = [render_row(root, patient, i, predictors, grad_ratio) for i in indices]
        width = min(r.shape[1] for r in rows)
        blocks.append(
            np.vstack([np.vstack([r[:, :width], np.full((6, width, 3), 40, np.uint8)]) for r in rows])
        )

    width = min(b.shape[1] for b in blocks)
    body = np.vstack(
        [np.vstack([b[:, :width], np.full((14, width, 3), 15, np.uint8)]) for b in blocks]
    )

    titles = ["raw B-mode", "+ GT (cyan)"]
    for position, (name, _) in enumerate(predictors):
        note = " [TRAINED ON]" if args.patient in train_patients[position] else ""
        titles.append(f"pred: {name}{note}")
    header = np.full((30, width, 3), 15, np.uint8)
    segment = width // max(len(titles), 1)
    for position, text in enumerate(titles):
        cv2.putText(header, text, (position * segment + 8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 220, 120), 1, cv2.LINE_AA)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([header, body]))
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
