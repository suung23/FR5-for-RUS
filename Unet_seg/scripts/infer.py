#!/usr/bin/env python3
"""Batch inference over an image directory, a single image, or a manifest split.

Writes predicted masks, optional probability maps and overlays, plus a JSONL log
of the full ControlState for every frame.

Example:
    python scripts/infer.py --config configs/slim_unet_production.yaml \
        --checkpoint checkpoints/slim_unet_production/best.pt \
        --input data/synthetic/images --output-dir runs/infer
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from _common import parse_overrides, prepare_manifest  # noqa: E402  (path setup)

from rus_perception.control.features import FeatureExtractionConfig
from rus_perception.data.io import load_grayscale
from rus_perception.inference.predictor import Predictor, PredictorConfig
from rus_perception.inference.sources import IMAGE_EXTENSIONS
from rus_perception.utils.config import load_config
from rus_perception.utils.logging_utils import CsvWriter, JsonlWriter, setup_logging
from rus_perception.utils.seeding import seed_everything

logger = logging.getLogger("infer")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch inference with a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Experiment config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to load")
    parser.add_argument("--input", default=None, help="Image file or directory")
    parser.add_argument("--split", default=None, help="Use this manifest split instead of --input")
    parser.add_argument("--output-dir", default="runs/infer", help="Destination directory")
    parser.add_argument("--device", default=None, help="Override the device")
    parser.add_argument("--threshold", type=float, default=None, help="Override postprocess.threshold")
    parser.add_argument("--save-probability", action="store_true", help="Also save probability .npy files")
    parser.add_argument("--save-overlay", action="store_true", help="Also save overlay PNGs")
    parser.add_argument("--no-save-mask", action="store_true", help="Do not write mask PNGs")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    return parser.parse_args()


def collect_inputs(args, config) -> list[tuple[str, Path]]:
    """Return ``(frame_id, path)`` pairs from --input or a manifest split."""
    if args.input:
        path = Path(args.input)
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            if not files:
                raise SystemExit(f"No images found in {path}.")
            return [(p.stem, p) for p in files]
        if path.is_file():
            return [(path.stem, path)]
        raise SystemExit(f"--input {path} does not exist.")

    manifest = prepare_manifest(config, None)
    if args.split:
        manifest = manifest.filter(split=args.split)
    if len(manifest) == 0:
        raise SystemExit("No frames selected; pass --input or check --split.")
    return [(r.frame_id, manifest.resolve(r.image_path)) for r in manifest]


def save_overlay(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    """Write an RGB overlay of ``mask`` on ``image``."""
    import cv2

    base = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    coloured = rgb.copy()
    coloured[mask > 0] = (0, 255, 0)
    blended = cv2.addWeighted(rgb, 0.65, coloured, 0.35, 0)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(str(path), blended)


def main() -> int:
    args = get_args()
    overrides = parse_overrides(args.overrides)
    if args.threshold is not None:
        overrides.setdefault("postprocess", {})["threshold"] = args.threshold

    config = load_config(args.config, overrides=overrides)
    output_dir = Path(args.output_dir)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    setup_logging(str(config.get("logging.level", "INFO")), output_dir / "infer.log")
    seed_everything(int(config.get("experiment.seed", 42)))

    inputs = collect_inputs(args, config)
    image_size = config.get("model.input_size") or config.get("data.image_size") or [128, 128]

    device = args.device
    if device is None:
        import torch

        configured = str(config.get("train.device", "auto"))
        device = ("cuda" if torch.cuda.is_available() else "cpu") if configured == "auto" else configured

    predictor = Predictor.from_checkpoint(
        args.checkpoint,
        model_config=config.section("model"),
        predictor_config=PredictorConfig(
            input_size=(int(image_size[0]), int(image_size[1])),
            intensity_normalization=str(config.get("data.intensity_normalization", "zero_one")),
            normalization_stats=config.get("data.normalization_stats"),
            device=device,
            restore_original_size=True,
        ),
        feature_config=FeatureExtractionConfig.from_dict(
            {"postprocess": config.section("postprocess"), **config.section("control")}
        ),
    )
    logger.info("Loaded %s on %s; %d input frame(s)", args.checkpoint, device, len(inputs))

    from PIL import Image

    valid_frames = 0
    with JsonlWriter(output_dir / "control_states.jsonl") as jsonl, CsvWriter(
        output_dir / "control_states.csv"
    ) as csv_writer:
        for index, (frame_id, path) in enumerate(inputs):
            image = load_grayscale(path)
            state = predictor.predict_control_state(
                image, metadata={"frame_id": frame_id, "timestamp": float(index)}
            )
            stem = frame_id.replace("/", "_")
            if not args.no_save_mask:
                Image.fromarray((state.binary_mask * 255).astype(np.uint8)).save(
                    output_dir / "masks" / f"{stem}_mask.png"
                )
            if args.save_probability:
                (output_dir / "probability").mkdir(parents=True, exist_ok=True)
                np.save(output_dir / "probability" / f"{stem}_prob.npy", state.probability_map)
            if args.save_overlay:
                (output_dir / "overlays").mkdir(parents=True, exist_ok=True)
                save_overlay(image, state.binary_mask, output_dir / "overlays" / f"{stem}_overlay.png")

            jsonl.write(state.to_dict())
            csv_writer.write(state.flat_record())
            valid_frames += int(state.valid_for_control)

    logger.info(
        "Wrote results for %d frame(s) to %s; %d/%d passed the validity gate.",
        len(inputs), output_dir, valid_frames, len(inputs),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
