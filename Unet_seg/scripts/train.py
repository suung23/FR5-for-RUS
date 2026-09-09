#!/usr/bin/env python3
"""Train a Standard U-Net or Slim U-Net from a YAML configuration.

Examples:
    python scripts/train.py --config configs/standard_unet_baseline.yaml
    python scripts/train.py --config configs/slim_unet_paper.yaml
    python scripts/train.py --config configs/slim_unet_temporal.yaml
    python scripts/train.py --config configs/slim_unet_temporal.yaml \
        --resume checkpoints/slim_unet_temporal/last.pt
    python scripts/train.py --config configs/slim_unet_temporal.yaml \
        --set loss.temporal.enabled=false          # temporal ablation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from _common import parse_overrides, prepare_manifest  # noqa: E402  (path setup)

from rus_perception.models.registry import build_model
from rus_perception.models.report import build_architecture_report
from rus_perception.training.trainer import Trainer
from rus_perception.utils.config import load_config
from rus_perception.utils.logging_utils import setup_logging
from rus_perception.utils.seeding import seed_everything

logger = logging.getLogger("train")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a bladder-ultrasound segmentation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume training from")
    parser.add_argument(
        "--init-weights",
        default=None,
        help="Fine-tuning: load only the model weights from this checkpoint (fresh optimizer, "
        "schedule and epoch counter). Use --resume to continue an interrupted run instead.",
    )
    parser.add_argument("--manifest", default=None, help="Override data.manifest")
    parser.add_argument("--output-dir", default=None, help="Override train.checkpoint_dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--device", default=None, help="Override train.device")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="Override any config value, e.g. --set loss.temporal.enabled=false",
    )
    parser.add_argument(
        "--skip-file-validation",
        action="store_true",
        help="Do not check that every file referenced by the manifest exists",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()
    overrides = parse_overrides(args.overrides)
    if args.epochs is not None:
        overrides.setdefault("train", {})["epochs"] = args.epochs
    if args.device is not None:
        overrides.setdefault("train", {})["device"] = args.device
    if args.output_dir is not None:
        overrides.setdefault("train", {})["checkpoint_dir"] = args.output_dir

    config = load_config(args.config, overrides=overrides)
    output_dir = Path(config.get("train.checkpoint_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(config.get("logging.level", "INFO")), output_dir / "train.log")
    seed_everything(int(config.get("experiment.seed", 42)))

    logger.info("Experiment: %s", config.get("experiment.name", "unnamed"))
    logger.info("Config    : %s", args.config)

    manifest = prepare_manifest(config, args.manifest, validate_files=not args.skip_file_validation)
    statistics = manifest.statistics()
    logger.info("Dataset statistics:\n%s", statistics.to_text())
    (output_dir / "dataset_statistics.json").write_text(
        json.dumps(statistics.to_dict(), indent=2), encoding="utf-8"
    )

    train_manifest = manifest.filter(split="train")
    val_manifest = manifest.filter(split="val")
    if len(train_manifest) == 0:
        raise SystemExit("The training split is empty; check split.strategy and the manifest.")
    if len(val_manifest) == 0:
        logger.warning("The validation split is empty; no model selection will be performed.")

    model = build_model(config.section("model"))
    input_size = config.get("model.input_size") or config.get("data.image_size") or [128, 128]
    report = build_architecture_report(
        model, (1, int(config.get("model.input_channels", 1)), int(input_size[0]), int(input_size[1]))
    )
    logger.info("Architecture:\n%s", report.to_table(max_layers=0))
    (output_dir / "architecture_report.json").write_text(report.to_json(), encoding="utf-8")

    trainer = Trainer(
        config=config,
        model=model,
        train_manifest=train_manifest,
        val_manifest=val_manifest if len(val_manifest) else None,
        output_dir=output_dir,
    )
    if args.resume:
        trainer.resume(args.resume)
    elif args.init_weights:
        from rus_perception.utils.checkpoint import load_checkpoint

        payload = load_checkpoint(args.init_weights, map_location="cpu")
        missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
        if unexpected or missing:
            raise SystemExit(
                f"--init-weights {args.init_weights} does not match the configured model "
                f"(missing {len(missing)}, unexpected {len(unexpected)} tensors)."
            )
        logger.info(
            "Initialised weights from %s (epoch %s, %s=%.4f); optimizer and schedule start fresh",
            args.init_weights,
            payload.get("epoch", "?"),
            payload.get("best_metric_name", "score"),
            float(payload.get("best_metric", float("nan"))),
        )

    state = trainer.fit()
    (output_dir / "history.json").write_text(json.dumps(state.history, indent=2), encoding="utf-8")
    logger.info(
        "Training finished. Best score %.4f at epoch %d. Checkpoints in %s",
        state.best_score,
        state.best_epoch + 1,
        output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
