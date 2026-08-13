#!/usr/bin/env python3
"""Evaluate a checkpoint: spatial accuracy, temporal stability and latency.

Sequences are evaluated in acquisition order without shuffling, because
shuffling would destroy the very property the temporal metrics measure.

Example:
    python scripts/evaluate.py --config configs/slim_unet_temporal.yaml \
        --checkpoint checkpoints/slim_unet_temporal/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


from _common import parse_overrides, prepare_manifest  # noqa: E402  (path setup)

from src.control.features import (
    FeatureExtractionConfig,
    warp_mask_with_flow,
)
from src.data.io import load_grayscale, load_mask, resize_image, resize_mask
from src.flow.precomputed import load_flow_pair
from src.inference.predictor import Predictor, PredictorConfig
from src.metrics.latency import LatencyTracker
from src.metrics.spatial import build_metric_report, compute_frame_metrics
from src.metrics.temporal import (
    TemporalFrameRecord,
    aggregate_temporal_metrics,
    compute_sequence_temporal_metrics,
)
from src.utils.config import load_config
from src.utils.logging_utils import CsvWriter, JsonlWriter, setup_logging
from src.utils.seeding import seed_everything

logger = logging.getLogger("evaluate")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained bladder-segmentation checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Experiment config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    parser.add_argument("--split", default=None, help="Override evaluation.split")
    parser.add_argument("--manifest", default=None, help="Override data.manifest")
    parser.add_argument("--output-dir", default=None, help="Where to write reports")
    parser.add_argument("--device", default=None, help="Override the device")
    parser.add_argument("--max-sequences", type=int, default=None, help="Evaluate at most N sequences")
    parser.add_argument("--no-hd95", action="store_true", help="Skip the HD95 metric")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> int:
    args = get_args()
    config = load_config(args.config, overrides=parse_overrides(args.overrides))
    split = args.split or str(config.get("evaluation.split", "test"))
    output_dir = Path(args.output_dir or Path(config.get("experiment.output_dir", "runs/eval")) / f"eval_{split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(config.get("logging.level", "INFO")), output_dir / "evaluate.log")
    seed_everything(int(config.get("experiment.seed", 42)))

    manifest = prepare_manifest(config, args.manifest).filter(split=split)
    if len(manifest) == 0:
        raise SystemExit(f"Split {split!r} is empty; nothing to evaluate.")

    image_size = config.get("model.input_size") or config.get("data.image_size") or [128, 128]
    predictor_config = PredictorConfig(
        input_size=(int(image_size[0]), int(image_size[1])),
        intensity_normalization=str(config.get("data.intensity_normalization", "zero_one")),
        normalization_stats=config.get("data.normalization_stats"),
        device=args.device or str(config.get("train.device", "auto")).replace("auto", "cpu"),
        restore_original_size=False,  # metrics are computed at the model's resolution
    )
    if args.device is None and str(config.get("train.device", "auto")) == "auto":
        import torch

        predictor_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_config = FeatureExtractionConfig.from_dict(
        {"postprocess": config.section("postprocess"), **config.section("control")}
    )
    predictor = Predictor.from_checkpoint(
        args.checkpoint,
        model_config=config.section("model"),
        predictor_config=predictor_config,
        feature_config=feature_config,
    )
    logger.info(
        "Evaluating %s on split %r: %d frames, %d patients",
        args.checkpoint, split, len(manifest), len(manifest.patients),
    )

    compute_hd95 = bool(config.get("evaluation.compute_hd95", True)) and not args.no_hd95
    spacing = float(config.get("evaluation.pixel_spacing", 1.0))
    latency = LatencyTracker(warmup=int(config.get("evaluation.warmup_frames", 3)))

    frame_metrics = []
    sequence_metrics = []
    target_size = (int(image_size[0]), int(image_size[1]))

    with JsonlWriter(output_dir / "control_states.jsonl") as jsonl, CsvWriter(
        output_dir / "frame_metrics.csv"
    ) as csv_writer:
        sequences = sorted(manifest.sequences().items())
        if args.max_sequences:
            sequences = sequences[: args.max_sequences]

        for (patient, sequence), records in sequences:
            previous_state = None
            temporal_records: list[TemporalFrameRecord] = []

            for record in records:  # chronological order, never shuffled
                image = resize_image(load_grayscale(manifest.resolve(record.image_path)), target_size)
                target = None
                if record.is_labeled:
                    target = resize_mask(load_mask(manifest.resolve(record.mask_path)), target_size)

                flow_backward = None
                if record.flow_backward_path:
                    path = manifest.resolve(record.flow_backward_path)
                    if path is not None and path.is_file():
                        try:
                            pair = load_flow_pair(path)
                            from src.data.video_dataset import resize_flow

                            flow_backward = resize_flow(pair.backward, target_size)
                        except ValueError as exc:
                            logger.warning("Ignoring unusable flow %s: %s", path, exc)

                state = predictor.predict_control_state(
                    image,
                    previous_state=previous_state,
                    flow=flow_backward,
                    metadata={
                        "frame_id": record.frame_id,
                        "timestamp": record.timestamp or 0.0,
                        "patient_id": record.patient_id,
                        "sequence_id": record.sequence_id,
                    },
                )
                latency.record("preprocessing", state.preprocessing_latency_ms)
                latency.record("inference", state.inference_latency_ms)
                latency.record("postprocessing", state.postprocessing_latency_ms)
                latency.record("control_features", state.control_feature_latency_ms)
                latency.record("end_to_end", state.end_to_end_latency_ms)
                jsonl.write(state.to_dict())

                if target is not None:
                    metrics = compute_frame_metrics(
                        prediction=state.binary_mask,
                        target=target,
                        frame_id=record.frame_id,
                        patient_id=record.patient_id,
                        sequence_id=record.sequence_id,
                        frame_index=record.frame_index,
                        compute_hd95=compute_hd95,
                        spacing=spacing,
                    )
                    frame_metrics.append(metrics)
                    csv_writer.write(metrics.to_dict())

                warped_previous = None
                warped_previous_probability = None
                previous_centroid = None
                previous_area = None
                if previous_state is not None and previous_state.binary_mask is not None:
                    if flow_backward is not None:
                        warped_previous = warp_mask_with_flow(
                            previous_state.binary_mask, flow_backward
                        )
                    else:
                        warped_previous = previous_state.binary_mask
                    warped_previous_probability = previous_state.probability_map
                    if previous_state.centroid_x_normalized is not None:
                        previous_centroid = (
                            previous_state.centroid_x_normalized,
                            previous_state.centroid_y_normalized,
                        )
                    previous_area = previous_state.mask_area_ratio

                temporal_records.append(
                    TemporalFrameRecord(
                        frame_id=record.frame_id,
                        mask=state.binary_mask,
                        probability_map=state.probability_map,
                        warped_previous_mask=warped_previous,
                        warped_previous_probability=warped_previous_probability,
                        centroid=(
                            (state.centroid_x_normalized, state.centroid_y_normalized)
                            if state.centroid_x_normalized is not None
                            else None
                        ),
                        previous_centroid=previous_centroid,
                        area_ratio=state.mask_area_ratio,
                        previous_area_ratio=previous_area,
                        valid_for_control=state.valid_for_control,
                        target=target,
                    )
                )
                previous_state = state

            sequence_metrics.append(
                compute_sequence_temporal_metrics(
                    temporal_records,
                    sequence_id=f"{patient}/{sequence}",
                    stability_iou_threshold=float(config.get("evaluation.stability_iou_threshold", 0.7)),
                    accuracy_dice_threshold=float(config.get("evaluation.accuracy_dice_threshold", 0.7)),
                )
            )

    spatial_report = build_metric_report(frame_metrics)
    temporal_report = aggregate_temporal_metrics(sequence_metrics)
    latency_report = latency.to_dict()

    payload = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "split": split,
        "num_frames": len(manifest),
        "num_patients": len(manifest.patients),
        "num_labeled_frames": len(frame_metrics),
        "spatial": spatial_report.to_dict(),
        "temporal": temporal_report,
        "temporal_per_sequence": [s.to_dict() for s in sequence_metrics],
        "latency": latency_report,
    }
    (output_dir / "evaluation.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(spatial_report.to_text())
    print("\n=== Temporal stability ===")
    for key, value in temporal_report.items():
        if isinstance(value, dict):
            continue
        print(f"{key:<44}: {'n/a' if value is None else value}")
    classification = temporal_report.get("frame_classification", {})
    if classification:
        print("frame classification (stability vs accuracy):")
        for key in sorted(classification):
            print(f"  {key:<24}: {classification[key]}")
        print(
            "  NOTE: 'stable_inaccurate' frames are consistent over time but wrong. "
            "Temporal stability alone never implies anatomical correctness."
        )
    print("\n=== Latency (model-only stages and end-to-end) ===")
    print(latency.to_text())
    print(f"\nReports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
