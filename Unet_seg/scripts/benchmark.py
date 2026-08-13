#!/usr/bin/env python3
"""Benchmark model-only and end-to-end latency, throughput and memory.

CUDA is synchronised around every measurement and warm-up iterations are
discarded, so the reported numbers are real execution times rather than kernel
enqueue times.

Example:
    python scripts/benchmark.py --checkpoint checkpoints/slim_unet_production/best.pt --device cuda
    python scripts/benchmark.py --model slim_unet --preset paper --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (path setup)

from src.control.features import FeatureExtractionConfig
from src.inference.predictor import Predictor, PredictorConfig
from src.metrics.latency import benchmark_callable, device_report, synchronize
from src.models.registry import build_model
from src.models.report import build_architecture_report
from src.utils.checkpoint import load_checkpoint
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("benchmark")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure inference latency and throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to benchmark")
    parser.add_argument("--config", default=None, help="Config providing the model section")
    parser.add_argument("--model", default="slim_unet", choices=["slim_unet", "standard_unet"])
    parser.add_argument("--preset", default="paper", help="Model preset when no checkpoint is given")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    parser.add_argument("--input-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--amp", action="store_true", help="Use autocast mixed precision (CUDA)")
    parser.add_argument("--json", default=None, help="Write the report to this JSON file")
    return parser.parse_args()


def main() -> int:
    args = get_args()
    setup_logging("INFO")

    import torch

    model_config = {"name": args.model, "preset": args.preset}
    if args.config:
        from src.utils.config import load_config

        model_config = load_config(args.config).section("model")
    if args.checkpoint:
        payload = load_checkpoint(args.checkpoint, map_location="cpu")
        stored = (payload.get("config") or {}).get("model")
        if stored and not args.config:
            model_config = stored

    model = build_model(model_config)
    if args.checkpoint:
        model.load_state_dict(load_checkpoint(args.checkpoint, map_location="cpu")["model_state"])
    model.eval()

    size = args.input_size or model_config.get("input_size") or [128, 128]
    height, width = int(size[0]), int(size[1])
    channels = int(model_config.get("input_channels", model_config.get("in_channels", 1)))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; benchmarking on CPU instead.")
        device = torch.device("cpu")
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    report = build_architecture_report(model.cpu(), (1, channels, height, width))
    model = model.to(device)

    dummy = torch.zeros(args.batch_size, channels, height, width, device=device)
    use_amp = args.amp and device.type == "cuda"

    def model_only() -> None:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=use_amp):
            model(dummy)

    model_stats = benchmark_callable(
        model_only, args.iterations, args.warmup, device, name="model_only"
    )

    predictor = Predictor(
        model,
        PredictorConfig(
            input_size=(height, width),
            device=str(device),
            amp=use_amp,
            restore_original_size=False,
        ),
        feature_config=FeatureExtractionConfig(),
    )
    frame = (np.random.default_rng(0).random((height, width)) * 255).astype(np.uint8)

    def end_to_end() -> None:
        predictor.predict_control_state(frame, metadata={"frame_id": "bench"})

    e2e_stats = benchmark_callable(
        end_to_end, max(10, args.iterations // 4), max(3, args.warmup // 4), device, name="end_to_end"
    )

    synchronize(device)
    devices = device_report(device)

    payload = {
        "model": model_config,
        "input_shape": [args.batch_size, channels, height, width],
        "trainable_parameters": report.trainable_parameters,
        "total_parameters": report.total_parameters,
        "estimated_macs": report.estimated_macs,
        "estimated_gmacs": report.estimated_macs / 1e9 if report.estimated_macs else None,
        "amp": use_amp,
        "model_only": model_stats.to_dict(),
        "end_to_end": e2e_stats.to_dict(),
        "device_report": devices,
    }

    print("=" * 88)
    print(f"Model            : {model_config.get('name')} (preset={model_config.get('preset')})")
    print(f"Input            : {tuple(payload['input_shape'])}   AMP: {use_amp}")
    print(f"Parameters       : {report.trainable_parameters:,} trainable")
    if report.estimated_macs:
        print(f"Estimated MACs   : {report.estimated_macs / 1e9:.3f} G (analytic, not measured)")
    print(f"Device           : {devices.get('gpu_name', devices['device'])}")
    print("-" * 88)
    print("Model-only latency (network forward pass only):")
    print("  " + model_stats.to_text())
    print("End-to-end latency (preprocess + inference + postprocess + control features):")
    print("  " + e2e_stats.to_text())
    if "gpu_max_memory_allocated_mb" in devices:
        print(f"Peak GPU memory  : {devices['gpu_max_memory_allocated_mb']:.1f} MB")
    else:
        print("Peak GPU memory  : not applicable (CPU run)")
    print("=" * 88)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Benchmark report written to %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
