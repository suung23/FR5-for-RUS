#!/usr/bin/env python3
"""Real-time monitoring application for bladder-lumen segmentation.

Displays the raw frame, the segmentation overlay, the probability map and the
control / temporal-stability values, with bounded rolling sparklines. Status
colour: green = valid, yellow = marginal, red = invalid for control.

This application reports perception state only. It never emits robot commands.

Examples:
    python scripts/live_monitor.py --config configs/realtime_monitor.yaml \
        --checkpoint checkpoints/best.pt --source path/to/video.mp4
    python scripts/live_monitor.py --config configs/realtime_monitor.yaml \
        --checkpoint checkpoints/best.pt --source 0                  # camera
    python scripts/live_monitor.py --config configs/realtime_monitor.yaml \
        --checkpoint checkpoints/best.pt --source data/frames --headless \
        --save-video runs/monitor/annotated.mp4
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from pathlib import Path


from _common import parse_overrides  # noqa: E402  (path setup)

from rus_perception.control.features import FeatureExtractionConfig
from rus_perception.flow.backends import build_flow_backend
from rus_perception.inference.predictor import Predictor, PredictorConfig
from rus_perception.inference.realtime import PipelineResult, RealtimeConfig, RealtimePipeline
from rus_perception.inference.sources import open_source
from rus_perception.inference.visualization import MonitorRenderer, RollingHistory, StatusColors
from rus_perception.utils.config import load_config
from rus_perception.utils.logging_utils import CsvWriter, JsonlWriter, setup_logging
from rus_perception.utils.seeding import seed_everything

logger = logging.getLogger("live_monitor")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time bladder-segmentation monitor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Monitor config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to run")
    parser.add_argument(
        "--source", required=True,
        help="Video file, image directory, camera index (e.g. 0) or stream URI",
    )
    parser.add_argument("--device", default=None, help="Override monitor.device")
    parser.add_argument("--headless", action="store_true", help="Do not open a display window")
    parser.add_argument("--save-video", default=None, help="Write an annotated video here")
    parser.add_argument("--jsonl", default=None, help="Override monitor.jsonl_log")
    parser.add_argument("--csv", default=None, help="Override monitor.csv_log")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal features")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> int:
    args = get_args()
    config = load_config(args.config, overrides=parse_overrides(args.overrides))
    monitor = config.section("monitor")

    setup_logging(str(config.get("logging.level", "INFO")))
    seed_everything(int(config.get("experiment.seed", 42)))

    headless = args.headless or bool(monitor.get("headless", False))
    device = args.device or str(monitor.get("device", "auto"))
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    image_size = config.get("model.input_size") or [256, 256]
    predictor = Predictor.from_checkpoint(
        args.checkpoint,
        model_config=config.section("model"),
        predictor_config=PredictorConfig(
            input_size=(int(image_size[0]), int(image_size[1])),
            intensity_normalization=str(config.get("data.intensity_normalization", "zero_one")),
            normalization_stats=config.get("data.normalization_stats"),
            device=device,
            amp=bool(monitor.get("amp", False)),
            channels_last=bool(monitor.get("channels_last", False)),
            restore_original_size=True,
        ),
        feature_config=FeatureExtractionConfig.from_dict(
            {"postprocess": config.section("postprocess"), **config.section("control")}
        ),
    )

    enable_temporal = bool(monitor.get("enable_temporal", True)) and not args.no_temporal
    flow_backend = None
    if enable_temporal:
        try:
            flow_backend = build_flow_backend(config.section("flow"))
            if flow_backend.name == "precomputed":
                logger.info(
                    "flow.backend is 'precomputed', which cannot estimate live flow; "
                    "the monitor will compare frames unwarped and mark them as such."
                )
                flow_backend = None
        except (KeyError, ImportError) as exc:
            logger.warning("Could not build the flow backend (%s); continuing without it.", exc)

    pipeline = RealtimePipeline(
        predictor,
        RealtimeConfig(
            max_queue_size=int(monitor.get("max_queue_size", 2)),
            drop_stale_frames=bool(monitor.get("drop_stale_frames", True)),
            threaded_capture=bool(monitor.get("threaded_capture", True)),
            warmup_frames=int(monitor.get("warmup_frames", 5)),
            enable_temporal=enable_temporal,
            flow_backend=flow_backend,
            snapshot_on_validity_change=bool(monitor.get("snapshot_on_validity_change", False)),
        ),
    )

    renderer = MonitorRenderer(
        colors=StatusColors.from_dict(monitor.get("status_colors")),
        history=RollingHistory(maxlen=int(monitor.get("history_length", 240))),
        marginal_quality_threshold=float(monitor.get("marginal_quality_threshold", 0.55)),
    )

    source = open_source(
        args.source, grayscale=True, frame_rate=float(monitor.get("frame_rate", 30.0))
    )
    logger.info(
        "Source %s (%s) | device=%s | temporal=%s | headless=%s",
        args.source, source.name, device, enable_temporal, headless,
    )

    jsonl_path = args.jsonl or monitor.get("jsonl_log")
    csv_path = args.csv or monitor.get("csv_log")
    video_path = args.save_video or monitor.get("save_video")
    snapshot_dir = Path(monitor.get("snapshot_dir", "runs/monitor/snapshots"))
    log_probability = bool(monitor.get("log_probability_map", False))

    jsonl = JsonlWriter(jsonl_path) if jsonl_path else None
    csv_writer = CsvWriter(csv_path) if csv_path else None
    writer = None
    window = "Slim U-Net bladder monitor"

    stop_event = threading.Event()

    def request_stop(signum, frame) -> None:  # noqa: ARG001
        logger.info("Signal %s received; shutting down gracefully.", signum)
        stop_event.set()

    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.signal(sig, request_stop)
        except (ValueError, OSError):  # not on the main thread
            pass

    import cv2

    frame_times: list[float] = []
    last_time = [time.perf_counter()]
    processed = [0]

    def on_result(result: PipelineResult) -> None:
        nonlocal writer
        now = time.perf_counter()
        delta = now - last_time[0]
        last_time[0] = now
        if delta > 0:
            frame_times.append(delta)
            if len(frame_times) > 30:
                frame_times.pop(0)
        fps = len(frame_times) / sum(frame_times) if frame_times else None

        state = result.state
        if jsonl is not None:
            jsonl.write(state.to_dict(include_probability_map=log_probability))
        if csv_writer is not None:
            csv_writer.write(state.flat_record())

        canvas = renderer.render(result.frame.image, state, fps=fps, dropped=result.dropped_frames)

        if video_path:
            if writer is None:
                Path(video_path).parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                rate = source.frame_rate or 20.0
                writer = cv2.VideoWriter(
                    str(video_path), fourcc, rate, (canvas.shape[1], canvas.shape[0])
                )
                if not writer.isOpened():
                    logger.error("Could not open the video writer for %s; disabling it.", video_path)
                    writer = False  # type: ignore[assignment]
            if writer:
                writer.write(canvas)

        if pipeline.config.snapshot_on_validity_change and result.validity_changed:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            name = f"frame{result.frame.index:06d}_{'valid' if state.valid_for_control else 'invalid'}.png"
            cv2.imwrite(str(snapshot_dir / name), canvas)
            logger.info(
                "Validity changed to %s at frame %d: %s",
                state.valid_for_control, result.frame.index, state.rejection_reasons or "-",
            )

        if not headless:
            scale = float(monitor.get("display_scale", 1.0))
            shown = canvas
            if scale != 1.0:
                shown = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(window, shown)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                stop_event.set()

        processed[0] += 1

    try:
        states = pipeline.run(
            source, on_result=on_result, max_frames=args.max_frames, stop_event=stop_event
        )
    finally:
        if writer:
            writer.release()
        if jsonl is not None:
            jsonl.close()
        if csv_writer is not None:
            csv_writer.close()
        if not headless:
            cv2.destroyAllWindows()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    valid = sum(1 for state in states if state.valid_for_control)
    print(f"\nProcessed {len(states)} frame(s); {valid} passed the validity gate.")
    print("\n=== Latency (warm-up excluded) ===")
    print(pipeline.latency.to_text())
    if jsonl_path:
        print(f"JSONL log: {jsonl_path}")
    if csv_path:
        print(f"CSV log  : {csv_path}")
    if video_path and writer is not False:
        print(f"Annotated video: {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
