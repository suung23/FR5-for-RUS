"""End-to-end pipeline tests: config, training, inference, monitoring, export.

These are smoke tests: they verify that the pieces fit together and that the
contracts between them hold, not that a 2-epoch model is accurate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rus_perception.control.features import FeatureExtractionConfig
from rus_perception.data.manifest import load_manifest
from rus_perception.inference.predictor import Predictor, PredictorConfig
from rus_perception.inference.realtime import FrameGrabber, RealtimeConfig, RealtimePipeline
from rus_perception.inference.sources import Frame, FrameSource, ImageDirectorySource, open_source
from rus_perception.inference.visualization import MonitorRenderer, RollingHistory
from rus_perception.models.registry import build_model
from rus_perception.training.trainer import Trainer, build_optimizer, build_scheduler, resolve_device
from rus_perception.utils.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from rus_perception.utils.config import Config, ConfigError, load_config, merge_dicts
from rus_perception.utils.logging_utils import CsvWriter, JsonlWriter
from rus_perception.utils.seeding import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


# -- configuration ---------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "standard_unet_baseline",
        "slim_unet_paper",
        "slim_unet_production",
        "slim_unet_temporal",
        "realtime_monitor",
    ],
)
def test_shipped_configs_load_validate_and_build_a_model(name: str) -> None:
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    model = build_model(config.section("model"))
    size = config.get("model.input_size")
    with torch.no_grad():
        output = model(torch.zeros(1, int(config.get("model.input_channels", 1)), 64, 64))
    assert output.shape[-2:] == (64, 64)
    assert size is not None and len(size) == 2


def test_config_dotted_access_and_overrides() -> None:
    config = load_config(CONFIG_DIR / "slim_unet_temporal.yaml")
    assert config.get("loss.temporal.lambda_temp_pixel") == pytest.approx(0.2)
    assert config.get("does.not.exist", "fallback") == "fallback"
    overridden = config.with_overrides({"loss": {"temporal": {"lambda_temp_pixel": 0.9}}})
    assert overridden.get("loss.temporal.lambda_temp_pixel") == pytest.approx(0.9)
    assert config.get("loss.temporal.lambda_temp_pixel") == pytest.approx(0.2)


def test_config_validation_rejects_broken_settings(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  name: not_a_model\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="model.name must be one of"):
        load_config(path)

    path.write_text("nonsense_section:\n  a: 1\nmodel:\n  name: slim_unet\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown top-level config section"):
        load_config(path)

    path.write_text(
        "model:\n  name: slim_unet\nloss:\n  bce_weight: 0\n  dice_weight: 0\n  jaccard_weight: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="no segmentation supervision"):
        load_config(path)


def test_config_rejects_a_warmup_longer_than_training(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model:\n  name: slim_unet\n"
        "train:\n  epochs: 3\n"
        "loss:\n  temporal:\n    enabled: true\n    warmup_epochs: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="never be active"):
        load_config(path)


def test_config_inheritance_merges_without_leaking_model_keys() -> None:
    """slim_unet_temporal inherits training settings but keeps a slim model section."""
    temporal = load_config(CONFIG_DIR / "slim_unet_temporal.yaml")
    assert temporal.get("model.name") == "slim_unet"
    assert "base_channels" not in temporal.section("model")
    assert temporal.get("loss.temporal.enabled") is True
    assert temporal.get("postprocess.threshold") is not None


def test_merge_dicts_is_recursive_and_non_destructive() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": [1, 2]}
    merged = merge_dicts(base, {"a": {"c": 3}, "d": [9]})
    assert merged == {"a": {"b": 1, "c": 3}, "d": [9]}
    assert base["a"]["c"] == 2


def test_config_require_raises_for_missing_values() -> None:
    config = Config({"model": {"name": "slim_unet"}})
    assert config.require("model.name") == "slim_unet"
    with pytest.raises(ConfigError, match="Required config value"):
        config.require("train.epochs")


# -- checkpoints -----------------------------------------------------------
def test_checkpoint_round_trip_carries_reproducibility_metadata(tmp_path) -> None:
    model = build_model({"name": "slim_unet", "channels": [8, 16]})
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metadata = CheckpointMetadata(
        epoch=7,
        best_metric=0.83,
        best_metric_name="val/selection_score",
        model_version="slim_unet-paper",
        seed=42,
        config={"model": {"name": "slim_unet", "channels": [8, 16]}},
    )
    path = save_checkpoint(tmp_path / "best.pt", model, metadata, optimizer=optimizer)
    payload = load_checkpoint(path)

    assert payload["epoch"] == 7
    assert payload["best_metric"] == pytest.approx(0.83)
    assert payload["model_version"] == "slim_unet-paper"
    assert payload["seed"] == 42
    assert payload["config"]["model"]["name"] == "slim_unet"
    assert payload["optimizer_state"] is not None
    assert len(payload["checkpoint_id"]) == 12
    assert "git_commit" in payload

    rebuilt = build_model(payload["config"]["model"])
    rebuilt.load_state_dict(payload["model_state"])


def test_loading_a_non_checkpoint_file_fails_clearly(tmp_path) -> None:
    path = tmp_path / "junk.pt"
    torch.save({"weights": 1}, path)
    with pytest.raises(ValueError, match="not a checkpoint"):
        load_checkpoint(path)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "absent.pt")


# -- optimiser / scheduler construction ------------------------------------
@pytest.mark.parametrize("name", ["adam", "adamw", "sgd", "rmsprop"])
def test_supported_optimizers(name: str) -> None:
    model = build_model({"name": "slim_unet", "channels": [4, 8]})
    optimizer = build_optimizer(model.parameters(), {"name": name, "lr": 1e-3})
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


@pytest.mark.parametrize("name,on_metric", [("none", False), ("cosine", False), ("step", False), ("plateau", True)])
def test_supported_schedulers(name: str, on_metric: bool) -> None:
    model = build_model({"name": "slim_unet", "channels": [4, 8]})
    optimizer = build_optimizer(model.parameters(), {"name": "adam"})
    scheduler, steps_on_metric = build_scheduler(optimizer, {"name": name}, epochs=10)
    assert steps_on_metric is on_metric
    assert (scheduler is None) == (name == "none")


def test_unknown_optimizer_and_scheduler_are_rejected() -> None:
    model = build_model({"name": "slim_unet", "channels": [4, 8]})
    with pytest.raises(ConfigError, match="Unknown optimizer"):
        build_optimizer(model.parameters(), {"name": "lbfgs_plus"})
    with pytest.raises(ConfigError, match="Unknown scheduler"):
        build_scheduler(torch.optim.Adam(model.parameters()), {"name": "magic"}, 5)


def test_resolve_device_falls_back_to_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")


# -- CPU training smoke test ----------------------------------------------
def make_training_config(manifest_path: Path, output_dir: Path, temporal: bool) -> Config:
    return Config(
        {
            "experiment": {"name": "smoke", "seed": 0},
            "model": {
                "name": "slim_unet",
                "input_channels": 1,
                "output_channels": 1,
                "channels": [8, 16, 32],
                "input_size": [64, 64],
                "dropout": 0.0,
            },
            "data": {
                "manifest": str(manifest_path),
                "image_size": [64, 64],
                "temporal_interval": 1,
                "require_labeled_current": True,
            },
            "augmentation": {"enabled": False},
            "loss": {
                "bce_weight": 1.0,
                "dice_weight": 1.0,
                "jaccard_weight": 1.0,
                "temporal": {
                    "enabled": temporal,
                    "lambda_temp_pixel": 0.2,
                    "lambda_temp_control": 0.05,
                    "temporal_warmup_epochs": 0,
                    "min_reliable_ratio": 0.0,
                },
            },
            "flow": {"backend": "identity", "params": {"warn": False}},
            "train": {
                "epochs": 2,
                "batch_size": 2,
                "num_workers": 0,
                "device": "cpu",
                "amp": False,
                "grad_clip": 1.0,
                "checkpoint_dir": str(output_dir),
                "optimizer": {"name": "adam", "lr": 1e-3},
                "scheduler": {"name": "none"},
                "model_selection": {
                    "spatial_weight": 0.8 if temporal else 1.0,
                    "temporal_weight": 0.2 if temporal else 0.0,
                },
            },
            "postprocess": {"threshold": 0.5},
        }
    ).validate()


@pytest.mark.parametrize("temporal", [False, True])
def test_cpu_training_smoke(synthetic_dataset, tmp_path, temporal: bool) -> None:
    manifest_path, manifest = synthetic_dataset
    output_dir = tmp_path / ("temporal" if temporal else "spatial")
    config = make_training_config(manifest_path, output_dir, temporal)

    seed_everything(0)
    trainer = Trainer(
        config=config,
        model=build_model(config.section("model")),
        train_manifest=manifest.filter(split="train"),
        val_manifest=manifest.filter(split="val"),
        output_dir=output_dir,
    )
    state = trainer.fit()

    assert len(state.history) == 2
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "best.pt").is_file()
    for record in state.history:
        assert np.isfinite(record["loss/total"])
        assert 0.0 <= record["val/dice"] <= 1.0
        assert "val/selection_score" in record
    if temporal:
        assert "loss/temporal_pixel" in state.history[0]
        assert "temporal/reliable_pixel_ratio" in state.history[0]


def test_model_selection_never_uses_training_loss_alone(synthetic_dataset, tmp_path) -> None:
    manifest_path, manifest = synthetic_dataset
    config = make_training_config(manifest_path, tmp_path / "sel", temporal=True)
    trainer = Trainer(
        config=config,
        model=build_model(config.section("model")),
        train_manifest=manifest.filter(split="train"),
        val_manifest=manifest.filter(split="val"),
        output_dir=tmp_path / "sel",
    )
    # Spatial accuracy must dominate the selection score.
    assert trainer.selection_spatial_weight > trainer.selection_temporal_weight
    high_spatial = trainer.selection_score({"val/dice": 0.9, "val/temporal_iou": 0.1})
    high_temporal = trainer.selection_score({"val/dice": 0.1, "val/temporal_iou": 0.9})
    assert high_spatial > high_temporal


def test_training_rejects_a_zero_spatial_selection_weight(synthetic_dataset, tmp_path) -> None:
    manifest_path, manifest = synthetic_dataset
    config = make_training_config(manifest_path, tmp_path / "bad", temporal=True)
    config.set("train.model_selection.spatial_weight", 0.0)
    with pytest.raises(ConfigError, match="must be > 0"):
        Trainer(
            config=config,
            model=build_model(config.section("model")),
            train_manifest=manifest.filter(split="train"),
            val_manifest=manifest.filter(split="val"),
            output_dir=tmp_path / "bad",
        )


def test_training_can_resume_from_a_checkpoint(synthetic_dataset, tmp_path) -> None:
    manifest_path, manifest = synthetic_dataset
    output_dir = tmp_path / "resume"
    config = make_training_config(manifest_path, output_dir, temporal=False)

    first = Trainer(
        config=config,
        model=build_model(config.section("model")),
        train_manifest=manifest.filter(split="train"),
        val_manifest=manifest.filter(split="val"),
        output_dir=output_dir,
    )
    first.fit()

    config.set("train.epochs", 3)
    second = Trainer(
        config=config,
        model=build_model(config.section("model")),
        train_manifest=manifest.filter(split="train"),
        val_manifest=manifest.filter(split="val"),
        output_dir=output_dir,
    )
    second.resume(output_dir / "last.pt")
    assert second.state.epoch == 2
    state = second.fit()
    assert state.history[-1]["epoch"] == 2


# -- inference -------------------------------------------------------------
def test_single_frame_inference_accepts_exactly_one_frame() -> None:
    """The runtime contract: one frame in, one probability map out."""
    predictor = Predictor(
        build_model({"name": "slim_unet", "channels": [8, 16]}),
        PredictorConfig(input_size=(64, 64), device="cpu"),
    )
    frame = (np.random.default_rng(0).random((80, 70)) * 255).astype(np.uint8)
    probability, latencies = predictor.predict_probability(frame)

    assert probability.shape == (80, 70), "output must be restored to the input geometry"
    assert 0.0 <= probability.min() and probability.max() <= 1.0
    assert set(latencies) == {
        "preprocessing_latency_ms",
        "inference_latency_ms",
        "postprocessing_latency_ms",
    }

    with pytest.raises(ValueError, match="2-D or 3-D"):
        predictor.preprocess(np.zeros((2, 3, 4, 5)))


def test_predictor_produces_a_complete_control_state(disc_image) -> None:
    predictor = Predictor(
        build_model({"name": "slim_unet", "channels": [8, 16]}),
        PredictorConfig(input_size=(64, 64), device="cpu"),
        feature_config=FeatureExtractionConfig(),
        model_version="slim-test",
        checkpoint_identifier="abc123",
    )
    state = predictor.predict_control_state(disc_image, metadata={"frame_id": "f1", "timestamp": 0.5})
    assert state.frame_id == "f1"
    assert state.model_version == "slim-test"
    assert state.checkpoint_id == "abc123"
    assert state.end_to_end_latency_ms > 0
    json.dumps(state.to_dict())


def test_predictor_can_be_rebuilt_from_a_checkpoint(synthetic_dataset, tmp_path) -> None:
    manifest_path, manifest = synthetic_dataset
    output_dir = tmp_path / "ckpt"
    config = make_training_config(manifest_path, output_dir, temporal=False)
    trainer = Trainer(
        config=config,
        model=build_model(config.section("model")),
        train_manifest=manifest.filter(split="train"),
        val_manifest=manifest.filter(split="val"),
        output_dir=output_dir,
    )
    trainer.fit()

    predictor = Predictor.from_checkpoint(output_dir / "best.pt")
    frame = (np.random.default_rng(1).random((64, 64)) * 255).astype(np.uint8)
    state = predictor.predict_control_state(frame, metadata={"frame_id": "x"})
    assert state.checkpoint_id != "unknown"
    assert state.probability_map is not None


# -- sequential evaluation -------------------------------------------------
def test_short_sequential_evaluation_preserves_order(synthetic_dataset) -> None:
    from rus_perception.metrics.temporal import TemporalFrameRecord, compute_sequence_temporal_metrics

    _, manifest = synthetic_dataset
    predictor = Predictor(
        build_model({"name": "slim_unet", "channels": [8, 16]}),
        PredictorConfig(input_size=(64, 64), device="cpu", restore_original_size=False),
    )
    (_, records) = sorted(manifest.filter(split="test").sequences().items())[0]
    assert [r.frame_index for r in records] == sorted(r.frame_index for r in records)

    from rus_perception.data.io import load_grayscale

    states = []
    previous = None
    temporal_records = []
    for record in records:
        image = load_grayscale(manifest.resolve(record.image_path))
        state = predictor.predict_control_state(
            image, previous_state=previous, metadata={"frame_id": record.frame_id}
        )
        temporal_records.append(
            TemporalFrameRecord(
                frame_id=record.frame_id,
                mask=state.binary_mask,
                warped_previous_mask=None if previous is None else previous.binary_mask,
                area_ratio=state.mask_area_ratio,
                valid_for_control=state.valid_for_control,
            )
        )
        states.append(state)
        previous = state

    metrics = compute_sequence_temporal_metrics(temporal_records, "seq")
    assert metrics.num_frames == len(records)
    assert metrics.num_transitions == len(records) - 1
    assert 0.0 <= metrics.invalid_control_frame_rate <= 1.0


# -- real-time pipeline ----------------------------------------------------
class ListSource(FrameSource):
    """A deterministic in-memory frame source for tests."""

    name = "list"

    def __init__(self, frames: list[np.ndarray], live: bool = False) -> None:
        self.frames = frames
        self._live = live
        self.released = False

    @property
    def is_live(self) -> bool:
        return self._live

    def __iter__(self):
        for index, image in enumerate(self.frames):
            yield Frame(index=index, timestamp=index * 0.05, image=image, source_id="test")

    def release(self) -> None:
        self.released = True


def test_realtime_pipeline_processes_every_frame_of_a_file_source() -> None:
    rng = np.random.default_rng(0)
    frames = [rng.random((64, 64)).astype(np.float32) for _ in range(12)]
    pipeline = RealtimePipeline(
        Predictor(
            build_model({"name": "slim_unet", "channels": [8, 16]}),
            PredictorConfig(input_size=(64, 64), device="cpu"),
        ),
        RealtimeConfig(max_queue_size=2, warmup_frames=0, enable_temporal=True),
    )
    source = ListSource(frames)
    states = pipeline.run(source)

    assert len(states) == len(frames), "a replayable source must not drop frames"
    assert source.released, "the source must be released even on the happy path"
    assert states[0].temporal_warped_iou is None, "the first frame has no predecessor"
    assert states[1].temporal_warped_iou is not None


def test_frame_grabber_queue_stays_bounded_and_drops_stale_frames() -> None:
    """Under back-pressure a live source must drop, not accumulate latency."""
    frames = [np.zeros((8, 8), np.float32) for _ in range(200)]
    grabber = FrameGrabber(ListSource(frames, live=True), max_queue_size=3, drop_stale=True)
    grabber.start()

    consumed = 0
    for _ in grabber:
        consumed += 1
        assert grabber.queue_size <= 3, "the capture queue grew beyond its bound"
    grabber.stop()

    assert consumed + grabber.dropped <= len(frames) + 1
    assert consumed >= 1


def test_realtime_pipeline_honours_max_frames_and_stop_event() -> None:
    import threading

    frames = [np.zeros((64, 64), np.float32) for _ in range(20)]
    predictor = Predictor(
        build_model({"name": "slim_unet", "channels": [8, 16]}),
        PredictorConfig(input_size=(64, 64), device="cpu"),
    )
    pipeline = RealtimePipeline(predictor, RealtimeConfig(warmup_frames=0, threaded_capture=False))
    assert len(pipeline.run(ListSource(frames), max_frames=5)) == 5

    stop = threading.Event()
    stop.set()
    assert pipeline.run(ListSource(frames), stop_event=stop) == []


def test_realtime_pipeline_releases_the_source_on_error() -> None:
    class ExplodingSource(ListSource):
        def __iter__(self):
            yield Frame(0, 0.0, np.zeros((64, 64), np.float32), "boom")
            raise RuntimeError("stream failed")

    source = ExplodingSource([])
    pipeline = RealtimePipeline(
        Predictor(
            build_model({"name": "slim_unet", "channels": [8, 16]}),
            PredictorConfig(input_size=(64, 64), device="cpu"),
        ),
        RealtimeConfig(threaded_capture=False, warmup_frames=0),
    )
    with pytest.raises(RuntimeError, match="stream failed"):
        pipeline.run(source)
    assert source.released


def test_image_directory_source_reads_in_name_order(synthetic_dataset) -> None:
    manifest_path, _ = synthetic_dataset
    directory = manifest_path.parent / "images"
    with ImageDirectorySource(directory, frame_rate=10.0) as source:
        frames = list(source)
    assert len(frames) > 0
    assert [f.index for f in frames] == sorted(f.index for f in frames)
    assert source.is_live is False
    assert frames[1].timestamp == pytest.approx(0.1)


def test_open_source_dispatch_and_errors(synthetic_dataset, tmp_path) -> None:
    manifest_path, _ = synthetic_dataset
    assert open_source(str(manifest_path.parent / "images")).name == "images"
    with pytest.raises(FileNotFoundError, match="neither an existing file"):
        open_source(str(tmp_path / "nowhere"))


def test_monitor_renderer_produces_a_canvas(disc_probability, disc_image) -> None:
    from rus_perception.control.features import extract_control_state

    renderer = MonitorRenderer(history=RollingHistory(maxlen=10))
    state = extract_control_state(disc_probability, disc_image, metadata={"frame_id": "f"})
    canvas = renderer.render(disc_image, state, fps=30.0, dropped=0)

    assert canvas.ndim == 3 and canvas.shape[2] == 3
    assert canvas.shape[1] == 4 * 360, "four side-by-side panels expected"
    assert renderer.status_text(state) in ("VALID", "MARGINAL", "INVALID")


def test_rolling_history_is_bounded(disc_probability) -> None:
    from rus_perception.control.features import extract_control_state

    history = RollingHistory(maxlen=5)
    state = extract_control_state(disc_probability)
    for _ in range(50):
        history.append(state)
    assert all(len(series) == 5 for series in history.series.values())
    assert history.total_samples() == 5


# -- logging ---------------------------------------------------------------
def test_jsonl_and_csv_writers_produce_readable_files(tmp_path, disc_probability) -> None:
    from rus_perception.control.features import extract_control_state

    state = extract_control_state(disc_probability, metadata={"frame_id": "f0"})
    jsonl_path = tmp_path / "log.jsonl"
    csv_path = tmp_path / "log.csv"

    with JsonlWriter(jsonl_path) as jsonl, CsvWriter(csv_path) as csv_writer:
        for index in range(3):
            state.frame_id = f"f{index}"
            jsonl.write(state.to_dict())
            csv_writer.write(state.flat_record())

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["frame_id"] == "f0"

    import csv as csv_module

    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    assert len(rows) == 3
    assert "control_quality_score" in rows[0]
    assert "valid_for_control" in rows[0]


def test_writers_reject_use_after_close(tmp_path) -> None:
    writer = JsonlWriter(tmp_path / "x.jsonl")
    writer.close()
    with pytest.raises(RuntimeError, match="already closed"):
        writer.write({"a": 1})


# -- CLI surface -----------------------------------------------------------
def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / name), *args],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
    )


@pytest.mark.parametrize(
    "script",
    [
        "train.py",
        "evaluate.py",
        "infer.py",
        "live_monitor.py",
        "precompute_flow.py",
        "export_onnx.py",
        "benchmark.py",
        "architecture_report.py",
        "make_synthetic_dataset.py",
    ],
)
def test_every_script_exposes_a_working_help(script: str) -> None:
    """Guards against README commands drifting away from the actual CLI."""
    result = run_script(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_architecture_report_cli_reports_exact_matches() -> None:
    result = run_script("architecture_report.py")
    assert result.returncode == 0, result.stderr
    assert "8,635,809" in result.stdout
    assert "4,705,377" in result.stdout
    assert result.stdout.count("EXACT MATCH") == 2


def test_onnx_export_matches_pytorch(synthetic_dataset, tmp_path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")

    manifest_path, manifest = synthetic_dataset
    output_dir = tmp_path / "onnx_ckpt"
    config = make_training_config(manifest_path, output_dir, temporal=False)
    trainer = Trainer(
        config=config,
        model=build_model(config.section("model")),
        train_manifest=manifest.filter(split="train"),
        val_manifest=manifest.filter(split="val"),
        output_dir=output_dir,
    )
    trainer.fit()

    onnx_path = tmp_path / "model.onnx"
    result = run_script(
        "export_onnx.py",
        "--checkpoint", str(output_dir / "best.pt"),
        "--output", str(onnx_path),
        "--input-size", "64", "64",
    )
    assert result.returncode == 0, result.stderr
    assert onnx_path.is_file()

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 1, 64, 64), np.float32)
    exported = session.run(["logits"], {"input": dummy})[0]

    model = build_model(config.section("model"))
    model.load_state_dict(load_checkpoint(output_dir / "best.pt")["model_state"])
    model.eval()
    with torch.no_grad():
        reference = model(torch.from_numpy(dummy)).numpy()
    assert np.abs(reference - exported).max() < 1e-4


def test_live_monitor_smoke_on_an_image_directory(synthetic_dataset, tmp_path) -> None:
    """Headless end-to-end run of the monitoring application."""
    manifest_path, manifest = synthetic_dataset
    output_dir = tmp_path / "monitor_ckpt"
    config = make_training_config(manifest_path, output_dir, temporal=False)
    trainer = Trainer(
        config=config,
        model=build_model(config.section("model")),
        train_manifest=manifest.filter(split="train"),
        val_manifest=manifest.filter(split="val"),
        output_dir=output_dir,
    )
    trainer.fit()

    jsonl_path = tmp_path / "monitor.jsonl"
    csv_path = tmp_path / "monitor.csv"
    result = run_script(
        "live_monitor.py",
        "--config", str(CONFIG_DIR / "realtime_monitor.yaml"),
        "--checkpoint", str(output_dir / "best.pt"),
        "--source", str(manifest_path.parent / "images"),
        "--headless",
        "--device", "cpu",
        "--max-frames", "6",
        "--jsonl", str(jsonl_path),
        "--csv", str(csv_path),
        "--set", "model.input_size=[64,64]",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Processed 6 frame(s)" in result.stdout

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    record = json.loads(lines[0])
    for key in ("valid_for_control", "control_quality_score", "rejection_reasons", "center_error_x"):
        assert key in record
    assert csv_path.read_text(encoding="utf-8").count("\n") == 7  # header + 6 rows


def test_end_to_end_cli_workflow(tmp_path) -> None:
    """make dataset -> precompute flow -> train -> evaluate, exactly as documented."""
    data_dir = tmp_path / "data"
    result = run_script(
        "make_synthetic_dataset.py",
        "--output-dir", str(data_dir),
        "--patients", "4", "--frames", "5", "--size", "64", "64", "--seed", "1",
    )
    assert result.returncode == 0, result.stderr
    manifest_path = data_dir / "manifest.csv"
    assert manifest_path.is_file()

    result = run_script(
        "precompute_flow.py",
        "--manifest", str(manifest_path),
        "--backend", "farneback",
        "--output-dir", str(data_dir / "flow"),
        "--update-manifest",
    )
    assert result.returncode == 0, result.stderr
    manifest = load_manifest(manifest_path)
    assert any(record.flow_backward_path for record in manifest)

    checkpoint_dir = tmp_path / "ckpt"
    result = run_script(
        "train.py",
        "--config", str(CONFIG_DIR / "slim_unet_temporal.yaml"),
        "--manifest", str(manifest_path),
        "--output-dir", str(checkpoint_dir),
        "--epochs", "2", "--device", "cpu",
        "--set", "model.input_size=[64,64]",
        "--set", "data.image_size=[64,64]",
        "--set", "train.batch_size=2",
        "--set", "train.num_workers=0",
        "--set", "train.amp=false",
        "--set", "loss.temporal.temporal_warmup_epochs=0",
        "--set", "loss.temporal.ramp_epochs=0",
    )
    assert result.returncode == 0, result.stderr
    assert (checkpoint_dir / "best.pt").is_file()
    assert (checkpoint_dir / "dataset_statistics.json").is_file()
    assert (checkpoint_dir / "architecture_report.json").is_file()

    eval_dir = tmp_path / "eval"
    result = run_script(
        "evaluate.py",
        "--config", str(CONFIG_DIR / "slim_unet_temporal.yaml"),
        "--checkpoint", str(checkpoint_dir / "best.pt"),
        "--manifest", str(manifest_path),
        "--output-dir", str(eval_dir),
        "--device", "cpu",
        "--set", "model.input_size=[64,64]",
        "--set", "data.image_size=[64,64]",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((eval_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["spatial"]["overall"]["num_frames"] > 0
    assert "warped_temporal_iou" in payload["temporal"]
    assert "stages" in payload["latency"]
    assert "Temporal stability" in result.stdout


def test_a_skipped_step_still_updates_the_grad_scaler(synthetic_dataset, tmp_path) -> None:
    """A non-finite gradient must not leave the AMP scaler in its unscaled state.

    ``unscale_()`` may be called at most once per optimizer between two
    ``update()`` calls. Skipping the update when the gradient norm is non-finite
    made the *next* iteration's ``unscale_()`` raise, which under AMP killed
    training mid-run as soon as one fp16 gradient overflowed.
    """
    manifest_path, manifest = synthetic_dataset
    output_dir = tmp_path / "scaler"
    config = make_training_config(manifest_path, output_dir, temporal=False)
    config.set("train.grad_clip", 1.0)

    trainer = Trainer(
        config=config,
        model=build_model(config.section("model")),
        train_manifest=manifest.filter(split="train"),
        val_manifest=None,
        output_dir=output_dir,
    )

    calls: list[str] = []
    scaler = trainer.scaler

    class SpyScaler:
        """Records the scaler protocol and enforces the unscale_/update contract."""

        def scale(self, loss):
            return scaler.scale(loss)

        def unscale_(self, optimizer):
            if calls and calls[-1] == "unscale_":
                raise RuntimeError(
                    "unscale_() has already been called on this optimizer since "
                    "the last update()."
                )
            calls.append("unscale_")
            return scaler.unscale_(optimizer)

        def step(self, optimizer):
            calls.append("step")
            return scaler.step(optimizer)

        def update(self):
            calls.append("update")
            return scaler.update()

    trainer.scaler = SpyScaler()
    # Every step overflows, so every step takes the skip path.
    trainer_module = sys.modules[Trainer.__module__]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        trainer_module.torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(float("inf")),
    )
    try:
        with pytest.raises(RuntimeError, match="No training batches"):
            trainer.train_epoch()
    finally:
        monkey.undo()

    assert calls, "the scaler was never exercised"
    assert "step" not in calls, "an overflowing step must not update the weights"
    # unscale_ and update strictly alternate: no unscale_ is left dangling.
    assert calls == ["unscale_", "update"] * (len(calls) // 2)
