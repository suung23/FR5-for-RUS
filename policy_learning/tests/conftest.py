"""공용 fixture. 전부 합성 데이터 — 실제 세션·체크포인트·네트워크 불필요."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from rus_policy.config import PolicyConfig  # noqa: E402
from rus_policy.session import SessionRecord, write_sessions_manifest  # noqa: E402
from rus_policy.synth import SynthConfig, generate_session  # noqa: E402


def small_config(**overrides) -> PolicyConfig:
    """CPU 에서 몇 초에 도는 작은 모델 설정."""
    cfg = PolicyConfig()
    cfg.perception.backend = "none"
    cfg.perception.frame_size = [32, 32]
    cfg.timing.obs_frames = 6
    cfg.timing.chunk_steps = 8
    cfg.model.d_model = 32
    cfg.model.n_heads = 4
    cfg.model.encoder_layers = 1
    cfg.model.decoder_layers = 1
    cfg.model.dim_feedforward = 64
    cfg.model.z_dim = 8
    cfg.model.frame_channels = [8, 16]
    cfg.model.frame_input_size = [32, 32]
    cfg.model.q_head_hidden = 32
    cfg.train.batch_size = 4
    cfg.train.epochs = 2
    cfg.train.z_samples_eval = 4
    cfg.train.log_every = 0
    cfg.train.num_workers = 0
    for k, v in overrides.items():
        section, key = k.split(".")
        setattr(getattr(cfg, section), key, v)
    return cfg


@pytest.fixture(scope="session")
def synthetic_sessions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[SessionRecord]]:
    root = tmp_path_factory.mktemp("sessions")
    records = []
    for i, split in enumerate(("train", "val", "test")):
        cfg = SynthConfig(duration_s=25.0, seed=10 + i, image_size=64, accel_noise=0.01,
                          t0_unix=1_700_000_000.0 + 1000.0 * i)
        d = generate_session(root, cfg, name=f"synth_{i}")
        records.append(SessionRecord(session_dir=d.name, subject=f"S{i}", source="freehand", split=split))
    manifest = root / "sessions.csv"
    write_sessions_manifest(manifest, records)
    return manifest, records


@pytest.fixture(scope="session")
def built_dataset(synthetic_sessions, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from rus_policy.dataset import build_dataset
    from rus_policy.session import read_sessions_manifest

    manifest, _ = synthetic_sessions
    cfg = small_config()
    out = tmp_path_factory.mktemp("dataset") / "dataset.h5"
    records = read_sessions_manifest(manifest, root=manifest.parent)
    result = build_dataset(cfg, out_path=out, records=records)
    assert result["n_samples"] > 0
    return out
