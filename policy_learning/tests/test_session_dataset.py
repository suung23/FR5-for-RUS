"""세션 로더 · 데이터셋 빌드 · torch Dataset."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from conftest import small_config
from rus_policy.dataset import OBS_VEC_DIM, OBS_VEC_NAMES, PolicyH5Dataset, assign_splits
from rus_policy.perception import STATE_DIM
from rus_policy.session import (
    IMU_COLUMNS, SessionRecord, load_session, read_sessions_manifest, write_sessions_manifest,
)


def test_load_session_layout(synthetic_sessions):
    manifest, records = synthetic_sessions
    s = load_session(manifest.parent / records[0].session_dir)
    assert s.frames.ndim == 3 and s.frames.dtype == np.uint8
    assert s.n_frames == s.frame_t_pc.size == s.frame_id.size
    assert np.all(np.diff(s.frame_t_pc) > 0)
    assert s.imu.acc.shape == (s.imu.n, 3) and s.imu.gyr.shape == (s.imu.n, 3)
    assert np.all(np.isfinite(s.imu.quaternion("chip")))
    assert 150 < s.imu.rate_hz < 260 and 6 < s.us_fps < 10
    # dev↔pc 시계: 기울기 ≈ 1, 잔차 ms 급
    a, b = s.imu.clock
    assert abs(b - 1.0) < 1e-3
    resid = s.imu.t_pc - s.imu.dev_to_pc(s.imu.t_dev)
    assert np.percentile(np.abs(resid), 95) < 0.01
    assert set(IMU_COLUMNS) - {"pc_unix", "dev_us", "ax", "ay", "az", "gx", "gy", "gz"} >= set(s.imu.columns)


def test_manifest_roundtrip(tmp_path):
    recs = [SessionRecord("a/b", "S1", "freehand", "train"), SessionRecord("c", "S2", "teleop", "")]
    p = tmp_path / "sessions.csv"
    write_sessions_manifest(p, recs)
    back = read_sessions_manifest(p, root=tmp_path)
    assert back[0].session_dir == str(tmp_path / "a/b") and back[1].source == "teleop" and back[1].split == ""
    with pytest.raises(ValueError):
        write_sessions_manifest(p, [SessionRecord("x", "S", "robot", "train")])
        read_sessions_manifest(p)


def test_assign_splits_subject_random_no_leak():
    cfg = small_config()
    cfg.split.strategy = "subject_random"
    recs = [SessionRecord(f"d{i}", f"S{i // 2}", "freehand", "") for i in range(10)]
    out = assign_splits(recs, cfg)
    assert set(out.values()) == {"train", "val", "test"}
    for i in range(0, 10, 2):
        assert out[f"d{i}"] == out[f"d{i + 1}"]          # 같은 피험자 → 같은 split
    cfg.split.strategy = "manifest"
    with pytest.raises(ValueError):
        assign_splits(recs, cfg)


def test_teleop_source_not_supported(synthetic_sessions, tmp_path):
    from rus_policy.dataset import build_dataset

    manifest, records = synthetic_sessions
    cfg = small_config()
    rec = SessionRecord(str(manifest.parent / records[0].session_dir), "S0", "teleop", "train")
    with pytest.raises(NotImplementedError):
        build_dataset(cfg, out_path=tmp_path / "x.h5", records=[rec])


def test_dataset_h5_layout(built_dataset):
    cfg = small_config()
    m, k = cfg.timing.obs_frames, cfg.timing.chunk_steps
    with h5py.File(built_dataset, "r") as fh:
        n = int(fh["schema"].attrs["n_samples"])
        assert n > 10
        assert fh["obs/frames"].shape == (n, m, 32, 32) and fh["obs/frames"].dtype == np.uint8
        assert fh["obs/state"].shape == (n, m, STATE_DIM)
        assert fh["obs/vec"].shape == (n, OBS_VEC_DIM)
        assert fh["label/P"].shape == (n, k + 1, 3) and fh["label/P6"].shape == (n, k + 1, 6)
        assert np.all(fh["label/P"][:, 0] == 0)
        assert np.all(fh["label/sigma_net"][...] > 0)
        # backend=none → Q 전부 무효, 토큰 −1
        assert not fh["label/Q_valid"][...].any()
        assert not fh["label/F_valid"][...].any()
        splits = fh["meta/split"][...].astype(str)
        assert set(splits) == {"train", "val", "test"}
        dt = fh["obs/frame_dt"][...]
        assert np.all(dt <= 0) and np.all(dt[fh["obs/frame_valid"][...]] >= -cfg.timing.obs_frame_max_age_s)
        meta = json.loads(fh["meta/json"][0])
        assert meta["token_at_anchor"] == -1 and "diag_zupt_v_end_raw_mm_s" in meta
        # Ã_{t−1}: 두 번째 샘플부터는 유효 (같은 세션, 공백 < 5 s)
        vec = fh["obs/vec"][...]
        assert vec[:, OBS_VEC_NAMES.index("prev_valid")].mean() > 0.5
        assert np.allclose(np.linalg.norm(vec[:, 5:8], axis=1), 1.0, atol=1e-5)   # 중력 단위벡터
        assert np.all(vec[:, OBS_VEC_NAMES.index("has_force")] == 0)
        sessions = json.loads(fh["schema"].attrs["sessions"])
        assert len(sessions) == 3 and all(s["samples"] > 0 for s in sessions)


def test_torch_dataset_items(built_dataset):
    import torch

    ds = PolicyH5Dataset(built_dataset, split="train", augment=True)
    assert len(ds) > 0
    item = ds[0]
    assert item["frames"].dtype == torch.float32 and 0.0 <= item["frames"].min() and item["frames"].max() <= 1.0
    assert item["P"].shape[-1] == 3 and item["Q_valid"].dtype == torch.bool
    assert torch.isfinite(item["state"]).all() and torch.isfinite(item["Q"]).all()
    va = PolicyH5Dataset(built_dataset, split="val")
    assert set(ds.indices).isdisjoint(set(va.indices))
    assert ds.summary()["freehand"] == len(ds)


def test_manifest_skips_comment_lines(tmp_path):
    p = tmp_path / "sessions.csv"
    lines = ["session_dir,subject,source,split,note", "# comment,,,,", "x,S,freehand,train,", ""]
    p.write_text(chr(10).join(lines), encoding="utf-8")
    recs = read_sessions_manifest(p, root=tmp_path)
    assert len(recs) == 1 and recs[0].subject == "S"


def test_config_override_coercion():
    from rus_policy.config import load_config

    c = load_config(None, ["train.lr=3e-4", "train.epochs=7", "train.amp=true", "model.frame_channels=[8,16]"])
    assert isinstance(c.train.lr, float) and c.train.lr == 3e-4 and c.train.epochs == 7 and c.train.amp is True
    assert c.model.frame_channels == [8, 16]
    with pytest.raises(ValueError):
        load_config(None, ["train.epochs=many"])
    with pytest.raises(ValueError):
        load_config(None, ["nosuch.key=1"])
