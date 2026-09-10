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


def test_q_area_label_present_and_masked_without_perception(built_dataset):
    import h5py

    with h5py.File(built_dataset, "r") as fh:
        assert "label/Q_area" in fh and "label/Q_area_valid" in fh
        qa_valid = np.asarray(fh["label/Q_area_valid"][...])
        assert qa_valid.shape[1] == fh["label/Q"].shape[1]
        # backend=none → 마스크가 없으니 전부 False
        assert not qa_valid.any()


def test_observation_anchor_augmentation(synthetic_sessions, tmp_path):
    """정지 A 안의 여러 시각을 '지금' 으로 잡으면 같은 라벨에 관측이 여러 개 붙는다 (2026-09-08 §8-2)."""
    import json

    import h5py

    from rus_policy.dataset import build_dataset
    from rus_policy.session import read_sessions_manifest

    manifest, _ = synthetic_sessions
    records = read_sessions_manifest(manifest, root=manifest.parent)
    cfg1 = small_config()
    r1 = build_dataset(cfg1, out_path=tmp_path / "one.h5", records=records)
    cfg3 = small_config(**{"dataset.obs_anchor_samples": 3})
    r3 = build_dataset(cfg3, out_path=tmp_path / "three.h5", records=records)
    assert r3["n_samples"] > r1["n_samples"]
    assert all(s["segments_used"] == s1["segments_used"] for s, s1 in zip(r3["sessions"], r1["sessions"]))
    with h5py.File(tmp_path / "three.h5", "r") as fh:
        metas = [json.loads(m) for m in np.asarray(fh["meta/json"][...]).astype(str)]
        offs = np.array([m["obs_offset_s"] for m in metas])
        P = np.asarray(fh["label/P"][...])
    assert (offs > 0).any() and offs.min() >= 0.0
    # 같은 구간의 증강 샘플은 라벨이 동일하다
    keys = [(m["segment_index"], m["t_anchor_pc"]) for m in metas]
    seen = {}
    for kk, p in zip(keys, P):
        if kk in seen:
            assert np.allclose(seen[kk], p)
        seen[kk] = p
    assert "chunk_truncated_fraction" in r3["sessions"][0]


def test_torch_dataset_items(built_dataset):
    import torch

    ds = PolicyH5Dataset(built_dataset, split="train", augment=True)
    assert len(ds) > 0
    item = ds[0]
    assert item["frames"].dtype == torch.float32 and 0.0 <= item["frames"].min() and item["frames"].max() <= 1.0
    assert item["P"].shape[-1] == 6 and item["Q_valid"].dtype == torch.bool   # 6 자유도 전체
    assert item["sigma_net"].shape[-1] == 6 and item["sigma_shape"].shape[-1] == 6
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
