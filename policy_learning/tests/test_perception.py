"""지각 특징 · 액션 토큰 · U-Net 백엔드 배선 (임시 무작위 slim_unet 체크포인트)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from conftest import small_config
from rus_policy.config import UNET_ROOT
from rus_policy.perception import (
    STATE_DIM, STATE_FEATURE_NAMES, TOKEN_CHECK_FILLING, TOKEN_HOLD, TOKEN_MOVE_LEFT, TOKEN_MOVE_RIGHT,
    PerceptionResult, action_token, apply_frame_transform, control_state_to_vector, empty_result,
)


def test_action_token_rule():
    assert action_token(None, None, 8.0) == TOKEN_CHECK_FILLING
    assert action_token(20.0, 0.0, 8.0) == TOKEN_CHECK_FILLING       # contrast ≤ 0 → 게이트
    assert action_token(20.0, -0.1, 8.0) == TOKEN_CHECK_FILLING
    assert action_token(3.0, 0.3, 8.0) == TOKEN_HOLD                  # |ê| < deadband
    assert action_token(-7.9, 0.3, 8.0) == TOKEN_HOLD
    assert action_token(12.0, 0.3, 8.0) == TOKEN_MOVE_LEFT            # ê = A − ĉ > 0
    assert action_token(-12.0, 0.3, 8.0) == TOKEN_MOVE_RIGHT


def test_frame_transform():
    f = np.arange(12, dtype=np.uint8).reshape(1, 3, 4)
    assert apply_frame_transform(f, "none") is f
    assert apply_frame_transform(f, "rot90_cw").shape == (1, 4, 3)
    assert np.array_equal(apply_frame_transform(f, "flip_h")[0, 0], f[0, 0, ::-1])
    assert np.array_equal(apply_frame_transform(apply_frame_transform(f, "rot180"), "rot180"), f)
    with pytest.raises(ValueError):
        apply_frame_transform(f, "diagonal")


def _fake_state(**kw):
    base = dict(centroid_x_px=100.0, centroid_y_px=120.0, mask_area_px=500, centroid_x_normalized=0.4,
                centroid_y_normalized=0.47, mask_area_ratio=0.05, major_axis_length=40.0, minor_axis_length=20.0,
                orientation_degrees=30.0, segmentation_confidence=0.9, lumen_surrounding_contrast=0.4,
                border_contact_ratio=0.0, largest_component_ratio=1.0, mean_boundary_entropy=0.2,
                control_quality_score=0.8, valid_for_control=True)
    base.update(kw)
    return SimpleNamespace(**base)


def test_control_state_to_vector():
    v, q, tok, e = control_state_to_vector(_fake_state(), beam_axis_px=128.0, image_size=(256, 256), deadband_px=8.0)
    assert v.shape == (STATE_DIM,) and len(STATE_FEATURE_NAMES) == STATE_DIM
    assert v[0] == 1 and v[1] == 1 and abs(v[2] - (-0.1)) < 1e-6 and q == 0.8
    assert e == 28.0 and tok == TOKEN_MOVE_LEFT and v[17 + TOKEN_MOVE_LEFT] == 1 and v[17:].sum() == 1
    assert abs(v[16] - 28.0 / 256) < 1e-6
    # 마스크 없음 → check-filling, 기하 특징 0
    v2, _, tok2, e2 = control_state_to_vector(_fake_state(centroid_x_px=None, mask_area_px=0,
                                                          lumen_surrounding_contrast=None),
                                              128.0, (256, 256), 8.0)
    assert tok2 == TOKEN_CHECK_FILLING and np.isnan(e2) and v2[0] == 0 and v2[2:9].sum() == 0 and v2[11] == 0


def test_perception_result_roundtrip(tmp_path):
    r = empty_result(5, (64, 64))
    r.save(tmp_path / "p.npz")
    back = PerceptionResult.load(tmp_path / "p.npz")
    assert back.state.shape == (5, STATE_DIM) and np.isnan(back.quality).all() and (back.token == -1).all()


@pytest.mark.skipif(not (UNET_ROOT / "rus_perception").is_dir(), reason="Unet_seg 없음")
def test_unet_backend_wiring(tmp_path, synthetic_sessions):
    """무작위 가중치의 작은 slim_unet 으로 Predictor → ControlState → 특징 벡터 경로를 통과한다."""
    torch = pytest.importorskip("torch")
    if str(UNET_ROOT) not in sys.path:
        sys.path.insert(0, str(UNET_ROOT))
    from rus_perception.models.registry import build_model
    from rus_perception.utils.checkpoint import CheckpointMetadata, save_checkpoint

    from rus_policy.perception import UnetPerception, build_backend, perceive_session
    from rus_policy.session import load_session

    size = 64
    model_cfg = {"name": "slim_unet", "preset": "paper", "channels": [4, 8, 16, 32, 64], "input_size": [size, size]}
    unet_cfg = {
        "model": model_cfg,
        "data": {"image_size": [size, size], "intensity_normalization": "per_image"},
        "postprocess": {"threshold": 0.5, "largest_component": True},
        "control": {
            "roi": {"mode": "fan", "apex_xy": [0.5, -0.15], "radius_range": [0.18, 1.12], "half_angle_deg": 29.0},
            "quality": {"aggregation": "geometric", "target_area_ratio": 0.07, "area_tolerance": 0.03},
        },
    }
    cfg_path = tmp_path / "unet.yaml"
    cfg_path.write_text(yaml.safe_dump(unet_cfg), encoding="utf-8")
    model = build_model(model_cfg)
    ckpt = tmp_path / "tiny.pt"
    save_checkpoint(ckpt, model, CheckpointMetadata(epoch=0, best_metric=0.0, best_metric_name="dice",
                                                    model_version="tiny", seed=0, config=unet_cfg))

    backend = UnetPerception(ckpt, cfg_path, device="cpu", hold_deadband_px=2.0)
    assert backend.image_size == (size, size) and backend.roi is not None
    assert 0 < backend.beam_axis_px < size
    manifest, records = synthetic_sessions
    session = load_session(manifest.parent / records[0].session_dir)
    frames = np.asarray(session.frames[:6])
    res = backend.run(frames, progress=False)
    assert res.state.shape == (6, STATE_DIM) and res.backend == "unet"
    assert np.isfinite(res.quality).all() and np.all((res.token >= 0) & (res.token <= 3))
    assert np.isfinite(res.state).all()
    assert res.checkpoint_id not in ("", "unknown", "none")

    # 캐시 경로: build_backend + perceive_session
    pcfg = small_config()
    pcfg.perception.backend = "unet"
    pcfg.paths.unet_checkpoint = str(ckpt)
    pcfg.paths.unet_config = str(cfg_path)
    pcfg.paths.perception_cache_dir = str(tmp_path / "cache")
    pcfg.perception.device = "cpu"
    b2 = build_backend(pcfg)
    full = perceive_session(pcfg, session, b2)
    assert full.state.shape[0] == session.n_frames
    cached = perceive_session(pcfg, session, b2)
    assert np.array_equal(cached.state, full.state)
    assert list((tmp_path / "cache").glob("*.npz"))

    # 없는 체크포인트는 친절한 오류
    pcfg.paths.unet_checkpoint = str(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError):
        build_backend(pcfg)
