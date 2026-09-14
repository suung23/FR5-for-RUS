"""scripts/run_policy.py 의 관측 조립 — 모델이 실제로 받아들이는지 ROS 없이 확인한다."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from conftest import small_config
from rus_policy.model import ACTION_DIM, build_policy
from rus_policy.perception import STATE_DIM
from run_policy import build_observation, quat_to_R


def _items(cfg, n, t_now, age_step=0.125):
    H, W = cfg.perception.frame_size
    rng = np.random.default_rng(0)
    return [(t_now - (n - 1 - i) * age_step,
             rng.integers(0, 255, (H, W), dtype=np.uint8),
             rng.normal(size=STATE_DIM).astype(np.float32),
             float(rng.uniform(0.5, 1.0))) for i in range(n)]


def test_quat_to_R_is_a_rotation():
    R = quat_to_R(0.5, 0.5, 0.5, 0.5)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0)


def test_observation_feeds_select_action():
    cfg = small_config()
    model = build_policy(cfg.model, cfg.timing).eval()
    t_now = time.time()
    obs = build_observation(cfg, _items(cfg, cfg.timing.obs_frames, t_now), t_now,
                            (1.0, 0.0, 0.0, 0.0), np.zeros(3, np.float32), t_now - 0.2, torch.device("cpu"))
    assert obs is not None
    q = obs.pop("quality_now")
    assert 0.0 <= q <= 1.0
    m, k = cfg.timing.obs_frames, cfg.timing.chunk_steps
    assert obs["frames"].shape == (1, m, *cfg.perception.frame_size)
    assert obs["frames"].min() >= 0.0 and obs["frames"].max() <= 1.0
    assert obs["frame_valid"].all() and (obs["frame_dt"] <= 0).all()
    with torch.no_grad():
        sel = model.select_action(obs, n_samples=4)
    assert sel["a"].shape == (1, k, ACTION_DIM) and torch.isfinite(sel["a"]).all()
    assert sel["net"].shape == (1, ACTION_DIM) and sel["Q_hat"].shape == (1, k)


def test_short_buffer_is_padded_and_marked_invalid():
    cfg = small_config()
    t_now = time.time()
    n = 3
    obs = build_observation(cfg, _items(cfg, n, t_now), t_now, None,
                            np.zeros(3, np.float32), None, torch.device("cpu"))
    v = obs["frame_valid"][0]
    assert (~v[:cfg.timing.obs_frames - n]).all() and v[-n:].all()
    assert obs["vec"][0, 3] == 0.0 and obs["vec"][0, 4] == pytest.approx(10.0)   # 직전 동작 없음


def test_stale_frames_are_invalidated_and_empty_returns_none():
    cfg = small_config()
    t_now = time.time()
    old = t_now - cfg.timing.obs_frame_max_age_s - 5.0
    items = _items(cfg, cfg.timing.obs_frames, old, age_step=0.0)
    assert build_observation(cfg, items, t_now, None, np.zeros(3, np.float32), None,
                             torch.device("cpu")) is None
    assert build_observation(cfg, [], t_now, None, np.zeros(3, np.float32), None,
                             torch.device("cpu")) is None


def test_gravity_channel_is_filled_from_pose():
    cfg = small_config()
    t_now = time.time()
    obs = build_observation(cfg, _items(cfg, cfg.timing.obs_frames, t_now), t_now,
                            (1.0, 0.0, 0.0, 0.0), np.zeros(3, np.float32), None, torch.device("cpu"))
    assert np.allclose(obs["vec"][0, 5:8].numpy(), [0.0, 0.0, 1.0], atol=1e-6)


def test_perception_path_comes_from_the_config_file_not_the_checkpoint():
    """지각 경로는 **지금의 설정 파일**이 이긴다.

    2026-09-14: U-Net 을 갈아 끼우고 policy_default.yaml 을 고쳤는데 실기는 옛 모델로
    돌았다. 러너가 정책 체크포인트 안에 저장된 설정(학습 당시 경로)을 썼기 때문이다.
    같은 프레임을 새 모델로 재생하면 60 개 중 59 개가 검출되는데, 기록에는 전부 미검출로
    남았다 — 화면에는 Q_seg 0.07 이 이어졌고 아무 오류도 없었다.

    정책 가중치는 체크포인트의 것이 맞다. 지각은 정책 학습 뒤에도 갈아 끼우는 부품이라
    출처가 다르면 파일이 이겨야 하고, 갈아 끼운 사실이 로그에 남아야 한다.
    """
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text()
    assert '"--perception-config"' in src, "지각 설정을 고를 인자가 없다"
    # 기본값은 그 설정 파일이어야 한다 — 다만 **cwd 에 매달리면 안 된다.** 예전 기본값은
    # 문자열 "policy_learning/configs/…" 라 저장소 뿌리에서만 맞았고, 런처가 띄우는 자리
    # (policy_learning 안)에서는 FileNotFoundError 로 죽었다 (2026-09-14 본 시험 ep0002).
    from run_policy import DEFAULT_PERCEPTION_CONFIG
    assert DEFAULT_PERCEPTION_CONFIG.is_absolute() and DEFAULT_PERCEPTION_CONFIG.is_file(), \
        f"기본 지각 설정이 없다: {DEFAULT_PERCEPTION_CONFIG}"
    assert DEFAULT_PERCEPTION_CONFIG.name == "policy_default.yaml", DEFAULT_PERCEPTION_CONFIG
    assert "default=str(DEFAULT_PERCEPTION_CONFIG)" in src, \
        "기본값이 설정 파일이 아니면 예전 함정으로 되돌아간다"
    head = src[src.index("model, cfg = load_policy(args.checkpoint"):]
    head = head[:head.index("backend = build_backend(cfg)")]
    assert "unet_checkpoint" in head and "setattr(cfg.paths" in head, \
        "체크포인트의 지각 경로를 덮어쓰지 않는다"
    assert "logging.warning" in head, "갈아 끼운 사실을 로그에 남기지 않는다"
    assert "U-Net=" in src, "실제로 실린 U-Net 을 기동 로그에 찍지 않는다"
