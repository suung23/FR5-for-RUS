"""⏳ 상수 추정 도구 — US 지연 상호상관, R_SP 주축 추정."""

from __future__ import annotations

import numpy as np
import pytest

from rus_policy.calibration import estimate_sensor_to_probe, estimate_us_latency, principal_rotation_axis
from rus_policy.imu_labels import quat_to_matrix


@pytest.mark.parametrize("fps,latency", [(8.0, 0.15), (20.0, 0.12), (30.0, 0.06)])
def test_latency_recovered_from_events(fps, latency):
    """IMU 스파이크 + latency 뒤에 영상 강도가 급변하면 그 latency 가 나온다 (프레임 간격 내 오차)."""
    rng = np.random.RandomState(1)
    T, imu_hz = 60.0, 200.0
    t_imu = np.arange(0.0, T, 1.0 / imu_hz)
    acc = np.tile([0.0, 0.0, 9.81], (t_imu.size, 1)) + rng.normal(0, 0.05, (t_imu.size, 3))
    events = np.sort(rng.uniform(2.0, T - 2.0, size=15))
    for te in events:
        i = int(te * imu_hz)
        acc[i:i + 6, 2] += 8.0 * np.exp(-np.arange(6) / 2.0)          # 떼는 순간의 충격
    t_fr = np.arange(0.0, T, 1.0 / fps) + rng.uniform(0, 1.0 / fps)
    intensity = 0.4 + rng.normal(0, 0.005, t_fr.size)
    level = np.zeros(t_fr.size)
    for te in events:                                                 # 젤에서 떼면 어두워졌다 다시 밝아진다
        on = (t_fr >= te + latency) & (t_fr < te + latency + 0.6)
        level[on] -= 0.25
    intensity += level
    est = estimate_us_latency(t_fr, intensity, t_imu, acc, max_lag_s=0.8)
    assert abs(est.latency_s - latency) <= 1.0 / fps + 0.01, est.as_dict()
    assert est.peak_corr > 0.1 and est.n_imu_events >= 10


def test_latency_sign_convention():
    """양수 = 영상이 늦다. 영상이 IMU 보다 먼저 변하면 음수가 나온다."""
    rng = np.random.RandomState(2)
    T, imu_hz, fps = 40.0, 200.0, 20.0
    t_imu = np.arange(0.0, T, 1.0 / imu_hz)
    acc = np.tile([0.0, 0.0, 9.81], (t_imu.size, 1)) + rng.normal(0, 0.05, (t_imu.size, 3))
    t_fr = np.arange(0.0, T, 1.0 / fps)
    intensity = 0.4 + rng.normal(0, 0.005, t_fr.size)
    for te in np.arange(3.0, T - 3.0, 3.0):
        i = int(te * imu_hz)
        acc[i:i + 6, 2] += 8.0
        intensity[(t_fr >= te - 0.2) & (t_fr < te + 0.3)] -= 0.25     # 영상이 0.2 s 먼저
    est = estimate_us_latency(t_fr, intensity, t_imu, acc)
    assert est.latency_s < -0.1


def test_principal_axis_sign_follows_rotation_direction():
    rng = np.random.RandomState(0)
    axis = np.array([0.0, 0.6, 0.8])
    w = np.outer(np.abs(np.sin(np.linspace(0, np.pi, 200))) * 2.0, axis) + rng.normal(0, 0.05, (200, 3))
    a = principal_rotation_axis(w)
    assert np.dot(a, axis) > 0.99
    a2 = principal_rotation_axis(-w)
    assert np.dot(a2, axis) < -0.99


def test_sensor_to_probe_recovers_known_rotation():
    """R_SP 가 알려진 회전이면, 프로브 x/y/z 회전의 센서 자이로 주축에서 그대로 복원된다."""
    rng = np.random.RandomState(3)
    q = np.array([np.cos(0.4), 0.3, -0.5, 0.2])
    R_true = quat_to_matrix(q / np.linalg.norm(q))          # 프로브 ← 센서
    segs = []
    for e in np.eye(3):
        w_P = np.outer(np.abs(np.sin(np.linspace(0, np.pi, 300))) * 1.5, e)   # 프로브 축 e 둘레 + 회전
        w_S = w_P @ R_true                                   # ω_S = R_SPᵀ ω_P
        segs.append(w_S + rng.normal(0, 0.03, w_S.shape))
    est = estimate_sensor_to_probe(segs)
    assert np.allclose(est.R_sp, R_true, atol=0.03), est.R_sp
    assert est.det == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.abs(est.angles_deg - 90.0) < 5.0) and est.residual_deg < 3.0


def test_sensor_to_probe_needs_three_segments():
    with pytest.raises(ValueError):
        estimate_sensor_to_probe([np.eye(3)])
