"""IMU 라벨 파이프라인 — 정지 판정 · ZUPT · 프레임 변환 · 합성 정답 대비."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rus_policy.config import ImuConfig, PolicyConfig
from rus_policy.imu_labels import (
    G0, cumtrapz, earth_convention, label_session, move_segments, quat_to_matrix, rotmat_log,
    segments_from_mask, sigma_for, still_mask, zupt_integrate,
)
from rus_policy.session import load_session
from rus_policy.synth import SynthConfig, _matrix_to_quat, generate_session

IMU_QC = Path(__file__).resolve().parents[2] / "imu_bench" / "qc_track"


def test_rotmat_log_roundtrip():
    rng = np.random.RandomState(0)
    for _ in range(20):
        r = rng.normal(size=3)
        r = r / np.linalg.norm(r) * rng.uniform(0.01, 3.0)      # |r| < π (로그맵의 주치)
        th = np.linalg.norm(r)
        k = r / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        assert np.allclose(rotmat_log(R), r, atol=1e-6)


def test_quat_matrix_consistency():
    rng = np.random.RandomState(1)
    for _ in range(10):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        assert np.allclose(quat_to_matrix(_matrix_to_quat(Q)), Q, atol=1e-9)


def test_zupt_cancels_constant_bias_exactly():
    t = np.arange(0, 2.0, 0.005)
    a = np.tile([0.03, -0.02, 0.01], (t.size, 1))          # 상수 바이어스만
    v, p, v_end = zupt_integrate(a, t, zupt=True)
    assert np.abs(v_end - np.array([0.03, -0.02, 0.01]) * (t[-1] - t[0])).max() < 1e-9
    assert np.abs(p[-1]).max() < 1e-9                        # 선형 디드리프트가 정확히 지운다


def test_cumtrapz_matches_analytic():
    t = np.linspace(0, 1, 201)
    y = (2 * t)[:, None]
    assert np.allclose(cumtrapz(y, t)[-1], 1.0, atol=1e-4)


def test_still_mask_matches_qc_common():
    """벡터화 판정이 imu_bench/qc_track/qc_common.still_mask 와 같은 마스크를 낸다."""
    if not (IMU_QC / "qc_common.py").is_file():
        pytest.skip("imu_bench/qc_track 없음")
    sys.path.insert(0, str(IMU_QC))
    import qc_common  # noqa: E402

    rng = np.random.RandomState(3)
    n = 3000
    t = np.arange(n) * 0.004
    gyr = rng.normal(0, 0.001, (n, 3))
    acc = rng.normal(0, 0.02, (n, 3)) + np.array([0, 0, G0])
    gyr[1000:1400] += 0.2 * np.sin(np.linspace(0, 6, 400))[:, None]
    acc[2000:2300, 0] += 0.5
    cfg = ImuConfig(still_gyro_sd=qc_common.STILL_GYRO_SD, still_gyro_mean=qc_common.STILL_GYRO_MEAN,
                    still_accel_sd=qc_common.STILL_ACCEL_SD, still_win_s=0.30)
    ours = still_mask(t, gyr, acc, cfg, dt=0.004)
    ref = qc_common.still_mask(t, gyr, acc, win_s=0.30, dt=0.004)
    assert ours.shape == ref.shape
    assert (ours == ref).mean() > 0.995         # 부동소수 경계에서 한두 표본은 다를 수 있다


def test_segments_and_move_segments():
    t = np.arange(0, 10, 0.01)
    still = np.zeros(t.size, bool)
    still[0:150] = True      # 1.5 s
    still[300:360] = True    # 0.6 s
    still[700:1000] = True   # 3 s
    segs = segments_from_mask(t, still, 0.5)
    assert len(segs) == 3
    cfg = ImuConfig()
    moves, stills = move_segments(t, still, cfg)
    assert len(moves) == 2
    m = moves[0]
    assert m.a0 == 0 and m.a1 == 149 and m.b0 == 300
    assert m.ia == int(149 - 0.25 * 149) and m.ib == int(300 + 0.25 * 59)
    assert abs(m.move_s - (t[300] - t[149])) < 1e-9


def test_earth_convention_detects_transposed():
    rng = np.random.RandomState(4)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    q = _matrix_to_quat(Q)
    a_S = Q.T @ np.array([0, 0, G0])                          # R(q)·a = +g z  → 'R'
    conv, amb = earth_convention(np.tile(q, (20, 1)), np.tile(a_S, (20, 1)))
    assert conv == "R" and not amb
    a_S2 = Q @ np.array([0, 0, G0])                            # R(q)ᵀ·a = +g z → 'R.T'
    conv2, _ = earth_convention(np.tile(q, (20, 1)), np.tile(a_S2, (20, 1)))
    assert conv2 == "R.T"


def test_sigma_window_is_capped_at_chunk_horizon(tmp_path):
    """이동이 chunk 창보다 길어도 σ 는 chunk 창 (1.6 s) 으로 잰다 (2026-09-08 §5.3a)."""
    from rus_policy.config import PolicyConfig
    from rus_policy.imu_labels import label_session
    from rus_policy.session import load_session
    from rus_policy.synth import SynthConfig, generate_session

    d = generate_session(tmp_path, SynthConfig(duration_s=40.0, seed=5, image_size=32, accel_noise=0.01,
                                                move_s=(2.5, 3.5)), name="long")
    s = load_session(d)
    cfg = PolicyConfig()
    labels = label_session(s.imu, cfg.timing, cfg.imu, cfg.labels)
    assert labels.labels, "이동 구간이 검출되어야 한다"
    horizon = cfg.timing.chunk_horizon_s
    for lab in labels.labels:
        assert lab.diagnostics["sigma_window_s"] <= horizon + 1e-9
        if lab.diagnostics["integration_window_s"] > horizon:
            assert lab.diagnostics["sigma_window_s"] == pytest.approx(horizon)
            assert lab.sigma_net[0] == pytest.approx(cfg.labels.sigma_translation_coeff_mm * horizon ** 1.5)
        assert lab.t_still_start_dev <= lab.t_anchor_dev


def test_sigma_tables():
    cfg = PolicyConfig().labels
    net, shape = sigma_for("freehand", 1.6, cfg)
    assert abs(net[0] - 0.7 * 1.6 ** 1.5) < 1e-9 and net[2] == cfg.sigma_rotation_deg
    assert np.all(shape <= net)
    net_t, _ = sigma_for("teleop", 1.6, cfg)
    assert net_t[0] == cfg.teleop_sigma_translation_mm


def test_sensor_to_probe_validation():
    cfg = ImuConfig(sensor_to_probe=[[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    with pytest.raises(ValueError):
        cfg.R_sp()


def test_labels_match_synthetic_truth(tmp_path):
    """저잡음 합성 세션: 회전 < 0.1°, 병진 평균 < 1 mm, 구간 검출률 > 90 %."""
    root = generate_session(tmp_path, SynthConfig(duration_s=60, seed=7, accel_noise=0.005, image_size=32,
                                                  t0_unix=1_700_000_000.0), name="s")
    s = load_session(root)
    cfg = PolicyConfig()
    L = label_session(s.imu, cfg.timing, cfg.imu, cfg.labels)
    truth = json.loads((root / "truth.json").read_text())["segments"]
    assert len(L.labels) >= 0.9 * len(truth)
    errs = []
    for lab in L.labels:
        ta, tb = lab.t_anchor_dev - 1.0, lab.t_end_dev - 1.0      # 합성 dev_us 는 t + 1 s
        m = [x for x in truth if ta <= x["t_move_start"] <= tb]
        if len(m) != 1:
            continue
        errs.append(lab.P[-1] - np.array([m[0]["dx_mm"], m[0]["dy_mm"], m[0]["dtheta_deg"]]))
    errs = np.abs(np.array(errs))
    assert len(errs) >= 0.85 * len(truth)
    assert errs[:, 2].mean() < 0.1
    assert errs[:, :2].mean() < 1.0
    # 라벨 σ 가 실제 오차를 덮는다 (보수적)
    sig = np.array([l.sigma_net for l in L.labels])
    assert np.median(sig[:, 0]) > errs[:, 0].mean()
    # P[0] = 0, 격자 길이 k+1, 중력이 프로브 −z (빔이 아래를 향함)
    lab = L.labels[0]
    assert np.all(lab.P6[0] == 0) and lab.P6.shape == (cfg.timing.chunk_steps + 1, 6)
    assert lab.gravity_P[2] < -0.99


def test_sensor_to_probe_rotation_rotates_labels(tmp_path):
    """R_SP 를 z 둘레 90° 로 두면 (x, y) 라벨이 그만큼 돈다."""
    root = generate_session(tmp_path, SynthConfig(duration_s=30, seed=8, accel_noise=0.005, image_size=32,
                                                  t0_unix=1_700_000_000.0), name="s")
    s = load_session(root)
    cfg = PolicyConfig()
    L0 = label_session(s.imu, cfg.timing, cfg.imu, cfg.labels)
    cfg.imu.sensor_to_probe = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    L1 = label_session(s.imu, cfg.timing, cfg.imu, cfg.labels)
    for a, b in zip(L0.labels, L1.labels):
        R = np.asarray(cfg.imu.sensor_to_probe, float)
        assert np.allclose(R @ a.P6[-1, :3], b.P6[-1, :3], atol=1e-6)
        assert np.isclose(a.P6[-1, 5], b.P6[-1, 5], atol=1e-6)
