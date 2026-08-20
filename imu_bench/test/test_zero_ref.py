"""영점 캘리브레이션 검증 — 센서 없이 합성 데이터로."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "host"))

import zero_ref  # noqa: E402
from fusion import quat_angle_deg, quat_to_matrix  # noqa: E402

G0 = zero_ref.G0


def make_samples(n=600, fs=245.0, q=None, scale=1.0, bias=None,
                 gyro_sd=2e-4, accel_sd=0.02, gyro_mean=None, seed=0):
    """임의 자세로 정지해 있는 센서가 낼 법한 샘플을 만든다.

    가속도계는 지구 프레임의 (0,0,g) 를 센서 프레임에서 보고, 스케일 오차와
    바이어스가 얹힌다.
    """
    rng = np.random.default_rng(seed)
    if q is None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    R = quat_to_matrix(q)                     # 센서 -> 지구
    g_sensor = R.T @ np.array([0.0, 0.0, G0])  # 정지 시 센서가 읽는 비력
    if bias is None:
        bias = np.zeros(3)

    t = np.arange(n) / fs
    out = []
    for i in range(n):
        acc = scale * g_sensor + bias + rng.normal(0, accel_sd, 3)
        gyr = rng.normal(0, gyro_sd, 3) + (np.zeros(3) if gyro_mean is None else gyro_mean)
        out.append((t[i], acc, gyr, q.copy()))
    return out


def test_recovers_gravity_and_scale_error():
    """스케일 오차 +2.28 % 를 넣으면 측정 중력이 그만큼 크게 나와야 한다."""
    ref = zero_ref.measure(make_samples(scale=1.0228))
    g = np.linalg.norm(ref.b_E)
    assert g == pytest.approx(G0 * 1.0228, rel=1e-3)
    assert ref.to_dict()["gravity_scale_error_pct"] == pytest.approx(2.28, abs=0.05)
    assert ref.quality["still"]


def test_bias_earth_is_vertical_when_attitude_is_right():
    """자세가 맞으면 지구프레임 바이어스는 (0,0,g) 여야 한다 — 수평 성분이 0."""
    q = np.array([0.9239, 0.2209, 0.1913, 0.2209])   # 임의 기울인 자세
    ref = zero_ref.measure(make_samples(q=q, scale=1.02))
    assert abs(ref.b_E[0]) < 0.02 and abs(ref.b_E[1]) < 0.02
    assert ref.b_E[2] == pytest.approx(G0 * 1.02, rel=1e-3)


def test_linear_accel_is_zero_at_rest():
    """영점을 적용하면 정지 상태의 선형가속도가 0 이어야 한다 — 그게 영점의 정의다."""
    samples = make_samples(scale=1.03, bias=np.array([0.05, -0.03, 0.02]))
    ref = zero_ref.measure(samples)
    resid = np.array([ref.linear_accel(q, a) for _, a, _, q in samples[::10]])
    assert np.linalg.norm(resid.mean(0)) < 5e-3


def test_relative_rotation_is_zero_at_reference():
    ref = zero_ref.measure(make_samples())
    assert quat_angle_deg(ref.relative_quat(ref.q0), np.array([1.0, 0, 0, 0])) < 1e-6


def test_moving_sensor_fails_the_still_gate():
    """움직이는 동안 잡은 영점은 **조용히 통과하면 안 된다.**"""
    moving = make_samples(gyro_sd=0.05, accel_sd=0.5)
    assert zero_ref.measure(moving).quality["still"] is False

    drifting = make_samples(gyro_mean=np.array([0.02, 0.0, 0.0]))
    assert zero_ref.measure(drifting).quality["still"] is False


def test_roundtrip_through_dict():
    ref = zero_ref.measure(make_samples(scale=1.01))
    back = zero_ref.ZeroReference.from_dict(ref.to_dict())
    assert np.allclose(back.q0, ref.q0)
    assert np.allclose(back.b_E, ref.b_E)
    assert back.quat_convention == ref.quat_convention


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        zero_ref.measure(make_samples(n=5))


# --- run_zero_calibration 배선 검증 (하드웨어 없이) ---------------------------

class StubStream:
    """ImuStream 의 캡처 API 만 흉내낸다. 실제 시리얼·스레드 없이 흐름을 검증한다."""

    def __init__(self, samples):
        self._samples = samples
        self._capturing = False
        self.zero = None
        self.captures = 0

    def start_capture(self):
        self._capturing = True
        self.captures += 1

    def capture_count(self):
        return len(self._samples) if self._capturing else 0

    def stop_capture(self):
        self._capturing = False
        return list(self._samples)


def test_run_zero_calibration_sets_stream_zero(capsys):
    from imu_fusion_view import run_zero_calibration

    stream = StubStream(make_samples())
    ref = run_zero_calibration(stream, 0.2, quiet=True)

    assert ref is not None
    assert stream.zero is ref
    assert ref.quality["still"]
    assert stream.captures == 1


def test_run_zero_calibration_retries_then_gives_up_on_motion(capsys):
    """움직이는 중이면 재시도하고, 끝내 실패하면 stream.zero 를 세우지 않는다.

    조용히 나쁜 영점을 물려 놓는 것이 가장 나쁜 실패다 — 로그는 정상으로 보이는데
    변위 계산만 틀린다.
    """
    from imu_fusion_view import run_zero_calibration

    stream = StubStream(make_samples(gyro_sd=0.05, accel_sd=0.5))
    ref = run_zero_calibration(stream, 0.05, retries=2, quiet=True)

    assert ref is None
    assert stream.zero is None
    assert stream.captures == 3          # 최초 1 회 + 재시도 2 회
