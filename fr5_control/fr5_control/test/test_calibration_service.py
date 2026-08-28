"""교정 세션 — 자동 캡처와 자세 분리도.

시리얼도 로봇도 여기서 다루지 않는다. 관절각이 들어왔을 때 **언제 잡고 언제 안
잡는가** 를 고정한다.

두 가지가 이 시험의 요점이다. 하나는 가만히 둔 자세를 초당 스무 번 잡지 않는 것,
다른 하나는 이미 잡은 자세와 비슷한 자세를 잡지 않는 것이다 — 비슷한 자세를 열두
번 모아도 커버리지는 오르지 않고, 그러면 조작자는 왜 유효하지 않은지 알 수 없다.
"""

import math

import numpy as np
import pytest

from fr5_control.calibration_service import (
    CalibrationSession,
    StillnessDetector,
)
from fr5_control.wrench_calibration import CalibrationPose


# -- 자동 캡처: 드래그가 멈춘 순간 --------------------------------------


def _drag(detector, start, end, now, steps=10, dt=0.05):
    """관절을 start 에서 end 까지 옮긴다. 마지막 반환값을 준다."""
    out = ""
    for i in range(steps):
        frac = (i + 1) / steps
        joints = [start + (end - start) * frac] * 6
        out = detector.update(joints, now + i * dt)
    return out, now + steps * dt


def _hold(detector, value, now, seconds=2.0, dt=0.05):
    """가만히 둔다. 그 사이 나온 판정을 모두 준다."""
    results = []
    for i in range(int(seconds / dt)):
        results.append(detector.update([value] * 6, now + i * dt))
    return results, now + seconds


def test_stillness_reports_settled_once_after_a_drag():
    detector = StillnessDetector()
    moving, now = _drag(detector, 0.0, 10.0, 0.0)
    assert moving == "moving"
    results, _ = _hold(detector, 10.0, now)
    assert results.count("settled") == 1


def test_stillness_does_not_report_again_while_still():
    """가만히 둔 자세를 초당 스무 번 잡으면 안 된다."""
    detector = StillnessDetector()
    _, now = _drag(detector, 0.0, 10.0, 0.0)
    first, now = _hold(detector, 10.0, now, seconds=3.0)
    again, _ = _hold(detector, 10.0, now, seconds=3.0)
    assert first.count("settled") == 1
    assert again.count("settled") == 0


def test_stillness_rearms_after_moving_again():
    detector = StillnessDetector()
    _, now = _drag(detector, 0.0, 10.0, 0.0)
    _, now = _hold(detector, 10.0, now)
    _, now = _drag(detector, 10.0, 25.0, now)
    results, _ = _hold(detector, 25.0, now)
    assert results.count("settled") == 1


def test_stillness_ignores_stillness_before_any_movement():
    """기동 직후 가만히 있는 것은 '방금 멈춘 것' 이 아니다."""
    detector = StillnessDetector()
    results, _ = _hold(detector, 0.0, 0.0, seconds=5.0)
    assert "settled" not in results


def test_stillness_waits_out_the_settle_window():
    """손을 뗀 직후의 흔들림을 잡으면 표본이 흔들린 값이 된다."""
    detector = StillnessDetector(settle_s=1.2)
    _, now = _drag(detector, 0.0, 10.0, 0.0)
    early, _ = _hold(detector, 10.0, now, seconds=1.0)
    assert "settled" not in early


def test_stillness_tolerates_controller_noise():
    """정지 중 관절각이 미세하게 떨려도 움직임으로 보지 않는다."""
    detector = StillnessDetector(move_threshold_deg=0.15)
    _, now = _drag(detector, 0.0, 10.0, 0.0)
    results = []
    for i in range(60):
        jitter = 0.04 if i % 2 else -0.04
        results.append(detector.update([10.0 + jitter] * 6, now + i * 0.05))
    assert results.count("settled") == 1


def test_stillness_reset_forgets_everything():
    detector = StillnessDetector()
    _drag(detector, 0.0, 10.0, 0.0)
    detector.reset()
    assert not detector.armed
    results, _ = _hold(detector, 10.0, 5.0, seconds=3.0)
    assert "settled" not in results


# -- 자세 분리도 ---------------------------------------------------------


def test_separation_is_wide_open_with_no_poses():
    assert CalibrationSession().separation_deg([0.0, 0.0, -9.8]) == 180.0


def test_separation_measures_the_gravity_direction():
    session = CalibrationSession()
    session.poses.append(CalibrationPose(
        gravity_sensor=np.array([0.0, 0.0, -9.80665]), wrench=np.zeros(6), label="a",
    ))
    assert session.separation_deg([0.0, 0.0, -9.80665]) == pytest.approx(0.0, abs=1e-6)
    assert session.separation_deg([0.0, -9.80665, 0.0]) == pytest.approx(90.0, abs=1e-6)
    assert session.separation_deg([0.0, 0.0, 9.80665]) == pytest.approx(180.0, abs=1e-6)


def test_separation_ignores_magnitude():
    """중력 크기는 어디서나 같다. 방향만이 새 정보다."""
    session = CalibrationSession()
    session.poses.append(CalibrationPose(
        gravity_sensor=np.array([0.0, 0.0, -9.80665]), wrench=np.zeros(6), label="a",
    ))
    assert session.separation_deg([0.0, 0.0, -1.0]) == pytest.approx(0.0, abs=1e-6)


def test_separation_takes_the_nearest_pose():
    session = CalibrationSession()
    for vector in ([0.0, 0.0, -9.8], [0.0, -9.8, 0.0]):
        session.poses.append(CalibrationPose(
            gravity_sensor=np.array(vector), wrench=np.zeros(6), label="x",
        ))
    # y 축에서 10° 떨어진 방향. 가장 가까운 자세와의 각이 나와야 한다.
    angle = math.radians(10.0)
    probe = [0.0, -9.8 * math.cos(angle), -9.8 * math.sin(angle)]
    assert session.separation_deg(probe) == pytest.approx(10.0, abs=1e-6)
