"""DLS 해법과 추종오차 감시의 단위 시험.

rclpy 와 PyKDL 없이 돈다 — 그래서 실로봇 없이도 회귀를 잡는다. KDL 을 쓰는
:class:`DlsSolver` 는 여기서 시험하지 않는다.
"""
import numpy as np
import pytest

from fr5_ik.dls_solver import dls_solve, tracking_error


def well_conditioned() -> np.ndarray:
    """특이점에서 먼 6x6 야코비안."""
    rng = np.random.default_rng(0)
    while True:
        jacobian = rng.normal(size=(6, 6))
        if np.linalg.cond(jacobian) < 8.0:
            return jacobian


def test_reproduces_twist_when_far_from_singularity():
    """조건수가 좋으면 DLS 는 지령을 사실상 그대로 낸다."""
    jacobian = well_conditioned()
    twist = np.array([0.01, -0.005, 0.008, 0.05, -0.02, 0.03])

    achieved = jacobian @ dls_solve(jacobian, twist, damping=0.02)

    assert np.allclose(achieved, twist, atol=1e-3)


def test_damping_trades_accuracy_for_stability():
    """감쇠를 키우면 추종이 나빠진다 — DLS 의 본질적 맞바꿈."""
    jacobian = well_conditioned()
    twist = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])

    gentle = np.linalg.norm(twist - jacobian @ dls_solve(jacobian, twist, 0.01))
    heavy = np.linalg.norm(twist - jacobian @ dls_solve(jacobian, twist, 1.0))

    assert heavy > gentle


def test_singular_jacobian_stays_bounded():
    """특이 자세에서도 관절속도가 발산하지 않아야 한다. 감쇠를 두는 이유다."""
    jacobian = np.zeros((6, 6))
    jacobian[0, 0] = 1.0  # 계수 1

    joint_velocity = dls_solve(jacobian, np.ones(6) * 0.01, damping=0.02)

    assert np.all(np.isfinite(joint_velocity))
    assert np.linalg.norm(joint_velocity) < 100.0


def test_zero_twist_gives_zero_velocity():
    assert np.allclose(dls_solve(well_conditioned(), np.zeros(6)), 0.0)


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="맞지 않는다"):
        dls_solve(np.zeros((6, 6)), np.zeros(3))


def test_rejects_negative_damping():
    with pytest.raises(ValueError, match="음수"):
        dls_solve(np.zeros((6, 6)), np.zeros(6), damping=-0.1)


def test_tracking_error_clean_when_reproduced():
    jacobian = well_conditioned()
    twist = np.array([0.0, 0.0, 0.005, 0.0, 0.0, 0.0])
    joint_velocity = dls_solve(jacobian, twist, 0.02)

    result = tracking_error(jacobian, twist, joint_velocity)

    assert not result.force_axes_degraded
    assert result.linear_norm < 1e-3


def test_tracking_error_flags_lost_normal_axis():
    """z 축(법선력)을 낼 수 없는 자세를 만들면 힘 축 열화가 잡혀야 한다.

    이것이 DLS 를 감시하는 이유다. DLS 는 우선순위를 표현하지 못해 힘 축을 조용히
    희생할 수 있고, 그러면 접촉력이 어긋나는데 로그에는 아무것도 남지 않는다.
    """
    jacobian = np.eye(6)
    jacobian[2, :] = 0.0  # 어떤 관절 조합으로도 z 병진을 만들 수 없다
    twist = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0])

    result = tracking_error(jacobian, twist, dls_solve(jacobian, twist, 0.02))

    assert result.force_axes_degraded
    assert result.per_axis[2] == pytest.approx(0.01, abs=1e-6)


def test_tracking_error_ignores_image_axis_loss():
    """영상 축(x, y, rz)이 뭉개지는 것은 힘 축 열화로 보고하지 않는다.

    영상 축은 slack 을 허용하는 축이므로, 여기서 경보를 울리면 진짜 위험한
    힘 축 경보가 묻힌다.
    """
    jacobian = np.eye(6)
    jacobian[0, :] = 0.0  # x 병진 불가
    twist = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = tracking_error(jacobian, twist, dls_solve(jacobian, twist, 0.02))

    assert not result.force_axes_degraded
    assert result.per_axis[0] == pytest.approx(0.01, abs=1e-6)
