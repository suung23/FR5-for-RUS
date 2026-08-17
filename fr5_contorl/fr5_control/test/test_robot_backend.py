"""로봇 백엔드 추상화와 mock 구현의 단위 시험 (DESIGN_NOTES §12.5).

rclpy 와 fairino SDK 없이 돈다. mock 을 두는 주된 이유가 **실패 경로 시험**이므로,
여기서 그 주입이 실제로 동작하는지 확인한다.
"""
import math

import numpy as np
import pytest

from fr5_control.robot_backend import (
    MockBackend,
    fr5_forward_kinematics,
    make_backend,
)

HOME = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
LINK_SUM = 0.152 + 0.425 + 0.39501 + 0.1021 + 0.102


def test_fk_returns_proper_rigid_transform():
    for joints in ([0.0] * 6, [0.3] * 6, [math.radians(d) for d in HOME]):
        transform = fr5_forward_kinematics(joints)
        rotation = transform[:3, :3]

        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
        assert transform[3, :].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_fk_stays_inside_reach():
    """말단은 링크 길이 합보다 멀리 갈 수 없다."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        joints = rng.uniform(-math.pi, math.pi, size=6)
        assert np.linalg.norm(fr5_forward_kinematics(joints)[:3, 3]) <= LINK_SUM + 1e-9


def test_fk_first_joint_rotates_about_base_z():
    """j1 만 돌리면 말단 높이가 변하지 않아야 한다 — 축 규약 확인."""
    base = fr5_forward_kinematics([0.0] * 6)
    turned = fr5_forward_kinematics([0.7, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert turned[2, 3] == pytest.approx(base[2, 3], abs=1e-9)
    assert np.linalg.norm(turned[:2, 3]) == pytest.approx(
        np.linalg.norm(base[:2, 3]), abs=1e-9
    )


def make_mock(**kwargs) -> MockBackend:
    backend = MockBackend(start_joints_deg=list(HOME), ft_noise_std=0.0, **kwargs)
    backend.connect()
    backend.servo_start()
    return backend


def test_servo_j_updates_state():
    backend = make_mock()
    target = list(HOME)
    target[1] += 2.0

    assert backend.servo_j(target, 0.008, 1) == 0
    assert backend.joint_positions_deg()[1] == pytest.approx(HOME[1] + 2.0)


def test_servo_j_rejected_before_servo_start():
    """서보 모드 밖의 지령은 조용히 먹히지 않고 오류를 낸다."""
    backend = MockBackend(start_joints_deg=list(HOME))
    backend.connect()

    assert backend.servo_j(HOME, 0.008, 1) != 0


def test_no_contact_above_plane_gives_no_force():
    backend = make_mock(contact_plane_z=-10.0)  # 평면이 한참 아래

    assert backend.wrench_raw()[2] == pytest.approx(0.0, abs=1e-9)


def test_penetration_produces_compressive_force():
    """평면 위에 두고 내리면 F_z 가 침투 깊이에 비례해 커져야 한다.

    Phase 1 의 admittance 가 0 이 아닌 힘을 받아볼 수 있어야 하므로 확인한다.
    """
    backend = make_mock(contact_plane_z=-10.0, contact_stiffness=2000.0)
    height = backend.tool_pose()[2] / 1000.0
    backend.contact_plane_z = height  # 지금 위치를 접촉면으로 삼는다

    assert backend.wrench_raw()[2] == pytest.approx(0.0, abs=1e-6)

    backend.contact_plane_z = height + 0.001  # 1 mm 침투
    force_z = backend.wrench_raw()[2]

    assert force_z < 0.0  # 조직이 프로브를 밀어낸다
    assert abs(force_z) == pytest.approx(2.0, rel=0.05)  # 2000 N/m x 1 mm


def test_wrench_dropout_injection():
    """F/T 두절은 실로봇에서 일부러 일으키기 곤란하다 — mock 이 존재하는 이유."""
    backend = make_mock(drop_wrench_after_s=0.0)
    assert backend.wrench_valid()

    backend.drop_wrench_after_s = 1e-9  # 이미 경과했다
    assert not backend.wrench_valid()
    assert backend.wrench_raw() == [0.0] * 6


def test_state_freeze_injection():
    """통신 두절 흉내: 지령은 받지만 상태가 갱신되지 않는다."""
    backend = make_mock(freeze_state_after_s=1e-9)
    target = list(HOME)
    target[0] += 5.0

    assert backend.servo_j(target, 0.008, 1) == 0
    assert backend.joint_positions_deg()[0] == pytest.approx(HOME[0])


def test_servo_end_zeroes_velocity():
    backend = make_mock()
    target = list(HOME)
    target[0] += 1.0
    backend.servo_j(target, 0.008, 1)

    backend.servo_end()

    assert backend.joint_velocities_deg_s() == [0.0] * 6


def test_make_backend_rejects_unknown_kind():
    """실로봇을 의도했는데 조용히 mock 이 도는 것이 가장 위험하다."""
    with pytest.raises(ValueError, match="알 수 없는 백엔드"):
        make_backend("realrobot", "192.168.58.3")


def test_make_backend_builds_mock():
    assert isinstance(make_backend("mock", "192.168.58.3"), MockBackend)
