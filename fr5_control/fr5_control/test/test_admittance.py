"""Admittance 제어 법칙과 wrench 프레임 이동의 단위 시험 (DESIGN_NOTES §4.3, §8).

rclpy 없이 돈다. 여기서 잡는 것은 **부호와 결합 구조** 다 — 실기에서 틀리면 프로브가
조직을 파고들거나 정렬이 오정렬을 키우는 방향으로 도는 오류들이다.
"""
import math

import numpy as np
import pytest

from fr5_control.admittance import (
    AdmittanceConfig,
    ContactSetpoint,
    SlewLimiter,
    admittance_velocity,
    compose_probe_twist,
    deadband,
)
from fr5_control.transforms import (
    probe_from_sensor,
    rotation_from_rpy,
    wrench_sensor_to_probe,
)

IDENTITY = np.eye(3)
NO_OFFSET = np.zeros(3)


# --------------------------------------------------------------------------
# wrench 프레임 이동
# --------------------------------------------------------------------------

def test_force_is_invariant_under_translation():
    """힘은 기준점 평행이동에 불변이다. 모멘트만 바뀐다."""
    wrench = [1.0, 2.0, -5.0, 0.0, 0.0, 0.0]
    offset = np.array([0.0, 0.0, 0.1])

    near = wrench_sensor_to_probe(wrench, IDENTITY, NO_OFFSET)
    far = wrench_sensor_to_probe(wrench, IDENTITY, offset)

    assert np.allclose(near[:3], far[:3])
    assert not np.allclose(near[3:], far[3:])


def test_lever_arm_term_swamps_alignment_signal():
    """§4.3 의 근거를 수치로 고정한다.

    r ≈ 0.1 m, F_z ≈ 5 N 이면 r×F 항이 0.5 N·m 다. 정렬 신호는 수십 mN·m 수준이므로
    이 변환을 빼먹으면 신호가 통째로 묻힌다.
    """
    alignment_signal = 0.02  # N·m, 실제 오정렬이 만드는 모멘트
    wrench = [0.0, 5.0, -5.0, alignment_signal, 0.0, 0.0]
    offset = np.array([0.0, 0.0, 0.1])

    moved = wrench_sensor_to_probe(wrench, IDENTITY, offset)

    lever = moved[3] - alignment_signal
    assert abs(lever) == pytest.approx(0.5, rel=0.01)
    assert abs(lever) > 20 * alignment_signal


def test_rotation_only_mount_leaves_z_force_untouched():
    """인라인 장착(센서 z ∥ 프로브 z)이면 F_z 가 그대로 넘어온다."""
    wrench = [0.0, 0.0, -4.0, 0.0, 0.0, 0.0]

    moved = wrench_sensor_to_probe(wrench, IDENTITY, np.array([0.0, 0.0, 0.05]))

    assert moved[2] == pytest.approx(-4.0)


def test_tilted_mount_mixes_lateral_force_into_normal():
    """센서를 기울여 달면 F_z^sensor 에 마찰이 섞인다 — 그래서 정의가 프로브 프레임이다."""
    rotation = rotation_from_rpy(math.radians(30.0), 0.0, 0.0)
    wrench = [0.0, 3.0, -4.0, 0.0, 0.0, 0.0]  # 측면력 3 N, 법선력 4 N

    moved = wrench_sensor_to_probe(wrench, rotation, NO_OFFSET)

    assert moved[2] != pytest.approx(-4.0, abs=0.1)
    assert np.linalg.norm(moved[:3]) == pytest.approx(5.0)  # 크기는 보존


def test_probe_from_sensor_identity_when_frames_coincide():
    rotation, offset = probe_from_sensor([0, 0, 0.1], [0, 0, 0], [0, 0, 0.1], [0, 0, 0])

    assert np.allclose(rotation, IDENTITY)
    assert np.allclose(offset, 0.0)


def test_probe_from_sensor_recovers_axial_offset():
    """센서가 프로브보다 J6 쪽으로 50 mm 뒤에 있으면 r 의 z 성분이 -0.05 여야 한다."""
    rotation, offset = probe_from_sensor([0, 0, 0.15], [0, 0, 0], [0, 0, 0.10], [0, 0, 0])

    assert np.allclose(rotation, IDENTITY)
    assert offset[2] == pytest.approx(-0.05)


# --------------------------------------------------------------------------
# 데드밴드 · 감쇠 · 슬루
# --------------------------------------------------------------------------

def test_deadband_is_continuous_at_the_threshold():
    """문턱만큼 빼는 방식이라 경계에서 출력이 튀지 않는다."""
    width = 0.1
    assert deadband(0.1, width) == pytest.approx(0.0)
    assert deadband(0.1001, width) == pytest.approx(0.0001, abs=1e-9)
    assert deadband(-0.1001, width) == pytest.approx(-0.0001, abs=1e-9)


def test_deadband_disabled_when_width_is_zero():
    assert deadband(0.03, 0.0) == pytest.approx(0.03)


def test_admittance_velocity_clamps():
    assert admittance_velocity(100.0, 1000.0, 0.010) == pytest.approx(0.010)
    assert admittance_velocity(-100.0, 1000.0, 0.010) == pytest.approx(-0.010)
    assert admittance_velocity(1.0, 1000.0, 0.010) == pytest.approx(0.001)


def test_admittance_rejects_zero_damping():
    with pytest.raises(ValueError, match="0보다 커야"):
        admittance_velocity(1.0, 0.0, 0.01)


def test_slew_limiter_bounds_the_rate():
    """§5.4: M* 계단 입력이 각속도 예산을 한 번에 소진하는 것을 막는다."""
    limiter = SlewLimiter(rate=0.05)  # N·m/s

    after_one_tick = limiter.step(0.10, dt=0.01)

    assert after_one_tick == pytest.approx(0.0005)
    assert after_one_tick < 0.10


def test_slew_limiter_reaches_target_eventually():
    limiter = SlewLimiter(rate=0.05)
    for _ in range(300):  # 3 s @ 100 Hz
        limiter.step(0.10, dt=0.01)
    assert limiter.value == pytest.approx(0.10)


def test_slew_limiter_prevents_step_in_commanded_velocity():
    """제한이 없으면 setpoint 계단이 그대로 각속도 계단이 된다 (§5.4).

    B_r = 0.5 에서 0.1 N·m 계단은 곧바로 ω = 0.2 rad/s — 각속도 예산 전체다.
    슬루를 걸면 한 틱에 실리는 변화가 rate·dt 로 묶인다.
    """
    damping, dt, target = 0.5, 0.01, 0.10

    unlimited = target / damping
    assert unlimited == pytest.approx(0.2)  # 예산 전체를 한 번에 소진

    limiter = SlewLimiter(rate=0.05)
    before = limiter.value
    after = limiter.step(target, dt)
    per_tick = (after - before) / damping

    assert per_tick == pytest.approx(0.05 * dt / damping)
    assert per_tick < unlimited / 100


def test_slew_limiter_disabled_when_rate_is_zero():
    """rate <= 0 은 제한 없음을 뜻한다. 계단이 그대로 통과해야 한다."""
    limiter = SlewLimiter(rate=0.0)
    assert limiter.step(0.10, dt=0.01) == pytest.approx(0.10)


# --------------------------------------------------------------------------
# 축 합성 — 부호와 직교성
# --------------------------------------------------------------------------

def config() -> AdmittanceConfig:
    return AdmittanceConfig(force_deadband=0.0, moment_deadband=0.0)


def test_under_force_commands_motion_into_tissue():
    """목표보다 덜 누르고 있으면 +z(침투 방향)로 가야 한다.

    부호가 반대면 프로브가 조직에서 떨어지면서 힘이 더 줄어드는 발산 루프가 된다.
    """
    wrench = [0, 0, -2.0, 0, 0, 0]  # F_n = +2 N (sign=-1)
    setpoint = ContactSetpoint(normal_force=4.0)

    twist, diag = compose_probe_twist(wrench, setpoint, np.zeros(6), config())

    assert diag["normal_force"] == pytest.approx(2.0)
    assert twist[2] > 0.0


def test_over_force_commands_retreat():
    wrench = [0, 0, -6.0, 0, 0, 0]  # F_n = +6 N
    setpoint = ContactSetpoint(normal_force=4.0)

    twist, _ = compose_probe_twist(wrench, setpoint, np.zeros(6), config())

    assert twist[2] < 0.0


def test_zero_moment_setpoint_is_pure_normal_alignment():
    wrench = [0, 0, -4.0, 0.05, -0.03, 0]
    twist, _ = compose_probe_twist(
        wrench, ContactSetpoint(normal_force=4.0), np.zeros(6), config()
    )
    assert twist[3] < 0.0  # M_x 를 0 으로 되돌리는 방향
    assert twist[4] > 0.0


def test_moment_bias_shifts_the_equilibrium():
    """§5.4: M* 를 편향시키면 그 모멘트에서 각속도가 0 이 된다 = 기울어진 평형."""
    bias = 0.04
    wrench = [0, 0, -4.0, bias, 0, 0]

    twist, _ = compose_probe_twist(
        wrench,
        ContactSetpoint(normal_force=4.0, moment_x=bias),
        np.zeros(6),
        config(),
    )

    assert twist[3] == pytest.approx(0.0, abs=1e-12)


def test_image_axes_pass_through_and_force_axes_are_overwritten():
    """면내 3축은 그대로 나가고, 힘 축에 실려온 영상 성분은 버려진다 (§8.3)."""
    image = np.array([0.005, -0.003, 999.0, 999.0, 999.0, 0.02])

    twist, _ = compose_probe_twist(
        [0, 0, -4.0, 0, 0, 0], ContactSetpoint(normal_force=4.0), image, config()
    )

    assert twist[0] == pytest.approx(0.005)
    assert twist[1] == pytest.approx(-0.003)
    assert twist[5] == pytest.approx(0.02)
    assert twist[2] != pytest.approx(999.0)
    assert twist[3] == pytest.approx(0.0)
    assert twist[4] == pytest.approx(0.0)


def test_deadband_applies_to_error_not_measurement():
    """편향 setpoint 에서 측정값이 크더라도 오차가 작으면 움직이지 않아야 한다.

    데드밴드를 측정값에 걸면 이 경우 계속 움직인다 (§8.2 경고).
    """
    cfg = AdmittanceConfig(force_deadband=0.0, moment_deadband=0.02)
    wrench = [0, 0, -4.0, 0.10, 0, 0]  # 측정 모멘트가 데드밴드의 5배

    twist, _ = compose_probe_twist(
        wrench,
        ContactSetpoint(normal_force=4.0, moment_x=0.105),  # 오차 0.005 < 0.02
        np.zeros(6),
        cfg,
    )

    assert twist[3] == pytest.approx(0.0)
