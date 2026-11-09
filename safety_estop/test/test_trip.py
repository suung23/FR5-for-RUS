"""정지 판정 — 문턱·확인창·래치·해제."""

import pytest

from safety_estop.trip import (
    SOURCE_FORCE,
    SOURCE_FORCE_STALE,
    SOURCE_HARDWARE,
    SOURCE_JOINT,
    TripLimits,
    TripMonitor,
    joint_reach,
)

LOWER = [-175.0, -265.0, -160.0, -265.0, -175.0, -165.0]
UPPER = [175.0, 85.0, 160.0, 85.0, 175.0, 165.0]
LIMITS = (LOWER, UPPER)


def monitor(**kw) -> TripMonitor:
    m = TripMonitor(TripLimits(**kw), LIMITS)
    assert m.arm(0.0, force_at=0.0, joints_at=0.0) is None
    return m


def feed(m, t, force=0.0, joints=None):
    return m.update(t, force_n=force, force_at=t, joints_deg=joints or [0.0] * 6, joints_at=t)


# -- 관절 여유 ------------------------------------------------------------


def test_reach_is_symmetric_on_an_asymmetric_axis():
    # J2 는 [-265, 85]. 중앙은 -90°, 반폭은 175°. 어느 쪽 끝이든 1.0 이어야 한다.
    assert joint_reach(85.0, -265.0, 85.0).reach == pytest.approx(1.0)
    assert joint_reach(-265.0, -265.0, 85.0).reach == pytest.approx(1.0)
    assert joint_reach(-90.0, -265.0, 85.0).reach == pytest.approx(0.0)
    # 0.9 가 걸리는 자리는 양쪽에 하나씩 있다.
    assert joint_reach(67.5, -265.0, 85.0).reach == pytest.approx(0.9)
    assert joint_reach(-247.5, -265.0, 85.0).reach == pytest.approx(0.9)


def test_margin_points_at_the_bound_being_approached():
    r = joint_reach(67.5, -265.0, 85.0)
    assert r.toward_upper and r.margin_deg == pytest.approx(17.5)
    r = joint_reach(-247.5, -265.0, 85.0)
    assert not r.toward_upper and r.margin_deg == pytest.approx(17.5)


def test_naive_fraction_of_upper_would_miss_the_lower_side():
    # 이 시험이 reach 정의의 존재 이유다. |q| >= 0.9*upper 로 셌다면 q = -247.5°
    # (하한까지 17.5° 남음)는 0.9*85 = 76.5 보다 크므로 걸리긴 하지만, q = -100°
    # (하한까지 165° 남고 안전하다)도 함께 걸린다.
    assert joint_reach(-100.0, -265.0, 85.0).reach == pytest.approx(0.0571, abs=1e-3)


# -- 힘 ------------------------------------------------------------------


def test_force_needs_the_confirm_window():
    m = monitor(force_trip_n=8.0, force_confirm_s=0.05)
    assert feed(m, 0.00, force=9.0) is None      # 넘기 시작
    assert feed(m, 0.03, force=9.0) is None      # 아직 확인창 안
    trip = feed(m, 0.06, force=9.0)
    assert trip is not None and trip.source == SOURCE_FORCE


def test_a_single_spike_does_not_trip():
    m = monitor(force_trip_n=8.0, force_confirm_s=0.05)
    assert feed(m, 0.00, force=9.0) is None
    assert feed(m, 0.01, force=2.0) is None
    assert feed(m, 0.04, force=9.0) is None      # 창이 다시 0 에서 시작한다
    assert feed(m, 0.06, force=9.0) is None
    assert feed(m, 0.10, force=9.0) is not None


def test_instant_value_skips_the_window():
    m = monitor(force_trip_n=8.0, force_instant_n=15.0, force_confirm_s=0.5)
    trip = feed(m, 0.0, force=16.0)
    assert trip is not None and "즉시" in trip.detail


def test_force_sign_does_not_matter():
    # 접촉력은 크기다 (probe.yaml ft_sensor.contact_force_mode: magnitude).
    m = monitor(force_trip_n=8.0, force_confirm_s=0.0)
    assert feed(m, 0.0, force=-9.0) is not None


# -- 관절 정지 ------------------------------------------------------------


def test_joint_trips_at_ninety_percent():
    m = monitor(joint_reach_trip=0.90, joint_confirm_s=0.0)
    safe = [0.0, -90.0, 0.0, -90.0, 0.0, 0.0]        # 전부 중앙 근처
    assert feed(m, 0.0, joints=safe) is None
    near = list(safe)
    near[1] = 67.5                                     # J2 가 90 %
    trip = feed(m, 0.1, joints=near)
    assert trip is not None and trip.source == SOURCE_JOINT and "J2" in trip.detail


def test_joint_detail_names_the_remaining_angle():
    m = monitor(joint_reach_trip=0.90, joint_confirm_s=0.0)
    joints = [0.0, -90.0, 0.0, -90.0, 0.0, 160.0]      # J6 는 ±165
    trip = feed(m, 0.0, joints=joints)
    assert trip is not None and "J6" in trip.detail and "5.0°" in trip.detail


# -- 신선도 --------------------------------------------------------------


def test_stale_force_trips_and_forbids_automatic_release():
    m = monitor(stale_s=0.25)
    trip = m.update(1.0, force_n=1.0, force_at=0.5, joints_deg=[0.0] * 6, joints_at=1.0)
    assert trip is not None and trip.source == SOURCE_FORCE_STALE
    # 힘을 못 믿는 상태에서 자동 후퇴는 허용하지 않는다.
    assert trip.allows_release is False


def test_nan_force_is_not_a_small_force():
    m = monitor()
    trip = feed(m, 0.0, force=float("nan"))
    assert trip is not None and trip.source == SOURCE_FORCE_STALE


def test_arming_requires_fresh_samples():
    m = TripMonitor(TripLimits(), LIMITS)
    assert "관절" in (m.arm(0.0, force_at=0.0, joints_at=None) or "")
    assert "힘" in (m.arm(0.0, force_at=None, joints_at=0.0) or "")
    assert m.armed is False


# -- 래치와 해제 ----------------------------------------------------------


def test_latch_survives_the_condition_going_away():
    m = monitor(force_trip_n=8.0, force_confirm_s=0.0)
    trip = feed(m, 0.0, force=9.0)
    assert trip is not None
    still = feed(m, 1.0, force=0.1)
    assert still is trip                       # 조건은 사라져도 래치는 남는다


def test_clear_refuses_while_the_probe_is_still_pressed():
    m = monitor(force_trip_n=8.0, force_confirm_s=0.0)
    feed(m, 0.0, force=9.0)
    reason = m.clear(1.0, force_n=9.0, joints_deg=[0.0] * 6, hardware_asserted=False)
    assert reason is not None and "닿아 있다" in reason
    assert m.latched is not None
    assert m.clear(2.0, force_n=0.1, joints_deg=[0.0] * 6, hardware_asserted=False) is None
    assert m.latched is None


def test_clear_refuses_while_the_button_is_still_pressed():
    m = monitor()
    m.update(0.0, force_n=0.0, force_at=0.0, joints_deg=[0.0] * 6, joints_at=0.0,
             hardware_asserted=True)
    assert m.latched is not None and m.latched.source == SOURCE_HARDWARE
    reason = m.clear(1.0, force_n=0.0, joints_deg=[0.0] * 6, hardware_asserted=True)
    assert reason is not None and "돌려 뽑고" in reason


def test_hardware_latches_even_when_disarmed():
    # 감시가 꺼져 있어도 물리 회로는 기록한다 — 판정이 아니라 관측이기 때문이다.
    m = TripMonitor(TripLimits(), LIMITS)
    assert m.armed is False
    trip = m.update(0.0, hardware_asserted=True)
    assert trip is not None and trip.source == SOURCE_HARDWARE


def test_cannot_raise_the_threshold_out_of_a_trip():
    m = monitor(force_trip_n=8.0, force_confirm_s=0.0)
    feed(m, 0.0, force=9.0)
    with pytest.raises(ValueError, match="걸려 있는 동안"):
        m.set_limits(TripLimits(force_trip_n=12.0))


def test_arm_refuses_while_latched():
    m = monitor(force_trip_n=8.0, force_confirm_s=0.0)
    feed(m, 0.0, force=9.0)
    assert "RESET" in (m.arm(1.0, force_at=1.0, joints_at=1.0) or "")


# -- 설정 규칙 ------------------------------------------------------------


def test_ceiling_is_not_raisable_from_the_console_side():
    with pytest.raises(ValueError, match="천장"):
        TripLimits(force_trip_n=25.0, force_instant_n=30.0, force_ceiling_n=20.0).validated()


def test_instant_below_the_window_threshold_is_refused():
    with pytest.raises(ValueError, match="확인창이 무의미"):
        TripLimits(force_trip_n=10.0, force_instant_n=5.0).validated()


def test_joint_threshold_cannot_sit_at_the_limit():
    with pytest.raises(ValueError):
        TripLimits(joint_reach_trip=1.0, joint_reach_ceiling=0.98).validated()
