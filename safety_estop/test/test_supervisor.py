"""상태기계 — 단계 전이와 그 전이를 막는 것들."""

import math

import pytest

from safety_estop.hardware import HardwareLoop
from safety_estop.recovery import Phase, RecoveryLimits
from safety_estop.supervisor import EstopSupervisor
from safety_estop.trip import TripLimits

RAD2DEG = 180.0 / math.pi
LOWER = [-175.0, -265.0, -160.0, -265.0, -175.0, -165.0]
UPPER = [175.0, 85.0, 160.0, 85.0, 175.0, 165.0]


def pose(j1):
    return [j1, -90.0, 0.0, -90.0, 0.0, 0.0]


def quiet_hw(t=0.0, present=False, **kw):
    return HardwareLoop(present=present, estop=kw.get("estop", False),
                        stop0=kw.get("stop0", False), stop1=kw.get("stop1", False), at=t)


def make(**trip_kw):
    sup = EstopSupervisor(
        TripLimits(force_trip_n=8.0, force_confirm_s=0.0, **trip_kw),
        RecoveryLimits(),
        (LOWER, UPPER),
        trail_step_deg=1.0,
    )
    return sup


def tick(sup, t, force=0.0, joints=None, hw=None):
    return sup.update(
        t, force_n=force, force_at=t,
        joints_deg=joints if joints is not None else pose(0.0), joints_at=t,
        hardware=hw or quiet_hw(t))


def arm(sup, t=0.0, joints=None):
    reason = sup.arm(t, force_at=t, joints_at=t, joints_deg=joints or pose(0.0))
    assert reason is None, reason
    return sup


# -- 켜기 ----------------------------------------------------------------


def test_starts_disarmed_and_claims_nothing():
    sup = make()
    assert sup.phase is Phase.DISARMED
    d = tick(sup, 0.0)
    assert d.inhibit is False


def test_arming_captures_the_pose_to_return_to():
    sup = arm(make(), joints=pose(12.0))
    assert sup.phase is Phase.ARMED
    assert sup.recorder.home_deg == pose(12.0)


def test_arming_without_joint_state_is_refused():
    sup = make()
    reason = sup.arm(0.0, force_at=0.0, joints_at=0.0, joints_deg=None)
    assert reason is not None and "돌아갈 자리" in reason
    assert sup.phase is Phase.DISARMED


# -- 걸기 ----------------------------------------------------------------


def test_force_trip_inhibits_motion():
    sup = arm(make())
    assert tick(sup, 0.1, force=1.0).inhibit is False
    d = tick(sup, 0.2, force=9.0)
    assert d.phase is Phase.TRIPPED and d.inhibit is True
    assert any("정지" in e for e in d.events)


def test_joint_trip_inhibits_motion():
    sup = arm(make(joint_reach_trip=0.90, joint_confirm_s=0.0))
    d = tick(sup, 0.1, joints=[0.0, 67.5, 0.0, -90.0, 0.0, 0.0])
    assert d.phase is Phase.TRIPPED and d.inhibit is True


def test_recovery_is_never_automatic():
    sup = arm(make())
    tick(sup, 0.1, force=9.0)
    for k in range(50):                       # 10 초를 돌려도
        d = tick(sup, 0.2 + k * 0.2, force=0.0)
    assert d.phase is Phase.TRIPPED           # 스스로 복귀하지 않는다
    assert d.recovery_velocity is None


def test_trail_is_recorded_only_while_armed():
    sup = make()
    tick(sup, 0.1, joints=pose(10.0))         # DISARMED — 기록하지 않는다
    assert len(sup.recorder) == 0
    arm(sup, 0.2, joints=pose(10.0))
    tick(sup, 0.3, joints=pose(20.0))
    assert len(sup.recorder) == 2


# -- 복귀 ----------------------------------------------------------------


def drive(sup, t0, *, force, steps=2000, dt=0.05, joints=None):
    """이상적인 로봇으로 복귀를 끝까지 돌린다. 마지막 결정을 돌려준다."""
    q = list(joints)
    t = t0
    for _ in range(steps):
        t += dt
        f = force(t) if callable(force) else force
        d = sup.update(t, force_n=f, force_at=t, joints_deg=q, joints_at=t,
                       hardware=quiet_hw(t))
        if d.recovery_velocity is not None:
            for i in range(6):
                q[i] += d.recovery_velocity[i] * RAD2DEG * dt
        if d.phase in (Phase.SAFE, Phase.TRIPPED):
            return d, q
    raise AssertionError("복귀가 끝나지 않았다")


def test_full_sequence_returns_to_the_initial_pose_and_stays_inhibited():
    sup = arm(make(), joints=pose(0.0))
    for k, j in enumerate([5.0, 10.0, 15.0, 20.0]):
        tick(sup, 0.1 * (k + 1), joints=pose(j))
    tick(sup, 1.0, force=9.0, joints=pose(20.0))
    assert sup.phase is Phase.TRIPPED

    assert sup.begin_recovery(1.1) is None
    assert sup.phase is Phase.RELEASING

    d, q = drive(sup, 1.1, force=0.0, joints=pose(20.0))
    assert d.phase is Phase.SAFE
    assert q[0] == pytest.approx(0.0, abs=0.5)
    # 복귀가 끝나도 금지는 풀리지 않는다.
    assert d.inhibit is True


def test_releasing_becomes_returning_when_the_force_drops():
    sup = arm(make(), joints=pose(0.0))
    tick(sup, 0.1, joints=pose(20.0))
    tick(sup, 0.2, force=9.0, joints=pose(20.0))
    sup.begin_recovery(0.3)
    d = sup.update(0.35, force_n=3.0, force_at=0.35, joints_deg=pose(20.0),
                   joints_at=0.35, hardware=quiet_hw(0.35))
    assert d.phase is Phase.RELEASING
    d = sup.update(0.40, force_n=0.1, force_at=0.40, joints_deg=pose(19.9),
                   joints_at=0.40, hardware=quiet_hw(0.40))
    assert d.phase is Phase.RETURNING


def test_rising_force_during_recovery_aborts_it():
    sup = arm(make(force_instant_n=15.0), joints=pose(0.0))
    tick(sup, 0.1, joints=pose(20.0))
    tick(sup, 0.2, force=9.0, joints=pose(20.0))
    sup.begin_recovery(0.3)
    d = sup.update(0.4, force_n=16.0, force_at=0.4, joints_deg=pose(20.0),
                   joints_at=0.4, hardware=quiet_hw(0.4))
    assert d.phase is Phase.TRIPPED
    assert "빠지는 방향이 아니다" in sup.halt_reason


def test_lost_joint_state_during_recovery_aborts_it():
    sup = arm(make(), joints=pose(0.0))
    tick(sup, 0.1, joints=pose(20.0))
    tick(sup, 0.2, force=9.0, joints=pose(20.0))
    sup.begin_recovery(0.3)
    d = sup.update(0.4, force_n=0.0, force_at=0.4, joints_deg=None, joints_at=None,
                   hardware=quiet_hw(0.4))
    assert d.phase is Phase.TRIPPED and "어디 있는지 모른 채" in sup.halt_reason


def test_stale_force_trip_refuses_automatic_recovery():
    sup = arm(make(stale_s=0.25), joints=pose(0.0))
    tick(sup, 0.1, joints=pose(20.0))
    # 힘 표본이 끊겼다.
    sup.update(1.0, force_n=1.0, force_at=0.1, joints_deg=pose(20.0), joints_at=1.0,
               hardware=quiet_hw(1.0))
    assert sup.phase is Phase.TRIPPED
    reason = sup.begin_recovery(1.1)
    assert reason is not None and "자동 복귀를 허용하지 않는다" in reason


# -- 물리 회로 ------------------------------------------------------------


def test_hardware_assertion_blocks_motion_and_recovery():
    sup = arm(make(), joints=pose(0.0))
    hw = quiet_hw(1.0, present=True, estop=True)
    d = sup.update(1.0, force_n=0.0, force_at=1.0, joints_deg=pose(0.0), joints_at=1.0,
                   hardware=hw)
    assert d.phase is Phase.TRIPPED and d.inhibit is True
    sup.hardware = hw.evaluate(1.0)
    assert "비상정지" in (sup.begin_recovery(1.1) or "")


def test_declared_hardware_that_cannot_be_read_blocks_motion():
    # 회로가 있다고 선언했는데 못 읽는다 = 고장. "안 눌렸다" 가 아니다.
    sup = arm(make(), joints=pose(0.0))
    stale = HardwareLoop(present=True, estop=False, stop0=False, stop1=False, at=0.0)
    d = sup.update(5.0, force_n=0.0, force_at=5.0, joints_deg=pose(0.0), joints_at=5.0,
                   hardware=stale)
    assert d.inhibit is True


def test_channel_disagreement_is_a_fault_not_a_clear_circuit():
    loop = HardwareLoop(present=True, estop=False, stop0=True, stop1=False, at=1.0)
    state = loop.evaluate(1.0)
    assert state.asserted and state.channels_disagree
    assert "배선 고장" in state.reason


def test_absent_hardware_is_reported_but_does_not_block():
    state = HardwareLoop(present=False, estop=False, stop0=False, stop1=False,
                         at=1.0).evaluate(1.0)
    assert state.blocks_motion is False
    assert "비상정지가 없다" in state.reason


# -- 리셋 ----------------------------------------------------------------


def test_reset_goes_to_disarmed_not_armed():
    sup = arm(make())
    tick(sup, 0.1, force=9.0)
    assert sup.reset(1.0, force_n=0.1, joints_deg=pose(0.0)) is None
    assert sup.phase is Phase.DISARMED
    assert sup.monitor.armed is False          # 리셋은 재기동이 아니다


def test_reset_is_refused_mid_recovery():
    sup = arm(make(), joints=pose(0.0))
    tick(sup, 0.1, joints=pose(20.0))
    tick(sup, 0.2, force=9.0, joints=pose(20.0))
    sup.begin_recovery(0.3)
    assert "진행 중" in (sup.reset(0.4, force_n=0.0, joints_deg=pose(20.0)) or "")


def test_disarm_cannot_erase_a_trip():
    sup = arm(make())
    tick(sup, 0.1, force=9.0)
    assert "걸려 있는 정지를 끄는 길은 없다" in (sup.disarm() or "")


def test_console_stop_is_not_an_emergency_stop():
    sup = arm(make())
    sup.stop(0.5, "콘솔 STOP")
    latched = sup.monitor.latched
    assert latched is not None and latched.source == "operator"
    assert latched.allows_release is True      # 사람이 세웠으니 사람이 복귀도 지시한다
