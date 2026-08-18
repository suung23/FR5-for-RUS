"""상태 머신과 복구 매핑의 단위 시험 (DESIGN_NOTES §10)."""
import pytest

from fr5_control.supervisor import (
    Mode,
    State,
    Supervisor,
    SupervisorConfig,
    SupervisorInputs,
    REASON_TO_RECOVERY,
    STATE_TO_MODE,
)

DT = 0.05  # 20 Hz


def config(**kwargs) -> SupervisorConfig:
    base = dict(contact_hold_s=0.2, valid_frames_enter_focus=3,
                invalid_frames_leave_scan=3, q_raw_degrade_hold_s=0.2)
    base.update(kwargs)
    return SupervisorConfig(**base)


def drive(sup: Supervisor, ticks: int, **kwargs) -> State:
    for _ in range(ticks):
        sup.step(DT, SupervisorInputs(**kwargs))
    return sup.state


def reach_scan(sup: Supervisor) -> None:
    """정상 경로로 STAGE2_SCAN 까지 올린다."""
    sup.step(DT, SupervisorInputs(operator_engage=True))
    drive(sup, 6, normal_force=1.0, operator_confirm=True)
    assert sup.state is State.STAGE1A_CONTACT
    drive(sup, 5, normal_force=4.0, q_raw=0.8, valid_for_control=True)
    assert sup.state is State.STAGE1B_FOCUS
    sup.step(DT, SupervisorInputs(normal_force=4.0, q_raw=0.8, search_done=True))
    assert sup.state is State.STAGE2_SCAN


# --------------------------------------------------------------------------
# 모드 계약 — admittance 가 유일한 발행자
# --------------------------------------------------------------------------

def test_every_state_maps_to_a_mode():
    """상태가 늘어나도 admittance 는 네 모드만 알면 된다."""
    assert set(STATE_TO_MODE) == set(State)


def test_teleop_state_gives_passthrough_mode():
    """TELEOP_APPROACH 에서 조작자는 z 를 포함한 6축이 필요하다."""
    sup = Supervisor(config())
    sup.step(DT, SupervisorInputs(operator_engage=True))

    assert sup.state is State.TELEOP_APPROACH
    assert sup.mode is Mode.TELEOP


def test_all_stage_states_share_contact_mode():
    for state in (State.STAGE1A_CONTACT, State.STAGE1B_FOCUS, State.STAGE2_SCAN):
        assert STATE_TO_MODE[state] is Mode.CONTACT


# --------------------------------------------------------------------------
# 정상 진행
# --------------------------------------------------------------------------

def test_contact_alone_does_not_start_automatic_control():
    """접촉 감지만으로 넘어가면 안 된다. 조작자 확인이 필요하다."""
    sup = Supervisor(config())
    sup.step(DT, SupervisorInputs(operator_engage=True))

    drive(sup, 20, normal_force=1.0, operator_confirm=False)

    assert sup.state is State.TELEOP_APPROACH


def test_normal_progression_to_scan():
    sup = Supervisor(config())
    reach_scan(sup)
    assert sup.mode is Mode.CONTACT


def test_quality_gate_blocks_focus_entry():
    """Q_raw 가 낮으면 유효 프레임이 쌓여도 Stage 1b 로 넘어가지 않는다."""
    sup = Supervisor(config(q_raw_enter_focus=0.6))
    sup.step(DT, SupervisorInputs(operator_engage=True))
    drive(sup, 6, normal_force=1.0, operator_confirm=True)

    drive(sup, 20, normal_force=4.0, q_raw=0.3, valid_for_control=True)

    assert sup.state is State.STAGE1A_CONTACT


# --------------------------------------------------------------------------
# 안전
# --------------------------------------------------------------------------

def test_safety_violation_preempts_from_any_state():
    for setup in (lambda s: None, reach_scan):
        sup = Supervisor(config())
        setup(sup)
        sup.step(DT, SupervisorInputs(safety_violation="force limit"))
        assert sup.state is State.RETREAT
        assert sup.mode is Mode.RETREAT


def test_retreat_requires_both_low_force_and_confirmation():
    sup = Supervisor(config())
    sup.step(DT, SupervisorInputs(safety_violation="force limit"))

    drive(sup, 5, normal_force=0.01, operator_confirm=False)
    assert sup.state is State.RETREAT

    drive(sup, 5, normal_force=5.0, operator_confirm=True)
    assert sup.state is State.RETREAT

    sup.step(DT, SupervisorInputs(normal_force=0.01, operator_confirm=True))
    assert sup.state is State.TELEOP_APPROACH


def test_retreat_threshold_below_contact_threshold_prevents_chatter():
    """후퇴 완료 힘이 접촉 판정 힘보다 크면 두 상태를 오간다."""
    with pytest.raises(ValueError, match="작아야"):
        SupervisorConfig(contact_force_n=0.2, retreat_force_n=0.5)


# --------------------------------------------------------------------------
# 복구 매핑
# --------------------------------------------------------------------------

def test_contact_failure_returns_further_back_than_lumen_loss():
    """접촉 불량은 Stage 1a 로, 방광 소실은 Stage 1b 로."""
    assert REASON_TO_RECOVERY["low_segmentation_confidence"] is State.STAGE1A_CONTACT
    assert REASON_TO_RECOVERY["empty_mask"] is State.STAGE1B_FOCUS


def test_in_plane_reasons_do_not_transition():
    """면내 문제는 policy 몫이지 상태 전이 사유가 아니다."""
    for reason in ("excessive_border_contact", "excessive_shadowing"):
        assert REASON_TO_RECOVERY[reason] is None


def test_recovery_picks_the_earliest_stage():
    """여러 사유가 겹치면 가장 앞 단계로 되돌린다.

    접촉이 나쁜데 방광만 다시 찾아봐야 소용없다.
    """
    sup = Supervisor(config())
    target = sup.request_recovery(["empty_mask", "low_segmentation_confidence"])

    assert target is State.STAGE1A_CONTACT


def test_recovery_returns_none_when_no_reason_maps():
    sup = Supervisor(config())
    assert sup.request_recovery(["excessive_border_contact"]) is None
    assert sup.request_recovery([]) is None


def test_scan_leaves_on_sustained_invalid_frames():
    sup = Supervisor(config(invalid_frames_leave_scan=3))
    reach_scan(sup)

    drive(sup, 3, normal_force=4.0, valid_for_control=False,
          rejection_reasons=["empty_mask"])

    assert sup.state is State.STAGE1B_FOCUS


def test_repeated_recovery_failure_returns_to_operator():
    """자동 복구를 무한히 시도하지 않는다."""
    sup = Supervisor(config(invalid_frames_leave_scan=1, max_recovery_attempts=2))
    reach_scan(sup)

    for _ in range(3):
        sup.state = State.STAGE2_SCAN
        sup.step(DT, SupervisorInputs(valid_for_control=False,
                                      rejection_reasons=["empty_mask"]))

    assert sup.state is State.TELEOP_APPROACH


def test_q_raw_degradation_triggers_force_research():
    """Q_raw 열화는 접촉 문제이므로 힘 재탐색으로 돌아간다 (§5.3)."""
    sup = Supervisor(config(q_raw_degrade_ratio=0.25, q_raw_degrade_hold_s=0.2))
    reach_scan(sup)
    assert sup.q_raw_reference == pytest.approx(0.8)

    drive(sup, 6, normal_force=4.0, q_raw=0.5, valid_for_control=True)

    assert sup.state is State.STAGE1B_FOCUS


def test_brief_q_raw_dip_does_not_trigger():
    """일시적 저하로 재탐색에 들어가면 스캔이 끊긴다."""
    sup = Supervisor(config(q_raw_degrade_hold_s=1.0))
    reach_scan(sup)

    drive(sup, 2, normal_force=4.0, q_raw=0.5, valid_for_control=True)
    drive(sup, 10, normal_force=4.0, q_raw=0.8, valid_for_control=True)

    assert sup.state is State.STAGE2_SCAN


def test_search_failure_returns_to_operator():
    sup = Supervisor(config())
    sup.step(DT, SupervisorInputs(operator_engage=True))
    drive(sup, 6, normal_force=1.0, operator_confirm=True)

    sup.step(DT, SupervisorInputs(normal_force=4.0, search_failed=True))

    assert sup.state is State.TELEOP_APPROACH
    assert "실패" in sup.last_transition
