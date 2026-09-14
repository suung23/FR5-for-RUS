"""텔레옵 힘 한계 감시 — 조절기가 안 도는 구간의 5 N 층.

이 층이 생긴 이유는 **있다고 적힌 보호가 없었기** 때문이다 (2026-09-14).
``safety.max_contact_force_n`` 후퇴는 ForceRegulator 안에 있고, 조절기는 접촉
프로빙에서만 돈다. 2026-09-11 에 힘 판정이 꺼지면서 그 모드의 유일한 문이
policy.enable 버튼이 됐고, 버튼 전 텔레옵 구간에는 힘을 보는 코드가 없었다.

그래서 이 파일이 재는 것은 "한계에서 물러나는가" 하나가 아니다. 안전층이 조작자를
가두지 않는가(탈출은 항상 더 빠를 수 있다), 고장 났을 때 무한히 물러나지 않는가,
그리고 **되돌려준 뒤 곧바로 다시 걸리지 않는가** 를 함께 본다. 마지막 항이 핵심이다:
팬텀은 점탄성이라 힘이 압입을 1.5~2.4 s 뒤따르므로, 이력 없이 만든 안전층은
한계 위에서 진동하는 장치가 된다.
"""
from fr5_ik.force_limit_override import (
    HOLD, PASS, RETREAT, ForceLimitOverride,
)
import pytest

LIMIT = 5.0
WARN = 4.5
RESUME = 3.0
SPEED = 0.005
CAP = 3.0

PUSH = [0.0, 0.0, 0.01, 0.0, 0.0, 0.0]        # 조작자가 전진
PULL = [0.0, 0.0, -0.002, 0.0, 0.0, 0.0]      # 조작자가 후퇴 (감시보다 느리게)
SIDE = [0.01, 0.02, 0.0, 0.1, 0.2, 0.3]       # 옆으로·비틀기만


def _ov(**kw):
    args = dict(limit_n=LIMIT, resume_n=RESUME, warn_n=WARN,
                retreat_speed_m_s=SPEED, max_override_s=CAP)
    args.update(kw)
    return ForceLimitOverride(**args)


# -- 불변식 -----------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(resume_n=4.8),                    # 해제선이 경고선 위
    dict(warn_n=5.5),                       # 경고선이 한계 위
    dict(resume_n=5.0),                     # 해제선 = 한계
    dict(retreat_speed_m_s=0.0),
    dict(retreat_speed_m_s=-0.005),
    dict(max_override_s=0.0),
])
def test_rejects_contradictory_settings(bad):
    """만들 때 막는다. 해제선이 경고선 위면 '돌려줬다' 는 말과 함께 못 움직인다."""
    with pytest.raises(ValueError):
        _ov(**bad)


# -- 통과 -------------------------------------------------------------------

def test_below_warn_passes_untouched():
    out = _ov().update(PUSH, 1.0, 0.01)
    assert out.twist == PUSH
    assert out.state == PASS
    assert not out.overriding


def test_disabled_passes_even_over_limit():
    """꺼 두면 아무것도 하지 않는다 — 배리어도 함께 꺼진다."""
    out = _ov(enabled=False).update(PUSH, 99.0, 0.01)
    assert out.twist == PUSH
    assert not out.overriding


# -- 배리어 -----------------------------------------------------------------

def test_barrier_blocks_advance_at_warn():
    out = _ov().update(PUSH, WARN, 0.01)
    assert out.twist[2] == 0.0
    assert out.overriding
    assert out.state == PASS      # 후퇴가 아니다. 전진만 막았다


def test_barrier_leaves_retreat_alone():
    """빼는 방향은 손대지 않는다. 안전층이 탈출을 막으면 안전층이 아니다."""
    out = _ov().update(PULL, 4.9, 0.01)
    assert out.twist == PULL
    assert not out.overriding


def test_barrier_leaves_other_axes_alone():
    """경고선은 침투축의 사건이다. 옆·회전은 조작자의 것으로 남는다."""
    out = _ov().update(SIDE, 4.9, 0.01)
    assert out.twist == SIDE


def test_barrier_can_be_disabled_without_losing_the_limit():
    ov = _ov(barrier_enabled=False)
    assert ov.update(PUSH, 4.9, 0.01).twist == PUSH      # 배리어 없음
    assert ov.update(PUSH, LIMIT, 0.01).state == RETREAT  # 한계는 살아 있다


# -- 한계 초과 --------------------------------------------------------------

def test_over_limit_overrides_operator():
    out = _ov().update(PUSH, LIMIT, 0.01)
    assert out.state == RETREAT
    assert out.overriding
    assert out.twist == [0.0, 0.0, -SPEED, 0.0, 0.0, 0.0]


def test_over_limit_closes_lateral_and_rotation():
    """한계를 넘은 힘으로 옆으로 끄는 것은 누르고 있는 것보다 나을 것이 없다."""
    out = _ov().update(SIDE, 6.0, 0.01)
    assert out.twist == [0.0, 0.0, -SPEED, 0.0, 0.0, 0.0]


def test_operator_may_escape_faster_never_slower():
    ov = _ov()
    fast = ov.update([0.0, 0.0, -0.02, 0.0, 0.0, 0.0], 6.0, 0.01)
    assert fast.twist[2] == -0.02        # 조작자가 더 빠르다 — 통과
    ov.reset()
    slow = ov.update(PULL, 6.0, 0.01)
    assert slow.twist[2] == -SPEED       # 조작자가 더 느리다 — 감시 속도


# -- 이력 -------------------------------------------------------------------

def test_stays_retreating_between_resume_and_limit():
    """한계 아래로 내려왔다고 풀리지 않는다. 풀리면 곧바로 다시 걸린다."""
    ov = _ov()
    ov.update(PUSH, LIMIT, 0.01)
    for force in (4.9, 4.0, 3.5, 3.01):
        out = ov.update(PUSH, force, 0.01)
        assert out.state == RETREAT, force
        assert out.twist[2] == -SPEED


def test_releases_at_resume():
    ov = _ov()
    ov.update(PUSH, LIMIT, 0.01)
    out = ov.update(PUSH, RESUME, 0.01)
    assert out.state == PASS
    assert not out.overriding
    assert out.twist == PUSH
    assert ov.elapsed_s == 0.0


def test_no_ping_pong_when_operator_keeps_pushing():
    """돌려준 뒤 조작자가 계속 밀어도 한계를 다시 넘지 않는다.

    되돌려준 힘(3.0 N)에서 조작자가 전진을 유지한다. 해제선과 경고선 사이는 조작자의
    구간이므로 전진이 열려 있고(그래서 이 층이 조작을 가두지 않는다), 경고선에 닿는
    순간 배리어가 전진을 끊는다. 후퇴로 다시 떨어지는 일 없이 그 자리에 선다 —
    안전층이 한계 위에서 진동하지 않는다.
    """
    ov = _ov()
    ov.update(PUSH, LIMIT, 0.01)
    assert ov.update(PUSH, RESUME, 0.01).state == PASS
    for force in (3.5, 4.0):
        out = ov.update(PUSH, force, 0.01)
        assert out.state == PASS and out.twist == PUSH, force
    for force in (WARN, 4.9):
        out = ov.update(PUSH, force, 0.01)
        assert out.state == PASS, force
        assert out.twist[2] == 0.0, force


# -- 시간 상한 --------------------------------------------------------------

def test_gives_up_and_stops_when_force_will_not_drop():
    """물러나는데 힘이 안 떨어지면 더 물러나는 것은 답이 아니다."""
    ov = _ov()
    ov.update(PUSH, 6.0, 0.01)
    out = None
    for _ in range(400):                       # 4 s > 상한 3 s
        out = ov.update(PUSH, 6.0, 0.01)
    assert out.state == HOLD
    assert out.twist == [0.0] * 6
    assert out.overriding


def test_hold_releases_when_force_finally_drops():
    ov = _ov()
    ov.update(PUSH, 6.0, 0.01)
    for _ in range(400):
        ov.update(PUSH, 6.0, 0.01)
    assert ov.state == HOLD
    out = ov.update(PUSH, 2.0, 0.01)
    assert out.state == PASS
    assert out.twist == PUSH


def test_retreat_does_not_time_out_early():
    ov = _ov()
    ov.update(PUSH, 6.0, 0.01)
    for _ in range(299):                       # 상한 직전까지
        out = ov.update(PUSH, 6.0, 0.01)
    assert out.state == RETREAT


# -- wrench 두절 ------------------------------------------------------------

def test_stale_wrench_while_passing_does_not_block():
    """힘을 모를 때 로봇을 세우는 것은 이 층의 일이 아니다 (상위 워치독의 몫)."""
    out = _ov().update(PUSH, 0.0, 0.01, wrench_fresh=False)
    assert out.twist == PUSH
    assert not out.overriding


def test_stale_wrench_cannot_trigger():
    out = _ov().update(PUSH, 99.0, 0.01, wrench_fresh=False)
    assert out.state == PASS


def test_stale_wrench_while_retreating_keeps_retreating():
    """'풀렸다' 는 판단은 힘으로만 한다. 무한 후퇴는 시간 상한이 막는다."""
    ov = _ov()
    ov.update(PUSH, 6.0, 0.01)
    out = ov.update(PUSH, 0.0, 0.01, wrench_fresh=False)
    assert out.state == RETREAT
    assert out.twist[2] == -SPEED
    for _ in range(400):
        out = ov.update(PUSH, 0.0, 0.01, wrench_fresh=False)
    assert out.state == HOLD


# -- 초기화 -----------------------------------------------------------------

def test_reset_clears_state_and_clock():
    ov = _ov()
    ov.update(PUSH, 6.0, 0.01)
    ov.update(PUSH, 6.0, 1.0)
    assert ov.elapsed_s > 0.0
    ov.reset()
    assert ov.state == PASS
    assert ov.elapsed_s == 0.0
