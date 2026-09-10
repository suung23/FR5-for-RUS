"""힘 탐색 배선 판정 — 같은 값을 되풀이해 밀지 않는가.

파라미터 set 은 us_diff_ik_node 가 조절기를 **다시 만드는** 일이라 싸지 않다. 20 Hz 로
같은 값을 밀면 그만큼 조절기가 다시 만들어진다. 탐색 자체는 ForceSetpointAdapter 의
시험이 맡고, 여기서는 그 사이의 판정만 본다. rclpy 없이 돈다.
"""

import math

from fr5_control.force_setpoint_adapter import SetpointPusher


def test_first_value_is_pushed():
    p = SetpointPusher(eps_n=0.01)
    assert p.should_push(3.0)
    p.mark(3.0)
    assert not p.should_push(3.0)


def test_only_moves_beyond_epsilon_are_pushed():
    p = SetpointPusher(eps_n=0.01)
    p.mark(3.0)
    assert not p.should_push(3.005)        # 잡음 수준 — 조절기를 다시 만들 이유가 없다
    assert p.should_push(3.02)             # 디더 반주기 전환은 이보다 훨씬 크다
    p.mark(3.02)
    assert not p.should_push(3.02)


def test_dither_switch_always_pushes():
    """디더 반폭이 기본 0.25 N 이므로 부호가 바뀌면 0.5 N 이 움직인다."""
    p = SetpointPusher(eps_n=0.01)
    p.mark(3.25)
    assert p.should_push(2.75)


def test_non_finite_setpoint_is_never_pushed():
    p = SetpointPusher(eps_n=0.01)
    assert not p.should_push(math.nan)
    assert not p.should_push(math.inf)
    p.mark(3.0)
    assert not p.should_push(math.nan)     # 이전 값을 지우지도 않는다
    assert p.should_push(3.5)
