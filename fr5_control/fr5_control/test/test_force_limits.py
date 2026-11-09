"""콘솔이 고치는 접촉력 창의 규칙."""

import math

import pytest

from fr5_control.force_limits import (
    DEFAULT_CEILING_N,
    MIN_FLOOR_N,
    WARN_GAP_N,
    floor_of,
    force_window,
)

#: probe.yaml contact_control.deadband_n
BAND = 0.05


def test_floor_puts_regulator_push_line_at_floor():
    # 조절기는 target - deadband 아래에서 밀어 넣는다. 그 선이 바닥이어야 한다.
    w = force_window(3.0, 5.0, BAND)
    assert w.hold_n - BAND == pytest.approx(3.0)
    assert (w.floor_n, w.warn_n, w.max_n, w.resume_n) == (3.0, 4.5, 5.0, 3.0)
    assert floor_of(w.hold_n, BAND) == pytest.approx(3.0)


def test_parameters_satisfy_both_node_invariants():
    for floor, mx in [(0.5, 1.6), (1.0, 4.5), (3.0, 5.0), (3.9, 5.0), (2.0, 3.1), (6.0, 10.0)]:
        w = force_window(floor, mx, BAND)
        assert w.hold_n < w.warn_n <= w.max_n          # ForceRegulator
        assert w.resume_n < w.warn_n <= w.max_n        # ForceLimitOverride
        assert w.warn_n == pytest.approx(mx - WARN_GAP_N)
        p = w.as_parameters()
        assert set(p) == {
            "contact_control.target_force_n",
            "safety.warn_contact_force_n",
            "safety.max_contact_force_n",
            "safety.teleop_resume_force_n",
        }
        assert "teleop.contact_probing_force_n" not in p   # 문턱은 콘솔의 것이 아니다


def test_ceiling_is_ten_and_cannot_be_raised_from_console():
    assert DEFAULT_CEILING_N == 10.0
    assert force_window(3.0, 10.0, BAND).max_n == 10.0
    with pytest.raises(ValueError, match="천장"):
        force_window(3.0, DEFAULT_CEILING_N + 0.1, BAND)
    # 브리지가 천장을 낮추면 그 아래만 허용한다.
    with pytest.raises(ValueError, match="천장"):
        force_window(2.0, 4.0, BAND, ceiling_n=3.5)
    assert force_window(2.0, 3.5, BAND, ceiling_n=3.5).max_n == 3.5


def test_window_too_narrow_is_refused():
    with pytest.raises(ValueError, match="낮아야"):
        force_window(4.5, 5.0, BAND)
    # 바닥 + deadband 가 max - 1.0 을 넘으면 거절 — deadband 가 창에 들어간다.
    with pytest.raises(ValueError, match="낮아야"):
        force_window(4.0, 5.0, BAND)
    assert force_window(4.0 - BAND, 5.0, BAND).hold_n == pytest.approx(4.0)
    # deadband 0 이면 바닥이 곧 목표.
    assert force_window(4.0, 5.0, 0.0).hold_n == 4.0


def test_floor_floor_and_bad_numbers():
    with pytest.raises(ValueError):
        force_window(MIN_FLOOR_N - 0.01, 5.0, BAND)
    with pytest.raises(ValueError):
        force_window(0.0, 5.0, BAND)
    with pytest.raises(ValueError):
        force_window(math.nan, 5.0, BAND)
    with pytest.raises(ValueError):
        force_window(3.0, math.inf, BAND)
    with pytest.raises(ValueError):
        force_window(3.0, 5.0, -0.1)
    with pytest.raises(ValueError):
        force_window("abc", 5.0, BAND)
    with pytest.raises(ValueError):
        force_window(None, 5.0, BAND)
