"""축 흔들기 — 규칙 기반 정렬에 필요한 **기하 이득**을 재는 모드.

왜 따로 재는가: 프리핸드의 centroid_dx↔θy 기울기 43 °/단위 는 "치우침이 이만큼일 때
시연자가 이만큼 돌렸다" 는 **행동** 기울기다. 제어기에 필요한 것은 "1° 돌리면 centroid 가
얼마나 움직이나" 이고 둘은 다른 양이다 (2026-09-12).
"""

import numpy as np
import pytest

from rus_policy.axis_probe import AXIS_INDEX, DEFAULT_ORDER, MOVE, REST, AxisProbe
from rus_policy.episode import CONDITIONS


def test_axis_probe_is_a_known_condition():
    assert "axis_probe" in CONDITIONS


def test_one_axis_at_a_time():
    """두 축이 동시에 돌면 이득이 섞여 분리되지 않는다 — 이 모드의 존재 이유다."""
    p = AxisProbe()
    for t in np.arange(0, p.total_s, 0.1):
        vel, _ = p.update(float(t))
        assert np.count_nonzero(vel) <= 1, (t, vel)


def test_every_axis_gets_both_signs():
    p = AxisProbe()
    seen = {}
    for t in np.arange(0, p.total_s, 0.05):
        vel, lab = p.update(float(t))
        if p.phase == MOVE:
            seen.setdefault(lab[:-1], set()).add(lab[-1])
    assert set(seen) == set(DEFAULT_ORDER)
    for name, signs in seen.items():
        assert signs == {"+", "−"}, (name, signs)


def test_rest_between_moves():
    """쉬는 구간이 있어야 시야 자체의 느린 변동을 기준선으로 뺄 수 있다."""
    p = AxisProbe()
    phases = [p.update(float(t))[0] for t in np.arange(0, p.slot_s * 2, 0.1)]
    moving = [bool(np.any(v)) for v in phases]
    assert True in moving and False in moving


def test_amplitude_follows_the_axis_kind():
    p = AxisProbe(amp_deg=3.0, amp_mm_s=5.0)
    got = {}
    for t in np.arange(0, p.total_s, 0.05):
        vel, lab = p.update(float(t))
        if p.phase == MOVE:
            got[lab[:-1]] = float(np.abs(vel).max())
    for name, v in got.items():
        assert v == pytest.approx(3.0 if name.startswith("th") else 5.0), (name, v)


def test_stops_after_the_schedule():
    """첫 호출이 기준 시각이 된다 — 순서를 다 돌면 0 을 내고 done 을 붙인다."""
    p = AxisProbe()
    p.update(0.0)
    vel, lab = p.update(p.total_s + 1.0)
    assert lab == "done" and not np.any(vel)


def test_beam_translation_is_never_commanded():
    """빔축 병진은 admittance 몫이라 이 모드가 건드리면 안 된다."""
    assert 2 not in AXIS_INDEX.values()
    p = AxisProbe()
    for t in np.arange(0, p.total_s, 0.1):
        assert p.update(float(t))[0][2] == 0.0


def test_unknown_axis_is_refused():
    with pytest.raises(ValueError, match="모르는 축"):
        AxisProbe(order=("thx", "롤"))
