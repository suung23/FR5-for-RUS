"""에피소드 판정 — 시작 조건, 성공 판정, 위약 버퍼."""

import numpy as np
import pytest

from rus_policy.episode import (AREA, COMPONENT, HAS_MASK, QUALITY, PlaceboBuffer, StartGate,
                                Thresholds, judge, longest_run_s)
from rus_policy.perception import STATE_DIM


def _state(area=0.0, comp=0.0, q=0.0, mask=1.0):
    s = np.zeros(STATE_DIM, np.float32)
    s[AREA], s[COMPONENT], s[QUALITY], s[HAS_MASK] = area, comp, q, mask
    return s


# ---- 시작 조건 ---------------------------------------------------------------

def test_gate_needs_the_window_not_one_frame():
    g = StartGate(confirm_s=1.0)
    s = _state(area=0.01, q=0.7)
    for t in np.arange(0.0, 0.9, 0.1):
        assert not g.update(float(t), s)
    for t in np.arange(0.9, 1.3, 0.1):
        g.update(float(t), s)
    assert g.is_open


def test_gate_resets_on_a_single_bad_sample():
    """분할이 한 장 튄 것과 '그 자세가 실제로 그렇다' 를 가르는 것이 창의 목적이다."""
    g = StartGate(confirm_s=1.0)
    good, bad = _state(area=0.01, q=0.7), _state(area=0.5, q=0.7)
    for t in np.arange(0.0, 0.9, 0.1):
        g.update(float(t), good)
    g.update(0.9, bad)
    assert g.held_s == 0.0 and not g.is_open


def test_gate_rejects_a_visible_bladder():
    """이미 잘 보이면 아무것도 안 해도 성공이라 찾는 능력을 못 잰다."""
    g = StartGate(area_max=0.02, quality_min=0.6, confirm_s=0.2)
    for t in np.arange(0.0, 1.0, 0.1):
        g.update(float(t), _state(area=0.10, q=0.9))
    assert not g.is_open


def test_gate_rejects_a_useless_image():
    g = StartGate(area_max=0.02, quality_min=0.6, confirm_s=0.2)
    for t in np.arange(0.0, 1.0, 0.1):
        g.update(float(t), _state(area=0.01, q=0.3))     # 안 보이지만 영상도 못 쓴다
    assert not g.is_open


# ---- 성공 판정 ---------------------------------------------------------------

def test_longest_run_measures_contiguous_time():
    t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert longest_run_s(t, [False, True, True, True, False, True]) == pytest.approx(2.0)
    assert longest_run_s(t, [False] * 6) == 0.0
    assert longest_run_s([0.0], [True]) == 0.0


def test_success_needs_a_sustained_run_not_one_good_frame():
    """한 프레임짜리 성공을 세면 무작위로 흔들기만 해도 위약이 이긴다."""
    thr = Thresholds(area_min=0.08, component_min=0.8, hold_s=3.0)
    t = np.arange(0.0, 10.0, 0.5)
    st = np.stack([_state(area=0.01, comp=0.9, q=0.7) for _ in t])
    st[7] = _state(area=0.2, comp=0.9, q=0.7)            # 딱 한 장만 좋다
    r = judge(t, st, thr)
    assert not r["success"] and r["best_run_s"] == 0.0

    st[6:14] = _state(area=0.2, comp=0.9, q=0.7)         # 4 s 연속
    r = judge(t, st, thr)
    assert r["success"] and r["best_run_s"] >= 3.0


def test_component_threshold_rejects_fragmented_masks():
    thr = Thresholds(area_min=0.08, component_min=0.8, hold_s=1.0)
    t = np.arange(0.0, 5.0, 0.5)
    st = np.stack([_state(area=0.2, comp=0.5, q=0.7) for _ in t])   # 넓지만 조각남
    assert not judge(t, st, thr)["success"]


def test_judge_reports_quality_summary():
    t = np.arange(0.0, 20.0, 1.0)
    st = np.stack([_state(area=0.01, comp=0.9, q=float(0.5 + 0.02 * i)) for i in range(len(t))])
    r = judge(t, st, Thresholds())
    assert r["n_samples"] == len(t)
    assert r["q_final_10s"] > r["q_mean"]                # 뒤가 좋아지면 꼬리 평균이 높다


def test_judge_checks_shape():
    with pytest.raises(ValueError):
        judge([0.0, 1.0], np.zeros((3, STATE_DIM)), Thresholds())


# ---- 위약 -------------------------------------------------------------------

def test_placebo_returns_nothing_until_the_delay_has_passed():
    b = PlaceboBuffer(delay_s=30.0)
    for t in np.arange(0.0, 29.0, 1.0):
        b.push(float(t), f"obs{t:.0f}")
    assert b.take(29.0) is None


def test_placebo_returns_the_delayed_observation():
    """시간은 단조로 들어온다 — 실시간 루프가 부르는 방식이다."""
    b = PlaceboBuffer(delay_s=30.0)
    for t in np.arange(0.0, 46.0, 1.0):
        b.push(float(t), f"obs{t:.0f}")
    assert b.take(45.0) == "obs15"
    for t in np.arange(46.0, 61.0, 1.0):
        b.push(float(t), f"obs{t:.0f}")
    assert b.take(60.0) == "obs30"


def test_placebo_drops_what_it_will_never_use():
    """90 s × 8 fps 를 다 들고 있을 이유가 없다."""
    b = PlaceboBuffer(delay_s=5.0)
    for t in np.arange(0.0, 100.0, 0.125):
        b.push(float(t), t)
        b.take(float(t))
    assert b.n_buffered < 100


def test_placebo_rejects_a_non_positive_delay():
    with pytest.raises(ValueError):
        PlaceboBuffer(delay_s=0.0)
