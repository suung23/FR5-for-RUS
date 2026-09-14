"""전파는 **메우기만** 한다 — 보고 있는 것을 덮거나, 없는 것을 만들지 않는다.

2026-09-14: 흐린 경계에서 모델이 통째로 놓치는데 사람은 두 번 그려도 Dice 0.865 로
일치했다. 방광은 0.1 초 만에 사라지지 않으므로 직전 마스크를 옮겨 메운다. 여기서 고정하는
것은 **언제 옮기지 않는가** 다 — 오탐 0/18 이라는 지금 성질이 그 조건들에서 나온다.
"""

import numpy as np
import pytest

from rus_policy.mask_propagation import MaskPropagator, PropagationConfig


def _mask(cx=40, cy=40, r=12, n=64):
    y, x = np.mgrid[0:n, 0:n]
    return ((x - cx) ** 2 + (y - cy) ** 2) < r * r


def _zero_flow(src, dst):
    return np.zeros(src.shape + (2,), np.float32)


def _shift_flow(dx):
    def f(src, dst):
        out = np.zeros(src.shape + (2,), np.float32)
        out[..., 0] = dx
        return out
    return f


def _prop(flow=_zero_flow, **kw):
    return MaskPropagator(PropagationConfig(**kw), flow)


def test_fills_a_missed_frame_from_the_last_confident_one():
    p = _prop()
    frame = np.full((64, 64), 100, np.uint8)
    p.observe(frame, _mask(), prob_max=1.0, t=0.0)
    d = p.propagate(frame, t=0.2, coupled=True)
    assert d.used and d.mask is not None and d.mask.sum() > 0


def test_never_propagates_without_contact():
    """접촉이 없으면 방광이 찍힐 수 없다. 여기서 마스크를 만들면 그것이 오탐이다."""
    p = _prop()
    frame = np.full((64, 64), 100, np.uint8)
    p.observe(frame, _mask(), prob_max=1.0, t=0.0)
    d = p.propagate(frame, t=0.1, coupled=False)
    assert not d.used and "접촉" in d.reason
    # 근거까지 버려야 한다 — 다시 닿았을 때 옛 마스크가 되살아나면 안 된다
    assert not p.propagate(frame, t=0.15, coupled=True).used


def test_expires_with_age():
    p = _prop(max_age_s=0.5)
    frame = np.full((64, 64), 100, np.uint8)
    p.observe(frame, _mask(), prob_max=1.0, t=0.0)
    assert p.propagate(frame, t=0.4, coupled=True).used
    p.observe(frame, _mask(), prob_max=1.0, t=0.0)
    d = p.propagate(frame, t=0.9, coupled=True)
    assert not d.used and "오래" in d.reason


def test_low_confidence_is_not_stored_as_evidence():
    """확신 없는 출력을 근거로 삼으면 그 오류가 뒤 프레임으로 번진다."""
    p = _prop(min_store_prob=0.99)
    frame = np.full((64, 64), 100, np.uint8)
    p.observe(frame, _mask(), prob_max=0.6, t=0.0)
    assert not p.propagate(frame, t=0.1, coupled=True).used


def test_rejects_when_the_scene_moved_too_much():
    p = _prop(flow=_shift_flow(30.0), max_flow_px=12.0)
    frame = np.full((64, 64), 100, np.uint8)
    p.observe(frame, _mask(), prob_max=1.0, t=0.0)
    d = p.propagate(frame, t=0.1, coupled=True)
    assert not d.used and "움직" in d.reason


def test_rejects_when_the_warped_area_changes_too_much():
    """마스크가 화면 밖으로 밀려 반쪽이 되면 그 전파는 근거가 아니다."""
    p = _prop(flow=_shift_flow(-9.0), area_ratio_band=(0.9, 1.1), max_flow_px=20.0)
    frame = np.full((64, 64), 100, np.uint8)
    p.observe(np.full((64, 64), 100, np.uint8), _mask(cx=8, r=12), prob_max=1.0, t=0.0)
    d = p.propagate(frame, t=0.1, coupled=True)
    assert not d.used and "면적" in d.reason


def test_disabled_does_nothing():
    p = _prop(enabled=False)
    frame = np.full((64, 64), 100, np.uint8)
    p.observe(frame, _mask(), prob_max=1.0, t=0.0)
    assert not p.propagate(frame, t=0.1, coupled=True).used
