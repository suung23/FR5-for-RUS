"""에피소드 판정 — 시작 조건, 성공 판정, 위약 버퍼."""

import numpy as np
import pytest

from rus_policy.episode import (AREA, CENTROID_DX, COMPONENT, HAS_MASK, QUALITY, PlaceboBuffer,
                                StartGate, Thresholds, judge, longest_run_s, randomize_direction)
from rus_policy.perception import STATE_DIM


_Q_RAW = 0.72          # 시작 조건의 Q_raw — 상태 벡터의 quality(=Q_seg) 와 다르다


def _state(area=0.0, comp=0.0, q=0.0, mask=1.0, dx=0.0):
    s = np.zeros(STATE_DIM, np.float32)
    s[AREA], s[COMPONENT], s[QUALITY], s[HAS_MASK], s[CENTROID_DX] = area, comp, q, mask, dx
    return s


# ---- 시작 조건 ---------------------------------------------------------------

def test_gate_needs_the_window_not_one_frame():
    g = StartGate(confirm_s=1.0)
    s = _state(area=0.01, q=0.7)
    for t in np.arange(0.0, 0.9, 0.1):
        assert not g.update(float(t), s, _Q_RAW)
    for t in np.arange(0.9, 1.3, 0.1):
        g.update(float(t), s, _Q_RAW)
    assert g.is_open


def test_gate_resets_on_a_single_bad_sample():
    """분할이 한 장 튄 것과 '그 자세가 실제로 그렇다' 를 가르는 것이 창의 목적이다."""
    g = StartGate(confirm_s=1.0)
    good, bad = _state(area=0.01, q=0.7), _state(area=0.5, q=0.7)
    for t in np.arange(0.0, 0.9, 0.1):
        g.update(float(t), good, _Q_RAW)
    g.update(0.9, bad, _Q_RAW)
    assert g.held_s == 0.0 and not g.is_open


def test_gate_rejects_a_visible_bladder():
    """이미 잘 보이면 아무것도 안 해도 성공이라 찾는 능력을 못 잰다."""
    g = StartGate(area_max=0.02, quality_min=0.6, confirm_s=0.2)
    for t in np.arange(0.0, 1.0, 0.1):
        g.update(float(t), _state(area=0.10), _Q_RAW)      # 이미 잘 보인다
    assert not g.is_open


def test_gate_rejects_a_useless_image():
    g = StartGate(area_max=0.02, quality_min=0.6, confirm_s=0.2)
    for t in np.arange(0.0, 1.0, 0.1):
        g.update(float(t), _state(area=0.01), 0.3)        # 안 보이는데 접촉도 나쁘다
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


# ---- 사전 등록 판정 (EVAL_PLAN_POLICY_RESCUE §4) --------------------------------

def test_edge_views_are_rejected_by_the_centroid_criterion():
    """면적·연결성분이 충분해도 가장자리에 걸친 뷰는 진단 가능 뷰가 아니다."""
    thr = Thresholds(area_min=0.08, component_min=0.8, centroid_max=0.30, hold_s=1.0)
    t = np.arange(0.0, 6.0, 0.5)
    centred = np.stack([_state(area=0.2, comp=0.9, dx=0.05) for _ in t])
    edged = np.stack([_state(area=0.2, comp=0.9, dx=0.45) for _ in t])
    assert judge(t, centred, thr)["success"]
    assert not judge(t, edged, thr)["success"]


def test_centroid_criterion_is_symmetric():
    thr = Thresholds(area_min=0.08, component_min=0.8, centroid_max=0.30, hold_s=1.0)
    t = np.arange(0.0, 6.0, 0.5)
    for dx in (-0.29, 0.29):
        assert judge(t, np.stack([_state(area=0.2, comp=0.9, dx=dx) for _ in t]), thr)["success"]
    for dx in (-0.31, 0.31):
        assert not judge(t, np.stack([_state(area=0.2, comp=0.9, dx=dx) for _ in t]), thr)["success"]


# ---- 위약: 같은 크기, 무작위 방향 (§5) -----------------------------------------

def test_placebo_preserves_the_magnitudes_exactly():
    """'같은 속도·같은 지속시간' 이 성립해야 '움직임의 양' 이 조건 사이에서 같다."""
    rng = np.random.default_rng(0)
    a = np.array([1.0, 2.0, 3.0, 0.4, -0.7, 1.1])
    for _ in range(50):
        b = randomize_direction(a, rng)
        assert sorted(np.abs(b[3:6])) == pytest.approx(sorted(np.abs(a[3:6])))
        assert np.linalg.norm(b[3:6]) == pytest.approx(np.linalg.norm(a[3:6]))


def test_placebo_leaves_translation_alone():
    """정책이 병진을 지령하지 않으므로 위약도 건드릴 것이 없다."""
    rng = np.random.default_rng(0)
    a = np.array([1.0, 2.0, 3.0, 0.4, -0.7, 1.1])
    assert np.array_equal(randomize_direction(a, rng)[:3], a[:3])


def test_placebo_actually_changes_direction():
    rng = np.random.default_rng(0)
    a = np.array([0.0, 0.0, 0.0, 0.4, -0.7, 1.1])
    seen = {tuple(np.sign(randomize_direction(a, rng)[3:6])) for _ in range(200)}
    assert len(seen) > 4                      # 부호 조합이 실제로 돌아다닌다


def test_placebo_does_not_mutate_its_input():
    rng = np.random.default_rng(0)
    a = np.array([1.0, 2.0, 3.0, 0.4, -0.7, 1.1])
    randomize_direction(a, rng)
    assert np.array_equal(a, [1.0, 2.0, 3.0, 0.4, -0.7, 1.1])


def test_gate_opens_with_no_mask_at_all():
    """'방광이 안 보인다' 의 가장 극단이 면적비 0 이다 — 마스크 없음을 요구로 걸면 안 된다.

    2026-09-11 실기: 면적비 0.000 · Q_raw 0.72 인 자세가 계속 폐기됐다.
    """
    g = StartGate(area_max=0.02, quality_min=0.6, confirm_s=1.0)
    s = _state(area=0.0, q=0.72, mask=0.0)          # 마스크가 아예 없다
    for t in np.arange(0.0, 1.5, 0.1):
        g.update(float(t), s, _Q_RAW)
    assert g.is_open


def test_success_still_requires_a_mask():
    """시작 조건과 반대다 — 진단 가능 뷰는 마스크가 있어야 한다."""
    thr = Thresholds(area_min=0.08, component_min=0.8, hold_s=1.0)
    t = np.arange(0.0, 6.0, 0.5)
    no_mask = np.stack([_state(area=0.2, comp=0.9, dx=0.0, mask=0.0) for _ in t])
    assert not judge(t, no_mask, thr)["success"]


def test_gate_uses_q_raw_not_the_segmentation_score():
    """계획서 §3 의 통제는 "접촉은 좋은데 방광이 없다" 이다 — 분할 점수로 재면 뜻이 뒤집힌다."""
    g = StartGate(area_max=0.02, quality_min=0.6, confirm_s=0.5)
    s = _state(area=0.0, q=0.95, mask=0.0)        # 상태의 quality(Q_seg)는 높지만
    for t in np.arange(0.0, 1.0, 0.1):
        g.update(float(t), s, 0.30)                # Q_raw 는 낮다 → 접촉 불량
    assert not g.is_open

    g.reset()
    for t in np.arange(0.0, 1.0, 0.1):
        g.update(float(t), _state(area=0.0, q=0.0, mask=0.0), 0.72)   # 반대
    assert g.is_open


def test_gate_treats_unmeasured_q_raw_as_not_met():
    g = StartGate(area_max=0.02, quality_min=0.6, confirm_s=0.5)
    for t in np.arange(0.0, 1.0, 0.1):
        g.update(float(t), _state(area=0.0), float("nan"))
    assert not g.is_open


def test_two_phase_separates_finding_from_holding():
    """찾기와 유지는 다른 능력이다 — 하나의 성공/실패로는 못 가린다.

    2026-09-12 조작자 정의: 방광이 없는 자리에서 시작해 찾고, 그 뒤 주사기로 용적을 바꿔도
    뷰를 유지하는가 · 그때 힘이 위험하지 않은가. 찾자마자 놓쳐도 기존 판정은 "성공" 이다.
    """
    import numpy as np

    from rus_policy.episode import Thresholds, judge, judge_two_phase
    from rus_policy.perception import STATE_FEATURE_NAMES

    I = {n: i for i, n in enumerate(STATE_FEATURE_NAMES)}
    thr = Thresholds()

    def frames(area_seq, hz=10.0):
        st = np.zeros((len(area_seq), len(STATE_FEATURE_NAMES)), np.float32)
        for i, a in enumerate(area_seq):
            st[i, I["has_mask"]] = 1.0 if a > 0 else 0.0
            st[i, I["area_ratio"]] = a
            st[i, I["largest_component_ratio"]] = 1.0
            st[i, I["segmentation_confidence"]] = 1.0
        return np.arange(len(area_seq)) / hz, st

    # 20 s 못 찾다가 4 s 찾고, 그 뒤 6 s 는 절반만 유지
    seq = [0.0] * 200 + [0.12] * 40 + ([0.12] * 5 + [0.0] * 5) * 6
    t, st = frames(seq)
    old = judge(t, st, thr)
    new = judge_two_phase(t, st, thr)
    assert old["success"] is True, "기존 판정은 찾기만 본다"
    assert new["found"] is True
    assert 19.0 < new["find_time_s"] < 24.0, new["find_time_s"]
    assert 0.3 < new["hold_good_fraction"] < 0.8, new["hold_good_fraction"]
    assert new["hold_worst_loss_s"] >= 0.35   # 5 프레임 = 0.4 s (마지막 자릿수는 부동소수점)

    # 못 찾으면 유지 항목은 재지 않는다 (NaN)
    t2, st2 = frames([0.0] * 100)
    none = judge_two_phase(t2, st2, thr)
    assert none["found"] is False and np.isnan(none["hold_good_fraction"])


def test_two_phase_reports_force_safety():
    """'그때의 힘은 위험하지 않은가' 도 같은 판정에서 나와야 한다."""
    import numpy as np

    from rus_policy.episode import Thresholds, judge_two_phase
    from rus_policy.perception import STATE_FEATURE_NAMES

    I = {n: i for i, n in enumerate(STATE_FEATURE_NAMES)}
    thr = Thresholds()
    st = np.zeros((100, len(STATE_FEATURE_NAMES)), np.float32)
    st[:, I["has_mask"]] = 1.0
    st[:, I["area_ratio"]] = 0.12
    st[:, I["largest_component_ratio"]] = 1.0
    t = np.arange(100) / 10.0
    ft = np.arange(100) / 10.0
    fn = np.full(100, 3.0)
    fn[50:60] = 4.8            # 1 s 동안 경고선 위
    fn[70:75] = 5.2            # 0.5 s 동안 한계선 위
    v = judge_two_phase(t, st, thr, force_t=ft, force_n=fn)
    assert abs(v["force_max_n"] - 5.2) < 1e-6
    assert 0.8 < v["time_above_warn_s"] < 1.7, v["time_above_warn_s"]   # 경고선 위 = 4.8·5.2 모두
    assert 0.4 < v["time_above_limit_s"] < 0.7, v["time_above_limit_s"]
