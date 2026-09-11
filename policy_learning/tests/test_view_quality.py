"""방광 뷰 품질 — 성공 판정과 같은 축으로 재는가, 그리고 정책 입력을 건드리지 않는가.

옛 Q_seg 는 면적 가중치가 15 % 라 방광이 없어도 0.89, 완벽해도 0.95 였다 (2026-09-11).
"""

import numpy as np
import pytest

from rus_policy.perception import STATE_FEATURE_NAMES
from rus_policy.view_quality import AREA_FULL, view_quality

I = {n: i for i, n in enumerate(STATE_FEATURE_NAMES)}


def _state(**kw):
    s = np.zeros(len(STATE_FEATURE_NAMES), np.float32)
    s[I["has_mask"]] = 1.0
    s[I["segmentation_confidence"]] = 1.0
    s[I["largest_component_ratio"]] = 1.0
    for k, v in kw.items():
        s[I[k]] = v
    return s


def test_no_bladder_scores_low():
    """못 찾았으면 점수에 드러나야 한다. 옛 Q 는 이때도 0.89 였다."""
    q, parts = view_quality(_state(has_mask=0.0, area_ratio=0.0))
    assert q < 0.2
    assert parts["presence_area"] == 0.0 and parts["centering"] == 0.0


def test_area_raises_score_up_to_the_success_threshold():
    qs = [view_quality(_state(area_ratio=a))[0] for a in (0.0, 0.02, 0.04, AREA_FULL)]
    assert qs == sorted(qs) and len(set(qs)) == 4, qs
    # 성공 문턱을 넘으면 더 크다고 더 좋아지지 않는다 — 판정도 그렇다.
    assert view_quality(_state(area_ratio=0.15))[0] == pytest.approx(
        view_quality(_state(area_ratio=AREA_FULL))[0])


def test_lateral_centering_matters():
    center = view_quality(_state(area_ratio=0.08, centroid_dx=0.0))[0]
    edge = view_quality(_state(area_ratio=0.08, centroid_dx=0.30))[0]   # 성공 경계
    off = view_quality(_state(area_ratio=0.08, centroid_dx=0.50))[0]    # 화면 끝
    assert center > edge > off


def test_score_spans_most_of_the_range():
    """없음 → 성공 사이가 넓어야 좇을 경사가 생긴다. 옛 Q 는 0.06 이었다."""
    none = view_quality(_state(has_mask=0.0))[0]
    success = view_quality(_state(area_ratio=0.08, centroid_dx=0.0))[0]
    assert success - none > 0.7


def test_area_and_centering_dominate_the_weights():
    from rus_policy.view_quality import WEIGHTS
    share = (WEIGHTS["presence_area"] + WEIGHTS["centering"]) / sum(WEIGHTS.values())
    assert share > 0.5, f"면적·중심 비중이 {share:.0%} 뿐이다"


def test_does_not_modify_the_policy_input():
    """정책은 상태 벡터의 quality 특징(옛 Q)으로 학습됐다. 그것을 바꾸면 처음 보는 입력이다."""
    s = _state(area_ratio=0.02, quality=0.877)
    before = s.copy()
    view_quality(s)
    np.testing.assert_array_equal(s, before)


def test_runner_keeps_feeding_the_trained_quality():
    """러너는 새 Q 를 발행하되, 정책 버퍼에는 state 를 손대지 않고 넣어야 한다."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text()
    assert "self.q_seg_pub.publish(Float32(data=float(q_view)))" in src
    # 버퍼에 들어가는 것은 view_quality 가 아니라 지각이 낸 state 그대로다.
    assert "self.buf.append((now_t, bm, state, q))" in src
