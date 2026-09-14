"""방광 뷰 품질 — 성공 판정과 같은 축으로 재는가, 그리고 정책 입력을 건드리지 않는가.

옛 Q_seg 는 면적 가중치가 15 % 라 방광이 없어도 0.89, 완벽해도 0.95 였다 (2026-09-11).
"""

import numpy as np
import pytest

from rus_policy.perception import STATE_FEATURE_NAMES
from rus_policy.episode import Thresholds
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
    assert parts["presence"] == 0.0 and parts["area"] == 0.0 and parts["centering"] == 0.0


def test_presence_is_the_heaviest_term():
    """방광을 보는 것이 주 목적이다 — 유무의 가중치가 가장 커야 한다 (2026-09-12)."""
    from rus_policy.view_quality import WEIGHTS
    assert WEIGHTS["presence"] == max(WEIGHTS.values())
    assert WEIGHTS["presence"] / sum(WEIGHTS.values()) > 0.35


def test_barely_visible_bladder_already_scores_far_above_none():
    """겨우 보이는 것과 안 보이는 것의 차이가 가장 커야 한다 — 찾는 것이 목적이다."""
    none = view_quality(_state(has_mask=0.0))[0]
    seen = view_quality(_state(area_ratio=0.005))[0]          # 0.5 %
    success = view_quality(_state(area_ratio=Thresholds().area_min))[0]
    assert seen - none > success - seen, (none, seen, success)


def test_area_raises_score_up_to_the_success_threshold():
    qs = [view_quality(_state(area_ratio=a))[0] for a in (0.0, AREA_FULL * 0.25, AREA_FULL * 0.5, AREA_FULL)]
    assert qs == sorted(qs) and len(set(qs)) == 4, qs
    # 성공 문턱을 넘으면 더 크다고 더 좋아지지 않는다 — 판정도 그렇다.
    assert view_quality(_state(area_ratio=0.15))[0] == pytest.approx(
        view_quality(_state(area_ratio=AREA_FULL))[0])


def test_lateral_centering_matters():
    center = view_quality(_state(area_ratio=Thresholds().area_min, centroid_dx=0.0))[0]
    edge = view_quality(_state(area_ratio=Thresholds().area_min, centroid_dx=0.30))[0]   # 성공 경계
    off = view_quality(_state(area_ratio=Thresholds().area_min, centroid_dx=0.50))[0]    # 화면 끝
    assert center > edge > off


def test_score_spans_most_of_the_range():
    """없음 → 성공 사이가 넓어야 좇을 경사가 생긴다. 옛 Q 는 0.06 이었다."""
    none = view_quality(_state(has_mask=0.0))[0]
    success = view_quality(_state(area_ratio=Thresholds().area_min, centroid_dx=0.0))[0]
    assert success - none > 0.7


def test_presence_area_and_centering_dominate_the_weights():
    from rus_policy.view_quality import WEIGHTS
    share = sum(WEIGHTS[k] for k in ("presence", "area", "centering")) / sum(WEIGHTS.values())
    assert share > 0.6, f"유무·면적·중심 비중이 {share:.0%} 뿐이다"


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


#: 이 팬텀에서 사람이 그린 방광 마스크의 최대 면적비 (라벨 78 장, 2026-09-14).
HUMAN_MAX_AREA = 0.0685


def test_quality_keeps_climbing_to_the_achievable_maximum():
    """포화점은 성공 문턱이 아니라 **도달 가능한 최대**를 넘겨 두어야 한다.

    Q_seg 는 판정에 쓰이지 않는다. 화면 표시이자 `search` 조건의 목적함수이고, 그
    조건은 이 값을 힐클라이밍한다. 성공 문턱(0.04)에서 포화시키면 0.04~0.0685 구간의
    기울기가 0 이 되어, 시야를 더 키울 수 있는데도 탐색기가 멈춘다 (2026-09-14 실측:
    기울기 2.59 → 0.00). 그래서 둘을 일부러 떼어 놓는다.
    """
    assert AREA_FULL > Thresholds().area_min, "포화점이 성공 문턱을 따라가면 탐색기가 눈을 잃는다"
    assert AREA_FULL >= HUMAN_MAX_AREA, "도달 가능한 최대 전에 포화하면 그 위 구간이 평평해진다"

    lo = view_quality(_state(area_ratio=Thresholds().area_min))[0]
    hi = view_quality(_state(area_ratio=HUMAN_MAX_AREA))[0]
    assert hi > lo + 0.05, "성공 문턱과 최대 사이에 오를 여지가 남아 있어야 한다"


def test_attempt_row_reports_the_view_score_not_the_legacy_one():
    """시도 요약의 q_seg 는 view_quality 여야 한다.

    2026-09-14 attempt_007: 마스크 0 %, 면적 0.0000 인데 q_mean 은 0.721 이었다. 옛 Q 는
    마스크가 없어도 0.7~0.9 를 낸다 — 방광을 찾는 일의 성적으로 쓰면 실패가 성공처럼 보인다.
    """
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    row = re.search(r'row = \{"n": self\.attempt.*?\}\n', src, re.S).group(0)
    assert '"q_seg": v.get("q_view_mean"' in row, row


def test_view_quality_is_zero_without_a_mask():
    """마스크가 없으면 점수가 0 에 가까워야 한다 — 이것이 옛 Q 와 갈리는 지점이다."""
    import numpy as np
    from rus_policy.perception import STATE_DIM
    from rus_policy.view_quality import view_quality
    s = np.zeros(STATE_DIM, np.float32)          # has_mask = 0
    score, parts = view_quality(s)
    assert score < 0.2, (score, parts)
    for k in ("presence", "area", "centering", "confidence", "component"):
        assert parts[k] == 0.0, (k, parts[k])
