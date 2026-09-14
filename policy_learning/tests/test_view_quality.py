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


def test_barely_visible_bladder_does_not_already_score_like_a_good_view():
    """🔁 2026-09-14 저녁: 이 테스트는 **뒤집혔다.**

    전에는 "겨우 보이는 것과 안 보이는 것의 차이가 가장 커야 한다" 를 고정했다. 그 성질
    자체는 맞지만, 당시 구현은 그것을 presence 가 면적 0.5 % 에서 포화하는 것으로
    이뤘다 — 그래서 화면의 0.5 % 만 보여도 가중치의 41 % 를 가져갔고, 조작자가
    "방광 전체가 가운데 오지도 않았는데 Q_seg 가 너무 높다" 고 지적한 자리가 됐다.

    유무는 여전히 가장 무겁다 (게이트로 들어가므로 무너지면 점수를 통째로 가져간다).
    다만 **0.5 % 는 "보인다" 가 아니다.** 겨우 보이는 뷰는 성공 문턱 뷰보다 뚜렷하게
    낮아야 한다.
    """
    none = view_quality(_state(has_mask=0.0))[0]
    seen = view_quality(_state(area_ratio=0.005))[0]          # 0.5 %
    success = view_quality(_state(area_ratio=Thresholds().area_min))[0]
    assert none == 0.0
    assert seen < 0.25, seen                       # 겨우 보이는 것은 낮은 점수다
    assert success - seen > seen - none, (none, seen, success)


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
    """없음 → 성공 사이가 넓어야 좇을 경사가 생긴다. 옛 Q 는 0.06 이었다.

    🔁 2026-09-14 저녁: 기준을 0.7 에서 0.6 으로 내렸다. **점수가 나빠져서가 아니라
    면적 눈금이 넓어져서다.** AREA_FULL 이 0.08 → 0.12 (채운 팬텀의 p90) 로 가면서
    성공 문턱(0.04)은 면적 항의 3 분의 1 자리가 됐다 — 최소 합격 뷰는 이제 "겨우
    합격" 으로 읽히는 것이 맞다. 실제로 좇을 경사는 오히려 넓어졌다: 아래에서 최대
    면적 뷰와의 차이를 함께 고정한다.
    """
    none = view_quality(_state(has_mask=0.0))[0]
    success = view_quality(_state(area_ratio=Thresholds().area_min, centroid_dx=0.0))[0]
    full = view_quality(_state(area_ratio=AREA_FULL, centroid_dx=0.0))[0]
    assert success - none > 0.6
    # 합격 문턱 위에도 좇을 경사가 남아 있어야 한다 — 여기가 0 이면 탐색이 멈춘다.
    assert full - success > 0.2, (success, full)


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


def test_the_pass_line_is_reachable():
    """0.97 이 구조적으로 가능한 값인가.

    합격선을 도달 불가능한 값으로 두면 성공률이 제어기와 무관하게 0 이 된다 —
    2026-09-14 에 면적 문턱 0.08 로 그 일이 있었다 (사람 라벨 최대가 0.0685 였다).
    면적 항이 포화하면(새 팬텀은 0.138 까지 보인다) 상한은 0.982 다.
    """
    from rus_policy.episode import Thresholds
    from rus_policy.view_quality import WEIGHTS
    tot = sum(WEIGHTS.values())
    # 대비는 실측이 0.67 이고 중심은 0.95 근처다 — 그 둘만 깎인 상태
    parts = {k: 1.0 for k in WEIGHTS}
    parts.update(contrast=0.667, centering=0.95)
    ceiling = sum(WEIGHTS[k] * v for k, v in parts.items()) / tot
    assert ceiling >= Thresholds().q_view_min, (
        f"합격선 {Thresholds().q_view_min} 이 상한 {ceiling:.3f} 보다 높다 — 도달 불가능하다")


def test_the_pass_line_is_the_same_everywhere():
    """제어기의 목표와 판정의 합격선이 어긋나면, 제어기가 멈춘 자리가 실패로 찍힌다."""
    import re
    from pathlib import Path
    from rus_policy.episode import Thresholds
    from rus_policy.rule_control import RuleController
    from rus_policy.axis_gains import AxisGains
    g = AxisGains(centroid_per_unit={"thy": 0.019}, sign_agreement={"thy": 0.9})
    assert RuleController(g).q_target == Thresholds().q_view_min
    root = Path(__file__).resolve().parents[1] / "scripts"
    for name, flag in (("run_policy.py", "--rule-q-target"),
                       ("run_policy.py", "--success-q-view-min"),
                       ("run_experiment.py", "--success-q-view-min")):
        src = (root / name).read_text(encoding="utf-8")
        m = re.search(rf'add_argument\("{flag}", type=float, default=([\d.]+)', src)
        assert m and float(m.group(1)) == Thresholds().q_view_min, f"{name} {flag}: {m and m.group(1)}"


# --- 2026-09-14 저녁: 조작자가 지적한 세 가지 ----------------------------------
#
# "방광의 전체가 화면에 가운데에 오지도 않았는데 Q_seg 의 점수가 너무 높다."
# 원인 셋을 기록 29,311 프레임에서 확인하고 고쳤다. 아래 셋이 그 셋을 고정한다.


def test_the_always_on_terms_add_nothing_when_satisfied():
    """1. **고정 오프셋이 없어야 한다.**

    presence·confidence·component 의 실측 평균은 1 에 붙어 있었다 (0.980 · 0.998 ·
    0.997). 가중 평균이던 시절 이 셋은 8/14.5 = 55 %p 를 늘 얹었고, 그래서 검출만 되면
    Q_seg 가 0.377 밑으로 내려가지 않았다. 늘 켜져 있는 항은 점수가 아니라 바닥이다.

    전제 조건은 곱으로 들어가야 한다 — 만족하면 아무것도 주지 않는다.
    """
    from rus_policy.view_quality import AREA_SEEN
    # 전제는 완벽하고 뷰만 나쁜 프레임: 유무·신뢰도·연결성분 만점, 그런데 잘렸고 치우쳤다.
    q, parts = view_quality(_state(area_ratio=AREA_SEEN, centroid_dx=0.45,
                                   border_contact_ratio=0.30))
    # float32 상태 벡터라 정확히 1.0 은 아니다 — 게이트가 "만족" 이라는 뜻이면 된다.
    assert min(parts["presence"], parts["confidence"], parts["component"]) == pytest.approx(1.0)
    assert q < 0.15, (q, parts)


def test_a_broken_precondition_takes_the_whole_score():
    """게이트는 만족하면 안 주지만 무너지면 전부 가져간다 — 그것이 곱의 값어치다."""
    good = view_quality(_state(area_ratio=AREA_FULL))[0]
    torn = view_quality(_state(area_ratio=AREA_FULL, largest_component_ratio=0.3))[0]
    assert torn < good * 0.4, (good, torn)


def test_presence_needs_a_real_bladder_not_a_speck():
    """2. **presence 포화점.** 0.5 % 는 "한두 픽셀이 아니다" 이지 "보고 있다" 가 아니다.

    이 팬텀의 사람 라벨 최대 면적이 6.85 % 다. 그 1/14 이 만점을 받으면 안 된다.
    """
    from rus_policy.view_quality import AREA_SEEN
    assert AREA_SEEN >= Thresholds().area_min / 2
    assert view_quality(_state(area_ratio=0.005))[1]["presence"] < 0.3
    assert view_quality(_state(area_ratio=AREA_SEEN))[1]["presence"] == pytest.approx(1.0)


def test_a_truncated_bladder_scores_far_below_a_whole_one():
    """3. **"전체가 화면 안에 있는가" 를 실제로 잰다.**

    ``border_contact_ratio`` 는 윤곽 중 시야 가장자리에 걸린 비율이다. 실측 분포가
    이분적이라 (75 % 가 정확히 0 · p90 0.226) 잘린 프레임은 0.22 근처에 모인다.
    옛 ``exp(-b/0.15)`` 는 거기서도 0.22 를 남겼고 가중치가 6.9 % 뿐이었다.
    """
    whole = view_quality(_state(area_ratio=0.05))[0]
    cut = view_quality(_state(area_ratio=0.05, border_contact_ratio=0.226))[0]
    assert cut < whole * 0.5, (whole, cut)
    assert view_quality(_state(area_ratio=0.05, border_contact_ratio=0.15))[1]["border"] == 0.0


def test_a_truncated_bladder_gets_no_centering_credit():
    """잘렸으면 보이는 조각의 중심은 해부학적 중심이 아니다.

    기록에서 |centroid_dx| 는 p95 가 0.111 로 거의 늘 가운데다 — 그 안에 "잘려서 남은
    조각이 가운데" 인 프레임이 섞여 있고, 조작자가 본 것이 그 프레임이다.
    ``border_contact_ratio`` 독스트링이 같은 말을 한다: 시야 밖으로 나간 방광의 중심은
    편향된 추정치다. 믿을 근거가 없으면 그 몫을 주지 않는다.
    """
    centred = view_quality(_state(area_ratio=0.05, centroid_dx=0.0))[1]
    cut = view_quality(_state(area_ratio=0.05, centroid_dx=0.0,
                              border_contact_ratio=0.226))[1]
    assert centred["centering"] == pytest.approx(1.0)
    assert cut["centering"] == 0.0, cut


def test_the_scale_actually_spans_its_range():
    """검출되었다는 이유만으로 얻는 바닥이 없어야 한다.

    옛 눈금에서 검출 프레임의 최솟값은 0.377, p5 는 0.706 이었다. 성공 판정에 탈락하는
    프레임(전체의 57 %)의 중앙값이 0.801 이라, 탈락한 뷰가 사람 눈에 "좋다" 로 읽혔다.
    """
    worst = view_quality(_state(area_ratio=0.001, centroid_dx=0.45,
                                border_contact_ratio=0.4))[0]
    best = view_quality(_state(area_ratio=AREA_FULL, centroid_dx=0.0,
                               lumen_contrast=0.5, contrast_valid=1.0))[0]
    assert worst < 0.05 and best > 0.95, (worst, best)
