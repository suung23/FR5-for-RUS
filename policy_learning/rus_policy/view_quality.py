"""방광 뷰 품질 — "방광 **전체**가 화면에, 크게, 가운데에 있는가".

    q, parts = view_quality(state)        # state: 지각 상태 벡터 (STATE_FEATURE_NAMES)

왜 따로 있는가
--------------
기존 Q_seg (``rus_perception.control.quality``) 는 가중치 합 6.5 중 면적이 1.0 (15 %) 이고
나머지 85 % 가 "마스크가 깨끗한가" (신뢰도·연결성·경계·시간 안정성) 다. 그래서
2026-09-11 팬텀에서 **방광이 전혀 안 보여도 0.89, 완벽하게 잡혀도 0.95** 였다 — 차이 0.06.

이 점수는 **성공 판정과 같은 축**으로 잰다: 방광이 보이는가 · 얼마나 크게 · 가운데인가 ·
잘리지 않았는가.

구조: 게이트 × 내용
-------------------
2026-09-14 개편. 그 전에는 일곱 항의 **가중 평균**이었는데, 기록 29,311 프레임에서
세 항의 평균이 1 에 붙어 있었다 — presence 0.980 · confidence 0.998 · component 0.997.
가중치로 8/14.5 = **55 %p 가 사실상 고정 오프셋**이었다는 뜻이다. 실제로 검출만 되면
Q_seg 가 0.377 아래로 내려간 적이 없고 p5 가 0.706 이었다. 성공 판정에 탈락하는
프레임(면적 < 4 % 또는 |dx| > 0.30, 전체의 57 %)의 **중앙값이 0.801** 이었다. 순서는
맞았지만(합격/탈락 AUC 0.958) 영점이 틀려서, 탈락한 뷰가 사람 눈에 "좋다" 로 읽혔다.

가중 평균은 늘 켜져 있는 항을 **더하기**로 넣는다. 그런 항은 점수가 아니라 바닥이다.
전제 조건은 **곱하기**로 들어가야 한다 — 만족하면 아무것도 주지 않고, 무너지면 전부
가져간다. 그래서:

    Q = gate × content
    gate    = presence · confidence · component      전제: 찾았고, 마스크가 믿을 만한가
    content = (3·area + 2·centering + 2·border + 0.5·contrast) / 7.5   뷰가 얼마나 좋은가

세 가지를 고쳤다 (2026-09-14 조작자 지적)
-----------------------------------------
1. **고정 오프셋 제거.** presence·confidence·component 가 게이트로 갔다. 실측 평균이
   1 이니 정상일 때 이 셋은 점수에 아무것도 더하지 않는다. 대신 무너지면(신뢰도가
   흔들리거나 마스크가 조각나거나 방광을 놓치면) 점수를 통째로 끌어내린다.
2. **presence 포화점 0.5 % → 2 %.** 이 팬텀의 사람 라벨 최대 면적이 6.85 % 인데
   그 1/14 만 보여도 가중치의 41 % 를 다 가져갔다. 2 % 는 성공 문턱(4 %)의 절반이다.
3. **잘림을 실제로 잰다.** ``border_contact_ratio`` 는 "방광 윤곽 중 시야 가장자리에
   걸린 비율" 로, 이 항목을 재는 데 정확히 맞는 양인데 가중치가 6.9 % 였고
   ``exp(-b/0.15)`` 라 너그러웠다. 가중치를 2.0 으로 올리고 선형 램프로 바꿨다.
   실측 분포가 이분적이라 (75 % 가 정확히 0, 걸리면 0.22~0.28) 0.15 에서 0 이 되는
   램프면 잘린 프레임이 확실히 0 으로 간다.

   그리고 **centering 에 잘림을 곱한다.** ``border_contact_ratio`` 의 독스트링이
   말하는 그대로다 — 방광의 일부가 시야 밖에 있으면 보이는 조각의 중심은 해부학적
   중심의 편향된 추정치다. 기록에서 |centroid_dx| 는 p95 가 0.111 로 거의 늘 가운데인데,
   그중에 "잘려서 남은 조각이 가운데" 인 프레임이 섞여 있다. 잘렸으면 가운데라는 말을
   믿을 근거가 없으므로 그 몫을 주지 않는다.

⚠️ **정책의 입력은 바꾸지 않는다.** 상태 벡터의 ``quality`` 특징(15 번)은 정책이 학습 때 본
값이고, 그것을 바꾸면 정책이 처음 보는 입력을 받는다 — 실험 도중 모델을 조용히 바꾸는 것과
같다. 이 점수는 화면(Q_seg)·기록·판정 보고와 ``search``/``rule`` 제어기의 목적함수에 쓴다.
정책이 **이 점수를 좇게** 하려면 이 점수로 라벨을 다시 만들고 재학습해야 한다.

⚠️ **눈금이 바뀌었으므로 ``Thresholds.q_view_min`` 도 함께 옮겼다.** 두 값은 같은 작동점을
가리킨다 (기록 프레임에서 판정이 일치하도록 맞췄다). 이미 기록된 에피소드는 states.npz 가
남아 있으므로 ``scripts/rejudge_runs.py`` 로 같은 자에 올릴 수 있다.

시간 항(temporal_iou · 중심/면적 안정성)은 넣지 않는다. 상태 벡터에 없고, "지금 이 뷰가
좋은가" 에는 필요 없다.
"""

from __future__ import annotations

import numpy as np

from .perception import STATE_FEATURE_NAMES

_IDX = {n: i for i, n in enumerate(STATE_FEATURE_NAMES)}

#: 이 면적이면 area 항이 1 이 된다.
#:
#: **성공 판정의 면적 문턱과 일부러 다르다.** 이 함수는 표시용이자 `search`·`rule` 조건의
#: 목적함수이고, 문턱과 같은 자리에서 포화시키면 그 위의 기울기가 0 이 된다. 판정과
#: 지표를 맞추는 편의보다 제어의 기울기가 우선이다 — 성공 판정은 Q 가 아니라
#: 면적·연결성분·중심이 따로 한다.
#:
#: 🔁 2026-09-14 저녁: 0.08 → 0.12. **0.08 은 이제 천장이 아니라 바닥이었다.**
#: 조작자가 "절반만 보이는데도 97 이 뜬다" 고 지적해서 최근 에피소드를 실측했더니,
#: 팬텀을 채운 뒤 면적비가 이렇게 올라와 있었다:
#:
#:     main_rule (09-14 14:20~14:53, 7,723 프레임)  p50 0.0822 · p90 0.1137 · 최대 0.1296
#:     그 전 세션들                                   p50 0.02~0.06 · 최대 0.085
#:
#: 즉 가장 무거운 내용 항(3/7.5 = 40 %)이 **절반 이상의 프레임에서 포화**해 있었다.
#: 아침에 presence 포화점을 0.5 % → 2 % 로 올린 것과 똑같은 고장이 위쪽 끝에서 난 것이다.
#: 0.12 는 채운 팬텀의 p90 자리다 — 상위 10 % 에서만 포화하므로 운용 구간 전체에 기울기가
#: 남는다. (사람 라벨 최대 0.0685 는 **채우기 전** 팬텀의 값이라 더는 기준이 아니다.)
#:
#: ⚠️ 이 값은 팬텀 충전 상태에 딸린 값이다. 용적을 크게 바꾸면 다시 재야 한다.
AREA_FULL = 0.12
#: 이 면적이면 presence 게이트가 1 이 된다. 0 에서 여기까지 가파르게 오른다 — 사실상
#: "제대로 보이는가" 이지만 계단이 아니라 이어져 있어야 오르막 탐색이 따라 오를 수 있다.
#:
#: 🔁 2026-09-14: 0.005 → 0.02. 0.5 % 는 "한두 픽셀이 아니다" 를 재는 값이었지 "방광을
#: 보고 있다" 를 재는 값이 아니었다. 사람 라벨의 면적 중앙값이 3.97 % 이므로 2 % 는
#: 그 절반, 성공 문턱(4 %)의 절반이다.
AREA_SEEN = 0.02
#: centroid_dx 는 [-0.5, 0.5] (0 = 가운데, ±0.5 = 화면 끝). 이 거리에서 centering 이 0.
#:
#: 🔁 2026-09-14: 0.5 → 0.35. 0.5 로는 성공 판정 탈락선인 |dx| = 0.30 에서도 0.4 가
#: 남았다. 판정이 "여긴 아니다" 라고 말하는 자리에서 점수가 40 % 를 주면 안 된다.
#: 0.35 면 그 자리가 0.143 이고 |dx| ≥ 0.35 에서 0 이다.
CENTER_ZERO = 0.35
#: 윤곽의 이 비율이 시야 가장자리에 걸리면 border(=잘리지 않음) 항이 0.
#:
#: 🔁 2026-09-14: exp(-b/0.15) → 선형 램프 1 - b/0.15. 실측 분포가 이분적이라
#: (75 % 가 정확히 0 · p90 0.226 · p99 0.279) 지수 감쇠는 잘린 프레임에도 0.22 를
#: 남겼다. 선형이면 0.15 를 넘는 순간 0 이 된다.
BORDER_ZERO = 0.15
#: 대비 기준 — 기존 Q 와 같다 (QualityConfig.contrast_reference).
CONTRAST_REF = 0.35

#: 게이트 항 — 만족하면 아무것도 주지 않고, 무너지면 전부 가져간다.
GATE_PARTS = ("presence", "confidence", "component")
#: 내용 항과 그 가중치. 합 7.5.
CONTENT_WEIGHTS = {
    "area": 3.0,            # 얼마나 크게
    "centering": 2.0,       # 좌우 가운데 (잘렸으면 이 몫을 주지 않는다)
    "border": 2.0,          # 잘리지 않았는가 — "전체가 화면에"
    "contrast": 0.5,
}

#: 항별 **손실 배분표**. ``rule_control.q_losses`` 가 "어느 축이 고칠 부족분인가" 를
#: 고를 때 쓴다.
#:
#: ⚠️ 이것으로 점수를 재구성할 수 없다. 점수는 게이트 × 내용이고 이 표는 **상대적
#: 순위**만 준다 — presence 가 모자라면 훑기, centering·border 가 모자라면 정렬,
#: 이 판단에 필요한 것은 크기 비교뿐이다 (``Q_REMEDY``).
WEIGHTS = {
    "presence": 6.0,
    "area": 3.0,
    "centering": 2.0,
    "border": 2.0,
    "confidence": 1.0,
    "component": 1.0,
    "contrast": 0.5,
}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def view_quality(state) -> tuple[float, dict]:
    """상태 벡터 하나 → (점수, 성분). 점수는 [0, 1].

    마스크가 없으면 모든 항이 0 이다 — "못 찾았다" 가 점수에 그대로 드러나야 한다.
    예전 Q 는 이때도 0.89 였다.
    """
    s = np.asarray(state, float).reshape(-1)
    has = s[_IDX["has_mask"]] > 0.5
    area = float(np.nan_to_num(s[_IDX["area_ratio"]]))
    dx = float(np.nan_to_num(s[_IDX["centroid_dx"]]))
    conf = float(np.nan_to_num(s[_IDX["segmentation_confidence"]]))
    comp = float(np.nan_to_num(s[_IDX["largest_component_ratio"]]))
    border = float(np.nan_to_num(s[_IDX["border_contact_ratio"]]))
    contrast = float(np.nan_to_num(s[_IDX["lumen_contrast"]]))
    contrast_ok = s[_IDX["contrast_valid"]] > 0.5

    if not has:
        parts = {k: 0.0 for k in WEIGHTS}
        return 0.0, parts

    # 잘림. 이 항은 "전체가 화면에 있는가" 이자, 아래에서 centering 을 믿을지의 근거다.
    contained = _clip01(1.0 - max(0.0, border) / BORDER_ZERO)
    parts = {
        "presence": _clip01(area / AREA_SEEN),
        "area": _clip01(area / AREA_FULL),
        # 잘렸으면 보이는 조각의 중심은 해부학적 중심이 아니다 — 그 몫을 주지 않는다.
        "centering": _clip01(1.0 - abs(dx) / CENTER_ZERO) * contained,
        "confidence": _clip01(conf),
        "component": _clip01(comp),
        "border": contained,
        "contrast": _clip01(contrast / CONTRAST_REF) if contrast_ok else 0.0,
    }
    gate = 1.0
    for k in GATE_PARTS:
        gate *= parts[k]
    total = sum(CONTENT_WEIGHTS.values())
    content = sum(w * parts[k] for k, w in CONTENT_WEIGHTS.items()) / total
    return _clip01(gate * content), parts
