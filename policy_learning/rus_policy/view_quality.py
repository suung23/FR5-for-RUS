"""방광 뷰 품질 — "방광을 찾았고, 화면에 크게, 가운데에 있는가".

    q, parts = view_quality(state)        # state: 지각 상태 벡터 (STATE_FEATURE_NAMES)

왜 따로 있는가
--------------
기존 Q_seg (``rus_perception.control.quality``) 는 가중치 합 6.5 중 면적이 1.0 (15 %) 이고
나머지 85 % 가 "마스크가 깨끗한가" (신뢰도·연결성·경계·시간 안정성) 다. 그래서
2026-09-11 팬텀에서 **방광이 전혀 안 보여도 0.89, 완벽하게 잡혀도 0.95** 였다 — 차이 0.06.
방광이 화면의 2 % 만 보이는 시작 자세에서 0.88 이 찍혔고, 성공 판정(면적 ≥ 8 %)과는
아무 관계가 없었다.

이 점수는 **성공 판정과 같은 축**으로 잰다:

  presence   방광이 **보이는가** — 면적 0.5 % 에서 이미 1. 가중치가 가장 크다 (41 %)
  area       얼마나 크게 — 0 → 1 (면적 ≥ 8 %, 성공 문턱)
  centering  1 (좌우 가운데) → 0 (화면 끝), 성공 문턱 |dx| ≤ 0.30 에서 0.4
  + 기존의 마스크 청결 항 (신뢰도·연결성분·경계·대비)

**유무가 가장 무겁다** (2026-09-12: "방광을 모니터링하는 것이 메인 목적"). 유무·면적·중심이
가중치의 76 % 다 (기존 Q_seg 에서 면적은 15 %, 유무는 항이 아예 없었다). 유무를 면적과 나눈
이유: 하나로 묶으면 "겨우 보이는 것" 과 "안 보이는 것" 의 차이가 면적 경사에 묻힌다 — 찾는
것이 목적이면 그 둘의 차이가 가장 커야 한다.

⚠️ **정책의 입력은 바꾸지 않는다.** 상태 벡터의 ``quality`` 특징(15 번)은 정책이 학습 때 본
값이고, 그것을 바꾸면 정책이 처음 보는 입력을 받는다 — 실험 도중 모델을 조용히 바꾸는 것과
같다. 이 점수는 화면(Q_seg)·기록·판정 보고에만 쓴다. 정책이 **이 점수를 좇게** 하려면 이
점수로 라벨을 다시 만들고 재학습해야 한다.

시간 항(temporal_iou · 중심/면적 안정성)은 넣지 않는다. 상태 벡터에 없고, "지금 이 뷰가
좋은가" 에는 필요 없다.
"""

from __future__ import annotations

import math

import numpy as np

from .perception import STATE_FEATURE_NAMES

_IDX = {n: i for i, n in enumerate(STATE_FEATURE_NAMES)}

#: 이 면적이면 area 항이 1 이 된다 — 성공 판정의 면적 문턱과 같다.
AREA_FULL = 0.08
#: 이 면적이면 presence 가 1 이 된다. 0 에서 여기까지 가파르게 오른다 — 사실상 "보이는가" 이지만
#: 계단이 아니라 이어져 있어야 오르막 탐색이 따라 오를 수 있다. 256×256 에서 0.5 % = 327 px 이라
#: 한두 픽셀짜리 오검출로는 안 켜진다.
AREA_SEEN = 0.005
#: centroid_dx 는 [-0.5, 0.5] (0 = 가운데, ±0.5 = 화면 끝). 이 거리에서 centering 이 0.
CENTER_ZERO = 0.5
#: 경계 접촉 감쇠 척도 — 기존 Q 와 같다 (QualityConfig.border_penalty_scale).
BORDER_SCALE = 0.15
#: 대비 기준 — 기존 Q 와 같다 (QualityConfig.contrast_reference).
CONTRAST_REF = 0.35

WEIGHTS = {
    "presence": 6.0,            # 방광이 보이는가 — 가장 무겁다
    "area": 3.0,                # 얼마나 크게
    "centering": 2.0,           # 좌우 가운데
    "confidence": 1.0,
    "component": 1.0,
    "border": 1.0,
    "contrast": 0.5,
}


def view_quality(state) -> tuple[float, dict]:
    """상태 벡터 하나 → (점수, 성분). 점수는 [0, 1].

    마스크가 없으면 presence · area · centering · confidence · component 가 모두 0 이다 —
    "못 찾았다" 가 점수에 그대로 드러나야 한다. 예전 Q 는 이때도 0.89 였다.
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

    parts = {
        "presence": min(1.0, max(0.0, area) / AREA_SEEN) if has else 0.0,
        "area": min(1.0, max(0.0, area) / AREA_FULL) if has else 0.0,
        "centering": max(0.0, 1.0 - abs(dx) / CENTER_ZERO) if has else 0.0,
        "confidence": min(1.0, max(0.0, conf)) if has else 0.0,
        "component": min(1.0, max(0.0, comp)) if has else 0.0,
        "border": math.exp(-max(0.0, border) / BORDER_SCALE),
        "contrast": min(1.0, max(0.0, contrast / CONTRAST_REF)) if (has and contrast_ok) else 0.0,
    }
    total = sum(WEIGHTS.values())
    score = sum(WEIGHTS[k] * v for k, v in parts.items()) / total
    return float(min(1.0, max(0.0, score))), parts
