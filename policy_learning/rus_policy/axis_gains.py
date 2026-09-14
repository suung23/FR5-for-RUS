"""실측 기하 이득 — 규칙 제어기의 축·부호·계수.

    g = AxisGains.load(path)          # 없으면 None
    dx_mm = g.centering_mm(centroid_dx)

무엇을 담는가
-------------
``scripts/analyze_axis_probe.py --save`` 가 쓴다. 축마다 두 값이다:

    centroid_per_unit   1 mm(또는 1°) 움직이면 방광 중심이 화면에서 얼마나 옮겨지는가
    area_per_unit       같은 지령이 보이는 면적을 얼마나 바꾸는가

**면내 축은 앞이 크고 뒤가 작다. 면외 축은 반대다.** 그 차이로 축의 정체가 갈린다.

왜 파일로 두는가
----------------
* 하드코딩하면 팬텀·프로브·자세가 바뀔 때 조용히 틀린다. 이 값은 장비의 성질이지
  알고리즘의 성질이 아니다.
* 프리핸드 데이터로는 못 구한다 — 거기서 나오는 기울기는 "치우침이 이만큼일 때 시연자가
  이만큼 움직였다" 는 **행동** 기울기이고, 제어기에 필요한 것은 **기하** 이득이다.
  2026-09-12 에 이 둘을 섞어 쓸 뻔했다.

2026-09-12 실측 (`runs/probe2`): x 병진이 면내(중심 0.00708 /mm, r=+1.00), y 병진이
면외(면적 0.00161 /mm). 회전 세 축은 그 구간에 방광이 안 보여 못 쟀다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#: 구간마다 지령 부호와 반응 부호가 이 비율 이상 맞아야 그 축을 믿는다.
#:
#: 상관(r) 을 쓰면 안 된다 — 크기에 무관해서 **작지만 깨끗한** 효과를 **크지만 살짝
#: 지저분한** 효과보다 위에 놓는다. 2026-09-14 실측에서 실행조차 안 된 y 병진이 r=0.90
#: 으로 thy(r=0.78, 실제 효과 14 배)를 이겼다. r 은 진단으로만 남긴다.
MIN_SIGN_AGREEMENT = 0.7
#: 옛 이름 (호환).
MIN_R = 0.8
#: 중심을 이만큼 못 옮기는 축은 정렬에 못 쓴다 [1/mm]. 잡음(1 s 표준편차 0.014)을
#: 한 걸음(수 mm)으로 이기려면 최소한 이 정도는 돼야 한다.
MIN_CENTERING_GAIN = 0.002

#: 제어기가 **스스로 고르면 안 되는** 축.
#:
#: ``thz`` 는 빔축(프로브 z) 둘레의 회전이다. 이것은 프로브를 옮기는 것이 아니라 **영상면
#: 자체를 자기 축으로 돌린다** — 장축/단축이 바뀌어 횡단면이 사면(斜面)이 된다. 그러면
#: 화면의 방광은 같은 방광이 아니고, 면적도 중심도 다른 절단면의 값이다.
#:
#: 이득만 보면 좋아 보이는 것이 함정이다 (2026-09-14 실측: 부호 일치 100 %, 면적 이득
#: +0.00401/° 로 세 축 중 최대). 면적이 크게 변하는 것은 **찾고 있어서가 아니라 절단면이
#: 갈리고 있어서**다. 부호가 깨끗한 것도 같은 이유다 — 기하가 단조롭다.
#:
#: 임상에서도 프로브 회전은 표준 횡단 뷰를 버리는 동작이라, 찾기 동작으로 쓸 수 없다.
#: 재는 것 자체는 막지 않는다 (axis_probe 는 그대로 세 축을 흔든다) — 고르지 못하게만 한다.
EXCLUDED_AXES: tuple[str, ...] = ("thz",)


@dataclass
class AxisGains:
    """축 이름 → 이득. 단위는 축에 따라 mm 또는 °."""

    centroid_per_unit: dict[str, float] = field(default_factory=dict)
    area_per_unit: dict[str, float] = field(default_factory=dict)
    r_centroid: dict[str, float] = field(default_factory=dict)
    sign_agreement: dict[str, float] = field(default_factory=dict)
    source: str = ""

    # ---- 파일 -------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> Optional["AxisGains"]:
        p = Path(path)
        if not p.is_file():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(centroid_per_unit={k: float(v) for k, v in d.get("centroid_per_unit", {}).items()},
                   area_per_unit={k: float(v) for k, v in d.get("area_per_unit", {}).items()},
                   r_centroid={k: float(v) for k, v in d.get("r_centroid", {}).items()},
                   sign_agreement={k: float(v) for k, v in d.get("sign_agreement", {}).items()},
                   source=str(d.get("source", "")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "centroid_per_unit": self.centroid_per_unit,
            "area_per_unit": self.area_per_unit,
            "r_centroid": self.r_centroid,
            "sign_agreement": self.sign_agreement,
            "source": self.source,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 축 고르기 ---------------------------------------------------------
    def centering_axis(self) -> Optional[str]:
        """좌우 정렬에 쓸 축 — 부호가 일관되면서 중심을 가장 크게 옮기는 축.

        부호 일치를 먼저 걸러 **방향이 믿을 만한** 축만 남기고, 그중 이득이 가장 큰 것을
        고른다. 이득이 클수록 같은 오차를 더 적게 움직여 고칠 수 있고, 잡음(1 s 표준편차
        0.014)에 덜 휘둘린다.
        """
        def trusted(k: str) -> bool:
            a = self.sign_agreement.get(k)
            if a is None:                       # 옛 파일 — 상관으로 대신한다
                return abs(self.r_centroid.get(k, 0.0)) >= MIN_R
            return a >= MIN_SIGN_AGREEMENT
        got = [(k, v) for k, v in self.centroid_per_unit.items()
               if abs(v) >= MIN_CENTERING_GAIN and trusted(k) and k not in EXCLUDED_AXES]
        return max(got, key=lambda kv: abs(kv[1]))[0] if got else None

    def search_axis(self) -> Optional[str]:
        """찾기에 쓸 축 — 면적을 가장 크게 바꾸되 정렬 축은 아닌 축 (면외).

        ``EXCLUDED_AXES`` 는 이득이 아무리 좋아도 안 고른다. thz 의 면적 이득이 세 축 중
        가장 큰데, 그것은 방광을 찾고 있다는 뜻이 아니라 절단면이 갈리고 있다는 뜻이다.
        """
        keep = self.centering_axis()
        got = [(k, v) for k, v in self.area_per_unit.items()
               if k != keep and v == v and k not in EXCLUDED_AXES]
        return max(got, key=lambda kv: abs(kv[1]))[0] if got else None

    # ---- 제어 --------------------------------------------------------------
    def centering_move(self, centroid_dx: float, axis: Optional[str] = None) -> float:
        """중심을 0 으로 되돌리는 **변위** [mm 또는 °]. 축이 없으면 0."""
        axis = axis or self.centering_axis()
        g = self.centroid_per_unit.get(axis or "", 0.0)
        return -float(centroid_dx) / g if g else 0.0
