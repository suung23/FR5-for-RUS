"""축 하나씩 흔들어 **기하 이득**을 잰다 — 어느 축이 영상을 어디로 옮기는가.

    probe = AxisProbe()
    vel6, label = probe.update(t, visible)   # [vx,vy,vz mm/s, wx,wy,wz °/s], 프로브 프레임

왜 필요한가
-----------
규칙 기반 정렬("방광이 오른쪽에 있으면 이쪽으로 돌려라")을 짜려면 **어느 축이 영상 속
방광을 좌우로 옮기는지와 그 부호**를 알아야 한다. 두 가지가 그것을 안 알려준다:

* **프리핸드 데이터의 기울기는 행동 기울기다.** centroid_dx ↔ θy 의 43 °/단위 는 "치우침이
  이만큼일 때 시연자가 이만큼 돌렸다" 이지, "1° 돌리면 centroid 가 이만큼 움직인다" 가 아니다.
  둘은 다른 양이고, 후자만 제어기에 쓸 수 있다 (2026-09-12 에 이 둘을 섞어 쓸 뻔했다).
* **기존 실기 로그로는 안 풀린다.** eval_v1 의 placebo(방향 무작위) 1767~1808 표본에서 세 회전
  축 모두 1 s 뒤 centroid_dx 변화와의 상관이 |r| ≤ 0.013 이었다 — 지령이 작고(중앙 1.2 °/s)
  여러 축이 섞여 돌아, 축별 이득이 분리되지 않는다.

그래서 **한 번에 한 축만**, 쉬는 구간을 사이에 두고, 양·음 두 방향으로 흔든다. 쉬는 구간이
있어야 시야 자체의 느린 변동(정지 상태에서도 0.6 s 창 간 0.148)을 기준선으로 뺄 수 있다.

**방광이 보이기 전에는 시작하지 않는다.** 안 보이는 채로 흔들면 면적도 중심도 0 이라 어떤
이득도 안 나온다 — 2026-09-12·14 에 세 번 연속으로 그렇게 날렸다 (앞 60 s 동안 마스크 0 %,
회전 세 축이 통째로 거기 들어갔다). 시작은 조작자가 자세를 잡은 **뒤**여야 한다.

읽는 법
-------
``scripts/analyze_axis_probe.py`` 가 구간마다 (지령, centroid_dx 변화, 면적 변화) 를 모아
축별 기울기를 낸다. 면내 축은 centroid_dx 를 크게 움직이고, 면외 축은 면적을 바꾸되
centroid_dx 는 거의 안 건드린다 — 그 차이로 축의 정체를 가른다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

#: vel6 안에서의 자리. 병진 z(빔축)는 admittance 몫이라 넣지 않는다.
AXIS_INDEX = {"x": 0, "y": 1, "thx": 3, "thy": 4, "thz": 5}
DEFAULT_ORDER: tuple[str, ...] = ("thx", "thy", "thz", "x", "y")

REST, MOVE = "쉼", "흔듦"


@dataclass
class AxisProbe:
    """정해진 순서로 축을 하나씩 ± 흔든다. rclpy 없이 돌아 시험할 수 있다.

    Args:
        order: 흔들 축 이름 순서.
        amp_deg: 회전 축의 각속도 [°/s].
        amp_mm_s: 병진 축의 속도 [mm/s].
        move_s: 한 번 흔드는 시간 [s].
        rest_s: 흔든 뒤 쉬는 시간 [s] — 기준선을 잡는 구간.
        repeats: 축·부호마다 반복 횟수.
    """

    order: Sequence[str] = DEFAULT_ORDER
    amp_deg: float = 3.0
    amp_mm_s: float = 5.0
    move_s: float = 1.5
    rest_s: float = 1.5
    repeats: int = 2

    #: 이 면적비 이상 보여야 시작한다. AREA_SEEN(0.005) 보다 넉넉히 잡는다 —
    #: 문턱에 걸친 상태로 시작하면 첫 구간에서 곧바로 사라진다.
    start_area: float = 0.010
    start_hold_s: float = 1.0         # 이만큼 계속 보여야 한다

    _t0: Optional[float] = field(default=None, init=False)
    _seen_since: Optional[float] = field(default=None, init=False)
    #: 진단용 — 시작을 기다린 시간 [s]
    waited_s: float = field(default=0.0, init=False)
    #: 진단용 — 마지막으로 낸 구간 이름 ("thy+" 등) 과 국면
    label: str = field(default="", init=False)
    phase: str = field(default=REST, init=False)

    def __post_init__(self) -> None:
        bad = [a for a in self.order if a not in AXIS_INDEX]
        if bad:
            raise ValueError(f"모르는 축: {bad} (가능: {sorted(AXIS_INDEX)})")

    @property
    def slot_s(self) -> float:
        return self.move_s + self.rest_s

    @property
    def n_slots(self) -> int:
        return len(self.order) * 2 * self.repeats

    @property
    def total_s(self) -> float:
        return self.n_slots * self.slot_s

    def _slot(self, k: int) -> tuple[str, float]:
        """k 번째 구간의 (축 이름, 부호). 부호는 **번갈아** 간다.

        같은 부호를 몰아서 하면 반복을 늘릴수록 시작 자세에서 멀어진다 (repeats=3 · 12.5°
        걸음이면 한쪽으로 37.5°). 번갈아 가면 매 구간이 직전 자세로 되돌아와, 큰 진폭에서도
        시작 자세 근처에 머문다 — 안전하고, 구간마다 기준선이 같아 측정도 낫다.
        """
        per_axis = 2 * self.repeats
        name = self.order[(k // per_axis) % len(self.order)]
        sign = 1.0 if (k % 2) == 0 else -1.0
        return name, sign

    def update(self, t: float, area: Optional[float] = None) -> tuple[np.ndarray, str]:
        """지금 내보낼 vel6 과 구간 이름. 순서를 다 돌면 0 을 내고 ``done`` 을 붙인다.

        ``area`` 를 주면 방광이 ``start_area`` 이상 ``start_hold_s`` 동안 보일 때까지
        시작하지 않는다 (그동안 ``wait`` 를 돌려준다). ``None`` 이면 곧바로 시작한다.
        """
        if self._t0 is None and area is not None:
            ok = area == area and area >= self.start_area
            if not ok:
                self._seen_since = None
            elif self._seen_since is None:
                self._seen_since = t
            if self._seen_since is None or (t - self._seen_since) < self.start_hold_s:
                if self._seen_since is None and self.waited_s == 0.0:
                    self.waited_s = 1e-9          # 기다린 적이 있다는 표시
                self.phase, self.label = REST, "wait"
                return np.zeros(6, float), self.label
            self.waited_s = t - (self._seen_since - self.start_hold_s)
        if self._t0 is None:
            self._t0 = t
        dt = t - self._t0
        vel = np.zeros(6, float)
        if dt >= self.total_s:
            self.phase, self.label = REST, "done"
            return vel, self.label
        k = int(dt // self.slot_s)
        name, sign = self._slot(k)
        in_slot = dt - k * self.slot_s
        self.label = f"{name}{'+' if sign > 0 else '−'}"
        if in_slot < self.move_s:
            self.phase = MOVE
            i = AXIS_INDEX[name]
            vel[i] = sign * (self.amp_deg if i >= 3 else self.amp_mm_s)
        else:
            self.phase = REST
        return vel, self.label

    def describe(self) -> str:
        if self._t0 is None:
            return (f"축 흔들기 **대기** — 방광이 면적비 {self.start_area:.3f} 이상으로 "
                    f"{self.start_hold_s:.0f} s 보여야 시작한다 "
                    f"({self.n_slots} 구간 · {self.total_s:.0f} s)")
        return f"축 흔들기 {self.label} {self.phase}"
