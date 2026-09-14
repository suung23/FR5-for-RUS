"""법선 정렬 — Q_seg 가 합격한 **여유에서만** 결합을 개선한다.

    na = NormalAlign()
    vel6, note = na.step(t, shear_xy=(fx, fy))

왜 여기 있는가
--------------
Q_raw(음향 결합)를 움직이는 고리는 힘 하나뿐이었다. 그런데 실측은 결합이 힘보다 **자세**에
훨씬 강하게 반응한다고 말한다 (2026-09-14 본 시험: Q_raw↔기울임 r=+0.49·+0.68 vs
Q_raw↔힘 r=−0.09·−0.27). 힘 탐색은 27 주기 만에 경사 0.0002 로 수렴했다 — 자기가 가진
축에서는 더 짜낼 것이 없다.

프로브가 피부 법선에서 벗어나면 구경 일부가 뜨고, 그때 **전단력이 생긴다.** 전단을 0 으로
만드는 회전이 곧 법선 정렬이다.

왜 합격선 위에서만 도는가
-------------------------
기울임(thx·thy)은 이미 Q_seg 의 것이다 — 방광을 가운데 두는 데 쓴다. 같은 축을 두 목적이
다투면 그 축은 둘 중 누구의 의도도 아니게 된다. 그래서 **Q_seg 가 합격선을 넘어 제어기가
할 일이 없는 동안에만** 돈다. 합격선 아래로 떨어지면 즉시 원래 일로 돌아간다.

부호를 왜 배우는가
------------------
"전단이 +x 면 어느 쪽으로 기울여야 하는가" 는 프로브 프레임 규약·마운트 방향·센서 축
순서가 함께 정하는 값이다. 틀리게 박아 두면 **전단을 키우는 쪽으로 돈다.** 이 파일은
``check_s`` 마다 전단이 줄었는지 보고, 늘었으면 부호를 뒤집는다 — 한 비트짜리 적응이고
틀려도 한 번의 확인 주기만 잃는다. 프리핸드로는 구할 수 없는 값이라 재는 수밖에 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

#: vel6 안에서 회전이 앉는 자리 (rule_control 과 같은 규약).
THX, THY = 3, 4


@dataclass
class NormalAlign:
    """전단력을 0 으로 미는 아주 느린 회전.

    Args:
        shear_ok_n: 전단이 이 아래면 충분히 법선이다 — 더 건드리지 않는다.
        k_p: 비례 이득 [°/s per N].
        max_deg_s: 속도 상한. **아주 느리다** — 남는 여유에서 하는 일이라 급할 이유가 없고,
            빠르면 Q_seg 를 흔든다.
        max_excursion_deg: 누적 회전 상한. 이만큼 갔는데도 전단이 안 줄면 자세가 아니라
            다른 것이 원인이다.
        check_s: 전단이 줄었는지 확인하는 간격. 부호는 여기서 배운다.
        grow_margin_n: 이만큼 늘었으면 "늘었다" 로 본다 (잡음으로 뒤집지 않는다).
    """

    shear_ok_n: float = 0.30
    k_p: float = 1.0
    max_deg_s: float = 0.5
    max_excursion_deg: float = 4.0
    check_s: float = 4.0
    grow_margin_n: float = 0.05

    sign: float = field(default=1.0, init=False)      # 실측으로 배운다
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2), init=False)
    flips: int = field(default=0, init=False)
    _t0: Optional[float] = field(default=None, init=False)
    _prev_t: Optional[float] = field(default=None, init=False)
    _mark_t: Optional[float] = field(default=None, init=False)
    _mark_shear: float = field(default=float("nan"), init=False)

    def reset(self) -> None:
        """자세를 새로 잡았다 — 누적을 비운다. **부호는 남긴다** (같은 장비의 성질이다)."""
        self.pos = np.zeros(2)
        self._t0 = self._prev_t = self._mark_t = None
        self._mark_shear = float("nan")

    @property
    def excursion_deg(self) -> float:
        return float(np.abs(self.pos).sum())

    def step(self, t: float, shear_xy: Optional[Sequence[float]]) -> tuple[np.ndarray, str]:
        """지금 낼 vel6 과 한 줄 설명. 할 일이 없으면 0 을 낸다."""
        vel = np.zeros(6, float)
        if shear_xy is None or not all(np.isfinite(shear_xy)):
            return vel, "법선 정렬 — 전단을 못 읽는다"
        fx, fy = float(shear_xy[0]), float(shear_xy[1])
        mag = float(np.hypot(fx, fy))

        if mag <= self.shear_ok_n:
            self._mark_t, self._mark_shear = None, float("nan")
            return vel, f"법선 정렬 — 전단 {mag:.2f} N (충분히 법선)"
        if self.excursion_deg >= self.max_excursion_deg:
            return vel, (f"법선 정렬 — 누적 {self.excursion_deg:.1f}° 로 상한이다 "
                         f"(전단 {mag:.2f} N) · 자세 말고 다른 원인이다")

        dt = 0.0 if self._prev_t is None else max(0.0, t - self._prev_t)
        self._prev_t = t
        # 부호 배우기: check_s 마다 전단이 줄었는지 본다. 늘었으면 뒤집는다.
        if self._mark_t is None:
            self._mark_t, self._mark_shear = t, mag
        elif (t - self._mark_t) >= self.check_s:
            if mag > self._mark_shear + self.grow_margin_n:
                self.sign = -self.sign
                self.flips += 1
            self._mark_t, self._mark_shear = t, mag

        # 전단 +x 를 줄이려면 y 축으로, +y 를 줄이려면 x 축으로 (부호는 배운 것을 쓴다).
        w = self.sign * self.k_p * np.array([-fy, fx], float)
        w = np.clip(w, -self.max_deg_s, self.max_deg_s)
        # **상한은 두 축의 합에 건다.** 축마다 걸면 합이 두 배까지 간다 — 프로브가 실제로
        # 시작 자세에서 얼마나 벗어났는지는 합이 말한다 (rule_control.excursion_deg 와 같은
        # 규약). 걸리면 그 걸음을 넘지 않는 만큼만 간다.
        nxt = self.pos + w * dt
        over = float(np.abs(nxt).sum()) - self.max_excursion_deg
        if over > 0.0:
            step = float(np.abs(w * dt).sum())
            w = w * max(0.0, (step - over) / step) if step > 1e-12 else w * 0.0
        self.pos = self.pos + w * dt
        vel[THX], vel[THY] = float(w[0]), float(w[1])
        return vel, (f"법선 정렬 — 전단 {mag:.2f} N → thx {w[0]:+.2f} thy {w[1]:+.2f} °/s "
                     f"(누적 {self.excursion_deg:.1f}° · 부호 {self.sign:+.0f}"
                     + (f", {self.flips} 회 뒤집음" if self.flips else "") + ")")
