"""접촉력 창(window)을 콘솔이 고칠 때의 규칙 — 브리지와 시뮬레이터가 같은 것을 쓴다.

조작자가 콘솔에서 고치는 값은 **둘**이다:

* ``floor_n`` — 프로빙(추론) 모드에서 프로브가 복부에 **최소한** 싣는 힘. 접촉력이
  이 아래로 내려가면 조절기가 곧바로 밀어 넣는다. "최소 접촉력" 이 이것이다.
* ``max_n``   — 하드 한계(``safety.max_contact_force_n``). 넘으면 강제 후퇴.

프로빙 모드로 **들어가는 것**은 힘이 아니라 콘솔의 추론 버튼이다
(``teleop.contact_probing_force_trigger: false``, 2026-09-11 이후). 그래서 접촉 문턱
(``teleop.contact_probing_force_n``)은 여기 없다 — 아무 모드도 그 값으로 바뀌지 않는다.

바닥이 조절기에 어떻게 닿는가
------------------------------
``ForceRegulator`` 는 목표 ``target`` 둘레 ``±deadband`` 안에서는 움직이지 않고, 그
아래로 벗어나면 밀어 넣고 위로 벗어나면 뺀다. 즉 조절기가 "밀어 넣기 시작하는 선" 은
``target - deadband`` 다. 바닥이 그 선이 되게 ``target = floor + deadband`` 로 보낸다.
deadband 는 노드가 들고 있는 값(``contact_control.deadband_n``, yaml 0.05)을 브리지가
읽어 넘긴다 — 여기서 상수로 들고 있으면 yaml 이 바뀔 때 조용히 어긋난다.

⚠️ "절대 아래로 안 내려간다" 는 제어기가 보장할 수 있는 문장이 아니다. 조절기는
바닥 아래를 **보고 나서** 민다. 팬텀은 점탄성이라 힘이 압입을 1.5~2.4 s 뒤따르고
(k_c 0.28~0.71 N/mm), 센서 잡음도 0.1 N 급이다. 바닥은 조절기가 반응하는 선이지
벽이 아니다. 복부가 움직이면(호흡) 짧게 바닥 아래로 내려가는 순간은 있다.

나머지 둘은 여기서 **유도**한다. 따로 노출하면 조작자가 불변식을 머리에 들고 있어야
하고, 그 불변식에 걸린 set 은 제어 노드가 통째로 거절한다.

* ``warn_n = max_n - WARN_GAP_N`` — "상승 금지" 선. probe.yaml 의 4.5/5.0 간격을 그대로.
* ``resume_n = floor_n``            — 텔레옵 한계 감시가 조작자에게 돌려주는 선.

제어 노드의 두 불변식 ``target < warn <= max`` (ForceRegulator) 와
``resume < warn <= limit`` (ForceLimitOverride) 는 ``floor + deadband <= max - MIN_WINDOW_N``
이면 둘 다 성립한다 (``MIN_WINDOW_N > WARN_GAP_N``).

**천장은 콘솔이 못 올린다.** ``ceiling_n`` 은 브리지 파라미터
(``bridge.force_limit_ceiling_n``)이고 기본 10.0 N 이다 (2026-10-19 조작자 결정 —
처음 5.0 으로 두었던 것을 올렸다). ⚠️ probe.yaml §2.1 의 조직 안전 근거값은 7 N 이고,
yaml 의 기본 하드 한계 5.0 은 팬텀에서 강제 후퇴가 실제로 걸리는 것을 확인한 값이다.
천장 10 N 은 콘솔이 **그 위까지 올릴 수 있다**는 뜻이지 그 값이 안전하다는 뜻이
아니다 — 조직으로 갈 때는 천장을 7 N 아래로 되돌릴 것.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: 경고선과 하드 한계 사이 [N]. probe.yaml 의 4.5 / 5.0.
WARN_GAP_N = 0.5
#: 조절기 목표(floor + deadband)와 max 사이 최소 간격 [N]. 경고선 아래에 목표가
#: 0.5 N 이상 남도록.
MIN_WINDOW_N = 1.0
#: 바닥의 바닥 [N]. 0 은 "유지하지 않음" 이라 뜻이 다르므로 막는다. 힘 대역을 훑는
#: 실험이 1 N 유지를 쓰므로 그 아래도 조금 남긴다.
MIN_FLOOR_N = 0.5
#: 천장의 기본값 [N]. 2026-10-19 조작자 결정으로 5.0 → 10.0. 모듈 docstring 의 경고를 볼 것.
DEFAULT_CEILING_N = 10.0


@dataclass(frozen=True)
class ForceWindow:
    """제어 노드에 한 번의 set_parameters 로 보낼 값들과, 콘솔이 보는 바닥."""

    floor_n: float
    hold_n: float
    warn_n: float
    max_n: float
    resume_n: float

    def as_parameters(self) -> dict[str, float]:
        """us_diff_ik_node 파라미터 이름 → 값. **한 요청**으로 보내야 한다 — 노드의
        콜백이 넷을 함께 받아야 불변식을 새 값끼리 검사한다."""
        return {
            "contact_control.target_force_n": self.hold_n,
            "safety.warn_contact_force_n": self.warn_n,
            "safety.max_contact_force_n": self.max_n,
            "safety.teleop_resume_force_n": self.resume_n,
        }


def force_window(floor_n: float, max_n: float, deadband_n: float,
                 ceiling_n: float = DEFAULT_CEILING_N) -> ForceWindow:
    """조작자가 준 (floor, max) 와 노드의 deadband 에서 창을 만든다. 규칙에 걸리면
    ``ValueError`` — 사유는 조작자에게 그대로 보여 줄 문장이다."""
    try:
        floor = float(floor_n)
        mx = float(max_n)
        band = float(deadband_n)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"힘은 숫자여야 한다: floor={floor_n!r}, max={max_n!r}, deadband={deadband_n!r}"
        ) from exc
    if not (math.isfinite(floor) and math.isfinite(mx) and math.isfinite(band)):
        raise ValueError("힘은 유한한 값이어야 한다")
    if band < 0.0:
        raise ValueError(f"deadband 는 음수일 수 없다: {band}")
    if floor < MIN_FLOOR_N:
        raise ValueError(f"최소 접촉력 {floor:.2f} N 은 {MIN_FLOOR_N:.1f} N 아래로 내릴 수 없다")
    if mx > ceiling_n + 1e-9:
        raise ValueError(
            f"한계 {mx:.2f} N 은 천장 {ceiling_n:.1f} N 을 넘는다 — "
            "천장은 콘솔이 아니라 probe.yaml 에서 올린다")
    hold = floor + band
    if hold > mx - MIN_WINDOW_N + 1e-9:
        raise ValueError(
            f"최소 접촉력 {floor:.2f} N 은 한계 {mx:.2f} N 보다 "
            f"{MIN_WINDOW_N + band:.2f} N 이상 낮아야 한다 (경고선이 그 사이에 선다)")
    return ForceWindow(floor_n=floor, hold_n=hold, warn_n=mx - WARN_GAP_N, max_n=mx,
                       resume_n=floor)


def floor_of(target_n: float, deadband_n: float) -> float:
    """노드가 들고 있는 (target, deadband) 에서 콘솔이 보여 줄 바닥. ``force_window`` 의 역."""
    return float(target_n) - float(deadband_n)
