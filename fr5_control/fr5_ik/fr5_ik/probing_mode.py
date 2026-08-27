"""접근 / 접촉 프로빙 모드 전환기 (DESIGN_NOTES §10.6, 2026-08-25).

teleop 한 세션은 두 단계로 나뉜다.

``approach``
    프로브가 아무것도 만지지 않은 단계. 초기 자세로 데려가는 것이 목적이므로
    자유공간 스케일(150 mm/s · 1.5 rad/s)을 쓴다.

``contact_probing``
    프로브가 조직·팬텀에 닿은 뒤. 접촉용 상한(10 mm/s · 0.2 rad/s)으로 내려간다.

전환은 **법선력이 문턱을 넘는 순간** 일어난다. 조작자가 손으로 바꾸는 것이 아니다 —
접촉은 예고 없이 시작되고, 그 순간에 손이 파라미터를 만지고 있을 리 없다.

``joint_command_limiter`` 와 같은 이유로 rclpy 없이 돈다. 이 판정의 버그는 실기 앞이
아니라 pytest 로 찾아야 하는 종류다.

왜 되돌리지 않는가
------------------
전환은 **단방향**이다. 힘이 다시 0 으로 떨어져도 접근 모드로 돌아가지 않는다.

프로브를 살짝 떼는 것은 접촉 작업의 일부이지 작업의 끝이 아니다. 그때마다 상한이
15 배로 뛰면, 조작자가 같은 손동작을 하는 동안 로봇의 반응이 달라진다 — 접촉 작업
중에 가장 원하지 않는 것이다. 되돌리려면 세션을 다시 시작한다
(``fr5_ik.teleop_frame`` 의 latch, ``fr5_control.contact_state`` 의 걸쇠와 같은 규약).

확인 시간
---------
문턱을 넘은 것이 잡음이나 충격 스파이크 한 번일 수 있다. 연속으로 일정 시간 넘어야
전환한다. 다만 **접촉 쪽으로 가는 전환이므로 짧게** 둔다 — 늦게 조이는 것이 일찍
조이는 것보다 위험하다. 접촉을 놓치면 자유공간 속도로 조직을 미는 시간이 길어진다.
"""
from __future__ import annotations

__all__ = ["APPROACH", "CONTACT_PROBING", "ProbingModeSwitch"]

#: 자유공간 스케일. 초기 자세 접근용.
APPROACH = "approach"

#: 접촉용 상한. 프로브가 닿은 뒤.
CONTACT_PROBING = "contact_probing"


class ProbingModeSwitch:
    """법선력으로 접근/접촉 프로빙 모드를 가른다.

    시간은 밖에서 받는다 (``dt_s``). 벽시계를 안 읽으므로 기록을 되돌려 그대로 다시
    판정할 수 있다.
    """

    def __init__(self, enter_force_n: float, confirm_s: float = 0.02) -> None:
        """전환기를 만든다.

        Args:
            enter_force_n: 접촉 프로빙으로 넘어가는 법선력 [N]. 양수 = 압축.
            confirm_s: 연속으로 문턱을 넘어야 하는 시간 [s]. 0 이면 표본 하나로 전환.

        Raises:
            ValueError: 문턱이 0 이하이거나 확인 시간이 음수다.
        """
        if enter_force_n <= 0.0:
            raise ValueError(f"enter_force_n 은 양수여야 한다: {enter_force_n}")
        if confirm_s < 0.0:
            raise ValueError(f"confirm_s 는 음수일 수 없다: {confirm_s}")

        self.enter_force_n = float(enter_force_n)
        self.confirm_s = float(confirm_s)
        self.mode = APPROACH
        self._above_s = 0.0

    def update(self, normal_force_n: float, dt_s: float) -> str:
        """표본 하나를 넣고 현재 모드를 돌려준다.

        Args:
            normal_force_n: 법선력 [N]. 양수 = 압축 (``F_n = sign x F_z``).
            dt_s: 직전 표본과의 간격 [s].

        Returns:
            :data:`APPROACH` 또는 :data:`CONTACT_PROBING`.
        """
        if self.mode is CONTACT_PROBING or self.mode == CONTACT_PROBING:
            return self.mode

        if normal_force_n >= self.enter_force_n:
            self._above_s += max(0.0, float(dt_s))
            if self._above_s >= self.confirm_s:
                self.mode = CONTACT_PROBING
        else:
            # 문턱 아래로 한 번만 내려가도 처음부터 다시 센다. "대체로 넘었다" 는
            # 접촉이 아니다.
            self._above_s = 0.0
        return self.mode

    @property
    def in_contact_probing(self) -> bool:
        """접촉 프로빙 모드인가."""
        return self.mode == CONTACT_PROBING

    @property
    def confirm_progress(self) -> float:
        """확인 창을 몇 % 지났는가. 0..1."""
        if self.mode == CONTACT_PROBING:
            return 1.0
        if self.confirm_s <= 0.0:
            return 1.0 if self._above_s > 0.0 else 0.0
        return min(1.0, self._above_s / self.confirm_s)

    def reset(self) -> None:
        """접근 모드로 되돌린다. 새 세션을 시작할 때만 쓴다."""
        self.mode = APPROACH
        self._above_s = 0.0
