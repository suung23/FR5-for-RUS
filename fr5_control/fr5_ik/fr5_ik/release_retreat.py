"""정책이 régime 을 놓은 뒤의 **이탈 후퇴** 판정. rclpy 없이 돈다.

왜 필요한가 (2026-09-12 실측)
-----------------------------
Stop 은 régime 만 푼다. 그 순간 프로브는 여전히 조직에 눌려 있다 — 마지막 에피소드의
접촉력은 **3.39 N** 이었다. 힘 조절기는 손을 떼고 z 는 조작자에게 돌아가는데, 조작자의
지령은 이제 접촉에 맞선다. 같은 세션에서 정책 전 텔레옵의 정지 시 지령 선행분은
0.50 · 0.56° 였고, Stop 뒤에는 2.87 · 2.77 · 1.77 · 3.58° 였다 (추종오차 밴드 5°).
즉 로봇이 지령을 5~7 배 더 못 따라간다 — 조작자에게는 "텔레옵이 안 된다" 로 보인다.

예전에는 이 자리를 워치독이 대신 처리했다. 런북 §G 의 "데드맨을 놓으면 워치독 후퇴" 가
그것인데, touch_twist 가 데드맨 해제 중에도 0 twist 를 계속 내므로 twist 는 신선하고
워치독은 걸리지 않는다. 즉 **놓아도 물러나지 않는다.**

그래서 놓는 쪽에서 물러난다. 새 동작이 아니라 워치독 후퇴와 같은 동작·같은 상한이다:
프로브 −z 로 ``speed_m_s``, ``until_force_n`` 아래로 떨어지면 종료, ``max_s`` 를 넘으면
포기하고 그 자리에 선다.

조작자가 이깁니다
-----------------
후퇴 중에 조작자가 0 이 아닌 지령을 내면 즉시 접는다. 이 후퇴는 편의이지 안전장치가
아니다 — 안전장치는 5 N 강제 후퇴이고 그것은 조절기 안에 따로 있다. 조작자가 손을
쓰기 시작한 순간에도 로봇이 자기 계획을 계속하면, 그것이 이 스택에서 가장 위험한
종류의 놀라움이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: 조작자가 "지령을 냈다" 고 볼 최소 크기. Touch 는 데드맨을 놓아도 정확히 0 을 내므로
#: (touch_twist_node.cpp) 0 과 구분하기만 하면 된다.
OPERATOR_EPS = 1e-6


@dataclass
class ReleaseRetreatDecision:
    """한 주기의 판정."""

    #: 프로브 −z 속도 [m/s]. ``None`` 이면 후퇴하지 않는다 (조작자 지령을 그대로 쓴다).
    v_z: Optional[float]
    #: 이번 주기에 후퇴가 끝났는가. 끝난 이유는 ``reason`` 에 있다.
    finished: bool
    #: 로그로 낼 사유. 상태가 바뀔 때만 채운다.
    reason: str = ""


class ReleaseRetreat:
    """이탈 후퇴 상태기계.

    Args:
        speed_m_s: 후퇴 속도 [m/s]. 워치독 후퇴와 같은 값을 쓴다.
        until_force_n: 이 접촉력 아래로 내려가면 끝난다 [N].
        max_s: 상한 시간 [s]. 힘이 안 떨어져도 여기서 멈춘다.
        enabled: 거짓이면 ``arm`` 이 아무것도 하지 않는다.
    """

    def __init__(self, speed_m_s: float, until_force_n: float, max_s: float,
                 enabled: bool = True) -> None:
        if speed_m_s <= 0.0:
            raise ValueError(f"speed_m_s 는 양수여야 한다: {speed_m_s}")
        if until_force_n < 0.0:
            raise ValueError(f"until_force_n 은 음수일 수 없다: {until_force_n}")
        if max_s <= 0.0:
            raise ValueError(f"max_s 는 양수여야 한다: {max_s}")
        self.speed_m_s = float(speed_m_s)
        self.until_force_n = float(until_force_n)
        self.max_s = float(max_s)
        self.enabled = bool(enabled)
        self.active = False

    def arm(self, force_n: float) -> bool:
        """régime 을 놓는 순간 부른다. 실제로 무장했으면 참.

        이미 접촉이 풀려 있으면 무장하지 않는다 — 뗀 프로브를 더 뒤로 물릴 이유가 없고,
        그 움직임은 조작자가 지시하지 않은 것이다.
        """
        if not self.enabled or force_n <= self.until_force_n:
            self.active = False
            return False
        self.active = True
        return True

    def cancel(self) -> None:
        self.active = False

    def update(self, elapsed_s: float, force_n: float, wrench_fresh: bool,
               operator_cmd_max: float) -> ReleaseRetreatDecision:
        """한 주기.

        Args:
            elapsed_s: 무장 이후 경과 [s].
            force_n: 지금 접촉력 [N]. 판정 스칼라는 호출자가 정한다 (‖F‖ 또는 F_n).
            wrench_fresh: 그 값이 신선한가. 낡았으면 힘으로 끝내지 않는다 —
                wrench 가 죽은 채로 "힘이 0 이다" 를 믿으면 영원히 물러난다.
            operator_cmd_max: 조작자 twist 의 최대 절대성분. 0 이 아니면 접는다.
        """
        if not self.active:
            return ReleaseRetreatDecision(None, False)
        if operator_cmd_max > OPERATOR_EPS:
            self.active = False
            return ReleaseRetreatDecision(
                None, True, "조작자가 지령을 냈다 — 이탈 후퇴를 접는다")
        if wrench_fresh and force_n < self.until_force_n:
            self.active = False
            return ReleaseRetreatDecision(
                None, True,
                f"이탈 후퇴 종료 — F {force_n:.2f} N < {self.until_force_n:.2f} N")
        if elapsed_s > self.max_s:
            self.active = False
            return ReleaseRetreatDecision(
                None, True,
                f"이탈 후퇴 시간 상한 {self.max_s:.1f} s — 그 자리에 선다 "
                + ("(힘이 안 떨어진다)" if wrench_fresh else "(wrench 두절)"))
        return ReleaseRetreatDecision(-self.speed_m_s, False)
