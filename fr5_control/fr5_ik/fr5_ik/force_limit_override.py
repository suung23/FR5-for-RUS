"""텔레옵 힘 한계 감시 — 조작자 위에 얹는 안전 루프 (2026-09-14).

왜 있는가
---------
``safety.max_contact_force_n`` (5.0) 은 2026-09-14 까지 **접촉 프로빙에서만** 살아
있었다. 그 5 N 후퇴는 :class:`fr5_ik.force_regulator.ForceRegulator` 안에 있고,
조절기는 ``mode_switch.in_contact_probing`` 일 때만 돈다. 그런데 2026-09-11 에
``teleop.contact_probing_force_trigger`` 가 false 로 내려가면서 그 모드로 들어가는
유일한 문이 콘솔의 policy.enable 버튼이 됐다.

즉 **버튼을 누르기 전 텔레옵 구간에는 힘을 보는 코드가 한 줄도 없었다.** 데드맨을
쥐고 있는 한 조작자는 5 N 이든 15 N 이든 그대로 밀 수 있었다. probe.yaml 은 그
구간을 두고 "5 N 후퇴가 유일한 보호다" 라고 적어 두었지만, 그 후퇴는 그 구간에서
실행되지 않았다 — 있다고 적힌 보호와 있는 보호가 달랐다.

이 모듈이 그 한 층이다. 조절기와 **겹치지 않는다**: 접촉 프로빙에서는 호출되지
않으며 (z 의 주인은 그때 조절기다), 접근/텔레옵에서만 조작자 지령 위에 얹힌다.

무엇을 하는가
-------------
``F ≥ limit`` 이면 조작자 지령을 **덮어쓰고** 프로브 −z 로 물러난다. ``F ≤ resume``
까지 내려오면 조작자에게 돌려준다.

이력(hysteresis)이 필수인 이유는 팬텀이 점탄성이기 때문이다. 실측 k_c 는
0.28~0.71 N/mm 이고 **힘이 압입을 1.5~2.4 s 뒤따른다** (force-hold 한계주기 분석,
2026-09-04). 해제선을 한계 바로 밑에 두면 후퇴를 멈춘 뒤에도 힘이 계속 올라와 곧바로
다시 걸리고, 그 진동의 꼭짓점은 매번 한계 위다. 해제선은 한계에서 한참 내려와야 한다.

같은 지연 때문에 **한계 초과를 막는 것은 이 루프가 아니다.** 힘이 5 N 으로 읽힐 때
프로브는 이미 그보다 깊이 들어가 있다. 이 루프는 초과를 짧게 끊는 층이고, 초과
자체를 막는 것은 아래 배리어와 침투축 속도 상한이다. 셋을 함께 봐야 한다.

배리어 — 넘기 전에
------------------
``F ≥ warn`` 이면 **전진 성분만** 0 으로 만든다 (``v_z > 0``). 빼는 방향은 손대지
않는다. 이것은 :mod:`fr5_ik.force_regulator` 가 ``target < warn`` 구조로 얻는
"경고 이상에서 전진 금지" 를 조작자 축에 그대로 옮긴 것이다.

배리어가 없으면 이 루프는 한계와 해제선 사이를 오간다: 조작자가 계속 밀고 있으므로
돌려주는 즉시 다시 올라간다. 그래서 기본값이 켜짐이다. 끄면 (``barrier_enabled=False``)
5 N 루프만 남는데, 그때 조작자는 **자기 손이 반복해서 무시당하는 것**을 보게 된다.

어느 프레임인가
---------------
twist 의 z 는 **프로브 침투축**이어야 한다. 호출자는 ``_remap_twist`` 뒤에 부를 것 —
텔레옵 노드가 내는 스타일러스 body 기준 z 는 걸쇠(``linear_frame: latched``) 를 지나며
프로브 z 와 어긋나므로, 그 자리에서 z 를 조이면 엉뚱한 축을 조인다.

빠져나가는 길은 항상 열려 있다
------------------------------
후퇴 중에도 조작자가 더 빨리 빼는 것은 통과시킨다 (``min(v_z_cmd, -retreat_speed)``).
나머지 다섯 축은 0 이다 — 한계를 넘은 힘으로 조직 위를 옆으로 끄는 것은 누르고 있는
것보다 나을 것이 없다.

wrench 가 두절되면
------------------
후퇴 중이었다면 **계속 물러난다.** 힘을 모르는 채로 "풀렸다" 고 판단할 수는 없고,
무한히 물러나는 것은 시간 상한(``max_override_s``)이 막는다 — 워치독 후퇴
(``us_diff_ik_node._retreat_twist``) 와 같은 규약이다. 상한에 닿으면 **선다**
(여섯 축 0). 힘이 안 떨어지는데 계속 물러나는 것은 그 자체로 다른 고장이다.

통과 중이었다면 지령을 그대로 보낸다. 이 층은 wrench 가 없을 때 로봇을 멈추는
장치가 아니다 — 그것은 상위의 몫이고, 여기서 함께 하면 두 곳이 같은 일을 한다.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PASS", "RETREAT", "HOLD", "ForceLimitOverride", "OverrideOutput"]

#: 조작자 지령이 (배리어를 빼면) 그대로 나간다.
PASS = "pass"

#: 한계를 넘어 물러나는 중. 조작자 축은 닫혀 있다.
RETREAT = "retreat"

#: 시간 상한에 닿아 선 상태. 힘이 해제선 아래로 내려와야 풀린다.
HOLD = "hold"


@dataclass(frozen=True)
class OverrideOutput:
    """한 주기의 결정.

    Attributes:
        twist: 하류로 내보낼 6 축 지령.
        state: :data:`PASS` · :data:`RETREAT` · :data:`HOLD`.
        overriding: 조작자 지령을 덮어썼는가. 진단 토픽·화면에 그대로 쓴다.
        reason: 무엇이 이 값을 정했는가. 바뀔 때만 로그로 낸다.
    """

    twist: list
    state: str
    overriding: bool
    reason: str


class ForceLimitOverride:
    """텔레옵 지령 위에 얹는 힘 한계 감시. 시간은 밖에서 받는다 (``dt_s``)."""

    def __init__(
        self,
        limit_n: float,
        resume_n: float,
        warn_n: float,
        retreat_speed_m_s: float,
        max_override_s: float,
        enabled: bool = True,
        barrier_enabled: bool = True,
    ) -> None:
        """감시를 만든다.

        Args:
            limit_n: 이 힘 이상이면 조작자를 덮어쓰고 물러난다 [N].
            resume_n: 이 힘 이하로 내려오면 조작자에게 돌려준다 [N].
            warn_n: 이 힘 이상에서는 전진 성분을 통과시키지 않는다 [N].
            retreat_speed_m_s: 후퇴 속도 [m/s]. 양수로 준다.
            max_override_s: 한 번의 후퇴가 이어질 수 있는 시간 상한 [s].
            enabled: 꺼 두면 지령을 그대로 통과시킨다 (배리어도 함께 꺼진다).
            barrier_enabled: 경고 이상 전진 금지를 쓸 것인가.

        Raises:
            ValueError: 값이 서로 모순된다.
        """
        if retreat_speed_m_s <= 0.0:
            raise ValueError(f"retreat_speed_m_s 는 양수여야 한다: {retreat_speed_m_s}")
        if max_override_s <= 0.0:
            raise ValueError(f"max_override_s 는 양수여야 한다: {max_override_s}")
        if not (resume_n < warn_n <= limit_n):
            # 해제선이 경고선 위에 있으면 돌려주는 순간이 곧 배리어가 막는 순간이라
            # 조작자는 풀렸다는 말을 듣고도 못 움직인다. 경고선이 한계 위면 배리어가
            # 영원히 안 걸리고, 이 층은 사후 후퇴만 남는다. 만들 때 막는다.
            raise ValueError(
                f"resume({resume_n}) < warn({warn_n}) <= limit({limit_n}) 이어야 한다"
            )

        self.limit_n = float(limit_n)
        self.resume_n = float(resume_n)
        self.warn_n = float(warn_n)
        self.retreat_speed_m_s = float(retreat_speed_m_s)
        self.max_override_s = float(max_override_s)
        self.enabled = bool(enabled)
        self.barrier_enabled = bool(barrier_enabled)

        self.state = PASS
        #: 지금 후퇴가 시작된 뒤로 흐른 시간 [s].
        self.elapsed_s = 0.0

    # -- 판정 ---------------------------------------------------------------

    def reset(self) -> None:
        """상태를 접촉 없는 처음으로 되돌린다.

        régime 이 바뀔 때 (접촉 프로빙 진입·이탈) 호출자가 부른다. 남은 ``elapsed_s``
        를 들고 régime 을 넘어가면, 돌아왔을 때 상한이 이미 차 있어 첫 후퇴가
        시작하자마자 HOLD 로 떨어진다.
        """
        self.state = PASS
        self.elapsed_s = 0.0

    def update(
        self,
        twist,
        force_n: float,
        dt_s: float,
        wrench_fresh: bool = True,
    ) -> OverrideOutput:
        """한 주기를 판정한다.

        Args:
            twist: 조작자 6 축 지령. z 는 **프로브 침투축** (양수 = 전진).
            force_n: 접촉력 [N]. 호출자가 정한 스칼라 (``‖F‖`` 또는 ``F_n``) 로,
                모드 스위치·조절기에 넣는 것과 **같은 값**이어야 한다.
            dt_s: 직전 호출로부터의 시간 [s].
            wrench_fresh: 힘 값을 믿을 수 있는가.

        Returns:
            이 주기의 결정.
        """
        out = [float(v) for v in twist]

        if not self.enabled:
            return OverrideOutput(out, PASS, False, "힘 한계 감시 꺼짐")

        if self.state in (RETREAT, HOLD):
            return self._continue_override(out, force_n, dt_s, wrench_fresh)

        if wrench_fresh and force_n >= self.limit_n:
            return self._enter_override(out, force_n)

        return self._pass(out, force_n, wrench_fresh)

    # -- 상태별 ------------------------------------------------------------

    def _pass(self, out: list, force_n: float, wrench_fresh: bool) -> OverrideOutput:
        """조작자에게 통과. 경고 이상이면 전진 성분만 뺀다."""
        self.state = PASS
        self.elapsed_s = 0.0

        if not wrench_fresh:
            # 힘을 모르면 배리어도 걸 수 없다. 지령은 그대로 보낸다 — 이 층이
            # wrench 두절로 로봇을 세우는 자리는 아니다.
            return OverrideOutput(out, PASS, False, "wrench 두절 — 힘 한계 감시 보류")

        if self.barrier_enabled and force_n >= self.warn_n and out[2] > 0.0:
            out[2] = 0.0
            return OverrideOutput(
                out, PASS, True,
                f"경고 {self.warn_n:.1f} N 이상 — 전진 금지 (지금 {force_n:.2f} N, "
                "빼는 방향은 그대로)",
            )

        return OverrideOutput(out, PASS, False, f"힘 {force_n:.2f} N — 조작자 통과")

    def _enter_override(self, out: list, force_n: float) -> OverrideOutput:
        """한계를 넘었다. 이번 주기부터 덮어쓴다."""
        self.state = RETREAT
        self.elapsed_s = 0.0
        return OverrideOutput(
            self._retreat_twist(out), RETREAT, True,
            f"⚠️ 한계 {self.limit_n:.1f} N 초과 ({force_n:.2f} N) — 조작자 지령을 덮고 "
            f"−z 로 {self.retreat_speed_m_s * 1000:.0f} mm/s 후퇴. "
            f"{self.resume_n:.1f} N 까지 내려오면 돌려준다",
        )

    def _continue_override(
        self, out: list, force_n: float, dt_s: float, wrench_fresh: bool
    ) -> OverrideOutput:
        """후퇴/정지를 이어 간다. 해제 조건은 힘, 안전장치는 시간이다."""
        self.elapsed_s += max(0.0, float(dt_s))

        if wrench_fresh and force_n <= self.resume_n:
            self.state = PASS
            self.elapsed_s = 0.0
            return OverrideOutput(
                out, PASS, False,
                f"힘 {force_n:.2f} N — 해제선 {self.resume_n:.1f} N 아래. 조작자에게 돌려준다",
            )

        if self.elapsed_s > self.max_override_s:
            # 물러났는데 힘이 안 떨어진다. 더 물러나는 것은 이 고장에 대한 답이
            # 아니다 (프로브가 걸렸거나, 힘이 프로브가 아닌 곳에서 온다). 선다.
            self.state = HOLD
            return OverrideOutput(
                [0.0] * 6, HOLD, True,
                f"후퇴 상한 {self.max_override_s:.1f} s 도달 — 정지한다. "
                + (f"힘이 안 떨어진다 ({force_n:.2f} N)" if wrench_fresh
                   else "wrench 도 두절이다") +
                ". 데드맨을 놓고 상황을 보라",
            )

        if not wrench_fresh:
            return OverrideOutput(
                self._retreat_twist(out), RETREAT, True,
                f"wrench 두절 — 후퇴를 이어 간다 ({self.elapsed_s:.1f}/"
                f"{self.max_override_s:.1f} s)",
            )

        return OverrideOutput(
            self._retreat_twist(out), RETREAT, True,
            f"한계 초과 후퇴 중 — {force_n:.2f} N → {self.resume_n:.1f} N "
            f"({self.elapsed_s:.1f}/{self.max_override_s:.1f} s)",
        )

    def _retreat_twist(self, out: list) -> list:
        """후퇴 한 주기. 조작자가 **더 빨리** 빼는 것은 통과시킨다.

        나머지 다섯 축은 0 이다: 한계를 넘은 힘으로 옆으로 끄는 것은 누르고 있는
        것보다 나을 것이 없다. 빼는 방향만은 조작자가 이 층보다 빠를 수 있어야
        한다 — 안전층이 조작자의 탈출을 늦추는 순간 그것은 안전층이 아니다.
        """
        return [0.0, 0.0, min(out[2], -self.retreat_speed_m_s), 0.0, 0.0, 0.0]
