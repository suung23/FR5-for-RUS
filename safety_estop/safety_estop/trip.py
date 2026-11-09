"""정지 판정 — 무엇이 걸렸는가, 그리고 그것이 어떤 복귀를 허용하는가.

ROS 를 모른다. 들어오는 것은 숫자와 시각뿐이고 나가는 것은 ``Trip`` 하나다.
떼어낸 이유는 ``joint_command_limiter`` 와 같다: 안전 판정의 버그는 전부 "시간이
지나야" 드러나는 종류(확인창, 래치, 해제 조건)라 실로봇 앞이 아니라 pytest 에서
찾아야 한다.

세 수준 중 **어디에 속하는가**
--------------------------------
이 모듈이 내는 것은 ISO 13850 의 비상정지가 **아니다**. 비상정지는 사람이 누르는
래칭형 물리 스위치이고, 그 회로는 소프트웨어를 통과하지 않는다 (``README.md`` §1).
여기서 자동으로 걸리는 것은 **보호정지** 다 — 힘 초과, 관절 한계 접근, 센서 정지.
``source == "hardware"`` 만이 물리 비상정지이며, 그때 이 모듈이 하는 일은 판정이
아니라 **관측** 이다. 컨트롤러가 이미 멈춘 것을 상태로 옮겨 적을 뿐이다.

구별을 코드에 남겨 두는 이유: 둘을 한 이름으로 부르면, 소프트웨어가 살아 있어야
작동하는 보호를 비상정지라고 믿게 된다. 그 믿음이 정확히 비상정지를 두지 않는
이유가 된다.

관절 여유를 어떻게 세는가
--------------------------
"관절 한계의 90 %" 를 ``|q| >= 0.9 * upper`` 로 읽으면 FR5 의 비대칭 축에서 뜻이
무너진다 — J2 는 [-265, 85] 라 ``0.9 * 85 = 76.5°`` 가 상한 쪽으로는 말이 되지만
하한 쪽(-265)에는 아무 의미가 없다. 그래서 **가동범위 중앙에서 한계까지 몇 %를
썼는가** 로 센다.

    reach = |q - center| / half_range,   center = (lo+hi)/2,  half = (hi-lo)/2

J2 라면 center = -90°, half = 175° 이므로 q = 0° 는 reach 0.51, q = +80° 는 0.97
이다. 어느 쪽 끝으로 가든 1.0 이 한계이고 0.9 가 문턱이다. 조작자에게는 남은
각도(``margin_deg``)로 함께 보여 준다 — 비율은 비교용, 각도는 판단용이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "TripLimits",
    "Trip",
    "TripMonitor",
    "JointReach",
    "joint_reach",
    "SOURCE_FORCE",
    "SOURCE_JOINT",
    "SOURCE_HARDWARE",
    "SOURCE_OPERATOR",
    "SOURCE_FORCE_STALE",
    "SOURCE_JOINT_STALE",
]

SOURCE_FORCE = "force"
SOURCE_JOINT = "joint"
SOURCE_HARDWARE = "hardware"
SOURCE_OPERATOR = "operator"
SOURCE_FORCE_STALE = "force_stale"
SOURCE_JOINT_STALE = "joint_stale"

#: 힘을 믿을 수 없을 때 자동 후퇴를 허용하는가. **허용하지 않는다** — 후퇴도
#: 운동이고, 운동의 멈출 조건을 그 못 믿을 힘이 쥐고 있다. 이 선택은 임의가 아니라
#: 위험분석이 돌려줘야 하는 답이며, 그때까지의 기본값은 "서고 사람을 부른다" 다.
_RELEASE_ALLOWED = {
    SOURCE_FORCE: True,        # 힘은 읽히고 있다 — 그 힘이 떨어지는 것을 보며 뺀다
    SOURCE_JOINT: True,        # 관절은 읽히고 있다 — 왔던 길을 되밟는다
    SOURCE_OPERATOR: True,     # 사람이 세웠다 — 사람이 복귀도 지시한다
    SOURCE_HARDWARE: False,    # 전원이 끊겼다. 소프트웨어 운동은 애초에 불가능하다
    SOURCE_FORCE_STALE: False,
    SOURCE_JOINT_STALE: False,
}


@dataclass(frozen=True)
class TripLimits:
    """조작자가 콘솔에서 고치는 값과, 고치지 못하는 천장.

    Args:
        force_trip_n: 접촉력 정지 문턱 [N]. ``force_confirm_s`` 동안 유지되면 건다.
        force_instant_n: 확인창 없이 즉시 거는 값 [N]. 한 표본이라도 넘으면 건다.
        force_confirm_s: 위 문턱의 확인창 [s]. 단발 스파이크로 서지 않기 위한 것이다.
        joint_reach_trip: 관절 여유 문턱 [0~1]. 0.90 = "중앙→한계의 90 % 를 썼다".
        joint_confirm_s: 관절 문턱의 확인창 [s].
        stale_s: 이보다 오래된 표본은 없는 것으로 본다 [s].
        trip_on_stale: 센서가 끊기면 정지를 걸 것인가. 끄면 **감시가 조용히 사라진다** —
            끌 수 있게 둔 것은 팬텀 벤치에서 센서 없이 관절만 보려는 경우 때문이고,
            그때도 상태에 ``monitoring: partial`` 로 남는다.
        force_ceiling_n: 콘솔이 올릴 수 있는 접촉력 문턱의 천장 [N]. 설정 파일에서만
            바뀐다 — ``force_limits.py`` 의 천장과 같은 이유다.
        joint_reach_ceiling: 관절 문턱의 천장. 1.0 은 "한계에 닿아야 선다" 라
            보호가 아니므로 그 아래로 묶는다.
    """

    force_trip_n: float = 8.0
    force_instant_n: float = 15.0
    force_confirm_s: float = 0.05
    joint_reach_trip: float = 0.90
    joint_confirm_s: float = 0.04
    stale_s: float = 0.25
    trip_on_stale: bool = True
    force_ceiling_n: float = 20.0
    joint_reach_ceiling: float = 0.98

    def validated(self) -> "TripLimits":
        """규칙에 걸리면 ``ValueError`` — 사유는 조작자에게 그대로 보여 줄 문장이다."""
        for name, value in (
            ("force_trip_n", self.force_trip_n),
            ("force_instant_n", self.force_instant_n),
            ("force_confirm_s", self.force_confirm_s),
            ("joint_reach_trip", self.joint_reach_trip),
            ("joint_confirm_s", self.joint_confirm_s),
            ("stale_s", self.stale_s),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} 이 유한한 값이어야 한다: {value!r}")
        if self.force_trip_n <= 0.0:
            raise ValueError("접촉력 문턱은 0 보다 커야 한다")
        if self.force_trip_n > self.force_ceiling_n + 1e-9:
            raise ValueError(
                f"접촉력 문턱 {self.force_trip_n:.1f} N 은 천장 "
                f"{self.force_ceiling_n:.1f} N 을 넘는다 — 천장은 콘솔이 아니라 "
                "estop.yaml 에서 올린다")
        if self.force_instant_n < self.force_trip_n:
            raise ValueError(
                f"즉시 정지값 {self.force_instant_n:.1f} N 이 확인창 문턱 "
                f"{self.force_trip_n:.1f} N 보다 낮다 — 확인창이 무의미해진다")
        if not (0.0 < self.joint_reach_trip <= self.joint_reach_ceiling + 1e-9):
            raise ValueError(
                f"관절 여유 문턱은 0 과 {self.joint_reach_ceiling:.2f} 사이여야 한다: "
                f"{self.joint_reach_trip}")
        if self.force_confirm_s < 0.0 or self.joint_confirm_s < 0.0:
            raise ValueError("확인창은 음수일 수 없다")
        if self.stale_s <= 0.0:
            raise ValueError("신선도 한계는 0 보다 커야 한다")
        return self


@dataclass(frozen=True)
class Trip:
    """걸린 정지 하나. 래치되며, 조작자가 사유를 읽기 전에는 지워지지 않는다."""

    source: str
    detail: str
    at: float
    value: float = float("nan")
    threshold: float = float("nan")

    @property
    def allows_release(self) -> bool:
        """이 사유에서 자동 접촉 해제(후퇴)를 허용해도 되는가.

        허용하지 않는다는 것은 "로봇이 그 자리에 선 채로 사람을 기다린다" 는 뜻이지
        안전하다는 뜻이 아니다. 닿은 채로 서 있는 것에도 위험이 있다 — 그래서 이
        값이 False 인 정지는 콘솔에 **사람이 해야 할 일** 과 함께 떠야 한다.
        """
        return _RELEASE_ALLOWED.get(self.source, False)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "detail": self.detail,
            "at": self.at,
            "value": None if math.isnan(self.value) else self.value,
            "threshold": None if math.isnan(self.threshold) else self.threshold,
            "allowsRelease": self.allows_release,
        }


@dataclass(frozen=True)
class JointReach:
    """관절 하나가 한계를 향해 얼마나 갔는가."""

    index: int
    position_deg: float
    reach: float
    margin_deg: float
    toward_upper: bool


def joint_reach(position_deg: float, lower_deg: float, upper_deg: float) -> JointReach:
    """``reach = |q - center| / half_range``. 모듈 docstring 의 정의 그대로."""
    center = 0.5 * (lower_deg + upper_deg)
    half = 0.5 * (upper_deg - lower_deg)
    if half <= 0.0:
        raise ValueError(f"관절 한계가 뒤집혀 있다: [{lower_deg}, {upper_deg}]")
    toward_upper = position_deg >= center
    bound = upper_deg if toward_upper else lower_deg
    return JointReach(
        index=-1,
        position_deg=float(position_deg),
        reach=abs(position_deg - center) / half,
        margin_deg=abs(bound - position_deg),
        toward_upper=toward_upper,
    )


class TripMonitor:
    """표본을 받아 정지를 건다. 한 번 걸리면 래치한다.

    래치가 핵심이다. 힘이 문턱을 스쳤다가 떨어지면 조건은 사라지지만 **일어난 일은
    사라지지 않는다**. 조건이 풀렸다고 감시가 스스로 복귀하면, 조작자는 로봇이 왜
    잠깐 멈췄는지 모른 채 계속 쓴다. 해제는 사람이 한다 (``clear``).
    """

    def __init__(self, limits: TripLimits, joint_limits_deg: tuple[list[float], list[float]]):
        self.limits = limits.validated()
        lower, upper = joint_limits_deg
        if len(lower) != 6 or len(upper) != 6:
            raise ValueError("관절 한계는 축마다 6 개씩이어야 한다")
        self.lower = [float(v) for v in lower]
        self.upper = [float(v) for v in upper]
        self._armed = False
        self._latched: Trip | None = None
        self._force_over_since: float | None = None
        self._joint_over_since: float | None = None
        #: 마지막으로 본 값들 — 콘솔이 "지금 어디까지 왔는가" 를 그린다.
        self.last_force_n: float | None = None
        self.last_reaches: list[JointReach] = []

    # -- 상태 -------------------------------------------------------------

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def latched(self) -> Trip | None:
        return self._latched

    def arm(
        self,
        now: float,
        *,
        force_at: float | None = None,
        joints_at: float | None = None,
        hardware_asserted: bool = False,
    ) -> str | None:
        """감시를 켠다. 켤 수 없으면 사유를 돌려준다 (켜지지 않는다).

        **보고 있지 않은 것은 지키지 못한다.** 그래서 켜는 조건에 표본의 신선도가
        들어간다. 이것이 없으면 센서가 아예 없는 상태에서도 ARMED 가 뜨고, 화면의
        그 한 단어가 있지도 않은 보호를 보증한다.
        """
        if self._latched is not None:
            return "래치된 정지가 남아 있다 — 사유를 확인하고 RESET 이 먼저다"
        if hardware_asserted:
            return "물리 비상정지가 눌려 있다"
        if joints_at is None or (now - joints_at) > self.limits.stale_s:
            return "관절 상태가 없다 — 제어 스택이 떠 있는가"
        if force_at is None or (now - force_at) > self.limits.stale_s:
            if self.limits.trip_on_stale:
                return "힘 표본이 없다 — 접촉력 감시 없이는 켤 수 없다"
        self._armed = True
        self._force_over_since = None
        self._joint_over_since = None
        return None

    def disarm(self) -> None:
        """감시를 끈다. **보호가 사라진다** — 노드는 이것을 경고로 남긴다."""
        self._armed = False
        self._force_over_since = None
        self._joint_over_since = None

    def set_limits(self, limits: TripLimits) -> None:
        """문턱을 바꾼다. 걸려 있는 동안에는 거절한다 — 걸린 이유를 지우는 가장
        쉬운 방법이 문턱을 올리는 것이어서는 안 된다."""
        if self._latched is not None:
            raise ValueError("정지가 걸려 있는 동안에는 문턱을 바꿀 수 없다")
        self.limits = limits.validated()
        self._force_over_since = None
        self._joint_over_since = None

    # -- 판정 -------------------------------------------------------------

    def trip_now(self, source: str, detail: str, now: float) -> Trip:
        """사람이나 상위 층이 직접 거는 정지 (콘솔 STOP, 하드웨어 관측)."""
        trip = Trip(source=source, detail=detail, at=now)
        if self._latched is None:
            self._latched = trip
        return self._latched

    def update(
        self,
        now: float,
        *,
        force_n: float | None = None,
        force_at: float | None = None,
        joints_deg: list[float] | None = None,
        joints_at: float | None = None,
        hardware_asserted: bool = False,
    ) -> Trip | None:
        """표본 한 묶음을 넣는다. 새로 걸렸으면 그 ``Trip``, 아니면 래치된 것.

        ``hardware_asserted`` 는 컨트롤러가 알려 준 물리 비상정지 상태다. 참이면
        **감시가 꺼져 있어도** 래치한다 — 이것은 판정이 아니라 기록이고, 전원이
        끊긴 사실을 콘솔이 모르고 있어서는 안 된다.
        """
        if hardware_asserted:
            return self.trip_now(
                SOURCE_HARDWARE,
                "물리 비상정지 회로가 열렸다 — 컨트롤러가 이미 멈춰 있다",
                now,
            )

        if not self._armed:
            # 감시가 꺼져 있어도 값은 따라간다. 콘솔이 "켜면 바로 걸릴 상태인가" 를
            # 보여 줘야 조작자가 켜기 전에 안다.
            self._observe(force_n, joints_deg)
            return self._latched

        self._observe(force_n, joints_deg)

        force_fresh = force_at is not None and (now - force_at) <= self.limits.stale_s
        joints_fresh = joints_at is not None and (now - joints_at) <= self.limits.stale_s

        # 한 번도 받은 적 없는 것(``*_at is None``)과 받다가 끊긴 것은 다른 고장이다.
        # 전자는 **기동 금지** 로 다룬다 (``can_arm``) — 켜지지 않은 감시를 정지로
        # 보고하면 콘솔에는 "로봇이 멈췄다" 만 뜨고 "센서가 아예 없다" 는 안 뜬다.
        if self.limits.trip_on_stale:
            if force_at is not None and not force_fresh:
                age = now - force_at
                return self.trip_now(
                    SOURCE_FORCE_STALE,
                    f"힘 표본이 {age * 1000:.0f} ms 오래됐다 — 힘 한계가 작동하지 않는다",
                    now,
                )
            if joints_at is not None and not joints_fresh:
                age = now - joints_at
                return self.trip_now(
                    SOURCE_JOINT_STALE,
                    f"관절 상태가 {age * 1000:.0f} ms 오래됐다 — 자세를 모른 채로 있다",
                    now,
                )

        if force_n is not None and force_fresh:
            trip = self._check_force(now, float(force_n))
            if trip is not None:
                return trip

        if joints_deg is not None and joints_fresh:
            trip = self._check_joints(now, joints_deg)
            if trip is not None:
                return trip

        return self._latched

    def _observe(self, force_n: float | None, joints_deg: list[float] | None) -> None:
        if force_n is not None and math.isfinite(force_n):
            self.last_force_n = float(force_n)
        if joints_deg is not None and len(joints_deg) >= 6:
            self.last_reaches = self._reaches(joints_deg)

    def _reaches(self, joints_deg) -> list[JointReach]:
        out = []
        for i in range(6):
            r = joint_reach(float(joints_deg[i]), self.lower[i], self.upper[i])
            out.append(JointReach(i, r.position_deg, r.reach, r.margin_deg, r.toward_upper))
        return out

    def _check_force(self, now: float, force_n: float) -> Trip | None:
        if not math.isfinite(force_n):
            # NaN 은 "모른다" 이지 "작다" 가 아니다. 신선도와 같은 취급을 한다.
            if self.limits.trip_on_stale:
                return self.trip_now(
                    SOURCE_FORCE_STALE, "힘 값이 NaN 이다 — 센서를 믿을 수 없다", now)
            return None

        magnitude = abs(force_n)
        if magnitude >= self.limits.force_instant_n:
            return self.trip_now_value(
                SOURCE_FORCE,
                f"접촉력 {magnitude:.2f} N — 즉시 정지값 "
                f"{self.limits.force_instant_n:.1f} N 을 넘었다",
                now, magnitude, self.limits.force_instant_n,
            )

        if magnitude >= self.limits.force_trip_n:
            if self._force_over_since is None:
                self._force_over_since = now
            held = now - self._force_over_since
            if held >= self.limits.force_confirm_s:
                return self.trip_now_value(
                    SOURCE_FORCE,
                    f"접촉력 {magnitude:.2f} N 이 문턱 {self.limits.force_trip_n:.1f} N 을 "
                    f"{held * 1000:.0f} ms 넘겼다",
                    now, magnitude, self.limits.force_trip_n,
                )
        else:
            self._force_over_since = None
        return None

    def _check_joints(self, now: float, joints_deg) -> Trip | None:
        worst: JointReach | None = None
        for r in self._reaches(joints_deg):
            if worst is None or r.reach > worst.reach:
                worst = r
        if worst is None:
            return None
        if worst.reach >= self.limits.joint_reach_trip:
            if self._joint_over_since is None:
                self._joint_over_since = now
            held = now - self._joint_over_since
            if held >= self.limits.joint_confirm_s:
                side = "상한" if worst.toward_upper else "하한"
                return self.trip_now_value(
                    SOURCE_JOINT,
                    f"J{worst.index + 1} 이 가동범위의 {worst.reach * 100:.0f} % 까지 갔다 "
                    f"({worst.position_deg:+.1f}°, {side}까지 {worst.margin_deg:.1f}° 남음)",
                    now, worst.reach, self.limits.joint_reach_trip,
                )
        else:
            self._joint_over_since = None
        return None

    def trip_now_value(self, source, detail, now, value, threshold) -> Trip:
        trip = Trip(source=source, detail=detail, at=now, value=value, threshold=threshold)
        if self._latched is None:
            self._latched = trip
        return self._latched

    # -- 해제 -------------------------------------------------------------

    def clear(
        self,
        now: float,
        *,
        force_n: float | None,
        joints_deg: list[float] | None,
        hardware_asserted: bool,
    ) -> str | None:
        """래치를 지운다. 조건이 아직 살아 있으면 거절하고 사유를 돌려준다.

        지운다고 감시가 켜지지는 않는다 — ``arm`` 이 따로 있다. 전원 복구나 노드
        재시작 뒤에 이전 상태가 이어지지 않게 하는 것이 이 두 단계의 목적이다.
        """
        if self._latched is None:
            return None
        if hardware_asserted:
            return "물리 비상정지가 아직 눌려 있다 — 버튼을 돌려 뽑고 컨트롤러를 리셋하라"
        if force_n is not None and math.isfinite(force_n):
            if abs(force_n) >= self.limits.force_trip_n:
                return (f"접촉력이 아직 {abs(force_n):.2f} N 이다 "
                        f"(문턱 {self.limits.force_trip_n:.1f} N) — 프로브가 닿아 있다")
        if joints_deg is not None and len(joints_deg) >= 6:
            for r in self._reaches(joints_deg):
                if r.reach >= self.limits.joint_reach_trip:
                    return (f"J{r.index + 1} 이 아직 가동범위의 {r.reach * 100:.0f} % 에 있다 — "
                            "복귀를 먼저 끝내라")
        self._latched = None
        self._force_over_since = None
        self._joint_over_since = None
        return None

    # -- 콘솔용 ------------------------------------------------------------

    def status(self) -> dict:
        """GUI 가 그릴 한 장. 값과 문턱을 함께 낸다 — 비율만 보내면 콘솔이 문턱을
        따로 들고 있어야 하고, 그 사본은 반드시 언젠가 어긋난다."""
        worst = max(self.last_reaches, key=lambda r: r.reach, default=None)
        return {
            "armed": self._armed,
            "latched": None if self._latched is None else self._latched.as_dict(),
            "limits": {
                "forceTripN": self.limits.force_trip_n,
                "forceInstantN": self.limits.force_instant_n,
                "forceCeilingN": self.limits.force_ceiling_n,
                "jointReachTrip": self.limits.joint_reach_trip,
                "jointReachCeiling": self.limits.joint_reach_ceiling,
                "tripOnStale": self.limits.trip_on_stale,
            },
            "forceN": self.last_force_n,
            "jointReach": [r.reach for r in self.last_reaches],
            "jointMarginDeg": [r.margin_deg for r in self.last_reaches],
            "worstJoint": None if worst is None else worst.index,
        }
