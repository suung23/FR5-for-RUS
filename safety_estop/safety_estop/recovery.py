"""복귀 — 선 자리에서 끝내지 않고 **왔던 길을 되밟아** 초기 자세로 돌아간다.

왜 MoveJ 가 아닌가
-------------------
"초기 자세로 돌아가라" 를 홈 관절각으로의 MoveJ 로 구현하면, 로봇은 지금 자세와
홈을 잇는 **관절공간 직선** 을 간다. 그 직선이 지나는 카테시안 경로는 아무도 보지
않았고, 접촉 프로빙 중이라면 그 선의 출발점은 **복부 안쪽을 향해 기울어진 프로브**
다. 즉 가장 위험한 순간에 가장 검증되지 않은 경로를 쓰게 된다.

대신 감시를 켠 순간부터 지나온 관절 자세를 ``PathRecorder`` 가 빵부스러기로 남기고,
복귀는 그 부스러기를 **역순으로** 되밟는다. 결과로 두 가지가 공짜로 따라온다.

1. 경로가 이미 한 번 지나온 곳이다. 정적인 주변물에 대해서는 새로 부딪힐 데가 없다
   (⚠️ **움직이는 것에 대해서는 아니다** — 사람과 침대는 그 사이에 움직인다. 복귀도
   감시 아래에서 일어나야 하는 이유이고, 이 모듈이 복귀 중에도 ``TripMonitor`` 를
   계속 돌리는 이유다).
2. 되밟기의 **첫 구간이 곧 접촉 해제** 다. 들어온 길의 역방향이 빠지는 방향이므로,
   후퇴 방향을 따로 계산할 필요가 없다 — 프로브 법선을 알아야 하는 야코비안이
   이 패키지에 없어도 된다.

속도는 두 단계다. 접촉이 남아 있는 동안(힘 > ``release_force_n``)은
``release_speed_deg_s``, 떨어진 뒤에는 ``return_speed_deg_s``. 같은 궤적을 두 속도로
간다.

무엇을 보장하지 않는가
-----------------------
* **부스러기는 유한하다.** 오래 돌면 가장 오래된 것부터 버린다 (``max_points``).
  버린 구간은 되밟을 수 없으므로, 그때의 "초기 자세" 는 **남아 있는 가장 오래된
  점** 이다. 상태에 ``trail_truncated`` 로 내보내 콘솔이 그 사실을 말하게 한다.
* **이 모듈은 로봇을 움직이지 않는다.** 관절속도 지령을 만들 뿐이고, 그 지령도
  ``us_servo_node`` 의 ``JointCommandLimiter`` 를 통과한다 — 속도·가속도·관절한계·
  추종오차 밴드는 전부 거기서 다시 걸린다. 복귀 지령이 규격 밖으로 나갈 수 없는
  이유가 이것이고, 여기서 그것들을 다시 구현하지 않는 이유이기도 하다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = ["Phase", "RecoveryLimits", "PathRecorder", "ReturnPlanner"]

_DEG2RAD = math.pi / 180.0


class Phase(str, Enum):
    """정지·복귀의 단계. 문자열 값이 그대로 콘솔로 간다.

    ``DISARMED`` 와 ``SAFE`` 를 나눈 것이 중요하다. 둘 다 "지금 감시하지 않는다" 이지만
    전자는 **아직 켜지 않았다**, 후자는 **정지를 겪고 복귀를 마쳤다** 이다. 한 단어로
    합치면 전원 복구 뒤의 화면과 사고 뒤의 화면이 같아진다.
    """

    DISARMED = "disarmed"
    ARMED = "armed"
    TRIPPED = "tripped"
    RELEASING = "releasing"
    RETURNING = "returning"
    SAFE = "safe"


@dataclass(frozen=True)
class RecoveryLimits:
    """복귀 운동의 한계. 전부 조작자가 콘솔에서 볼 수 있어야 하는 값이다.

    Args:
        release_speed_deg_s: 접촉이 남아 있는 동안의 관절속도 상한 [°/s].
        return_speed_deg_s: 접촉이 떨어진 뒤의 관절속도 상한 [°/s].
        release_force_n: 이 아래로 떨어지면 접촉이 풀린 것으로 본다 [N].
        waypoint_tol_deg: 빵부스러기 하나를 "닿았다" 로 볼 거리 [°] (최대 성분 기준).
        gain_per_s: 목표점을 향한 P 게인 [1/s]. 속도 상한으로 잘린다.
        max_release_s: 접촉 해제 단계의 상한 [s]. 넘으면 선다 — 물러났는데 힘이 안
            떨어지면 더 물러나는 것은 그 고장에 대한 답이 아니다 (probe.yaml 의
            ``teleop_max_override_s`` 와 같은 논리).
        max_return_s: 복귀 전체의 상한 [s].
        stall_s: 이 시간 동안 관절이 ``stall_deg`` 만큼도 못 움직이면 선다 [s].
        stall_deg: 위의 거리 [°].
    """

    release_speed_deg_s: float = 2.0
    return_speed_deg_s: float = 8.0
    release_force_n: float = 0.5
    waypoint_tol_deg: float = 0.5
    gain_per_s: float = 1.5
    max_release_s: float = 10.0
    max_return_s: float = 120.0
    stall_s: float = 3.0
    stall_deg: float = 0.2

    def validated(self) -> "RecoveryLimits":
        if self.release_speed_deg_s <= 0 or self.return_speed_deg_s <= 0:
            raise ValueError("복귀 속도는 0 보다 커야 한다")
        if self.release_speed_deg_s > self.return_speed_deg_s:
            raise ValueError(
                "접촉 해제 속도가 복귀 속도보다 빠르다 — 닿아 있는 동안 더 느려야 한다")
        if self.waypoint_tol_deg <= 0 or self.gain_per_s <= 0:
            raise ValueError("허용오차와 게인은 0 보다 커야 한다")
        if self.max_release_s <= 0 or self.max_return_s <= 0:
            raise ValueError("시간 상한은 0 보다 커야 한다")
        return self


class PathRecorder:
    """지나온 관절 자세를 남긴다. 첫 점이 곧 "초기 자세" 다.

    거리 필터를 둔다 — 125 Hz 를 전부 남기면 1 분에 7500 점이고, 되밟을 때 그
    대부분은 앞 점과 구별되지 않는다. ``min_step_deg`` 이상 움직였을 때만 남긴다.
    """

    def __init__(self, min_step_deg: float = 1.0, max_points: int = 2000):
        if min_step_deg <= 0:
            raise ValueError("기록 간격은 0 보다 커야 한다")
        if max_points < 2:
            raise ValueError("최소 두 점은 들고 있어야 한다 (시작과 현재)")
        self.min_step_deg = float(min_step_deg)
        self.max_points = int(max_points)
        self._points: list[list[float]] = []
        self.truncated = False

    def reset(self, start_deg) -> None:
        """감시를 켜는 순간 호출한다. 여기가 돌아갈 자리다."""
        self._points = [[float(v) for v in start_deg[:6]]]
        self.truncated = False

    def record(self, joints_deg) -> None:
        if not self._points:
            self.reset(joints_deg)
            return
        last = self._points[-1]
        if max(abs(float(joints_deg[i]) - last[i]) for i in range(6)) < self.min_step_deg:
            return
        self._points.append([float(v) for v in joints_deg[:6]])
        if len(self._points) > self.max_points:
            # 가장 오래된 것부터 버린다. 버린 순간 "초기 자세" 의 뜻이 바뀌므로
            # 그 사실을 들고 있는다.
            self._points.pop(0)
            self.truncated = True

    @property
    def points(self) -> list[list[float]]:
        return self._points

    @property
    def home_deg(self) -> list[float] | None:
        """되밟아 도달할 자세. 부스러기가 잘렸다면 **원래 초기 자세가 아니다**."""
        return list(self._points[0]) if self._points else None

    def __len__(self) -> int:
        return len(self._points)


class ReturnPlanner:
    """부스러기를 역순으로 소비하며 관절속도 지령을 만든다.

    ``step`` 이 돌려주는 것은 ``(velocity_rad_s, done, reason)`` 이다. ``reason`` 이
    비어 있지 않으면 **중단** 이다 — 시간 상한, 정체, 또는 경로 소진. 중단은 실패가
    아니라 "여기서부터는 사람이 한다" 이고, 노드는 그 문장을 그대로 콘솔에 올린다.
    """

    def __init__(self, trail: list[list[float]], limits: RecoveryLimits, started_at: float):
        self.limits = limits.validated()
        # 역순 사본을 들고 소비한다. 원본(기록기)은 건드리지 않는다 — 복귀가
        # 중단되고 다시 시작될 수 있고, 그때 같은 길이 남아 있어야 한다.
        self._remaining = [list(p) for p in reversed(trail)]
        self.started_at = float(started_at)
        self.released_at: float | None = None
        self._last_progress_at = float(started_at)
        self._last_pose: list[float] | None = None
        self.in_contact = True

    @property
    def remaining(self) -> int:
        return len(self._remaining)

    def step(
        self,
        now: float,
        joints_deg,
        force_n: float | None,
    ) -> tuple[list[float], bool, str]:
        """한 주기. 관절속도 [rad/s] 6 개와 완료 여부, 중단 사유를 돌려준다."""
        q = [float(v) for v in joints_deg[:6]]

        # 접촉 판정. 힘을 모르면 **닿아 있다고 본다** — 모르는 쪽에서 더 느리게
        # 움직이는 것이 틀릴 때 덜 해롭다.
        if force_n is None or not math.isfinite(force_n):
            self.in_contact = True
        else:
            self.in_contact = abs(force_n) > self.limits.release_force_n
        if not self.in_contact and self.released_at is None:
            self.released_at = now

        if self.in_contact and (now - self.started_at) > self.limits.max_release_s:
            return self._halt(
                f"접촉 해제에 {self.limits.max_release_s:.0f} s 를 썼는데 힘이 "
                f"{self.limits.release_force_n:.1f} N 아래로 안 떨어진다 — 프로브가 "
                "걸렸거나 힘이 프로브가 아닌 곳에서 온다")
        if (now - self.started_at) > self.limits.max_return_s:
            return self._halt(f"복귀가 {self.limits.max_return_s:.0f} s 를 넘겼다")

        # 정체 감시. 지령은 나가는데 관절이 안 움직이면 복귀가 아니라 밀고 있는 것이다.
        if self._last_pose is None:
            self._last_pose = q
        elif max(abs(q[i] - self._last_pose[i]) for i in range(6)) >= self.limits.stall_deg:
            self._last_pose = q
            self._last_progress_at = now
        elif (now - self._last_progress_at) > self.limits.stall_s:
            return self._halt(
                f"{self.limits.stall_s:.0f} s 동안 {self.limits.stall_deg:.1f}° 도 못 움직였다 — "
                "무언가에 걸려 있다")

        # 닿은 부스러기는 버린다. 한 주기에 여러 개를 지나칠 수 있다 (빠르게
        # 움직였거나 부스러기가 촘촘한 구간).
        while self._remaining and self._max_err(q, self._remaining[0]) < self.limits.waypoint_tol_deg:
            self._remaining.pop(0)

        if not self._remaining:
            return [0.0] * 6, True, ""

        target = self._remaining[0]
        speed_cap = (
            self.limits.release_speed_deg_s if self.in_contact
            else self.limits.return_speed_deg_s
        )
        error = [target[i] - q[i] for i in range(6)]
        vel = [self.limits.gain_per_s * e for e in error]
        peak = max(abs(v) for v in vel)
        if peak > speed_cap:
            scale = speed_cap / peak
            vel = [v * scale for v in vel]
        return [v * _DEG2RAD for v in vel], False, ""

    def _halt(self, reason: str) -> tuple[list[float], bool, str]:
        return [0.0] * 6, False, reason

    @staticmethod
    def _max_err(a, b) -> float:
        return max(abs(a[i] - b[i]) for i in range(6))
