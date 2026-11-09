"""상태기계 — 정지 판정·물리 회로·복귀를 하나의 단계로 묶는다.

여기도 ROS 를 모른다. ``update`` 하나가 주기마다 불리고, 돌려주는 ``Decision`` 이
노드가 실제로 내보낼 전부다: 운동 금지 플래그, 복귀 관절속도, 단계, 그리고 사람에게
보여 줄 사건들.

단계 사이의 규칙 — 전부 **사람이 한 칸씩 민다**
------------------------------------------------
::

    DISARMED ──arm()──▶ ARMED ──(판정/물리회로)──▶ TRIPPED
                          ▲                            │
                          │                     begin_recovery()
                      reset()                          │
                          │                            ▼
                        SAFE ◀──경로 소진──  RELEASING ─▶ RETURNING
                                                   (힘이 떨어지면)

자동으로 일어나는 전이는 **하나뿐이다** — ARMED → TRIPPED. 나머지는 전부 콘솔의
명시적 조작이다. 특히:

* TRIPPED 에서 복귀로 **스스로 넘어가지 않는다.** 걸린 이유를 사람이 읽기 전에
  로봇이 움직이기 시작하면, 그 움직임이 원인을 덮는다.
* 복귀가 끝나도 SAFE 에서 멈춘다. 운동 금지는 **풀리지 않는다.** 복귀가 끝났다는
  것은 로봇이 안전한 자세에 있다는 뜻이지, 다시 쓸 준비가 됐다는 뜻이 아니다.
* ``reset()`` 은 DISARMED 로 간다. ARMED 가 아니다 — 리셋이 곧 재기동이 되면
  ISO 13850 이 금지하는 바로 그 거동이 된다. 다시 켜는 것은 또 한 번의 조작이다.

프로세스가 죽었다 살아나면 DISARMED 에서 시작한다. 이전 단계를 디스크에 남겨
이어받지 않는다 — 전원 복구 뒤에 이전 명령이 재개되지 않아야 한다는 요구가 바로
이 한 줄이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hardware import HardwareLoop, HardwareLoopState
from .recovery import Phase, PathRecorder, RecoveryLimits, ReturnPlanner
from .trip import SOURCE_OPERATOR, TripLimits, TripMonitor

__all__ = ["EstopSupervisor", "Decision"]


@dataclass
class Decision:
    """한 주기의 결론. 노드는 이것을 그대로 토픽으로 옮긴다."""

    phase: Phase
    #: 참이면 평범한 운동 지령(텔레옵·정책·IK)을 실행하지 않는다.
    inhibit: bool
    #: 복귀 중에만 채워진다 [rad/s] × 6. ``inhibit`` 이 참인 동안 **이 채널만**
    #: 로봇을 움직일 수 있다 — 권한이 한 곳에 있다는 것이 금지의 뜻이다.
    recovery_velocity: list[float] | None = None
    #: 로그·콘솔에 한 번씩만 올릴 사건들.
    events: list[str] = field(default_factory=list)


class EstopSupervisor:
    def __init__(
        self,
        trip_limits: TripLimits,
        recovery_limits: RecoveryLimits,
        joint_limits_deg: tuple[list[float], list[float]],
        *,
        trail_step_deg: float = 1.0,
        trail_max_points: int = 2000,
    ):
        self.monitor = TripMonitor(trip_limits, joint_limits_deg)
        self.recovery_limits = recovery_limits.validated()
        self.recorder = PathRecorder(trail_step_deg, trail_max_points)
        self.phase = Phase.DISARMED
        self.planner: ReturnPlanner | None = None
        #: 복귀가 중단된 사유. 비어 있지 않으면 콘솔이 그 문장을 띄운다.
        self.halt_reason = ""
        self.hardware: HardwareLoopState | None = None
        #: 직전 주기에 만든 복귀 지령. 복귀 단계 밖에서는 항상 None 이다.
        self._last_velocity: list[float] | None = None
        self._events: list[str] = []

    # -- 조작자 명령 -------------------------------------------------------

    def arm(self, now: float, *, force_at=None, joints_at=None, joints_deg=None) -> str | None:
        if self.phase not in (Phase.DISARMED, Phase.SAFE):
            return f"지금 단계는 {self.phase.value} 다 — ARM 은 해제 상태에서만 된다"
        hw = self.hardware
        reason = self.monitor.arm(
            now, force_at=force_at, joints_at=joints_at,
            hardware_asserted=bool(hw and hw.blocks_motion))
        if reason is not None:
            return reason
        if joints_deg is None:
            self.monitor.disarm()
            return "관절 상태가 없다 — 돌아갈 자리를 모른 채로 켤 수 없다"
        self.recorder.reset(joints_deg)
        self.phase = Phase.ARMED
        self._events.append(
            "감시 시작 — 지금 자세를 초기 자세로 삼는다: "
            + ", ".join(f"{v:+.1f}" for v in joints_deg[:6]))
        return None

    def disarm(self) -> str | None:
        if self.phase not in (Phase.ARMED, Phase.DISARMED):
            return f"지금 단계는 {self.phase.value} 다 — 걸려 있는 정지를 끄는 길은 없다"
        self.monitor.disarm()
        self.phase = Phase.DISARMED
        self._events.append("⚠️ 감시 해제 — 힘·관절 정지가 더 이상 걸리지 않는다")
        return None

    def stop(self, now: float, detail: str = "콘솔 STOP") -> None:
        """조작자가 콘솔에서 세운다. **비상정지가 아니다** — GUI·ROS·네트워크가
        살아 있어야 닿는 경로이고, 그래서 화면에서도 비상정지와 다른 이름으로 뜬다."""
        self.monitor.trip_now(SOURCE_OPERATOR, detail, now)
        if self.phase in (Phase.RELEASING, Phase.RETURNING):
            self.planner = None
            self.halt_reason = "조작자가 복귀를 중단했다"
        self.phase = Phase.TRIPPED
        self._events.append(f"정지 — {detail}")

    def begin_recovery(self, now: float) -> str | None:
        """복귀를 시작한다. 조작자의 명시적 확인 뒤에만 불린다.

        **이것은 운동 명령이다.** 거절 조건이 길고, 길어야 한다.
        """
        if self.phase != Phase.TRIPPED:
            return f"지금 단계는 {self.phase.value} 다 — 복귀는 TRIPPED 에서만 시작한다"
        latched = self.monitor.latched
        if latched is None:
            return "걸린 정지가 없다"
        hw = self.hardware
        if hw is not None and hw.blocks_motion:
            return hw.reason
        if not latched.allows_release:
            return (f"이 정지({latched.source})에서는 자동 복귀를 허용하지 않는다 — "
                    "로봇을 직접 풀고, 원인을 고친 뒤 RESET 하라")
        trail = self.recorder.points
        if len(trail) < 1:
            return "되밟을 경로가 없다 — 감시를 켠 뒤 움직인 기록이 없다"
        self.planner = ReturnPlanner(trail, self.recovery_limits, now)
        self.halt_reason = ""
        self.phase = Phase.RELEASING
        note = " ⚠️ 경로가 잘려 있다 — 돌아갈 자리는 원래 초기 자세가 아니다" if self.recorder.truncated else ""
        self._events.append(
            f"복귀 시작 — 지나온 {len(trail)} 점을 역순으로 되밟는다." + note)
        return None

    def abort_recovery(self, reason: str = "조작자 중단") -> str | None:
        if self.phase not in (Phase.RELEASING, Phase.RETURNING):
            return f"복귀 중이 아니다 (지금 {self.phase.value})"
        self.planner = None
        self.halt_reason = reason
        self.phase = Phase.TRIPPED
        self._events.append(f"복귀 중단 — {reason}")
        return None

    def reset(self, now: float, *, force_n=None, joints_deg=None) -> str | None:
        """래치를 지우고 DISARMED 로. 감시는 **켜지지 않는다**."""
        if self.phase in (Phase.RELEASING, Phase.RETURNING):
            return "복귀가 진행 중이다 — 끝내거나 중단한 뒤에 리셋한다"
        hw = self.hardware
        reason = self.monitor.clear(
            now, force_n=force_n, joints_deg=joints_deg,
            hardware_asserted=bool(hw and hw.asserted))
        if reason is not None:
            return reason
        self.monitor.disarm()
        self.planner = None
        self.halt_reason = ""
        self.phase = Phase.DISARMED
        self._events.append("리셋 — 해제 상태. 다시 쓰려면 ARM 을 한 번 더 누른다")
        return None

    def set_trip_limits(self, limits: TripLimits) -> str | None:
        try:
            self.monitor.set_limits(limits)
        except ValueError as exc:
            return str(exc)
        self._events.append(
            f"문턱 변경 — 접촉력 {limits.force_trip_n:.1f} N "
            f"(즉시 {limits.force_instant_n:.1f}) · 관절 {limits.joint_reach_trip * 100:.0f} %")
        return None

    def drain_events(self) -> list[str]:
        """쌓인 사건을 가져가고 비운다. 명령 처리 직후처럼 ``update`` 를 거치지
        않는 경로에서 노드가 로그에 옮기기 위한 것이다."""
        out = list(self._events)
        self._events.clear()
        return out

    # -- 주기 -------------------------------------------------------------

    def update(
        self,
        now: float,
        *,
        force_n: float | None,
        force_at: float | None,
        joints_deg: list[float] | None,
        joints_at: float | None,
        hardware: HardwareLoop,
    ) -> Decision:
        hw = hardware.evaluate(now)
        was_asserted = self.hardware.asserted if self.hardware else False
        self.hardware = hw
        if hw.asserted and not was_asserted:
            self._events.append(f"물리 회로 — {hw.reason}")

        previous = self.phase

        if self.phase in (Phase.ARMED, Phase.DISARMED):
            trip = self.monitor.update(
                now, force_n=force_n, force_at=force_at,
                joints_deg=joints_deg, joints_at=joints_at,
                hardware_asserted=hw.blocks_motion)
            if self.phase == Phase.ARMED and joints_deg is not None:
                self.recorder.record(joints_deg)
            if trip is not None:
                self.phase = Phase.TRIPPED
                if previous != Phase.TRIPPED:
                    self._events.append(f"🛑 정지 [{trip.source}] {trip.detail}")

        elif self.phase in (Phase.RELEASING, Phase.RETURNING):
            self._step_recovery(now, force_n, joints_deg, hw)

        elif self.phase == Phase.TRIPPED:
            # 걸린 뒤에도 값은 계속 본다 — 콘솔이 "힘이 떨어졌는가" 를 보여 줘야
            # 조작자가 복귀를 누를 때를 안다. 새 래치는 걸리지 않는다 (이미 있다).
            self.monitor.update(
                now, force_n=force_n, force_at=force_at,
                joints_deg=joints_deg, joints_at=joints_at,
                hardware_asserted=hw.blocks_motion)

        decision = Decision(
            phase=self.phase,
            inhibit=self.phase in (Phase.TRIPPED, Phase.RELEASING, Phase.RETURNING, Phase.SAFE)
            or hw.blocks_motion,
            events=list(self._events),
        )
        self._events.clear()
        if self.phase in (Phase.RELEASING, Phase.RETURNING) and self._last_velocity is not None:
            decision.recovery_velocity = self._last_velocity
        return decision

    def _step_recovery(self, now, force_n, joints_deg, hw: HardwareLoopState) -> None:
        self._last_velocity = None
        if hw.blocks_motion:
            self.abort_recovery(hw.reason)
            return
        if joints_deg is None:
            self.abort_recovery("관절 상태가 끊겼다 — 어디 있는지 모른 채로는 못 움직인다")
            return
        # 복귀 중에도 힘은 본다. 되밟기가 접촉을 **늘리는** 방향으로 가고 있다면
        # 그것은 경로가 틀렸다는 뜻이고, 더 가는 것이 답이 아니다.
        if (force_n is not None and abs(force_n) >= self.monitor.limits.force_instant_n):
            self.abort_recovery(
                f"복귀 중 접촉력이 {abs(force_n):.1f} N 으로 올랐다 — 빠지는 방향이 아니다")
            return
        assert self.planner is not None
        velocity, done, reason = self.planner.step(now, joints_deg, force_n)
        if reason:
            self.abort_recovery(reason)
            return
        if done:
            self.planner = None
            self.phase = Phase.SAFE
            self._events.append(
                "복귀 완료 — 초기 자세. 운동 금지는 아직 풀리지 않았다 (RESET 이 남아 있다)")
            return
        self.phase = Phase.RETURNING if not self.planner.in_contact else Phase.RELEASING
        self._last_velocity = velocity

    # -- 콘솔용 ------------------------------------------------------------

    def status(self) -> dict:
        trip = self.monitor.status()
        return {
            "phase": self.phase.value,
            "inhibit": self.phase in (
                Phase.TRIPPED, Phase.RELEASING, Phase.RETURNING, Phase.SAFE),
            "hardware": None if self.hardware is None else self.hardware.as_dict(),
            "haltReason": self.halt_reason,
            "trail": {
                "points": len(self.recorder),
                "truncated": self.recorder.truncated,
                "remaining": None if self.planner is None else self.planner.remaining,
                "homeDeg": self.recorder.home_deg,
            },
            "recovery": {
                "releaseSpeedDegS": self.recovery_limits.release_speed_deg_s,
                "returnSpeedDegS": self.recovery_limits.return_speed_deg_s,
                "releaseForceN": self.recovery_limits.release_force_n,
                "inContact": None if self.planner is None else self.planner.in_contact,
            },
            **trip,
        }
