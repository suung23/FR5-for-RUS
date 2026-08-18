"""상태 머신과 실패 복구 (DESIGN_NOTES §10).

실전에서 가장 중요한 부분이다. 핵심은 **실패 종류마다 다른 곳으로 돌아간다**는 것 —
접촉 불량은 Stage 1a 로, 방광 소실은 Stage 1b 로 간다. Slim U-Net 과 원시 품질기가
machine-readable 한 거부 사유를 주므로 이유별 매핑이 가능하다.

## 모드 — admittance 가 유일한 twist 발행자다

teleop 과 admittance 가 둘 다 ``desired_twist`` 를 발행하면 다툰다. 그리고
``TELEOP_APPROACH`` 에서 조작자는 z 를 포함한 6축이 필요한데 z 는 admittance 소유다.

그래서 **admittance 를 유일한 발행자로 두고 모드로 동작을 바꾼다**:

    IDLE     zero twist
    TELEOP   조작자 6축을 그대로 통과 (힘 루프 정지)
    CONTACT  힘 3축 + 영상 3축 (정상 동작)
    RETREAT  프로브 −z 로 후퇴

mux 경쟁이 사라지고 발행자가 하나로 유지된다.

ROS 에 의존하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "State",
    "Mode",
    "SupervisorConfig",
    "SupervisorInputs",
    "Supervisor",
    "STATE_TO_MODE",
    "REASON_TO_RECOVERY",
]


class State(Enum):
    """감독기 상태 (DESIGN_NOTES §10)."""

    IDLE = "IDLE"
    TELEOP_APPROACH = "TELEOP_APPROACH"
    STAGE1A_CONTACT = "STAGE1A_CONTACT"
    STAGE1B_FOCUS = "STAGE1B_FOCUS"
    STAGE2_SCAN = "STAGE2_SCAN"
    RETREAT = "RETREAT"


class Mode(Enum):
    """admittance 에 내리는 동작 모드."""

    IDLE = "idle"
    TELEOP = "teleop"
    CONTACT = "contact"
    RETREAT = "retreat"


#: 상태 → 모드. 상태가 늘어나도 admittance 는 이 네 가지만 알면 된다.
STATE_TO_MODE: dict[State, Mode] = {
    State.IDLE: Mode.IDLE,
    State.TELEOP_APPROACH: Mode.TELEOP,
    State.STAGE1A_CONTACT: Mode.CONTACT,
    State.STAGE1B_FOCUS: Mode.CONTACT,
    State.STAGE2_SCAN: Mode.CONTACT,
    State.RETREAT: Mode.RETREAT,
}

#: 거부 사유 → 복구 목적지 (DESIGN_NOTES §10.2, §10.3).
#: 값이 ``None`` 이면 상태 전이 없이 policy 목표에만 반영한다.
REASON_TO_RECOVERY: dict[str, State | None] = {
    # -- 세그멘테이션 계열 --------------------------------------------
    "empty_mask": State.STAGE1B_FOCUS,
    "mask_area_too_small": State.STAGE1B_FOCUS,
    "mask_area_too_large": State.STAGE1B_FOCUS,
    "fragmented_mask": State.STAGE1B_FOCUS,
    "low_segmentation_confidence": State.STAGE1A_CONTACT,
    "high_boundary_entropy": State.STAGE1A_CONTACT,
    "excessive_border_contact": None,          # 면내 문제 — policy 몫
    "low_temporal_warped_iou": None,
    "excessive_centroid_jump": None,
    # -- 원시 접촉 계열 ------------------------------------------------
    "no_contact": State.TELEOP_APPROACH,
    "poor_acoustic_coupling": State.STAGE1A_CONTACT,
    "low_near_field_echo": State.STAGE1A_CONTACT,
    # 힘으로 해결되지 않는 유일한 원시 실패. 힘을 올리면 오히려 악화된다.
    "excessive_shadowing": None,
}


@dataclass
class SupervisorConfig:
    """전이 임계 (DESIGN_NOTES §10.1). 전부 🟡 제안값이다."""

    contact_force_n: float = 0.5          # 접촉으로 인정할 최소 힘
    contact_hold_s: float = 0.2
    q_raw_enter_focus: float = 0.6        # STAGE1A → 1B
    valid_frames_enter_focus: int = 10
    invalid_frames_leave_scan: int = 15   # ≈0.5 s @ 30 Hz
    q_raw_degrade_ratio: float = 0.25     # 재탐색 트리거
    q_raw_degrade_hold_s: float = 3.0
    max_recovery_attempts: int = 2        # 초과하면 TELEOP 로 반환
    retreat_force_n: float = 0.2          # 이 아래면 후퇴 완료

    def __post_init__(self) -> None:
        if self.contact_force_n <= 0 or self.retreat_force_n <= 0:
            raise ValueError("힘 임계는 양수여야 한다")
        if self.retreat_force_n >= self.contact_force_n:
            raise ValueError(
                "retreat_force_n 은 contact_force_n 보다 작아야 한다. "
                "아니면 후퇴 직후 다시 접촉으로 판정되어 진동한다."
            )
        if not 0.0 < self.q_raw_degrade_ratio < 1.0:
            raise ValueError("q_raw_degrade_ratio 는 (0, 1) 이어야 한다")


@dataclass
class SupervisorInputs:
    """한 틱의 관측. 없는 값은 ``None`` 으로 둔다."""

    normal_force: float = 0.0
    q_raw: float | None = None
    valid_for_control: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    search_done: bool = False
    search_failed: bool = False
    operator_engage: bool = False
    operator_confirm: bool = False
    #: 안전 위반 사유. 값이 있으면 어느 상태에서든 즉시 RETREAT.
    safety_violation: str | None = None


class Supervisor:
    """관측을 상태로, 상태를 모드로 바꾼다.

    시간은 밖에서 ``dt`` 로 주입한다.
    """

    def __init__(self, config: SupervisorConfig | None = None) -> None:
        self.config = config or SupervisorConfig()
        self.state = State.IDLE
        self._contact_timer = 0.0
        self._valid_streak = 0
        self._invalid_streak = 0
        self._degrade_timer = 0.0
        self._recovery_attempts = 0
        self._q_raw_reference: float | None = None
        self._last_transition: str | None = None

    # -- 조회 -----------------------------------------------------------

    @property
    def mode(self) -> Mode:
        return STATE_TO_MODE[self.state]

    @property
    def last_transition(self) -> str | None:
        """직전 전이의 사유. 로그·진단용이며 읽고 나면 비워지지 않는다."""
        return self._last_transition

    @property
    def q_raw_reference(self) -> float | None:
        """Stage 1 종료 시점의 ``Q_raw``. 열화 판정의 기준선."""
        return self._q_raw_reference

    # -- 진행 -----------------------------------------------------------

    def step(self, dt: float, inputs: SupervisorInputs) -> State:
        """한 틱 진행하고 새 상태를 돌려준다."""
        if inputs.safety_violation and self.state is not State.RETREAT:
            self._go(State.RETREAT, f"안전 위반: {inputs.safety_violation}")
            return self.state

        handler = {
            State.IDLE: self._idle,
            State.TELEOP_APPROACH: self._teleop,
            State.STAGE1A_CONTACT: self._stage1a,
            State.STAGE1B_FOCUS: self._stage1b,
            State.STAGE2_SCAN: self._stage2,
            State.RETREAT: self._retreat,
        }[self.state]
        handler(dt, inputs)
        return self.state

    def request_recovery(self, reasons: list[str]) -> State | None:
        """거부 사유 목록에서 복구 목적지를 고른다.

        여러 사유가 동시에 뜨면 **가장 앞 단계로** 되돌린다. 접촉이 나쁜데 방광만
        다시 찾으려 해봐야 소용없기 때문이다.
        """
        order = [State.TELEOP_APPROACH, State.STAGE1A_CONTACT, State.STAGE1B_FOCUS]
        targets = [
            REASON_TO_RECOVERY[r]
            for r in reasons
            if REASON_TO_RECOVERY.get(r) is not None
        ]
        if not targets:
            return None
        return min(targets, key=order.index)

    # -- 상태별 처리 -----------------------------------------------------

    def _go(self, state: State, reason: str) -> None:
        if state is self.state:
            return
        self._last_transition = f"{self.state.value} → {state.value} ({reason})"
        self.state = state
        self._contact_timer = 0.0
        self._valid_streak = 0
        self._invalid_streak = 0
        self._degrade_timer = 0.0

    def _idle(self, dt: float, inputs: SupervisorInputs) -> None:
        if inputs.operator_engage:
            self._recovery_attempts = 0
            self._go(State.TELEOP_APPROACH, "조작자 engage")

    def _teleop(self, dt: float, inputs: SupervisorInputs) -> None:
        cfg = self.config
        if inputs.normal_force > cfg.contact_force_n:
            self._contact_timer += dt
        else:
            self._contact_timer = 0.0

        # 접촉 감지만으로는 넘어가지 않는다. 조작자가 확인해야 자동 제어가 시작된다.
        if self._contact_timer >= cfg.contact_hold_s and inputs.operator_confirm:
            self._recovery_attempts = 0
            self._go(State.STAGE1A_CONTACT, "접촉 감지 + 조작자 확인")

    def _stage1a(self, dt: float, inputs: SupervisorInputs) -> None:
        cfg = self.config
        if inputs.search_failed:
            self._escalate("Stage 1a 탐색 실패")
            return

        if inputs.valid_for_control:
            self._valid_streak += 1
        else:
            self._valid_streak = 0

        quality_ok = inputs.q_raw is not None and inputs.q_raw >= cfg.q_raw_enter_focus
        if quality_ok and self._valid_streak >= cfg.valid_frames_enter_focus:
            self._go(State.STAGE1B_FOCUS, f"Q_raw≥{cfg.q_raw_enter_focus} & valid×{self._valid_streak}")

    def _stage1b(self, dt: float, inputs: SupervisorInputs) -> None:
        if inputs.search_failed:
            self._escalate("Stage 1b 탐색 실패")
            return
        if inputs.search_done:
            self._q_raw_reference = inputs.q_raw
            self._recovery_attempts = 0
            self._go(State.STAGE2_SCAN, "F* 수렴")

    def _stage2(self, dt: float, inputs: SupervisorInputs) -> None:
        cfg = self.config

        if inputs.valid_for_control:
            self._invalid_streak = 0
        else:
            self._invalid_streak += 1
            if self._invalid_streak >= cfg.invalid_frames_leave_scan:
                target = self.request_recovery(inputs.rejection_reasons) or State.STAGE1B_FOCUS
                self._recovery_attempts += 1
                if self._recovery_attempts > cfg.max_recovery_attempts:
                    self._escalate("복구 반복 실패")
                else:
                    self._go(target, f"valid=False ×{self._invalid_streak}")
                return

        # Q_raw 열화 — 접촉 문제이므로 힘 재탐색으로 돌아간다 (§5.3 진단 분기).
        if self._q_raw_reference and inputs.q_raw is not None:
            degraded = inputs.q_raw < self._q_raw_reference * (1.0 - cfg.q_raw_degrade_ratio)
            self._degrade_timer = self._degrade_timer + dt if degraded else 0.0
            if self._degrade_timer >= cfg.q_raw_degrade_hold_s:
                self._recovery_attempts += 1
                if self._recovery_attempts > cfg.max_recovery_attempts:
                    self._escalate("Q_raw 열화 반복")
                else:
                    self._go(State.STAGE1B_FOCUS, "Q_raw 열화 지속")

    def _retreat(self, dt: float, inputs: SupervisorInputs) -> None:
        # 안전 위반으로 들어왔더라도, 힘이 빠지고 조작자가 확인해야 나간다.
        if inputs.normal_force < self.config.retreat_force_n and inputs.operator_confirm:
            self._go(State.TELEOP_APPROACH, "후퇴 완료 + 조작자 확인")

    def _escalate(self, reason: str) -> None:
        """자동 복구를 포기하고 조작자에게 돌려준다."""
        self._recovery_attempts = 0
        self._go(State.TELEOP_APPROACH, reason)
