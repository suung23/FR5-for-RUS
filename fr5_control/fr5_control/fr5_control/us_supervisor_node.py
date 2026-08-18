"""감독 노드 — 상태 전이와 모드 발행 (DESIGN_NOTES §10).

    /us/quality_raw          Float32           Q_raw
    /us/valid_for_control    Bool
    /us/rejection_reasons    String            '|' 로 이어붙인 사유 코드
    /fr5_right/wrench        WrenchStamped
    /operator/engage         Bool              래치 아님 — 눌린 동안 True
    /operator/confirm        Bool
    /diag/force_search       Float32MultiArray [phase, target, F*, best_Q]
              │
              ▼
    /supervisor/state        String
    /supervisor/mode         String            admittance 가 구독

상태 판정 로직은 :mod:`fr5_control.supervisor` 에 있고 이 노드는 배선만 한다.

세 개의 개별 토픽(`quality_raw`, `valid_for_control`, `rejection_reasons`)은 임시다.
`us_interfaces` 가 생기면 한 메시지로 합친다 — 지금은 perception_node 자체가 없어
형식을 확정할 수 없다.
"""
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from fr5_control.supervisor import (
    State,
    Supervisor,
    SupervisorConfig,
    SupervisorInputs,
)

#: force_search 노드가 진단 배열의 0번에 싣는 phase 코드.
_SEARCH_PHASE_DONE = 3.0
_SEARCH_PHASE_FAILED = 4.0


class UsSupervisorNode(Node):
    """관측을 모아 상태 머신을 돌리고 모드를 발행한다."""

    def __init__(self) -> None:
        super().__init__("us_supervisor_node")

        self._declare_parameters()
        self.robot_name = self.get_parameter("robot.name").value
        rate_hz = float(self.get_parameter("supervisor.rate_hz").value)

        self.supervisor = Supervisor(
            SupervisorConfig(
                contact_force_n=float(self.get_parameter("supervisor.contact_force_n").value),
                contact_hold_s=float(self.get_parameter("supervisor.contact_hold_s").value),
                q_raw_enter_focus=float(self.get_parameter("supervisor.q_raw_enter_focus").value),
                valid_frames_enter_focus=int(
                    self.get_parameter("supervisor.valid_frames_enter_focus").value
                ),
                invalid_frames_leave_scan=int(
                    self.get_parameter("supervisor.invalid_frames_leave_scan").value
                ),
                q_raw_degrade_ratio=float(
                    self.get_parameter("supervisor.q_raw_degrade_ratio").value
                ),
                q_raw_degrade_hold_s=float(
                    self.get_parameter("supervisor.q_raw_degrade_hold_s").value
                ),
                max_recovery_attempts=int(
                    self.get_parameter("supervisor.max_recovery_attempts").value
                ),
                retreat_force_n=float(self.get_parameter("watchdog.retreat_until_force_n").value),
            )
        )
        self.normal_force_sign = float(self.get_parameter("ft_sensor.normal_force_sign").value)
        self.hard_force_limit = float(self.get_parameter("safety.max_normal_force_n").value)
        self.wrench_timeout = float(self.get_parameter("watchdog.wrench_timeout_s").value)

        # -- 관측 --------------------------------------------------------
        self.normal_force = 0.0
        self.wrench_stamp = None
        self.q_raw: float | None = None
        self.valid = False
        self.reasons: list[str] = []
        self.search_done = False
        self.search_failed = False
        self.engage = False
        self.confirm = False

        ns = f"/{self.robot_name}"
        self.create_subscription(WrenchStamped, f"{ns}/wrench", self._on_wrench, 10)
        self.create_subscription(Float32, "/us/quality_raw", self._on_quality, 10)
        self.create_subscription(Bool, "/us/valid_for_control", self._on_valid, 10)
        self.create_subscription(String, "/us/rejection_reasons", self._on_reasons, 10)
        self.create_subscription(Float32MultiArray, "/diag/force_search", self._on_search, 10)
        self.create_subscription(Bool, "/operator/engage", self._on_engage, 10)
        self.create_subscription(Bool, "/operator/confirm", self._on_confirm, 10)

        self.state_pub = self.create_publisher(String, "/supervisor/state", 10)
        self.mode_pub = self.create_publisher(String, "/supervisor/mode", 10)

        self.create_timer(1.0 / rate_hz, self._loop)
        self.get_logger().info(f"Supervisor {rate_hz:.0f} Hz · 시작 상태 {self.supervisor.state.value}")

    def _declare_parameters(self) -> None:
        self.declare_parameter("robot.name", "fr5_right")
        self.declare_parameter("supervisor.rate_hz", 20.0)
        self.declare_parameter("supervisor.contact_force_n", 0.5)
        self.declare_parameter("supervisor.contact_hold_s", 0.2)
        self.declare_parameter("supervisor.q_raw_enter_focus", 0.6)
        self.declare_parameter("supervisor.valid_frames_enter_focus", 10)
        self.declare_parameter("supervisor.invalid_frames_leave_scan", 15)
        self.declare_parameter("supervisor.q_raw_degrade_ratio", 0.25)
        self.declare_parameter("supervisor.q_raw_degrade_hold_s", 3.0)
        self.declare_parameter("supervisor.max_recovery_attempts", 2)
        self.declare_parameter("watchdog.retreat_until_force_n", 0.2)
        self.declare_parameter("watchdog.wrench_timeout_s", 0.05)
        self.declare_parameter("ft_sensor.normal_force_sign", -1.0)
        self.declare_parameter("safety.max_normal_force_n", 7.0)

    # -- 입력 ------------------------------------------------------------

    def _on_wrench(self, msg: WrenchStamped) -> None:
        # 프로브 프레임 변환은 admittance 가 한다. 여기서 필요한 것은 크기 판정뿐이라
        # 센서 z 를 그대로 쓴다 — 인라인 장착이면 같고, 아니면 보수적으로 크게 잡힌다.
        self.normal_force = self.normal_force_sign * msg.wrench.force.z
        self.wrench_stamp = self.get_clock().now()

    def _on_quality(self, msg: Float32) -> None:
        """NaN 은 "점수를 낼 수 없는 프레임"이다 — 0 이 아니다.

        perception_node 는 원시 품질을 낼 수 없을 때 NaN 을 보낸다. 이것을 0 으로
        받으면 열화 판정(`q_raw < 기준 x 0.75`)이 즉시 참이 되어 재탐색이 잘못 걸린다.
        """
        value = float(msg.data)
        self.q_raw = None if math.isnan(value) else value

    def _on_valid(self, msg: Bool) -> None:
        self.valid = bool(msg.data)

    def _on_reasons(self, msg: String) -> None:
        self.reasons = [r for r in msg.data.split("|") if r]

    def _on_search(self, msg: Float32MultiArray) -> None:
        if not msg.data:
            return
        phase = msg.data[0]
        self.search_done = phase == _SEARCH_PHASE_DONE
        self.search_failed = phase == _SEARCH_PHASE_FAILED

    def _on_engage(self, msg: Bool) -> None:
        self.engage = bool(msg.data)

    def _on_confirm(self, msg: Bool) -> None:
        self.confirm = bool(msg.data)

    # -- 루프 ------------------------------------------------------------

    def _safety_violation(self) -> str | None:
        """상태와 무관하게 즉시 RETREAT 시키는 조건."""
        if self.wrench_stamp is None:
            return None  # 아직 한 번도 안 왔다 — 기동 중일 수 있다
        age = (self.get_clock().now() - self.wrench_stamp).nanoseconds / 1e9
        if age > self.wrench_timeout:
            return f"F/T 두절 {age * 1000:.0f} ms"
        if self.normal_force > self.hard_force_limit:
            return f"접촉력 {self.normal_force:.2f} N > {self.hard_force_limit} N"
        return None

    def _loop(self) -> None:
        previous = self.supervisor.state
        rate = float(self.get_parameter("supervisor.rate_hz").value)

        self.supervisor.step(
            1.0 / rate,
            SupervisorInputs(
                normal_force=self.normal_force,
                q_raw=self.q_raw,
                valid_for_control=self.valid,
                rejection_reasons=list(self.reasons),
                search_done=self.search_done,
                search_failed=self.search_failed,
                operator_engage=self.engage,
                operator_confirm=self.confirm,
                safety_violation=self._safety_violation(),
            ),
        )

        if self.supervisor.state is not previous:
            self.get_logger().info(self.supervisor.last_transition or "전이")
            # 상태가 바뀌면 탐색 완료 플래그를 소비한 것으로 본다. 남겨두면 다음
            # 상태에서 즉시 다시 발동한다.
            self.search_done = False
            self.search_failed = False

        self.state_pub.publish(String(data=self.supervisor.state.value))
        self.mode_pub.publish(String(data=self.supervisor.mode.value))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = UsSupervisorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
