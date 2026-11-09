"""ROS 배선. 판정도 복귀도 여기 없다 — ``supervisor`` 를 주기마다 돌리고 결과를
토픽으로 옮긴다.

무엇을 듣고 무엇을 내는가
--------------------------
듣는 것 (전부 ``/{robot}`` 네임스페이스)::

    joint_states            sensor_msgs/JointState   us_servo_node
    wrench_px6d             geometry_msgs/WrenchStamped  telemetry_bridge (보상된 값)
    safety/hardware         std_msgs/String (JSON)   us_servo_node — 컨트롤러 안전입력
    safety/estop_cmd        std_msgs/String (JSON)   telemetry_bridge — 콘솔 명령

내는 것::

    safety/inhibit          std_msgs/Bool            래치·TRANSIENT_LOCAL
    safety/recovery_cmds    sensor_msgs/JointState   복귀 중에만, 관절속도 [rad/s]
    safety/estop_state      std_msgs/String (JSON)   래치·TRANSIENT_LOCAL

``inhibit`` 과 ``estop_state`` 를 래치(TRANSIENT_LOCAL)로 내는 것이 중요하다. 콘솔이나
제어 노드가 **늦게 붙어도** 현재 금지 상태를 즉시 받는다. volatile 로 내면 다음
전이까지 아무것도 못 받고, 전이가 없는 세션에서는 영영 못 받는다 — us_servo 의
``probing_mode`` 구독이 같은 함정을 다룬다.

명령의 답은 토픽으로 돌아간다
------------------------------
``estop_cmd`` 는 단방향이고, 거절 사유는 ``estop_state`` 의 ``lastCommand`` 에 실려
돌아간다. 콘솔은 **보낸 명령이 아니라 돌아온 상태** 를 근거로 화면을 그린다
(``ForceLimits`` 플레이트와 같은 규약: readback 이 진실, 입력은 요청).

⚠️ 이 노드가 죽으면 ``inhibit`` 은 래치된 마지막 값으로 남는다. 금지 중이었다면
금지가 유지된다 (안전한 쪽). 감시 중이었다면 **감시가 조용히 사라진다** — 그래서
``us_servo_node`` 가 이 노드의 심박을 따로 보고, 끊기면 선다.
"""

from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from .hardware import HardwareLoop
from .recovery import RecoveryLimits
from .supervisor import EstopSupervisor
from .trip import TripLimits

LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class EstopNode(Node):

    def __init__(self) -> None:
        super().__init__("safety_estop_node")
        self._declare()

        self.robot_name = str(self.get_parameter("robot.name").value)
        rate = float(self.get_parameter("estop.update_hz").value)

        trip = TripLimits(
            force_trip_n=float(self.get_parameter("estop.force_trip_n").value),
            force_instant_n=float(self.get_parameter("estop.force_instant_n").value),
            force_confirm_s=float(self.get_parameter("estop.force_confirm_s").value),
            joint_reach_trip=float(self.get_parameter("estop.joint_reach_trip").value),
            joint_confirm_s=float(self.get_parameter("estop.joint_confirm_s").value),
            stale_s=float(self.get_parameter("estop.stale_s").value),
            trip_on_stale=bool(self.get_parameter("estop.trip_on_stale").value),
            force_ceiling_n=float(self.get_parameter("estop.force_ceiling_n").value),
            joint_reach_ceiling=float(self.get_parameter("estop.joint_reach_ceiling").value),
        )
        recovery = RecoveryLimits(
            release_speed_deg_s=float(self.get_parameter("recovery.release_speed_deg_s").value),
            return_speed_deg_s=float(self.get_parameter("recovery.return_speed_deg_s").value),
            release_force_n=float(self.get_parameter("recovery.release_force_n").value),
            waypoint_tol_deg=float(self.get_parameter("recovery.waypoint_tol_deg").value),
            gain_per_s=float(self.get_parameter("recovery.gain_per_s").value),
            max_release_s=float(self.get_parameter("recovery.max_release_s").value),
            max_return_s=float(self.get_parameter("recovery.max_return_s").value),
            stall_s=float(self.get_parameter("recovery.stall_s").value),
            stall_deg=float(self.get_parameter("recovery.stall_deg").value),
        )
        joint_limits = (
            [float(v) for v in self.get_parameter("safety.joint_limits_deg.lower").value],
            [float(v) for v in self.get_parameter("safety.joint_limits_deg.upper").value],
        )

        try:
            self.sup = EstopSupervisor(
                trip, recovery, joint_limits,
                trail_step_deg=float(self.get_parameter("recovery.trail_step_deg").value),
                trail_max_points=int(self.get_parameter("recovery.trail_max_points").value),
            )
        except ValueError as exc:
            # 설정이 틀린 채로 뜨지 않는다. 올라와 있는 안전 노드는 보호를 주장한다.
            raise SystemExit(f"estop 설정이 규칙에 걸린다: {exc}") from exc

        self.hw_present = bool(self.get_parameter("hardware.present").value)
        if not self.hw_present:
            self.get_logger().warn(
                "⚠️ hardware.present = false — 이 스택에는 물리 비상정지가 없다. "
                "여기서 거는 것은 전부 **보호정지**이고, GUI·ROS·이 노드가 살아 "
                "있어야만 작동한다. 환자·사람 앞에서 쓸 구성이 아니다 (README §1)")

        self._joints: list[float] | None = None
        self._joints_at: float | None = None
        self._force_n: float | None = None
        self._force_at: float | None = None
        self._hw = HardwareLoop(self.hw_present, False, False, False, at=-1e9)
        self._last_command: dict | None = None
        self._published_inhibit: bool | None = None
        self._published_phase: str | None = None

        ns = f"/{self.robot_name}"
        self.create_subscription(JointState, f"{ns}/joint_states", self._on_joints, 10)
        self.create_subscription(
            WrenchStamped,
            f"{ns}/" + str(self.get_parameter("estop.wrench_topic").value),
            self._on_wrench, 10)
        self.create_subscription(String, f"{ns}/safety/hardware", self._on_hardware, LATCHED)
        self.create_subscription(String, f"{ns}/safety/estop_cmd", self._on_command, 10)

        self.inhibit_pub = self.create_publisher(Bool, f"{ns}/safety/inhibit", LATCHED)
        self.state_pub = self.create_publisher(String, f"{ns}/safety/estop_state", LATCHED)
        self.recovery_pub = self.create_publisher(
            JointState, f"{ns}/safety/recovery_cmds", 10)

        # 기동 순간의 상태를 **먼저** 낸다. 늦게 붙는 쪽이 아무것도 못 받은 채
        # "금지 없음" 으로 가정하는 구간을 없앤다.
        self._publish_inhibit(False, force=True)
        self._publish_state()

        self.create_timer(1.0 / rate, self._tick)
        self.create_timer(1.0 / float(self.get_parameter("estop.state_hz").value),
                          self._publish_state)
        self.get_logger().info(
            f"비상정지 감시 {rate:.0f} Hz · 접촉력 {trip.force_trip_n:.1f} N "
            f"(즉시 {trip.force_instant_n:.1f}) · 관절 {trip.joint_reach_trip * 100:.0f} % · "
            f"복귀 {recovery.release_speed_deg_s:.0f}→{recovery.return_speed_deg_s:.0f} °/s"
        )
        self.get_logger().info("단계: DISARMED — ARM 을 눌러야 감시가 시작된다")

    # -- 설정 -------------------------------------------------------------

    def _declare(self) -> None:
        self.declare_parameter("robot.name", "fr5_right")
        self.declare_parameter("estop.update_hz", 50.0)
        self.declare_parameter("estop.state_hz", 10.0)
        self.declare_parameter("estop.wrench_topic", "wrench_px6d")
        self.declare_parameter("estop.force_trip_n", 8.0)
        self.declare_parameter("estop.force_instant_n", 15.0)
        self.declare_parameter("estop.force_confirm_s", 0.05)
        self.declare_parameter("estop.joint_reach_trip", 0.90)
        self.declare_parameter("estop.joint_confirm_s", 0.04)
        self.declare_parameter("estop.stale_s", 0.25)
        self.declare_parameter("estop.trip_on_stale", True)
        self.declare_parameter("estop.force_ceiling_n", 20.0)
        self.declare_parameter("estop.joint_reach_ceiling", 0.98)
        self.declare_parameter("hardware.present", False)
        self.declare_parameter("recovery.release_speed_deg_s", 2.0)
        self.declare_parameter("recovery.return_speed_deg_s", 8.0)
        self.declare_parameter("recovery.release_force_n", 0.5)
        self.declare_parameter("recovery.waypoint_tol_deg", 0.5)
        self.declare_parameter("recovery.gain_per_s", 1.5)
        self.declare_parameter("recovery.max_release_s", 10.0)
        self.declare_parameter("recovery.max_return_s", 120.0)
        self.declare_parameter("recovery.stall_s", 3.0)
        self.declare_parameter("recovery.stall_deg", 0.2)
        self.declare_parameter("recovery.trail_step_deg", 1.0)
        self.declare_parameter("recovery.trail_max_points", 2000)
        self.declare_parameter(
            "safety.joint_limits_deg.lower", [-175.0, -265.0, -160.0, -265.0, -175.0, -165.0])
        self.declare_parameter(
            "safety.joint_limits_deg.upper", [175.0, 85.0, 160.0, 85.0, 175.0, 165.0])

    # -- 입력 -------------------------------------------------------------

    def _on_joints(self, msg: JointState) -> None:
        if len(msg.position) < 6:
            return
        # 토픽은 라디안이다. 감시와 한계는 전부 도로 센다 (probe.yaml 과 같은 단위).
        self._joints = [math.degrees(v) for v in msg.position[:6]]
        self._joints_at = time.time()

    def _on_wrench(self, msg: WrenchStamped) -> None:
        f = msg.wrench.force
        self._force_n = math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)
        self._force_at = time.time()

    def _on_hardware(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        self._hw = HardwareLoop(
            present=bool(data.get("present", self.hw_present)),
            estop=bool(data.get("estop", False)),
            stop0=bool(data.get("stop0", False)),
            stop1=bool(data.get("stop1", False)),
            at=time.time(),
        )

    # -- 명령 -------------------------------------------------------------

    def _on_command(self, msg: String) -> None:
        try:
            request = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("estop 명령이 JSON 이 아니다")
            return
        if not isinstance(request, dict):
            return
        command = str(request.get("command", ""))
        now = time.time()
        reason: str | None = None

        if command == "arm":
            reason = self.sup.arm(
                now, force_at=self._force_at, joints_at=self._joints_at,
                joints_deg=self._joints)
        elif command == "disarm":
            reason = self.sup.disarm()
        elif command == "stop":
            self.sup.stop(now, str(request.get("detail", "콘솔 STOP")))
        elif command == "recover":
            reason = self.sup.begin_recovery(now)
        elif command == "abort":
            reason = self.sup.abort_recovery(str(request.get("detail", "조작자 중단")))
        elif command == "reset":
            reason = self.sup.reset(now, force_n=self._force_n, joints_deg=self._joints)
        elif command == "limits":
            reason = self._set_limits(request)
        else:
            reason = f"모르는 명령: {command}"

        self._last_command = {
            "command": command,
            "ok": reason is None,
            "reason": reason or "",
            "at": now,
            "requestId": request.get("requestId"),
        }
        if reason is None:
            self.get_logger().info(f"estop 명령 [{command}] 수락")
        else:
            self.get_logger().warn(f"estop 명령 거절 [{command}]: {reason}")
        self._drain_events()
        self._publish_state()

    def _set_limits(self, request: dict) -> str | None:
        """콘솔이 고치는 문턱 둘 — 접촉력과 관절 여유. 나머지는 설정 파일의 것이다."""
        current = self.sup.monitor.limits
        try:
            force = float(request.get("forceTripN", current.force_trip_n))
            reach = float(request.get("jointReachTrip", current.joint_reach_trip))
            instant = float(request.get("forceInstantN", current.force_instant_n))
        except (TypeError, ValueError):
            return "문턱은 숫자여야 한다"
        return self.sup.set_trip_limits(
            TripLimits(
                force_trip_n=force,
                force_instant_n=instant,
                force_confirm_s=current.force_confirm_s,
                joint_reach_trip=reach,
                joint_confirm_s=current.joint_confirm_s,
                stale_s=current.stale_s,
                trip_on_stale=current.trip_on_stale,
                force_ceiling_n=current.force_ceiling_n,
                joint_reach_ceiling=current.joint_reach_ceiling,
            ))

    # -- 주기 -------------------------------------------------------------

    def _tick(self) -> None:
        now = time.time()
        decision = self.sup.update(
            now,
            force_n=self._force_n,
            force_at=self._force_at,
            joints_deg=self._joints,
            joints_at=self._joints_at,
            hardware=self._hw,
        )
        for event in decision.events:
            self.get_logger().warn(event)

        self._publish_inhibit(decision.inhibit)

        if decision.recovery_velocity is not None:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.velocity = [float(v) for v in decision.recovery_velocity]
            self.recovery_pub.publish(msg)

        if decision.phase.value != self._published_phase:
            self._published_phase = decision.phase.value
            self.get_logger().warn(f"단계 → {decision.phase.value.upper()}")
            self._publish_state()

    def _drain_events(self) -> None:
        for event in self.sup.drain_events():
            self.get_logger().warn(event)

    def _publish_inhibit(self, value: bool, force: bool = False) -> None:
        if not force and value == self._published_inhibit:
            return
        self._published_inhibit = value
        self.inhibit_pub.publish(Bool(data=bool(value)))
        self.get_logger().warn(
            "운동 금지 — 평범한 지령은 실행되지 않는다. 복귀 채널만 로봇을 움직인다"
            if value else "운동 금지 해제")

    def _publish_state(self) -> None:
        state = self.sup.status()
        state["robot"] = self.robot_name
        state["at"] = time.time()
        state["lastCommand"] = self._last_command
        self.state_pub.publish(String(data=json.dumps(state)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EstopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 내려가면서 금지를 **풀지 않는다**. 감시가 사라지는 것과 금지가 풀리는 것은
        # 다른 일이고, 둘을 같이 하면 노드를 죽이는 것이 정지를 지우는 방법이 된다.
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
