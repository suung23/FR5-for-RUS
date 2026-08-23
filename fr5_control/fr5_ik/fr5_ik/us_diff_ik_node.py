"""프로브 프레임 twist 를 관절속도로 옮기는 100 Hz 노드 (DESIGN_NOTES Phase 0).

복강경 스택의 ``freespace_two_twist`` 에서 왔지만 세 가지가 다르다.

1. **단일팔.** 듀얼암 상대 자세 발행이 없다.
2. **고정 주기 루프.** 이전 노드는 twist 콜백이 올 때만 돌았다. admittance 를 물리려면
   주기가 고정되어야 한다 (DESIGN_NOTES §11).
3. **워치독 상위 층.** twist 가 끊기면 프로브 −z 로 후퇴한다 (DESIGN_NOTES §12.2).
   후퇴는 카테시안 개념이라 관절 공간의 servo_node 가 할 수 없다 — 프로브 프레임을
   아는 것은 이 노드뿐이다.

기구학은 :mod:`fr5_ik.dls_solver` 가 담당하며, ``solve(V, q)`` 하나가 나중에 QP 로
갈아끼울 접합면이다.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, WrenchStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray

from fr5_ik.dls_solver import DlsSolver
from fr5_ik.teleop_frame import LinearFrameMapper

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

#: fr5_control.robot_backend 의 사본과 같은 링크 파라미터.
_FR5_LINKS = (
    (1.5708, 0.0, 0.0, 0.0, 0.0, 0.152),
    (0.0, 0.0, 0.0, -0.425, 0.0, 0.0),
    (0.0, 0.0, 0.0, -0.39501, 0.0, 0.0),
    (1.5708, 0.0, 0.0, 0.0, 0.0, 0.1021),
    (-1.5708, 0.0, 0.0, 0.0, 0.0, 0.102),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)


def build_chain(probe_xyz, probe_rpy):
    """FR5 6축 체인 끝에 프로브 고정 세그먼트를 붙인다."""
    import PyKDL as kdl

    chain = kdl.Chain()
    for index, (roll, pitch, yaw, x, y, z) in enumerate(_FR5_LINKS, start=1):
        chain.addSegment(
            kdl.Segment(
                f"link{index}",
                kdl.Joint(f"j{index}", kdl.Joint.RotZ),
                kdl.Frame(kdl.Rotation.RPY(roll, pitch, yaw), kdl.Vector(x, y, z)),
            )
        )
    chain.addSegment(
        kdl.Segment(
            "probe",
            kdl.Joint("probe_fixed", kdl.Joint.Fixed),
            kdl.Frame(
                kdl.Rotation.RPY(*probe_rpy),
                kdl.Vector(*probe_xyz),
            ),
        )
    )
    return chain


class UsDiffIkNode(Node):
    """100 Hz 미분 역기구학 + 추종오차 감시 + 후퇴 워치독."""

    def __init__(self) -> None:
        super().__init__("us_diff_ik_node")

        self._declare_parameters()
        self.robot_name = self.get_parameter("robot.name").value

        probe_xyz, probe_rpy = self._resolve_tool()

        self.rate_hz = float(self.get_parameter("rates.cartesian_hz").value)
        self.twist_timeout = float(self.get_parameter("watchdog.twist_timeout_s").value)
        self.twist_hold = min(
            float(self.get_parameter("watchdog.twist_hold_s").value), self.twist_timeout
        )
        self.joint_timeout = float(self.get_parameter("watchdog.joint_state_timeout_s").value)
        self.retreat_speed = float(self.get_parameter("watchdog.retreat_speed_m_s").value)
        self.retreat_force = float(self.get_parameter("watchdog.retreat_until_force_n").value)
        self.max_retreat = float(self.get_parameter("watchdog.max_retreat_s").value)

        self.max_linear = float(self.get_parameter("safety.max_linear_vel_m_s").value)
        self.max_angular = float(self.get_parameter("safety.max_angular_vel_rad_s").value)
        self.normal_sign = float(self.get_parameter("ft_sensor.normal_force_sign").value)

        # 병진 기준 프레임. "latched" 가 표류를 없애는 쪽, "probe" 가 예전 거동이다.
        self.linear_frame = str(self.get_parameter("teleop.linear_frame").value)
        self.engage_gap = float(self.get_parameter("teleop.engage_gap_s").value)
        self.mapper = LinearFrameMapper(
            tip_roll_deg=float(self.get_parameter("teleop.tip_roll_deg").value),
            relatch_on_engage=bool(self.get_parameter("teleop.relatch_on_engage").value),
        )

        self.solver = DlsSolver(
            lambda: build_chain(probe_xyz, probe_rpy),
            damping=float(self.get_parameter("ik.dls_damping").value),
        )
        self.linear_threshold = float(self.get_parameter("ik.tracking_error_linear_m_s").value)
        self.angular_threshold = float(self.get_parameter("ik.tracking_error_angular_rad_s").value)

        # -- 상태 -------------------------------------------------------
        self.q = None
        self.twist_cmd = np.zeros(6)
        self.normal_force = 0.0
        self.wrench_stamp = None
        self.last_twist_time = None
        self.last_joint_time = None
        self.retreat_started = None
        self._retreat_announced = False
        self.rot_stylus = None
        self.last_stylus_time = None
        self._warned_no_stylus = False

        # -- 인터페이스 --------------------------------------------------
        ns = f"/{self.robot_name}"
        self.create_subscription(JointState, f"{ns}/joint_states", self._on_joints, 10)
        self.create_subscription(Twist, f"{ns}/desired_twist", self._on_twist, 10)
        self.create_subscription(WrenchStamped, f"{ns}/wrench", self._on_wrench, 10)
        # 스타일러스 자세. teleop 이 데드맨을 잡고 있는 동안에만 발행하므로,
        # **발행이 끊겼다 다시 오는 것 자체가 재파지 신호**다 (별도 토픽이 필요 없다).
        side = str(self.robot_name).rsplit("_", 1)[-1]
        self.create_subscription(
            PoseStamped, f"/touch/{side}/stylus_pose", self._on_stylus, 10
        )

        self.velocity_pub = self.create_publisher(JointState, f"{ns}/joint_velocity_cmds", 10)
        self.error_pub = self.create_publisher(Float32MultiArray, "/diag/twist_tracking_error", 10)
        self.retreat_pub = self.create_publisher(Bool, "/diag/retreating", 10)

        self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info(
            f"미분 IK {self.rate_hz:.0f} Hz · DLS λ={self.solver.damping} · "
            f"twist ZOH 유지 {self.twist_hold * 1000:.0f} ms → 감쇠 → "
            f"워치독 {self.twist_timeout * 1000:.0f} ms → 프로브 −z 후퇴"
        )
        # 클램프를 기동 로그에 찍는다. 이 값이 안 보이면 freespace 인자를 빠뜨렸는지
        # 조작감만으로는 구분할 수 없다 — teleop 스케일을 아무리 올려도 여기서 잘리므로
        # "스케일이 안 먹는다" 로만 나타난다. 실제로 그 혼동이 있었다 (2026-08-19).
        self.get_logger().info(
            f"속도 클램프: 병진 {self.max_linear * 1000:.0f} mm/s · "
            f"회전 {self.max_angular:.2f} rad/s"
            + ("   ← 접촉용 한계다. 자유공간이면 freespace:=true 를 빠뜨린 것이다"
               if self.max_linear <= 0.02 else "")
        )
        # 병진 기준이 무엇인지 안 찍으면, 축이 어긋났을 때 원인을 조작감으로만
        # 판단하게 된다 — 2026-08-21 에 실제로 그렇게 시간을 썼다.
        self.get_logger().info(
            f"병진 기준 프레임: {self.linear_frame}"
            + ("  (데드맨 첫 파지에 고정 → 표류 없음)" if self.linear_frame == "latched"
               else "  ← 예전 거동이다. 손 자세가 돌면 축이 따라 돈다")
        )

    # -- 설정 ------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("robot.name", "fr5_right")

        self.declare_parameter("tool.j6_to_probe_xyz", [float("nan")] * 3)
        self.declare_parameter("tool.j6_to_probe_rpy", [float("nan")] * 3)
        self.declare_parameter("tool.allow_missing_tool", False)

        self.declare_parameter("rates.cartesian_hz", 100.0)
        self.declare_parameter("ik.dls_damping", 0.02)
        self.declare_parameter("ik.tracking_error_linear_m_s", 0.002)
        self.declare_parameter("ik.tracking_error_angular_rad_s", 0.05)

        self.declare_parameter("safety.max_linear_vel_m_s", 0.010)
        self.declare_parameter("safety.max_angular_vel_rad_s", 0.2)

        self.declare_parameter("watchdog.twist_timeout_s", 0.1)
        self.declare_parameter("watchdog.twist_hold_s", 0.04)
        self.declare_parameter("watchdog.joint_state_timeout_s", 0.2)
        self.declare_parameter("watchdog.retreat_speed_m_s", 0.005)
        self.declare_parameter("watchdog.retreat_until_force_n", 0.2)
        self.declare_parameter("watchdog.max_retreat_s", 3.0)

        self.declare_parameter("ft_sensor.normal_force_sign", -1.0)

        # teleop 노드와 **같은** probe.yaml 항목을 읽는다. 값이 갈라지면 병진이
        # 조용히 90° 틀어지므로, 여기서 따로 기본값을 만들지 않는다.
        self.declare_parameter("teleop.tip_roll_deg", 90.0)
        self.declare_parameter("teleop.linear_frame", "latched")
        self.declare_parameter("teleop.relatch_on_engage", False)
        self.declare_parameter("teleop.engage_gap_s", 0.3)

    def _resolve_tool(self):
        """프로브 변환을 읽는다. 미측정이면 기동을 거부한다.

        Raises:
            RuntimeError: CAD 값이 없는데 ``allow_missing_tool`` 이 꺼져 있을 때.
                0 으로 대충 채우면 조용히 틀린 기하로 돌아간다 — 그것이 가장 나쁘다.
        """
        xyz = list(self.get_parameter("tool.j6_to_probe_xyz").value)
        rpy = list(self.get_parameter("tool.j6_to_probe_rpy").value)
        missing = any(v is None or math.isnan(float(v)) for v in list(xyz) + list(rpy))

        if not missing:
            return [float(v) for v in xyz], [float(v) for v in rpy]

        if not bool(self.get_parameter("tool.allow_missing_tool").value):
            raise RuntimeError(
                "tool.j6_to_probe_* 가 아직 측정되지 않았다 (프로브 마운트 CAD 대기). "
                "Phase 0 의 자유공간 검증만 하려면 tool.allow_missing_tool 을 켜라. "
                "접촉 제어에는 켠 채로 쓰면 안 된다."
            )
        self.get_logger().warn(
            "프로브 변환 없음 — J6 플랜지를 툴 프레임으로 삼는다. "
            "자유공간 twist 추종 검증 전용이며 접촉 제어에는 기하가 틀린다."
        )
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    # -- 입력 ------------------------------------------------------------

    def _on_joints(self, msg: JointState) -> None:
        lookup = dict(zip(msg.name, msg.position))
        try:
            self.q = np.array([lookup[name] for name in JOINT_NAMES], dtype=float)
        except KeyError:
            if len(msg.position) >= 6:
                self.q = np.array(msg.position[:6], dtype=float)
            else:
                return
        self.last_joint_time = self.get_clock().now()

    def _on_twist(self, msg: Twist) -> None:
        self.twist_cmd = np.array(
            [
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z,
            ],
            dtype=float,
        )
        self.last_twist_time = self.get_clock().now()

    def _on_stylus(self, msg: PoseStamped) -> None:
        """스타일러스 자세. 발행이 끊겼다 다시 오면 재파지로 본다.

        ``touch_twist_node`` 는 데드맨을 잡고 있는 동안에만 이 토픽을 낸다. 그래서
        ``engage_gap_s`` 보다 긴 공백 뒤의 첫 메시지가 곧 "데드맨을 새로 잡았다" 이고,
        전용 토픽을 새로 만들 필요가 없다.
        """
        now = self.get_clock().now()
        orientation = msg.pose.orientation
        self.rot_stylus = Rotation.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_matrix()

        gap = (
            None
            if self.last_stylus_time is None
            else (now - self.last_stylus_time).nanoseconds / 1e9
        )
        self.last_stylus_time = now

        if self.q is None:
            return  # 관절각을 모르면 기준을 잡을 수 없다. 다음 메시지에서 다시 본다.
        if gap is not None and gap <= self.engage_gap and self.mapper.ready:
            return  # 파지가 이어지는 중

        if self.mapper.engage(self.solver.rotation_base_probe(self.q), self.rot_stylus):
            self.get_logger().info(
                f"병진 기준 프레임 고정 (파지 {self.mapper.engage_count}회차). "
                f"이 순간의 축이 세션 내내 유지된다 — "
                f"손 자세가 돌아도 '오른쪽'은 계속 같은 방향이다."
            )

    def _on_wrench(self, msg: WrenchStamped) -> None:
        self.normal_force = self.normal_sign * msg.wrench.force.z
        self.wrench_stamp = self.get_clock().now()

    # -- 제어 ------------------------------------------------------------

    def _clamp(self, twist: np.ndarray) -> np.ndarray:
        """병진·회전 크기를 각각 상한으로 묶는다. 방향은 보존한다."""
        out = np.array(twist, dtype=float)
        linear = np.linalg.norm(out[:3])
        if linear > self.max_linear:
            out[:3] *= self.max_linear / linear
        angular = np.linalg.norm(out[3:])
        if angular > self.max_angular:
            out[3:] *= self.max_angular / angular
        return out

    def _hold_fade(self, now) -> float:
        """직전 twist 를 아직 몇 % 로 믿을지. 오래된 지령일수록 0 에 가깝다.

        이 루프는 100 Hz 인데 Touch 는 50 Hz 로 발행한다. 매 주기 최신 twist 를
        그대로 다시 쓰는 것(ZOH)이 기본인데, 그대로 두면 발행이 끊긴 뒤에도
        **워치독 만료(100 ms)까지** 마지막 지령이 계속 재사용된다. 손을 멈춘 뒤에도
        최대 100 ms 분의 속도가 관절에 더 실린다는 뜻이고, 이것이 서보 단에서
        지령 선행분으로 쌓인다.

        그래서 한 발행 주기(``twist_hold_s``)까지만 그대로 믿고, 그 뒤로는 워치독
        만료까지 선형으로 0 으로 보낸다. 정상 조작(20 ms 간격)에서는 항상 1.0 이라
        조작감에 영향이 없고, 발행이 끊겼을 때만 동작한다.
        """
        age = (now - self.last_twist_time).nanoseconds / 1e9
        if age <= self.twist_hold:
            return 1.0
        span = self.twist_timeout - self.twist_hold
        if span <= 0.0:
            return 0.0
        return max(0.0, 1.0 - (age - self.twist_hold) / span)

    def _retreat_twist(self, now) -> np.ndarray | None:
        """후퇴 twist. 종료 조건에 도달했으면 ``None``.

        프로브 +z 가 조직 침투 방향이므로 후퇴는 −z 다. 종료 조건은 힘이지만,
        wrench 까지 함께 죽으면 무한히 물러나므로 시간 상한을 함께 둔다.
        """
        if self.retreat_started is None:
            self.retreat_started = now
            self._retreat_announced = False

        elapsed = (now - self.retreat_started).nanoseconds / 1e9
        wrench_fresh = (
            self.wrench_stamp is not None
            and (now - self.wrench_stamp).nanoseconds / 1e9 < 0.5
        )

        if wrench_fresh and self.normal_force < self.retreat_force:
            return None  # 접촉이 풀렸다
        if elapsed > self.max_retreat:
            self.get_logger().error(
                f"후퇴 시간 상한 {self.max_retreat} s 도달 — 정지한다. "
                f"{'힘이 안 떨어진다' if wrench_fresh else 'wrench 도 두절이다'}."
            )
            return None

        if not self._retreat_announced:
            self.get_logger().warn(
                f"twist 두절 — 프로브 −z 로 {self.retreat_speed * 1000:.0f} mm/s 후퇴"
            )
            self._retreat_announced = True
        return np.array([0.0, 0.0, -self.retreat_speed, 0.0, 0.0, 0.0])

    def _remap_linear(self, twist: np.ndarray) -> np.ndarray:
        """병진만 표류하지 않는 고정 프레임으로 옮긴다 (fr5_ik.teleop_frame 참조).

        회전은 손대지 않는다. "손목을 비틀면 프로브가 비틀린다" 는 대응은 조작자가
        프로브를 보면서 직접 확인·보정하는 축이고, 병진처럼 화면 밖으로 어긋나
        버리는 종류가 아니다.

        후퇴 twist 에는 적용하지 않는다 — 후퇴는 조작자 의도가 아니라 프로브 −z 라는
        기하학적 정의이므로 프로브 프레임 그대로여야 한다.
        """
        if self.linear_frame != "latched" or not self.mapper.ready:
            if self.linear_frame == "latched" and not self._warned_no_stylus:
                self.get_logger().warn(
                    "stylus_pose 가 아직 없어 병진을 예전(프로브 body) 기준으로 낸다. "
                    "데드맨을 한 번 잡으면 기준이 잡힌다. 계속 이 상태라면 "
                    "touch_twist 노드가 떠 있는지 확인하라."
                )
                self._warned_no_stylus = True
            return twist

        out = np.array(twist, dtype=float)
        out[:3] = self.mapper.to_probe(
            out[:3], self.solver.rotation_base_probe(self.q), self.rot_stylus
        )
        return out

    def _control_loop(self) -> None:
        now = self.get_clock().now()

        if self.q is None or self.last_joint_time is None:
            self.get_logger().warn("관절 상태 대기 중", throttle_duration_sec=2.0)
            return
        if (now - self.last_joint_time).nanoseconds / 1e9 > self.joint_timeout:
            self.get_logger().error("관절 상태 두절 — 지령 중단", throttle_duration_sec=1.0)
            return

        twist_stale = (
            self.last_twist_time is None
            or (now - self.last_twist_time).nanoseconds / 1e9 > self.twist_timeout
        )

        if twist_stale:
            twist = self._retreat_twist(now)
            if twist is None:
                self._publish_velocity(np.zeros(6))
                self.retreat_pub.publish(Bool(data=False))
                return
            self.retreat_pub.publish(Bool(data=True))
        else:
            if self.retreat_started is not None:
                self.get_logger().info("twist 복귀 — 후퇴 해제")
            self.retreat_started = None
            twist = self._clamp(self.twist_cmd) * self._hold_fade(now)
            twist = self._remap_linear(twist)
            self.retreat_pub.publish(Bool(data=False))

        try:
            joint_velocity, error = self.solver.solve_with_diagnostics(
                twist, self.q, self.linear_threshold, self.angular_threshold
            )
        except Exception as exc:
            self.get_logger().error(f"IK 실패: {exc}", throttle_duration_sec=1.0)
            self._publish_velocity(np.zeros(6))
            return

        if error.force_axes_degraded:
            self.get_logger().warn(
                f"힘 축 추종 열화 — e_z={error.per_axis[2]:.4f} m/s, "
                f"e_rx={error.per_axis[3]:.4f}, e_ry={error.per_axis[4]:.4f} rad/s. "
                f"특이점 근처일 수 있다 (QP 검토 근거).",
                throttle_duration_sec=2.0,
            )

        self._publish_velocity(joint_velocity)
        self.error_pub.publish(Float32MultiArray(data=error.as_list()))

    def _publish_velocity(self, joint_velocity: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.velocity = [float(v) for v in joint_velocity]
        self.velocity_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = UsDiffIkNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f"[us_diff_ik_node] 기동 거부: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
