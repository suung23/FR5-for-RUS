"""초음파 프로브용 단일팔 서보 노드 (DESIGN_NOTES Phase 0).

복강경 스택의 ``fr5_servo_joint_control_node`` 에서 온 것이지만 세 가지가 다르다.

1. **단일팔.** 듀얼암 인스턴스와 상대 자세 발행이 없다.
2. **그리퍼 없음.** 이전 노드는 그리퍼를 움직이려고 ``ServoMoveEnd`` → ``MoveGripper``
   → ``ServoMoveStart`` 를 했다. 접촉 중에 서보 모드가 끊기는 것은 그대로 사고다.
3. **wrench 발행.** F/T 를 상태 패키지에서 읽어 100 Hz 로 낸다.

워치독은 2층 구조의 **하위 층**이다 (DESIGN_NOTES §12.2). 관절 지령이 끊기면 정지한다.
후퇴하지 않는다 — 이 노드는 관절 공간에서 돌아 법선 방향을 모르고, 방향을 모르는 채
움직이는 것보다 서는 편이 안전하다. 후퇴는 프로브 프레임을 아는 ``us_diff_ik_node`` 몫이다.
"""
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Pose, TransformStamped, WrenchStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from fr5_control.robot_backend import make_backend

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class UsServoNode(Node):
    """관절 속도 지령을 적분해 ServoJ 로 흘리고, 상태와 wrench 를 발행한다."""

    def __init__(self) -> None:
        super().__init__("us_servo_node")

        self._declare_parameters()
        self.robot_name = self.get_parameter("robot.name").value
        robot_ip = self.get_parameter("robot.ip").value
        backend_kind = self.get_parameter("robot.backend").value

        self.max_joint_vel = float(self.get_parameter("safety.max_joint_vel_rad_s").value)
        self.limits_lower = list(self.get_parameter("safety.joint_limits_deg.lower").value)
        self.limits_upper = list(self.get_parameter("safety.joint_limits_deg.upper").value)
        self.cmd_timeout = float(self.get_parameter("watchdog.joint_cmd_timeout_s").value)
        self.retreat_force = float(self.get_parameter("watchdog.retreat_until_force_n").value)

        servo_hz = float(self.get_parameter("rates.servo_hz").value)
        wrench_hz = float(self.get_parameter("rates.wrench_publish_hz").value)
        status_hz = float(self.get_parameter("rates.status_publish_hz").value)
        self.cmd_t = 1.0 / servo_hz

        self._validate_limits()

        # -- 백엔드 ------------------------------------------------------
        self.backend = make_backend(
            backend_kind,
            robot_ip,
            contact_plane_z=float(self.get_parameter("mock.contact_plane_z").value),
            contact_stiffness=float(self.get_parameter("mock.contact_stiffness_n_m").value),
            ft_noise_std=float(self.get_parameter("mock.ft_noise_std_n").value),
            drop_wrench_after_s=float(self.get_parameter("mock.drop_wrench_after_s").value),
            freeze_state_after_s=float(self.get_parameter("mock.freeze_state_after_s").value),
        )
        self.backend.connect()
        if backend_kind == "mock":
            self.get_logger().warn(
                "MOCK 백엔드로 기동한다. 실로봇이 아니며 접촉력은 합성값이다."
            )
        self.get_logger().info(f"{self.robot_name} 연결 — {self.backend.name}")

        self.target_vel_rad = [0.0] * 6
        self.commanded_deg = list(self.backend.joint_positions_deg())
        self.home_deg = list(self.commanded_deg)
        self.cmd_id = 0
        self.get_logger().info(f"홈 자세 저장: {[round(v, 2) for v in self.home_deg]}")

        self.backend.servo_start()

        # -- 인터페이스 --------------------------------------------------
        ns = f"/{self.robot_name}"
        self.create_subscription(JointState, f"{ns}/joint_velocity_cmds", self._on_velocity, 10)

        self.joint_pub = self.create_publisher(JointState, f"{ns}/joint_states", 10)
        self.pose_pub = self.create_publisher(Pose, f"{ns}/ee_wrt_base", 10)
        self.wrench_pub = self.create_publisher(WrenchStamped, f"{ns}/wrench", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        now = self.get_clock().now()
        self.last_cmd_time = now
        self.last_loop_time = now
        self._stopped_by_watchdog = False

        self.create_timer(self.cmd_t, self._control_loop)
        self.create_timer(1.0 / wrench_hz, self._publish_wrench)
        self.create_timer(1.0 / status_hz, self._publish_status)

        self.get_logger().info(
            f"서보 {servo_hz:.0f} Hz · wrench {wrench_hz:.0f} Hz · "
            f"워치독 {self.cmd_timeout * 1000:.0f} ms · 최대 관절속도 {self.max_joint_vel} rad/s"
        )

    # -- 설정 ------------------------------------------------------------

    def _declare_parameters(self) -> None:
        """probe.yaml 의 항목을 선언한다. 기본값은 설정 누락을 드러내되 안전한 쪽으로."""
        self.declare_parameter("robot.name", "fr5_right")
        self.declare_parameter("robot.ip", "192.168.58.3")
        self.declare_parameter("robot.backend", "mock")

        self.declare_parameter("safety.max_joint_vel_rad_s", 0.5)
        self.declare_parameter("safety.joint_limits_deg.lower", [-175.0, -265.0, -160.0, -265.0, -175.0, -165.0])
        self.declare_parameter("safety.joint_limits_deg.upper", [175.0, 85.0, 160.0, 85.0, 175.0, 165.0])

        self.declare_parameter("watchdog.joint_cmd_timeout_s", 0.1)
        self.declare_parameter("watchdog.retreat_until_force_n", 0.2)

        self.declare_parameter("rates.servo_hz", 125.0)
        self.declare_parameter("rates.wrench_publish_hz", 100.0)
        self.declare_parameter("rates.status_publish_hz", 30.0)

        self.declare_parameter("ft_sensor.normal_force_sign", -1.0)

        self.declare_parameter("mock.contact_plane_z", 0.30)
        self.declare_parameter("mock.contact_stiffness_n_m", 2000.0)
        self.declare_parameter("mock.ft_noise_std_n", 0.02)
        self.declare_parameter("mock.drop_wrench_after_s", 0.0)
        self.declare_parameter("mock.freeze_state_after_s", 0.0)

    def _validate_limits(self) -> None:
        if len(self.limits_lower) != 6 or len(self.limits_upper) != 6:
            raise ValueError("safety.joint_limits_deg 는 상하한 각각 6개여야 한다")
        if any(lo >= hi for lo, hi in zip(self.limits_lower, self.limits_upper)):
            raise ValueError(f"관절 한계가 뒤집혀 있다: {self.limits_lower} / {self.limits_upper}")
        if self.max_joint_vel <= 0.0:
            raise ValueError("safety.max_joint_vel_rad_s 는 0보다 커야 한다")

    # -- 제어 ------------------------------------------------------------

    def _on_velocity(self, msg: JointState) -> None:
        if len(msg.velocity) < 6:
            return
        try:
            lag = (self.get_clock().now() - rclpy.time.Time.from_msg(msg.header.stamp)).nanoseconds / 1e9
            if lag > self.cmd_timeout:
                self.get_logger().warn(
                    f"관절 지령이 {lag * 1000:.0f} ms 늦어 버린다", throttle_duration_sec=1.0
                )
                return
        except (TypeError, ValueError):
            pass  # 타임스탬프 없는 발행자는 통과시킨다

        self.target_vel_rad = list(msg.velocity[:6])
        self.last_cmd_time = self.get_clock().now()

    def _control_loop(self) -> None:
        try:
            now = self.get_clock().now()
            dt = (now - self.last_loop_time).nanoseconds / 1e9
            self.last_loop_time = now
            # 타이머가 밀리거나 튀어도 한 스텝에 큰 각도가 실리지 않게 묶는다
            dt = min(max(dt, 0.002), 0.1)

            since_cmd = (now - self.last_cmd_time).nanoseconds / 1e9
            if since_cmd > self.cmd_timeout:
                # 하위 워치독: 정지. 후퇴는 프로브 프레임을 아는 상위 층 몫이다.
                if not self._stopped_by_watchdog and any(self.target_vel_rad):
                    self.get_logger().warn(
                        f"관절 지령 두절 {since_cmd * 1000:.0f} ms — 정지. "
                        f"후퇴는 us_diff_ik_node 가 수행한다."
                    )
                    self._stopped_by_watchdog = True
                self.target_vel_rad = [0.0] * 6
            else:
                self._stopped_by_watchdog = False

            for i in range(6):
                velocity = max(min(self.target_vel_rad[i], self.max_joint_vel), -self.max_joint_vel)
                candidate = self.commanded_deg[i] + math.degrees(velocity) * dt
                self.commanded_deg[i] = max(
                    min(candidate, self.limits_upper[i]), self.limits_lower[i]
                )

            self.cmd_id += 1
            error = self.backend.servo_j(self.commanded_deg, dt, self.cmd_id)
            if error != 0:
                self.get_logger().error(f"ServoJ 오류 {error}", throttle_duration_sec=1.0)
        except Exception as exc:
            self.get_logger().error(f"제어 루프 예외: {exc}", throttle_duration_sec=1.0)

    # -- 발행 ------------------------------------------------------------

    def _publish_wrench(self) -> None:
        """센서 프레임 wrench 를 그대로 낸다.

        Note:
            **아직 변환되지 않은 값이다.** 프로브 접촉면으로의 기준점 이동과 중력 보상은
            Phase 1 에서 들어간다 (DESIGN_NOTES §4.3, §2.3). 그 전까지 이 토픽의
            ``Mx, My`` 를 자세 정렬에 쓰면 안 된다 — 지렛대 항이 정렬 신호를 덮는다.
        """
        try:
            if not self.backend.wrench_valid():
                self.get_logger().warn("F/T 비활성 — wrench 발행 중단", throttle_duration_sec=2.0)
                return
            values = self.backend.wrench_raw()
            msg = WrenchStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f"{self.robot_name}_ft_sensor"
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = values[0:3]
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = values[3:6]
            self.wrench_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"wrench 읽기 실패: {exc}", throttle_duration_sec=2.0)

    def _publish_status(self) -> None:
        try:
            stamp = self.get_clock().now().to_msg()

            joint_msg = JointState()
            joint_msg.header.stamp = stamp
            joint_msg.name = JOINT_NAMES
            joint_msg.position = [math.radians(d) for d in self.backend.joint_positions_deg()]
            joint_msg.velocity = [math.radians(d) for d in self.backend.joint_velocities_deg_s()]
            joint_msg.effort = list(self.backend.joint_torques())
            self.joint_pub.publish(joint_msg)

            pose = self.backend.tool_pose()
            x, y, z = (v / 1000.0 for v in pose[0:3])
            quat = Rotation.from_euler("xyz", pose[3:6], degrees=True).as_quat()

            pose_msg = Pose()
            pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = x, y, z
            (
                pose_msg.orientation.x,
                pose_msg.orientation.y,
                pose_msg.orientation.z,
                pose_msg.orientation.w,
            ) = quat
            self.pose_pub.publish(pose_msg)

            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = f"{self.robot_name}_base_link"
            transform.child_frame_id = f"{self.robot_name}_tool"
            transform.transform.translation.x = x
            transform.transform.translation.y = y
            transform.transform.translation.z = z
            (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ) = quat
            self.tf_broadcaster.sendTransform(transform)
        except Exception as exc:
            self.get_logger().warn(f"상태 발행 실패: {exc}", throttle_duration_sec=2.0)

    # -- 종료 ------------------------------------------------------------

    def shutdown(self) -> None:
        """서보를 끊고, 접촉이 없을 때에만 홈으로 보낸다.

        이전 스택은 종료 시 무조건 ``MoveJ`` 로 초기 자세에 복귀했다. 프로브가 피부에
        닿아 있으면 그대로 긁으며 이동한다. 여기서는 접촉을 확인하고, 접촉 중이면
        홈잉을 **거부**한다 — 이 노드는 후퇴할 방향을 모르므로 스스로 풀 수 없다.
        """
        try:
            self.target_vel_rad = [0.0] * 6
            self.backend.servo_end()
        except Exception as exc:
            self.get_logger().error(f"서보 종료 실패: {exc}")

        try:
            sign = float(self.get_parameter("ft_sensor.normal_force_sign").value)
            normal_force = sign * self.backend.wrench_raw()[2] if self.backend.wrench_valid() else 0.0
            if normal_force > self.retreat_force:
                self.get_logger().error(
                    f"접촉 상태에서 종료 요청 (F_n = {normal_force:.2f} N > "
                    f"{self.retreat_force} N). 홈잉을 건너뛴다 — 프로브가 조직을 긁으며 "
                    f"이동한다. 먼저 후퇴시킨 뒤 다시 종료하라."
                )
            else:
                self.get_logger().info("홈 자세로 복귀 중...")
                if self.backend.move_j(self.home_deg, vel=15.0) != 0:
                    self.get_logger().error("MoveJ 실패")
        except Exception as exc:
            self.get_logger().error(f"종료 절차 예외: {exc}")
        finally:
            self.backend.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = UsServoNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
