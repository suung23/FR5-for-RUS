"""초음파 프로브용 단일팔 서보 노드 (DESIGN_NOTES Phase 0).

복강경 스택의 ``fr5_servo_joint_control_node`` 에서 온 것이지만 세 가지가 다르다.

1. **단일팔.** 듀얼암 인스턴스와 상대 자세 발행이 없다.
2. **그리퍼 없음.** 이전 노드는 그리퍼를 움직이려고 ``ServoMoveEnd`` → ``MoveGripper``
   → ``ServoMoveStart`` 를 했다. 접촉 중에 서보 모드가 끊기는 것은 그대로 사고다.
3. **wrench 발행.** F/T 를 상태 패키지에서 읽어 100 Hz 로 낸다.
4. **모드 인지** (2026-09-03). ``us_diff_ik`` 가 선언하는 ``probing_mode`` 를 구독한다.
   접촉 프로빙에서는 관절 지령이 힘 조절기에서 오고 목표에 붙을수록 작아지는데,
   teleop 용 idle 문턱(``IDLE_VEL_EPS_RAD_S``)이 그것을 전부 "정지 의도" 로 버렸다 —
   팬텀 실측에서 조절기가 18 초 동안 전진을 지령하는데 팁이 2 µm 만 움직인 원인.
   그래서 접촉 프로빙 중에는 크기 추측을 끄고 0 인지 아닌지만 본다
   (``JointCommandLimiter.step`` 의 ``explicit_intent``).

워치독은 2층 구조의 **하위 층**이다 (DESIGN_NOTES §12.2). 관절 지령이 끊기면 정지한다.
후퇴하지 않는다 — 이 노드는 관절 공간에서 돌아 법선 방향을 모르고, 방향을 모르는 채
움직이는 것보다 서는 편이 안전하다. 후퇴는 프로브 프레임을 아는 ``us_diff_ik_node`` 몫이다.
"""
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Pose, TransformStamped, WrenchStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from fr5_control.joint_command_limiter import JointCommandLimiter
from fr5_control.robot_backend import make_backend

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

#: fairino 컨트롤러 오류코드 중 이 노드가 실제로 마주치는 것들.
#: 숫자만 찍으면 매번 문서를 뒤져야 해서 뜻을 여기 붙여 둔다.
#: 출처: FAIRINO Error Code Comparison Table.
SERVOJ_ERRORS = {
    14: "Interface execution failed — 컨트롤러가 폴트 상태다 (보통 선행 오류의 후속 증상)",
    29: "ServoJ joint overrun — 관절 지령이 적정 범위를 벗어났다",
}

#: fairino 문서가 정한 ServoJ ``cmdT`` 권장 범위 [s].
#: 이전 코드는 dt 를 [0.002, 0.1] 로 묶어 그대로 cmdT 로 넘겼다. 상한 0.1 은
#: 규격의 6 배라, 루프가 한 번만 크게 밀려도 규격 밖 지령이 나간다.
CMD_T_MIN = 0.001
CMD_T_MAX = 0.016


class UsServoNode(Node):
    """관절 속도 지령을 적분해 ServoJ 로 흘리고, 상태와 wrench 를 발행한다."""

    def __init__(self) -> None:
        super().__init__("us_servo_node")

        self._declare_parameters()
        self.robot_name = self.get_parameter("robot.name").value
        robot_ip = self.get_parameter("robot.ip").value
        backend_kind = self.get_parameter("robot.backend").value

        self.max_joint_vel = float(self.get_parameter("safety.max_joint_vel_rad_s").value)
        self.max_joint_accel = float(self.get_parameter("safety.max_joint_accel_rad_s2").value)
        self.max_joint_decel = float(self.get_parameter("safety.max_joint_decel_rad_s2").value)
        self.max_follow_error = float(self.get_parameter("safety.max_follow_error_deg").value)
        self.limits_lower = list(self.get_parameter("safety.joint_limits_deg.lower").value)
        self.limits_upper = list(self.get_parameter("safety.joint_limits_deg.upper").value)
        self.resync_rate_deg_s = float(self.get_parameter("safety.idle_resync_rate_deg_s").value)
        self.resync_gain = float(self.get_parameter("safety.idle_resync_gain_per_s").value)
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
        self._servo_faulted = False

        # 지령 적분·속도 상한·정지 되감기는 전부 여기에 있다. 노드는 배선만 한다.
        self.limiter = JointCommandLimiter(
            start_deg=self.backend.joint_positions_deg(),
            max_vel=self.max_joint_vel,
            max_accel=self.max_joint_accel,
            max_decel=self.max_joint_decel,
            max_follow_error=self.max_follow_error,
            limits_lower=self.limits_lower,
            limits_upper=self.limits_upper,
            resync_rate_deg_s=self.resync_rate_deg_s,
            resync_gain=self.resync_gain,
        )
        self.home_deg = list(self.limiter.commanded_deg)
        self.cmd_id = 0
        self.get_logger().info(f"홈 자세 저장: {[round(v, 2) for v in self.home_deg]}")

        self.backend.servo_start()

        # -- 인터페이스 --------------------------------------------------
        ns = f"/{self.robot_name}"
        self.create_subscription(JointState, f"{ns}/joint_velocity_cmds", self._on_velocity, 10)

        #: us_diff_ik 가 선언한 모드가 접촉 프로빙인가. 참이면 idle 크기 추측을 끈다.
        self._contact_probing = False
        # 발행자(us_diff_ik)가 RELIABLE + TRANSIENT_LOCAL 로 래치해서 낸다. 같은
        # 내구성으로 구독해야 늦게 붙어도 현재 모드를 즉시 받는다 — volatile 로
        # 구독하면 다음 전환까지 아무것도 못 받고, 전환이 없는 세션에서는 영영
        # 못 받는다 (capture.py 가 2026-09-02 에 같은 함정을 밟았다).
        self.create_subscription(
            String,
            f"{ns}/probing_mode",
            self._on_probing_mode,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

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
        # 정지 거동은 가속 상한이 아니라 이 두 값이 정한다. teleop 이 "멈춰도 더 간다"
        # 로 느껴질 때 먼저 볼 곳이므로 기동 로그에 남긴다.
        self.get_logger().info(
            f"가속 {self.max_joint_accel} / 감속 {self.max_joint_decel} rad/s² · "
            f"정지 되감기 {self.limiter.resync_rate_deg_s:.0f} °/s "
            f"(게인 {self.resync_gain:.0f}/s, 추종오차 밴드 ±{self.max_follow_error}°)"
        )

    # -- 설정 ------------------------------------------------------------

    def _declare_parameters(self) -> None:
        """probe.yaml 의 항목을 선언한다. 기본값은 설정 누락을 드러내되 안전한 쪽으로."""
        self.declare_parameter("robot.name", "fr5_right")
        self.declare_parameter("robot.ip", "192.168.58.3")
        self.declare_parameter("robot.backend", "mock")

        self.declare_parameter("safety.max_joint_vel_rad_s", 0.5)
        self.declare_parameter("safety.max_joint_accel_rad_s2", 8.0)
        self.declare_parameter("safety.max_joint_decel_rad_s2", 24.0)
        self.declare_parameter("safety.idle_resync_rate_deg_s", 90.0)
        self.declare_parameter("safety.idle_resync_gain_per_s", 12.0)
        self.declare_parameter("safety.max_follow_error_deg", 5.0)
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
        if self.max_joint_accel <= 0.0:
            raise ValueError("safety.max_joint_accel_rad_s2 는 0보다 커야 한다")
        if self.max_joint_decel < self.max_joint_accel:
            raise ValueError(
                "safety.max_joint_decel_rad_s2 는 가속 상한 이상이어야 한다 "
                f"({self.max_joint_decel} < {self.max_joint_accel}). 감속을 가속보다 "
                "느리게 잡으면 정지가 가속보다 오래 걸린다 — 지금 고치려는 증상 그대로다."
            )
        if self.resync_gain <= 0.0 or self.resync_rate_deg_s <= 0.0:
            raise ValueError("safety.idle_resync_* 는 0보다 커야 한다 (0 이면 되감기가 꺼진다)")
        if self.max_follow_error <= 0.0:
            raise ValueError("safety.max_follow_error_deg 는 0보다 커야 한다")

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

    def _on_probing_mode(self, msg: String) -> None:
        """us_diff_ik 가 선언한 모드를 받는다.

        "contact_probing" 으로 시작하는 값(순수 접촉 프로빙과 면내 회전 하위 모드 —
        fr5_ik.probing_mode 참조)이면 접촉 프로빙이다. 이 모드에서 관절 지령의
        출처는 힘 조절기이고, 조절기는 밴드 안에서 정확히 0 을 내므로 크기로 정지
        의도를 추측할 이유가 없다 — 추측하면 수렴 중의 미소 지령이 전부 버려진다.
        """
        probing = msg.data.startswith("contact_probing")
        if probing != self._contact_probing:
            self.get_logger().info(
                "접촉 프로빙 진입 — 미소 관절 지령을 그대로 실행한다 "
                "(idle 크기 추측 꺼짐, 정지 판정은 지령 == 0 일 때만)"
                if probing
                else "접촉 프로빙 이탈 — teleop idle 문턱 복원"
            )
        self._contact_probing = probing

    def _control_loop(self) -> None:
        try:
            now = self.get_clock().now()
            dt = (now - self.last_loop_time).nanoseconds / 1e9
            self.last_loop_time = now
            # 타이머가 밀리거나 튀어도 한 스텝에 큰 각도가 실리지 않게 묶는다.
            # 상한은 fairino 의 cmdT 권장 상한이다 — 이 값이 그대로 cmdT 로 나가므로
            # 규격 밖으로 나가면 컨트롤러가 지령을 거부한다 (오류 29).
            dt = min(max(dt, CMD_T_MIN), CMD_T_MAX)

            since_cmd = (now - self.last_cmd_time).nanoseconds / 1e9
            if since_cmd > self.cmd_timeout:
                # 하위 워치독: 정지. 후퇴는 프로브 프레임을 아는 상위 층 몫이다.
                if not self._stopped_by_watchdog and any(self.target_vel_rad):
                    self.get_logger().warn(
                        f"관절 지령 두절 {since_cmd * 1000:.0f} ms — 정지. "
                        f"남은 지령 선행분은 되감아 버린다. "
                        f"후퇴는 us_diff_ik_node 가 수행한다."
                    )
                    self._stopped_by_watchdog = True
                self.target_vel_rad = [0.0] * 6
            else:
                self._stopped_by_watchdog = False

            # 폴트가 한 번 걸리면 지령을 계속 밀어넣지 않는다. 컨트롤러는 어차피 전부
            # 거부하고, 그 사이에도 commanded_deg 는 적분을 계속해 실제와의 간격이
            # 무한히 벌어진다 — 그 상태로 서보가 살아나면 그것이 곧 튀는 동작이다.
            if self._servo_faulted:
                self.target_vel_rad = [0.0] * 6
                self.limiter.hard_reset(self.backend.joint_positions_deg())
                return

            actual_deg = self.backend.joint_positions_deg()
            was_idle = self.limiter.idle

            # 속도 상한 · 가속/감속 상한 · 관절 한계 · 추종오차 밴드, 그리고 정지 중
            # 지령 되감기가 전부 여기서 일어난다 (joint_command_limiter 참조).
            # 접촉 프로빙에서는 힘 조절기의 미소 지령이 의도이므로 크기 추측을 끈다.
            commanded_deg = self.limiter.step(
                self.target_vel_rad, actual_deg, dt,
                explicit_intent=self._contact_probing,
            )

            # 정지로 넘어가는 순간의 선행분을 기록한다. 되감기가 없던 시절에는
            # 이만큼이 손을 멈춘 뒤에 그대로 실행됐고, 그것이 곧 "멈춰도 더 간다" 였다.
            # 이 값이 추종오차 밴드에 붙어 있으면 밴드가 포화한 것이다 — 그 구간에서
            # 조작이 뻣뻣해진다. 밴드를 줄이거나 IK 출력을 줄여야 한다는 신호다.
            if self.limiter.idle and not was_idle and self.limiter.lead_at_stop_deg > 0.5:
                self.get_logger().info(
                    f"정지 — 지령 선행분 {self.limiter.lead_at_stop_deg:.2f}° 되감는다"
                    + (
                        "  ← 추종오차 밴드 포화"
                        if self.limiter.lead_at_stop_deg > 0.9 * self.max_follow_error
                        else ""
                    ),
                    throttle_duration_sec=2.0,
                )

            self.cmd_id += 1
            error = self.backend.servo_j(commanded_deg, dt, self.cmd_id)
            if error != 0:
                self._servo_faulted = True
                meaning = SERVOJ_ERRORS.get(error, "미상 — FAIRINO Error Code Table 참조")
                self.get_logger().error(
                    f"ServoJ 오류 {error}: {meaning}. 서보 지령을 중단한다. "
                    f"관절={[round(v, 2) for v in self.limiter.commanded_deg]} "
                    f"속도={[round(v, 3) for v in self.limiter.applied_vel_rad]} "
                    f"cmdT={dt * 1000:.1f} ms"
                )
                self.get_logger().error(
                    "복구는 자동으로 하지 않는다 — 폴트 시점의 손 자세 그대로 서보가 "
                    "살아나면 그것이 곧 급발진이다. 노드를 내렸다 다시 띄워라."
                )
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
                # 폴트 상태에서는 MoveJ 도 14 로 떨어진다. 지우고 나서 보낸다.
                if self._servo_faulted:
                    reset = self.backend.reset_error()
                    self.get_logger().info(f"컨트롤러 폴트 해제 시도 — ResetAllError={reset}")
                self.get_logger().info("홈 자세로 복귀 중...")
                move_error = self.backend.move_j(self.home_deg, vel=15.0)
                if move_error != 0:
                    meaning = SERVOJ_ERRORS.get(move_error, "미상")
                    self.get_logger().error(f"MoveJ 실패 {move_error}: {meaning}")
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
