"""100 Hz admittance 노드 — 힘 축과 영상 축을 합쳐 프로브 twist 를 낸다 (DESIGN_NOTES §8).

    /fr5_right/wrench          WrenchStamped  센서 프레임 (us_servo_node)
    /control/contact_setpoint  WrenchStamped  force.z=F_n*, torque.x=M*_x, torque.y=M*_y
    /control/image_twist       Twist          linear.x, linear.y, angular.z
    /control/teleop_twist      Twist          조작자 6축 (TELEOP 모드에서만)
    /supervisor/mode           String          idle | teleop | contact | retreat
              │
              ▼
    /fr5_right/desired_twist   Twist          프로브 프레임 6-DoF, 100 Hz

## 이 노드가 유일한 twist 발행자다

teleop 과 admittance 가 둘 다 ``desired_twist`` 를 발행하면 다툰다. 그리고
``TELEOP_APPROACH`` 에서 조작자는 z 를 포함한 6축이 필요한데 z 는 힘 루프 소유다.
그래서 조작자 twist 를 이 노드가 받아 **모드에 따라 통과시킨다**:

    idle      영(zero) twist
    teleop    조작자 6축 그대로 통과, 힘 루프 정지
    contact   힘 3축 + 영상 3축 (정상 동작)
    retreat   프로브 −z 로 후퇴

`contact_setpoint` 에 ``WrenchStamped`` 를 쓰는 이유: 세 값이 **한 묶음으로 한 시각에**
나와야 하는데 ``Float32`` 셋으로 쪼개면 동기가 깨진다. 의미도 정확히 들어맞는다 —
목표 접촉 wrench 다. 사용하지 않는 성분(force.x/y, torque.z)은 무시한다.

**wrench 가 끊기면 이 노드는 발행을 멈춥니다 — 0 을 내지 않습니다.**
직관에 반하지만 의도된 설계다. 발행이 멈추면 한 층 아래 ``us_diff_ik_node`` 의 twist
워치독(0.1 s)이 걸려 프로브 −z 후퇴가 시작된다 (§12.2 2층 구조). 0 을 내면 diff_ik 는
"정상적으로 정지 지령을 받았다"고 보고 접촉 상태 그대로 멈춰 선다.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, WrenchStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from fr5_control.admittance import (
    AdmittanceConfig,
    ContactSetpoint,
    SlewLimiter,
    compose_probe_twist,
)
from fr5_control.transforms import probe_from_sensor, wrench_sensor_to_probe


class UsAdmittanceNode(Node):
    """접촉 wrench 를 속도 지령으로 바꾸고 영상 축과 합성한다."""

    def __init__(self) -> None:
        super().__init__("us_admittance_node")

        self._declare_parameters()
        self.robot_name = self.get_parameter("robot.name").value
        self.rate_hz = float(self.get_parameter("rates.cartesian_hz").value)

        self.rotation, self.offset = self._resolve_frames()

        self.config = AdmittanceConfig(
            damping_linear=float(self.get_parameter("admittance.damping_linear_ns_m").value),
            damping_angular=float(self.get_parameter("admittance.damping_angular_nms_rad").value),
            max_linear=float(self.get_parameter("safety.max_linear_vel_m_s").value),
            max_angular=float(self.get_parameter("safety.max_angular_vel_rad_s").value),
            force_deadband=float(self.get_parameter("ft_sensor.force_deadband_n").value),
            moment_deadband=float(self.get_parameter("ft_sensor.moment_deadband_nm").value),
            moment_sign=float(self.get_parameter("admittance.moment_sign").value),
        )
        self.normal_force_sign = float(self.get_parameter("ft_sensor.normal_force_sign").value)

        self.force_min = float(self.get_parameter("setpoint.f_normal_min_n").value)
        self.force_max = float(self.get_parameter("setpoint.f_normal_max_n").value)
        self.moment_bias_max = float(self.get_parameter("setpoint.max_moment_bias_nm").value)
        self.hard_force_limit = float(self.get_parameter("safety.max_normal_force_n").value)
        self.warn_force_limit = float(self.get_parameter("safety.warn_normal_force_n").value)
        self.hard_moment_limit = float(self.get_parameter("safety.max_moment_nm").value)
        self.wrench_timeout = float(self.get_parameter("watchdog.wrench_timeout_s").value)
        self.image_timeout = float(self.get_parameter("setpoint.us_twist_timeout_s").value)

        self._check_limit_consistency()

        # setpoint 는 계단으로 뛰면 안 된다 (§5.4). 힘은 0 에서 시작해 램프로 올린다.
        force_slew = float(self.get_parameter("setpoint.force_slew_n_s").value)
        moment_slew = float(self.get_parameter("setpoint.moment_bias_slew_nm_s").value)
        self.force_limiter = SlewLimiter(force_slew, initial=0.0)
        self.moment_x_limiter = SlewLimiter(moment_slew, initial=0.0)
        self.moment_y_limiter = SlewLimiter(moment_slew, initial=0.0)

        # -- 입력 상태 ---------------------------------------------------
        self.wrench_raw: np.ndarray | None = None
        self.wrench_stamp = None
        self.commanded = ContactSetpoint()
        self.image_twist = np.zeros(6)
        self.image_stamp = None
        self.teleop_twist = np.zeros(6)
        self.teleop_stamp = None
        self.last_step_time = None
        self._suppressed = False

        # 모드가 한 번도 안 왔으면 default_mode 로 돈다 — supervisor 없이 단독 기동해
        # 시험할 수 있어야 하기 때문이다. 한 번이라도 받은 뒤 끊기면 발행을 멈춘다.
        self.mode = str(self.get_parameter("admittance.default_mode").value)
        self.mode_stamp = None
        self.mode_timeout = float(self.get_parameter("admittance.mode_timeout_s").value)
        self.retreat_speed = float(self.get_parameter("watchdog.retreat_speed_m_s").value)

        ns = f"/{self.robot_name}"
        self.create_subscription(WrenchStamped, f"{ns}/wrench", self._on_wrench, 10)
        self.create_subscription(
            WrenchStamped, "/control/contact_setpoint", self._on_setpoint, 10
        )
        self.create_subscription(Twist, "/control/image_twist", self._on_image_twist, 10)
        self.create_subscription(Twist, "/control/teleop_twist", self._on_teleop_twist, 10)
        self.create_subscription(String, "/supervisor/mode", self._on_mode, 10)

        self.twist_pub = self.create_publisher(Twist, f"{ns}/desired_twist", 10)
        self.diag_pub = self.create_publisher(Float32MultiArray, "/diag/admittance", 10)

        self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info(
            f"Admittance {self.rate_hz:.0f} Hz · B_z={self.config.damping_linear} · "
            f"B_r={self.config.damping_angular} · F_n 한계 {self.hard_force_limit} N"
        )

    # -- 설정 ------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("robot.name", "fr5_right")
        self.declare_parameter("rates.cartesian_hz", 100.0)

        self.declare_parameter("tool.j6_to_probe_xyz", [float("nan")] * 3)
        self.declare_parameter("tool.j6_to_probe_rpy", [float("nan")] * 3)
        self.declare_parameter("tool.allow_missing_tool", False)
        self.declare_parameter("ft_sensor.j6_to_sensor_xyz", [float("nan")] * 3)
        self.declare_parameter("ft_sensor.j6_to_sensor_rpy", [float("nan")] * 3)

        self.declare_parameter("ft_sensor.normal_force_sign", -1.0)
        self.declare_parameter("ft_sensor.force_deadband_n", 0.1)
        self.declare_parameter("ft_sensor.moment_deadband_nm", 0.01)

        self.declare_parameter("admittance.damping_linear_ns_m", 1000.0)
        self.declare_parameter("admittance.damping_angular_nms_rad", 0.5)
        self.declare_parameter("admittance.moment_sign", 1.0)
        # supervisor 없이 단독 기동할 때 쓰는 모드. 한 번이라도 모드를 받으면
        # 그 뒤부터는 두절 시 발행을 멈춘다.
        self.declare_parameter("admittance.default_mode", "contact")
        self.declare_parameter("admittance.mode_timeout_s", 0.5)
        self.declare_parameter("watchdog.retreat_speed_m_s", 0.005)

        self.declare_parameter("setpoint.f_normal_min_n", 1.0)
        self.declare_parameter("setpoint.f_normal_max_n", 7.0)
        self.declare_parameter("setpoint.max_moment_bias_nm", 0.10)
        self.declare_parameter("setpoint.force_slew_n_s", 2.0)
        self.declare_parameter("setpoint.moment_bias_slew_nm_s", 0.05)
        self.declare_parameter("setpoint.us_twist_timeout_s", 0.5)

        self.declare_parameter("safety.max_linear_vel_m_s", 0.010)
        self.declare_parameter("safety.max_angular_vel_rad_s", 0.2)
        self.declare_parameter("safety.max_normal_force_n", 7.0)
        self.declare_parameter("safety.warn_normal_force_n", 6.0)
        self.declare_parameter("safety.max_moment_nm", 0.3)
        self.declare_parameter("watchdog.wrench_timeout_s", 0.05)

    def _check_limit_consistency(self) -> None:
        """서로 다른 절에 흩어진 한계가 모순되지 않는지 본다.

        `setpoint.f_normal_max_n` 과 `safety.max_normal_force_n` 은 같은 물리량을
        가리키므로 어긋나면 어느 쪽이 이기는지 코드를 읽어야만 알게 된다.
        """
        if abs(self.force_max - self.hard_force_limit) > 1e-9:
            raise RuntimeError(
                f"setpoint.f_normal_max_n({self.force_max})와 "
                f"safety.max_normal_force_n({self.hard_force_limit})가 다르다. "
                f"같은 값이어야 한다."
            )
        if self.moment_bias_max >= self.hard_moment_limit:
            raise RuntimeError(
                f"setpoint.max_moment_bias_nm({self.moment_bias_max})가 "
                f"safety.max_moment_nm({self.hard_moment_limit}) 이상이다. "
                f"과도응답 여유가 없다."
            )

    def _resolve_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """센서 → 프로브 변환. 미측정이면 기동을 거부한다 (§4.3).

        Raises:
            RuntimeError: CAD/장착 값이 없는데 ``allow_missing_tool`` 이 꺼져 있을 때.
        """
        probe_xyz = list(self.get_parameter("tool.j6_to_probe_xyz").value)
        probe_rpy = list(self.get_parameter("tool.j6_to_probe_rpy").value)
        sensor_xyz = list(self.get_parameter("ft_sensor.j6_to_sensor_xyz").value)
        sensor_rpy = list(self.get_parameter("ft_sensor.j6_to_sensor_rpy").value)

        values = probe_xyz + probe_rpy + sensor_xyz + sensor_rpy
        missing = any(v is None or math.isnan(float(v)) for v in values)

        if not missing:
            return probe_from_sensor(probe_xyz, probe_rpy, sensor_xyz, sensor_rpy)

        if not bool(self.get_parameter("tool.allow_missing_tool").value):
            raise RuntimeError(
                "tool.j6_to_probe_* 또는 ft_sensor.j6_to_sensor_* 가 미측정이다. "
                "wrench 기준점 이동 없이는 M_x/M_y 자세 정렬이 성립하지 않는다 — "
                "r×F 항이 0.5 N·m 로 정렬 신호를 덮는다 (§4.3)."
            )
        self.get_logger().error(
            "센서/프로브 변환 없음 — 항등변환으로 돈다. "
            "M_x/M_y 는 지렛대 항에 오염되어 있으므로 자세 정렬을 신뢰하지 말 것. "
            "z축 힘 추종만 유효하다."
        )
        return np.eye(3), np.zeros(3)

    # -- 입력 ------------------------------------------------------------

    def _on_wrench(self, msg: WrenchStamped) -> None:
        self.wrench_raw = np.array(
            [
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
            ]
        )
        self.wrench_stamp = self.get_clock().now()

    def _on_setpoint(self, msg: WrenchStamped) -> None:
        """접촉 목표. 여기서 곧바로 범위를 자른다 — 슬루는 제어 루프에서 건다."""
        self.commanded = ContactSetpoint(
            normal_force=float(np.clip(msg.wrench.force.z, self.force_min, self.force_max)),
            moment_x=float(np.clip(msg.wrench.torque.x, -self.moment_bias_max, self.moment_bias_max)),
            moment_y=float(np.clip(msg.wrench.torque.y, -self.moment_bias_max, self.moment_bias_max)),
        )

    def _on_image_twist(self, msg: Twist) -> None:
        """policy 의 면내 지령. 힘 축 성분이 실려 와도 합성 단계에서 버려진다."""
        self.image_twist = self._twist_array(msg)
        self.image_stamp = self.get_clock().now()

    def _on_teleop_twist(self, msg: Twist) -> None:
        """조작자 지령. TELEOP 모드에서만 **6축 전부** 통과한다."""
        self.teleop_twist = self._twist_array(msg)
        self.teleop_stamp = self.get_clock().now()

    def _on_mode(self, msg: String) -> None:
        if msg.data != self.mode:
            self.get_logger().info(f"모드 {self.mode} → {msg.data}")
        self.mode = msg.data
        self.mode_stamp = self.get_clock().now()

    @staticmethod
    def _twist_array(msg: Twist) -> np.ndarray:
        return np.array(
            [msg.linear.x, msg.linear.y, msg.linear.z,
             msg.angular.x, msg.angular.y, msg.angular.z]
        )

    # -- 제어 ------------------------------------------------------------

    def _age(self, stamp) -> float:
        if stamp is None:
            return math.inf
        return (self.get_clock().now() - stamp).nanoseconds / 1e9

    def _suppress(self, reason: str) -> None:
        """발행을 멈춘다. 아래층 워치독이 후퇴를 시작한다 (§12.2).

        슬루 상태를 0 으로 되돌려, 복귀 시 힘이 다시 램프로 올라가게 한다.
        """
        if not self._suppressed:
            self.get_logger().error(f"{reason} — twist 발행 중단. 하위 워치독이 후퇴시킨다.")
            self._suppressed = True
        self.force_limiter.reset(0.0)
        self.moment_x_limiter.reset(0.0)
        self.moment_y_limiter.reset(0.0)

    def _control_loop(self) -> None:
        now = self.get_clock().now()
        dt = 1.0 / self.rate_hz if self.last_step_time is None else \
            (now - self.last_step_time).nanoseconds / 1e9
        self.last_step_time = now
        dt = min(max(dt, 1e-3), 0.1)

        # 모드가 끊기면 감독자가 죽은 것이다. 마지막 모드를 붙들고 계속 도는 것보다
        # 발행을 멈춰 후퇴시키는 편이 안전하다.
        if self.mode_stamp is not None and self._age(self.mode_stamp) > self.mode_timeout:
            self._suppress("supervisor 모드 두절")
            return

        if self.mode == "idle":
            self._publish(np.zeros(6))
            self._suppressed = False
            return

        if self.mode == "teleop":
            self._passthrough()
            return

        if self.wrench_raw is None or self._age(self.wrench_stamp) > self.wrench_timeout:
            self._suppress("F/T 두절")
            return

        if self.mode == "retreat":
            # 프로브 −z. 힘 루프를 끄고 일정 속도로 물러난다. 종료 판정은 supervisor 가 한다.
            self._publish(np.array([0.0, 0.0, -self.retreat_speed, 0.0, 0.0, 0.0]))
            self._suppressed = False
            return

        wrench_probe = wrench_sensor_to_probe(self.wrench_raw, self.rotation, self.offset)

        normal_force = self.normal_force_sign * wrench_probe[2]
        moment_norm = math.hypot(wrench_probe[3], wrench_probe[4])
        if normal_force > self.hard_force_limit:
            self._suppress(f"접촉력 한계 초과 {normal_force:.2f} N > {self.hard_force_limit} N")
            return
        if moment_norm > self.hard_moment_limit:
            self._suppress(f"모멘트 한계 초과 {moment_norm:.3f} N·m > {self.hard_moment_limit}")
            return
        if normal_force > self.warn_force_limit:
            self.get_logger().warn(
                f"접촉력 {normal_force:.2f} N — 소프트 한계 {self.warn_force_limit} N 초과",
                throttle_duration_sec=1.0,
            )

        self._suppressed = False

        # setpoint 슬루. 계단 입력이 각속도 예산을 한 번에 소진하는 것을 막는다.
        settled = ContactSetpoint(
            normal_force=self.force_limiter.step(self.commanded.normal_force, dt),
            moment_x=self.moment_x_limiter.step(self.commanded.moment_x, dt),
            moment_y=self.moment_y_limiter.step(self.commanded.moment_y, dt),
        )

        # US 가 늦으면 면내 축만 죽인다. 힘 루프는 계속 돈다 (§8.3).
        image_twist = self.image_twist
        if self._age(self.image_stamp) > self.image_timeout:
            image_twist = np.zeros(6)

        twist, diag = compose_probe_twist(
            wrench_probe, settled, image_twist, self.config, self.normal_force_sign
        )
        self._publish(twist)
        self.diag_pub.publish(
            Float32MultiArray(
                data=[
                    diag["normal_force"], settled.normal_force, diag["error_force"],
                    diag["moment_x"], settled.moment_x, diag["error_moment_x"],
                    diag["moment_y"], settled.moment_y, diag["error_moment_y"],
                ]
            )
        )

    def _passthrough(self) -> None:
        """TELEOP 모드 — 조작자 6축을 그대로 낸다 (§10 모드 계약).

        힘 루프를 돌리지 않는 이유: 접근 단계에는 접촉이 없어 ``F_n = 0`` 이고, 힘 루프를
        켜두면 ``F_n*`` 를 향해 스스로 내려가 조작자와 z 를 다툰다.

        접촉이 없는 동안에도 안전 클램프는 유지한다. 그리고 조작자 지령이 끊기면
        발행을 멈춰 하위 워치독이 후퇴시키게 한다 — 프로브가 마지막 속도로 계속
        가는 것을 막는다.
        """
        if self.teleop_stamp is None or self._age(self.teleop_stamp) > self.mode_timeout:
            self._suppress("teleop 지령 두절")
            return
        self._suppressed = False

        twist = np.array(self.teleop_twist, dtype=float)
        linear = np.linalg.norm(twist[:3])
        if linear > self.config.max_linear:
            twist[:3] *= self.config.max_linear / linear
        angular = np.linalg.norm(twist[3:])
        if angular > self.config.max_angular:
            twist[3:] *= self.config.max_angular / angular
        self._publish(twist)

    def _publish(self, twist: np.ndarray) -> None:
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = (float(v) for v in twist[:3])
        msg.angular.x, msg.angular.y, msg.angular.z = (float(v) for v in twist[3:])
        self.twist_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = UsAdmittanceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f"[us_admittance_node] 기동 거부: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
