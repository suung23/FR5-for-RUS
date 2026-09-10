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

import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, WrenchStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray, Float64, String

from fr5_ik.dls_solver import DlsSolver
from fr5_ik.force_regulator import ForceRegulator, RegulatorOutput
from fr5_ik.probing_mode import APPROACH, CONTACT_PROBING, ProbingModeSwitch
from fr5_ik.teleop_frame import TeleopFrameMapper

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
        #: twist 두절 중 힘 조절을 잇는다는 안내를 이미 냈는가.
        self._force_hold_announced = False
        self.max_retreat = float(self.get_parameter("watchdog.max_retreat_s").value)

        # 접근 상한. freespace:=true 면 launch 가 덮어쓴 값이 여기로 들어온다.
        self.approach_linear = float(self.get_parameter("safety.max_linear_vel_m_s").value)
        self.approach_angular = float(self.get_parameter("safety.max_angular_vel_rad_s").value)
        self.contact_linear = float(
            self.get_parameter("safety.contact.max_linear_vel_m_s").value
        )
        self.contact_angular = float(
            self.get_parameter("safety.contact.max_angular_vel_rad_s").value
        )
        # 지금 쓰는 상한. 모드가 바뀌면 이 둘만 갈아 끼운다.
        self.max_linear = self.approach_linear
        self.max_angular = self.approach_angular

        self.allow_teleop_lateral = bool(
            self.get_parameter("contact_control.allow_teleop_lateral").value
        )
        self.allow_inplane_rotation = bool(
            self.get_parameter("contact_control.allow_teleop_inplane_rotation").value
        )
        self.regulator = self._build_regulator()
        self.regulator_reason = ""
        self.force_hold_enabled = bool(
            self.get_parameter("contact_control.force_hold_enabled").value
        )
        self.require_calibration = bool(
            self.get_parameter("contact_control.require_valid_calibration").value
        )
        #: 브리지가 알려 주는 교정 유효성. None 이면 아직 못 들었다는 뜻이고,
        #: 그것은 "유효하다" 와 다르다 — 모르는 것을 통과시키지 않는다.
        self.calibration_valid = None
        # 접촉 판정 보류를 한 번만 알리기 위한 표시. 렌치는 50 Hz 로 들어온다.
        self._mode_gate_warned = False
        self._calibration_blocked_announced = False
        #: 보상 전 렌치를 버리고 있다는 사실을 한 번만 알리기 위한 표시.
        self._raw_wrench_warned = False

        self.contact_probing_enabled = bool(
            self.get_parameter("teleop.contact_probing_enabled").value
        )
        # 이탈 문턱이 0 이하면 "되돌아가지 않는다" 는 뜻이다 — ROS2 파라미터는 None 을
        # 표현하지 못하므로 그 자리를 0 이 대신한다. 그 해석은 _build_switch 안에 있고,
        # 기동과 런타임 변경이 **같은 함수**를 쓰므로 둘이 갈릴 수 없다.
        self.mode_switch = self._build_switch()
        self.normal_sign = float(self.get_parameter("ft_sensor.normal_force_sign").value)
        # 판정과 제어가 무엇을 "접촉력" 으로 부를지. §4.4 · probing_mode 참조.
        self.contact_force_mode = str(
            self.get_parameter("ft_sensor.contact_force_mode").value
        ).strip().lower()
        if self.contact_force_mode not in ("magnitude", "normal"):
            raise RuntimeError(
                f"ft_sensor.contact_force_mode 는 magnitude 또는 normal 이어야 한다: "
                f"{self.contact_force_mode!r}"
            )

        # 병진 기준 프레임. "latched" 가 표류를 없애는 쪽, "probe" 가 예전 거동이다.
        self.linear_frame = str(self.get_parameter("teleop.linear_frame").value)
        self.angular_frame = str(self.get_parameter("teleop.angular_frame").value)
        self.engage_gap = float(self.get_parameter("teleop.engage_gap_s").value)
        self.mapper = TeleopFrameMapper(
            tip_roll_deg=float(self.get_parameter("teleop.tip_roll_deg").value),
            relatch_on_engage=bool(self.get_parameter("teleop.relatch_on_engage").value),
            operator_yaw_deg=float(self.get_parameter("teleop.operator_yaw_deg").value),
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
        #: 법선력 ``F_n = sign x F_z``. 양수 = 압축.
        self.normal_force = 0.0
        #: 접촉력 크기 ``‖F‖``. 부호가 없으므로 인장도 양수로 나온다 — 아래
        #: :meth:`_control_force` 가 그 구멍을 F_n 의 부호로 막는다.
        self.contact_force_mag = 0.0
        self.wrench_stamp = None
        self.last_twist_time = None
        self.last_joint_time = None
        self.retreat_started = None
        self._retreat_announced = False
        #: 시간 상한 도달을 이미 알렸는가. 이것도 한 번만 찍는다 — 아래 참조.
        self._retreat_gave_up = False
        self.rot_stylus = None
        self.last_stylus_time = None
        self._warned_no_stylus = False

        # -- 인터페이스 --------------------------------------------------
        ns = f"/{self.robot_name}"
        self.create_subscription(JointState, f"{ns}/joint_states", self._on_joints, 10)
        self.create_subscription(Twist, f"{ns}/desired_twist", self._on_twist, 10)
        wrench_topic = f"{ns}/{self.get_parameter('ft_sensor.wrench_topic').value}"
        self.create_subscription(WrenchStamped, wrench_topic, self._on_wrench, 10)
        self.create_subscription(
            Bool, f"{ns}/calibration_valid", self._on_calibration, 10
        )
        # 스타일러스 자세. teleop 이 데드맨을 잡고 있는 동안에만 발행하므로,
        # **발행이 끊겼다 다시 오는 것 자체가 재파지 신호**다 (별도 토픽이 필요 없다).
        side = str(self.robot_name).rsplit("_", 1)[-1]
        self.create_subscription(
            PoseStamped, f"/touch/{side}/stylus_pose", self._on_stylus, 10
        )

        self.velocity_pub = self.create_publisher(JointState, f"{ns}/joint_velocity_cmds", 10)
        self.error_pub = self.create_publisher(Float32MultiArray, "/diag/twist_tracking_error", 10)
        # 콘솔이 로봇의 **선언된** 모드를 볼 수 있어야 한다. GUI 가 자기 판정으로
        # 추측하면 로봇이 실제로 쓰는 상한과 어긋날 수 있다. latched 로 내보내
        # 늦게 붙은 구독자도 현재 모드를 즉시 받는다.
        # TRANSIENT_LOCAL. 모드는 전환할 때만 발행하므로, 늦게 붙는 구독자가
        # volatile 이면 다음 전환까지 아무것도 못 받는다 — 단방향 전환이라 그
        # "다음" 이 영영 안 올 수도 있다. 콘솔 브리지를 재시작하면 로봇이 이미 접촉
        # 프로빙인데 화면은 접근으로 보이게 된다.
        self.mode_pub = self.create_publisher(
            String,
            f"{ns}/probing_mode",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.retreat_pub = self.create_publisher(Bool, "/diag/retreating", 10)

        # 힘 축이 무엇을 지령받는가. `regulator_reason` 은 지금까지 노드 안에서만
        # 살아 있었다 — 밖에서 보면 "안 움직인다" 와 "움직이라고 했는데 안 갔다" 가
        # 구별되지 않는다. 그 둘을 가르는 것이 이 토픽의 전부다.
        #   [접촉력, 목표, 지령 v_z, 달성 오차 e_z, 접촉프로빙 여부]
        self.force_diag_pub = self.create_publisher(
            Float32MultiArray, "/diag/force_regulation", 10)
        self.force_reason_pub = self.create_publisher(
            String, "/diag/force_regulation_reason", 10)
        self._last_reason = None
        self._last_v_z = 0.0

        # 조작자 위치(미러) 상태와 요청. 모드와 같은 이유로 latched 다 — 콘솔이
        # 늦게 붙어도 지금 어느 매핑으로 도는지 즉시 알아야 한다. 화면이 축을
        # 추측하면 "밀면 반대로 온다" 를 조작자가 혼자 판단하게 된다.
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.teleop_frame_pub = self.create_publisher(
            String, f"{ns}/teleop_frame", latched_qos
        )
        # 콘솔에서 바꾸는 길. **파라미터가 진실이고 이 토픽은 그 파라미터를 쓰는
        # 창구다** — 요청이 오면 자기 파라미터를 갱신하므로 `ros2 param get` 과
        # 화면이 갈라지지 않는다.
        self.create_subscription(
            Float64, f"{ns}/teleop_frame_request", self._on_frame_request, latched_qos
        )
        # 면내 회전 모드도 같은 규약이다 — 파라미터가 진실이고 토픽은 그것을 쓰는
        # 창구다. 그래야 `ros2 param get` 과 화면이 갈라지지 않는다.
        self.create_subscription(
            Bool, f"{ns}/inplane_rotation_request", self._on_inplane_request, latched_qos
        )
        # 그리고 그 진실을 되돌려 알린다. 요청과 상태를 나누는 이유는 **미리 걸어
        # 둘 수 있어야** 하기 때문이다: 접촉은 예고 없이 시작되고 그 순간 조작자의
        # 손은 스타일러스에 있지 콘솔에 있지 않다. 접근 중에 켜 두면 전환과 함께
        # 적용되고, 화면은 그동안 "걸어 뒀다" 를 보여 줘야 한다 — 그것을 모드
        # 문자열로 말할 수는 없다. 모드는 지금 실제 속도 상한이 무엇인가이고,
        # 접근 중에는 접근이기 때문이다.
        self.inplane_pub = self.create_publisher(
            Bool,
            f"{ns}/inplane_rotation_state",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self._publish_inplane_state()
        self._publish_teleop_frame()

        # 힘 파라미터가 런타임에 바뀌면 조절기를 다시 만든다. 모드 스위치가
        # 만들어진 뒤에 걸어야 한다 — 콜백이 이탈 문턱을 참조한다.
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info(
            f"미분 IK {self.rate_hz:.0f} Hz · DLS λ={self.solver.damping} · "
            f"twist ZOH 유지 {self.twist_hold * 1000:.0f} ms → 감쇠 → "
            f"워치독 {self.twist_timeout * 1000:.0f} ms → 프로브 −z 후퇴"
        )
        self.get_logger().info(f"wrench 구독: {wrench_topic}")
        # 클램프를 기동 로그에 찍는다. 이 값이 안 보이면 freespace 인자를 빠뜨렸는지
        # 조작감만으로는 구분할 수 없다 — teleop 스케일을 아무리 올려도 여기서 잘리므로
        # "스케일이 안 먹는다" 로만 나타난다. 실제로 그 혼동이 있었다 (2026-08-19).
        self.get_logger().info(
            f"속도 클램프: 접근 {self.approach_linear * 1000:.0f} mm/s · "
            f"{self.approach_angular:.2f} rad/s  →  접촉 "
            f"{self.contact_linear * 1000:.0f} mm/s · {self.contact_angular:.2f} rad/s "
            + (f"({self.mode_switch.enter_force_n:.1f} N 에서 전환, "
               + ("복귀 가능" if self.mode_switch.reversible else "단방향") + ")"
               if self.contact_probing_enabled else "(전환 꺼짐)")
            + ("   ← 접근 상한이 이미 접촉용이다. freespace:=true 를 빠뜨린 것이다"
               if self.approach_linear <= 0.02 else "")
        )
        # 무엇을 접촉력으로 보는지, 그리고 어느 힘을 유지하는지를 기동 로그에 남긴다.
        # 이 두 줄이 없으면 "왜 안 잡히나 / 왜 이만큼만 누르나" 를 조작감으로만
        # 판단하게 된다. 문턱이 한 자릿수 N 으로 내려온 뒤로는 그 차이가 눈에 잘 안 띈다.
        self.get_logger().info(
            f"접촉력 기준: {self.contact_force_mode}"
            + ("  (‖F‖ = √(Fx²+Fy²+Fz²), 부호는 F_n 이 준다)"
               if self.contact_force_mode == "magnitude"
               else f"  (F_n = {self.normal_sign:+.0f} x F_z)")
        )
        self.get_logger().info(
            f"힘 유지: 목표 {self.regulator.target_force_n:.1f} ± "
            f"{self.regulator.deadband_n:.1f} N · B_z {self.regulator.admittance_b_z:.0f} N·s/m · "
            f"경고 {self.regulator.warn_force_n:.1f} · 한계 {self.regulator.max_force_n:.1f} N"
            + (f" · 이탈 {self.mode_switch.release_force_n:.1f} N / "
               f"{self.mode_switch.release_confirm_s:.1f} s"
               if self.mode_switch.reversible else " · 이탈 없음(단방향)")
        )
        if not self.force_hold_enabled:
            # 기동부터 대조군이면 화면에 크게 말한다. 본 시험을 이 상태로 받으면
            # 파일은 멀쩡해 보이는데 제어가 없던 것이고, 그것은 나중에 알기 어렵다.
            self.get_logger().warn(
                "⚠️ 힘 유지가 꺼진 채로 기동한다 (contact_control.force_hold_enabled=false) — "
                "대조군이다. z 를 잡지 않고 힘을 기록만 한다."
            )
        if self.mode_switch.reversible and (
            self.regulator.target_force_n - self.regulator.deadband_n
            <= self.mode_switch.release_force_n
        ):
            # 유지 밴드의 아래끝이 이탈 문턱 아래로 내려가면, 힘을 정상적으로 잡고
            # 있는 동안에도 이탈 조건이 성립한다 — 모드가 접촉과 접근을 오간다.
            self.get_logger().warn(
                f"⚠️ 유지 밴드 아래끝 "
                f"{self.regulator.target_force_n - self.regulator.deadband_n:.2f} N 이 "
                f"이탈 문턱 {self.mode_switch.release_force_n:.2f} N 이하다 — "
                "힘을 잡고 있는 중에 접근으로 되돌아갈 수 있다. "
                "contact_control.target_force_n 을 올리거나 "
                "teleop.contact_probing_release_n 을 내려라."
            )
        if not self.contact_probing_enabled:
            self.get_logger().warn(
                "⚠️ 접촉 프로빙 전환이 꺼져 있다 (teleop.contact_probing_enabled=false). "
                "힘이 얼마가 되든 접근 상한 "
                f"{self.approach_linear * 1000:.0f} mm/s 를 유지한다 — 접촉해도 속도가 "
                "줄지 않는다. 남는 안전층은 힘 한계 "
                f"{float(self.get_parameter('safety.max_contact_force_n').value):.1f} N 뿐이다. "
                "검증 절차 전용이며, "
                "팬텀 작업에는 켜고 써라."
            )
        # 늦게 붙는 구독자(콘솔 브리지)가 현재 모드를 즉시 받도록 한 번 낸다.
        self._publish_mode()
        # 병진 기준이 무엇인지 안 찍으면, 축이 어긋났을 때 원인을 조작감으로만
        # 판단하게 된다 — 2026-08-21 에 실제로 그렇게 시간을 썼다.
        self.get_logger().info(
            f"기준 프레임: 병진 {self.linear_frame} · 회전 {self.angular_frame}"
            + ("  (데드맨 첫 파지에 고정 → 표류 없음)"
               if self.linear_frame == "latched" and self.angular_frame == "latched"
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

        self.declare_parameter("safety.warn_contact_force_n", 4.5)
        self.declare_parameter("safety.max_contact_force_n", 5.0)
        self.declare_parameter("safety.max_linear_vel_m_s", 0.010)
        self.declare_parameter("safety.max_angular_vel_rad_s", 0.2)

        # 접촉 프로빙 상한. launch 의 freespace 오버라이드가 safety.max_* 를 덮어써도
        # 이쪽은 건드리지 않는다 — 두 벌이 동시에 있어야 런타임에 전환할 수 있다.
        self.declare_parameter("safety.contact.max_linear_vel_m_s", 0.010)
        self.declare_parameter("safety.contact.max_angular_vel_rad_s", 0.2)

        # 이 힘을 넘으면 접촉 프로빙으로 넘어간다. 단방향이다.
        # 접촉 프로빙 전환을 켤지.
        #
        # 끄면 힘이 얼마가 되든 접근 상한을 유지한다. 전자저울 검증처럼 **의도적으로
        # 문턱을 넘겨 누르면서 teleop 을 계속해야 하는** 절차를 위한 것이다 — 전환은
        # 단방향이라 한 번 걸리면 그 세션에서 다시 못 나온다.
        #
        # ⚠️ 이것은 안전 거동을 끄는 스위치다. 접촉 시 속도를 15 배 줄이는 층이
        # 사라지고, 남는 것은 힘 한계(safety.max_contact_force_n)뿐이다. 그래서
        # 기본값은 켜짐이고, 끄면 기동 로그에 경고가 찍힌다.
        self.declare_parameter("teleop.contact_probing_enabled", True)
        # 2026-08-31: 8.0 → 1.0 → 2.0. 함께 판정 기준이 F_n 에서 접촉력 크기 ‖F‖ 로
        # 바뀌었다 (ft_sensor.contact_force_mode). 이 크기의 문턱은 F_z 하나로는
        # 못 잡는다 — 프로브가 조금만 기울어 닿으면 상당 부분이 횡력으로 가기 때문이다.
        # 값의 근거는 probe.yaml 이 들고 있다. 여기 기본값은 그것을 따라간다.
        self.declare_parameter("teleop.contact_probing_force_n", 2.0)
        self.declare_parameter("teleop.contact_probing_confirm_s", 0.02)
        # 접근으로 되돌아가는 문턱. 0 이하 = 되돌아가지 않는다(예전 단방향 거동).
        #
        # 1 N 문턱에서는 스치기만 해도 전환되므로 단방향을 유지할 수 없다. 접촉
        # 프로빙에서는 조작자의 여섯 축이 모두 0 이라 스스로 빠져나올 수단도 없다 —
        # 데드맨을 놓아 워치독 후퇴를 부르는 것이 유일한 길이고, 그 후퇴의 종료
        # 조건(watchdog.retreat_until_force_n)과 같은 자리에 이 값을 둔다.
        self.declare_parameter("teleop.contact_probing_release_n", 0.3)
        # 이탈 확인 창. 진입(20 ms)보다 훨씬 길다 — 누르는 중의 순간적인 힘 감소로
        # 접근 속도가 되살아나는 것이 이 판정에서 가장 위험한 오작동이다.
        self.declare_parameter("teleop.contact_probing_release_confirm_s", 0.5)
        # 교정이 유효할 때만 힘 기반 동작을 연다 (사양 §6·§7).
        self.declare_parameter("contact_control.require_valid_calibration", True)

        # 접촉 프로빙에서 로봇이 스스로 잡는 힘. §7 admittance 의 첫 구현이다.
        # 2026-08-31: 5.0 → 3.0, 그리고 **무엇의 N 인지도 바뀌었다** — 법선력이 아니라
        # 접촉력 크기다 (contact_force_mode). 진입 문턱(1.0)보다 높으므로, 닿는
        # 순간부터 로봇이 스스로 여기까지 파고든다. probe.yaml 이 값의 출처이며
        # 그 근거도 거기 있다.
        self.declare_parameter("contact_control.target_force_n", 3.0)
        self.declare_parameter("contact_control.deadband_n", 0.5)
        self.declare_parameter("contact_control.admittance_b_z", 1000.0)
        self.declare_parameter("contact_control.retreat_speed_m_s", 0.005)
        # 힘 유지를 켤지. **거짓이 위약(placebo) 대조군이다** — 접촉 모드도, 속도
        # 상한도, 안전층도 그대로 두고 z 조절만 놓는다. 프로브는 그 자리에 선 채
        # 팬텀이 변하는 대로 힘이 흐르고, 그것이 "제어가 없었다면 얼마였나" 다.
        #
        # 이 값 없이는 교란 시험이 무엇을 보였는지 말할 수 없다. 힘이 목표 근처에
        # 머문 것이 제어 덕분인지, 애초에 교란이 그 정도였을 뿐인지 구별되지 않는다.
        # 대조군을 같은 접촉점·같은 주입량으로 이어서 받아야 그 차이가 곧 제어의
        # 기여가 된다.
        self.declare_parameter("contact_control.force_hold_enabled", True)
        # 접촉 프로빙에서 조작자의 병진·회전을 그대로 통과시킬지. 기본은 거짓 —
        # 로봇이 힘만 잡고 나머지는 정지한다. policy 가 영상축을 맡기 전까지
        # 조작자가 미끄러뜨리며 쓰고 싶으면 켠다.
        self.declare_parameter("contact_control.allow_teleop_lateral", False)
        # 면내 회전 모드 (2026-08-31).
        #
        # 접촉 프로빙에서 힘은 로봇이 잡되, 조작자에게 **ω_y 하나만** 돌려준다.
        # 영상면은 프로브의 x–z 평면이고 +y 는 그 면의 법선(elevational)이므로,
        # y 둘레 회전은 면을 **자기 자신으로** 옮긴다 — 즉 빔이 훑는 평면이 공간에서
        # 바뀌지 않는다. 초음파에서 rocking 이라 부르는 조작이 이것이다.
        #
        # 나머지 축을 열지 않는 이유는 각각 면을 벗어나기 때문이다: ω_z 는 영상면을
        # 통째로 돌리고(§8.3 에서 영상축), ω_x 는 면을 기울여 빼내며, v_y 는 면 밖으로
        # 미끄러진다. v_x 는 면 안에 남지만 회전이 아니라 병진이라 여기 넣지 않았다 —
        # 필요하면 별도로 연다.
        #
        # ⚠️ §8.2 의 rx/ry 자세 정렬이 들어오면 ω_y 를 admittance 가 쓰려 한다
        # (§8.3 의 힘축 셋 중 하나다). 그때 둘 중 누가 쓰는지는 arbiter 가 정해야
        # 하며, 지금은 정렬이 미구현이라 이 축이 비어 있어서 성립하는 모드다.
        self.declare_parameter("contact_control.allow_teleop_inplane_rotation", False)

        self.declare_parameter("watchdog.twist_timeout_s", 0.1)
        self.declare_parameter("watchdog.twist_hold_s", 0.04)
        self.declare_parameter("watchdog.joint_state_timeout_s", 0.2)
        self.declare_parameter("watchdog.retreat_speed_m_s", 0.005)
        self.declare_parameter("watchdog.retreat_until_force_n", 0.2)
        self.declare_parameter("watchdog.max_retreat_s", 3.0)

        self.declare_parameter("ft_sensor.normal_force_sign", -1.0)
        # 접촉 판정과 힘 유지가 무엇을 "접촉력" 으로 볼 것인가.
        #
        #   magnitude: ‖F‖ = √(Fx²+Fy²+Fz²). 기본. 방향에 무관하므로 프로브가
        #              기울어 닿아도 잡힌다. 1 N 문턱은 이것이 전제다.
        #   normal:    F_n = sign x F_z. 2026-08-31 이전 거동. 전자저울 검증처럼
        #              축방향으로만 누르는 절차에서 두 값이 같아야 함을 확인할 때 쓴다.
        self.declare_parameter("ft_sensor.contact_force_mode", "magnitude")
        # 어느 wrench 를 볼 것인가. 우리 PX6D 는 컨트롤러에 안 물리므로 기본
        # `wrench`(us_servo 발행)는 0 이다. probe.yaml 이 `wrench_px6d` 를 준다.
        self.declare_parameter("ft_sensor.wrench_topic", "wrench")

        # teleop 노드와 **같은** probe.yaml 항목을 읽는다. 값이 갈라지면 병진이
        # 조용히 90° 틀어지므로, 여기서 따로 기본값을 만들지 않는다.
        self.declare_parameter("teleop.tip_roll_deg", 90.0)
        self.declare_parameter("teleop.linear_frame", "latched")
        # 회전도 같은 고정 프레임을 쓸지. "latched" 또는 "body".
        #
        # 조작 중에 바뀌면 지령 방향이 한 주기에 튀므로, 값은 **다음 파지에서만**
        # 반영한다. 데드맨을 놓았다 다시 잡으면 적용되고, 그 사이에 로봇이 예상 못 한
        # 방향으로 움직이는 일은 없다.
        self.declare_parameter("teleop.angular_frame", "body")
        self.declare_parameter("teleop.relatch_on_engage", False)
        self.declare_parameter("teleop.engage_gap_s", 0.3)
        # 조작자가 로봇의 어느 쪽에 서 있는가 [도]. base 수직축 둘레 회전이며,
        # 180 이 "마주보기"(흔히 말하는 미러 모드)다. teleop_frame 모듈 문서 참조.
        self.declare_parameter("teleop.operator_yaw_deg", 0.0)

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

        # 파지 경계에서만 다시 읽는다. 조작 중 `ros2 param set` 을 해도 손을 놓았다
        # 잡기 전까지는 축이 바뀌지 않으므로, 비교하다가 로봇이 튀는 일이 없다.
        previous_angular = self.angular_frame
        self.angular_frame = str(self.get_parameter("teleop.angular_frame").value)
        if self.angular_frame != previous_angular:
            self.get_logger().info(
                f"회전 기준 프레임 변경: {previous_angular} → {self.angular_frame}"
            )

        # 조작자 위치도 파지 경계에서만 바뀐다. 조작 중에 뒤집히면 같은 손동작에
        # 로봇이 반대로 가고, 그 순간의 반사적인 교정은 상황을 악화시킨다.
        requested = float(self.get_parameter("teleop.operator_yaw_deg").value)
        self.mapper.request_operator_yaw(requested)
        pending = self.mapper.pending_yaw_deg

        latched = self.mapper.engage(self.solver.rotation_base_probe(self.q), self.rot_stylus)

        if pending is not None:
            self.get_logger().warn(
                f"조작자 위치 {self.mapper.operator_yaw_deg:+.0f}° 적용 — "
                + ("**마주보기(미러)**: 앞뒤와 좌우가 뒤집힌다. "
                   if self.mapper.mirrored else "나란히: 지금까지의 축이다. ")
                + "침투축(프로브 z)은 바뀌지 않는다."
            )
            self._publish_teleop_frame()
        if latched:
            self.get_logger().info(
                f"기준 프레임 고정 (파지 {self.mapper.engage_count}회차). "
                f"이 순간의 축이 세션 내내 유지된다 — "
                f"손 자세가 돌아도 '오른쪽'은 계속 같은 방향이다."
            )

    def _on_frame_request(self, msg: Float64) -> None:
        """콘솔이 조작자 위치를 바꿔 달라고 한다.

        **즉시 바뀌지 않는다.** 자기 파라미터에 적어 두면 다음 파지 경계에서
        ``_on_stylus`` 가 그것을 읽어 적용한다. 파라미터를 진실로 두는 덕에
        ``ros2 param get`` 과 화면이 갈라지지 않는다.
        """
        wanted = float(msg.data)
        if not math.isfinite(wanted):
            self.get_logger().error(f"조작자 위치 요청이 숫자가 아니다: {msg.data}")
            return
        self.set_parameters([Parameter("teleop.operator_yaw_deg", value=wanted)])
        self.mapper.request_operator_yaw(wanted)
        if self.mapper.pending_yaw_deg is None:
            self.get_logger().info(f"조작자 위치 {wanted:+.0f}° — 이미 그 값이다")
        else:
            self.get_logger().info(
                f"조작자 위치 {wanted:+.0f}° 예약 — **다음 파지부터** 적용된다. "
                "손을 놓았다 다시 잡아라."
            )
        self._publish_teleop_frame()

    def _publish_teleop_frame(self) -> None:
        """지금 어느 매핑으로 도는지 한 장으로 알린다 (콘솔용)."""
        msg = String()
        msg.data = json.dumps({
            "operatorYawDeg": self.mapper.operator_yaw_deg,
            "pendingYawDeg": self.mapper.pending_yaw_deg,
            "mirrored": self.mapper.mirrored,
            "linearFrame": self.linear_frame,
            "angularFrame": self.angular_frame,
            "tipRollDeg": float(self.get_parameter("teleop.tip_roll_deg").value),
            "engageCount": self.mapper.engage_count,
        })
        self.teleop_frame_pub.publish(msg)

    def _on_calibration(self, msg: Bool) -> None:
        """브리지가 알려 주는 교정 유효성."""
        if self.calibration_valid is not msg.data:
            self.get_logger().info(f"교정 유효성: {msg.data}")
        self.calibration_valid = bool(msg.data)

    def _control_force(self) -> float:
        """판정과 힘 유지가 함께 보는 스칼라 [N]. 양수 = 누름.

        ``magnitude`` 에서는 ``‖F‖`` 인데, 크기에는 부호가 없으므로 프로브가 **당겨질
        때도 양수**로 나온다. 그대로 쓰면 인장이 접촉으로 잡히고, 조절기는 "이미 충분히
        누르고 있다" 고 판단해 물러난다 — 실제로는 붙어서 끌려가는 중인데.

        그래서 법선력의 부호를 크기에 얹는다. ``F_n < 0`` (당김) 이면 음수를 돌려주고,
        그러면 문턱을 넘지 못하며 조절기는 전진 쪽으로 판단한다. 부호만 빌려 오고
        크기는 세 축 전부에서 온다.
        """
        if self.contact_force_mode == "normal":
            return self.normal_force
        return (
            self.contact_force_mag if self.normal_force >= 0.0 else -self.contact_force_mag
        )

    #: 브리지가 중력보상을 실었을 때 붙이는 프레임 이름의 꼬리.
    #: 원값은 ``…_ft_sensor`` 로 나가므로 이 하나로 둘이 갈린다.
    COMPENSATED_FRAME_SUFFIX = "_probe"

    def _on_wrench(self, msg: WrenchStamped) -> None:
        # **이 메시지가 보상된 것인지 메시지 자신에게 묻는다.**
        #
        # 브리지는 보상할 수 없을 때 원값을 같은 토픽으로 낸다. 예전에는 그것을
        # 아래 교정 게이트(calibration_valid 토픽)가 막는다고 보았는데, 둘은 서로
        # **다른 토픽**이라 잠깐 어긋날 수 있다. 게이트가 열리는 순간 구독 큐에
        # 남아 있던 보상 전 표본이 그대로 판정에 들어간다.
        #
        # 2026-09-04 02:15:09 에 그것이 일어났다. 교정 유효 20 ms 뒤에 F 10.52 N
        # 으로 접촉 전환이 걸렸는데, 같은 자세의 보상값은 0.71 N 이었다 — 10.5 는
        # 마운트·프로브 자중, 즉 보상 전 값이다. 게다가 그 값은 한계 5.0 N 을
        # 넘으므로 조절기의 첫 분기가 **강제 후퇴**다. 아무것도 닿지 않은 세션
        # 시작에 로봇이 움직였고, has_contacted 걸쇠까지 남았다.
        #
        # 프레임 이름은 그 메시지와 함께 온다. 토픽 사이의 시간차가 끼어들 자리가
        # 없다.
        if not msg.header.frame_id.endswith(self.COMPENSATED_FRAME_SUFFIX):
            if not self._raw_wrench_warned:
                self._raw_wrench_warned = True
                self.get_logger().warn(
                    f"보상 전 렌치를 버린다 (frame_id={msg.header.frame_id!r}) — "
                    "자중이 실려 있어 접촉과 구분되지 않는다. 교정이 실리면 브리지가 "
                    f"…{self.COMPENSATED_FRAME_SUFFIX} 프레임으로 낸다."
                )
            return
        self._raw_wrench_warned = False

        now = self.get_clock().now()
        previous = self.wrench_stamp
        self.normal_force = self.normal_sign * msg.wrench.force.z
        # 세 축 크기. 브리지가 교정을 실었으면 이 토픽은 이미 중력보상된 프로브
        # 프레임 접촉 렌치다 (telemetry_bridge.publish_wrench).
        self.contact_force_mag = float(
            math.sqrt(
                msg.wrench.force.x ** 2
                + msg.wrench.force.y ** 2
                + msg.wrench.force.z ** 2
            )
        )
        self.wrench_stamp = now

        # 교정이 유효하지 않으면 **모드 판정 자체를 하지 않는다.**
        #
        # 보상 전 렌치에는 마운트·프로브 자중이 그대로 실려 있다 (실측 약 10 N).
        # 그 값은 자세에 따라 부호가 바뀌므로, 프로브가 위를 향하는 것만으로도
        # 8 N 문턱을 넘는다. 전환은 **단방향** 이라 한 번 걸리면 접근 속도가
        # 세션 내내 15 배로 묶인다 — 아무것도 닿지 않았는데.
        #
        # 힘 축을 여는 것은 이미 교정으로 막혀 있었지만(_apply_force_regulation),
        # 속도 상한을 갈아 끼우는 것은 막혀 있지 않았다. 접촉 판정이 못 믿을
        # 값에서 나온다면 그 판정으로 무엇도 바꾸면 안 된다.
        if not self.contact_probing_enabled:
            return

        if self.require_calibration and self.calibration_valid is not True:
            if not self._mode_gate_warned:
                self._mode_gate_warned = True
                self.get_logger().warn(
                    "교정이 유효하지 않아 접촉 판정을 보류한다 — 보상 전 렌치에는 "
                    "자중이 실려 있어 접촉과 구분되지 않는다. 접근 상한을 유지한다."
                )
            return

        # 모드 판정은 힘이 들어오는 순간에 한다. 제어 루프(100 Hz)에 맡기면 센서가
        # 그보다 빨리 올 때 문턱을 넘는 순간을 지나칠 수 있다.
        dt = 0.0 if previous is None else (now - previous).nanoseconds / 1e9
        was_contact = self.mode_switch.in_contact_probing
        self.mode_switch.update(self._control_force(), dt)
        if self.mode_switch.in_contact_probing and not was_contact:
            self._enter_contact_probing()
        elif was_contact and not self.mode_switch.in_contact_probing:
            self._leave_contact_probing()

    def _enter_contact_probing(self) -> None:
        """접촉 프로빙으로 넘어간다. 상한을 갈아 끼우고 알린다."""
        self.max_linear = self.contact_linear
        self.max_angular = self.contact_angular
        self.get_logger().warn(
            f"접촉 프로빙 전환 — F {self._control_force():.2f} N "
            f"({self.contact_force_mode}, 문턱 {self.mode_switch.enter_force_n:.1f} N; "
            f"F_n {self.normal_force:+.2f} · ‖F‖ {self.contact_force_mag:.2f}). "
            f"속도 상한 {self.approach_linear * 1000:.0f} → "
            f"{self.contact_linear * 1000:.0f} mm/s · "
            f"{self.approach_angular:.2f} → {self.contact_angular:.2f} rad/s. "
            f"힘 유지 시작 — 목표 {self.regulator.target_force_n:.1f} ± "
            f"{self.regulator.deadband_n:.1f} N."
            + (
                f" {self.mode_switch.release_force_n:.1f} N 아래로 "
                f"{self.mode_switch.release_confirm_s:.1f} s 지속되면 접근으로 되돌아간다."
                if self.mode_switch.reversible
                else " 되돌아가지 않는다 — 접근 속도가 다시 필요하면 세션을 새로 시작하라."
            )
        )
        self._publish_mode()

    def _leave_contact_probing(self) -> None:
        """접근으로 되돌아간다. 접촉이 확실히 끝났을 때만 여기 온다.

        이탈 문턱과 확인 창이 이미 그것을 보장한다 (``probing_mode`` 참조). 여기서는
        상한을 되돌리고 조절기의 마지막 사유를 지운다 — 그 문장이 화면에 남아 있으면
        힘을 안 잡고 있는데 잡고 있는 것처럼 읽힌다.

        ``has_contacted`` 걸쇠는 지우지 않는다. "이번 세션에 무언가에 닿았다" 는 사실은
        모드가 되돌아간다고 사라지지 않으며, 콘솔의 접촉 걸쇠 표시가 그것을 본다.
        """
        self.max_linear = self.approach_linear
        self.max_angular = self.approach_angular
        self.regulator_reason = ""
        self.get_logger().warn(
            f"접근 전환 — F {self._control_force():.2f} N 이 "
            f"{self.mode_switch.release_force_n:.1f} N 아래로 "
            f"{self.mode_switch.release_confirm_s:.1f} s 지속됐다. "
            f"속도 상한 {self.contact_linear * 1000:.0f} → "
            f"{self.approach_linear * 1000:.0f} mm/s · "
            f"{self.contact_angular:.2f} → {self.approach_angular:.2f} rad/s. "
            f"힘 유지 해제 — z 축이 조작자에게 돌아갔다."
        )
        self._publish_mode()

    #: 바뀌면 조절기를 다시 만들어야 하는 파라미터.
    REGULATOR_PARAMS = (
        "contact_control.target_force_n",
        "contact_control.deadband_n",
        "contact_control.admittance_b_z",
        "contact_control.retreat_speed_m_s",
        "safety.warn_contact_force_n",
        "safety.max_contact_force_n",
    )

    #: 바뀌면 모드 스위치를 다시 만들어야 하는 파라미터.
    SWITCH_PARAMS = (
        "teleop.contact_probing_force_n",
        "teleop.contact_probing_release_n",
        "teleop.contact_probing_confirm_s",
        "teleop.contact_probing_release_confirm_s",
    )

    def _build_regulator(self, overrides=None) -> ForceRegulator:
        """현재 파라미터로 조절기를 만든다. ``overrides`` 는 아직 반영 전인 값이다."""
        def get(name):
            if overrides and name in overrides:
                return float(overrides[name])
            return float(self.get_parameter(name).value)
        return ForceRegulator(
            target_force_n=get("contact_control.target_force_n"),
            deadband_n=get("contact_control.deadband_n"),
            admittance_b_z=get("contact_control.admittance_b_z"),
            max_speed_m_s=self.contact_linear,
            warn_force_n=get("safety.warn_contact_force_n"),
            max_force_n=get("safety.max_contact_force_n"),
            retreat_speed_m_s=get("contact_control.retreat_speed_m_s"),
        )

    def _build_switch(self, overrides=None) -> ProbingModeSwitch:
        """현재 파라미터로 모드 스위치를 만든다."""
        def get(name):
            if overrides and name in overrides:
                return float(overrides[name])
            return float(self.get_parameter(name).value)
        release = get("teleop.contact_probing_release_n")
        return ProbingModeSwitch(
            enter_force_n=get("teleop.contact_probing_force_n"),
            confirm_s=get("teleop.contact_probing_confirm_s"),
            release_force_n=release if release > 0.0 else None,
            release_confirm_s=get("teleop.contact_probing_release_confirm_s"),
        )

    def _on_set_parameters(self, params):
        """힘 관련 파라미터를 **실제로** 반영한다.

        예전에는 조절기를 기동 때 한 번만 만들었고, 그 뒤의 ``ros2 param set`` 은
        파라미터만 바꾸고 조절기는 옛 값을 계속 썼다 — **조용히**. 목표를 1 N 으로
        바꿨다고 믿으면서 3 N 으로 누르는 일이 가능했다는 뜻이고, 힘 대역을 훑는
        실험이라면 결과 전체가 틀어진다.

        새 값으로 조절기를 **먼저 만들어 본다.** 생성자가 ``target < warn <= max``
        같은 불변식을 검사하므로, 못 만들면 사유와 함께 거절한다 — 거절된 set 은
        아무것도 바꾸지 않고 그 사실이 로그에 남는다.
        """
        from rcl_interfaces.msg import SetParametersResult

        touched = {p.name: p.value for p in params if p.name in self.REGULATOR_PARAMS}
        switched = {p.name: p.value for p in params if p.name in self.SWITCH_PARAMS}

        # 힘 유지 on/off 는 조절기를 다시 만들 필요가 없다. 대신 **로봇이 하는 일이
        # 바뀌므로** 지나가는 줄로 두지 않는다 — 대조군을 켜 놓은 채 본 시험을
        # 받으면 그 캡처는 조용히 쓸모가 없어지고, 파일만 봐서는 알기 어렵다.
        for p in params:
            if p.name != "contact_control.force_hold_enabled":
                continue
            self.force_hold_enabled = bool(p.value)
            if self.force_hold_enabled:
                self.get_logger().warn("힘 유지 켜짐 — z 를 로봇이 다시 잡는다")
            else:
                self.get_logger().warn(
                    "⚠️ 힘 유지 꺼짐 (대조군) — z 를 놓는다. 프로브는 그 자리에 서고 "
                    f"힘은 팬텀이 정하는 대로 흐른다. 한계 "
                    f"{self.regulator.max_force_n:.1f} N 후퇴는 그대로 살아 있다."
                )

        # 접촉 프로빙 전환 on/off. 조작자가 GUI 에서 régime 을 직접 고르는 길이며,
        # **동작 명령이 아니다** — 켜면 접촉이 잡힐 때 로봇이 z 를 가져가도 된다는
        # 허가이고, 끄면 그 허가를 거둔다.
        #
        # 예전에는 이 파라미터를 기동 때 한 번만 읽어 `self.contact_probing_enabled`
        # 에 넣어 두어서, `ros2 param set` 이 값만 바꾸고 거동은 그대로였다 —
        # 조용히. 위 force_hold_enabled 와 같은 종류의 함정이다.
        for p in params:
            if p.name != "teleop.contact_probing_enabled":
                continue
            self.contact_probing_enabled = bool(p.value)
            if self.contact_probing_enabled:
                self.get_logger().warn(
                    "접촉 프로빙 전환 켜짐 — 접촉이 잡히면 로봇이 z 를 가져간다")
            else:
                # **끄는 쪽은 지금 상태에서 빠져나와야 한다.** 아래 판정 루프는
                # 꺼져 있으면 곧바로 return 하므로, 이미 프로빙 중이면 거기 갇힌다 —
                # 조작자는 껐다고 믿는데 로봇은 접촉 상한과 힘 유지를 그대로 쥔다.
                if self.mode_switch.in_contact_probing:
                    self.mode_switch.mode = APPROACH
                    self._leave_contact_probing()
                    self.get_logger().warn(
                        "접촉 프로빙 전환 꺼짐 — 접근으로 되돌리고 z 를 놓는다")
                else:
                    self.get_logger().warn(
                        "접촉 프로빙 전환 꺼짐 — 접촉이 잡혀도 접근 상한을 유지한다")

        if not touched and not switched:
            return SetParametersResult(successful=True)

        # 스위치를 먼저 본다. 문턱은 접촉 판정 자체를 바꾸므로 조절기보다 앞선다.
        if switched:
            try:
                switch = self._build_switch(switched)
            except ValueError as exc:
                self.get_logger().error(f"전환 문턱 거절: {exc}")
                return SetParametersResult(successful=False, reason=str(exc))
            # **상태는 넘겨받는다.** 문턱을 바꿨다고 지금 접촉 중이라는 사실이나
            # 걸쇠가 사라지면, 파라미터 하나 만졌다고 로봇이 접근 속도로 돌아간다.
            switch.mode = self.mode_switch.mode
            switch.has_contacted = self.mode_switch.has_contacted
            self.mode_switch = switch
            self.get_logger().info(
                f"전환 문턱 갱신 — 진입 {switch.enter_force_n:.2f} N · 이탈 "
                + (f"{switch.release_force_n:.2f} N" if switch.reversible else "없음")
                + f" (모드 {switch.mode} 유지)"
            )
            self._publish_mode()

        if not touched:
            return SetParametersResult(successful=True)
        try:
            regulator = self._build_regulator(touched)
        except ValueError as exc:
            self.get_logger().error(f"힘 파라미터 거절: {exc}")
            return SetParametersResult(successful=False, reason=str(exc))

        self.regulator = regulator
        self.get_logger().info(
            f"힘 유지 갱신 — 목표 {regulator.target_force_n:.2f} ± "
            f"{regulator.deadband_n:.2f} N · 경고 {regulator.warn_force_n:.2f} · "
            f"한계 {regulator.max_force_n:.2f} N ("
            + ", ".join(f"{k.split('.')[-1]}={v}" for k, v in touched.items()) + ")"
        )
        # 밴드 아래끝이 이탈 문턱을 뚫으면 모드가 접촉과 접근을 오간다. 기동 때와
        # 같은 검사를 여기서도 한다 — 런타임 변경이라고 덜 위험한 것이 아니다.
        if self.mode_switch.reversible and (
            regulator.target_force_n - regulator.deadband_n
            <= self.mode_switch.release_force_n
        ):
            self.get_logger().warn(
                f"⚠️ 유지 밴드 아래끝 "
                f"{regulator.target_force_n - regulator.deadband_n:.2f} N 이 "
                f"이탈 문턱 {self.mode_switch.release_force_n:.2f} N 이하다 — "
                "힘을 잡고 있는 중에 접근으로 되돌아갈 수 있다."
            )
        return SetParametersResult(successful=True)

    def _publish_inplane_state(self) -> None:
        self.inplane_pub.publish(Bool(data=bool(self.allow_inplane_rotation)))

    def _on_inplane_request(self, msg: Bool) -> None:
        """면내 회전 모드를 켜고 끈다. 파라미터를 갱신해 두 창구를 같게 만든다.

        **접촉 프로빙이 아니어도 받는다.** 접근 중에 걸어 두면 전환하는 순간부터
        적용된다 — 접촉이 시작된 뒤에야 켤 수 있게 하면, 정작 켜야 할 때 조작자는
        손을 쓰고 있다.
        """
        want = bool(msg.data)
        if want == self.allow_inplane_rotation:
            return
        self.allow_inplane_rotation = want
        self.set_parameters([
            Parameter("contact_control.allow_teleop_inplane_rotation",
                      Parameter.Type.BOOL, want)
        ])
        self.get_logger().info(
            "면내 회전 모드 " + ("켬 — ω_y 가 조작자에게 열린다 (영상면 유지)"
                              if want else "끔 — 접촉 프로빙에서 여섯 축 모두 로봇이 잡는다")
            + ("" if self.mode_switch.in_contact_probing else " · 접촉 전이라 예약 상태다")
        )
        self._publish_inplane_state()
        self._publish_mode()

    def _publish_mode(self) -> None:
        msg = String()
        # 접촉 프로빙 안의 하위 모드까지 알린다. 화면이 "힘은 로봇이 잡고 회전은
        # 내가 한다" 를 알아야 조작자가 손을 움직여도 되는지를 안다.
        mode = self.mode_switch.mode
        if mode == CONTACT_PROBING and self.allow_inplane_rotation:
            mode = "contact_probing_inplane"
        msg.data = mode
        self.mode_pub.publish(msg)

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
            self._retreat_gave_up = False

        elapsed = (now - self.retreat_started).nanoseconds / 1e9
        wrench_fresh = (
            self.wrench_stamp is not None
            and (now - self.wrench_stamp).nanoseconds / 1e9 < 0.5
        )

        # 종료 조건도 판정과 같은 스칼라를 본다. F_n 만 보면 프로브가 옆으로 눌린 채
        # 남아 있어도 "풀렸다" 가 되고, 그 상태로 접근 속도가 되살아난다.
        if wrench_fresh and self._control_force() < self.retreat_force:
            return None  # 접촉이 풀렸다
        if elapsed > self.max_retreat:
            # **한 번만 찍는다.** 여기서 None 을 내면 호출자는 속도 0 을 내고
            # 돌아가지만 retreat_started 는 그대로 두므로, twist 가 돌아올 때까지
            # 다음 주기에도 같은 가지로 들어온다. 예전에는 그때마다 error 를 찍어서
            # 100 Hz 로 같은 줄이 쏟아졌다.
            #
            # 그게 왜 문제인가: 2026-09-04 에 touch_twist 가 장치를 못 잡고 죽었는데,
            # 그 사실을 말하는 유일한 줄("No haptic devices found")이 몇 초 만에
            # 수천 줄 밑으로 밀려났다. 조작자에게 남은 화면은 로봇이 무엇을 못 하는지
            # 반복하는 문장뿐이고, **왜** 그런지는 스크롤 밖에 있었다.
            #
            # 로봇은 이 상태에서 이미 정지해 있다. 반복해서 알릴 새 소식이 없다.
            if not self._retreat_gave_up:
                self.get_logger().error(
                    f"후퇴 시간 상한 {self.max_retreat} s 도달 — 정지한다. "
                    f"{'힘이 안 떨어진다' if wrench_fresh else 'wrench 도 두절이다'}. "
                    "twist 가 돌아올 때까지 이 자리에 선다 — 상류(touch_twist)를 보라."
                )
                self._retreat_gave_up = True
            return None

        if not self._retreat_announced:
            self.get_logger().warn(
                f"twist 두절 — 프로브 −z 로 {self.retreat_speed * 1000:.0f} mm/s 후퇴"
            )
            self._retreat_announced = True
        return np.array([0.0, 0.0, -self.retreat_speed, 0.0, 0.0, 0.0])

    def _apply_force_regulation(self, twist: np.ndarray) -> np.ndarray:
        """접촉 프로빙에서 침투축을 로봇이 잡는다.

        접근 모드에서는 아무것도 하지 않는다 — 그때 z 는 조작자의 것이다.

        접촉 프로빙에서는 **z 를 조절기가 덮어쓴다.** 조작자가 누르는 깊이를 맞추는
        것이 아니라, 목표 힘 구간에 들어오도록 로봇이 스스로 움직인다. 나머지 다섯
        축은 기본적으로 0 이다 (``contact_control.allow_teleop_lateral`` 로 켤 수 있다)
        — policy 가 영상축을 맡기 전까지는 미끄러뜨릴 주체가 없고, 힘을 잡는 동안
        옆으로 흐르면 그 힘이 무엇에 대한 힘인지 알 수 없게 된다.

        **데드맨은 그대로 살아 있다.** 이 함수는 twist 가 신선할 때만 불린다. 조작자가
        손을 놓으면 상위 워치독이 후퇴를 맡고 여기는 아예 안 온다 — 자율 모드라고
        해서 조작자가 멈출 수단을 잃지는 않는다.

        wrench 가 오래되면 조절하지 않는다. 힘을 모르는 채로 힘을 잡을 수는 없다.
        """
        if not self.mode_switch.in_contact_probing:
            return twist

        # 전환이 꺼져 있으면 모드가 접촉 프로빙이 될 수 없으므로 위에서 이미
        # 돌아갔다. 그래도 남겨 두되 **twist 를 돌려준다** — 예전에는 여기서 값 없이
        # return 해 None 이 나갔고, 그 None 은 제어 루프의 솔버까지 가서 터진다.
        # 도달하지 않는 줄이라 증상이 없었을 뿐, 조건 하나가 바뀌면 정지가 아니라
        # 예외로 나타날 자리였다.
        if not self.contact_probing_enabled:
            return twist

        # 교정이 유효하지 않으면 힘 기반 동작을 열지 않는다. 보정되지 않은 값으로
        # 힘을 잡으면 자세가 바뀔 때마다 목표가 수 N 씩 어긋난 채로 조직을 민다.
        if self.require_calibration and self.calibration_valid is not True:
            if not self._calibration_blocked_announced:
                self.get_logger().error(
                    "접촉 프로빙인데 교정이 유효하지 않다 — 힘 축을 놓는다. "
                    "콘솔의 Sensor calibration 에서 교정을 마쳐라."
                )
                self._calibration_blocked_announced = True
            self.regulator_reason = "교정 무효 — 힘 제어 차단"
            return np.zeros(6)
        self._calibration_blocked_announced = False

        now = self.get_clock().now()
        wrench_fresh = (
            self.wrench_stamp is not None
            and (now - self.wrench_stamp).nanoseconds / 1e9 < 0.5
        )
        if not wrench_fresh:
            self.regulator_reason = "wrench 두절 — 힘 조절 중단, 정지"
            self.get_logger().error(
                "접촉 프로빙 중 wrench 두절 — 힘 축을 놓는다", throttle_duration_sec=1.0
            )
            return np.zeros(6)

        force = self._control_force()
        out = self.regulator.update(force)

        # 대조군: z 를 잡지 않는다.
        #
        # 조절기를 **부른 뒤에** 덮어쓴다. 그래야 한계 초과 판정이 대조군에서도
        # 그대로 돌고, 넘었을 때는 조절기가 낸 강제 후퇴를 그대로 쓴다. 대조군은
        # "제어를 안 한다" 이지 "안전층을 끈다" 가 아니다 — 팬텀에 물을 넣다 보면
        # 힘은 조작자가 아니라 주사기가 올린다.
        if not self.force_hold_enabled and force < self.regulator.max_force_n:
            out = RegulatorOutput(
                v_z=0.0,
                reason=f"힘 유지 꺼짐 (대조군) — z 고정, 지금 {force:.3f} N",
            )

        self.regulator_reason = out.reason
        self._last_v_z = float(out.v_z)

        regulated = np.zeros(6)
        if self.allow_teleop_lateral:
            regulated = np.array(twist, dtype=float)
            regulated[2] = 0.0
        elif self.allow_inplane_rotation:
            # 면내 회전만 돌려준다. twist 는 이미 접촉 상한으로 묶여 있으므로
            # (`_clamp` 뒤에 불린다) 여기서 다시 제한하지 않는다.
            #
            # **면을 벗어나지 않는다** 는 것이 이 한 줄의 전부다: 영상면은 x–z 이고
            # ω_y 는 그 면을 자기 자신으로 옮긴다. 다른 회전축을 함께 열면 그 성질이
            # 사라지므로, 여기서 twist 를 통째로 복사하지 않는 것이 요점이다.
            regulated[4] = float(twist[4])
        regulated[2] = out.v_z
        return regulated

    def _remap_twist(self, twist: np.ndarray) -> np.ndarray:
        """지령을 표류하지 않는 고정 프레임으로 옮긴다 (fr5_ik.teleop_frame 참조).

        병진과 회전을 각각 켤 수 있다 (``teleop.linear_frame`` ·
        ``teleop.angular_frame``). 둘은 **같은 ``R_latch``** 를 공유하므로, 손을
        대각선으로 움직일 때 병진과 회전이 서로 다른 "오른쪽" 을 갖는 일이 없다.

        후퇴 twist 에는 적용하지 않는다 — 후퇴는 조작자 의도가 아니라 프로브 −z 라는
        기하학적 정의이므로 프로브 프레임 그대로여야 한다.
        """
        want_linear = self.linear_frame == "latched"
        want_angular = self.angular_frame == "latched"
        mirrored = self.mapper.operator_yaw_deg != 0.0
        if not (want_linear or want_angular or mirrored):
            return twist

        if (want_linear or want_angular) and not self.mapper.ready:
            if not self._warned_no_stylus:
                self.get_logger().warn(
                    "stylus_pose 가 아직 없어 예전(프로브 body) 기준으로 낸다. "
                    "데드맨을 한 번 잡으면 기준이 잡힌다. 계속 이 상태라면 "
                    "touch_twist 노드가 떠 있는지 확인하라."
                )
                self._warned_no_stylus = True
            want_linear = want_angular = False

        rot_base_probe = self.solver.rotation_base_probe(self.q)
        out = np.array(twist, dtype=float)
        # 기준을 고정하는 축은 to_probe 로 (조작자 회전이 R_latch 안에 들어 있다),
        # body 로 두는 축은 to_probe_body 로 조작자 회전만 받는다. 한쪽만 뒤집히면
        # 미는 방향과 비트는 방향이 서로 다른 세계에 있게 된다.
        out[:3] = (
            self.mapper.to_probe(out[:3], rot_base_probe, self.rot_stylus)
            if want_linear
            else self.mapper.to_probe_body(out[:3], rot_base_probe)
        )
        out[3:] = (
            self.mapper.to_probe(out[3:], rot_base_probe, self.rot_stylus)
            if want_angular
            else self.mapper.to_probe_body(out[3:], rot_base_probe)
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

        # 접촉 프로빙에서는 twist 두절이 후퇴 사유가 아니다.
        #
        # 이 모드에서 조작자에게 열린 축은 **하나도 없다** — 여섯 축이 모두 0 이고
        # z 는 조절기가 덮어쓴다. 그런데 예전에는 twist 가 0.1 s 끊기면 워치독
        # 분기로 빠져 `_apply_force_regulation` 이 아예 호출되지 않았다. 즉 잡을 축이
        # 없는 조작자가 데드맨을 쥐고 있어야만 로봇이 자기 힘을 유지했고, 놓으면
        # 목표에 도달하는 움직임 자체가 멈췄다.
        #
        # 힘 축의 보호는 데드맨이 아니라 조절기 안에 있다 — 한계 초과 시 강제 후퇴,
        # 교정 무효 시 차단, wrench 두절 시 정지. 그 셋은 아래에서 그대로 산다.
        if twist_stale and self.mode_switch.in_contact_probing:
            if not self._force_hold_announced:
                self.get_logger().info(
                    "twist 두절 — 접촉 프로빙이라 힘 조절을 계속한다 (조작자 축 없음)"
                )
                self._force_hold_announced = True
            twist = self._apply_force_regulation(np.zeros(6))
            self.retreat_pub.publish(Bool(data=False))
            twist_stale = False

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
            self._retreat_gave_up = False
            self._force_hold_announced = False
            twist = self._clamp(self.twist_cmd) * self._hold_fade(now)
            twist = self._remap_twist(twist)
            twist = self._apply_force_regulation(twist)
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

        probing = self.mode_switch.in_contact_probing
        self.force_diag_pub.publish(Float32MultiArray(data=[
            float(self._control_force()),
            float(self.regulator.target_force_n),
            float(self._last_v_z if probing else 0.0),
            float(error.per_axis[2]),
            1.0 if probing else 0.0,
        ]))
        # 사유는 바뀔 때만 낸다 — 100 Hz 로 같은 문자열을 흘리면 로그가 아니라 잡음이다.
        if self.regulator_reason != self._last_reason:
            self._last_reason = self.regulator_reason
            self.force_reason_pub.publish(String(data=self.regulator_reason))

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
    except (KeyboardInterrupt, ExternalShutdownException):
        # 후자도 Ctrl-C 다. launch 아래에서는 SIGINT 를 rclpy 의 전역 처리기가 먼저
        # 받아 컨텍스트를 내리고, 그러면 spin 은 KeyboardInterrupt 가 아니라
        # ExternalShutdownException 을 낸다. 그것을 안 잡으면 **정상 종료 때마다**
        # 트레이스백과 exit 1 이 나오고, launch 는 "process has died" 로 적는다.
        #
        # 조용한 종료가 목적이 아니다. 매번 나오는 트레이스백은 조작자에게
        # 종료 로그를 읽지 않는 습관을 만들고, 그러면 진짜 종료 실패
        # (us_servo 의 "종료 절차 예외", 접촉 중 홈잉 거부)가 같은 소음에 묻힌다.
        pass
    except RuntimeError as exc:
        print(f"[us_diff_ik_node] 기동 거부: {exc}")
    finally:
        # 정리 중에 신호가 한 번 더 오면 (Ctrl-C 연타, 또는 launch 의 신호와
        # 터미널의 신호가 겹칠 때) KeyboardInterrupt 가 **이 블록 안에서** 뜬다.
        # except 절은 이미 지나갔으므로 그때는 아무도 안 잡고, 정상 종료가 다시
        # 트레이스백으로 끝난다. 정리 도중의 중단은 그 자체로 소식이 아니다.
        try:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
