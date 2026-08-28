"""ROS 2 → 웹소켓 텔레메트리 브리지. teleop 콘솔 GUI 가 붙는 자리다.

    ros2 run fr5_control telemetry_bridge
    ros2 run fr5_control telemetry_bridge --ros-args -p bridge.px6d_port:=/dev/ttyACM0

GUI 는 자기 힘으로 로봇 상태를 알 방법이 없다. FR5 컨트롤러는 자체 XML-RPC/UDP 만
말하고 웹소켓도 ROS 브리지도 열지 않는다. 이 노드가 그 간극을 메운다 — 이미 돌고 있는
Phase 0 스택의 토픽을 구독해 GUI 의 계약 형식(JSON)으로 밀어 준다.

**로봇을 움직이는 명령은 받지 않는다.** 소켓으로 들어오는 것 중 이 노드가 처리하는
것은 **교정 명령뿐**이고, 그중 어느 것도 로봇을 움직이지 않는다 — 표본을 모으고,
모델을 풀고, 프로파일을 저장할 뿐이다. 자세 이동은 조작자가 손으로 한다 (§7).

원래 이 브리지는 완전 단방향이었다. 교정 절차가 "지금 자세에서 모아라" 를 어딘가에서
말해야 해서 좁은 채널을 열었고, 그 경계는 **동작 명령 없음**으로 유지한다. 속도 클램프와
워치독을 가진 제어 스택이 로봇을 움직이는 유일한 경로라는 성질은 그대로다.

구독:
  {ns}/joint_states   sensor_msgs/JointState     위치·속도·토크
  {ns}/ee_wrt_base    geometry_msgs/Pose         TCP 자세
  {ns}/wrench         geometry_msgs/WrenchStamped  컨트롤러 경유 F/T

**PX6D 는 컨트롤러를 거치지 않는다.** USB 변형이라 `robot_state_pkg.ft_sensor_data`
에 값이 안 들어오고, 그 결과 `{ns}/wrench` 는 보통 비어 있다. `bridge.px6d_port` 를
주면 이 노드가 센서를 직접 시리얼로 읽어 그 값을 쓴다 — 지금으로서는 그것이 유일한
실제 힘 경로다 (DESIGN_NOTES §2.1 · [[paxini-px6d-ft-sensor]]).

기동하면 어떤 소스가 살아 있는지 로그로 찍는다. GUI 가 "연결됐는데 값이 없다" 를
보여줄 때 여기부터 본다.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, WrenchStamped
from std_msgs.msg import Bool, Float64, String
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSPresetProfiles, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

from fr5_control.calibration_service import (
    AUTO_MIN_SEPARATION_DEG,
    CalibrationSession,
    StillnessDetector,
)
from fr5_control.wrench_calibration import CalibrationPose, MIN_POSES
from fr5_control.us_servo_node import JOINT_NAMES
from fr5_control.wrench_frames import FrameRegistration, GRAVITY_B
from fr5_control.wrench_profile import CalibrationProfile, compensate

#: 관절이 이보다 느리면 정지로 본다 [rad/s]. 컨트롤러는 teleop 여부를 말해 주지
#: 않으므로 움직임으로 추론한다. 추론이라는 사실은 프레임에 함께 실어 보낸다.
_IDLE_VEL_RAD_S = 0.002


class TelemetryBridge(Node):
    """ROS 토픽을 모아 웹소켓으로 밀어 주는 노드."""

    def __init__(self) -> None:
        """노드를 만들고 구독·서버를 띄운다."""
        super().__init__("fr5_telemetry_bridge")

        self.declare_parameter("robot.name", "fr5_right")
        self.declare_parameter("bridge.host", "0.0.0.0")
        self.declare_parameter("bridge.port", 8765)
        self.declare_parameter("bridge.telemetry_hz", 30.0)
        # 화면용 렌치 발행 주기. 센서는 1 kHz 지만 화면은 초당 한 번이면 된다 —
        # 조작자가 읽는 것은 "지금 몇 뉴턴인가" 이지 그 안의 파형이 아니고, 더 자주
        # 바꿔 봐야 읽히지 않는 자릿수만 흔들린다.
        #
        # **제어와는 무관하다.** us_diff_ik 는 ROS 토픽으로 전 표본을 받는다.
        # 접촉 판정과 힘 제어는 그쪽에서 1 kHz 로 돈다.
        #
        # 각 프레임은 그 1 초의 **평균** 이다. 평균만 보내면 그 사이의 스파이크가
        # 사라지므로, 같은 창의 **극값** 도 함께 보낸다 (아래 forceExtremes).
        self.declare_parameter("bridge.wrench_hz", 1.0)
        # 그래프용 파형의 구간 수 [Hz]. **읽는 숫자와 그리는 선은 다른 문제다.**
        #
        # 위의 1 Hz 는 조작자가 읽는 숫자에 맞춘 값이다 — 더 자주 바꾸면 안 읽힌다.
        # 그런데 같은 주기로 선까지 그리면 20 초 그래프가 점 스무 개짜리 계단이 되고,
        # 그러면 1 kHz 센서를 달아 놓고 초당 한 점만 보는 셈이다.
        #
        # 그래서 프레임 주기는 그대로 두고, 그 한 장 안에 **그 1 초의 파형**을 접어
        # 보낸다 (아래 forceWaveform). 구간마다 [평균, 최소, 최대] 라 구간 안의
        # 스파이크도 남는다 — 화면 폭이 20 000 점을 그릴 수 없으므로 표본을 골라
        # 버리는 대신 접는 것이다. 고르면 앨리어싱이고, 접으면 아니다.
        #
        # 0 이하면 끈다. 100 Hz 면 구간 10 ms, 초당 100 구간이다.
        self.declare_parameter("bridge.wrench_waveform_hz", 100.0)
        self.declare_parameter("bridge.px6d_port", "")

        # 로봇 관절각을 컨트롤러에서 **읽기로만** 가져온다. bridge.px6d_port 와
        # 대칭인 기능이다.
        #
        # 교정 때문에 있다. 중력 식별은 조작자가 로봇을 손으로 여러 자세에 옮기며
        # 하는데, 그러려면 드래그 모드를 켜야 하고 그동안 us_servo 가 ServoJ 를
        # 쏘고 있으면 서로 싸운다. 그렇다고 us_servo 를 끄면 {ns}/joint_states 가
        # 끊겨 자세를 알 수 없다 — 자세를 모르면 중력 식별 자체가 불가능하다.
        #
        # 그래서 브리지가 직접 읽는다. **명령은 보내지 않는다** — 연결과 상태
        # 패키지 읽기가 전부다. "교정 절차는 로봇을 자동으로 움직이지 않는다" 는
        # 규칙이 여기서도 유지된다.
        self.declare_parameter("bridge.robot_ip", "")
        self.declare_parameter("bridge.robot_poll_hz", 20.0)
        self.declare_parameter("bridge.px6d_baud", 921600)
        # PX6D 직결 값을 ROS 로도 낸다. 제어 스택이 실제 힘을 볼 수 있는 유일한 길이다.
        #
        # us_servo 가 이미 {ns}/wrench 를 내므로 **다른 토픽**을 쓴다. 같은 토픽에
        # 발행자가 둘이면 구독자는 컨트롤러의 0 과 PX6D 값을 번갈아 받게 되고, 그
        # 섞임은 로그에 드러나지 않는다.
        self.declare_parameter("bridge.publish_wrench", True)
        self.declare_parameter("bridge.wrench_topic", "wrench_px6d")

        # 교정 프로파일. 저장·되읽기 경로이며, 없으면 원값을 그대로 낸다 —
        # 교정이 없는데 있는 척하지 않는다.
        self.declare_parameter("bridge.calibration_path", "")
        # ---- 적층 기하 — probe.yaml 에서 온다 ✅ 2026-08-27 -------------------
        #
        # 예전에는 여기서 registration.mounting_angle_deg / r_sensor_to_probe_m /
        # flange_to_sensor_rpy 를 **따로** 선언했다. 그런데 이 노드는 ros2 run 으로
        # 뜨느라 probe.yaml 을 받지 않았고, 그래서 세 값이 전부 노드 기본값으로만
        # 돌았다. 결과가 무증상 버그였다 — r_sensor_to_probe_m 이 0 이라 모멘트
        # 기준점 이동이 꺼진 채로 오래 돌았는데, 축방향 압축에서는 r ∥ f 라 차이가
        # 0 이어서 화면으로는 정상으로 보였다.
        #
        # 이제 셋 다 **유도한다** (FrameRegistration.from_stack). 독립된 설정값이
        # 아니라 적층 사슬 위의 서로 다른 구간이므로, 유도하면 어긋날 수 없다.
        # 값을 고칠 곳은 probe.yaml 하나다.
        #
        # 기본값이 .nan 인 것은 의도다. 이 노드는 파라미터 파일 없이 뜨면 기하를
        # 모르는 채로 도는데, 그 상태에서 0 을 대신 쓰면 위와 같은 무증상 오류가
        # 다시 생긴다. 없으면 기동을 거부하고 무엇을 하라는지 말한다.
        self.declare_parameter("tool.j6_to_probe_xyz", [float("nan")] * 3)
        self.declare_parameter("tool.j6_to_probe_rpy", [float("nan")] * 3)
        self.declare_parameter("ft_sensor.j6_to_sensor_xyz", [float("nan")] * 3)
        self.declare_parameter("ft_sensor.j6_to_sensor_rpy", [float("nan")] * 3)

        self.robot_name = self.get_parameter("robot.name").value
        ns = f"/{self.robot_name}"

        self._lock = threading.Lock()
        self._joints: JointState | None = None
        self._joints_at = 0.0
        self._pose: Pose | None = None
        self._wrench: tuple[float, ...] | None = None
        self._wrench_at = 0.0
        self._wrench_source = "none"
        # PX6D 직결일 때만 채워진다. 회선 품질을 GUI 가 그대로 보여 준다 —
        # "값이 이상하다" 와 "선이 이상하다" 를 조작자가 가를 수 있어야 한다.
        self._px6d_crc = 0
        self._px6d_hz = 0.0

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(JointState, f"{ns}/joint_states", self._on_joints, sensor_qos)
        self.create_subscription(Pose, f"{ns}/ee_wrt_base", self._on_pose, 10)
        self.create_subscription(WrenchStamped, f"{ns}/wrench", self._on_wrench, sensor_qos)

        self._pose_quat = None          # {B}R{F} 를 만들 원본

        # 자동 캡처 — 드래그가 멈추면 자세를 잡는다.
        self.auto_capture = False
        self.auto_seconds = 2.0
        self.auto_note = ""
        self.stillness = StillnessDetector()

        # 화면용 렌치는 **구간 평균**으로 낸다.
        #
        # 센서는 1 kHz 로 오는데 화면은 그보다 훨씬 느리게 그린다. 최신 표본 하나만
        # 골라 보내면 스무 개 중 하나를 임의로 뽑는 셈이라 값이 튄다 — 앨리어싱이고,
        # 조작자에게는 "숫자가 정신없다" 로 보인다. 구간을 평균하면 같은 주기로도
        # 값이 가라앉는다. 잃는 것은 그 구간 안의 스파이크인데, 그것은 애초에
        # 화면으로 볼 수 있는 것이 아니다 (한계 감시는 제어 스택이 전 표본으로 한다).
        self._display_sum = np.zeros(6)
        self._display_count = 0
        #: 같은 창의 힘 극값. 평균이 지운 스파이크를 피크 표시가 잃지 않게 한다.
        self._display_min = np.full(3, np.inf)
        self._display_max = np.full(3, -np.inf)

        # 그래프용 파형 누적기. 위의 평균이 "지금 몇 뉴턴인가" 라면 이쪽은 "그 1 초가
        # 어떤 모양이었나" 다. 표본이 들어오는 대로 구간에 접어 두고, 프레임을 낼 때
        # 모아 둔 구간을 통째로 실어 보낸다.
        self._wave_hz = float(self.get_parameter("bridge.wrench_waveform_hz").value)
        self._wave_dt = 1.0 / self._wave_hz if self._wave_hz > 0.0 else 0.0
        #: 닫힌 구간들 — (구간 끝 시각, 평균 f[3], 최소 f[3], 최대 f[3]).
        self._wave_bins: list[tuple] = []
        #: 채우는 중인 구간 — [마감 시각, 합 f[3], 표본 수, 최소 f[3], 최대 f[3]].
        self._wave_open: list | None = None
        # GUI 가 붙어 있지 않으면 아무도 비워 주지 않는다. 5 초어치에서 오래된 것부터
        # 버린다 — 늦게 붙은 화면에 5 초 전 파형을 밀어 넣어 봐야 읽을 것이 없다.
        self._wave_cap = int(self._wave_hz * 5.0) if self._wave_hz > 0.0 else 0
        self.session = CalibrationSession()
        self.calibration_path = str(self.get_parameter("bridge.calibration_path").value) or (
            os.path.expanduser("~/.ros/fr5_px6d_calibration.json")
        )
        self.pose_log_path = self.calibration_path.replace(".json", "") + "_poses.jsonl"
        self.registration = self._registration_from_params()
        self.profile = self._load_profile()

        self._mode = None
        # 발행자가 TRANSIENT_LOCAL 이므로 구독도 맞춘다. 안 맞추면 QoS 불일치로
        # 아예 연결되지 않는다 — 조용히, 오류 없이.
        self.create_subscription(
            String,
            f"{ns}/probing_mode",
            self._on_mode,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        # 조작자 위치(미러) 상태. us_diff_ik 가 latched 로 내므로 늦게 떠도 즉시 받는다.
        self._teleop_frame = None
        self.create_subscription(
            String,
            f"{ns}/teleop_frame",
            self._on_teleop_frame,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        # 콘솔이 그것을 바꾸는 길.
        #
        # 이 노드가 로봇을 **움직이는** 것은 여전히 아니다 — 조작자가 손으로 내는
        # 지령을 어느 방향으로 읽을지만 바꾼다. 그래도 안전과 무관하지 않아서,
        # 적용은 us_diff_ik 가 다음 파지 경계까지 미룬다. 여기서 즉시 반영하면
        # 조작 중에 축이 뒤집힐 수 있다.
        self.teleop_frame_req = self.create_publisher(
            Float64,
            f"{ns}/teleop_frame_request",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self.wrench_pub = None
        if bool(self.get_parameter("bridge.publish_wrench").value):
            topic = f"{ns}/{self.get_parameter('bridge.wrench_topic').value}"
            self.wrench_pub = self.create_publisher(WrenchStamped, topic, 10)
            self._wrench_frame = f"{self.robot_name}_ft_sensor"
            self.get_logger().info(f"PX6D wrench 발행: {topic}")

        # 제어 스택이 교정 유효성을 알아야 힘 기반 동작을 열지 말지 정한다.
        # latched: 늦게 뜨는 IK 노드가 즉시 현재 상태를 받는다.
        self.calib_pub = self.create_publisher(
            Bool, f"{ns}/calibration_valid",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_timer(1.0, self._publish_calibration_valid)

        self.get_logger().info(
            f"구독: {ns}/joint_states · {ns}/ee_wrt_base · {ns}/wrench · {ns}/probing_mode"
        )

        robot_ip = str(self.get_parameter("bridge.robot_ip").value)
        if robot_ip:
            self._start_robot_reader(
                robot_ip, float(self.get_parameter("bridge.robot_poll_hz").value)
            )

        px6d_port = str(self.get_parameter("bridge.px6d_port").value)
        if px6d_port:
            self._start_px6d(px6d_port, int(self.get_parameter("bridge.px6d_baud").value))

        self._start_server()

    # -- 적층 기하 ---------------------------------------------------------

    def _registration_from_params(self) -> FrameRegistration:
        """probe.yaml 의 적층 값에서 등록을 유도한다.

        Returns:
            :class:`FrameRegistration`. 네 항목이 서로 정합적임이 구조적으로 보장된다.

        Raises:
            RuntimeError: 적층 값이 없거나(.nan) 장착각 하나로 표현되지 않을 때.
                조용히 0 으로 돌지 않는다 — 그 조합이 정확히 이 노드가 오래 앓던
                무증상 오류다.
        """
        try:
            reg = FrameRegistration.from_stack(
                self.get_parameter("ft_sensor.j6_to_sensor_xyz").value,
                self.get_parameter("ft_sensor.j6_to_sensor_rpy").value,
                self.get_parameter("tool.j6_to_probe_xyz").value,
                self.get_parameter("tool.j6_to_probe_rpy").value,
            )
        except ValueError as exc:
            # 경로를 못 찾는다고 여기서 또 죽으면 안 된다 — 오류 처리기가 내는 오류는
            # 원래 오류를 가리고, 조작자는 엉뚱한 것을 쫓게 된다.
            try:
                share = get_package_share_directory("fr5_control")
            except Exception:  # noqa: BLE001
                share = "<workspace>/install/fr5_control/share/fr5_control"
            raise RuntimeError(
                f"적층 기하를 읽지 못했다: {exc}\n"
                "이 노드는 probe.yaml 을 받아야 한다. 파라미터 파일을 붙여 다시 띄워라:\n"
                f"  ros2 run fr5_control telemetry_bridge --ros-args "
                f"--params-file {share}/config/probe.yaml"
            ) from exc

        self.get_logger().info(
            f"적층 기하 (probe.yaml 유도): 장착각 {reg.mounting_angle_deg:+.1f}° · "
            f"레버암 {np.linalg.norm(reg.r_sensor_to_probe_m) * 1000.0:.0f} mm · "
            f"플랜지→센서 yaw {math.degrees(reg.flange_to_sensor_rpy[2]):+.1f}°"
            + (" · 뒤집힘" if reg.axial_flip else "")
        )
        return reg

    # -- 교정 프로파일 -----------------------------------------------------

    def _load_profile(self):
        """저장된 교정을 되읽는다. 없거나 깨졌으면 ``None`` 이다.

        읽지 못한 것을 조용히 넘기지 않는다 — 교정이 없는데 보상된 값처럼 보이는
        것이 이 시스템에서 가장 위험한 실패다.
        """
        try:
            profile = CalibrationProfile.load(self.calibration_path)
        except FileNotFoundError:
            self.get_logger().warn(
                f"교정 파일 없음: {self.calibration_path} — 원값을 그대로 낸다"
            )
            return None
        except Exception as exc:
            self.get_logger().error(f"교정 파일을 읽지 못했다 ({exc}) — 원값을 그대로 낸다")
            return None

        valid, issues = profile.validity()
        age_h = profile.age_s() / 3600.0
        if valid:
            self.get_logger().info(
                f"교정 적재 — {age_h:.1f} 시간 전, 질량 "
                f"{profile.gravity.mass_kg * 1000:.0f} g, 장착각 "
                f"{profile.registration.mounting_angle_deg:.1f}°"
            )
        else:
            self.get_logger().warn(
                f"교정이 유효하지 않다 ({'; '.join(issues)}) — 값은 내되 접촉 제어는 막힌다"
            )
        # 등록은 프로파일 것을 따른다. 파라미터는 새 교정을 시작할 때의 초기값이다.
        self.registration = profile.registration
        return profile

    def _tare_off_axis_deg(self):
        """작업 영점을 잰 자세에서 지금 몇 도 떨어졌는가. 영점이 없으면 ``None``."""
        if self.profile is None or self.profile.tare_flange_z is None:
            return None
        now = self._flange_z_component()
        if now is None:
            return None
        cos_a = max(-1.0, min(1.0, float(now) * float(self.profile.tare_flange_z)))
        return round(math.degrees(math.acos(cos_a)), 1)

    def _flange_z_component(self):
        """플랜지 z 축의 베이스 z 성분. 자세를 모르면 ``None``."""
        rot = self.rotation_base_flange()
        return None if rot is None else float(rot[2, 2])

    def gravity_in_flange(self, rot_base_flange):
        """``{F}g`` — 순기구학만으로 나오는 값.

        ``gravity_in_sensor`` 와 달리 플랜지→센서 회전이 섞이지 않는다. 그 회전은
        적합이 함께 푸는 미지수이므로, 자세를 기록할 때는 가정이 안 섞인 이쪽을
        같이 남긴다.
        """
        return np.asarray(rot_base_flange, dtype=float).T @ GRAVITY_B

    def rotation_base_flange(self):
        """``{B}R{F}``. 자세도 관절각도 없으면 ``None``.

        자세 토픽을 먼저 쓰고, 없으면 **관절각에서 순기구학으로 만든다.**

        예전에는 자세 토픽만 봤다. 그런데 그 토픽은 us_servo 가 발행하므로,
        제어 스택 없이 도는 교정 모드에서는 영영 오지 않았고 ``calib.pose`` 가
        전부 거절됐다 — 관절각은 멀쩡히 들어오고 있는데도. 자세의 출처가 하나로
        묶여 있으면 그 하나가 없을 때 통째로 못 쓰게 된다.
        """
        with self._lock:
            quat = self._pose_quat
            joints = self._joints
        if quat is not None:
            return Rotation.from_quat(quat).as_matrix()
        if joints is not None and len(joints.position) >= 6:
            from fr5_control.robot_backend import fr5_forward_kinematics

            return fr5_forward_kinematics(list(joints.position)[:6])[:3, :3]
        return None

    # -- 구독 콜백 ---------------------------------------------------------

    def _on_joints(self, msg: JointState) -> None:
        with self._lock:
            self._joints = msg
            self._joints_at = time.time()

    def _on_pose(self, msg: Pose) -> None:
        with self._lock:
            self._pose = msg
            o = msg.orientation
            self._pose_quat = [o.x, o.y, o.z, o.w]

    def _on_mode(self, msg: String) -> None:
        """로봇이 선언한 프로빙 모드. 콘솔이 추측하지 않게 그대로 전달한다."""
        if msg.data != self._mode:
            self.get_logger().info(f"프로빙 모드: {msg.data}")
        self._mode = msg.data

    def _on_teleop_frame(self, msg: String) -> None:
        """us_diff_ik 가 선언한 teleop 매핑. 콘솔이 추측하지 않게 그대로 전달한다."""
        try:
            frame = json.loads(msg.data)
        except (TypeError, ValueError):
            self.get_logger().error(f"teleop_frame 을 읽지 못했다: {msg.data!r}")
            return
        if self._teleop_frame != frame:
            self.get_logger().info(
                f"teleop 매핑: 조작자 {frame.get('operatorYawDeg')}°"
                + (f" (대기 {frame.get('pendingYawDeg')}°)"
                   if frame.get("pendingYawDeg") is not None else "")
            )
        self._teleop_frame = frame

    def _on_wrench(self, msg: WrenchStamped) -> None:
        # 컨트롤러 경유 값은 PX6D 직결이 없을 때만 쓴다. 둘 다 있으면 직결이 이긴다 —
        # 우리 개체는 컨트롤러에 물려 있지 않으므로 그쪽 값은 0 이거나 남의 것이다.
        if self._wrench_source == "px6d_serial":
            return
        w = msg.wrench
        with self._lock:
            self._wrench = (w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z)
            self._wrench_at = time.time()
            self._wrench_source = "controller"

    # -- PX6D 직결 ---------------------------------------------------------

    def _auto_capture_step(self) -> None:
        """드래그가 멈추면 자세를 자동으로 잡는다.

        조작자가 자세마다 버튼을 누르려면 한 손이 묶이는데, 그 손은 로봇을 붙잡고
        있어야 한다. 그래서 멈춘 것을 보고 잡는다.

        **로봇을 움직이지 않는다.** 관절각을 보고 수집을 시작할 뿐이다.
        """
        if not self.auto_capture or self.session.capture.active:
            return
        with self._lock:
            joints = self._joints
        if joints is None or len(joints.position) < 6:
            return

        degrees = [float(v) * 57.29577951308232 for v in list(joints.position)[:6]]
        if self.stillness.update(degrees, time.time()) != "settled":
            return

        rot = self.rotation_base_flange()
        if rot is None:
            return
        gravity = self.registration.gravity_in_sensor(rot)

        separation = self.session.separation_deg(gravity)
        if separation < AUTO_MIN_SEPARATION_DEG:
            # 비슷한 자세를 열두 번 잡아도 커버리지는 오르지 않는다. 조용히 넘기지
            # 말고 무엇이 모자란지 말한다 — 조작자는 "왜 안 잡히지" 를 알아야 한다.
            self.session.last_error = (
                f"직전 자세들과 {separation:.0f}° 밖에 안 떨어졌다 "
                f"({AUTO_MIN_SEPARATION_DEG:.0f}° 이상 필요) — 더 크게 돌려라"
            )
            return

        label = f"auto {len(self.session.poses) + 1}"
        self.session.start_pose(
            label, gravity, self.auto_seconds,
            gravity_flange=self.gravity_in_flange(rot),
        )
        self.session.last_error = ""
        self.get_logger().info(
            f"자동 캡처: {label} — 정지 감지 (직전 자세들과 {separation:.0f}°)"
        )

    def _write_profile(self, registration, note: str):
        """지금 세션 상태로 프로파일을 쓴다.

        수동 저장과 자동 저장이 **같은 경로**를 쓰게 하려고 뺐다. 둘이 갈라지면
        "손으로 저장한 것" 과 "저절로 저장된 것" 이 다른 물건이 되고, 나중에
        어느 쪽으로 저장했는지에 따라 결과가 달라진다.

        Returns:
            ``(profile, valid, issues)``. 저장에 실패하면 예외가 그대로 올라간다.
        """
        with self._lock:
            pose_deg = list(self._joints.position) if self._joints else []
        profile = CalibrationProfile(
            registration=registration,
            bias=self.session.bias_result,
            gravity=self.session.gravity_model,
            robot_pose_deg=[float(v) * 57.29577951308232 for v in pose_deg],
            mounting_note=note,
        )

        # 직전 프로파일을 남겨 둔다. 자동 저장은 조작자가 누르지 않아도 덮어쓰므로,
        # 새로 쓴 것이 더 나쁠 때 돌아갈 자리가 없으면 안 된다.
        try:
            if os.path.exists(self.calibration_path):
                os.replace(self.calibration_path, self.calibration_path + ".prev")
        except OSError as exc:  # 백업 실패가 저장을 막을 이유는 없다
            self.get_logger().warn(f"직전 교정 백업 실패: {exc}")

        profile.save(self.calibration_path)
        self.profile = profile
        self.registration = registration
        valid, issues = profile.validity()
        return profile, valid, issues

    def _append_pose_log(self, pose) -> None:
        """잡은 자세를 **즉시** 파일에 덧붙인다.

        세션 상태는 이 프로세스의 메모리에만 있다. 실제로 열두 자세를 잡아 둔
        브리지를 종료해야 코드를 고칠 수 있는 상황이 생겼고, 그때 그 자세들을
        꺼낼 방법이 없었다 — 다시 잡는 것 말고는. 한 줄씩 덧붙이는 파일이면
        중간에 죽어도 잡은 데까지는 남는다.

        저장 실패가 수집을 막지는 않는다. 기록은 보조 수단이고, 진행 중인 절차를
        기록 문제로 끊으면 그것이 더 큰 손실이다.
        """
        try:
            with open(self.pose_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "at": time.time(),
                    "label": pose.label,
                    "wrench": [float(v) for v in np.asarray(pose.wrench).reshape(6)],
                    "gravity_sensor": [
                        float(v) for v in np.asarray(pose.gravity_sensor).reshape(3)
                    ],
                    "gravity_flange": (
                        None if pose.gravity_flange is None
                        else [float(v) for v in np.asarray(pose.gravity_flange).reshape(3)]
                    ),
                }, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.get_logger().warn(f"자세 기록 실패: {exc}")

    def _load_pose_log(self) -> list:
        """기록 파일의 자세를 :class:`CalibrationPose` 로 되살린다.

        플랜지 중력이 없는 줄은 건너뛴다 — 그 줄은 정렬을 함께 풀 수 없는 옛
        기록이고, 섞으면 지금 자세들까지 옛 가정으로 끌어내린다.
        """
        out = []
        if not os.path.exists(self.pose_log_path):
            return out
        with open(self.pose_log_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("gravity_flange") is None:
                    continue
                out.append(CalibrationPose(
                    gravity_sensor=np.asarray(row["gravity_sensor"], dtype=float),
                    gravity_flange=np.asarray(row["gravity_flange"], dtype=float),
                    wrench=np.asarray(row["wrench"], dtype=float),
                    label=str(row.get("label", "")),
                ))
        return out

    def _auto_fit_and_save(self) -> None:
        """자세가 충분히 모이면 스스로 풀고 저장한다.

        자동 캡처를 켜 두면 조작자의 두 손은 로봇에 있다. 그 상태에서 화면으로
        돌아와 Fit·Save 를 누르라고 하면 자동 캡처의 이유가 없어진다.

        **유효할 때만 쓴다.** 유효하지 않은 적합은 남기지 않고, 왜 아직 아닌지는
        세션 상태로 계속 보인다. 자세를 더 잡으면 다시 풀어 다시 쓴다 — 자세가
        늘수록 모델이 나아지므로 마지막에 쓴 것이 가장 좋은 것이다.
        """
        if not self.auto_capture or len(self.session.poses) < MIN_POSES:
            return
        if not self.session.fit():
            return
        try:
            _, valid, issues = self._write_profile(self.registration, self.auto_note)
        except Exception as exc:  # noqa: BLE001 - 저장 실패가 세션을 끝내면 안 된다
            self.session.last_error = f"자동 저장 실패: {exc}"
            self.get_logger().error(self.session.last_error)
            return
        model = self.session.gravity_model
        self.get_logger().info(
            f"자동 저장 — 자세 {model.poses} 개 · 커버리지 {model.coverage:.3f} · "
            f"질량 {model.mass_kg * 1000:.0f} g · 잔차 {model.rms_force_n:.3f} N · "
            f"{'유효' if valid else '무효: ' + '; '.join(issues)}"
        )

    def _fit_issues(self) -> list:
        """적합이 왜 안 됐는지 — **비어 있으면 안 된다**.

        모델이 나왔으면 그 모델의 문제 목록을, 모델조차 못 만들었으면 세션이 남긴
        사유를 낸다. 예전에는 후자에서 빈 목록이 나가 화면에 "유효하지 않음" 만
        뜨고 이유가 없었다 — 조작자가 무엇을 고쳐야 할지 알 수 없었다.
        """
        model = self.session.gravity_model
        if model is not None:
            return list(model.issues)
        if self.session.last_error:
            return [self.session.last_error]
        return ["중력 모델을 풀지 못했다 (사유 불명)"]

    def _start_robot_reader(self, ip: str, poll_hz: float) -> None:
        """관절각을 컨트롤러에서 직접 **읽는** 스레드를 띄운다.

        us_servo 가 없어도 자세를 알 수 있게 한다. 교정 중에는 조작자가 드래그
        모드로 로봇을 옮기므로 서보 제어 루프가 돌면 안 되고, 그러면
        ``{ns}/joint_states`` 도 끊긴다.

        **명령 경로는 없다.** 이 스레드가 하는 일은 상태 패키지를 읽어
        :class:`sensor_msgs.msg.JointState` 로 바꿔 담는 것뿐이다.
        """
        thread = threading.Thread(
            target=self._robot_loop, args=(ip, max(1.0, poll_hz)),
            name="robot-read", daemon=True,
        )
        thread.start()
        self.get_logger().info(f"로봇 읽기 전용 연결 시도: {ip} @ {poll_hz:.0f} Hz")

    def _robot_loop(self, ip: str, poll_hz: float) -> None:
        """상태 패키지를 주기적으로 읽어 관절각을 갱신한다."""
        from fr5_control.robot_backend import FairinoBackend, fr5_forward_kinematics

        try:
            backend = FairinoBackend(ip)
            backend.connect()
        except Exception as exc:  # noqa: BLE001 - 어떤 실패든 브리지는 살아야 한다
            self.get_logger().error(f"로봇 읽기 연결 실패 ({ip}): {exc}")
            return
        self.get_logger().info(f"로봇 읽기 전용 연결됨: {ip} — 명령은 보내지 않는다")

        period = 1.0 / poll_hz
        while rclpy.ok():
            try:
                positions_deg = backend.joint_positions_deg()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"관절각 읽기 실패: {exc}")
                time.sleep(period)
                continue

            radians = [float(v) * 0.017453292519943295 for v in positions_deg]
            message = JointState()
            message.name = list(JOINT_NAMES)
            message.position = radians

            # **자세도 같이 채운다.** 관절각만 채우면 calib.pose 가 전부 거절된다 —
            # 중력 식별은 {B}R{F} 를 알아야 성립하고, 그 값은 평소 us_servo 가
            # {ns}/ee_wrt_base 로 발행하는데 교정 모드에는 us_servo 가 없다.
            #
            # 컨트롤러의 tool_pose() 를 쓰지 않고 순기구학으로 만든다. tool_pose 는
            # 컨트롤러에 설정된 TCP 를 따르므로 그쪽 설정이 바뀌면 조용히 달라진다.
            # 순기구학은 base → J6 로 정의가 하나뿐이다.
            transform = fr5_forward_kinematics(radians)
            quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
            pose = Pose()
            pose.position.x = float(transform[0, 3])
            pose.position.y = float(transform[1, 3])
            pose.position.z = float(transform[2, 3])
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            with self._lock:
                # 토픽이 살아 있으면 그쪽이 이긴다 — us_servo 가 도는 평소 운전에서
                # 읽기 스레드가 더 낡은 값으로 덮어쓰지 않게 한다.
                fresh = self._joints_at and (time.time() - self._joints_at) < 0.5
                if not fresh:
                    self._joints = message
                    self._joints_at = time.time()
                    self._pose = pose
                    self._pose_quat = [float(v) for v in quat]
            time.sleep(period)

    def _start_px6d(self, port: str, baud: int) -> None:
        """센서를 시리얼로 직접 읽는 스레드를 띄운다."""
        try:
            import serial  # noqa: F401
        except ImportError:
            self.get_logger().error("pyserial 이 없어 PX6D 직결을 건너뛴다")
            return

        thread = threading.Thread(
            target=self._px6d_loop, args=(port, baud), name="px6d", daemon=True
        )
        thread.start()
        self.get_logger().info(f"PX6D 직결 시도: {port} @ {baud}")

    def _px6d_loop(self, port_name: str, baud: int) -> None:
        import serial

        from fr5_control.px6d_protocol import (
            build_command,
            build_set_rate,
            CMD_STREAM_START,
            CMD_STREAM_STOP,
            FrameParser,
            STREAM_START_DATA,
        )

        while rclpy.ok():
            try:
                with serial.Serial(port_name, baud, timeout=0.01) as port:
                    parser = FrameParser()
                    port.write(build_command(CMD_STREAM_STOP, 0x00))
                    time.sleep(0.1)
                    port.reset_input_buffer()
                    port.write(build_set_rate(1000))
                    time.sleep(0.05)
                    port.write(build_command(CMD_STREAM_START, STREAM_START_DATA))
                    port.reset_input_buffer()
                    parser.reset()

                    self.get_logger().info(f"PX6D 스트리밍 시작: {port_name}")
                    with self._lock:
                        self._wrench_source = "px6d_serial"

                    # 스트림을 켜고 여기까지 오는 사이의 재동기는 회선 오류가 아니다.
                    # 첫 프레임이 선 시점을 기준으로 삼는다 (px6d_monitor 와 같은 규약).
                    crc_base = None
                    stamps: list[float] = []

                    while rclpy.ok():
                        chunk = port.read(port.in_waiting or 1)
                        if not chunk:
                            continue
                        for frame in parser.feed(chunk):
                            if not frame.is_wrench:
                                continue
                            now = time.time()
                            if crc_base is None:
                                crc_base = parser.crc_errors
                            stamps.append(now)
                            while stamps and now - stamps[0] > 1.0:
                                stamps.pop(0)
                            values = frame.wrench()
                            with self._lock:
                                self._wrench = values
                                self._wrench_at = now
                                self._px6d_crc = parser.crc_errors - crc_base
                                self._px6d_hz = float(len(stamps))
                                self._feed_display(
                                    np.asarray(values, dtype=float), now
                                )
                            self._publish_wrench(values)
                            # 수집 중이면 같은 표본이 교정으로도 간다. 별도 경로를
                            # 두면 "화면에 보이는 값" 과 "교정이 쓴 값" 이 갈린다.
                            self.session.feed(values)
                            done = self.session.finish_if_due()
                            if done:
                                self.get_logger().info(
                                    f"교정 수집 완료: {done} — {self.session.last_error or 'ok'}"
                                )
                            # 수집이 끝난 **뒤에** 자동 캡처를 본다. 순서를 바꾸면
                            # 방금 끝낸 수집이 아직 active 로 보여 한 자세를 건너뛴다.
                            if done == "pose":
                                if self.session.poses:
                                    self._append_pose_log(self.session.poses[-1])
                                self._auto_fit_and_save()
                            self._auto_capture_step()
            except Exception as exc:  # 케이블이 빠져도 노드는 살아 있어야 한다
                self.get_logger().warn(
                    f"PX6D 읽기 실패 ({exc}) — 2 초 뒤 재시도", throttle_duration_sec=5.0
                )
                with self._lock:
                    self._wrench_source = "none"
                time.sleep(2.0)

    def _publish_calibration_valid(self) -> None:
        """교정 유효성을 주기적으로 낸다. 프로파일이 없으면 거짓이다."""
        valid = False
        if self.profile is not None:
            valid, _ = self.profile.validity()
        self.calib_pub.publish(Bool(data=bool(valid)))

    def _publish_wrench(self, values) -> None:
        """제어 스택이 볼 wrench 를 낸다 — **교정이 있으면 보상된 값** 으로.

        예전에는 원시값을 그대로 냈다. 그러면 접촉 문턱(8 N)이 자중이 실린 수에
        걸리는데, 그 자중은 자세에 따라 −8 에서 +8 N 까지 움직인다. 무접촉인데도
        문턱을 넘거나, 실제로 8 N 을 눌러도 안 넘는 일이 자세마다 달라진다.
        교정을 하는 이유가 정확히 그것을 없애는 것인데, 그 결과가 제어까지 닿지
        않으면 화면만 맞고 로봇은 여전히 원시값으로 판단한다.

        교정이 없거나 유효하지 않으면 원시값을 낸다. 그 경우 접촉 판정은
        ``us_diff_ik`` 가 교정 게이트로 막으므로, 못 믿을 값으로 무엇도 바뀌지
        않는다.
        """
        if self.wrench_pub is None:
            return

        published = values
        frame = self._wrench_frame
        if self.profile is not None:
            valid, _ = self.profile.validity()
            rot = self.rotation_base_flange()
            if valid and rot is not None:
                published = compensate(self.profile, values, rot).contact_probe
                # 프레임 이름도 바뀐다. 보상된 값은 센서 축이 아니라 프로브 축에
                # 있고, 이름을 그대로 두면 TF 를 쓰는 쪽이 조용히 틀린다.
                frame = f"{self.robot_name}_probe"

        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = published[0:3]
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = published[3:6]
        self.wrench_pub.publish(msg)

    # -- 프레임 조립 -------------------------------------------------------

    def telemetry_frame(self) -> dict:
        """GUI 계약의 RobotTelemetry 한 장."""
        now = time.time()
        with self._lock:
            joints = self._joints
            joints_at = self._joints_at
            pose = self._pose

        # 토픽이 끊긴 지 오래면 연결되지 않은 것으로 보고한다. 마지막 자세를 계속
        # 내보내면 GUI 는 멈춘 로봇과 살아 있는 로봇을 구별할 수 없다.
        fresh = joints is not None and (now - joints_at) < 0.5

        # 종류를 명시한다. 한 소켓에 두 종류가 흐르므로 받는 쪽이 모양으로
        # 추측하게 두면, 필드가 빠진 프레임이 다른 종류로 오인된다.
        frame: dict = {
            "type": "telemetry",
            "timestamp": int(now * 1000),
            "connected": bool(fresh),
        }
        if not fresh:
            return frame

        assert joints is not None
        frame["jointPositions"] = list(joints.position)
        if joints.velocity:
            frame["jointVelocities"] = list(joints.velocity)
            moving = any(abs(v) > _IDLE_VEL_RAD_S for v in joints.velocity)
            # 컨트롤러는 teleop 여부를 말해 주지 않는다. 움직임에서 추론한 값이며
            # safetyState 는 추론하지 않는다 — 안전 상태를 지어내면 안 된다.
            frame["robotState"] = "teleop" if moving else "idle"

        if self._mode:
            frame["probingMode"] = self._mode

        if self._teleop_frame is not None:
            frame["teleopFrame"] = self._teleop_frame

        frame["calibration"] = self.calibration_summary()

        if pose is not None:
            frame["tcpPose"] = {
                "position": [pose.position.x, pose.position.y, pose.position.z],
                "quaternion": [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
            }
        return frame

    def _feed_display(self, sample, now: float) -> None:
        """센서 표본 하나를 화면용 누적기에 넣는다. **락을 쥔 채** 부른다.

        두 가지를 동시에 쌓는다: 프레임 하나로 낼 구간 평균·극값과, 그래프가 그릴
        파형 구간이다. 같은 표본에서 나와야 화면의 숫자와 선이 서로 다른 것을
        말하지 않는다.
        """
        self._display_sum += sample
        self._display_count += 1
        self._display_min = np.minimum(self._display_min, sample[:3])
        self._display_max = np.maximum(self._display_max, sample[:3])

        if self._wave_dt <= 0.0:
            return

        current = self._wave_open
        if current is None or now >= current[0]:
            if current is not None:
                self._close_wave_bin(current)
            current = self._wave_open = [
                now + self._wave_dt, np.zeros(3), 0, np.full(3, np.inf), np.full(3, -np.inf)
            ]
        force = sample[:3]
        current[1] += force
        current[2] += 1
        current[3] = np.minimum(current[3], force)
        current[4] = np.maximum(current[4], force)

    def _close_wave_bin(self, current: list) -> None:
        """채우던 구간을 접어 대기열에 넣는다. **락을 쥔 채** 부른다."""
        if current[2] <= 0:
            return
        self._wave_bins.append(
            (current[0], current[1] / current[2], current[3].copy(), current[4].copy())
        )
        overflow = len(self._wave_bins) - self._wave_cap
        if overflow > 0:
            del self._wave_bins[:overflow]

    def _drain_waveform(self, now: float) -> dict | None:
        """모아 둔 구간을 GUI 계약의 파형 한 벌로 낸다. **락 바깥에서** 부른다."""
        with self._lock:
            bins = self._wave_bins
            self._wave_bins = []
        if not bins:
            return None
        # 시각은 절대값이 아니라 **이 프레임을 낸 순간으로부터 몇 ms 전인가** 로 낸다.
        # 브리지와 브라우저는 서로 다른 시계를 쓰므로, 절대 시각을 보내면 화면이
        # 그것을 자기 시계에 맞춰 다시 해석해야 하고 그 보정이 곧 오차가 된다.
        out = []
        for end_at, mean, low, high in bins:
            out.append([
                max(0, int(round((now - end_at) * 1000.0))),
                round(float(mean[0]), 4),
                round(float(mean[1]), 4),
                round(float(mean[2]), 4),
                round(float(low[2]), 4),
                round(float(high[2]), 4),
            ])
        return {"hz": round(self._wave_hz, 1), "bins": out}

    def wrench_frame(self) -> dict | None:
        """GUI 계약의 WrenchSample 한 장. 값이 신선하지 않으면 ``None``."""
        now = time.time()
        with self._lock:
            latest = self._wrench
            at = self._wrench_at
            source = self._wrench_source
            crc = self._px6d_crc
            sensor_hz = self._px6d_hz
            # 직전 프레임 이후 모인 표본의 평균. 없으면 최신값으로 떨어진다
            # (컨트롤러 경유처럼 표본이 드문 경로도 있다).
            if self._display_count:
                values = list(self._display_sum / self._display_count)
                averaged = self._display_count
                extremes = [
                    [float(v) for v in self._display_min],
                    [float(v) for v in self._display_max],
                ]
                self._display_sum = np.zeros(6)
                self._display_count = 0
                self._display_min = np.full(3, np.inf)
                self._display_max = np.full(3, -np.inf)
            else:
                values = latest
                averaged = 0
                extremes = None
        if values is None or (now - at) > 0.5:
            return None

        sample = {
            "type": "wrench",
            "timestamp": int(now * 1000),
            "force": [values[0], values[1], values[2]],
            "torque": [values[3], values[4], values[5]],
            "source": source,
            # 이 한 장이 몇 표본의 평균인가. 0 이면 평균하지 않은 최신값이다.
            "averagedSamples": averaged,
            # 같은 창의 축별 힘 [최소, 최대]. 평균이 지운 스파이크는 여기 남는다 —
            # 피크 표시가 1 초 창 안의 순간 최대를 놓치면 안 된다.
            "forceExtremes": extremes,
            # 플랜지 z 축이 베이스에서 위·아래 어느 쪽을 보는가 (+1 아래, -1 위).
            # 무접촉 검증이 자세대별로 나눠 보는 데 쓴다 — 보상이 틀리면 특정
            # 자세대에서만 틀리고, 전체 평균은 그것을 감춘다.
            "flangeZ": self._flange_z_component(),
        }

        # 이 프레임이 덮는 구간의 파형. 한 장에 접어 보내므로 프레임 수는 늘지 않는다 —
        # 1 kHz 를 그대로 흘리면 초당 천 장의 JSON 이고, 화면은 그것을 파싱하느라
        # 정작 그리지 못한다. 항목: [몇 ms 전, Fx, Fy, Fz, Fz 최소, Fz 최대].
        waveform = self._drain_waveform(now)
        if waveform is not None:
            sample["forceWaveform"] = waveform

        # 보상 단계를 모두 싣는다. 검증 화면이 "원값은 이런데 보상 뒤엔 이렇다" 를
        # 나란히 보여 줘야 하고, 어느 단계에서 이상해지는지가 곧 어느 교정이 틀렸는지다.
        if self.profile is not None:
            out = compensate(self.profile, np.asarray(values, dtype=float),
                             self.rotation_base_flange())
            sample["compensated"] = out.to_dict()
            valid, issues = self.profile.validity()
            sample["calibrationValid"] = valid
            sample["calibrationIssues"] = issues
        else:
            sample["calibrationValid"] = False
            sample["calibrationIssues"] = ["교정 프로파일 없음"]
        # 센서 회선 지표는 직결일 때만 뜻이 있다. 컨트롤러 경유 값에 붙이면
        # 재지 않은 것을 잰 것처럼 보인다.
        if source == "px6d_serial":
            sample["crcErrors"] = int(crc)
            sample["sensorHz"] = round(sensor_hz, 1)
        return sample

    def calibration_summary(self) -> dict:
        """화면이 그릴 교정 상태 한 벌."""
        reg = self.registration
        rot = reg.rotation_probe_from_sensor()
        summary = {
            "path": self.calibration_path,
            "mountingAngleDeg": reg.mounting_angle_deg,
            "axialFlip": reg.axial_flip,
            "leverSensorToProbeM": [
                float(v) for v in np.asarray(reg.r_sensor_to_probe_m).reshape(3)
            ],
            "rotationProbeFromSensor": [[float(v) for v in row] for row in rot],
            "session": self.session.status(),
            "autoCapture": self.auto_capture,
            "autoArmed": self.stillness.armed,
        }
        if self.profile is None:
            summary.update({"present": False, "valid": False,
                            "issues": ["교정 프로파일 없음"]})
            return summary

        valid, issues = self.profile.validity()
        g = self.profile.gravity
        summary.update({
            "present": True,
            "valid": valid,
            "issues": issues,
            "createdAt": int(self.profile.created_at * 1000),
            "ageHours": round(self.profile.age_s() / 3600.0, 2),
            "mountingNote": self.profile.mounting_note,
            "workingTare": (
                None if self.profile.working_tare is None
                else [float(v) for v in np.asarray(self.profile.working_tare).reshape(6)]
            ),
            # 영점을 잰 자세에서 지금 몇 도 떨어져 있는가. 이 영점은 상수라
            # 멀어질수록 뺀 만큼이 오차로 돌아온다.
            "tareOffAxisDeg": self._tare_off_axis_deg(),
            "biasAccepted": self.profile.bias.accepted,
            "biasReason": self.profile.bias.reason,
            "bias": [float(v) for v in self.profile.bias.bias],
            "massKg": None if g is None else g.mass_kg,
            "comSensorM": None if g is None else [float(v) for v in g.com_sensor_m],
            "rmsForceN": None if g is None else g.rms_force_n,
            "rmsTorqueNm": None if g is None else g.rms_torque_nm,
            "perAxisForceN": None if g is None else [float(v) for v in g.per_axis_force_n],
            "perAxisTorqueNm": None if g is None else [float(v) for v in g.per_axis_torque_nm],
            "coverage": None if g is None else g.coverage,
            "poses": None if g is None else g.poses,
        })
        return summary

    # -- 교정 명령 ---------------------------------------------------------

    def handle_command(self, request) -> dict | None:
        """교정 명령 하나를 처리한다.

        받는 것은 다섯 가지뿐이다 — 영점 수집, 자세 수집, 적합, 저장, 초기화.
        **어느 것도 로봇을 움직이지 않는다.** 모르는 명령은 조용히 버리는 대신
        거절 사유를 돌려준다: 무시된 명령이 성공처럼 보이면 조작자가 하지 않은 교정을
        했다고 믿게 된다.
        """
        if not isinstance(request, dict):
            return None
        command = request.get("command")
        if not command:
            return None

        def ok(**extra):
            return {"type": "ack", "command": command, "ok": True, **extra}

        def fail(reason):
            self.get_logger().warn(f"교정 명령 거절 [{command}]: {reason}")
            return {"type": "ack", "command": command, "ok": False, "reason": reason}

        if command == "calib.bias":
            if self._wrench_source != "px6d_serial":
                return fail("PX6D 직결이 아니다 — 컨트롤러 경유 값으로는 교정할 수 없다")
            self.session.start_bias(float(request.get("seconds", 4.0)))
            self.get_logger().info("전자 영점 수집 시작 — 아무것도 닿지 않게 하라")
            return ok()

        if command == "calib.pose":
            rot = self.rotation_base_flange()
            if rot is None:
                return fail("로봇 자세를 아직 못 받았다")
            if self._wrench_source != "px6d_serial":
                return fail("PX6D 직결이 아니다")
            label = str(request.get("label", f"pose {len(self.session.poses) + 1}"))
            self.session.start_pose(
                label,
                self.registration.gravity_in_sensor(rot),
                float(request.get("seconds", 2.0)),
                gravity_flange=self.gravity_in_flange(rot),
            )
            self.get_logger().info(f"자세 수집 시작: {label} — 로봇을 움직이지 마라")
            return ok(label=label)

        if command == "calib.cancel":
            self.session.cancel()
            return ok()

        if command == "calib.auto":
            enabled = bool(request.get("enabled", True))
            if enabled and self._wrench_source != "px6d_serial":
                return fail("PX6D 직결이 아니다 — 컨트롤러 경유 값으로는 교정할 수 없다")
            self.auto_capture = enabled
            self.auto_seconds = float(request.get("seconds", self.auto_seconds))
            self.auto_note = str(request.get("note", self.auto_note))
            self.stillness.reset()
            self.session.last_error = ""
            self.get_logger().info(
                "자동 캡처 켬 — 로봇을 옮기고 놓으면 잡는다" if enabled
                else "자동 캡처 끔"
            )
            return ok(enabled=enabled)

        if command == "calib.tare":
            # 작업 자세 영점.
            #
            # 프로브가 아래를 볼 것을 요구한다. 이 영점은 상수이므로 잰 자세에서만
            # 정확하고, 접근을 시작하는 자세가 아닌 곳에서 재면 정작 쓸 자세에서
            # 틀린다. "지금 자세" 라고만 하면 그 실수가 조용히 일어난다.
            if self.profile is None or self.profile.gravity is None:
                return fail("중력 보상 교정이 먼저다 — 그 위에 얹는 영점이다")
            rot = self.rotation_base_flange()
            if rot is None:
                return fail("로봇 자세를 아직 못 받았다")

            flange_z = float(rot[2, 2])
            if flange_z > -0.9:
                angle = math.degrees(math.acos(max(-1.0, min(1.0, -flange_z))))
                return fail(
                    f"프로브가 아래에서 {angle:.0f}° 벗어나 있다 — 이 영점은 잰 "
                    "자세에서만 정확하므로 작업 자세(아래 수직)에서 재야 한다"
                )

            with self._lock:
                latest = self._wrench
            if latest is None:
                return fail("센서 값이 없다")

            current = compensate(self.profile, latest, rot).contact_probe
            magnitude = float(np.linalg.norm(np.asarray(current)[:3]))
            if magnitude > float(request.get("maxForceN", 2.0)):
                return fail(
                    f"지금 {magnitude:.2f} N 이 실려 있다 — 무언가 닿아 있다. "
                    "그 힘을 영점에 넣으면 접촉을 접촉으로 못 읽는다"
                )

            with self._lock:
                pose_deg = list(self._joints.position) if self._joints else []
            # 이미 적용된 영점 위에 더한다. 두 번 재도 결과가 누적되지 않고
            # "지금이 0" 이라는 같은 뜻이 된다.
            previous = (
                np.zeros(6) if self.profile.working_tare is None
                else np.asarray(self.profile.working_tare, dtype=float).reshape(6)
            )
            self.profile.working_tare = previous + np.asarray(current, dtype=float)
            self.profile.tare_flange_z = flange_z
            self.profile.tare_pose_deg = [
                float(v) * 57.29577951308232 for v in pose_deg
            ]
            try:
                self.profile.save(self.calibration_path)
            except OSError as exc:
                return fail(f"저장 실패: {exc}")
            self.get_logger().info(
                f"작업 영점 — {magnitude:.3f} N 을 0 으로. 아래에서 "
                f"{math.degrees(math.acos(max(-1.0, min(1.0, -flange_z)))):.0f}° 자세"
            )
            return ok(removedN=magnitude, flangeZ=flange_z)

        if command == "calib.tare.clear":
            if self.profile is None:
                return fail("교정 프로파일이 없다")
            self.profile.working_tare = None
            self.profile.tare_flange_z = None
            self.profile.tare_pose_deg = []
            try:
                self.profile.save(self.calibration_path)
            except OSError as exc:
                return fail(f"저장 실패: {exc}")
            self.get_logger().info("작업 영점 해제")
            return ok()

        if command == "calib.load":
            # 기록해 둔 자세를 세션으로 되불러온다.
            #
            # 자동으로 불러오지 않는 것이 중요하다. 마운트를 바꾸거나 프로브를
            # 다시 물린 뒤라면 옛 자세는 다른 물건의 자중이고, 그것을 조용히
            # 섞으면 유효해 보이는 틀린 교정이 나온다. 불러오는 것은 조작자가
            # 지금 조립이 그때와 같다고 판단했을 때다.
            try:
                loaded = self._load_pose_log()
            except OSError as exc:
                return fail(f"자세 기록을 읽을 수 없다: {exc}")
            if not loaded:
                return fail(f"기록된 자세가 없다 ({self.pose_log_path})")
            self.session.poses.extend(loaded)
            self.session.gravity_model = None
            self.session.last_error = ""
            self.get_logger().info(
                f"자세 {len(loaded)} 개 되불러옴 — 총 {len(self.session.poses)} 개"
            )
            return ok(loaded=len(loaded), total=len(self.session.poses))

        if command == "calib.fit":
            valid = self.session.fit()
            return ok(valid=valid, issues=self._fit_issues())

        if command == "calib.save":
            if self.session.bias_result is None:
                return fail("전자 영점을 먼저 잰다")
            if self.session.gravity_model is None:
                # 적합을 **시도했다가 실패한 것** 과 아예 안 한 것을 구별해서 말한다.
                # 예전에는 둘 다 "calib.fit 을 먼저 하라" 로 나가서, 방금 fit 을 누른
                # 조작자에게 아무 뜻도 없는 문장이 돌아갔다.
                reason = self.session.last_error
                return fail(
                    f"중력 모델을 풀지 못했다 — {reason}" if reason
                    else "중력 모델을 먼저 푼다 (calib.fit)"
                )
            if not self.session.gravity_model.valid:
                return fail(
                    "중력 모델이 유효하지 않다 — " + "; ".join(self.session.gravity_model.issues)
                )
            reg = self.registration
            angle = request.get("mountingAngleDeg")
            if angle is not None:
                reg = FrameRegistration(
                    mounting_angle_deg=float(angle),
                    axial_flip=bool(request.get("axialFlip", reg.axial_flip)),
                    r_sensor_to_probe_m=reg.r_sensor_to_probe_m,
                    flange_to_sensor_rpy=reg.flange_to_sensor_rpy,
                )
            try:
                _, valid, issues = self._write_profile(reg, str(request.get("note", "")))
            except Exception as exc:  # noqa: BLE001
                return fail(f"저장 실패: {exc}")
            self.get_logger().info(
                f"교정 저장 — {self.calibration_path} · "
                f"{'유효' if valid else '무효: ' + '; '.join(issues)}"
            )
            return ok(valid=valid, issues=issues, path=self.calibration_path)

        if command == "teleop.operator_frame":
            # 조작자가 로봇의 어느 쪽에 서 있는가. 로봇을 움직이는 명령이 아니라
            # 손 지령을 어느 방향으로 읽을지의 문제다. 적용 시점은 us_diff_ik 가
            # 정한다 — 다음 파지 경계다.
            try:
                yaw = float(request.get("yawDeg"))
            except (TypeError, ValueError):
                return fail(f"yawDeg 가 숫자가 아니다: {request.get('yawDeg')!r}")
            if not math.isfinite(yaw):
                return fail("yawDeg 가 유한한 값이어야 한다")
            if self._teleop_frame is None:
                # 상태를 한 번도 못 받았다는 것은 us_diff_ik 가 없다는 뜻이다.
                # 요청을 보내 봐야 아무도 받지 않는데 성공처럼 보이면 안 된다.
                return fail("us_diff_ik 가 매핑을 아직 알리지 않았다 — 노드가 떠 있는가")
            self.teleop_frame_req.publish(Float64(data=yaw))
            self.get_logger().info(f"조작자 위치 요청 {yaw:+.0f}° → us_diff_ik")
            return ok(yawDeg=yaw, appliesOn="next-grip")

        if command == "calib.reset":
            self.stillness.reset()
            self.session.drop_poses()
            self.session.bias_result = None
            return ok()

        return fail(f"모르는 명령: {command}")

    # -- 웹소켓 ------------------------------------------------------------

    def _start_server(self) -> None:
        host = str(self.get_parameter("bridge.host").value)
        port = int(self.get_parameter("bridge.port").value)
        thread = threading.Thread(
            target=self._serve, args=(host, port), name="ws", daemon=True
        )
        thread.start()

    def _serve(self, host: str, port: int) -> None:
        try:
            import websockets
        except ImportError:
            self.get_logger().error(
                "websockets 가 없다: pip install websockets — 브리지를 열 수 없다"
            )
            return

        async def pump(connection):
            """교정 명령을 받는다. 동작 명령은 없다 — 모르는 것은 버린다."""
            try:
                async for message in connection:
                    try:
                        request = json.loads(message)
                    except Exception:
                        continue
                    reply = self.handle_command(request)
                    if reply is not None:
                        await connection.send(json.dumps(reply))
            except Exception:
                pass

        async def handler(connection):
            peer = getattr(connection, "remote_address", "?")
            self.get_logger().info(f"GUI 접속: {peer}")
            reader = asyncio.ensure_future(pump(connection))
            tele_dt = 1.0 / max(1.0, float(self.get_parameter("bridge.telemetry_hz").value))
            wrench_dt = 1.0 / max(1.0, float(self.get_parameter("bridge.wrench_hz").value))
            next_tele = next_wrench = 0.0
            try:
                while True:
                    now = time.monotonic()
                    if now >= next_tele:
                        next_tele = now + tele_dt
                        await connection.send(json.dumps(self.telemetry_frame()))
                    if now >= next_wrench:
                        next_wrench = now + wrench_dt
                        sample = self.wrench_frame()
                        if sample is not None:
                            await connection.send(json.dumps(sample))
                    await asyncio.sleep(min(tele_dt, wrench_dt) / 2.0)
            except Exception:
                pass
            finally:
                reader.cancel()
                self.get_logger().info(f"GUI 연결 종료: {peer}")

        async def main() -> None:
            async with websockets.serve(handler, host, port):
                self.get_logger().info(f"텔레메트리 브리지 대기: ws://{host}:{port}")
                await asyncio.Future()

        try:
            asyncio.run(main())
        except Exception as exc:
            self.get_logger().error(f"브리지 종료: {exc}")


def main(argv=None) -> None:
    """노드를 띄우고 Ctrl-C 까지 돈다."""
    rclpy.init(args=argv)
    node = TelemetryBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
