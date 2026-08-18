"""FR5 하드웨어 접근을 인터페이스 뒤로 감춘다 (DESIGN_NOTES §12.5).

실로봇 앞에서만 검증할 수 있는 구조로는 안전 로직을 시험하기 어렵다. 워치독 만료,
F/T 드롭아웃, 통신 두절 같은 **실패 경로는 실로봇에서 일부러 일으키기 곤란**하다.
그래서 두 구현을 둔다.

    FairinoBackend   실로봇. fairino Robot.RPC 위임
    MockBackend      무하드웨어. 지령을 적분하고 합성 wrench 를 낸다

주의: mock 은 기구학만 맞다. 조직 점탄성, 실제 F/T 잡음, 컨트롤러 지연은 재현하지
않는다. mock 통과는 "로직이 돈다"는 뜻이지 "제어가 맞다"는 뜻이 아니다.
"""
from __future__ import annotations

import abc
import math
import random
import threading
import time

import numpy as np

__all__ = ["RobotBackend", "FairinoBackend", "MockBackend", "make_backend", "fr5_forward_kinematics"]

#: KDL 체인(fr5_ik)과 동일한 링크 파라미터. 각 항은 관절 뒤에 곱해지는 고정 변환이며
#: ``(roll, pitch, yaw, x, y, z)`` 로 준다. 제어의 권위는 fr5_ik 의 KDL 체인이고,
#: 여기 사본은 MockBackend 가 자기 툴 자세를 만들기 위해서만 쓴다.
_FR5_LINKS = (
    (1.5708, 0.0, 0.0, 0.0, 0.0, 0.152),
    (0.0, 0.0, 0.0, -0.425, 0.0, 0.0),
    (0.0, 0.0, 0.0, -0.39501, 0.0, 0.0),
    (1.5708, 0.0, 0.0, 0.0, 0.0, 0.1021),
    (-1.5708, 0.0, 0.0, 0.0, 0.0, 0.102),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """고정축 XYZ 오일러각을 회전행렬로. KDL ``Rotation::RPY`` 와 같은 규약."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fr5_forward_kinematics(joint_rad) -> np.ndarray:
    """base → J6 동차변환 4x4.

    KDL 규약을 그대로 따른다: 세그먼트 변환은 ``Rz(q_i) * F_i`` 이고, ``F_i`` 는
    고정 팁 변환이다.
    """
    transform = np.eye(4)
    for angle, (roll, pitch, yaw, x, y, z) in zip(joint_rad, _FR5_LINKS):
        segment = np.eye(4)
        segment[:3, :3] = _rot_z(float(angle)) @ _rpy(roll, pitch, yaw)
        segment[:3, 3] = _rot_z(float(angle)) @ np.array([x, y, z])
        transform = transform @ segment
    return transform


def _matrix_to_pose(transform: np.ndarray) -> list[float]:
    """4x4 → ``[x, y, z (mm), rx, ry, rz (deg)]``. fairino ``tl_cur_pos`` 형식."""
    rotation = transform[:3, :3]
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:  # 짐벌락: roll 을 0 으로 두고 yaw 로 몰아준다
        roll, yaw = 0.0, math.atan2(-rotation[0, 1], rotation[1, 1])
    else:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return [
        transform[0, 3] * 1000.0,
        transform[1, 3] * 1000.0,
        transform[2, 3] * 1000.0,
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    ]


class RobotBackend(abc.ABC):
    """제어 노드가 로봇에 대해 알아도 되는 것의 전부."""

    @abc.abstractmethod
    def connect(self) -> None:
        """연결을 맺는다. 실패하면 예외를 던진다."""

    @abc.abstractmethod
    def joint_positions_deg(self) -> list[float]:
        """현재 관절각 6개, 도."""

    @abc.abstractmethod
    def joint_velocities_deg_s(self) -> list[float]:
        """현재 관절속도 6개, 도/초."""

    @abc.abstractmethod
    def joint_torques(self) -> list[float]:
        """관절 토크 6개."""

    @abc.abstractmethod
    def tool_pose(self) -> list[float]:
        """툴 자세 ``[x, y, z (mm), rx, ry, rz (deg)]``."""

    @abc.abstractmethod
    def wrench_raw(self) -> list[float]:
        """센서 프레임 wrench ``[Fx, Fy, Fz, Mx, My, Mz]``, N 과 N·m."""

    @abc.abstractmethod
    def wrench_valid(self) -> bool:
        """F/T 센서가 활성이고 값이 신선한가."""

    @abc.abstractmethod
    def servo_start(self) -> None:
        """서보 스트리밍 모드 진입."""

    @abc.abstractmethod
    def servo_j(self, joint_pos_deg, cmd_t: float, cmd_id: int) -> int:
        """관절각 지령 한 스텝. 0 이 성공, 그 외는 오류코드."""

    @abc.abstractmethod
    def servo_end(self) -> None:
        """서보 스트리밍 모드 종료."""

    @abc.abstractmethod
    def move_j(self, joint_pos_deg, vel: float) -> int:
        """블로킹 관절 이동. 서보 모드 밖에서만 쓴다."""

    def close(self) -> None:
        """자원을 놓는다. 여러 번 불러도 안전해야 한다."""

    @property
    def name(self) -> str:
        return type(self).__name__


class FairinoBackend(RobotBackend):
    """fairino ``Robot.RPC`` 위임.

    wrench 는 ``robot_state_pkg`` 에서 직접 읽는다. 이 SDK 판의
    ``FT_GetForceTorqueRCS`` 는 XML-RPC 를 하지 않고 상태 패키지를 읽을 뿐이라,
    RPC 왕복 지연 없이 컨트롤러 UDP 스트림 레이트로 값이 갱신된다.
    """

    def __init__(self, ip: str) -> None:
        self.ip = ip
        self._robot = None

    def connect(self) -> None:
        from fr5_control.fairino import Robot  # 실로봇에서만 필요

        self._robot = Robot.RPC(self.ip)
        if self._robot.robot_state_pkg is None:
            raise RuntimeError(f"{self.ip} 연결은 되었으나 상태 패키지가 비어 있다")

    @property
    def _state(self):
        if self._robot is None:
            raise RuntimeError("connect() 를 먼저 불러야 한다")
        return self._robot.robot_state_pkg

    def joint_positions_deg(self) -> list[float]:
        return list(self._state.jt_cur_pos)

    def joint_velocities_deg_s(self) -> list[float]:
        return list(self._state.actual_qd)

    def joint_torques(self) -> list[float]:
        return list(self._state.jt_cur_tor)

    def tool_pose(self) -> list[float]:
        return list(self._state.tl_cur_pos)

    def wrench_raw(self) -> list[float]:
        return list(self._state.ft_sensor_data)

    def wrench_valid(self) -> bool:
        return bool(self._state.ft_sensor_active)

    def servo_start(self) -> None:
        self._robot.ServoMoveStart()

    def servo_j(self, joint_pos_deg, cmd_t: float, cmd_id: int) -> int:
        return self._robot.ServoJ(
            joint_pos=list(joint_pos_deg),
            axisPos=[0.0] * 4,
            acc=0.0,
            vel=0.0,
            cmdT=cmd_t,
            filterT=0.0,
            gain=0.0,
            id=cmd_id,
        )

    def servo_end(self) -> None:
        self._robot.ServoMoveEnd()

    def move_j(self, joint_pos_deg, vel: float) -> int:
        return self._robot.MoveJ(
            joint_pos=list(joint_pos_deg), tool=0, user=0, vel=vel, blendT=-1.0
        )

    def close(self) -> None:
        if self._robot is None:
            return
        try:
            self._robot.ServoMoveEnd()
        except Exception:
            pass
        self._robot = None


class MockBackend(RobotBackend):
    """하드웨어 없이 도는 대역품.

    지령을 완전 추종한다고 가정하고 (``servo_j`` 로 받은 각을 그대로 상태에 반영),
    가상 평면에 파고든 깊이에 비례한 합성 ``F_z`` 를 낸다. 실패 주입이 이 클래스를
    두는 주된 이유다 — 워치독과 후퇴 경로는 실로봇에서 시험하기 위험하다.

    Args:
        start_joints_deg: 초기 관절각.
        contact_plane_z: 이보다 아래(base +z 기준, m)로 내려가면 접촉으로 친다.
        contact_stiffness: 합성 접촉 강성 [N/m].
        ft_noise_std: wrench 잡음 표준편차 [N].
        tool_offset_z: J6 에서 프로브 접촉면까지 거리 [m]. CAD 전에는 0 이고,
            그때는 J6 플랜지 자체를 접촉점으로 삼는다.
        drop_wrench_after_s: 이 시간 뒤 F/T 를 죽인다. 0 이면 비활성.
        freeze_state_after_s: 이 시간 뒤 상태 갱신을 멈춘다. 0 이면 비활성.
    """

    def __init__(
        self,
        start_joints_deg=None,
        contact_plane_z: float = 0.30,
        contact_stiffness: float = 2000.0,
        ft_noise_std: float = 0.02,
        tool_offset_z: float = 0.0,
        drop_wrench_after_s: float = 0.0,
        freeze_state_after_s: float = 0.0,
        seed: int = 0,
    ) -> None:
        self._lock = threading.Lock()
        self._q_deg = list(start_joints_deg or [0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
        self._qd_deg_s = [0.0] * 6
        self._last_cmd_time = None
        self._servo_active = False
        self._started_at = time.monotonic()

        self.contact_plane_z = contact_plane_z
        self.contact_stiffness = contact_stiffness
        self.ft_noise_std = ft_noise_std
        self.tool_offset_z = tool_offset_z
        self.drop_wrench_after_s = drop_wrench_after_s
        self.freeze_state_after_s = freeze_state_after_s
        self._rng = random.Random(seed)

    # -- 주입된 실패 ------------------------------------------------------

    def _elapsed(self) -> float:
        return time.monotonic() - self._started_at

    def _state_frozen(self) -> bool:
        return 0.0 < self.freeze_state_after_s <= self._elapsed()

    # -- 기구학 -----------------------------------------------------------

    def _probe_transform(self) -> np.ndarray:
        joint_rad = [math.radians(d) for d in self._q_deg]
        transform = fr5_forward_kinematics(joint_rad)
        if self.tool_offset_z:
            offset = np.eye(4)
            offset[2, 3] = self.tool_offset_z
            transform = transform @ offset
        return transform

    # -- RobotBackend -----------------------------------------------------

    def connect(self) -> None:
        self._started_at = time.monotonic()

    def joint_positions_deg(self) -> list[float]:
        with self._lock:
            return list(self._q_deg)

    def joint_velocities_deg_s(self) -> list[float]:
        with self._lock:
            return list(self._qd_deg_s)

    def joint_torques(self) -> list[float]:
        return [0.0] * 6

    def tool_pose(self) -> list[float]:
        with self._lock:
            return _matrix_to_pose(self._probe_transform())

    def wrench_raw(self) -> list[float]:
        """접촉 평면에 파고든 깊이에 비례한 합성 wrench.

        프로브 축이 평면 법선과 나란하지 않으면 접촉점이 프로브 중앙에서 벗어나므로
        모멘트가 생긴다. 그 편심을 기울기로 근사해 ``Mx, My`` 를 만든다 — Phase 2 의
        자세 정렬 로직이 0 이 아닌 신호를 받아볼 수 있어야 하기 때문이다.
        """
        if not self.wrench_valid():
            return [0.0] * 6
        with self._lock:
            transform = self._probe_transform()
        penetration = self.contact_plane_z - float(transform[2, 3])
        if penetration <= 0.0:
            force_z = 0.0
        else:
            force_z = self.contact_stiffness * penetration

        # 프로브 z축과 평면 법선(base +z) 사이의 기울기
        probe_axis = transform[:3, 2]
        tilt_x = float(probe_axis[1])
        tilt_y = float(-probe_axis[0])

        noise = lambda: self._rng.gauss(0.0, self.ft_noise_std)  # noqa: E731
        return [
            noise(),
            noise(),
            -force_z + noise(),  # 조직이 프로브를 밀어내는 방향
            force_z * tilt_x * 0.02 + noise() * 0.01,
            force_z * tilt_y * 0.02 + noise() * 0.01,
            noise() * 0.01,
        ]

    def wrench_valid(self) -> bool:
        if 0.0 < self.drop_wrench_after_s <= self._elapsed():
            return False
        return True

    def servo_start(self) -> None:
        self._servo_active = True
        self._last_cmd_time = None

    def servo_j(self, joint_pos_deg, cmd_t: float, cmd_id: int) -> int:
        if not self._servo_active:
            return -1
        if self._state_frozen():
            return 0  # 지령은 받지만 상태가 갱신되지 않는다 — 통신 두절 흉내
        now = time.monotonic()
        with self._lock:
            dt = cmd_t if self._last_cmd_time is None else max(1e-3, now - self._last_cmd_time)
            self._qd_deg_s = [(new - old) / dt for new, old in zip(joint_pos_deg, self._q_deg)]
            self._q_deg = list(joint_pos_deg)
        self._last_cmd_time = now
        return 0

    def servo_end(self) -> None:
        self._servo_active = False
        with self._lock:
            self._qd_deg_s = [0.0] * 6

    def move_j(self, joint_pos_deg, vel: float) -> int:
        with self._lock:
            self._q_deg = list(joint_pos_deg)
            self._qd_deg_s = [0.0] * 6
        return 0

    def close(self) -> None:
        self.servo_end()


def make_backend(kind: str, ip: str, **mock_kwargs) -> RobotBackend:
    """설정 문자열로 백엔드를 고른다.

    Raises:
        ValueError: 알 수 없는 종류. 조용히 mock 으로 떨어지지 않는다 — 실로봇을
            의도했는데 mock 이 도는 것이 가장 위험하다.
    """
    if kind == "fairino":
        return FairinoBackend(ip)
    if kind == "mock":
        return MockBackend(**mock_kwargs)
    raise ValueError(f"알 수 없는 백엔드 '{kind}'. 'fairino' 또는 'mock' 이어야 한다.")
