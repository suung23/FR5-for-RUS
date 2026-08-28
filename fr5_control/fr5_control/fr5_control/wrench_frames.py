"""프레임 정의와 wrench 변환 — {B} {F} {S} {M} {P} (DESIGN_NOTES §4, 2026-08-26).

PX6D 는 FR5 플랜지와 프로브 마운트 **사이**에 물려 있다. 그래서 센서가 재는 것은
프로브 접촉점의 힘이 아니라, 센서 원점에서 본 센서 축 기준의 wrench 다. 제어는
``{P}`` 에서 하므로 그 사이를 메우는 것이 이 모듈이다.

프레임
------
``{B}``  FR5 베이스.
``{F}``  플랜지 (J6).
``{S}``  PX6D 측정 프레임. 제조사 축 규약을 따른다.
``{M}``  프로브 마운트.
``{P}``  프로브 제어 프레임. 프로브 표면의 음향/접촉 기준점에 둔다.

프로브 축 규약 (§4.1 과 같다)::

    +z_P   축 방향. 조직 표면 법선이자 압축 방향.
    +x_P   영상면 내 lateral.
    +y_P   영상면 밖 elevational.

**부호 규약은 여기 한 곳에서만 정한다.** ``F_normal = F_{z_P}`` 이고 양수가 압축이다.
제어 코드 곳곳에 부호 뒤집기를 흩어 두면, 어느 것이 무엇을 상쇄하는지 아무도 추적할
수 없게 된다 — 그래서 뒤집기는 등록(registration) 층의 회전 행렬 안으로 흡수한다.

장착 각
-------
센서 축이 영상 표시축이나 로봇 베이스축과 정렬되어 있다고 가정하지 않는다. 실측
장착은 횡단면에서 돌아가 있다::

    +y_S  수평 기준에서 약 +45°  (우상 대각)
    +x_S  수평 기준에서 약 -45°  (우하 대각)

이 둘은 직교하고 ``z_S = x_S × y_S = z_P`` 가 되므로, 등록 회전은 **z 둘레 회전
하나**로 표현된다::

    {P}R{S} = Rz(-θ),   θ = 장착각 (기본 45°)

각도는 교정 파라미터다. 실물 확인 뒤 UI 에서 미세 조정한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "GRAVITY_B",
    "FrameRegistration",
    "skew",
    "rotation_z",
    "wrench_rotate",
    "wrench_translate",
]

#: 베이스 프레임 중력 가속도 [m/s²]. 표준 중력.
GRAVITY_B = np.array([0.0, 0.0, -9.80665])


def skew(v) -> np.ndarray:
    """벡터를 외적 행렬로 바꾼다.

    성질: 이 행렬에 b 를 곱한 것이 ``np.cross(v, b)`` 와 같다.
    """
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_z(angle_rad: float) -> np.ndarray:
    """축 z 둘레 회전 행렬을 만든다."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@dataclass
class FrameRegistration:
    """센서 → 프로브 등록. 부호 규약이 사는 유일한 자리다.

    Attributes:
        mounting_angle_deg: 횡단면 장착각 θ. ``{P}R{S} = Rz(-θ)``. 기본 45°.
        axial_flip: 조립이 뒤집혀 달렸는가. 참이면 ``Rx(180°)`` 를 앞에 곱한다.

            **이것이 부호 뒤집기의 전부다.** 스칼라 부호를 따로 두지 않는 이유는,
            스칼라는 힘에만 적용되고 모멘트에는 잊혀지기 때문이다. 180° 회전은
            힘과 모멘트에 똑같이 작용하므로 그런 비대칭이 생기지 않는다.

            ``Rx(180°) = diag(1, -1, -1)`` 이라 z 와 함께 y(elevational)도 뒤집힌다.
            그것이 물리적으로 맞다 — 조립을 lateral 축 둘레로 뒤집으면 실제로 두 축이
            함께 뒤집힌다.
        r_sensor_to_probe_m: ``{P}`` 기준, 센서 원점에서 프로브 제어점까지 [m].
            모멘트 기준점 이동에 쓴다 (§4.3). 0 이 아니면 지렛대 항이 정렬 신호를
            덮으므로 반드시 채워야 한다.
        flange_to_sensor_rpy: ``{F}`` → ``{S}`` 고정 회전 [rad], 고정축 XYZ.
            중력 보상에서 ``{S}R{B}`` 를 만들 때 쓴다.
    """

    mounting_angle_deg: float = 45.0
    axial_flip: bool = False
    r_sensor_to_probe_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    flange_to_sensor_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def rotation_probe_from_sensor(self) -> np.ndarray:
        """센서 → 프로브 회전 ``{P}R{S}``. 직교이고 ``det = +1`` 이다.

        진짜 회전이어야 하는 이유: 모멘트는 유사벡터라 반사가 섞이면 벡터와 다르게
        변환된다. 축 교환이나 부호 대각행렬로 "단순화" 하면 힘은 맞는데 **모멘트만
        거울상**이 되고, 화면으로는 알아채기 어렵다.
        """
        rot = rotation_z(-math.radians(self.mounting_angle_deg))
        if self.axial_flip:
            rot = _rotation_x(math.pi) @ rot
        return rot

    def rotation_flange_from_sensor(self) -> np.ndarray:
        """플랜지 → 센서 회전 ``{F}R{S}``. 고정축 XYZ 오일러각에서 만든다."""
        roll, pitch, yaw = np.asarray(self.flange_to_sensor_rpy, dtype=float).reshape(3)
        return _rotation_z_yx(yaw, pitch, roll)

    def rotation_sensor_from_base(self, rot_base_flange: np.ndarray) -> np.ndarray:
        """베이스 → 센서 회전 ``{S}R{B}``. 중력을 센서 축으로 내리는 데 쓴다.

        Args:
            rot_base_flange: ``{B}R{F}``. 로봇 순기구학에서 온다.
        """
        rot_base_sensor = np.asarray(rot_base_flange, dtype=float) @ self.rotation_flange_from_sensor()
        return rot_base_sensor.T

    def gravity_in_sensor(self, rot_base_flange: np.ndarray) -> np.ndarray:
        """이 자세에서 중력이 센서 축에 어떻게 실리는가. ``{S}g`` [m/s²]."""
        return self.rotation_sensor_from_base(rot_base_flange) @ GRAVITY_B

    def to_dict(self) -> dict:
        """저장용."""
        return {
            "mounting_angle_deg": float(self.mounting_angle_deg),
            "axial_flip": bool(self.axial_flip),
            "r_sensor_to_probe_m": [float(v) for v in np.asarray(self.r_sensor_to_probe_m).reshape(3)],
            "flange_to_sensor_rpy": [float(v) for v in np.asarray(self.flange_to_sensor_rpy).reshape(3)],
        }

    @staticmethod
    def from_stack(
        sensor_xyz,
        sensor_rpy,
        probe_xyz,
        probe_rpy,
        *,
        tolerance_deg: float = 0.5,
    ) -> "FrameRegistration":
        """probe.yaml 의 적층 값에서 등록을 **유도한다** (2026-08-27).

        이 함수가 있는 이유는 중복 관리를 없애기 위해서다. ``mounting_angle_deg`` 와
        ``r_sensor_to_probe_m`` 은 독립된 설정값이 아니라 ``tool.j6_to_probe_*`` 와
        ``ft_sensor.j6_to_sensor_*`` 에서 **계산되는 값**이다. 따로 적어 두면 적층
        치수를 고칠 때 한쪽만 고쳐지고, 어긋나도 각각은 그럴듯해 보여 알아채기 어렵다
        (``probe_tcp.compose_stack`` 의 주석이 말하는 그 문제다).

        유도::

            {P}R{S} = {F}R{P}ᵀ · {F}R{S}          = Rz(-θ)  또는  Rx(180°)·Rz(-θ)
            r_{S→P} = {F}R{P}ᵀ · (p_{F→P} - p_{F→S})       ({P} 기준, §4.3 이 요구하는 형태)

        Args:
            sensor_xyz: ``ft_sensor.j6_to_sensor_xyz`` [m].
            sensor_rpy: ``ft_sensor.j6_to_sensor_rpy`` [rad], 고정축 XYZ.
            probe_xyz: ``tool.j6_to_probe_xyz`` [m].
            probe_rpy: ``tool.j6_to_probe_rpy`` [rad], 고정축 XYZ.
            tolerance_deg: ``{P}R{S}`` 가 z 둘레 회전에서 벗어나도 되는 한도.

        Returns:
            네 항목이 서로 정합적인 :class:`FrameRegistration`.

        Raises:
            ValueError: 값에 NaN 이 있거나(= 아직 안 쟀다는 뜻이다), ``{P}R{S}`` 가
                z 둘레 회전으로 표현되지 않을 때. 둘 다 **조용히 넘어가면 안 되는**
                조건이다 — NaN 을 0 으로 채우면 틀린 기하로 돌고, z 둘레가 아닌 장착은
                각도 하나로 표현할 수 없어 θ 가 무의미해진다.
        """
        arrays = []
        for name, value in (
            ("j6_to_sensor_xyz", sensor_xyz),
            ("j6_to_sensor_rpy", sensor_rpy),
            ("j6_to_probe_xyz", probe_xyz),
            ("j6_to_probe_rpy", probe_rpy),
        ):
            arr = np.asarray(value, dtype=float).reshape(3)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{name} 에 NaN 이 있다 — 아직 실측되지 않았다는 뜻이다. "
                    "0 으로 채우면 조용히 틀린 기하로 돌기 때문에 여기서 멈춘다."
                )
            arrays.append(arr)
        p_sensor, rpy_sensor, p_probe, rpy_probe = arrays

        rot_flange_sensor = _rotation_z_yx(*rpy_sensor[::-1])
        rot_flange_probe = _rotation_z_yx(*rpy_probe[::-1])

        rot_probe_sensor = rot_flange_probe.T @ rot_flange_sensor
        lever = rot_flange_probe.T @ (p_probe - p_sensor)

        # z 둘레 회전인가. 뒤집혀 달렸으면 Rx(180°) 를 걷어낸 뒤 다시 본다.
        axial_flip = bool(rot_probe_sensor[2, 2] < 0.0)
        planar = _rotation_x(math.pi) @ rot_probe_sensor if axial_flip else rot_probe_sensor
        residual = math.degrees(
            math.acos(max(-1.0, min(1.0, planar[2, 2])))
        )
        if residual > tolerance_deg:
            raise ValueError(
                f"{{P}}R{{S}} 가 z 둘레 회전이 아니다 (z 축이 {residual:.2f}° 기울어 있다). "
                "장착각 하나로는 표현할 수 없는 기하이며, mounting_angle_deg 를 무엇으로 "
                "적든 틀린다. 적층 rpy 를 다시 확인하라."
            )

        # {P}R{S} = Rz(-θ) 이므로 θ 는 그 yaw 의 부호를 뒤집은 것이다.
        theta = -math.degrees(math.atan2(planar[1, 0], planar[0, 0]))

        return FrameRegistration(
            mounting_angle_deg=theta,
            axial_flip=axial_flip,
            r_sensor_to_probe_m=lever,
            flange_to_sensor_rpy=rpy_sensor.copy(),
        )

    @staticmethod
    def from_dict(data: dict) -> "FrameRegistration":
        """저장본에서 되살린다."""
        return FrameRegistration(
            mounting_angle_deg=float(data.get("mounting_angle_deg", 45.0)),
            axial_flip=bool(data.get("axial_flip", False)),
            r_sensor_to_probe_m=np.asarray(
                data.get("r_sensor_to_probe_m", [0.0, 0.0, 0.0]), dtype=float
            ).reshape(3),
            flange_to_sensor_rpy=np.asarray(
                data.get("flange_to_sensor_rpy", [0.0, 0.0, 0.0]), dtype=float
            ).reshape(3),
        )


def _rotation_z_yx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """고정축 XYZ (= Rz·Ry·Rx) 회전."""
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def wrench_rotate(rotation: np.ndarray, wrench) -> np.ndarray:
    """힘과 모멘트에 **같은** 회전을 적용해 wrench 를 돌린다.

    Args:
        rotation: 3×3 진짜 회전.
        wrench: ``[Fx, Fy, Fz, Mx, My, Mz]``.

    Returns:
        회전된 6 벡터.
    """
    rot = np.asarray(rotation, dtype=float)
    w = np.asarray(wrench, dtype=float).reshape(6)
    return np.concatenate([rot @ w[:3], rot @ w[3:]])


def wrench_translate(wrench, lever) -> np.ndarray:
    """모멘트 기준점을 옮긴다 (§4.3).

    ``τ_contact = τ_ext - r × f_ext`` 에서 ``r`` 은 센서 원점 → 프로브 제어점이며
    **같은 프레임**으로 표현되어 있어야 한다.

    ``r ≈ 0.1 m``, ``F_z ≈ 5 N`` 이면 오프셋이 0.5 N·m 다 — 정렬 신호(수십 mN·m)를
    통째로 덮는다. 이 항을 빼먹으면 rx/ry 정렬이 원리적으로 동작하지 않는다.

    Args:
        wrench: 같은 프레임의 ``[f, τ]``.
        lever: 같은 프레임의 ``r``.

    Returns:
        기준점이 옮겨진 6 벡터. 힘은 변하지 않는다.
    """
    w = np.asarray(wrench, dtype=float).reshape(6)
    r = np.asarray(lever, dtype=float).reshape(3)
    return np.concatenate([w[:3], w[3:] - np.cross(r, w[:3])])
