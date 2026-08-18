"""강체 변환과 wrench 프레임 이동 (DESIGN_NOTES §4.3).

ROS 에도 로봇 SDK 에도 의존하지 않는다. 그래야 하드웨어 없이 시험된다.

핵심은 힘과 모멘트가 다르게 변환된다는 것이다:

    F_probe = R · F_sensor
    M_probe = R · M_sensor + r × F_probe

**힘은 기준점 평행이동에 불변이고 모멘트만 바뀐다.** r ≈ 0.1 m, F_z ≈ 5 N 이면
r × F 항이 0.5 N·m 인데, 자세 정렬 신호는 수십 mN·m 수준이라 이 항을 빼먹으면
신호가 통째로 묻힌다.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "rotation_from_rpy",
    "homogeneous",
    "wrench_sensor_to_probe",
    "probe_from_sensor",
]


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """고정축 XYZ 오일러각 → 3x3 회전행렬. KDL ``Rotation::RPY`` 와 같은 규약."""
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


def homogeneous(xyz, rpy) -> np.ndarray:
    """``[x,y,z]`` 와 ``[r,p,y]`` 로 4x4 동차변환을 만든다."""
    transform = np.eye(4)
    transform[:3, :3] = rotation_from_rpy(*(float(v) for v in rpy))
    transform[:3, 3] = [float(v) for v in xyz]
    return transform


def probe_from_sensor(
    j6_to_probe_xyz, j6_to_probe_rpy, j6_to_sensor_xyz, j6_to_sensor_rpy
) -> tuple[np.ndarray, np.ndarray]:
    """센서 프레임 → 프로브 프레임 변환을 ``(R, r)`` 로 돌려준다.

    두 변환 모두 J6 플랜지 기준으로 주어지므로 ``T_probe_sensor = T_j6_probe⁻¹ · T_j6_sensor``.

    Returns:
        ``R``: 센서 → 프로브 회전행렬.
        ``r``: **프로브 프레임에서 본 센서 원점 위치.** 모멘트 이동 항 ``r × F`` 에 쓴다.
    """
    t_probe = homogeneous(j6_to_probe_xyz, j6_to_probe_rpy)
    t_sensor = homogeneous(j6_to_sensor_xyz, j6_to_sensor_rpy)

    rotation_probe = t_probe[:3, :3]
    # T⁻¹ 을 명시적으로 쓴다: 회전은 전치, 병진은 -Rᵀp
    t_probe_inv = np.eye(4)
    t_probe_inv[:3, :3] = rotation_probe.T
    t_probe_inv[:3, 3] = -rotation_probe.T @ t_probe[:3, 3]

    t_rel = t_probe_inv @ t_sensor
    return t_rel[:3, :3], t_rel[:3, 3]


def wrench_sensor_to_probe(wrench, rotation: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """센서 프레임 wrench 를 프로브 접촉면 기준으로 옮긴다.

    Args:
        wrench: ``[Fx, Fy, Fz, Mx, My, Mz]``, 센서 프레임.
        rotation: 센서 → 프로브 회전행렬.
        offset: 프로브 프레임에서 본 센서 원점 위치 ``r``.

    Returns:
        프로브 프레임 wrench 6개.

    Raises:
        ValueError: wrench 가 6개가 아닐 때.
    """
    wrench = np.asarray(wrench, dtype=float).reshape(-1)
    if wrench.size != 6:
        raise ValueError(f"wrench 는 6개여야 한다, {wrench.size}개를 받았다")

    force = rotation @ wrench[:3]
    # 모멘트만 평행이동 항을 받는다. 힘은 기준점과 무관하다.
    moment = rotation @ wrench[3:] + np.cross(offset, force)
    return np.concatenate((force, moment))
