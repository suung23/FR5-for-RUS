"""Admittance 제어 법칙과 setpoint 슬루 제한 (DESIGN_NOTES §8).

ROS 에 의존하지 않는다. 노드는 이 모듈을 호출하는 껍데기이고, 제어 법칙 자체는
하드웨어 없이 시험된다.

감쇠 지배형(damping-dominant)을 쓴다. 질량항을 생략하는 이유는 접촉 안정성 측면에서
가장 견고하고 튜닝할 파라미터가 하나뿐이기 때문이다:

    v = clamp( (목표 − 측정) / B , ±상한 )

세 축이 같은 형태를 공유한다. z 는 힘, rx/ry 는 모멘트를 쓸 뿐이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "deadband",
    "admittance_velocity",
    "SlewLimiter",
    "AdmittanceConfig",
    "ContactSetpoint",
    "compose_probe_twist",
]


def deadband(error: float, width: float) -> float:
    """``|error| < width`` 이면 0, 아니면 문턱만큼 당겨서 돌려준다.

    문턱을 빼는(soft) 방식이라 데드밴드 경계에서 출력이 튀지 않는다. 단순히 0으로
    자르면 경계에서 ``width`` 만큼의 계단이 생긴다.

    Note:
        **오차에 적용해야 한다.** 측정값에 걸면 setpoint 가 편향된 상태(§5.4)에서
        정상 동작점이 데드밴드 밖으로 나가 루프가 항상 움직인다.
    """
    if width <= 0.0:
        return float(error)
    if abs(error) <= width:
        return 0.0
    return float(error - width) if error > 0.0 else float(error + width)


def admittance_velocity(error: float, damping: float, limit: float) -> float:
    """``clamp(error / damping, ±limit)``.

    Raises:
        ValueError: 감쇠가 0 이하일 때. 0 이면 속도가 발산한다.
    """
    if damping <= 0.0:
        raise ValueError(f"감쇠는 0보다 커야 한다, {damping} 을 받았다")
    return float(np.clip(error / damping, -abs(limit), abs(limit)))


class SlewLimiter:
    """setpoint 가 계단으로 뛰지 않게 기울기를 제한한다.

    §5.4 참조: ``M*`` 가 계단으로 0.1 N·m 뛰면 admittance 가 곧바로
    ``ω = 0.1 / 0.5 = 0.2 rad/s`` 를 내어 각속도 예산 전체를 한 번에 소진한다.

    Args:
        rate: 초당 최대 변화량. 0 이하이면 제한하지 않는다.
        initial: 시작값.
    """

    def __init__(self, rate: float, initial: float = 0.0) -> None:
        self.rate = float(rate)
        self.value = float(initial)

    def reset(self, value: float = 0.0) -> None:
        """내부 상태를 즉시 맞춘다. 상태 전이처럼 연속성이 필요 없을 때만 쓴다."""
        self.value = float(value)

    def step(self, target: float, dt: float) -> float:
        """``dt`` 만큼 목표를 향해 기울기 제한하며 이동한 값."""
        if self.rate <= 0.0 or dt <= 0.0:
            self.value = float(target)
            return self.value
        max_step = self.rate * dt
        self.value += float(np.clip(target - self.value, -max_step, max_step))
        return self.value


@dataclass
class AdmittanceConfig:
    """세 힘 축의 감쇠·상한·데드밴드.

    Attributes:
        damping_linear: ``B_z`` [N·s/m]. 1 N 오차가 만드는 속도의 역수.
        damping_angular: ``B_r`` [N·m·s/rad].
        max_linear: ``v_z`` 상한 [m/s].
        max_angular: ``ω_x, ω_y`` 상한 [rad/s].
        force_deadband: 힘 **오차** 데드밴드 [N].
        moment_deadband: 모멘트 **오차** 데드밴드 [N·m].
        moment_sign: 모멘트 → 회전 방향 부호. ⏳ 하드웨어에서 검증한다.
            틀리면 정렬 루프가 오정렬을 키우는 방향으로 돈다.
    """

    damping_linear: float = 1000.0
    damping_angular: float = 0.5
    max_linear: float = 0.010
    max_angular: float = 0.2
    force_deadband: float = 0.1
    moment_deadband: float = 0.01
    moment_sign: float = 1.0

    def __post_init__(self) -> None:
        if self.damping_linear <= 0.0 or self.damping_angular <= 0.0:
            raise ValueError("감쇠는 0보다 커야 한다")
        if self.max_linear <= 0.0 or self.max_angular <= 0.0:
            raise ValueError("속도 상한은 0보다 커야 한다")
        if self.force_deadband < 0.0 or self.moment_deadband < 0.0:
            raise ValueError("데드밴드는 음수일 수 없다")


@dataclass
class ContactSetpoint:
    """policy 또는 Stage 1 탐색이 내리는 접촉 목표 (§5).

    speed 지령이 아니라 **setpoint** 다. 100 Hz 힘 루프가 이것을 추종한다.
    """

    normal_force: float = 0.0   # F_n*  [N], 양수 = 압축
    moment_x: float = 0.0       # M*_x  [N·m]
    moment_y: float = 0.0       # M*_y  [N·m]


def compose_probe_twist(
    wrench_probe,
    setpoint: ContactSetpoint,
    image_twist,
    config: AdmittanceConfig,
    normal_force_sign: float = -1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """힘 축 3개와 영상 축 3개를 합쳐 프로브 프레임 twist 를 만든다 (§8.3).

    축 분할이 완전 직교라 단순 결합으로 충분하다 — 힘 축은 ``(v_z, ω_x, ω_y)``,
    영상 축은 ``(v_x, v_y, ω_z)`` 이며 태스크 공간에서 겹치지 않는다. 관절 공간에서
    다투는 것은 특이점 근처뿐이고 그것은 IK 의 추종 모니터가 잡는다.

    Args:
        wrench_probe: 프로브 프레임 wrench 6개 (이미 §4.3 변환을 거친 값).
        setpoint: 슬루 제한을 **이미 통과한** 접촉 목표.
        image_twist: policy 의 ``(v_x, v_y, ω_z)``. 힘 축 성분은 무시된다.
        config: 감쇠·상한·데드밴드.
        normal_force_sign: ``F_n = sign · F_z^probe``. ⏳ 하드웨어 검증 대상.

    Returns:
        ``(twist 6개, 진단 dict)``. 진단에는 실제 접촉력과 축별 오차가 담긴다.
    """
    wrench_probe = np.asarray(wrench_probe, dtype=float).reshape(-1)
    image_twist = np.asarray(image_twist, dtype=float).reshape(-1)

    normal_force = normal_force_sign * wrench_probe[2]
    moment_x, moment_y = wrench_probe[3], wrench_probe[4]

    # 데드밴드는 오차에 건다 — 측정값이 아니다 (§8.2).
    error_force = deadband(setpoint.normal_force - normal_force, config.force_deadband)
    error_mx = deadband(setpoint.moment_x - moment_x, config.moment_deadband)
    error_my = deadband(setpoint.moment_y - moment_y, config.moment_deadband)

    velocity_z = admittance_velocity(error_force, config.damping_linear, config.max_linear)
    omega_x = admittance_velocity(
        config.moment_sign * error_mx, config.damping_angular, config.max_angular
    )
    omega_y = admittance_velocity(
        config.moment_sign * error_my, config.damping_angular, config.max_angular
    )

    twist = np.array(
        [image_twist[0], image_twist[1], velocity_z, omega_x, omega_y, image_twist[5]]
    )
    diagnostics = {
        "normal_force": float(normal_force),
        "moment_x": float(moment_x),
        "moment_y": float(moment_y),
        "error_force": float(error_force),
        "error_moment_x": float(error_mx),
        "error_moment_y": float(error_my),
    }
    return twist, diagnostics
