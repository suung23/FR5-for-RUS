"""감쇠 최소자승 역속도 해법과 축별 추종오차 감시 (DESIGN_NOTES §9).

DLS 로 상위 아키텍처를 먼저 검증하기로 했다. 6-DoF 로봇이 6-DoF twist 를 받으므로
특이점에서 멀면 지령을 거의 그대로 재현한다.

문제는 특이점·관절한계 근처에서 감쇠가 걸릴 때 **어느 축이 희생됐는지 알 수 없다**는
점이다. DLS 는 축 우선순위를 표현하지 못해 모든 축을 균등하게 뭉갠다. 힘 축이 조용히
뭉개지면 접촉력이 어긋나는데 로그에는 아무것도 남지 않는다.

그래서 해를 순전파해 축별 오차를 낸다:

    V_achieved = J q̇        e = V_desired − V_achieved

비용이 사실상 0 이고 (야코비안은 이미 계산됐다), **QP 가 실제로 필요한가를 데이터로
판정**해 준다. 필요 없으면 DLS 확정, 필요하면 근거를 갖고 교체한다.

``solve(V_desired, q) -> q̇`` 시그니처는 QP 로 갈아끼울 때 그대로 유지되는 접합면이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["dls_solve", "TrackingError", "tracking_error", "DlsSolver"]


def dls_solve(jacobian: np.ndarray, twist: np.ndarray, damping: float = 0.02) -> np.ndarray:
    """``q̇ = Jᵀ(JJᵀ + λ²I)⁻¹ V``.

    Args:
        jacobian: ``6 x n`` 야코비안. twist 와 같은 프레임이어야 한다.
        twist: 목표 twist 6개 ``[vx, vy, vz, wx, wy, wz]``.
        damping: 감쇠 λ. 크면 특이점 근처에서 안정하지만 추종이 나빠진다.

    Returns:
        관절속도 ``n`` 개.

    Raises:
        ValueError: 모양이 맞지 않거나 감쇠가 음수일 때.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    twist = np.asarray(twist, dtype=float).reshape(-1)
    if jacobian.ndim != 2 or jacobian.shape[0] != twist.size:
        raise ValueError(
            f"야코비안 {jacobian.shape} 와 twist {twist.shape} 의 행 수가 맞지 않는다"
        )
    if damping < 0.0:
        raise ValueError(f"감쇠는 음수일 수 없다: {damping}")

    rows = jacobian.shape[0]
    normal = jacobian @ jacobian.T + (damping**2) * np.eye(rows)
    return jacobian.T @ np.linalg.solve(normal, twist)


@dataclass
class TrackingError:
    """DLS 해가 지령 twist 를 얼마나 재현했는지."""

    #: 축별 오차 ``V_desired − J q̇``, 6개.
    per_axis: np.ndarray
    #: 병진 오차 크기 [m/s].
    linear_norm: float
    #: 회전 오차 크기 [rad/s].
    angular_norm: float
    #: 힘 축(z, rx, ry)이 임계를 넘었는가. 이쪽이 뭉개지면 접촉력이 어긋난다.
    force_axes_degraded: bool

    def as_list(self) -> list[float]:
        """진단 토픽에 실을 평평한 리스트."""
        return [float(v) for v in self.per_axis]


def tracking_error(
    jacobian: np.ndarray,
    twist_desired: np.ndarray,
    joint_velocity: np.ndarray,
    linear_threshold: float = 0.002,
    angular_threshold: float = 0.05,
) -> TrackingError:
    """해를 순전파해 지령과의 축별 차이를 재고 힘 축 열화를 판정한다.

    힘 축은 프로브 프레임의 ``z``(인덱스 2), ``rx``(3), ``ry``(4) — admittance 가
    담당하는 축이다 (DESIGN_NOTES §5).
    """
    jacobian = np.asarray(jacobian, dtype=float)
    twist_desired = np.asarray(twist_desired, dtype=float).reshape(-1)
    achieved = jacobian @ np.asarray(joint_velocity, dtype=float).reshape(-1)
    error = twist_desired - achieved

    degraded = bool(
        abs(error[2]) > linear_threshold
        or abs(error[3]) > angular_threshold
        or abs(error[4]) > angular_threshold
    )
    return TrackingError(
        per_axis=error,
        linear_norm=float(np.linalg.norm(error[:3])),
        angular_norm=float(np.linalg.norm(error[3:])),
        force_axes_degraded=degraded,
    )


class DlsSolver:
    """프로브 프레임 twist 를 관절속도로 옮긴다.

    QP 로 교체할 때 이 클래스만 갈아끼우면 되도록 ``solve(V, q)`` 하나만 노출한다.
    QP 가 값어치를 하는 지점은 여기다 — "Fz 축 추종은 hard constraint, 영상 축은
    slack 허용"을 DLS 는 표현하지 못한다.

    Args:
        chain_builder: 인자 없이 ``PyKDL.Chain`` 을 돌려주는 호출체. KDL 의존을
            생성자 밖에 두어, 순수 수치 부분(:func:`dls_solve`)은 KDL 없이도 시험된다.
        damping: DLS 감쇠 λ.
    """

    def __init__(self, chain_builder, damping: float = 0.02) -> None:
        import PyKDL as kdl  # ROS 쪽에서만 필요

        self._kdl = kdl
        self.chain = chain_builder()
        self.num_joints = self.chain.getNrOfJoints()
        self.damping = float(damping)
        self._jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self._fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

    def _joint_array(self, q):
        array = self._kdl.JntArray(self.num_joints)
        for i in range(self.num_joints):
            array[i] = float(q[i])
        return array

    def jacobian(self, q) -> np.ndarray:
        """체인 말단(프로브)에서의 야코비안, **base 프레임 표현**."""
        jac = self._kdl.Jacobian(self.num_joints)
        self._jac_solver.JntToJac(self._joint_array(q), jac)
        return np.array(
            [[jac[r, c] for c in range(self.num_joints)] for r in range(6)], dtype=float
        )

    def rotation_base_probe(self, q) -> np.ndarray:
        """base → 프로브 회전행렬."""
        frame = self._kdl.Frame()
        self._fk_solver.JntToCart(self._joint_array(q), frame)
        return np.array([[frame.M[r, c] for c in range(3)] for r in range(3)], dtype=float)

    def solve(self, twist_probe: np.ndarray, q: np.ndarray) -> np.ndarray:
        """프로브(body) 프레임 twist → 관절속도.

        Args:
            twist_probe: ``[vx, vy, vz, wx, wy, wz]``, 프로브 프레임.
            q: 현재 관절각 [rad].

        Returns:
            관절속도 [rad/s].
        """
        rotation = self.rotation_base_probe(q)
        twist_probe = np.asarray(twist_probe, dtype=float).reshape(-1)
        twist_base = np.concatenate(
            (rotation @ twist_probe[:3], rotation @ twist_probe[3:])
        )
        return dls_solve(self.jacobian(q), twist_base, self.damping)

    def solve_with_diagnostics(
        self,
        twist_probe: np.ndarray,
        q: np.ndarray,
        linear_threshold: float = 0.002,
        angular_threshold: float = 0.05,
    ) -> tuple[np.ndarray, TrackingError]:
        """:meth:`solve` 에 축별 추종오차를 함께 돌려준다.

        오차는 **프로브 프레임**으로 되돌려 보고한다. 힘 축 판정이 프로브 축 기준이라야
        의미가 있기 때문이다.
        """
        rotation = self.rotation_base_probe(q)
        twist_probe = np.asarray(twist_probe, dtype=float).reshape(-1)
        twist_base = np.concatenate(
            (rotation @ twist_probe[:3], rotation @ twist_probe[3:])
        )
        jacobian = self.jacobian(q)
        joint_velocity = dls_solve(jacobian, twist_base, self.damping)

        achieved_base = jacobian @ joint_velocity
        achieved_probe = np.concatenate(
            (rotation.T @ achieved_base[:3], rotation.T @ achieved_base[3:])
        )
        error = twist_probe - achieved_probe
        degraded = bool(
            abs(error[2]) > linear_threshold
            or abs(error[3]) > angular_threshold
            or abs(error[4]) > angular_threshold
        )
        return joint_velocity, TrackingError(
            per_axis=error,
            linear_norm=float(np.linalg.norm(error[:3])),
            angular_norm=float(np.linalg.norm(error[3:])),
            force_axes_degraded=degraded,
        )
