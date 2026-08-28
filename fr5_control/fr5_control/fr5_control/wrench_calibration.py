"""PX6D 2 단계 교정 — 전자 영점과 중력 보상 (2026-08-26).

두 절차는 **다른 것**이며 하나로 합치면 안 된다.

A. 전자 영점 (정적 바이어스)
    조립을 단 채, 아무것과도 닿지 않은 상태에서 채널 바이어스를 잰다. 로봇은 한 자세에
    가만히 있는다. 이것은 증폭단 오프셋과 그 자세에서의 자중을 **합쳐서** 잡는다.

    **이것만으로는 부족하다.** 자세가 바뀌면 달린 프로브·홀더의 중력이 센서 축에
    다르게 실린다. 한 자세에서 0 으로 맞춘 값이 다른 자세에서 수 N 씩 어긋난다.

B. 다자세 중력 보상
    여러 정적 자세에서 무부하 데이터를 모아 **유효 질량**과 **무게중심**을 푼다. 그러면
    자세에 의존하는 중력 wrench 를 예측해 뺄 수 있다.

왜 선형으로 풀리는가
--------------------
힘::

    f = m·g_S + b_f      →  [g_S | I] · [m, b_f]     (m, b_f 에 대해 선형)

모멘트::

    τ = (m·r) × g_S + b_τ = -[g_S]× (m·r) + b_τ

``c = m·r`` 을 미지수로 두면 이것도 선형이다. 그다음 ``r = c / m``. 비선형 최적화가
필요 없고, 해가 유일한지·자세가 충분한지를 특이값으로 바로 판정할 수 있다.

로버스트
--------
한 자세에서 케이블이 걸리거나 누가 스쳐도 그 자세만 크게 틀어진다. 보통 최소자승은
그 하나에 전체 해가 끌려간다. Huber 가중 IRLS 로 그런 자세의 영향력을 줄인다 —
버리지는 않는다. 버리면 왜 버렸는지가 기록에 안 남는다.

⚠️ 이 모듈은 **로봇을 움직이지 않는다.** 자세 이동은 조작자가 손으로 한다 (§7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fr5_control.wrench_frames import skew

__all__ = [
    "BiasResult",
    "CalibrationPose",
    "GravityModel",
    "estimate_static_bias",
    "fit_gravity_model",
    "pose_coverage",
]

#: 자세가 충분히 흩어졌다고 볼 최소 개수 (사양 §4B).
MIN_POSES = 12

#: 자세 다양성 하한. 중력 방향 행렬의 σ₃/σ₁ 이다.
#: 모든 자세가 같은 방향이면 rank 1 이 되어 0 에 가까워지고, 그때 COM 은 풀리지 않는다.
MIN_COVERAGE = 0.15


@dataclass
class BiasResult:
    """전자 영점 결과.

    Attributes:
        bias: 6 채널 평균 ``b_S``.
        std: 채널별 표준편차. 판정 근거이자 기록이다.
        samples: 쓴 표본 수.
        duration_s: 수집 시간.
        accepted: 분산 판정을 통과했는가.
        reason: 통과·거부 이유. 사람이 읽는다.
    """

    bias: np.ndarray
    std: np.ndarray
    samples: int
    duration_s: float
    accepted: bool
    reason: str

    def to_dict(self) -> dict:
        """저장용."""
        return {
            "bias": [float(v) for v in self.bias],
            "std": [float(v) for v in self.std],
            "samples": int(self.samples),
            "duration_s": float(self.duration_s),
            "accepted": bool(self.accepted),
            "reason": self.reason,
        }


@dataclass
class CalibrationPose:
    """중력 식별용 한 자세.

    Attributes:
        gravity_sensor: ``{S}g`` [m/s²]. 이 자세에서 중력이 센서 축에 어떻게 실리는가.
        wrench: 그 자세의 평균 wrench (전자 영점을 **빼기 전**의 원값이어야 한다 —
            잔여 바이어스를 같이 풀기 때문이다).
        label: 조작자에게 보여 준 자세 이름. 기록용.
    """

    gravity_sensor: np.ndarray
    wrench: np.ndarray
    label: str = ""
    #: ``{F}g`` [m/s²] — 순기구학에서 바로 나오는, **가정이 섞이지 않은** 값.
    #: ``gravity_sensor`` 는 플랜지→센서 회전을 알아야 만들 수 있고 그 회전이
    #: 미지수이므로, 정렬을 함께 풀 때는 이쪽을 쓴다. 없으면 옛 기록이라는
    #: 뜻이고, 그때는 두 값이 같다고 본 것이다 (회전 = 단위행렬 가정).
    gravity_flange: np.ndarray | None = None


@dataclass
class GravityModel:
    """중력 wrench 모델과 그 신뢰도.

    Attributes:
        mass_kg: 유효 페이로드 질량.
        com_sensor_m: ``{S}`` 기준 무게중심.
        residual_bias: 중력으로 설명되지 않고 남은 6 채널 바이어스.
        rms_force_n: 힘 잔차 RMS.
        rms_torque_nm: 모멘트 잔차 RMS.
        per_axis_force_n: 축별 힘 잔차 RMS.
        per_axis_torque_nm: 축별 모멘트 잔차 RMS.
        coverage: 자세 다양성 0..1.
        poses: 쓴 자세 수.
        valid: 이 모델을 제어에 써도 되는가.
        issues: 경고·거부 사유. 비어 있으면 문제 없음.
    """

    mass_kg: float
    com_sensor_m: np.ndarray
    residual_bias: np.ndarray
    rms_force_n: float
    rms_torque_nm: float
    per_axis_force_n: np.ndarray
    per_axis_torque_nm: np.ndarray
    coverage: float
    poses: int
    valid: bool
    issues: list = field(default_factory=list)
    #: ``{S}R{F}``, 같은 자세들에서 함께 푼 것. 단위행렬이면 풀지 않았다는 뜻이다.
    rotation_sensor_from_flange: np.ndarray = field(
        default_factory=lambda: np.eye(3)
    )
    #: 정렬 적합의 특이값 산포. 0 에 가까워야 회전이라 부를 수 있다.
    alignment_spread: float = 0.0

    def predict(self, gravity_sensor) -> np.ndarray:
        """이 자세에서 예상되는 중력 wrench (잔여 바이어스 포함).

        Args:
            gravity_sensor: ``{S}g`` [m/s²].

        Returns:
            6 벡터. 런타임에 원값에서 이것을 뺀다.
        """
        g = np.asarray(gravity_sensor, dtype=float).reshape(3)
        force = self.mass_kg * g
        torque = np.cross(self.com_sensor_m, force)
        return np.concatenate([force, torque]) + self.residual_bias

    def to_dict(self) -> dict:
        """저장용."""
        return {
            "mass_kg": float(self.mass_kg),
            "rotation_sensor_from_flange": [
                [float(v) for v in row] for row in self.rotation_sensor_from_flange
            ],
            "alignment_spread": float(self.alignment_spread),
            "com_sensor_m": [float(v) for v in self.com_sensor_m],
            "residual_bias": [float(v) for v in self.residual_bias],
            "rms_force_n": float(self.rms_force_n),
            "rms_torque_nm": float(self.rms_torque_nm),
            "per_axis_force_n": [float(v) for v in self.per_axis_force_n],
            "per_axis_torque_nm": [float(v) for v in self.per_axis_torque_nm],
            "coverage": float(self.coverage),
            "poses": int(self.poses),
            "valid": bool(self.valid),
            "issues": list(self.issues),
        }

    @staticmethod
    def from_dict(data: dict) -> "GravityModel":
        """저장본에서 되살린다."""
        return GravityModel(
            mass_kg=float(data["mass_kg"]),
            com_sensor_m=np.asarray(data["com_sensor_m"], dtype=float).reshape(3),
            residual_bias=np.asarray(data["residual_bias"], dtype=float).reshape(6),
            rms_force_n=float(data["rms_force_n"]),
            rms_torque_nm=float(data["rms_torque_nm"]),
            per_axis_force_n=np.asarray(data["per_axis_force_n"], dtype=float).reshape(3),
            per_axis_torque_nm=np.asarray(data["per_axis_torque_nm"], dtype=float).reshape(3),
            coverage=float(data["coverage"]),
            poses=int(data["poses"]),
            valid=bool(data["valid"]),
            rotation_sensor_from_flange=np.asarray(
                data.get("rotation_sensor_from_flange", np.eye(3).tolist()), dtype=float
            ).reshape(3, 3),
            alignment_spread=float(data.get("alignment_spread", 0.0)),
            issues=list(data.get("issues", [])),
        )


def estimate_static_bias(
    samples,
    duration_s: float,
    max_force_std_n: float = 0.15,
    max_torque_std_nm: float = 0.01,
    min_samples: int = 300,
) -> BiasResult:
    """전자 영점. 접촉이 전혀 없는 상태에서 채널 평균을 잡는다.

    Args:
        samples: ``(N, 6)`` 원시 wrench.
        duration_s: 수집에 걸린 시간.
        max_force_std_n: 힘 채널 표준편차 상한. 넘으면 거부한다.
        max_torque_std_nm: 모멘트 채널 표준편차 상한.
        min_samples: 최소 표본 수. 1 kHz 에서 300 이면 0.3 초다.

    Returns:
        :class:`BiasResult`. ``accepted`` 가 거짓이면 값을 쓰지 말 것.

    분산으로 거부하는 이유: 조용해야 할 신호가 시끄럽다는 것은 **무언가 닿아 있거나
    케이블이 당기고 있다**는 뜻이다. 그 상태의 평균을 영점으로 저장하면 그 접촉이
    영점 안으로 들어가 버리고, 이후 모든 측정이 조용히 틀어진다.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"samples 는 (N, 6) 이어야 한다: {arr.shape}")

    n = arr.shape[0]
    mean = arr.mean(axis=0) if n else np.zeros(6)
    std = arr.std(axis=0) if n else np.zeros(6)

    if n < min_samples:
        return BiasResult(mean, std, n, duration_s, False,
                          f"표본 부족 {n} < {min_samples}")
    if duration_s < 3.0:
        return BiasResult(mean, std, n, duration_s, False,
                          f"수집 시간 부족 {duration_s:.1f} s < 3.0 s")

    force_std = float(std[:3].max())
    torque_std = float(std[3:].max())
    if force_std > max_force_std_n:
        return BiasResult(mean, std, n, duration_s, False,
                          f"힘 잡음 과다 {force_std:.3f} > {max_force_std_n:.3f} N "
                          "— 무언가 닿아 있거나 케이블이 당기고 있다")
    if torque_std > max_torque_std_nm:
        return BiasResult(mean, std, n, duration_s, False,
                          f"모멘트 잡음 과다 {torque_std:.4f} > {max_torque_std_nm:.4f} N·m")

    return BiasResult(mean, std, n, duration_s, True,
                      f"통과 — {n} 표본 / {duration_s:.1f} s, "
                      f"힘 σ≤{force_std:.3f} N · 모멘트 σ≤{torque_std:.4f} N·m")


def pose_coverage(gravity_vectors) -> tuple:
    """자세가 중력 방향을 얼마나 흩어 놓았는가.

    Args:
        gravity_vectors: ``(N, 3)`` 각 자세의 ``{S}g``.

    Returns:
        ``(coverage, singular_values)``. coverage 는 ``σ₃/σ₁`` 로 0..1 이다.

    왜 특이값인가: 모든 자세가 중력을 같은 방향으로 받으면 방향 행렬의 rank 가 1 이
    되고, 그 데이터로는 COM 의 두 성분이 원리적으로 관측되지 않는다. 잔차는 작게
    나오는데 해는 틀린 상태가 되므로 — 잔차만 보면 못 잡는다.
    """
    dirs = np.asarray(gravity_vectors, dtype=float).reshape(-1, 3)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    unit = dirs / norms
    sv = np.linalg.svd(unit, compute_uv=False)
    sv = np.pad(sv, (0, max(0, 3 - len(sv))))
    coverage = float(sv[2] / sv[0]) if sv[0] > 1e-9 else 0.0
    return coverage, sv


def _irls(design: np.ndarray, target: np.ndarray, huber_delta: float,
          iterations: int = 12) -> np.ndarray:
    """Huber 가중 반복 최소자승.

    한 자세가 오염돼도 전체 해가 그쪽으로 끌려가지 않게 한다. 완전히 버리지 않는
    이유는, 버린 자세가 기록에 안 남으면 나중에 "왜 이 값이 나왔나" 를 재구성할 수
    없기 때문이다.
    """
    weights = np.ones(design.shape[0])
    solution = np.zeros(design.shape[1])
    for _ in range(iterations):
        w = np.sqrt(weights)[:, None]
        solution, *_ = np.linalg.lstsq(design * w, target * w[:, 0], rcond=None)
        residual = np.abs(design @ solution - target)
        scale = max(huber_delta, 1e-9)
        weights = np.where(residual <= scale, 1.0, scale / np.maximum(residual, 1e-12))
    return solution


@dataclass
class SensorAlignment:
    """플랜지에서 본 센서의 자세, 데이터에서 푼 것.

    Attributes:
        rotation_sensor_from_flange: ``{S}R{F}``. 직교이고 ``det = +1``.
        mass_kg: 같은 적합에서 함께 나온 페이로드 질량.
        bias_force: 중력으로 설명되지 않은 힘 바이어스.
        scale_spread: 세 특이값의 (최대−최소)/평균. 0 에 가까워야 한다 — 크면
            적합이 회전이 아닌 것을 회전이라고 우긴 것이다.
        residual_n: 힘 잔차 RMS.
    """

    rotation_sensor_from_flange: np.ndarray
    mass_kg: float
    bias_force: np.ndarray
    scale_spread: float
    residual_n: float


def solve_axial_alignment(gravity_flange, forces) -> SensorAlignment:
    """센서 정렬을 **z 둘레 회전 하나** 로 제약해 푼다.

    적층이 동축이라는 사실을 쓴다. 그것은 가정이 아니라 측정이다 — 축 방향 압축
    (``Sz+``)이 Fz 채널로 깨끗하게 갔고(분리비 4.9), 그것이 곧 센서 z 와 프로브 z
    가 같은 축이라는 뜻이다. 남는 자유도는 그 축 둘레 회전 하나뿐이다.

    왜 제약이 필요한가. 이 조립체의 자중은 약 200 g 이라 중력 신호가 2 N 인데
    센서 오프셋은 8.5 N 이다. 신호 대 오프셋이 이만큼 나쁘면 자유 회전 3 자유도는
    잡히지 않는다 — 실측에서 특이값 산포 0.30 으로 "회전이 아니다" 가 나왔다.
    미지수를 1 로 줄이면 같은 데이터에서 잔차가 0.183 → 0.047 N 으로 떨어진다.

    **미지수를 줄인 것이지 가정을 더한 것이 아니다.** 동축은 이미 확인된 사실이고,
    확인된 것을 모델에 알려 주면 남은 것을 더 잘 푼다.

    각도는 0.25° 격자로 훑는다. 미지수가 하나뿐이라 전역 최소를 놓칠 일이 없고,
    각도에 대해 비볼록한 문제를 반복법으로 푸는 것보다 이쪽이 확실하다.

    Args:
        gravity_flange: 자세별 ``{F}g`` [m/s²], ``(n, 3)``.
        forces: 자세별 측정 힘 [N], ``(n, 3)``.

    Returns:
        :class:`SensorAlignment`. ``scale_spread`` 는 여기서 **등방성 오차** 다 —
        풀어낸 3x3 이 ``m·I`` 에서 얼마나 벗어났는가.

    Raises:
        ValueError: 자세가 4 개 미만일 때.
    """
    g = np.asarray(gravity_flange, dtype=float).reshape(-1, 3)
    f = np.asarray(forces, dtype=float).reshape(-1, 3)
    n = g.shape[0]
    if n < 4:
        raise ValueError(f"정렬을 풀려면 자세가 4 개 이상 있어야 한다: {n}")

    best = None
    for degrees in np.arange(-180.0, 180.0, 0.25):
        angle = math.radians(float(degrees))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotation = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])

        rotated = (rotation @ g.T).T
        design = np.hstack([rotated, np.ones((n, 1))])
        solution, *_ = np.linalg.lstsq(design, f, rcond=None)
        scale_block = solution[:3, :]
        mass = float(np.trace(scale_block) / 3.0)

        residual = float(np.sqrt(((f - design @ solution) ** 2).mean()))
        # 질량은 방향에 상관없이 같아야 한다 — 풀어낸 블록이 m·I 에 가까울수록
        # 그 각도가 맞다. 잔차만 보면 등방성을 버린 해가 이길 수 있다.
        isotropy = float(np.linalg.norm(scale_block - mass * np.eye(3)))
        score = residual + isotropy
        if best is None or score < best[0]:
            best = (score, rotation, mass, solution[3, :].copy(), residual, isotropy)

    _, rotation, mass, bias, residual, isotropy = best
    # 두 솔버의 ``scale_spread`` 를 **같은 척도** 로 맞춘다. 자유 해에서는
    # (σmax − σmin)/σmean, 즉 배율이 방향에 따라 몇 % 다른가였다. 여기서도 같은
    # 뜻이어야 하므로, m·I 에서 벗어난 Frobenius 노름을 m·I 자신의 노름
    # (|m|·√3) 으로 나눈다. 나누는 값을 빠뜨리면 같은 이름의 두 수에 같은 문턱을
    # 적용하면서 서로 다른 것을 재게 된다.
    reference = max(abs(mass), 1e-9) * math.sqrt(3.0)
    return SensorAlignment(
        rotation_sensor_from_flange=rotation,
        mass_kg=mass,
        bias_force=bias,
        scale_spread=isotropy / reference,
        residual_n=residual,
    )


def solve_sensor_alignment(gravity_flange, forces) -> SensorAlignment:
    """센서가 플랜지에 어떻게 앉아 있는지를 **데이터에서** 푼다.

    왜 이것이 필요한가. 중력 보상은 ``{S}g`` 를 알아야 하고, 그것을 만들려면
    플랜지→센서 회전을 이미 알아야 한다. 그 회전을 단위행렬로 **가정** 하면,
    센서가 실제로 돌아 앉아 있을 때 모델 중력이 측정된 힘과 다른 방향을 가리킨다.
    상관이 없으니 최소제곱은 질량을 0 쪽으로 밀고, 실측에서 **음수 질량** 이
    나왔다. 크기는 맞는데 방향이 틀린 것이 그런 모양으로 드러난다.

    푸는 방법. 모델은

        f = m · ({S}R{F} · g_F) + b

    인데, ``A = m · {S}R{F}`` 로 묶으면 **A 에 대해 선형** 이다 (미지수 12,
    자세 하나당 식 3). 선형으로 A 와 b 를 푼 뒤 A 를 SVD 로 분해하면 특이값이
    질량이고, 가장 가까운 정규직교 행렬이 회전이다 (Procrustes).

    즉 **가정을 하나 없앤다.** 회전을 따로 재는 절차를 더하는 대신, 이미 잡은
    자세들이 그것도 함께 말하게 한다.

    Args:
        gravity_flange: 자세별 ``{F}g`` [m/s²], ``(n, 3)``.
        forces: 자세별 측정 힘 [N], ``(n, 3)``. 전자 영점을 빼기 전 원값.

    Returns:
        :class:`SensorAlignment`.

    Raises:
        ValueError: 자세가 4 개 미만이거나 중력 방향이 한 평면에 몰려 있어
            회전이 결정되지 않을 때.
    """
    g = np.asarray(gravity_flange, dtype=float).reshape(-1, 3)
    f = np.asarray(forces, dtype=float).reshape(-1, 3)
    n = g.shape[0]
    if n < 4:
        raise ValueError(f"정렬을 풀려면 자세가 4 개 이상 있어야 한다: {n}")
    if np.linalg.matrix_rank(g - g.mean(axis=0), tol=1e-6) < 3:
        raise ValueError(
            "중력 방향이 한 평면에 몰려 있다 — 회전의 한 축이 결정되지 않는다"
        )

    # [A | b] 를 한 번에. 각 자세가 [g_F | 1] 한 줄이고, 오른쪽이 측정 힘이다.
    design = np.hstack([g, np.ones((n, 1))])
    solution, *_ = np.linalg.lstsq(design, f, rcond=None)
    a_matrix = solution[:3, :].T          # (3, 3), f ≈ A·g + b
    bias = solution[3, :].copy()

    u, singular, vt = np.linalg.svd(a_matrix)
    # det 를 +1 로 강제한다. 반사가 섞이면 힘은 맞아도 모멘트가 거울상이 된다.
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    rotation = u @ correction @ vt
    mass = float(singular.mean())
    spread = float((singular.max() - singular.min()) / max(singular.mean(), 1e-9))

    predicted = (rotation * mass) @ g.T + bias.reshape(3, 1)
    residual = float(np.sqrt(((f.T - predicted) ** 2).mean()))

    return SensorAlignment(
        rotation_sensor_from_flange=rotation,
        mass_kg=mass,
        bias_force=bias,
        scale_spread=spread,
        residual_n=residual,
    )


def fit_gravity_model(
    poses,
    min_poses: int = MIN_POSES,
    min_coverage: float = MIN_COVERAGE,
    max_rms_force_n: float = 0.30,
    max_rms_torque_nm: float = 0.02,
    plausible_mass_kg: tuple = (0.02, 3.0),
    plausible_com_m: float = 0.35,
    solve_alignment: bool = True,
    max_alignment_spread: float = 0.15,
    axial_alignment: bool = True,
) -> GravityModel:
    """다자세 중력 모델을 푼다.

    Args:
        poses: :class:`CalibrationPose` 목록.
        min_poses: 최소 자세 수.
        min_coverage: 최소 자세 다양성.
        max_rms_force_n: 힘 잔차 RMS 상한.
        max_rms_torque_nm: 모멘트 잔차 RMS 상한.
        plausible_mass_kg: 질량이 이 범위를 벗어나면 물리적으로 말이 안 된다.
        plausible_com_m: COM 이 센서에서 이보다 멀면 마찬가지다.
        solve_alignment: 플랜지→센서 회전을 **가정하지 않고 함께 푼다.** 자세가
            ``gravity_flange`` 를 들고 있어야 하고, 없으면 조용히 건너뛴다.
        max_alignment_spread: 정렬 산포 상한. 넘으면 회전 하나로 설명되지 않는다.
        axial_alignment: 정렬을 **z 둘레 회전 하나** 로 제약한다. 적층이 동축이라는
            측정 사실을 쓰는 것이며, 자중이 가벼워 신호가 약할 때 이것이 결정적이다.
            거짓이면 자유 회전 3 자유도로 푼다.

    Returns:
        :class:`GravityModel`. ``valid`` 가 거짓이면 ``issues`` 에 이유가 있다.

    Raises:
        ValueError: 자세가 하나도 없다.
    """
    poses = list(poses)
    if not poses:
        raise ValueError("자세가 없다")

    wrench = np.array([np.asarray(p.wrench, dtype=float).reshape(6) for p in poses])
    n = len(poses)

    # --- 정렬: 플랜지→센서 회전을 가정하지 않고 데이터에서 푼다 -------------
    #
    # 이 단계가 없으면 아래 힘 적합은 **가정된** {S}g 를 쓴다. 그 가정이 틀리면
    # 모델 중력이 측정된 힘과 다른 방향을 가리키고, 상관이 없으니 질량이 0 쪽으로
    # 밀린다 — 실측에서 음수 질량으로 나타났다.
    rotation = np.eye(3)
    spread = 0.0
    alignment_issue = ""
    have_flange = all(p.gravity_flange is not None for p in poses)
    if solve_alignment and have_flange and n >= 4:
        flange = np.array(
            [np.asarray(p.gravity_flange, dtype=float).reshape(3) for p in poses]
        )
        solver = solve_axial_alignment if axial_alignment else solve_sensor_alignment
        try:
            alignment = solver(flange, wrench[:, :3])
        except ValueError as exc:
            alignment_issue = f"정렬을 풀 수 없다 — {exc}"
        else:
            rotation = alignment.rotation_sensor_from_flange
            spread = alignment.scale_spread
            if spread > max_alignment_spread:
                alignment_issue = (
                    f"정렬 특이값 산포 {spread:.3f} > {max_alignment_spread} — "
                    "회전 하나로 설명되지 않는다 (자세 중 하나가 흔들렸거나 "
                    "수집 중 로봇이 움직였다)"
                )
        # 이후 계산은 **푼 회전으로 만든** {S}g 를 쓴다.
        grav = np.array([rotation @ flange[i] for i in range(n)])
    else:
        grav = np.array(
            [np.asarray(p.gravity_sensor, dtype=float).reshape(3) for p in poses]
        )

    # --- 힘: [g | I] · [m, b_f] -----------------------------------------
    design_f = np.zeros((3 * n, 4))
    target_f = wrench[:, :3].reshape(-1)
    for i in range(n):
        design_f[3 * i:3 * i + 3, 0] = grav[i]
        design_f[3 * i:3 * i + 3, 1:4] = np.eye(3)
    sol_f = _irls(design_f, target_f, huber_delta=0.5)
    mass = float(sol_f[0])
    bias_f = sol_f[1:4]

    # --- 모멘트: [-[g]× | I] · [c, b_τ],  c = m·r -------------------------
    design_t = np.zeros((3 * n, 6))
    target_t = wrench[:, 3:].reshape(-1)
    for i in range(n):
        design_t[3 * i:3 * i + 3, 0:3] = -skew(grav[i])
        design_t[3 * i:3 * i + 3, 3:6] = np.eye(3)
    sol_t = _irls(design_t, target_t, huber_delta=0.05)
    first_moment = sol_t[0:3]
    bias_t = sol_t[3:6]

    com = first_moment / mass if abs(mass) > 1e-6 else np.zeros(3)
    residual_bias = np.concatenate([bias_f, bias_t])

    model = GravityModel(
        mass_kg=mass,
        com_sensor_m=com,
        residual_bias=residual_bias,
        rms_force_n=0.0,
        rms_torque_nm=0.0,
        per_axis_force_n=np.zeros(3),
        per_axis_torque_nm=np.zeros(3),
        coverage=0.0,
        poses=n,
        valid=False,
        rotation_sensor_from_flange=rotation,
        alignment_spread=spread,
    )

    # --- 잔차 ------------------------------------------------------------
    residuals = np.array([wrench[i] - model.predict(grav[i]) for i in range(n)])
    model.per_axis_force_n = np.sqrt((residuals[:, :3] ** 2).mean(axis=0))
    model.per_axis_torque_nm = np.sqrt((residuals[:, 3:] ** 2).mean(axis=0))
    model.rms_force_n = float(np.sqrt((residuals[:, :3] ** 2).mean()))
    model.rms_torque_nm = float(np.sqrt((residuals[:, 3:] ** 2).mean()))
    model.coverage, _ = pose_coverage(grav)

    # --- 판정 ------------------------------------------------------------
    issues = []
    if alignment_issue:
        issues.append(alignment_issue)
    if n < min_poses:
        issues.append(f"자세 부족 {n} < {min_poses}")
    if model.coverage < min_coverage:
        issues.append(
            f"자세 다양성 부족 {model.coverage:.3f} < {min_coverage:.2f} — "
            "중력 방향이 한쪽에 몰려 COM 이 관측되지 않는다"
        )
    lo, hi = plausible_mass_kg
    if not (lo <= mass <= hi):
        issues.append(f"질량 비현실적 {mass:.3f} kg (허용 {lo}–{hi})")
    com_norm = float(np.linalg.norm(com))
    if com_norm > plausible_com_m or not math.isfinite(com_norm):
        issues.append(f"COM 비현실적 |r| = {com_norm:.3f} m > {plausible_com_m} m")
    if model.rms_force_n > max_rms_force_n:
        issues.append(f"힘 잔차 과다 {model.rms_force_n:.3f} > {max_rms_force_n} N")
    if model.rms_torque_nm > max_rms_torque_nm:
        issues.append(f"모멘트 잔차 과다 {model.rms_torque_nm:.4f} > {max_rms_torque_nm} N·m")

    model.issues = issues
    model.valid = not issues
    return model
