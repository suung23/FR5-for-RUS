"""프로브 TCP 식별 — ``tool.j6_to_probe_xyz`` 와 ``j6_to_probe_rpy`` 를 잰다.

**무엇을 재는가.** ``us_diff_ik`` 는 FR5 6축 체인 끝에 고정 세그먼트 하나를 붙여
프로브 프레임 ``{P}`` 를 만든다 (``build_chain``). 그 세그먼트가 곧 ``{F}T{P}``
— 플랜지에서 프로브 제어점까지의 변환이고, 병진은 **플랜지 프레임 기준 미터**,
회전은 **고정축 XYZ 라디안** (``kdl.Rotation.RPY`` 규약) 이다.

**왜 재야 하는가.** 값이 없으면 노드는 ``allow_missing_tool`` 로 플랜지를 툴
프레임으로 대신 쓴다. 그러면 로봇은 **플랜지 중심으로 회전한다.** 프로브 끝이
플랜지에서 멀수록 같은 회전 지령이 프로브를 크게 휘두르므로, 조작자는 원하는
자세를 만들려고 병진을 손으로 섞게 된다 — "회전이 둔하다" 로 느껴지는 것이
이것이다. 접촉 제어에서는 더 나빠서, 레버암이 틀리면 모멘트 채널 전체가 틀린다.

절차는 두 단계이고 **둘 다 로봇을 움직이지 않는다.** 조작자가 드래그 모드로
직접 옮기고, 이 모듈은 그때의 관절각만 받는다.

**1단계 — 피벗 (병진).** 프로브의 같은 지점을 공간의 같은 고정점에 댄 채 자세만
바꿔 여러 번 잡는다. 자세 ``i`` 에서 플랜지가 ``(R_i, t_i)`` 일 때 고정점 ``c`` 는

    R_i · p + t_i = c        (모든 i 에 대해 같은 c)

이고, 미지수는 ``p`` (구하려는 오프셋) 와 ``c`` 로 6개다. 자세마다 식 3개가
나오므로 최소제곱으로 푼다.

**두 자세로는 원리적으로 풀리지 않는다.** 식 6 개에 미지수 6 개라 될 것 같지만,
두 자세를 빼면 ``(R_i - R_j)·p = t_j - t_i`` 이고 ``R_i - R_j = R_i(I - R_i^T R_j)``
인데 **회전행렬은 항상 고유값 1 을 가지므로** ``I - R`` 은 언제나 특이하다. 즉
``det(R_i - R_j) = 0`` 이 항상 성립하고, 두 자세의 상대회전축 방향 성분은 결코
구속되지 않는다. 최소는 **상대회전축이 서로 나란하지 않은 3 자세** 이며, 접촉
오차를 흡수하려면 그보다 여유가 있어야 한다.

**2단계 — 평면 접촉 (회전).** 프로브 면을 평평한 기준면에 밀착시킨다. 그러면
프로브 축 ``+z_P`` 는 그 면의 법선 반대 방향이므로, 플랜지 프레임에서 본
``z_P`` 를 자세마다 하나씩 얻는다. 남는 자유도 하나(축 둘레 회전)는 배열 방향을
알아야 정해지며, 곧은 모서리에 배열 장축을 맞추는 것으로 같은 방식으로 잰다.

**정확도의 한계는 수학이 아니라 접촉이다.** 초음파 프로브 면은 뾰족하지 않아서
"같은 점" 을 다시 대는 것이 어렵다. 면 중심을 표시하고 원뿔 팁 위에서 자세를
바꾸면 1~2 mm 급으로 잡히는데, 그 오차는 잔차로 그대로 드러난다 — 그래서 이
모듈은 값과 함께 **잔차와 자세 다양성을 항상 같이 낸다.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fr5_control.robot_backend import fr5_forward_kinematics

__all__ = [
    "MIN_PIVOT_POSES",
    "MIN_ORIENTATION_SPREAD_DEG",
    "PivotPose",
    "PivotResult",
    "AxisResult",
    "pose_from_joints",
    "orientation_spread_deg",
    "solve_pivot",
    "fit_plane",
    "axis_in_flange",
    "rotation_from_axes",
    "rpy_from_rotation",
    "rotation_from_rpy",
    "StackSegment",
    "StackResult",
    "compose_stack",
]

#: 피벗 최소 자세 수. 원리적 최소는 3 이고 (위 docstring 참조), 3 에서는 잔차가
#: 0 이 되어 버려 접촉을 잘 짚었는지 알 방법이 없다. 4 부터 잔차가 의미를 갖는다.
MIN_PIVOT_POSES = 4

#: 자세 다양성 하한 [도]. 이보다 좁으면 계수행렬이 병약해 오프셋이 안 잡힌다.
MIN_ORIENTATION_SPREAD_DEG = 30.0

#: 피벗 잔차 상한 [mm]. 이보다 크면 매번 같은 점을 대지 못한 것이다.
MAX_PIVOT_RMS_MM = 2.0

#: Huber 전환점 [mm]. 이 안쪽은 최소제곱, 바깥은 선형 — 한 자세를 헛짚어도
#: 나머지가 끌려가지 않는다.
HUBER_DELTA_MM = 1.5


@dataclass(frozen=True)
class PivotPose:
    """한 자세에서의 플랜지 위치·자세. 관절각도 함께 남긴다 (재현용)."""

    label: str
    joint_deg: tuple[float, ...]
    rotation: np.ndarray
    position: np.ndarray


@dataclass
class PivotResult:
    """피벗 해와 그것을 믿어도 되는지에 대한 근거."""

    offset_flange_m: np.ndarray
    pivot_base_m: np.ndarray
    residuals_mm: np.ndarray
    rms_mm: float
    max_mm: float
    spread_deg: float
    singular_values: np.ndarray
    issues: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """문제가 하나도 없을 때만 참."""
        return not self.issues


@dataclass
class AxisResult:
    """평면 접촉으로 얻은 축 방향과 자세 간 일치도."""

    axis_flange: np.ndarray
    spread_deg: float
    issues: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """문제가 하나도 없을 때만 참."""
        return not self.issues


def pose_from_joints(joint_deg, label: str = "") -> PivotPose:
    """관절각[도] → 플랜지 자세. 순기구학은 제어 스택과 같은 것을 쓴다."""
    joints = tuple(float(v) for v in joint_deg)
    if len(joints) != 6:
        raise ValueError(f"관절각은 6 개여야 한다: {len(joints)}")
    transform = fr5_forward_kinematics([math.radians(v) for v in joints])
    return PivotPose(
        label=label,
        joint_deg=joints,
        rotation=np.array(transform[:3, :3], dtype=float),
        position=np.array(transform[:3, 3], dtype=float),
    )


def orientation_spread_deg(rotations) -> float:
    """자세들이 서로 얼마나 벌어져 있는지 — 최대 쌍별 측지각 [도].

    조작자가 바로 행동으로 옮길 수 있는 수 하나가 필요해서 이것을 쓴다.
    작으면 "더 기울여서 한 번 더" 다.
    """
    mats = [np.asarray(r, dtype=float) for r in rotations]
    if len(mats) < 2:
        return 0.0
    worst = 0.0
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            trace = float(np.trace(mats[i] @ mats[j].T))
            angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
            worst = max(worst, angle)
    return math.degrees(worst)


def _pivot_lstsq(poses, weights):
    """가중 최소제곱 한 번. ``[p; c]`` 와 자세별 잔차[m] 를 돌려준다."""
    rows = []
    rhs = []
    for pose, weight in zip(poses, weights):
        block = np.hstack([pose.rotation, -np.eye(3)])
        rows.append(weight * block)
        rhs.append(weight * (-pose.position))
    a = np.vstack(rows)
    b = np.concatenate(rhs)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    offset, pivot = solution[:3], solution[3:]
    residuals = np.array(
        [np.linalg.norm(p.rotation @ offset + p.position - pivot) for p in poses]
    )
    return offset, pivot, residuals


def solve_pivot(
    poses,
    *,
    min_poses: int = MIN_PIVOT_POSES,
    min_spread_deg: float = MIN_ORIENTATION_SPREAD_DEG,
    max_rms_mm: float = MAX_PIVOT_RMS_MM,
    huber_delta_mm: float = HUBER_DELTA_MM,
    iterations: int = 8,
) -> PivotResult:
    """고정점을 여러 자세로 짚은 결과에서 플랜지→프로브 오프셋을 푼다.

    Args:
        poses: :class:`PivotPose` 들. 모두 **같은 물리적 점** 을 짚은 것이어야 한다.
        min_poses: 이보다 적으면 무효로 표시한다.
        min_spread_deg: 자세 다양성 하한.
        max_rms_mm: 잔차 RMS 상한.
        huber_delta_mm: Huber 전환점.
        iterations: IRLS 반복 수.

    Returns:
        해와 잔차와 문제 목록. **문제가 있어도 값은 돌려준다** — 조작자가 무엇이
        모자란지 보고 자세를 더 잡을 수 있어야 하기 때문이다.
    """
    poses = list(poses)
    if len(poses) < 2:
        raise ValueError("피벗은 최소 2 자세가 있어야 계수행렬을 세울 수 있다")

    weights = np.ones(len(poses))
    offset = pivot = residuals = None
    for _ in range(iterations):
        offset, pivot, residuals = _pivot_lstsq(poses, weights)
        scale = huber_delta_mm / 1000.0
        weights = np.where(
            residuals <= scale,
            1.0,
            np.sqrt(scale / np.maximum(residuals, 1e-12)),
        )

    residuals_mm = residuals * 1000.0
    rms = float(np.sqrt(np.mean(residuals_mm**2)))
    spread = orientation_spread_deg([p.rotation for p in poses])

    stacked = np.vstack([np.hstack([p.rotation, -np.eye(3)]) for p in poses])
    singular = np.linalg.svd(stacked, compute_uv=False)

    issues: list[str] = []
    if len(poses) < min_poses:
        issues.append(f"자세가 {len(poses)} 개다. {min_poses} 개 이상 필요하다")
    if spread < min_spread_deg:
        issues.append(
            f"자세 다양성 {spread:.0f}° 로 좁다 ({min_spread_deg:.0f}° 이상 필요). "
            "더 크게 기울여서 잡아라"
        )
    if singular[-1] < 1e-6:
        issues.append(
            "계수행렬이 특이하다 — 오프셋의 한 방향이 구속되지 않는다. "
            "자세가 2 개뿐이거나, 상대회전축이 모두 나란하다. 축을 바꿔 기울여라"
        )
    if rms > max_rms_mm:
        issues.append(
            f"잔차 RMS {rms:.2f} mm 가 크다 ({max_rms_mm:.1f} mm 이하여야 한다). "
            "매번 같은 점을 짚지 못한 것이다"
        )

    return PivotResult(
        offset_flange_m=offset,
        pivot_base_m=pivot,
        residuals_mm=residuals_mm,
        rms_mm=rms,
        max_mm=float(np.max(residuals_mm)),
        spread_deg=spread,
        singular_values=singular,
        issues=issues,
    )


def fit_plane(points):
    """점들에 평면을 맞춘다. ``(법선, 평면 위 한 점, 잔차 RMS [mm])``.

    법선은 **+z_B 쪽** 으로 맞춰 돌려준다. 부호가 자세마다 뒤집히면 그것을 쓰는
    쪽에서 축이 뒤집히기 때문이다.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("평면을 맞추려면 점이 3 개 이상 있어야 한다")
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    residual = float(np.sqrt(np.mean(((pts - centroid) @ normal) ** 2)) * 1000.0)
    return normal, centroid, residual


def axis_in_flange(rotations, direction_base, *, max_spread_deg: float = 5.0) -> AxisResult:
    """기준면·기준모서리에 맞춘 자세들에서 그 방향을 플랜지 프레임으로 옮긴다.

    Args:
        rotations: 밀착시킨 자세들의 ``{B}R{F}``.
        direction_base: 그 자세에서 프로브 축이 향하는 베이스 프레임 방향.
            평면 밀착이면 **바깥 법선의 반대** 다 — ``+z_P`` 는 조직으로 파고드는
            방향이므로 면을 향한다.
        max_spread_deg: 자세 간 불일치 상한.

    Returns:
        평균 축과 자세 간 최대 이탈각. 이탈이 크면 밀착이 안 된 자세가 있다.
    """
    mats = [np.asarray(r, dtype=float) for r in rotations]
    if not mats:
        raise ValueError("자세가 하나도 없다")
    target = np.asarray(direction_base, dtype=float)
    target = target / np.linalg.norm(target)

    axes = np.array([r.T @ target for r in mats])
    mean = axes.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-9:
        raise ValueError("축들이 서로 상쇄된다 — 자세 기록이 잘못됐다")
    mean = mean / norm

    spread = math.degrees(
        max(math.acos(max(-1.0, min(1.0, float(a @ mean)))) for a in axes)
    )

    issues: list[str] = []
    if len(mats) < 2:
        issues.append("자세가 1 개다. 최소 2 개는 잡아야 일치도를 볼 수 있다")
    elif spread > max_spread_deg:
        issues.append(
            f"자세 간 축 불일치 {spread:.1f}° ({max_spread_deg:.0f}° 이하여야 한다). "
            "밀착되지 않은 자세가 있다"
        )
    return AxisResult(axis_flange=mean, spread_deg=spread, issues=issues)


def rotation_from_axes(z_axis, x_hint) -> np.ndarray:
    """프로브 z 축과 배열 방향 힌트로 ``{F}R{P}`` 를 만든다.

    ``x_hint`` 는 z 축에 직교하지 않아도 된다 — 직교 성분만 쓴다. 측정으로 얻은
    두 방향이 정확히 직교할 리 없고, 억지로 직교화하지 않으면 회전행렬이 아니게
    된다.
    """
    z = np.asarray(z_axis, dtype=float)
    z = z / np.linalg.norm(z)
    hint = np.asarray(x_hint, dtype=float)
    x = hint - float(hint @ z) * z
    norm = np.linalg.norm(x)
    if norm < 1e-6:
        raise ValueError("배열 방향이 프로브 축과 거의 나란하다 — 힌트가 못 쓸 값이다")
    x = x / norm
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def rpy_from_rotation(rotation) -> tuple[float, float, float]:
    """``{F}R{P}`` → 고정축 XYZ ``(roll, pitch, yaw)`` [라디안].

    ``kdl.Rotation.RPY`` 와 같은 규약이다. ``probe.yaml`` 에 그대로 들어간다.
    """
    r = np.asarray(rotation, dtype=float)
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:  # 짐벌락: roll 을 0 으로 두고 yaw 로 몰아준다
        return 0.0, pitch, math.atan2(-r[0, 1], r[1, 1])
    return math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0])


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """고정축 XYZ → 회전행렬. :func:`rpy_from_rotation` 의 역이다."""
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


# ======================================================================
# 실측 치수로 조립하기
# ======================================================================
#
# 피벗은 로봇 프레임에서 벡터 전체를 주지만 접촉 정확도가 한계다. 캘리퍼는
# 길이에 강하지만 플랜지 프레임의 x/y 중 어디를 향하는지는 알려주지 않는다.
# 둘은 서로를 검증하는 관계이고, 아래는 캘리퍼 쪽이다.
#
# **세 파라미터를 하나의 기술에서 뽑는 것이 요점이다.** 적층은
#
#     플랜지 → 마운트① → 센서 → 마운트② → 프로브
#
# 이고, probe.yaml 의 ``tool.j6_to_probe_*``, ``ft_sensor.j6_to_sensor_*``,
# 그리고 렌치 보상의 레버암 ``r_{S→P}`` 는 모두 이 하나의 사슬 위의 서로 다른
# 구간이다. 따로 적어 넣으면 조용히 어긋날 수 있다 — 어긋나도 각각은 그럴듯해
# 보이므로 알아채기 어렵다. 여기서는 같은 사슬에서 계산하므로 어긋날 수 없다.


@dataclass(frozen=True)
class StackSegment:
    """적층의 한 마디. 변환은 **직전 마디의 프레임 기준** 이다.

    ``kdl.Frame(R, p)`` 와 같은 규약으로, 동차변환 ``[R | p]`` 는 벡터를
    ``R·v + p`` 로 옮긴다. 즉 **``xyz_mm`` 은 회전하기 전 프레임에서 잰 값**
    이다 — 회전한 축을 따라 나가는 값이 아니다. 꺾인 마운트를 적을 때 이 둘을
    바꿔 적으면 길이는 맞는데 방향이 틀린, 알아채기 어려운 오류가 된다.

    꺾인 뒤 그 방향으로 나가는 것을 적으려면 마디를 둘로 나눈다 — 회전만 하는
    마디 하나, 그 다음 프레임에서 병진하는 마디 하나.
    """

    name: str
    xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def matrix(self) -> np.ndarray:
        """이 마디의 4x4 동차변환."""
        transform = np.eye(4)
        transform[:3, :3] = rotation_from_rpy(*[math.radians(v) for v in self.rpy_deg])
        transform[:3, 3] = np.array(self.xyz_mm, dtype=float) / 1000.0
        return transform


@dataclass
class StackResult:
    """적층에서 뽑은 probe.yaml 값들. 서로 정합적임이 구조적으로 보장된다."""

    j6_to_sensor_xyz: np.ndarray
    j6_to_sensor_rpy: np.ndarray
    j6_to_probe_xyz: np.ndarray
    j6_to_probe_rpy: np.ndarray
    r_sensor_to_probe_m: np.ndarray
    frames: dict


def compose_stack(
    segments,
    *,
    sensor_frame: str = "sensor",
    probe_frame: str = "probe",
) -> StackResult:
    """실측 치수 사슬에서 세 파라미터를 한꺼번에 뽑는다.

    Args:
        segments: 플랜지에서 시작하는 :class:`StackSegment` 들, 순서대로.
        sensor_frame: 센서 측정 프레임에 해당하는 마디 이름.
        probe_frame: 프로브 제어점에 해당하는 마디 이름.

    Returns:
        :class:`StackResult`. ``r_sensor_to_probe_m`` 은 **프로브 프레임 기준**
        으로, ``wrench_translate`` 가 요구하는 형태다.

    Raises:
        KeyError: 지정한 이름의 마디가 사슬에 없을 때.
    """
    frames: dict = {}
    cumulative = np.eye(4)
    for segment in segments:
        cumulative = cumulative @ segment.matrix()
        frames[segment.name] = cumulative.copy()

    for name in (sensor_frame, probe_frame):
        if name not in frames:
            raise KeyError(f"사슬에 '{name}' 마디가 없다: {list(frames)}")

    sensor = frames[sensor_frame]
    probe = frames[probe_frame]

    # 센서 → 프로브 병진을 프로브 프레임에서 본 것. 레버암은 그 프레임에서 쓴다.
    lever_probe = probe[:3, :3].T @ (probe[:3, 3] - sensor[:3, 3])

    return StackResult(
        j6_to_sensor_xyz=sensor[:3, 3].copy(),
        j6_to_sensor_rpy=np.array(rpy_from_rotation(sensor[:3, :3])),
        j6_to_probe_xyz=probe[:3, 3].copy(),
        j6_to_probe_rpy=np.array(rpy_from_rotation(probe[:3, :3])),
        r_sensor_to_probe_m=lever_probe,
        frames=frames,
    )
