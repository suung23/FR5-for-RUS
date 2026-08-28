"""robot 값과 전자저울 ground truth 를 맞추고 지표를 낸다.

ROS 를 쓰지 않는다. 측정한 기계가 아닌 곳에서도 돌아가야 하고, 분석이 로봇에
붙어 있어야만 돌아가면 결과를 나중에 다시 볼 수 없다.

**임의로 만들지 않고 임의로 버리지 않는다.** 빠진 값은 빠진 대로 제외 사유와 함께
남고, 보정처럼 보이는 계산은 하지 않는다. 특히 중력보상 잔차는 robot 값에서 다시
빼지 않는다 — 그것을 빼면 최종 잔차가 인위적으로 작아지고, 그 수는 아무것도
뜻하지 않게 된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .tables import InputError, as_bool, as_float, index_by_sample, read_rows, require_columns

#: 표준중력. 전자저울 질량을 힘으로 바꿀 때 쓴다.
G = 9.80665

ROBOT_COLUMNS = (
    "sample_id",
    "robot_normal_force_instant_N",
    "sensor_calibration_valid",
    "gravity_compensation_valid",
    "probe_frame_transform_valid",
    "capture_valid",
)

TRIAL_COLUMNS = (
    "sample_id",
    "scale_mass_g",
    "scale_reference_force_N",
    "robot_normal_force_instant_N",
    "residual_robot_minus_scale_N",
    "estimated_gravity_error_normal_N",
    "estimated_gravity_uncertainty_normal_N",
    "scale_standard_uncertainty_N",
    "combined_reference_uncertainty_N",
    "included_in_primary_analysis",
    "exclusion_reason",
)


@dataclass
class Trial:
    """한 capture 와 그에 짝지어진 전자저울 값."""

    sample_id: int
    robot_n: float | None = None
    scale_mass_g: float | None = None
    scale_n: float | None = None
    residual_n: float | None = None
    gravity_error_n: float | None = None
    gravity_uncertainty_n: float | None = None
    scale_uncertainty_n: float = 0.0
    combined_uncertainty_n: float | None = None
    included: bool = False
    exclusion_reason: str = ""
    orientation: tuple | None = None
    std_250ms_n: float | None = None
    data_age_ms: float | None = None


@dataclass
class GravityMetrics:
    """중력보상 성능. 자료가 없으면 :attr:`available` 이 거짓이다."""

    available: bool = False
    n_poses: int = 0
    bias_n: float = 0.0
    sd_n: float = 0.0
    rmse_n: float = 0.0
    max_abs_n: float = 0.0
    lower_95_n: float = 0.0
    upper_95_n: float = 0.0
    robust_2p5_n: float = 0.0
    robust_97p5_n: float = 0.0
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    orientations: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    pose_ids: list = field(default_factory=list)
    method: str = "unavailable"
    note: str = ""
    #: 자세 순서에 대한 추세 — Spearman 상관과 p 값.
    #:
    #: 잔차가 무작위이면 순서와 상관이 없어야 한다. 상관이 있으면 계통 성분이
    #: 남아 있다는 뜻이고, 그것은 SD 로 요약할 수 있는 잡음이 아니다. 사양이
    #: "pose 에 따른 residual trend" 를 요구하는 이유다.
    trend_rho: float = float("nan")
    trend_p: float = float("nan")


def load_gravity_validation(path: str) -> GravityMetrics:
    """무부하 다자세 검증 결과를 읽어 중력보상 지표를 낸다.

    Raises:
        InputError: 파일이 못 쓸 상태일 때. 파일이 **없는 것** 은 여기서 다루지
            않는다 — 호출한 쪽이 "중력 불확도 없음" 으로 계속 갈 수 있어야 한다.
    """
    rows = read_rows(path)
    require_columns(rows, ("pose_id", "gravity_residual_Fz_P_N"), path)

    residuals = []
    orientations = []
    pose_ids = []
    for row in rows:
        pose = (row.get("pose_id") or "").strip()
        if as_bool(row.get("valid_pose"), True) is False:
            continue
        value = as_float(row.get("gravity_residual_Fz_P_N"), "gravity_residual_Fz_P_N",
                         pose, allow_blank=True)
        if value is None:
            continue
        residuals.append(value)
        pose_ids.append(pose)
        angles = [
            as_float(row.get(name), name, pose, allow_blank=True)
            for name in ("tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad")
        ]
        orientations.append(angles if all(a is not None for a in angles) else None)

    if not residuals:
        raise InputError(
            f"{path}: 쓸 수 있는 자세가 없다 — gravity_residual_Fz_P_N 이 모두 "
            "비었거나 valid_pose 가 모두 FALSE 다"
        )

    values = np.asarray(residuals, dtype=float)
    have_orientation = all(item is not None for item in orientations)
    metrics = GravityMetrics(
        available=True,
        n_poses=int(values.size),
        bias_n=float(values.mean()),
        # 표본 표준편차. 자세 몇 개로 모집단 SD 를 말하지 않는다.
        sd_n=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        rmse_n=float(np.sqrt((values ** 2).mean())),
        max_abs_n=float(np.abs(values).max()),
        residuals=values,
        orientations=(
            np.asarray(orientations, dtype=float) if have_orientation
            else np.zeros((values.size, 3))
        ),
        pose_ids=pose_ids,
    )
    metrics.lower_95_n = metrics.bias_n - 1.96 * metrics.sd_n
    metrics.upper_95_n = metrics.bias_n + 1.96 * metrics.sd_n
    metrics.robust_2p5_n = float(np.percentile(values, 2.5))
    metrics.robust_97p5_n = float(np.percentile(values, 97.5))
    if values.size >= 4:
        from scipy import stats

        rho, pvalue = stats.spearmanr(np.arange(values.size), values)
        metrics.trend_rho = float(rho)
        metrics.trend_p = float(pvalue)
    return metrics


def assign_gravity_error(trials, gravity: GravityMetrics, orientation_available: bool,
                         max_angle_deg: float = 30.0) -> str:
    """Capture 마다 예상 중력보상 오차와 불확도를 붙인다.

    자세 정보가 양쪽에 다 있으면 **가장 가까운 검증 자세** 의 잔차를 쓴다. 보간을
    쓰지 않는 이유는 자료가 적기 때문이다 — 열몇 개 자세에 곡면을 맞추면 자료가
    말하지 않은 것을 말하게 된다. 사양도 자료가 부족하면 보간을 강요하지 말라고
    한다.

    자세를 못 쓰면 전체 잔차의 표준편차를 모든 capture 에 똑같이 준다.

    Returns:
        쓴 방법 — ``"pose-specific"`` 또는 ``"global"``.
    """
    if not gravity.available:
        for trial in trials:
            trial.gravity_error_n = None
            trial.gravity_uncertainty_n = None
        return "unavailable"

    usable_poses = orientation_available and bool(np.any(gravity.orientations))
    if not usable_poses:
        for trial in trials:
            trial.gravity_error_n = gravity.bias_n
            trial.gravity_uncertainty_n = gravity.sd_n
        return "global"

    limit = math.radians(max_angle_deg)
    used_pose_specific = False
    for trial in trials:
        if trial.orientation is None:
            trial.gravity_error_n = gravity.bias_n
            trial.gravity_uncertainty_n = gravity.sd_n
            continue
        here = np.asarray(trial.orientation, dtype=float)
        distances = np.linalg.norm(gravity.orientations - here, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= limit:
            trial.gravity_error_n = float(gravity.residuals[nearest])
            trial.gravity_uncertainty_n = gravity.sd_n
            used_pose_specific = True
        else:
            # 가까운 자세가 없으면 전체값으로 떨어진다. 먼 자세의 잔차를 끌어다
            # 쓰는 것은 추정이 아니라 추측이다.
            trial.gravity_error_n = gravity.bias_n
            trial.gravity_uncertainty_n = gravity.sd_n
    return "pose-specific" if used_pose_specific else "global"


def build_trials(robot_path: str, scale_path: str) -> list:
    """Capture 와 전자저울 표를 sample_id 로 맞춘다.

    Raises:
        InputError: 열이 없거나, id 가 중복이거나, 전자저울 표에 capture 에 없는
            id 가 있을 때. 후자는 표가 서로 다른 실험이라는 뜻이다.
    """
    robot_rows = read_rows(robot_path)
    require_columns(robot_rows, ROBOT_COLUMNS, robot_path)
    robot = index_by_sample(robot_rows, robot_path)

    scale_rows = read_rows(scale_path)
    require_columns(scale_rows, ("sample_id", "scale_mass_g"), scale_path)
    scale = index_by_sample(scale_rows, scale_path)

    unknown = sorted(set(scale) - set(robot))
    if unknown:
        raise InputError(
            f"{scale_path} 에 capture 에 없는 sample_id 가 있다: {unknown}\n"
            "  두 표가 같은 실험의 것인지 확인하라."
        )

    trials = []
    for key in sorted(robot):
        row = robot[key]
        trial = Trial(sample_id=key)

        angles = [
            as_float(row.get(name), name, str(key), allow_blank=True)
            for name in ("tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad")
        ]
        if all(a is not None for a in angles):
            trial.orientation = tuple(angles)
        trial.std_250ms_n = as_float(
            row.get("robot_normal_force_std_250ms_N"),
            "robot_normal_force_std_250ms_N", str(key), allow_blank=True,
        )
        trial.data_age_ms = as_float(
            row.get("force_data_age_ms"), "force_data_age_ms", str(key), allow_blank=True,
        )
        trial.robot_n = as_float(
            row.get("robot_normal_force_instant_N"),
            "robot_normal_force_instant_N", str(key), allow_blank=True,
        )

        reasons = []
        if as_bool(row.get("capture_valid"), True) is False:
            reasons.append("capture_valid=FALSE")
        for flag in ("sensor_calibration_valid", "gravity_compensation_valid",
                     "probe_frame_transform_valid"):
            if as_bool(row.get(flag), True) is False:
                reasons.append(f"{flag}=FALSE")
        if trial.robot_n is None:
            reasons.append("robot force 없음")

        scale_row = scale.get(key)
        if scale_row is None:
            reasons.append("전자저울 값 미입력")
        else:
            if as_bool(scale_row.get("valid_trial"), True) is False:
                reasons.append("valid_trial=FALSE")
            mass = as_float(scale_row.get("scale_mass_g"), "scale_mass_g", str(key),
                            allow_blank=True)
            if mass is None:
                reasons.append("scale_mass_g 비어 있음")
            else:
                if mass < 0:
                    raise InputError(
                        f"sample {key}: scale_mass_g 가 음수다 ({mass}). "
                        "전자저울 질량은 g 단위 양수여야 한다."
                    )
                if mass > 20000:
                    raise InputError(
                        f"sample {key}: scale_mass_g 가 {mass} 다 — kg 을 g 로 "
                        "잘못 적었을 수 있다 (20 kg 초과)."
                    )
                trial.scale_mass_g = mass
                trial.scale_n = mass / 1000.0 * G

        if trial.robot_n is not None and trial.scale_n is not None:
            trial.residual_n = trial.robot_n - trial.scale_n

        trial.included = not reasons
        trial.exclusion_reason = "; ".join(reasons)
        trials.append(trial)
    return trials


def scale_standard_uncertainty(resolution_g: float) -> float:
    """전자저울 해상도에서 오는 표준불확도 [N].

    균등분포의 표준편차, 곧 ``Δ/√12`` 다. 해상도는 g 단위로 받아 힘으로 바꾼다.
    """
    return (resolution_g / 1000.0 * G) / math.sqrt(12.0)


@dataclass
class Accuracy:
    """robot 과 전자저울의 일치도."""

    n: int
    bias_n: float
    mae_n: float
    rmse_n: float
    max_abs_n: float
    residual_sd_n: float
    full_scale_n: float
    mae_pct_fs: float
    rmse_pct_fs: float
    max_pct_fs: float
    slope: float
    intercept: float
    r_squared: float
    slope_ci: tuple
    intercept_ci: tuple
    ba_bias_n: float
    ba_lower_n: float
    ba_upper_n: float


def accuracy_metrics(trials) -> Accuracy | None:
    """유효한 trial 만으로 정확도를 계산한다. 두 개 미만이면 ``None``."""
    used = [t for t in trials if t.included]
    if len(used) < 2:
        return None

    robot = np.array([t.robot_n for t in used], dtype=float)
    scale = np.array([t.scale_n for t in used], dtype=float)
    error = robot - scale
    n = error.size

    # 전 구간(full scale)은 기준값의 최대치로 둔다. 0 이면 %FS 는 뜻이 없다.
    full_scale = float(scale.max()) if scale.size else 0.0

    design = np.vstack([scale, np.ones_like(scale)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, robot, rcond=None)
    predicted = design @ np.array([slope, intercept])
    ss_res = float(((robot - predicted) ** 2).sum())
    ss_tot = float(((robot - robot.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    slope_ci = (float("nan"), float("nan"))
    intercept_ci = (float("nan"), float("nan"))
    if n > 2:
        from scipy import stats

        dof = n - 2
        mse = ss_res / dof
        cov = mse * np.linalg.inv(design.T @ design)
        t_crit = float(stats.t.ppf(0.975, dof))
        slope_se, intercept_se = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
        slope_ci = (slope - t_crit * slope_se, slope + t_crit * slope_se)
        intercept_ci = (intercept - t_crit * intercept_se, intercept + t_crit * intercept_se)

    ba_bias = float(error.mean())
    ba_sd = float(error.std(ddof=1)) if n > 1 else 0.0

    # %FS 는 기준 최대값이 0 이면 뜻이 없다. 0 으로 나누는 대신 NaN 을 남기고,
    # 보고서는 그것을 대시로 찍는다.
    def percent(value):
        return float(value / full_scale * 100.0) if full_scale else float("nan")

    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt((error ** 2).mean()))
    max_abs = float(np.abs(error).max())
    return Accuracy(
        n=n,
        bias_n=ba_bias,
        mae_n=mae,
        rmse_n=rmse,
        max_abs_n=max_abs,
        residual_sd_n=ba_sd,
        full_scale_n=full_scale,
        mae_pct_fs=percent(mae),
        rmse_pct_fs=percent(rmse),
        max_pct_fs=percent(max_abs),
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        slope_ci=slope_ci,
        intercept_ci=intercept_ci,
        ba_bias_n=ba_bias,
        ba_lower_n=ba_bias - 1.96 * ba_sd,
        ba_upper_n=ba_bias + 1.96 * ba_sd,
    )


def repeatability(trials, tolerance_n: float = 0.25) -> list:
    """같은 목표 힘을 여러 번 잰 것이 있으면 그 반복성을 낸다.

    목표 힘 수준을 따로 기록하지 않으므로, 전자저울 기준값이 서로
    ``tolerance_n`` 안쪽인 것들을 한 수준으로 묶는다. 두 번 이상 잰 수준만 낸다 —
    한 번 잰 것에 SD 를 붙이면 반복성이 아니라 잡음이다.

    Returns:
        ``(수준 평균 [N], 개수, SD [N], CV [%])`` 목록.
    """
    used = sorted([t for t in trials if t.included], key=lambda t: t.scale_n)
    groups: list = []
    for trial in used:
        if groups and abs(trial.scale_n - groups[-1][0][0]) <= tolerance_n:
            groups[-1].append((trial.scale_n, trial.robot_n))
        else:
            groups.append([(trial.scale_n, trial.robot_n)])

    out = []
    for group in groups:
        if len(group) < 2:
            continue
        robot = np.array([item[1] for item in group], dtype=float)
        level = float(np.mean([item[0] for item in group]))
        sd = float(robot.std(ddof=1))
        mean = float(robot.mean())
        cv = sd / abs(mean) * 100.0 if abs(mean) > 1e-9 else float("nan")
        out.append((level, len(group), sd, cv))
    return out


def finalise_uncertainty(trials, gravity: GravityMetrics, resolution_g: float) -> float:
    """전자저울·중력보상 불확도를 trial 마다 채우고 결합값을 돌려준다.

    **잔차를 고치는 값이 아니다.** 결과를 읽을 때의 폭일 뿐이며, 어디에서도
    robot 값에서 빼지 않는다.
    """
    u_scale = scale_standard_uncertainty(resolution_g)
    u_gravity = gravity.sd_n if gravity.available else 0.0
    combined = math.sqrt(u_scale ** 2 + u_gravity ** 2)
    for trial in trials:
        trial.scale_uncertainty_n = u_scale
        trial.combined_uncertainty_n = math.sqrt(
            u_scale ** 2 + (trial.gravity_uncertainty_n or 0.0) ** 2
        )
    return combined
