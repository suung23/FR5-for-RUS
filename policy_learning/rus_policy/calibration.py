"""⏳ 상수 두 개를 데이터로 채우는 도구 (2026-09-08, POLICY_LEARNING_MATH §7.1 · README §3).

1. ``estimate_us_latency``  — `timing.us_latency_s`.
   두 스트림에 동시에 찍히는 물리 사건을 쓴다: 프로브를 젤에서 급히 떼면 (또는 갖다 대면) 영상 평균강도가
   급변하고 가속도가 튄다. 영상 평균강도 미분의 절대값과 고역통과한 |가속도| 를 IMU 시간축에서 상호상관해
   피크 지연을 잡는다. **양수 = 영상이 IMU 보다 늦다** = `us_latency_s` 에 그대로 넣는 값
   (dataset.py 가 `frame_t − us_latency_s` 로 뺀다). 세션 하나에 사건 10–20 회면 충분하다.

2. ``estimate_sensor_to_probe`` — `imu.sensor_to_probe` (R_SP).
   프로브를 **프로브 축 x → y → z 순서로** 각각 오른손 양의 방향으로 크게 돌리고 사이사이 정지한다.
   회전 구간마다 자이로 표본의 주축 (SVD 첫 특이벡터, 부호는 평균 회전 방향) 이 그 프로브 축의
   센서 프레임 표현 ŝ 이고,  ω_P = R_SP ω_S  이므로  R_SP 의 **행** 이 ŝ_x, ŝ_y, ŝ_z 다.
   세 행을 쌓고 가장 가까운 회전행렬로 투영한다. 축 사이 각도 잔차와 det 를 같이 낸다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["LatencyEstimate", "estimate_us_latency", "estimate_sensor_to_probe", "SensorToProbeEstimate"]


# --------------------------------------------------------------------------- 1. 지연
@dataclass
class LatencyEstimate:
    latency_s: float           # + 면 영상이 늦다
    peak_corr: float           # 정규화 상호상관 피크
    second_ratio: float        # 피크 / 두 번째 국소 피크 (1 에 가까우면 애매)
    n_image_events: int        # 영상 급변 횟수 (임계 초과)
    n_imu_events: int
    lags_s: np.ndarray
    corr: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {"latency_s": self.latency_s, "peak_corr": self.peak_corr, "second_ratio": self.second_ratio,
                "n_image_events": self.n_image_events, "n_imu_events": self.n_imu_events}


def _highpass_abs(x: np.ndarray, t: np.ndarray, win_s: float) -> np.ndarray:
    """|x − 이동평균| (창 win_s). 중력·저주파 자세 변화를 지운다."""
    n = x.size
    if n < 3:
        return np.abs(x - x.mean())
    dt = float(np.median(np.diff(t)))
    k = max(3, int(round(win_s / max(dt, 1e-6))))
    cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(n)
    lo = np.maximum(0, i - k // 2)
    hi = np.minimum(n, i + k // 2 + 1)
    mean = (cs[hi] - cs[lo]) / (hi - lo)
    return np.abs(x - mean)


def _standardize(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def estimate_us_latency(frame_t: np.ndarray, frame_mean: np.ndarray, imu_t: np.ndarray, acc: np.ndarray,
                        max_lag_s: float = 1.0, hp_win_s: float = 0.5, event_z: float = 3.0) -> LatencyEstimate:
    """영상 평균강도열과 가속도로 US 지연을 추정한다. 시각은 둘 다 같은 시계 (pc) 여야 한다."""
    frame_t = np.asarray(frame_t, float)
    frame_mean = np.asarray(frame_mean, float)
    imu_t = np.asarray(imu_t, float)
    acc = np.asarray(acc, float)
    if frame_t.size < 4 or imu_t.size < 8:
        raise ValueError("프레임 4 장, IMU 8 표본 이상이 필요합니다")
    a_mag = np.linalg.norm(acc, axis=1) if acc.ndim == 2 else np.abs(acc)
    imu_ev = _highpass_abs(a_mag, imu_t, hp_win_s)
    # 영상 사건: 평균강도 미분의 절대값 (프레임 간격으로 나눈다), 프레임 중간 시각에 놓는다
    dI = np.abs(np.diff(frame_mean)) / np.maximum(np.diff(frame_t), 1e-6)
    t_mid = 0.5 * (frame_t[1:] + frame_t[:-1])
    # IMU 시간축으로 (영점 유지 보간: 사건은 두 프레임 사이 어딘가 — 그 구간 전체에 편다)
    img_ev = np.interp(imu_t, t_mid, dI, left=0.0, right=0.0)
    x = _standardize(img_ev)
    y = _standardize(imu_ev)
    dt = float(np.median(np.diff(imu_t)))
    max_lag = max(1, int(round(max_lag_s / max(dt, 1e-6))))
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(lags.size)
    n = x.size
    for j, L in enumerate(lags):
        # L > 0: 영상 신호를 L 만큼 앞으로 당겨 IMU 와 맞춘다 (영상이 늦다)
        if L >= 0:
            a, b = x[L:], y[: n - L]
        else:
            a, b = x[: n + L], y[-L:]
        corr[j] = float((a * b).mean()) if a.size > 8 else 0.0
    k = int(np.argmax(corr))
    peak = float(corr[k])
    # 두 번째 국소 피크
    local = [j for j in range(1, corr.size - 1) if corr[j] >= corr[j - 1] and corr[j] >= corr[j + 1] and j != k]
    second = max((corr[j] for j in local), default=0.0)
    ratio = float(peak / second) if second > 1e-9 else float("inf")
    thr_i = np.mean(dI) + event_z * np.std(dI)
    thr_m = np.mean(imu_ev) + event_z * np.std(imu_ev)
    return LatencyEstimate(latency_s=float(lags[k] * dt), peak_corr=peak, second_ratio=ratio,
                           n_image_events=int((dI > thr_i).sum()), n_imu_events=int((imu_ev > thr_m).sum()),
                           lags_s=lags * dt, corr=corr)


# --------------------------------------------------------------------------- 2. R_SP
@dataclass
class SensorToProbeEstimate:
    R_sp: np.ndarray                 # (3,3) 가장 가까운 회전행렬
    axes_sensor: np.ndarray          # (3,3) 원시 주축 (행 = x,y,z 회전의 센서 프레임 방향)
    angles_deg: np.ndarray           # 원시 주축 사이 각도 (xy, yz, zx) — 90 에 가까워야 한다
    det: float
    residual_deg: float              # 투영 전후 행 벡터의 최대 각도 차

    def as_yaml_rows(self) -> list[list[float]]:
        return [[round(float(v), 6) for v in row] for row in self.R_sp]


def principal_rotation_axis(gyr: np.ndarray) -> np.ndarray:
    """(n,3) 자이로 표본 → 단위 주축. 부호는 평균 회전 방향 (오른손 양의 회전이 + 가 되게)."""
    g = np.asarray(gyr, float)
    if g.ndim != 2 or g.shape[0] < 3:
        raise ValueError("자이로 표본은 (n≥3, 3) 이어야 합니다")
    _, _, vt = np.linalg.svd(g - 0.0, full_matrices=False)   # 원점 기준 (평균을 빼지 않는다: 방향이 정보)
    axis = vt[0]
    if (g @ axis).sum() < 0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def estimate_sensor_to_probe(gyr_segments: Sequence[np.ndarray]) -> SensorToProbeEstimate:
    """세 회전 구간 (프로브 x, y, z 축 순) 의 자이로 표본 → R_SP."""
    if len(gyr_segments) != 3:
        raise ValueError("회전 구간은 정확히 세 개 (x, y, z 순) 여야 합니다")
    rows = np.stack([principal_rotation_axis(g) for g in gyr_segments])      # ŝ_x, ŝ_y, ŝ_z
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))  # noqa: E731
    angles = np.array([ang(rows[0], rows[1]), ang(rows[1], rows[2]), ang(rows[2], rows[0])])
    U, _, Vt = np.linalg.svd(rows)
    R = U @ Vt
    if np.linalg.det(R) < 0:                       # 반사 → 마지막 특이방향 뒤집기
        U[:, -1] *= -1
        R = U @ Vt
    resid = max(ang(rows[i], R[i]) for i in range(3))
    return SensorToProbeEstimate(R_sp=R, axes_sensor=rows, angles_deg=angles, det=float(np.linalg.det(R)),
                                 residual_deg=resid)
