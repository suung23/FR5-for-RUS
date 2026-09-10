"""IMU → chunk 라벨. ``docs/POLICY_LEARNING_MATH.md`` §2 · §5.2 의 구현.

원칙 (L3): 라벨은 **정지 → 이동 → 정지로 괄호 친 구간의 순변위**다. 연속 궤적에서
per-step 속도 라벨은 만들지 않는다. 구간 하나 = ACT chunk 하나.

파이프라인 (imu_bench/qc_track/analyze_track.py 의 C 단계와 같다):

    still_mask        이동창 자이로·가속도 산포로 정지 판정  (qc_common.still_mask 와 동일 기준, 벡터화)
    move_segments     정지-이동-정지 괄호. 앵커는 정지 구간 끝쪽 25 % 안
    earth_accel       a_E = R_SE(q)·a_S,  규약(R vs Rᵀ)은 데이터로 판정 (zero_ref._earth_convention)
    bias              앵커 정지 구간의 a_E 평균 (중력 포함) 을 뺀다
    zupt_integrate    사다리꼴 이중적분 + 끝 속도 0 선형 디드리프트
    probe frame       앵커 시점의 프로브 프레임으로 회전  P_P = R_SP · R_SE(q_a)ᵀ · P_E
    chunk grid        t_a + i/f_p (i = 0..k) 에서 표본화. 이동이 끝나면 값이 유지된다

라벨 단위: 병진 **mm**, 회전 **deg**. 축 순서 (x lateral, y elevational, z beam).
policy 가 쥔 3축은 (v_x, v_y, ω_z) → P[:, (0, 1, 5)] 이고, 힘축 (z, ω_x, ω_y) 는
관측·진단용으로만 남긴다 (§5.3b masking).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from .config import ImuConfig, LabelConfig, TimingConfig


def imu_config_for_session(meta: dict[str, Any] | None, imu_cfg: ImuConfig) -> ImuConfig:
    """세션 메타 ``imu.sensor_to_probe`` 가 있으면 설정의 R_SP 대신 그것을 쓴 ImuConfig 사본을 돌려준다.

    R_SP 는 IMU 마운트에 붙는 값이라 세션이 알고 있는 것이 우선이다 — 합성 세션(rus_policy.synth) 은 센서 = 프로브(I) 로
    만들어지고, 실기 세션은 수집기가 마운트 값을 남길 수 있다. 없으면 설정값(2026-09-10 board-6-qc 실측) 을 쓴다."""
    R = ((meta or {}).get("imu") or {}).get("sensor_to_probe")
    if R is None:
        return imu_cfg
    cfg = replace(imu_cfg, sensor_to_probe=[[float(v) for v in row] for row in R])
    cfg.R_sp()   # 검증 (3x3 회전행렬)
    return cfg

G0 = 9.80665
# 레거시 3 축 투영. 학습은 2026-09-10 부터 label/P6 (6 자유도 전체) 를 쓴다 — 품질을 좌우하는
# θx(기울임)·θy(부채질)·z(빔방향) 를 걸러내면 Q̂ 이 행동을 볼 수 없기 때문. label/P 는 호환용.
POLICY_AXES = (0, 1, 5)   # (x, y, θz) in the 6-dim (x, y, z, θx, θy, θz)


# --------------------------------------------------------------------------- 회전 유틸
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    return q / n


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """wxyz (…,4) → (…,3,3). fusion.quat_to_matrix 와 같은 식 (R = 센서→지구, 규약 'R')."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), float)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotmat_log(R: np.ndarray) -> np.ndarray:
    """(…,3,3) → 회전벡터 (…,3) [rad]. 각도 π 근처는 대칭식으로 처리."""
    R = np.asarray(R, float)
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(tr)
    axis = np.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    small = theta < 1e-6
    sin_t = np.sin(theta)
    scale = np.where(small, 0.5, theta / np.where(small, 1.0, 2.0 * sin_t))
    out = axis * scale[..., None]
    near_pi = theta > np.pi - 1e-3
    if np.any(near_pi):
        Rp = R[near_pi]
        d = np.clip((np.diagonal(Rp, axis1=-2, axis2=-1) + 1.0) * 0.5, 0.0, None)
        v = np.sqrt(d)
        # 부호는 비대각 성분으로 결정
        v[..., 1] *= np.sign(Rp[..., 0, 1] + 1e-12) * np.sign(v[..., 0] + 1e-12)
        v[..., 2] *= np.sign(Rp[..., 0, 2] + 1e-12) * np.sign(v[..., 0] + 1e-12)
        v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        out[near_pi] = v * theta[near_pi][..., None]
    return out


def gravity_in_frame(R_frame_from_earth: np.ndarray) -> np.ndarray:
    """지구 +z (위) 단위벡터를 주어진 프레임에서 표현."""
    return R_frame_from_earth @ np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- 정지 판정
def _window_bounds(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(n)
    lo = np.maximum(0, i - k // 2)
    hi = np.minimum(n, i + k // 2 + 1)
    return lo, hi


def _rolling_mean_std(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n,d) 의 창별 평균·표준편차 (모집단, ddof=0 — numpy .std 기본과 같음)."""
    x = np.asarray(x, float)
    x0 = x - x.mean(axis=0, keepdims=True)         # 취소 오차 방지
    cs = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x0, axis=0)], axis=0)
    cs2 = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x0 * x0, axis=0)], axis=0)
    cnt = (hi - lo)[:, None].astype(float)
    mean = (cs[hi] - cs[lo]) / cnt
    var = (cs2[hi] - cs2[lo]) / cnt - mean * mean
    return mean + x.mean(axis=0, keepdims=True), np.sqrt(np.maximum(var, 0.0))


def still_mask(t: np.ndarray, gyr: np.ndarray, acc: np.ndarray, cfg: ImuConfig,
               dt: Optional[float] = None) -> np.ndarray:
    """qc_common.still_mask 와 같은 판정을 벡터화. 창 폭은 dt 로 고정하는 것을 권장."""
    t = np.asarray(t, float)
    n = t.size
    if n < 8:
        return np.zeros(n, bool)
    dt = float(np.median(np.diff(t))) if dt is None else float(dt)
    k = max(3, int(round(cfg.still_win_s / max(dt, 1e-6))))
    lo, hi = _window_bounds(n, k)
    _, g_sd = _rolling_mean_std(gyr, lo, hi)
    gm = np.linalg.norm(gyr, axis=1)[:, None]
    gm_mean, _ = _rolling_mean_std(gm, lo, hi)
    _, a_sd = _rolling_mean_std(acc, lo, hi)
    out = ((g_sd.max(axis=1) < cfg.still_gyro_sd)
           & (gm_mean[:, 0] < cfg.still_gyro_mean)
           & (a_sd.max(axis=1) < cfg.still_accel_sd)
           & ((hi - lo) >= 3))
    return out


def segments_from_mask(t: np.ndarray, mask: np.ndarray, min_s: float) -> list[tuple[int, int]]:
    """mask 참 연속 구간 [(i0, i1)] (양끝 포함). min_s 미만은 버린다."""
    m = np.asarray(mask, bool)
    if m.size == 0:
        return []
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if t[e] - t[s] >= min_s]


@dataclass
class MoveSegment:
    """정지 A → 이동 → 정지 B. 인덱스는 IMU 표 기준."""
    ia: int            # 앵커 (정지 A 의 끝쪽 25 % 안)
    ib: int            # 정지 B 안의 앵커
    a0: int            # 정지 A 시작
    a1: int            # 정지 A 끝 (= 이동 시작 직전)
    b0: int            # 정지 B 시작 (= 이동 끝)
    b1: int            # 정지 B 끝
    move_s: float

    @property
    def index(self) -> tuple[int, int]:
        return self.ia, self.ib


def move_segments(t: np.ndarray, still: np.ndarray, cfg: ImuConfig) -> tuple[list[MoveSegment], list[tuple[int, int]]]:
    """analyze_track.move_segments 와 같은 규칙. (구간 목록, 정지 구간 목록).

    창 폭 win_s 의 이동창 판정은 정지 구간의 양 끝을 약 win_s/2 씩 깎아 낸다 (창이 운동에
    걸치면 산포가 커지므로). 따라서 **물리적 정지 길이 min_still_s** 를 요구하려면 검출된
    마스크 길이는 min_still_s − win_s 이상이면 된다. 검출된 정지 표본 하나는 그 주변 ±win/2
    가 전부 저산포라는 뜻이라 ZUPT 앵커로 쓰기에 충분하다.
    """
    min_detected = max(0.02, cfg.min_still_s - cfg.still_win_s)
    stills = segments_from_mask(t, still, min_detected)
    out: list[MoveSegment] = []
    for k in range(len(stills) - 1):
        (a0, a1), (b0, b1) = stills[k], stills[k + 1]
        ia = int(a1 - cfg.anchor_fraction * (a1 - a0))
        ib = int(b0 + cfg.anchor_fraction * (b1 - b0))
        move_s = float(t[b0] - t[a1])
        if t[ib] - t[ia] < 0.2 or move_s > cfg.max_move_s or move_s < cfg.min_move_s:
            continue
        out.append(MoveSegment(ia=ia, ib=ib, a0=a0, a1=a1, b0=b0, b1=b1, move_s=move_s))
    return out, stills


# --------------------------------------------------------------------------- 지구 프레임
def earth_convention(quat: np.ndarray, acc: np.ndarray, tol: float = 0.05) -> tuple[str, bool]:
    """정지 표본에서 지구 z 가 +g 가 되는 규약을 고른다 ('R' = R(q)·a, 'R.T' = R(q)ᵀ·a).

    반환 (규약, 모호함). 센서 z 가 중력과 (반)평행이면 두 규약이 같은 답을 내 판별이 안 된다
    — 그때는 'R' (BNO085 RV 의 기본, zero_ref 와 동일 판정) 로 두고 모호 플래그를 켠다.
    zero_ref 가 세션 메타에 있으면 그쪽의 quat_convention 을 우선한다 (label_session).
    """
    R = quat_to_matrix(quat)
    fwd = np.einsum("nij,nj->ni", R, acc)
    inv = np.einsum("nji,nj->ni", R, acc)

    def score(v):
        return abs(v[:, 2].mean() - G0) + np.abs(v[:, :2].mean(0)).sum()

    s_f, s_i = score(fwd), score(inv)
    ambiguous = abs(s_f - s_i) < tol
    if ambiguous:
        return "R", True
    return ("R" if s_f <= s_i else "R.T"), False


def sensor_to_earth_rotations(quat: np.ndarray, convention: str) -> np.ndarray:
    R = quat_to_matrix(quat)
    return R if convention == "R" else np.swapaxes(R, -1, -2)


def cumtrapz(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """0 에서 시작하는 사다리꼴 누적적분 (가변 dt). analyze_track._cumtrapz 참조."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    out = np.zeros_like(y)
    if len(t) > 1:
        dt = np.diff(t)[:, None] if y.ndim == 2 else np.diff(t)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt, axis=0)
    return out


def zupt_integrate(a_lin: np.ndarray, t: np.ndarray, zupt: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """가속도 → (속도, 변위, ZUPT 전 끝 속도). 끝 속도 0 조건의 선형 디드리프트."""
    v = cumtrapz(a_lin, t)
    v_end_raw = v[-1].copy() if len(v) else np.zeros(a_lin.shape[1])
    if zupt and len(v) > 1:
        ramp = (t - t[0]) / max(t[-1] - t[0], 1e-9)
        v = v - v[-1] * ramp[:, None]
    p = cumtrapz(v, t)
    return v, p, v_end_raw


# --------------------------------------------------------------------------- 라벨
@dataclass
class SegmentLabel:
    """chunk 하나의 라벨. 모두 앵커 시점의 프로브 프레임 기준."""
    P6: np.ndarray                 # (k+1, 6) [x,y,z mm, θx,θy,θz deg], P6[0] = 0
    sigma_net: np.ndarray          # (3,) policy 축 (x mm, y mm, θz deg)
    sigma_shape: np.ndarray        # (3,)
    grid_t_dev: np.ndarray         # (k+1,) chunk 격자 시각 (dev)
    t_anchor_dev: float
    t_anchor_pc: float
    t_still_start_dev: float       # 정지 A 시작 (관측 창 증강의 하한, dataset.py)
    t_end_dev: float               # 정지 B 앵커
    move_s: float
    still_before_s: float
    still_after_s: float
    gravity_P: np.ndarray          # (3,) 지구 +z 를 프로브 프레임에서 (roll/pitch 대용, §1.7)
    R_PE: np.ndarray               # (3,3) 지구→프로브 (앵커)
    net_E_m: np.ndarray            # (3,) 지구 프레임 순변위 [m] (Ã_{t−1} 계산용)
    net_rotvec_S_rad: np.ndarray   # (3,) 센서 프레임 순회전 [rad]
    seg: MoveSegment
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def P(self) -> np.ndarray:
        """policy 축 (k+1, 3): x mm, y mm, θz deg."""
        return self.P6[:, list(POLICY_AXES)]

    @property
    def net_mm(self) -> float:
        return float(np.linalg.norm(self.P6[-1, :3]))


@dataclass
class SessionLabels:
    labels: list[SegmentLabel]
    still: np.ndarray                 # (n,) 정지 마스크
    stills: list[tuple[int, int]]
    convention: str
    R_SE: np.ndarray                  # (n,3,3)
    diagnostics: dict[str, Any]


def sigma_for(source: str, tau_s: float, lab: LabelConfig) -> tuple[np.ndarray, np.ndarray]:
    """(σ_net, σ_shape) — policy 축 (x mm, y mm, θz deg). §5.3a 의 소스별 표."""
    if source == "teleop":
        s_t, s_r = lab.teleop_sigma_translation_mm, lab.teleop_sigma_rotation_deg
    else:
        s_t = lab.sigma_translation_coeff_mm * float(tau_s) ** 1.5     # (2.1)
        s_r = lab.sigma_rotation_deg
    s_t = max(s_t, lab.sigma_floor_mm)
    s_r = max(s_r, lab.sigma_floor_deg)
    net = np.array([s_t, s_t, s_r], float)
    shape = np.maximum(net * lab.sigma_shape_fraction,
                       [lab.sigma_floor_mm, lab.sigma_floor_mm, lab.sigma_floor_deg])
    return net, shape


def label_segment(t_dev: np.ndarray, t_pc: np.ndarray, acc: np.ndarray, R_SE: np.ndarray,
                  still: np.ndarray, seg: MoveSegment, timing: TimingConfig, imu_cfg: ImuConfig,
                  lab: LabelConfig, source: str = "freehand",
                  R_sp: Optional[np.ndarray] = None) -> SegmentLabel:
    """구간 하나 → SegmentLabel."""
    R_sp = imu_cfg.R_sp() if R_sp is None else np.asarray(R_sp, float)
    ia, ib = seg.ia, seg.ib
    sl = slice(ia, ib + 1)
    tt = t_dev[sl]

    # 지구 프레임 선형가속: 앵커 정지 구간의 평균(중력 포함)을 바이어스로 뺀다 (단계 C)
    aE = np.einsum("nij,nj->ni", R_SE[sl], acc[sl])
    still_a = np.arange(seg.a0, seg.a1 + 1)
    bias_E = np.einsum("nij,nj->ni", R_SE[still_a], acc[still_a]).mean(axis=0)
    a_lin = aE - bias_E
    v, p_E, v_end_raw = zupt_integrate(a_lin, tt, zupt=True)

    # 회전: 앵커 대비 상대회전 (센서 프레임) → 프로브 프레임
    R_a = R_SE[ia]
    dR = np.einsum("ji,njk->nik", R_a, R_SE[sl])          # R_aᵀ R_t
    r_S = rotmat_log(dR)                                   # (n,3) rad, 센서 프레임
    R_PE = R_sp @ R_a.T
    p_P = p_E @ R_PE.T                                     # (n,3) m
    r_P = r_S @ R_sp.T                                     # (n,3) rad

    # chunk 격자 표본화. 구간 밖(t > t_b) 은 마지막 값 유지 (정지이므로 물리적으로도 맞다)
    grid = tt[0] + np.arange(timing.chunk_steps + 1) * timing.chunk_dt
    P6 = np.zeros((timing.chunk_steps + 1, 6), float)
    for j in range(3):
        P6[:, j] = np.interp(grid, tt, p_P[:, j]) * 1000.0            # mm
        P6[:, 3 + j] = np.degrees(np.interp(grid, tt, r_P[:, j]))    # deg
    P6[0] = 0.0

    tau = float(tt[-1] - tt[0])
    # σ 는 라벨이 실제로 덮는 창(chunk 지평) 으로 잰다 (2026-09-08). 이동이 chunk 창보다 길면 P_k 는
    # 1.6 s 시점의 변위이고, 그 오차는 c·(1.6)^1.5 이지 c·τ^1.5 (τ = 구간 전체) 가 아니다.
    tau_eff = min(tau, float(timing.chunk_steps * timing.chunk_dt))
    s_net, s_shape = sigma_for(source, tau_eff, lab)
    still_before = float(t_dev[seg.a1] - t_dev[seg.a0])
    still_after = float(t_dev[seg.b1] - t_dev[seg.b0])
    diag = {
        "zupt_v_end_raw_mm_s": float(np.linalg.norm(v_end_raw) * 1000.0),
        "integration_window_s": tau,
        "sigma_window_s": tau_eff,
        "bias_E_norm": float(np.linalg.norm(bias_E)),
        "still_frac": float(np.mean(still[sl])),
        "net_mm": float(np.linalg.norm(p_P[-1]) * 1000.0),
        "chunk_covers_move": float(grid[-1] >= t_dev[seg.b0]),
    }
    return SegmentLabel(
        P6=P6, sigma_net=s_net, sigma_shape=s_shape, grid_t_dev=grid,
        t_anchor_dev=float(tt[0]), t_anchor_pc=float(t_pc[ia]), t_still_start_dev=float(t_dev[seg.a0]),
        t_end_dev=float(tt[-1]),
        move_s=seg.move_s, still_before_s=still_before, still_after_s=still_after,
        gravity_P=gravity_in_frame(R_PE), R_PE=R_PE, net_E_m=p_E[-1].copy(),
        net_rotvec_S_rad=r_S[-1].copy(), seg=seg, diagnostics=diag,
    )


def label_session(imu, timing: TimingConfig, imu_cfg: ImuConfig, lab: LabelConfig,
                  source: str = "freehand", convention: Optional[str] = None) -> SessionLabels:
    """ImuTable → 세션의 모든 chunk 라벨. convention 은 zero_ref.quat_convention 이 있으면 그 값."""
    t = imu.t_dev
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.004
    still = still_mask(t, imu.gyr, imu.acc, imu_cfg, dt=dt)
    segs, stills = move_segments(t, still, imu_cfg)
    quat = imu.quaternion(imu_cfg.quaternion)
    ok = np.all(np.isfinite(quat), axis=1)
    if not ok.all():
        # 남은 NaN 쿼터니언은 앞 값으로 채운다
        idx = np.where(ok, np.arange(len(ok)), 0)
        np.maximum.accumulate(idx, out=idx)
        quat = quat[idx]
        if not ok[0]:
            first = int(np.argmax(ok)) if ok.any() else 0
            quat[:first] = quat[first]
    ambiguous = False
    if convention in ("R", "R.T"):
        conv = convention
    else:
        sel = still if still.any() else np.ones(len(quat), bool)
        conv, ambiguous = earth_convention(quat[sel], imu.acc[sel])
    convention = conv
    R_SE = sensor_to_earth_rotations(quat, convention)
    R_sp = imu_cfg.R_sp()

    labels = [label_segment(t, imu.t_pc, imu.acc, R_SE, still, seg, timing, imu_cfg, lab,
                            source=source, R_sp=R_sp) for seg in segs]
    diag = {
        "imu_rate_hz": float(1.0 / dt) if dt > 0 else float("nan"),
        "still_fraction": float(still.mean()) if still.size else 0.0,
        "n_still_segments": len(stills),
        "n_move_segments": len(labels),
        "convention": convention,
        "convention_ambiguous": bool(ambiguous),
        "gravity_check_m_s2": float(np.linalg.norm(
            np.einsum("nij,nj->ni", R_SE[still], imu.acc[still]).mean(0))) if still.any() else float("nan"),
    }
    return SessionLabels(labels=labels, still=still, stills=stills, convention=convention,
                         R_SE=R_SE, diagnostics=diag)


def previous_motion(labels: list[SegmentLabel], i: int, imu_cfg: ImuConfig) -> tuple[np.ndarray, float, float]:
    """Ã_{t−1}: 직전 구간의 실현 운동을 **현재** 앵커의 프로브 프레임으로. (vec3, gap_s, valid)."""
    cur = labels[i]
    if i == 0:
        return np.zeros(3), float("inf"), 0.0
    prev = labels[i - 1]
    gap = cur.t_anchor_dev - prev.t_end_dev
    if gap > imu_cfg.prev_motion_max_gap_s or gap < 0:
        return np.zeros(3), float(gap), 0.0
    d_P = cur.R_PE @ prev.net_E_m * 1000.0                        # mm, 현재 프로브 프레임
    # 회전: 이전 구간 센서 프레임 회전벡터 → 프로브 (프레임이 조금 달라도 짧은 공백에서는 근사)
    r_P = np.degrees(imu_cfg.R_sp() @ prev.net_rotvec_S_rad)
    return np.array([d_P[0], d_P[1], r_P[2]], float), float(gap), 1.0
