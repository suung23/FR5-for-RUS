"""합성 세션 — 하드웨어 없이 파이프라인 전체를 검증한다 (§8 "지금 할 수 있는 것").

``us_imu_collect.py`` 와 **같은 레이아웃** (imu_<stamp>.csv 33열, us_frames.bin, us_index.csv,
session.meta.json) 을 만든다. 추가로 ``truth.json`` 에 구간별 정답을 남긴다.

시연 프로토콜 (§2.3): [정지] → [이동 τ≈0.5–1.5 s, 최소저크] → [정지] 반복.
  * 프로브 자세: 빔축(+z_probe)이 아래(중력 방향). 센서 = 프로브 (R_SP = I).
  * 방광은 프로브 기준 (x_off, y_off) 에 있고, 시연자는 이동으로 그것을 중심에 놓는다
    (Δx ≈ −x_off·(0.85–1.0), Δy ≈ −y_off·…). 정지 중에 방광 위치가 새로 뽑힌다.
  * 영상: 어두운 타원 (루멘). 위치 = −x_off/s, 크기 = 슬라이스 두께 sqrt(1 − (y_off/R_y)²),
    장축 회전 = φ_off (시연자는 Δθ_z = −φ_off 로 정렬한다).
    |y_off| 의 부호는 영상에 나타나지 않는다 — §3.2 의 이봉성이 그대로 재현된다.
  * IMU: a_S = R_SEᵀ(p̈_E + g ẑ) + b_S + n,   ω_S + b_g + n,   칩 쿼터니언 = R_SE (규약 'R').
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .imu_labels import G0
from .session import IMU_COLUMNS


@dataclass
class SynthConfig:
    duration_s: float = 60.0
    imu_hz: float = 200.0
    us_fps: float = 8.0
    still_s: tuple[float, float] = (0.4, 1.2)
    move_s: tuple[float, float] = (0.5, 1.4)
    offset_mm: float = 15.0            # 방광 오프셋 |x|,|y| 최대
    rot_deg: float = 6.0               # 방광 장축의 프로브 기준 회전 오프셋 최대 (시연자는 이를 되돌린다)
    accel_noise: float = 0.03          # m/s^2 (백색)
    gyro_noise: float = 0.002          # rad/s
    accel_bias: float = 0.03           # m/s^2 상수 바이어스 (§2.3 의 b)
    gyro_bias: float = 0.001           # rad/s
    us_latency_s: float = 0.0          # 프레임 수신 지연 (pc_unix 에 더해짐)
    move_wobble_deg: float = 1.5       # 이동 중 손의 회전 흔들림 (실제 프리핸드는 병진에 회전이 따라온다)
    still_tremor_deg: float = 0.02     # 정지 중 미세 떨림 (판정 임계 아래)
    image_size: int = 256
    mm_per_px: float = 0.5
    seed: int = 0
    t0_unix: Optional[float] = None


def _min_jerk(n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def _rotmat_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(3,3) → wxyz (Shepperd)."""
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def _euler_zyx_deg(R: np.ndarray) -> tuple[float, float, float]:
    pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def render_frame(rng: np.random.RandomState, size: int, mm_per_px: float, x_off_mm: float, y_off_mm: float,
                 theta_deg: float, radius_mm: float = 22.0, ry_mm: float = 26.0) -> np.ndarray:
    """B-mode 흉내: 부채꼴 ROI 안의 스페클 + 어두운 타원 루멘 (+ 아래 밝은 후벽)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = size / 2 - x_off_mm / mm_per_px
    cy = size * 0.45
    slice_scale = float(np.sqrt(max(0.0, 1.0 - (y_off_mm / ry_mm) ** 2)))
    a = radius_mm / mm_per_px * slice_scale
    b = 0.7 * a
    th = np.radians(-theta_deg)
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    inside = (u / max(a, 1e-3)) ** 2 + (v / max(b, 1e-3)) ** 2 <= 1.0 if a > 1.0 else np.zeros_like(xx, bool)
    speckle = rng.gamma(shape=3.0, scale=1.0 / 3.0, size=(size, size)).astype(np.float32)
    img = 0.45 * speckle
    img[inside] *= 0.12
    # 후벽 향상
    wall = ((u / max(a * 1.15, 1e-3)) ** 2 + (v / max(b * 1.15, 1e-3)) ** 2 <= 1.0) & ~inside & (dy > 0)
    img[wall] *= 1.6
    # 부채꼴 ROI (꼭짓점 위 중앙, 반각 29°)
    apex = np.array([size / 2, -0.15 * size])
    ang = np.arctan2(xx - apex[0], yy - apex[1])
    r = np.hypot(xx - apex[0], yy - apex[1])
    fan = (np.abs(ang) <= np.radians(29.0)) & (r <= 1.12 * size) & (r >= 0.18 * size)
    img[~fan] = 0.0
    img *= np.clip(1.15 - 0.5 * (yy / size), 0.5, 1.2)     # 깊이 감쇠
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def generate_session(out_dir: str | Path, cfg: SynthConfig = SynthConfig(), name: Optional[str] = None
                     ) -> Path:
    rng = np.random.RandomState(cfg.seed)
    t0 = cfg.t0_unix if cfg.t0_unix is not None else time.time()
    stamp = name or ("us_imu_synth_" + time.strftime("%Y%m%d_%H%M%S", time.localtime(t0)) + f"_{cfg.seed}")
    root = Path(out_dir) / stamp
    root.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / cfg.imu_hz
    n = int(cfg.duration_s * cfg.imu_hz)
    t = np.arange(n) * dt

    # ---------------- 궤적 계획: 프로브 프레임 (x,y) 변위와 θ_z, 방광 오프셋 시퀀스
    R_SE0 = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # 빔 +z 가 지구 −z (아래)
    p_E = np.zeros((n, 3))          # 지구 프레임 변위 [m]
    theta = np.zeros(n)             # θ_z [rad] (센서 z 둘레 누적)
    x_off = np.zeros(n)             # 방광 오프셋 [mm], 현재 프로브 기준
    y_off = np.zeros(n)
    phi_off = np.zeros(n)           # 방광 장축 회전 오프셋 [rad], 현재 프로브 기준
    truth: list[dict[str, Any]] = []

    def _sample_offsets():
        o = rng.uniform(-cfg.offset_mm, cfg.offset_mm, 2)
        ph = np.radians(rng.uniform(-cfg.rot_deg, cfg.rot_deg)) if rng.rand() < 0.7 else 0.0
        return o, ph

    cur_off, cur_phi = _sample_offsets()
    i = int(round(rng.uniform(*cfg.still_s) / dt))
    x_off[:i], y_off[:i], phi_off[:i] = cur_off[0], cur_off[1], cur_phi
    p_acc = np.zeros(3)
    th_acc = 0.0
    while True:
        n_move = int(round(rng.uniform(*cfg.move_s) / dt))
        n_still = int(round(rng.uniform(*cfg.still_s) / dt))
        if i + n_move + n_still >= n:
            break
        gain = rng.uniform(0.85, 1.0, 3)
        d_mm = -cur_off * gain[:2]
        d_th = -cur_phi * gain[2]                               # 장축을 프로브 x 축에 정렬
        prof = _min_jerk(n_move)
        sl = slice(i, i + n_move)
        R_start = R_SE0 @ _rotmat_z(th_acc)                    # 이동 시작 시점의 센서→지구
        d_E = R_start @ np.array([d_mm[0] * 1e-3, d_mm[1] * 1e-3, 0.0])
        p_E[sl] = p_acc[None, :] + d_E[None, :] * prof[:, None]
        theta[sl] = th_acc + d_th * prof
        # 프로브가 움직이는 동안 방광 오프셋은 반대로 줄어든다
        x_off[sl] = cur_off[0] + d_mm[0] * prof
        y_off[sl] = cur_off[1] + d_mm[1] * prof
        phi_off[sl] = cur_phi + d_th * prof
        truth.append({"t_move_start": float(t[i]), "t_move_end": float(t[i + n_move - 1]),
                      "dx_mm": float(d_mm[0]), "dy_mm": float(d_mm[1]), "dtheta_deg": float(np.degrees(d_th)),
                      "x_off_before_mm": float(cur_off[0]), "y_off_before_mm": float(cur_off[1]),
                      "phi_off_before_deg": float(np.degrees(cur_phi))})
        p_acc = p_E[i + n_move - 1].copy()
        th_acc = theta[i + n_move - 1]
        i += n_move
        # 정지: 잠깐 뒤 방광 위치가 새로 뽑힌다 (다음 시연의 목표)
        sl = slice(i, i + n_still)
        p_E[sl] = p_acc
        theta[sl] = th_acc
        left = cur_off + d_mm
        left_phi = cur_phi + d_th
        jump = i + max(1, int(0.12 * n_still))
        x_off[i:jump], y_off[i:jump], phi_off[i:jump] = left[0], left[1], left_phi
        cur_off, cur_phi = _sample_offsets()
        x_off[jump:i + n_still], y_off[jump:i + n_still], phi_off[jump:i + n_still] = cur_off[0], cur_off[1], cur_phi
        i += n_still
    p_E[i:] = p_acc
    theta[i:] = th_acc
    x_off[i:], y_off[i:], phi_off[i:] = cur_off[0], cur_off[1], cur_phi

    # ---------------- 지구 프레임 운동학 → IMU
    v_E = np.gradient(p_E, dt, axis=0)
    a_E = np.gradient(v_E, dt, axis=0)
    # 손 흔들림: 이동 중에는 큰 저주파 wobble, 정지 중에는 작은 떨림 (roll/pitch 성분)
    moving = np.linalg.norm(v_E, axis=1) > 1e-4
    wob = np.zeros((n, 2))
    for ax in range(2):
        f1, f2 = rng.uniform(1.5, 3.0), rng.uniform(4.0, 7.0)
        ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
        w = np.sin(2 * np.pi * f1 * t + ph1) + 0.4 * np.sin(2 * np.pi * f2 * t + ph2)
        amp = np.where(moving, cfg.move_wobble_deg, cfg.still_tremor_deg)
        # 이동/정지 전환에서 진폭을 부드럽게
        kern = np.ones(int(0.15 / dt)) / max(1, int(0.15 / dt))
        amp = np.convolve(amp, kern, mode="same")
        wob[:, ax] = np.radians(amp) * w
    # wobble 을 자세에 반영: R_SE = R_SE0 · Rz(θ) · Rx(wob_x) · Ry(wob_y)
    def _rx(a):
        c, s_ = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])

    def _ry(a):
        c, s_ = np.cos(a), np.sin(a)
        return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])

    R_SE = np.array([R_SE0 @ _rotmat_z(th) @ _rx(wx) @ _ry(wy) for th, wx, wy in zip(theta, wob[:, 0], wob[:, 1])])
    # 각속도 (센서 프레임): ω̂ = Rᵀ Ṙ
    Rdot = np.gradient(R_SE, dt, axis=0)
    W = np.einsum("nji,njk->nik", R_SE, Rdot)
    omega_S = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], axis=1)
    bias_S = rng.normal(0, cfg.accel_bias, 3)
    gyro_b = rng.normal(0, cfg.gyro_bias, 3)
    acc_S = np.einsum("nji,nj->ni", R_SE, a_E + np.array([0.0, 0.0, G0])) + bias_S         + rng.normal(0, cfg.accel_noise, (n, 3))
    gyr_S = omega_S + gyro_b + rng.normal(0, cfg.gyro_noise, (n, 3))
    quat = np.array([_matrix_to_quat(R) for R in R_SE])
    quat += rng.normal(0, 2e-4, quat.shape)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    # ---------------- IMU CSV (33열)
    imu_path = root / f"imu_{stamp}.csv"
    pc = t0 + t + rng.normal(0, 0.0008, n)
    dev_us = np.round(t * 1e6 + 1_000_000).astype(np.int64)
    mag = np.array([22.0, 5.0, -40.0])
    with open(imu_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(IMU_COLUMNS)
        for j in range(n):
            R = R_SE[j]
            roll, pitch, yaw = _euler_zyx_deg(R)
            m_S = R.T @ mag
            row = [f"{pc[j]:.6f}", int(dev_us[j]),
                   *[f"{v:.5f}" for v in acc_S[j]], *[f"{v:.6f}" for v in gyr_S[j]],
                   *[f"{v:.2f}" for v in m_S], *[f"{v:.2f}" for v in m_S],
                   *[f"{v:.6f}" for v in quat[j]], f"{roll:.3f}", f"{pitch:.3f}", f"{yaw:.3f}",
                   *[f"{v:.6f}" for v in quat[j]], f"{roll:.3f}", f"{pitch:.3f}", f"{yaw:.3f}",
                   "0.00", "", "", "", 3, 3, 3, 3]
            w.writerow(row)
    meta = {"source": "synthetic (rus_policy.synth)", "port": "none", "use_mag": True, "n_samples": n,
            "start_unix": t0, "columns": IMU_COLUMNS, "closed": True, "end_unix": float(t0 + t[-1]),
            "duration_s": float(t[-1]), "synth": {k: (list(v) if isinstance(v, tuple) else v)
                                                   for k, v in asdict(cfg).items()}}
    (root / f"imu_{stamp}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------------- US 프레임
    n_frames = int(cfg.duration_s * cfg.us_fps)
    ft = np.arange(n_frames) / cfg.us_fps + rng.uniform(0, 0.02, n_frames)
    ft = np.clip(ft, 0, t[-1])
    idx = np.searchsorted(t, ft).clip(0, n - 1)
    frames = np.empty((n_frames, cfg.image_size, cfg.image_size), np.uint8)
    for f in range(n_frames):
        j = idx[f]
        frames[f] = render_frame(rng, cfg.image_size, cfg.mm_per_px, x_off[j], y_off[j], np.degrees(phi_off[j]))
    frames.tofile(root / "us_frames.bin")
    with open(root / "us_index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pc_unix", "us_seq", "frame_id", "byte_offset"])
        per = cfg.image_size * cfg.image_size
        for f in range(n_frames):
            w.writerow([f"{t0 + ft[f] + cfg.us_latency_s:.6f}", f, f % 256, f * per])

    session_meta = {
        "created": stamp, "time_axis": "pc_unix (synthetic)", "join_key": "pc_unix",
        "caveat": "synthetic session generated by rus_policy.synth — not real data",
        "us": {"host": "synthetic", "frames": n_frames, "frame_shape": [cfg.image_size, cfg.image_size],
               "dtype": "uint8", "bin": "us_frames.bin", "index": "us_index.csv"},
        "imu": {"port": "none", "rows": n, "csv": imu_path.name, "meta": imu_path.name.replace(".csv", ".meta.json"),
                # 합성 세션은 센서 = 프로브. 실기 설정의 R_SP 가 적용되지 않도록 세션이 직접 밝힌다 (imu_labels.imu_config_for_session)
                "sensor_to_probe": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
    }
    (root / "session.meta.json").write_text(json.dumps(session_meta, indent=2), encoding="utf-8")
    (root / "truth.json").write_text(json.dumps({
        "segments": truth, "R_SE0": R_SE0.tolist(), "accel_bias_S": bias_S.tolist(),
        "gyro_bias_S": gyro_b.tolist(), "mm_per_px": cfg.mm_per_px, "t0_unix": t0,
        "note": "dx,dy in the probe frame at move start (x lateral, y elevational); dtheta about beam axis",
    }, indent=2), encoding="utf-8")
    return root
