#!/usr/bin/env python3
"""자체 시험 — 로봇 없이, **알려진 값을 심은** 가짜 세션을 만든다.

    python3 simulate_session.py                 # raw_data_sim/ 에 생성
    python3 analyze_track.py --run raw_data_sim

분석기가 심은 값을 되찾아 오는지가 이 파일의 존재 이유다. 되찾지 못하면 그건
센서 이야기가 아니라 **분석기 버그**이고, 실기 데이터로는 그 둘을 구분할 수
없다. (Surgilogger QC 도 이 과정에서 지연보정 부호 버그를 하나 잡았다.)

심는 것과 되찾아야 하는 것
--------------------------
    자세 지연      --imu-lag 25 ms        회전 실효 지연 ~25 ms
    고정회전 A,B   임의의 두 회전         정렬 잔차 ~0 deg (잡음 수준)
    지렛대 r       --lever 60,-20,35 mm   회귀 r ~ 같은 값
    가속도 스케일  --accel-scale 1.023    측정 중력 +2.3 %, 캘리브가 되돌림
    가속도 바이어스 --accel-bias           ZUPT(C) 가 원리적으로 지운다

되찾지 못하는 것도 있다 — 그것도 결과다. 예컨대 6 축 퓨전의 heading 은 원리적으로
관측되지 않으므로 azimuth 오차가 자란다. 그게 정상이라는 것을 시뮬레이터가
보여 준다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import protocol                                                 # noqa: E402
import qc_common as qc                                          # noqa: E402

FINE_HZ = 1000.0            # 궤적 생성 격자. 여기서 미분해 GT/IMU 로 내려보낸다


# ------------------------------------------------------------------ 회전 유틸
def rotvec_to_R(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def R_to_quat_xyzw(R):
    R = np.asarray(R, float)
    m = R.reshape(-1, 3, 3)
    q = np.empty((m.shape[0], 4))
    for i, a in enumerate(m):
        t = a[0, 0] + a[1, 1] + a[2, 2]
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[i] = [(a[2, 1] - a[1, 2]) / s, (a[0, 2] - a[2, 0]) / s,
                    (a[1, 0] - a[0, 1]) / s, 0.25 * s]
        elif a[0, 0] > a[1, 1] and a[0, 0] > a[2, 2]:
            s = np.sqrt(1.0 + a[0, 0] - a[1, 1] - a[2, 2]) * 2
            q[i] = [0.25 * s, (a[0, 1] + a[1, 0]) / s, (a[0, 2] + a[2, 0]) / s,
                    (a[2, 1] - a[1, 2]) / s]
        elif a[1, 1] > a[2, 2]:
            s = np.sqrt(1.0 + a[1, 1] - a[0, 0] - a[2, 2]) * 2
            q[i] = [(a[0, 1] + a[1, 0]) / s, 0.25 * s, (a[1, 2] + a[2, 1]) / s,
                    (a[0, 2] - a[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + a[2, 2] - a[0, 0] - a[1, 1]) * 2
            q[i] = [(a[0, 2] + a[2, 0]) / s, (a[1, 2] + a[2, 1]) / s, 0.25 * s,
                    (a[1, 0] - a[0, 1]) / s]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(R.shape[:-2] + (4,))


def slerp(qa, qb, u):
    """쿼터니언 구면 보간.

    축각으로 뽑아 보간하면 **180 deg 근처에서 무너진다** — vee 식의 1/sin(ang) 이
    발산해 회전이 난수가 된다. align 블록에 큰 자세(+-90, 180 deg)를 넣자마자
    가속도 잔차가 17 m/s^2 로 튀어 그 사실이 드러났다.
    """
    qa = np.asarray(qa, float); qb = np.asarray(qb, float)
    d = float(np.dot(qa, qb))
    if d < 0:                      # 짧은 쪽으로 돈다 (q 와 -q 는 같은 회전)
        qb, d = -qb, -d
    if d > 0.9995:
        q = qa + u * (qb - qa)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - u) * th) * qa + np.sin(u * th) * qb) / np.sin(th)


def minjerk(s):
    """0..1 -> 0..1, 양 끝에서 속도·가속도가 0. 정지-이동-정지의 '이동' 이다."""
    s = np.clip(s, 0.0, 1.0)
    return 6 * s ** 5 - 15 * s ** 4 + 10 * s ** 3


# --------------------------------------------------------------- 궤적 생성
def probe_waypoints(rng, n, home_R):
    """프로빙하듯 — 프로브를 거의 연직으로 세운 채 조금씩 옮기고 기울인다.

    기울기 <= 25 deg, 이동 20~150 mm. 회전만 하는 구간과 병진만 하는 구간을
    각각 넣는다 (지렛대 항과 병진 가속을 데이터로 가르려면 둘 다 필요하다 —
    protocol.PROMPTS['probe'] 가 사람에게 시키는 것과 같은 것이다).
    """
    p0 = np.array([0.45, 0.0, 0.35])
    out = [(p0.copy(), home_R.copy())]
    p, R = p0.copy(), home_R.copy()
    for k in range(n):
        kind = k % 4
        if kind == 0:                       # 병진만
            p = p + rng.uniform(-0.08, 0.08, 3) * np.array([1, 1, 0.4])
        elif kind == 2:                     # 회전만
            R = rotvec_to_R(rng.uniform(-0.35, 0.35, 3)) @ home_R
        else:                               # 섞어서
            p = p + rng.uniform(-0.05, 0.05, 3) * np.array([1, 1, 0.4])
            R = rotvec_to_R(rng.uniform(-0.25, 0.25, 3)) @ home_R
        out.append((p.copy(), R.copy()))
    return out


def build_probe(rng, seconds, t0, hold_s=None, move_s=None):
    """정지-이동-정지 리듬의 궤적을 FINE_HZ 격자로 만든다."""
    hold_s = protocol.PROBE_HOLD_S if hold_s is None else hold_s
    move_s = protocol.PROBE_MOVE_S if move_s is None else move_s
    home = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # +z 가 아래
    n_wp = int(seconds / (hold_s + np.mean(move_s))) + 2
    wps = probe_waypoints(rng, n_wp, home)
    # 이동 길이를 섞는다 — 짧은 재배치(0.4~1.5 s)와 긴 이동을 둘 다 넣어야
    # '창 길이별 오차' 표의 짧은 칸이 채워진다. 실제 프로빙도 그렇게 움직인다.
    box = {"i": 0}

    def dur():
        box["i"] += 1
        return rng.uniform(0.4, 1.5) if box["i"] % 2 else rng.uniform(*move_s)

    t, P, R = interp_track(wps, t0, hold_s=hold_s, move_s=dur)
    keep = t - t0 <= seconds
    return t[keep], P[keep], R[keep]


def build_align(rng, t0, n_poses=None):
    """정지 자세 n 개. 서로 충분히 다른 방향으로 기울인다.

    이 블록은 두 가지에 쓰인다 — (A,B) 정렬과 **가속도계 다자세 보정**. 뒤엣것
    때문에 자세가 서로 충분히 달라야 하고, 정지 구간이 길어야 한다.
    """
    n_poses = protocol.ALIGN_TARGET_POSES if n_poses is None else n_poses
    home = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    # 두 축을 동시에 기울인 자세를 반드시 섞는다 — 한 축으로만 기울이면 그 축
    # 둘레가 관측되지 않아 A,B 가 유일하게 풀리지 않는다.
    # 앞 6 개는 중력이 센서 세 축을 양방향으로 훑게 하는 큰 자세다 (가속도계
    # 스케일·바이어스 분리), 뒤 6 개는 작은 복합 기울임 (고정회전 관측도).
    rots = [(0, 0, 0), (np.pi, 0, 0), (np.pi / 2, 0, 0), (-np.pi / 2, 0, 0),
            (0, np.pi / 2, 0), (0, -np.pi / 2, 0),
            (0.30, 0, 0), (0, 0.30, 0), (0.22, 0.22, 0), (-0.22, -0.22, 0),
            (0, 0, 0.60), (0.20, -0.20, 0.35)][:n_poses]
    p0 = np.array([0.45, 0.0, 0.35])
    wps = [(p0 + rng.uniform(-0.03, 0.03, 3), rotvec_to_R(np.array(rv)) @ home)
           for rv in rots]
    return interp_track(wps, t0, hold_s=protocol.ALIGN_HOLD_S + 0.5, move_s=2.0)


def interp_track(wps, t0, hold_s, move_s):
    """웨이포인트를 정지-이동-정지로 잇는다. move_s 는 상수이거나 함수.

    **위치와 자세를 같은 f(t) 로 움직인다.** 따로 움직이면 이음매에서 위치가
    튀고, 그 계단이 2 계 미분에서 폭발해 (실제로 그렇게 만들었다) 가속도 모형
    동정이 통째로 망가진다.
    """
    t, P, RR = [], [], []
    now = 0.0
    for k in range(len(wps)):
        pa, Ra = wps[k - 1] if k else wps[0]
        pb, Rb = wps[k]
        dur_m = move_s() if callable(move_s) else move_s
        for dur, mode in ((dur_m, "move"), (hold_s, "hold")):
            if k == 0 and mode == "move":
                continue
            n = max(2, int(round(dur * FINE_HZ)))
            f = minjerk(np.arange(n) / n) if mode == "move" else np.ones(n)
            qa, qb = R_to_quat_xyzw(Ra), R_to_quat_xyzw(Rb)
            for j in range(n):
                t.append(now + j / FINE_HZ)
                P.append(pa + (pb - pa) * f[j])
                RR.append(qc.quat_xyzw_to_R(slerp(qa, qb, f[j])))
            now += n / FINE_HZ
    return t0 + np.array(t), np.array(P), np.array(RR)


def build_still(t0, seconds):
    n = int(seconds * FINE_HZ)
    home = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    return (t0 + np.arange(n) / FINE_HZ,
            np.tile([0.45, 0.0, 0.35], (n, 1)),
            np.tile(home, (n, 1, 1)))


def build_sweep(t0, axis):
    """스크립트 자극 블록. protocol.plan_sweep 과 **같은 사양**으로 만든다."""
    home = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    p0 = np.array([0.45, 0.0, 0.35])
    t, P, RR, segs = [], [], [], []
    now = 0.0
    for seg in protocol.plan_sweep(axis):
        f = seg["freq_hz"]
        A = np.radians(seg["amp_deg"])
        lead = protocol.LEAD_S
        dur = seg["seconds"] + 2 * lead
        n = int(round(dur * FINE_HZ))
        tt = np.arange(n) / FINE_HZ
        env = protocol.raised_cosine(np.clip(tt / lead, 0, 1)) * \
            protocol.raised_cosine(np.clip((dur - tt) / lead, 0, 1))
        ang = A * env * np.sin(2 * np.pi * f * (tt - lead))
        # tilt 는 base +y 둘레 -> 침투축이 base x 쪽으로 기운다 (tip_x 채널).
        axis_v = {"tilt": np.array([0, 1.0, 0]), "roll": np.array([0, 0, 1.0])}[axis]
        for k in range(n):
            t.append(now + tt[k])
            P.append(p0)
            RR.append(rotvec_to_R(axis_v * ang[k]) @ home)
        segs.append({"freq_hz": f, "t_start": t0 + now + lead,
                     "t_stop": t0 + now + dur - lead, "cap": seg["cap"]})
        now += dur + protocol.SETTLE_S
        n2 = int(protocol.SETTLE_S * FINE_HZ)
        for k in range(n2):
            t.append(now - protocol.SETTLE_S + k / FINE_HZ)
            P.append(p0)
            RR.append(home)
    return t0 + np.array(t), np.array(P), np.array(RR), segs


# ------------------------------------------------------------ 센서 합성
def synth_block(block, t, P, R, args, rng, segs=None):
    """GT 궤적 -> (로봇 표, IMU 표). 심는 오차는 전부 여기서 들어간다."""
    dt = 1.0 / FINE_HZ
    A_wb = rotvec_to_R(np.array([0.03, -0.02, 0.9]))       # base <- imu world
    B_it = rotvec_to_R(np.array([0.6, 0.1, -0.2]))         # imu <- tool
    r = np.asarray(args.lever, float) / 1000.0

    # --- IMU 자리의 참 운동 --------------------------------------------
    p_imu = P + np.einsum("nij,j->ni", R, r)
    a_world = np.gradient(np.gradient(p_imu, dt, axis=0), dt, axis=0)
    R_imu = np.einsum("ji,njk,lk->nil", A_wb, R, B_it)      # A^T R B^T
    dR = np.gradient(R_imu, dt, axis=0)
    W = np.einsum("nji,njk->nik", R_imu, dR)               # R^T Rdot (반대칭)
    gyro_true = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], 1)
    # 비력 = R_imu^T (a + g) — 단, **IMU 자신의 world 프레임에서** 다.
    # a_world 와 중력은 base 프레임 값이므로 A^T 로 IMU world 로 먼저 옮긴다.
    # 이걸 빼먹으면 (실제로 빼먹었다) 되돌린 가속도가 A 만큼 돌아간 채 나오고,
    # 정지 시 +g 검산은 그대로 통과해 버려서 조용히 지나간다. 증상은 '변위
    # 오차가 참 모형을 넣어도 30 mm' 였다.
    a_imuworld = np.einsum("ji,nj->ni", A_wb, a_world + np.array([0.0, 0.0, qc.G0]))
    f_sensor = np.einsum("nji,nj->ni", R_imu, a_imuworld)

    # --- 심는 오차 ------------------------------------------------------
    S = np.diag([args.accel_scale, args.accel_scale * 0.999, args.accel_scale * 1.002])
    S[0, 1] += args.nonorth                                  # 비직교 한 항
    f_sensor = np.einsum("ij,nj->ni", S, f_sensor) + np.asarray(args.accel_bias, float)

    # --- GT 표본 --------------------------------------------------------
    gi = np.arange(0, len(t), max(1, int(round(FINE_HZ / args.gt_hz))))
    t_gt = t[gi] + rng.normal(0, args.gt_jitter_ms * 1e-3, gi.size)
    t_gt = np.sort(t_gt)
    robot = {
        "pose/t": t_gt, "pose/pc_ts": t_gt + rng.uniform(0.001, 0.004, gi.size),
        "pose/paired": np.ones(gi.size),
        "pose/position": P[gi] + rng.normal(0, args.gt_pos_noise_mm * 1e-3, (gi.size, 3)),
        "pose/quat_xyzw": R_to_quat_xyzw(R[gi]),
        "joints/t": t_gt, "joints/pc_ts": t_gt,
        "joints/position": np.zeros((gi.size, 6)),
        "joints/velocity": np.zeros((gi.size, 6)),
        "joints/effort": np.zeros((gi.size, 6)),
        "wrench/t": t_gt, "wrench/data": np.zeros((gi.size, 6)),
        "twist/t": np.empty(0), "twist/data": np.empty((0, 6)),
        "track_err/t": np.empty(0), "track_err/data": np.empty((0, 2)),
        "retreat/t": np.empty(0), "retreat/data": np.empty((0, 1)),
    }

    # --- IMU 표본 (지연을 심는다) ----------------------------------------
    t_imu = np.arange(t[0] + args.imu_lag * 1e-3, t[-1], 1.0 / args.imu_hz)
    src = t_imu - args.imu_lag * 1e-3          # 이 시각의 물리가 지금 보고된다

    def samp(arr):
        return np.column_stack([np.interp(src, t, arr[:, k]) for k in range(arr.shape[1])])

    acc = samp(f_sensor) + rng.normal(0, args.acc_noise, (src.size, 3))
    gyr = samp(gyro_true) + rng.normal(0, args.gyr_noise, (src.size, 3))
    Rq = np.stack([[np.interp(src, t, R_imu[:, i, j]) for j in range(3)] for i in range(3)], 0)
    R_s = np.array([qc.project_SO3(m) for m in np.transpose(Rq, (2, 0, 1))])
    # heading 드리프트: 자기환경이 나쁠 때 칩 RV 가 겪는 것과 같은 모양
    if args.yaw_drift_deg_per_min:
        d = np.radians(args.yaw_drift_deg_per_min) * (src - src[0]) / 60.0
        R_s = np.einsum("nij,njk->nik",
                        np.array([rotvec_to_R(np.array([0, 0, v])) for v in d]), R_s)
    quat = R_to_quat_xyzw(R_s)
    chip_q = np.column_stack([quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]])
    field = np.array([20.0, 0.0, -43.0])
    mag = np.einsum("nji,j->ni", R_s, field) + rng.normal(0, 0.4, (src.size, 3))

    dev_us = (t_imu - t_imu[0]) * 1e6 * (1.0 + args.skew_ppm * 1e-6)
    # pc_ts 는 시리얼 read 로 계단 양자화된다 — 실측과 같은 모양으로 만든다
    step = 0.005
    t_pc = t_imu[0] + np.ceil((t_imu - t_imu[0]) / step) * step + \
        rng.uniform(0.0005, 0.003, t_imu.size)

    imu = {"t": t_imu, "t_pc": t_pc, "dev_us": dev_us, "acc": acc, "gyr": gyr,
           "mag": mag, "mag_raw": mag, "chip_q": chip_q, "host_q": chip_q,
           "diff_deg": np.zeros(t_imu.size)}
    _ = (block, segs)
    return robot, imu, A_wb, B_it


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "raw_data_sim"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--imu-lag", type=float, default=25.0, help="자세·가속 지연 [ms]")
    ap.add_argument("--lever", default="60,-20,35", help="지렛대 r [mm], tool 프레임")
    ap.add_argument("--accel-scale", type=float, default=1.023)
    ap.add_argument("--nonorth", type=float, default=0.004)
    ap.add_argument("--accel-bias", default="0.02,-0.03,0.05", help="센서프레임 [m/s^2]")
    ap.add_argument("--yaw-drift-deg-per-min", type=float, default=0.0)
    ap.add_argument("--acc-noise", type=float, default=0.02)
    ap.add_argument("--gyr-noise", type=float, default=0.002)
    ap.add_argument("--imu-hz", type=float, default=250.0)
    ap.add_argument("--gt-hz", type=float, default=200.0)
    ap.add_argument("--gt-jitter-ms", type=float, default=0.3)
    ap.add_argument("--gt-pos-noise-mm", type=float, default=0.05)
    ap.add_argument("--skew-ppm", type=float, default=-22.0)
    ap.add_argument("--still-seconds", type=float, default=40.0)
    ap.add_argument("--probe-seconds", type=float, default=120.0)
    ap.add_argument("--no-sweeps", action="store_true")
    args = ap.parse_args()
    args.lever = [float(v) for v in args.lever.split(",")]
    args.accel_bias = [float(v) for v in args.accel_bias.split(",")]

    qc.use_raw_dir(args.out)
    qc.ensure_dirs()
    rng = np.random.default_rng(args.seed)

    plan = [("still", lambda t0: build_still(t0, args.still_seconds) + (None,)),
            ("align", lambda t0: build_align(rng, t0) + (None,)),
            ("probe", lambda t0: build_probe(rng, args.probe_seconds, t0) + (None,)),
            ("probe_cal", lambda t0: build_probe(rng, args.probe_seconds, t0) + (None,))]
    if not args.no_sweeps:
        plan += [("sweep_tilt", lambda t0: build_sweep(t0, "tilt")),
                 ("sweep_roll", lambda t0: build_sweep(t0, "roll"))]

    t0 = 1.79e9
    program = {"blocks": [], "simulated": True, "injected": vars(args)}
    for name, build in plan:
        t, P, R, segs = build(t0)
        robot, imu, A_wb, B_it = synth_block(name, t, P, R, args, rng, segs)
        qc.save_table(os.path.join(qc.DIR_ROBOT, name), robot,
                      {"block": name, "pose_rate_hz": args.gt_hz, "paired_frac": 1.0,
                       "retreat_events": 0, "simulated": 1})
        slope, off, info = qc.fit_clock(imu["dev_us"] * 1e-6, imu["t_pc"])
        attrs = {"block": name, "n": int(imu["t"].size), "crc_errors": 0, "seq_gaps": 0,
                 "zero_still": 1, "mag_cal": 1, "use_mag": 1, "simulated": 1,
                 **{f"clock_{k}": v for k, v in info.items()}}
        qc.save_table(os.path.join(qc.DIR_IMU, name), imu, attrs)

        # 영점: 이 블록의 정지 구간에서 잡은 지구프레임 가속도 평균.
        # 실기 파이프라인(zero_ref.measure)이 하는 것과 같은 계산이다.
        still = qc.still_mask(imu["t"], imu["gyr"], imu["acc"])
        if still.sum() > 100:
            R_w = qc.quat_wxyz_to_R(imu["chip_q"])
            aE = np.einsum("nij,nj->ni", R_w, imu["acc"])
            b_E = aE[still].mean(0)
            # 키 이름을 **실기 로거와 같게** 쓴다 (zero_ref.to_dict()).
            # 예전에는 여기만 "b_E" 라 분석기가 시뮬에서만 영점을 찾았고, 실기
            # 에서는 조용히 블록 내 평균으로 대체됐다 — 자체 시험이 전부 통과하는
            # 채로. 시뮬이 실기와 다른 것을 쓰면 시험한 것과 실행한 것이 갈린다.
            qc.dump_json(os.path.join(qc.DIR_META, f"zero_{name}.json"),
                         {"accel_bias_earth": b_E.tolist(),
                          "gravity_measured": float(np.linalg.norm(b_E)),
                          "quality": {"still": True, "n": int(still.sum())}})
        entry = {"block": name, "t_start": float(t[0]), "t_stop": float(t[-1])}
        if segs:
            entry["segments"] = segs
        program["blocks"].append(entry)
        print(f"  {name:<12} {t[-1] - t[0]:6.1f} s   GT {robot['pose/t'].size:6d}"
              f"   IMU {imu['t'].size:6d}   정지 {100 * still.mean():3.0f} %")
        t0 = t[-1] + 5.0

    program["truth"] = {"A_base_from_imuworld": A_wb.tolist(),
                        "B_imu_from_tool": B_it.tolist(),
                        "lever_mm": args.lever, "imu_lag_ms": args.imu_lag,
                        "accel_scale": args.accel_scale}
    qc.dump_json(qc.PROGRAM_JSON, program)
    print(f"\n  -> {args.out}")
    print(f"  심은 값: 지연 {args.imu_lag:.0f} ms, 지렛대 {args.lever} mm, "
          f"스케일 {100 * (args.accel_scale - 1):+.1f} %")
    print(f"  확인:  python3 analyze_track.py --run {os.path.basename(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
