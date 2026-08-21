#!/usr/bin/env python3
"""IMU 추적 정확도 분석 — FR5 FK(GT) 대비 회전과 병진을 얼마나 따라가는가.

    python3 analyze_track.py
    python3 analyze_track.py --run runs/run01
    python3 analyze_track.py --sources chip            # 오프라인 재퓨전 생략 (빠름)

읽는 순서가 곧 신뢰의 순서다
----------------------------
1. **시간축**  레이트·지터·구멍·시계 skew. 이게 먼저다 — 시간축이 틀어진 데이터로
   낸 오차는 오차가 아니라 정렬 실패다.
2. **정렬**    IMU world <-> 로봇 base 고정회전 (A, B). 정지 자세에서만 푼다.
   움직이는 데이터로 풀면 **지연이 정렬오차로 둔갑한다.**
3. **회전**    지연(상호상관) -> 지연 보정 후 각오차 -> 드리프트. 9 축 vs 6 축.
4. **병진**    지렛대 r -> ZUPT 구간 분할 -> A/B/C 단계별 오차 -> 스케일 캘리브.
5. **판정**    통과 기준과 **기준값(reference.py)** 을 같이 찍는다.

세 가지 추정기를 같이 쓴다 (Surgilogger QC 와 같은 이유)
--------------------------------------------------------
· 상호상관 — teleop 구간의 실효 지연 하나. **그 운동의 스펙트럼에 딸린 값**이라
  조건이 바뀌면 바뀐다. 그래서 achieved 스펙트럼을 같이 찍는다.
· 록인    — sweep 블록이 있을 때만. 구간마다 한 주파수라 위상이 정의된다.
· L/T 피팅 — 위상과 이득을 같이 맞춰 순수지연 L 과 1 차 지체 T 를 가른다.
  IMU 는 T ~ 0 이어야 정상이다 (대역폭 손실이 없다는 뜻).

병진은 왜 다르게 평가하나
-------------------------
IMU 는 병진을 **직접 재지 못한다.** 이중적분은 상수 바이어스에 대해 t^2 로 자란다.
그래서 회전처럼 "전 구간 시계열 오차" 를 내면 그 숫자는 적분 길이의 함수일 뿐
센서의 성질이 아니다. 대신 **양 끝이 정지인 구간**으로 잘라 구간별 변위 오차를
내고(ZUPT), 창 길이별로 표를 만든다. 그 표가 "몇 초짜리 움직임까지 믿을 수
있는가" 라는 사용 가능 영역이다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "host"))

import protocol                                                 # noqa: E402
import qc_common as qc                                          # noqa: E402
import reference as ref                                         # noqa: E402

RESAMPLE_HZ = 200.0
MAX_LAG_S = 0.5             # IMU 지연은 수십 ms 대다. 1 s 까지 열어 두면 가짜
                            # 최대점(운동의 반주기)을 물 수 있다.
# 채널을 이렇게 고른 이유는 qc_common.tilt_deg / tip_xy_deg / spin_deg 의 문서주석에
# 있다. 요약: 프로빙 자세(프로브가 거의 연직)에서 '하방각 + 지평선 기준 롤' 은
# 무너진다. tip_x/tip_y/spin 은 무너지지 않고, tilt 는 기준 보고서의 하방각과
# 정확히 여각이라 그 표와 바로 비교된다.
ANGLE_CHANNELS = ("tip_x", "tip_y", "spin", "tilt")
REF_CHANNEL = {"tilt": "depression", "spin": "roll"}     # 기준 보고서와 짝지어지는 것


# ------------------------------------------------------------------- 적재
def load_block(block, sources=("chip",)):
    """(gt, imu). 없으면 (None, None)."""
    rb, ra = qc.load_table(os.path.join(qc.DIR_ROBOT, block))
    ib, ia = qc.load_table(os.path.join(qc.DIR_IMU, block))
    if rb is None or ib is None:
        return None, None

    t = np.asarray(rb["pose/t"], float)
    p = np.asarray(rb["pose/position"], float)
    q = np.asarray(rb["pose/quat_xyzw"], float)
    ok = np.isfinite(t) & np.all(np.isfinite(p), 1) & np.all(np.isfinite(q), 1)
    t, p, q = t[ok], p[ok], q[ok]
    order = np.argsort(t)
    t, p, q = t[order], p[order], q[order]
    R = qc.quat_xyzw_to_R(q)
    u = qc.probe_axis(R)
    tx, ty = qc.tip_xy_deg(u)
    gt = {"t": t, "p": p, "R": R, "u": u,
          "tilt": qc.tilt_deg(u), "tip_x": tx, "tip_y": ty,
          "spin": qc.unwrap_deg(qc.spin_deg(R)),
          "spin_valid": qc.spin_valid(u), "tip_valid": qc.tip_valid(u),
          "attrs": ra, "audit": qc.rate_audit(t, f"{block}/pose"),
          "paired_frac": float(np.asarray(rb["pose/paired"], float).mean()),
          "retreat": np.asarray(rb.get("retreat/data", np.empty((0, 1))), float),
          "track_err": np.asarray(rb.get("track_err/data", np.empty((0, 2))), float)}

    ti = np.asarray(ib["t"], float)
    imu = {"t": ti, "t_pc": np.asarray(ib["t_pc"], float),
           "acc": np.asarray(ib["acc"], float), "gyr": np.asarray(ib["gyr"], float),
           "mag": np.asarray(ib["mag"], float),
           "mag_raw": np.asarray(ib.get("mag_raw", ib["mag"]), float),
           "cal_status": (np.asarray(ib["cal_status"], float)
                          if "cal_status" in ib else None),
           "attrs": ia, "audit": qc.rate_audit(ti, f"{block}/imu"),
           "R": {}}
    imu["R"]["chip"] = qc.quat_wxyz_to_R(np.asarray(ib["chip_q"], float))
    if "host9" in sources or "host6" in sources:
        for name, use_mag in (("host9", True), ("host6", False)):
            if name in sources:
                imu["R"][name] = refuse(imu, use_mag=use_mag)
    return gt, imu


def mag_model_residual(pairs_in, restarts=12):
    """m = K R h + b 를 **로봇의 알려진 자세로** 푼다.

    K 가 소프트아이언·축 비직교·스케일, 그리고 센서<->플랜지 고정회전까지 통째로
    흡수하므로 (K X 도 그냥 3x3 이다), 이 모형은 **센서에 고정된 왜곡을 남김없이**
    담는다. 핸드아이 해가 따로 필요 없다. 남는 잔차는 정의상 센서에 안 붙은 것이다.

    회전을 안 쓰고 불변량만으로 같은 것을 풀면 조건수가 나빠 잔차가 부풀 수 있다
    (2026-08-20 에 그렇게 6.7 deg 를 얻었는데, 회전을 넣고 풀자 5.6 deg 였고
    하드아이언 12 uT 가 |m| 산포를 대부분 설명했다). 그래서 회전이 있으면 이쪽을 쓴다.

    **비대칭에 주의.** 잔차가 크면 '이 모형으로 설명 안 된다' 가 증명되지만,
    잔차가 작다고 모형이 맞는 것은 아니다 — 자세가 한 축으로만 퍼지면 잔차 0
    이면서 해가 틀릴 수 있다 (합성 검증에서 확인). 그래서 자세 다양성을 같이 낸다.
    """
    H = []
    for gt, imu in pairs_in:
        mask = qc.still_mask(imu["t"], imu["gyr"], imu["acc"])
        for (t0, t1, _i0, _i1) in qc.segments_from_mask(imu["t"], mask, 0.8):
            span = t1 - t0
            lo, hi = t0 + 0.2 * span, t1 - 0.2 * span
            mi = (imu["t"] >= lo) & (imu["t"] <= hi)
            mr = (gt["t"] >= lo) & (gt["t"] <= hi)
            if mi.sum() < 50 or mr.sum() < 5:
                continue
            m = imu["mag_raw"][mi]
            n = np.linalg.norm(m, axis=1)
            ok = np.isfinite(m).all(1) & (n > 1.0) & (n < 200.0)
            if ok.sum() < 30:
                continue
            # R_eb = R_be^T — base 벡터를 센서 프레임으로 보낸다
            H.append((qc.project_SO3(gt["R"][mr].mean(0)).T, m[ok].mean(0),
                      float(np.linalg.norm(m[ok].std(0)))))
    if len(H) < 8:
        return None
    Rs = np.array([x[0] for x in H])
    M = np.array([x[1] for x in H])
    noise = float(np.median([x[2] for x in H]))

    def solve(seed):
        rng = np.random.default_rng(seed)
        h = rng.normal(size=3)
        h /= np.linalg.norm(h)
        K, b = np.eye(3), np.zeros(3)
        for _ in range(300):
            U = np.einsum("nij,j->ni", Rs, h)
            sol, *_ = np.linalg.lstsq(np.hstack([U, np.ones((len(U), 1))]), M, rcond=None)
            K, b = sol[:3].T, sol[3]
            h, *_ = np.linalg.lstsq(np.concatenate([K @ R for R in Rs], 0),
                                    (M - b).reshape(-1), rcond=None)
            n = np.linalg.norm(h)
            if n < 1e-12:
                break
            h, K = h / n, K * n
        return K, b, h

    best = None
    for seed in range(restarts):
        K, b, h = solve(seed)
        P = np.array([K @ R @ h + b for R in Rs])
        r = float(np.linalg.norm(M - P, axis=1).mean())
        if best is None or r < best[0]:
            best = (r, K, b, h, P)
    resid, K, b, h, P = best
    ang = np.degrees(np.arccos(np.clip(
        (M * P).sum(1) / np.linalg.norm(M, axis=1) / np.linalg.norm(P, axis=1), -1, 1)))
    field = float(np.linalg.norm(K @ h))
    # 자세 다양성 — 회전축이 한 방향으로 몰리면 해가 유일하지 않다.
    # 자세 다양성 — 기준 자세 대비 회전벡터들이 세 축으로 고루 퍼졌는가.
    # 최소/최대 특이값 비. 0 에 가까우면 한 축(또는 한 평면)으로만 퍼진 것이고,
    # 그때는 잔차가 작아도 해가 유일하지 않다.
    axes = qc.so3_log_deg(Rs[1:], np.repeat(Rs[:1], len(Rs) - 1, axis=0))
    sv = np.linalg.svd(axes, compute_uv=False) if len(axes) > 2 else np.zeros(3)
    div = float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0
    return {"n_holds": len(H), "resid_uT": resid, "noise_uT": noise,
            "resid_over_noise": float(resid / noise) if noise > 0 else float("nan"),
            "dir_err_deg": float(ang.mean()), "dir_err_max_deg": float(ang.max()),
            "field_uT": field, "hard_iron_uT": float(np.linalg.norm(b)),
            "resid_pct_of_field": float(100 * resid / field) if field > 0 else float("nan"),
            "pose_diversity": div}


def mag_position_test(gt, imu, max_dR_deg=3.0, min_dp_mm=100.0):
    """**모형 없는** 판정 — 자세가 같고 위치만 다른 정지구간 쌍을 비교한다.

    센서에 고정된 자기 왜곡은 자세의 함수다 (m = K R h + b). 자세가 같으면
    그 예측은 완전히 같다 — K, h, b 가 무엇이든. 그러므로 측정된 m 이 다르면
    남는 설명은 '장이 위치에 따라 다르다' 하나뿐이다. 적합도 필요 없고, 적합이
    유일한지 걱정할 필요도 없다.

    비교의 바닥은 **구간 안 잡음**이다. 그보다 큰 차이만 뜻이 있다.
    """
    mask = qc.still_mask(imu["t"], imu["gyr"], imu["acc"])
    segs = qc.segments_from_mask(imu["t"], mask, 0.8)
    H = []
    for (t0, t1, _i0, _i1) in segs:
        span = t1 - t0
        lo, hi = t0 + 0.2 * span, t1 - 0.2 * span
        mi = (imu["t"] >= lo) & (imu["t"] <= hi)
        mr = (gt["t"] >= lo) & (gt["t"] <= hi)
        if mi.sum() < 50 or mr.sum() < 5:
            continue
        m = imu["mag_raw"][mi]
        n = np.linalg.norm(m, axis=1)
        ok = np.isfinite(m).all(1) & (n > 1.0) & (n < 200.0)
        if ok.sum() < 30:
            continue
        H.append({"R": qc.project_SO3(gt["R"][mr].mean(0)),
                  "p": gt["p"][mr].mean(0) * 1000.0,
                  "m": m[ok].mean(0),
                  "sd": float(np.linalg.norm(m[ok].std(0)))})
    if len(H) < 3:
        return None

    noise = float(np.median([x["sd"] for x in H]))
    pairs = []
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            c = (np.trace(H[i]["R"].T @ H[j]["R"]) - 1.0) / 2.0
            dR = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
            dp = float(np.linalg.norm(H[i]["p"] - H[j]["p"]))
            if dR > max_dR_deg or dp < min_dp_mm:
                continue
            mi_, mj_ = H[i]["m"], H[j]["m"]
            d = float(np.linalg.norm(mj_ - mi_))
            ang = float(np.degrees(np.arccos(np.clip(
                mi_ @ mj_ / np.linalg.norm(mi_) / np.linalg.norm(mj_), -1.0, 1.0))))
            pairs.append({"dR_deg": dR, "dp_mm": dp, "dm_uT": d, "dm_deg": ang})
    if not pairs:
        return {"n_holds": len(H), "n_pairs": 0, "noise_uT": noise}
    dm = np.array([p["dm_uT"] for p in pairs])
    return {"n_holds": len(H), "n_pairs": len(pairs), "noise_uT": noise,
            "dm_median_uT": float(np.median(dm)), "dm_max_uT": float(dm.max()),
            "dm_over_noise": float(np.median(dm) / noise) if noise > 0 else float("nan"),
            "dm_deg_median": float(np.median([p["dm_deg"] for p in pairs])),
            "dp_max_mm": float(max(p["dp_mm"] for p in pairs)),
            "pairs": pairs[:50]}


def chip_cal_status(imu):
    """BNO085 가 스스로 매긴 보정 정확도 (0=미보정 ~ 3=완전), 센서별 중앙값과
    '3 에 도달한 시간 비율'.

    이게 없으면 자기 교란 판정이 반쪽이다 — 자세오차가 큰 이유가 '환경이
    나쁘다' 인지 '칩 보정이 애초에 수렴을 안 했다' 인지 못 가른다. 둘은 대처가
    정반대다. 2026-08-20 run 은 이 열이 없어 그 질문에 답하지 못했다.
    """
    cs = imu.get("cal_status")
    if cs is None or not np.size(cs):
        return None
    names = ("acc", "gyr", "mag", "rv")
    out = {}
    for i, nm in enumerate(names):
        v = cs[:, i]
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        out[nm] = {"median": float(np.median(v)),
                   "frac_full": float(np.mean(v >= 3)),
                   "min": float(v.min())}
    return out or None


def magnetic_disturbance(imu, deep=False):
    """자력계가 **보정으로 고칠 수 있는 상태인가.**

    회전 불변량 두 개를 쓴다 — 세기 |m| 과 복각(중력과 자기벡터 사이각). 지구
    자기장 안이라면 둘 다 자세와 무관한 상수여야 한다. 하드아이언·소프트아이언은
    정의상 **센서 프레임에 고정된** 왜곡이므로, 그런 왜곡만 보정이 지울 수 있고
    그때는 어떤 (S, b) 가 두 불변량을 동시에 상수로 만든다.

    그래서 '그런 (S, b) 가 존재하는가' 를 직접 푼다. 최적해에서도 복각이 흩어져
    있으면, 남은 것은 **팔이 어디 있느냐에 따라 장 자체가 달라지는 몫**이고 이건
    어떤 보정 계수로도 못 지운다. 그 각도가 곧 heading 에 실리는 오차다.

    2026-08-20 실기에서 12 파라미터 최적해로도 복각이 +-6.7 deg 남았다 — 정상적인
    보정이면 1 deg 아래다. 그 4.8 deg 가 정렬 잔차 4.69 deg 로 그대로 나왔다.
    """
    a, m0, mr = imu["acc"], imu["mag"], imu["mag_raw"]
    mask = qc.still_mask(imu["t"], imu["gyr"], imu["acc"])
    if mask.sum() < 200:
        return None
    out = {}
    for tag, m in (("cal", m0), ("raw", mr)):
        A, M = a[mask], m[mask]
        n = np.linalg.norm(M, axis=1)
        ok = np.isfinite(M).all(1) & np.isfinite(A).all(1) & (n > 1.0) & (n < 200.0)
        if ok.sum() < 200:
            continue
        A, M = A[ok], M[ok]
        out[tag] = {"norm_pct_sd": float(100 * np.std(np.linalg.norm(M, axis=1))
                                         / np.mean(np.linalg.norm(M, axis=1))),
                    "dip_sd_deg": float(_dip_deg(A, M).std())}
    # 12 파라미터 적합은 비싸고, **자세가 다양한 블록에서만** 뜻이 있다 (한 자세만
    # 있으면 어떤 보정이든 그 한 점을 맞출 수 있어 0 이 나온다). align 에서만 켠다.
    if deep and "raw" in out:
        out["irreducible_dip_sd_deg"] = _irreducible_dip(a[mask], mr[mask])
    return out or None


def _dip_deg(A, M):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((A * M).sum(1), -1, 1)))


def _irreducible_dip(A, M, n_pose=40):
    """전체 3x3 소프트아이언 + 하드아이언(12 param)을 **불변량 산포를 줄이라고**
    직접 맞춘 뒤에도 남는 복각 산포. 보정의 이론적 한계다.

    자세마다 한 점으로 줄여 푼다 — 같은 자세의 수천 샘플이 최적화를 지배하면
    자세 다양성이 아니라 체류시간을 맞추게 된다.
    """
    from scipy.optimize import minimize

    ok = np.isfinite(M).all(1) & np.isfinite(A).all(1)
    A, M = A[ok], M[ok]
    n = np.linalg.norm(M, axis=1)
    ok = (n > 1.0) & (n < 200.0)
    A, M = A[ok], M[ok]
    if len(A) < 200:
        return float("nan")
    k = max(1, len(A) // n_pose)
    A = np.array([A[i:i + k].mean(0) for i in range(0, len(A) - k + 1, k)])
    M = np.array([M[i:i + k].mean(0) for i in range(0, len(M) - k + 1, k)])

    def cost(p):
        Mx = (M - p[:3]) @ p[3:12].reshape(3, 3).T
        nn = np.linalg.norm(Mx, axis=1)
        if np.any(nn < 1e-6):
            return 1e6
        return (100 * nn.std() / nn.mean()) ** 2 + _dip_deg(A, Mx).std() ** 2

    best = None
    rng = np.random.default_rng(0)
    for _ in range(16):
        x0 = np.r_[rng.normal(0, 20, 3), (np.eye(3) + rng.normal(0, 0.2, (3, 3))).ravel()]
        r = minimize(cost, x0, method="Nelder-Mead",
                     options=dict(maxiter=20000, maxfev=20000, xatol=1e-8, fatol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    Mx = (M - best.x[:3]) @ best.x[3:12].reshape(3, 3).T
    return float(_dip_deg(A, Mx).std())


def refuse(imu, use_mag=True, beta=0.05):
    """원시 acc/gyr/mag 로 호스트 Madgwick 을 **오프라인에서 다시** 돌린다.

    같은 기록에서 9 축과 6 축 자세를 둘 다 뽑기 위한 것이다. 로봇은 큰 강자성체라
    자세마다 자기환경이 달라진다 — heading 이 얼마나 버티는지는 두 결과를
    나란히 놓아야만 답이 난다. 칩 RV 만 남기면 그 질문 자체가 사라진다.
    """
    from fusion import Madgwick, quat_to_matrix, seed_quat

    t, a, g, m = imu["t"], imu["acc"], imu["gyr"], imu["mag"]
    filt = Madgwick(beta=beta)
    # **반드시 seed 한다.** 항등에서 출발하면 프로빙 자세(프로브가 아래를 봄)는
    # 참 자세가 180 도 떨어져 있어 수렴이 아주 느리거나 아예 안 된다 — 그러면
    # '6 축이 30 도 틀린다' 는 결론이 나오는데 그건 센서가 아니라 초기값 이야기다.
    # 6 축은 heading 을 정할 수 없으므로 임의의 수평 기준으로 seed 한다 (요 원점은
    # 어차피 관측 불가라 정렬 A,B 가 흡수한다 — PER_BLOCK_ALIGN 참조).
    ref = (m[0] if (use_mag and np.all(np.isfinite(m[0])))
           else (np.array([1.0, 0.0, 0.0])
                 if abs(a[0][0]) < 0.9 * np.linalg.norm(a[0]) else np.array([0.0, 1.0, 0.0])))
    q0 = seed_quat(a[0], ref)
    if q0 is not None:
        filt.q = q0
    out = np.empty((t.size, 3, 3))
    prev = t[0]
    for k in range(t.size):
        dt = t[k] - prev
        prev = t[k]
        if not (0.0 < dt < 0.5):
            dt = 0.004
        mm = m[k] if (use_mag and np.all(np.isfinite(m[k]))) else None
        filt.update(g[k], a[k], mm, dt)
        out[k] = quat_to_matrix(filt.q)
    return out


def add_channels(imu, src, A, B):
    """R_hat = A R_imu B 로 base 프레임 자세를 만들고 각도 채널을 붙인다."""
    R = np.einsum("ij,njk,kl->nil", A, imu["R"][src], B)
    u = qc.probe_axis(R)
    tx, ty = qc.tip_xy_deg(u)
    return {"R": R, "u": u, "tilt": qc.tilt_deg(u), "tip_x": tx, "tip_y": ty,
            "spin": qc.unwrap_deg(qc.spin_deg(R))}


# ------------------------------------------------------------------- 정렬
def solve_AB(X, Y):
    """X_i = A Y_i B 를 푼다. (Surgilogger QC 와 같은 해법)

    X 는 로봇 tool 자세(base<-tool), Y 는 IMU 자세(world<-imu). 둘 다 같은 강체를
    보므로 A(base<-world) 와 B(imu<-tool) 는 상수다. 상대회전으로 넘기면 A 가
    소거된다: X_0^T X_i = B^T (Y_0^T Y_i) B. D=B 로 두면 N_i D = D M_i 라는
    선형식이고, vec 를 취해 최소특이벡터로 푼다. 널벡터는 부호가 자유로우므로
    det<0 이면 뒤집는다 — 안 그러면 반사가 섞여 180 도 틀린 해가 나온다.

    s[-2] 가 **관측도**다. 0 에 가까우면 정지 자세들이 한 축으로만 기울어 B 의
    그 축 둘레 성분이 관측되지 않은 것이고, 그러면 이 정렬을 믿으면 안 된다.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if len(X) < protocol.ALIGN_MIN_POSES:
        raise SystemExit(f"정지 자세가 {protocol.ALIGN_MIN_POSES} 개 미만이다 — 정렬을 풀 수 없다")
    rows = [np.kron(np.eye(3), Y[0].T @ Y[i]) - np.kron((X[0].T @ X[i]).T, np.eye(3))
            for i in range(1, len(X))]
    _, s, Vt = np.linalg.svd(np.vstack(rows))
    D = Vt[-1].reshape(3, 3, order="F")
    if np.linalg.det(D) < 0:
        D = -D
    B = qc.project_SO3(D)
    A = qc.project_SO3(sum(X[i] @ B.T @ Y[i].T for i in range(len(X))))
    resid = np.array([qc.geodesic_deg(X[i], A @ Y[i] @ B) for i in range(len(X))])
    return A, B, float(s[-2]), resid


def still_poses(gt, imu, src, min_s=None):
    """정지 구간마다 (GT 자세 평균, IMU 자세 평균) 한 쌍.

    정지 판정은 **IMU 로** 한다 (로봇 FK 는 지령이 멈춰도 미세하게 흔들린다).
    구간의 가운데 60 % 만 쓴다 — 앞뒤는 아직 정착 중이라 지연이 섞인다.
    """
    min_s = protocol.ALIGN_HOLD_S if min_s is None else min_s
    mask = qc.still_mask(imu["t"], imu["gyr"], imu["acc"])
    segs = qc.segments_from_mask(imu["t"], mask, min_s)
    X, Y, used = [], [], []
    for (t0, t1, i0, i1) in segs:
        span = t1 - t0
        lo, hi = t0 + 0.2 * span, t1 - 0.2 * span
        mg = (gt["t"] >= lo) & (gt["t"] <= hi)
        mi = (imu["t"] >= lo) & (imu["t"] <= hi)
        if mg.sum() < 5 or mi.sum() < 20:
            continue
        X.append(qc.project_SO3(gt["R"][mg].mean(axis=0)))
        Y.append(qc.project_SO3(imu["R"][src][mi].mean(axis=0)))
        used.append({"t_start": t0, "t_stop": t1, "n_gt": int(mg.sum()),
                     "n_imu": int(mi.sum())})
        _ = (i0, i1)
    return X, Y, used, segs


def alignment(gt, imu, src):
    X, Y, used, _ = still_poses(gt, imu, src)
    A, B, obs, resid = solve_AB(X, Y)
    return A, B, {"source": src, "n": len(X), "observability": float(obs),
                  "resid_deg": resid.tolist(),
                  "resid_rms_deg": float(np.sqrt((resid ** 2).mean())),
                  "resid_max_deg": float(resid.max()),
                  "holds": used}


# --------------------------------------------------------------- 상호상관
def resample(t, x, grid):
    ok = np.isfinite(t) & np.isfinite(x)
    if ok.sum() < 2:
        return np.full(grid.shape, np.nan)
    return np.interp(grid, t[ok], x[ok], left=np.nan, right=np.nan)


def xcorr_lag(dt, a_gt, b_sn, max_lag=MAX_LAG_S):
    """센서가 GT 보다 늦은 시간 [s] 과 그때의 상관. **부호가 있다.**

    sn(t) ~ gt(t - tau) 이므로 sn 을 tau 만큼 **당겨서** gt 와 겹칠 때 최대다.
    최대점은 포물선으로 보간한다 — 격자(5 ms)보다 잘게 읽기 위해서.

    음수(센서가 GT 보다 **앞섬**)도 찾는다. GT 쪽에는 컨트롤러 UDP 상태 패키지가
    PC 에 닿는 몫이 이미 실려 있고(qc_common.ROBOT_STATE_LAG_NOTE), 그 몫이 IMU
    자신의 지연보다 크면 센서가 앞선 것으로 보인다. 한쪽만 뒤지면 그 경우가 0 으로
    잘려 "지연 0 ms — 통과" 라는 **거짓 통과**가 된다. 실제로 이 리그의 첫 run 이
    네 채널 모두 정확히 0.0 ms 를 냈고, 양쪽을 열자 -15 ~ -50 ms 로 드러났다.
    """
    a = np.asarray(a_gt, float).copy(); b = np.asarray(b_sn, float).copy()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 64:
        return np.nan, np.nan
    a -= np.mean(a[ok]); b -= np.mean(b[ok])
    a[~ok] = 0.0; b[~ok] = 0.0
    kmax = int(max_lag / dt)
    ks = np.arange(-kmax, kmax + 1)
    cs = np.empty(ks.size)
    for i, k in enumerate(ks):
        if k >= 0:
            aa, bb = (a[:len(a) - k] if k else a), b[k:]
        else:
            aa, bb = a[-k:], b[:len(b) + k]
        n = np.sqrt(np.dot(aa, aa) * np.dot(bb, bb))
        cs[i] = np.dot(aa, bb) / n if n > 0 else 0.0
    i = int(np.argmax(cs))
    d = 0.0
    if 0 < i < cs.size - 1:
        y0, y1, y2 = cs[i - 1], cs[i], cs[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return float((ks[i] + d) * dt), float(cs[i])


def shift_apply(x, dt, lag):
    """x 를 lag 만큼 앞으로 당긴 신호 (지연 보정). 길이 유지, 끝은 nan."""
    n = len(x)
    idx = np.arange(n) + lag / dt
    return np.interp(idx, np.arange(n), x, left=np.nan, right=np.nan)


def motion_spectrum(dt, x):
    """이 블록이 실제로 어떤 주파수로 흔들렸나.

    상호상관 실효 지연은 **그 운동의 스펙트럼에 딸린 값**이다. teleop 은 매번
    스펙트럼이 다르므로, 숫자만 옮겨 적으면 다음 run 과 비교가 안 된다.
    그래서 대역폭(누적 파워 90 % 지점)과 중앙 주파수를 같이 남긴다.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 64:
        return {}
    x = x - x.mean()
    f = np.fft.rfftfreq(x.size, dt)
    P = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    P[0] = 0.0
    c = np.cumsum(P) / max(P.sum(), 1e-30)
    return {"f_peak_hz": float(f[int(np.argmax(P))]),
            "f_median_hz": float(np.interp(0.5, c, f)),
            "f_p90_hz": float(np.interp(0.9, c, f))}


# ------------------------------------------------------------------- 록인
def lockin(t, x, f, t0):
    """(진폭, 위상 rad, 잔차비). 위상 기준시각 t0 을 GT 와 센서가 **공유**해야 한다."""
    t = np.asarray(t, float); x = np.asarray(x, float)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if t.size < 16:
        return np.nan, np.nan, np.nan
    tc = t - t0
    w = 2 * np.pi * f
    M = np.column_stack([np.cos(w * tc), np.sin(w * tc), np.ones_like(tc), tc])
    coef, *_ = np.linalg.lstsq(M, x, rcond=None)
    a, b = coef[0], coef[1]
    amp = float(np.hypot(a, b))
    resid = float(np.sqrt(np.mean((x - M @ coef) ** 2)))
    return amp, float(np.arctan2(-b, a)), (resid / amp if amp > 0 else np.inf)


def wrap_pi(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def fit_LT(freqs, dphase, gain, weight):
    """위상과 이득을 같이 맞춰 (L, T).

        위상(w) = -(wL + atan(wT))        이득(w) = 1/sqrt(1+(wT)^2)

    이득을 같이 쓰는 이유: 위상만 맞추면 낮은 주파수에서 atan(wT) ~ wT 라 L 과 T 가
    맞바꿔진다. 이득은 L 에 무관하므로 T 를 따로 묶어 준다. 그래도 완전히
    없어지지는 않으니 **L+T 를 먼저 보고**, L 과 T 를 따로 인용할 때는 이 축퇴를
    함께 밝힌다.
    """
    from scipy.optimize import least_squares
    f = np.asarray(freqs, float)
    w = 2 * np.pi * f
    ph = np.asarray(dphase, float); g = np.asarray(gain, float)
    wt = np.asarray(weight, float)
    ok = np.isfinite(ph) & np.isfinite(g) & (g > 0) & np.isfinite(wt)
    if ok.sum() < 2:
        return {"L": np.nan, "T": np.nan, "ok": False, "n": int(ok.sum())}
    w, ph, g, wt = w[ok], ph[ok], g[ok], wt[ok]

    def resid(p):
        L, T = p
        return np.concatenate([wt * wrap_pi(ph + (w * L + np.arctan(w * T))),
                               0.5 * wt * (np.log(g) + 0.5 * np.log1p((w * T) ** 2))])

    best = None
    for L0 in (0.0, 0.02, 0.05):
        for T0 in (0.0, 0.02, 0.1):
            try:
                r = least_squares(resid, [L0, T0], bounds=([-0.05, 0.0], [0.5, 1.0]))
            except Exception:                                   # noqa: BLE001
                continue
            if best is None or r.cost < best.cost:
                best = r
    if best is None:
        return {"L": np.nan, "T": np.nan, "ok": False, "n": int(ok.sum())}
    L, T = best.x
    return {"L": float(L), "T": float(T), "ok": True, "n": int(ok.sum()),
            "cost": float(best.cost), "dc_group_delay": float(L + T)}


# ------------------------------------------------------------------- 병진
def earth_accel(R_base_sensor, acc):
    """센서 프레임 가속도를 **로봇 base 프레임**으로 돌린다.

    입력은 R_base<-sensor = A R_imu 다. R_imu (IMU 자신의 world 기준) 를 그대로
    넣으면 안 된다 — 그러면 가속도는 IMU world 에, GT 변위는 base 에 있게 되어
    **두 좌표계가 A 만큼 어긋난 채 빼는** 것이 된다. 중력 z 는 두 프레임이 같아
    검산(+g)을 통과해 버리므로, 이 실수는 조용히 지나간다. 실제로 그렇게 만들었다가
    시뮬레이터가 잡았다 (심은 값 대비 변위 오차가 100 배로 나왔다).

    규약(센서->지구인가 그 역인가)은 문서가 아니라 데이터로 정한다: 정지 시
    z 성분이 +g 에 가까워지는 쪽이 맞는 것이다. 반대로 잡으면 중력이 안 빠지고
    오차가 100 배가 되는데, 그 증상은 '센서가 나쁘다' 로 오인된다.
    """
    fwd = np.einsum("nij,nj->ni", R_base_sensor, acc)
    inv = np.einsum("nji,nj->ni", R_base_sensor, acc)
    score = lambda v: abs(np.nanmean(v[:, 2]) - qc.G0) + np.abs(np.nanmean(v[:, :2], 0)).sum()
    return (fwd, "R") if score(fwd) <= score(inv) else (inv, "R.T")


def _cumtrapz(y, dt):
    """0 에서 시작하는 사다리꼴 누적적분.

    직사각형(cumsum*dt)을 쓰면 **상수 입력에서도 ZUPT 가 정확히 0 을 안 낸다** —
    v[k] 가 b*(k+1)dt 라 선형 디드리프트의 램프(k/(n-1))와 반 칸 어긋나기
    때문이다. 0.03 m/s^2 바이어스, 2 s 구간에서 0.3 mm 가 남는다. 오차 바닥이
    0.2 mm 대인 QC 에서 그건 무시할 수 없는 몫이고, 무엇보다 **파이프라인이
    스스로 만든 오차**다.
    """
    out = np.zeros_like(np.asarray(y, float))
    if len(out) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt, axis=0)
    return out


def zupt_integrate(a_lin, dt, zupt=True):
    """가속도 -> 변위. zupt 면 구간 끝 속도가 0 이라는 조건으로 선형 디드리프트.

    ZUPT 가 **상수 가속도 바이어스를 원리적으로 제거한다** — 그게 이 QC 가
    "양 끝이 정지" 인 구간만 보는 이유다. 실제 프로빙의 리듬(자세를 잡고 ->
    눌러 보고 -> 멈춰 영상을 보고 -> 옮긴다)이 그 조건을 공짜로 만족한다.
    """
    v = _cumtrapz(np.asarray(a_lin, float), dt)
    if zupt and len(v) > 1:
        v = v - v[-1] * np.linspace(0.0, 1.0, len(v))[:, None]
    return _cumtrapz(v, dt)


# ------------------------------------------------- 가속도 모형 동정 (C, d, r)
def _ident_design(P, Rg, R_bs, a, dt, win_s=0.25):
    """한 블록의 설계행렬 (M, rhs). 15 열 = C(9) + d(3) + r(3).

        R_bs (C a + d) - g  =  p_ddot + R_ddot r

    **대역폭을 맞추는 것이 중요하다.** 우변은 Savitzky-Golay 미분이라 이미
    저역통과를 통과한 신호다. 좌변을 날것으로 두면 센서 쪽에만 남은 고주파가
    통째로 잔차가 되어, 모형이 설명할 수 있는 몫까지 잡음에 묻힌다. 그래서
    좌변 열에 **같은 창의 평활(deriv=0)** 을 건다. 선형 연산이므로 해는 안 변한다.
    """
    from scipy.signal import savgol_filter

    win = max(7, int(round(win_s / dt)) | 1)
    ok = (np.all(np.isfinite(P), 1) & np.all(np.isfinite(Rg.reshape(9, -1)), 0)
          & np.all(np.isfinite(a), 1) & np.all(np.isfinite(R_bs.reshape(-1, 9)), 1))
    if ok.sum() < win * 8:
        return None
    i0, i1 = np.where(ok)[0][[0, -1]]
    i1 += 1
    P, Rg, R_bs, a = P[i0:i1], Rg[:, :, i0:i1], R_bs[i0:i1], a[i0:i1]
    p_dd = savgol_filter(P, win, 3, deriv=2, delta=dt, axis=0)
    R_dd = savgol_filter(Rg, win, 3, deriv=2, delta=dt, axis=2)

    sm = lambda x: savgol_filter(x, win, 3, axis=0)
    cols = [sm(R_bs[:, :, i] * a[:, j:j + 1]) for i in range(3) for j in range(3)]
    cols += [sm(R_bs[:, :, i]) for i in range(3)]
    cols += [-R_dd[:, j, :].T for j in range(3)]        # 이미 같은 창을 통과했다
    M = np.stack(cols, axis=2).reshape(-1, 15)
    rhs = (p_dd + np.array([0.0, 0.0, qc.G0])).reshape(-1)
    good = np.isfinite(rhs) & np.all(np.isfinite(M), 1)
    return M[good], rhs[good]


def identify_accel_model(idents, r0=None, lam=1e-3):
    r"""스케일·비직교 C, 바이어스 d, 지렛대 r 을 **한꺼번에** 푼다.

        R_bs(t) (C a(t) + d)  -  g   =   p_ddot(t)  +  R_ddot(t) r
        \_____ 센서가 말하는 가속 _____/     \__ 로봇이 아는 IMU 자리의 가속 __/

    셋을 나눠 풀면 안 되는 이유가 실측으로 드러났다. 지렛대만 따로 회귀하면
    좌변이 **스케일 오차 x 중력**에 지배된다 — 2.3 % 스케일 오차는 0.23 m/s^2 를
    만드는데, 이 리그의 실제 가속(0.05 m/s^2)과 지렛대 항(0.02 m/s^2)은 그보다
    한 자릿수 작다. 그래서 r 회귀가 아무것도 설명하지 못하고 엉뚱한 값을 자신
    있게 돌려준다 (시뮬레이터에서 60,-20,35 mm 를 39,-65,9 로 냈다).

    15 개 미지수에 대해 **선형**이므로 최소자승으로 풀린다. 항등(C=I, d=0, r=r0)
    쪽으로 Tikhonov 를 약하게 걸어, 자세 다양성이 부족할 때 해가 날아가지 않게
    한다. 파라미터마다 단위가 다르므로(C 무차원, d m/s^2, r m) 열 노름으로
    정규화한 뒤 건다 — 안 그러면 d 만 움직이는 해가 나온다.

    **여러 블록을 같이 넣는다.** 관측 조건이 서로 다르기 때문이다.
      · C, d 는 **여러 자세**가 있어야 관측된다 -> align 블록(정지 자세 10 개)이
        가장 좋은 자료다. 정지 상태에서는 p_ddot = R_ddot = 0 이라 식이
        R_bs(Ca+d) = g 로 줄어, 고전적인 다자세 가속도계 보정 그 자체가 된다.
      · r 은 **회전**이 있어야 관측된다 (R_ddot = 0 이면 r 이 식에서 사라진다)
        -> probe 블록의 회전 구간이 그 몫을 낸다.
    """
    des = [d for d in (_ident_design(**x) for x in idents) if d is not None]
    if not des:
        return None
    M = np.vstack([d[0] for d in des])
    rhs = np.concatenate([d[1] for d in des])
    r0 = np.zeros(3) if r0 is None else np.asarray(r0, float)
    theta0 = np.concatenate([np.eye(3).reshape(-1), np.zeros(3), r0])
    resid0 = rhs - M @ theta0
    w = np.maximum(np.linalg.norm(M, axis=0), 1e-9)
    Aug = np.vstack([M / w, lam * np.eye(15)])
    dth, *_ = np.linalg.lstsq(Aug, np.concatenate([resid0, np.zeros(15)]), rcond=None)
    theta = theta0 + dth / w
    resid1 = rhs - M @ theta

    C = theta[:9].reshape(3, 3)
    d = theta[9:12]
    r = theta[12:]
    sing = np.linalg.svd(C, compute_uv=False)
    cx, cy = C[:, 0], C[:, 1]
    return {"C": C.tolist(), "bias_ms2": d.tolist(), "r_m": r.tolist(),
            "lever_mm": (r * 1000.0).tolist(),
            "lever_norm_mm": float(np.linalg.norm(r) * 1000.0),
            "scale_pct": [float(100.0 * (v - 1.0)) for v in sing],
            "nonorthogonality_deg": float(np.degrees(np.arccos(np.clip(
                np.dot(cx, cy) / (np.linalg.norm(cx) * np.linalg.norm(cy)), -1, 1))) - 90.0),
            "accel_resid_before_ms2": float(np.sqrt(np.mean(resid0 ** 2))),
            "accel_resid_after_ms2": float(np.sqrt(np.mean(resid1 ** 2))),
            "n_blocks": len(des), "n": int(len(rhs) // 3)}


def apply_model(a, model):
    """a_corr = C a + d. model 이 없으면 그대로 돌려준다."""
    if not model:
        return a
    return np.einsum("ij,nj->ni", np.asarray(model["C"], float), a) \
        + np.asarray(model["bias_ms2"], float)


def gt_point_path(gt, grid, r):
    """GT 위치를 **IMU 자리**로 옮긴 궤적. p_imu = p + R r.

    이걸 안 하면 IMU 는 자기 자리의 운동을 보고하는데 GT 는 플랜지 원점의 운동을
    말하게 되어, 회전할 때마다 |r| 만큼의 차이가 '센서 오차' 로 잡힌다.
    """
    P = np.column_stack([resample(gt["t"], gt["p"][:, k], grid) for k in range(3)])
    if r is None:
        return P
    Rr = np.stack([resample(gt["t"], (gt["R"] @ np.asarray(r, float))[:, k], grid)
                   for k in range(3)], 1)
    return P + Rr


def gt_still_mask(P, R, dt, win_s=0.30, pos_sd_mm=0.3, ang_sd_deg=0.05):
    """GT 로 본 정지. **미분하지 않고** 이동창 안의 산포로 본다.

    속도를 수치미분해서 문턱을 걸면 안 된다 — 0.05 mm 짜리 위치 잡음도 200 Hz
    에서 미분하면 10 mm/s 가 되어, 완전히 서 있는 구간이 '움직이는 중' 으로
    나온다. 산포는 잡음을 평균해 없앤다.
    """
    from scipy.ndimage import uniform_filter1d
    k = max(3, int(round(win_s / dt)))
    P = np.nan_to_num(P, nan=0.0)
    m = uniform_filter1d(P, k, axis=0, mode="nearest")
    sd = np.sqrt(np.maximum(uniform_filter1d(P ** 2, k, axis=0, mode="nearest") - m ** 2, 0))
    pos_ok = np.linalg.norm(sd, axis=1) * 1000.0 < pos_sd_mm
    Rf = R.reshape(len(R), 9)
    mr = uniform_filter1d(Rf, k, axis=0, mode="nearest")
    sdr = np.sqrt(np.maximum(uniform_filter1d(Rf ** 2, k, axis=0, mode="nearest") - mr ** 2, 0))
    ang_ok = np.degrees(np.linalg.norm(sdr, axis=1) / np.sqrt(2)) < ang_sd_deg
    return pos_ok & ang_ok


def move_segments(t, still, dt):
    """정지-이동-정지. (t_a, t_b, i_a, i_b, move_s) 목록."""
    segs = qc.segments_from_mask(t, still, protocol.ZUPT_MIN_STILL_S)
    out = []
    for k in range(len(segs) - 1):
        (a0, a1, ia0, ia1), (b0, b1, ib0, ib1) = segs[k], segs[k + 1]
        # 앵커는 정지 구간의 **끝쪽 25 % 안**에 잡는다. 한가운데로 잡으면 멈춤
        # 길이만큼 적분창이 늘어 오차가 부풀고, 경계로 잡으면 아직 정착 중인
        # 샘플이 물려 ZUPT 의 전제(끝 속도 0)가 깨진다.
        ia = int(ia1 - 0.25 * (ia1 - ia0))
        ib = int(ib0 + 0.25 * (ib1 - ib0))
        move_s = float(b0 - a1)
        if t[ib] - t[ia] < 0.2 or move_s > protocol.ZUPT_MAX_MOVE_S:
            continue
        out.append((float(t[ia]), float(t[ib]), int(ia), int(ib), move_s))
        _ = (a0, b1, dt)
    return out, segs


STAGES = ("A", "B", "C", "D")
STAGE_LABEL = {"A": "무보정", "B": "영점 바이어스", "C": "+ZUPT", "D": "+캘리브레이션"}


def _stage_accels(aE_raw, aE_cal, bias_E):
    """단계별 (선형가속, ZUPT 여부). 항상 넷을 나란히 낸다 — C 만 보면 어느
    보정이 실제로 일을 하는지 알 수 없고, 영점이 깨진 run 을 못 알아챈다."""
    g_nom = np.array([0.0, 0.0, qc.G0])
    bias = np.asarray(bias_E, float) if bias_E is not None else g_nom
    out = {"A": (aE_raw - g_nom, False), "B": (aE_raw - bias, False),
           "C": (aE_raw - bias, True)}
    if aE_cal is not None:
        out["D"] = (aE_cal - g_nom, True)      # 모형이 바이어스를 이미 품고 있다
    return out


def displacement_table(dt, aE_raw, aE_cal, bias_E, still, P_gt, segs):
    """구간마다 A/B/C/D 단계의 변위 오차."""
    rows = []
    for (ta, tb, ia, ib, move_s) in segs:
        n = ib - ia
        if n < 8:
            continue
        d_gt = P_gt[ib] - P_gt[ia]
        if not np.all(np.isfinite(d_gt)) or not np.all(np.isfinite(aE_raw[ia:ib + 1])):
            continue
        stages = _stage_accels(aE_raw[ia:ib + 1],
                               None if aE_cal is None else aE_cal[ia:ib + 1], bias_E)
        eps = {k: float(np.linalg.norm(zupt_integrate(lin, dt, z)[-1] - d_gt) * 1000.0)
               for k, (lin, z) in stages.items()}
        dist = float(np.linalg.norm(d_gt) * 1000.0)
        rows.append({"t_start": ta, "t_stop": tb, "seconds": float(tb - ta),
                     "move_seconds": move_s, "dist_mm": dist, "eps_mm": eps,
                     "eps_rel_pct": {k: (100.0 * v / dist if dist > 1.0 else np.nan)
                                     for k, v in eps.items()},
                     "still_frac": float(np.mean(still[ia:ib + 1]))})
    return rows


def error_floor(grid, dt, aE_raw, aE_cal, bias_E, still, windows=(0.5, 1.0, 2.0, 5.0)):
    """정지 구간에만 같은 파이프라인을 돌려 "0 이 나와야 하는데 얼마인가".

    로봇 GT 없이도 성능 상한을 알 수 있는 자리이고, imu_bench QC_PLAN §9 의
    표와 **같은 방식**이라 그 값과 직접 비교된다. 여기 값이 그쪽보다 크게 나쁘면
    센서가 아니라 이 리그의 진동·자기환경·영점을 의심할 자리다.
    """
    segs = qc.segments_from_mask(grid, still, min(windows))
    out = {}
    for W in windows:
        n = int(round(W / dt))
        vals = {k: [] for k in STAGES}
        for (_, _, i0, i1) in segs:
            for s0 in range(i0, i1 - n, max(1, n // 2)):
                a = aE_raw[s0:s0 + n + 1]
                if a.shape[0] < n or not np.all(np.isfinite(a)):
                    continue
                stages = _stage_accels(a, None if aE_cal is None else aE_cal[s0:s0 + n + 1],
                                       bias_E)
                for k, (lin, z) in stages.items():
                    vals[k].append(np.linalg.norm(zupt_integrate(lin, dt, z)[-1]) * 1000.0)
        if vals["C"]:
            out[W] = {k: float(np.median(v)) for k, v in vals.items() if v}
            out[W]["n"] = len(vals["C"])
    return out


# =================================================================== 본체
# 6 축 퓨전은 heading 이 **원리적으로 관측되지 않는다.** 그래서 블록마다 요 원점이
# 다르고, align 블록에서 푼 고정회전을 probe 블록에 그대로 쓰면 그 사이의 표류가
# 통째로 '오차' 로 잡혀 49 도 같은 숫자가 나온다 (실제로 그렇게 나왔다). 이 소스는
# **분석하는 블록 자신의 정지 자세로** 정렬한다. 그러면 재는 것이 '블록 안에서
# 얼마나 흘렀나' 가 되어, 물어볼 수 있는 유일한 질문에 답하게 된다.
# 영점 파일에서 지구프레임 가속도 바이어스를 담고 있는 키. **두 갈래 다** 본다 —
# 실기 로거는 host/zero_ref.py 의 ZeroReference.to_dict() 를 그대로 쓰고(accel_bias_earth),
# 예전 시뮬레이터는 b_E 로 썼다. test_zero_bias_key_contract 가 이 목록이 실기
# 로거의 출력과 맞물려 있는지 지킨다.
ZERO_BIAS_KEYS = ("accel_bias_earth", "b_E")

PER_BLOCK_ALIGN = ("host6",)
REFUSE_SETTLE_S = 3.0       # 오프라인 재퓨전의 수렴 과도구간. 드리프트와 구분 안 된다


def analyse_rotation(gt, imu, src, A, B, block, settle_s=0.0):
    """지연 -> 지연 보정 후 각오차 -> 드리프트. 채널별로, 그리고 전체 geodesic 으로."""
    sn = add_channels(imu, src, A, B)
    dt = 1.0 / RESAMPLE_HZ
    lo = max(gt["t"][0], imu["t"][0]) + float(settle_s)
    hi = min(gt["t"][-1], imu["t"][-1])
    if hi - lo < 5.0:
        return None
    grid = np.arange(lo, hi, dt)

    out = {"block": block, "source": src, "span_s": float(hi - lo),
           "settle_s": float(settle_s), "channels": {}}
    lags = {}
    for ch in ANGLE_CHANNELS:
        g = resample(gt["t"], gt[ch], grid)
        s = resample(imu["t"], sn[ch], grid)
        if ch in ("spin", "tip_x", "tip_y"):
            # 기준선이 무너지는 자세의 구간을 뺀다 (spin: 축이 base +-x 를 겨눌 때,
            # tip_*: 축이 아래를 안 볼 때). 안 빼면 감긴 값이 오차로 둔갑한다.
            key = "spin_valid" if ch == "spin" else "tip_valid"
            bad = resample(gt["t"], (~gt[key]).astype(float), grid) > 0.5
            g = np.where(bad, np.nan, g)
            s = np.where(bad, np.nan, s)
        lag, corr = xcorr_lag(dt, g, s)
        lags[ch] = lag
        e_raw = s - g
        e_cor = shift_apply(s, dt, lag) - g if np.isfinite(lag) else np.full_like(g, np.nan)
        out["channels"][ch] = {
            "latency_ms": float(lag * 1000.0) if np.isfinite(lag) else float("nan"),
            "corr": corr,
            "amp_gt_deg": float(np.nanstd(g) * np.sqrt(2)),
            "err_rms_deg": float(np.sqrt(np.nanmean(e_raw ** 2))),
            "err_p95_deg": float(np.nanpercentile(np.abs(e_raw), 95)),
            "err_rms_delay_corrected_deg": float(np.sqrt(np.nanmean(e_cor ** 2))),
            "err_p95_delay_corrected_deg": float(np.nanpercentile(np.abs(e_cor), 95)),
            "drift_deg_per_min": _drift(grid, e_cor),
            "spectrum": motion_spectrum(dt, g),
            "valid_frac": float(np.mean(np.isfinite(e_raw))),
        }

    # 채널이 아니라 회전 그 자체의 오차. 채널은 축이 겹치므로 합산하면 이중계산이다.
    Rg = np.stack([[resample(gt["t"], gt["R"][:, i, j], grid) for j in range(3)]
                   for i in range(3)], 0)
    lag_r = float(np.nanmedian([lags[c] for c in ANGLE_CHANNELS]))
    Rs = np.stack([[resample(imu["t"], sn["R"][:, i, j], grid) for j in range(3)]
                   for i in range(3)], 0)
    ok = np.all(np.isfinite(Rg.reshape(9, -1)), 0) & np.all(np.isfinite(Rs.reshape(9, -1)), 0)
    Rg_n = np.transpose(Rg[:, :, ok], (2, 0, 1))
    Rs_n = np.transpose(Rs[:, :, ok], (2, 0, 1))
    Rg_n = np.array([qc.project_SO3(m) for m in Rg_n])
    Rs_n = np.array([qc.project_SO3(m) for m in Rs_n])
    geo = qc.geodesic_deg(Rs_n, Rg_n)
    # 지연 보정판: 센서 자세를 lag 만큼 당겨서 다시 잰다
    k = int(round(lag_r / dt)) if np.isfinite(lag_r) else 0
    if 0 < k < len(geo) - 10:
        geo_c = qc.geodesic_deg(Rs_n[k:], Rg_n[:len(Rg_n) - k])
    elif 10 - len(geo) < k < 0:
        # 센서가 앞선 경우. 반대로 GT 를 당겨서 겹친다. 여기를 빼먹으면 음수
        # 지연일 때만 조용히 '보정 전' 값이 보정 후 칸에 앉는다.
        geo_c = qc.geodesic_deg(Rs_n[:len(Rs_n) + k], Rg_n[-k:])
    else:
        geo_c = geo
    axis = qc.so3_log_deg(Rs_n, Rg_n)
    out["geodesic"] = {
        "lag_ms": float(lag_r * 1000.0) if np.isfinite(lag_r) else float("nan"),
        "rms_deg": float(np.sqrt(np.mean(geo ** 2))),
        "p95_deg": float(np.percentile(geo, 95)),
        "max_deg": float(geo.max()),
        "rms_delay_corrected_deg": float(np.sqrt(np.mean(geo_c ** 2))),
        "p95_delay_corrected_deg": float(np.percentile(geo_c, 95)),
        "axis_rms_deg": np.sqrt((axis ** 2).mean(axis=0)).tolist(),
        "n": int(ok.sum()),
    }
    out["_grid"] = grid
    out["_sn"] = sn
    out["_lags"] = lags
    return out


def _drift(t, e):
    """오차의 선형 추세 [deg/min]. 자기환경이 나쁘면 azimuth 에서 먼저 보인다."""
    ok = np.isfinite(e)
    if ok.sum() < 100:
        return float("nan")
    a = np.polyfit(t[ok] - t[ok][0], e[ok], 1)[0]
    return float(a * 60.0)


def analyse_lockin(block, gt, imu, src, A, B, program):
    """sweep 블록이 있을 때만. 주파수별 지연·이득 -> L/T 분리."""
    blk = next((b for b in program.get("blocks", []) if b.get("block") == block), None)
    if blk is None or not blk.get("segments"):
        return None
    sn = add_channels(imu, src, A, B)
    # sweep_tilt 는 base +y 축 둘레로 흔든다 -> 침투축이 base x 쪽으로 기울므로
    # tip_x 에 나타난다 (protocol.plan_sweep 의 축 정의와 짝을 이룬다).
    ch = {"sweep_tilt": "tip_x", "sweep_roll": "spin", "sweep_trans": None}[block]
    if ch is None:
        return None
    rows = []
    for seg in blk["segments"]:
        f = seg["freq_hz"]
        t0, t1 = seg["t_start"], seg["t_stop"]
        lo, hi = t0 + 1.5 / f, t1 - 1.5 / f          # 램프 구간을 뺀다
        if hi - lo < 2.0 / f:
            continue
        mg = (gt["t"] >= lo) & (gt["t"] <= hi)
        ms = (imu["t"] >= lo) & (imu["t"] <= hi)
        ag, pg, rg = lockin(gt["t"][mg], gt[ch][mg], f, t0)
        as_, ps, rs = lockin(imu["t"][ms], sn[ch][ms], f, t0)
        dph = float(wrap_pi(ps - pg))
        rows.append({"freq_hz": f, "amp_gt": ag, "amp_sn": as_,
                     "gain": (as_ / ag if ag > 0 else np.nan),
                     "dphase_rad": dph,
                     "latency_ms": float(-dph / (2 * np.pi * f) * 1000.0),
                     "resid_gt": rg, "resid_sn": rs,
                     "cap": seg.get("cap")})
    if not rows:
        return None
    wt = np.array([1.0 / max(r["resid_sn"], 0.05) for r in rows], float)
    fit = fit_LT([r["freq_hz"] for r in rows], [r["dphase_rad"] for r in rows],
                 [r["gain"] for r in rows], wt)
    return {"channel": ch, "segments": rows, "fit": fit}


def analyse_translation(gt, imu, src, A, B, block, r_cad=None, lag_s=0.0, model=None):
    """구간별 변위 오차 (A/B/C/D) + 오차 바닥. 모형(C,d,r)은 밖에서 받는다.

    모형을 여기서 풀지 않는 이유: **적합과 검증을 나누기 위해서다** (QC_PLAN §6).
    같은 블록으로 풀고 같은 블록으로 평가한 숫자는 아무것도 말하지 않는다.
    analyse() 가 probe 로 풀어서 모든 블록에 같은 모형을 넣는다.
    """
    dt = 1.0 / RESAMPLE_HZ
    lo = max(gt["t"][0], imu["t"][0]); hi = min(gt["t"][-1], imu["t"][-1])
    if hi - lo < 5.0:
        return None
    grid = np.arange(lo, hi, dt)

    # 센서 시간축을 **지연만큼 앞으로 당겨서** 격자에 얹는다. 지연을 안 빼면
    # 모형 동정이 무너진다 — R 의 2 계 미분을 쓰므로 고주파가 지배하고, 25 ms
    # 어긋난 고주파는 서로 아무 관계가 없는 신호가 된다.
    t_i = imu["t"] - float(lag_s)
    a_grid = np.column_stack([resample(t_i, imu["acc"][:, k], grid) for k in range(3)])
    gyr_g = np.column_stack([resample(t_i, imu["gyr"][:, k], grid) for k in range(3)])
    R_world = np.transpose(np.stack(
        [[resample(t_i, imu["R"][src][:, i, j], grid) for j in range(3)]
         for i in range(3)], 0), (2, 0, 1))
    # base <- sensor.  B 는 안 곱한다 — 가속도는 **센서 프레임** 값이고 B 는
    # imu <- tool 이라 여기에 낄 자리가 없다. (자세 채널 쪽은 A R B 로 tool
    # 자세를 만들지만, 이쪽은 센서 축 그대로 돌려야 한다.)
    R_bs = np.einsum("ij,njk->nik", A, R_world)
    _ = B

    P_grid = np.column_stack([resample(gt["t"], gt["p"][:, k], grid) for k in range(3)])
    Rg_grid = np.transpose(np.stack(
        [[resample(gt["t"], gt["R"][:, i, j], grid) for j in range(3)]
         for i in range(3)], 0), (2, 0, 1))

    aE_raw, conv = earth_accel(R_bs, a_grid)
    aE_cal = None
    if model:
        aE_cal, _ = earth_accel(R_bs, apply_model(a_grid, model))

    r_used = (np.asarray(r_cad, float) if r_cad is not None
              else (np.asarray(model["r_m"], float) if model else None))
    P_gt = gt_point_path(gt, grid, r_used)

    # 정지 판정은 IMU 와 GT 를 **둘 다** 만족해야 한다.
    #   IMU 만 쓰면: 아주 느린 이동의 양 끝(최소저크는 시작·끝에서 가속도도
    #     각속도도 0 이다)을 정지로 오인해 두 정지 구간이 이어져 버린다.
    #   GT 만 쓰면: 센서가 실제로 무엇을 보고 있었는지와 무관해진다.
    # 실행 중(zero_ref)에는 IMU 만 쓸 수 있지만, 분석에는 GT 가 있으니 쓴다.
    still = qc.still_mask(grid, gyr_g, a_grid) & gt_still_mask(P_grid, Rg_grid, dt)
    segs, holds = move_segments(grid, still, dt)

    zero = qc.load_json(os.path.join(qc.DIR_META, f"zero_{block}.json"))
    # zero_ref.measure 는 **IMU 자신의 world 프레임**에서 b_E 를 잰다 (칩 쿼터니언만
    # 쓴다). 여기 가속도는 base 프레임이므로 A 를 곱해 옮겨야 한다. 안 곱하면
    # 바이어스가 A 만큼 돌아간 채 빠져서, B 단계가 A 단계보다 나아지지 않는다.
    #
    # 키 이름이 두 갈래라 **둘 다** 본다. 실기 로거는 zero_ref.to_dict() 의
    # ``accel_bias_earth`` 로 쓰고 시뮬레이터는 ``b_E`` 로 썼다. 한쪽만 보면
    # 시뮬에서는 멀쩡한데 실기에서만 조용히 영점을 잃는다 — 자체 시험 19 개가
    # 전부 통과하는 채로 실기의 B 단계가 블록 내 평균으로 대체됐고, probe_cal
    # 에서는 그 평균이 나빠 B 가 A 보다 못한 표가 나왔다.
    b_E = None
    if zero:
        for key in ZERO_BIAS_KEYS:
            if key in zero:
                b_E = np.asarray(zero[key], float)
                break
    bias_E = (A @ b_E) if b_E is not None else None
    zero_from = "meta" if b_E is not None else "none"
    if bias_E is None and still.sum() > 200:
        bias_E = np.nanmean(aE_raw[still], axis=0)
        zero_from = "this_block"

    rows = displacement_table(dt, aE_raw, aE_cal, bias_E, still, P_gt, segs)
    floor = error_floor(grid, dt, aE_raw, aE_cal, bias_E, still)

    seg_data = []
    for (ta, tb, ia, ib, move_s) in segs:
        _ = (ta, tb, move_s)
        d_gt = P_gt[ib] - P_gt[ia]
        if np.all(np.isfinite(d_gt)) and np.all(np.isfinite(a_grid[ia:ib + 1])):
            seg_data.append({"R": R_bs[ia:ib + 1], "a": a_grid[ia:ib + 1],
                             "dt": dt, "d_gt": d_gt})

    out = {"block": block, "source": src, "convention": conv,
           "lag_used_ms": 1000.0 * lag_s,
           "lever_arm_used_mm": ((np.asarray(r_used) * 1000.0).tolist()
                                 if r_used is not None else None),
           "lever_arm_source": ("cad" if r_cad is not None
                                else ("model" if model else "none")),
           "model_applied": bool(model),
           "zero_from": zero_from if bias_E is not None else "none",
           "bias_E_ms2": bias_E.tolist() if bias_E is not None else None,
           "gravity_ms2": float(np.linalg.norm(bias_E)) if bias_E is not None else float("nan"),
           "n_holds": len(holds), "n_segments": len(rows),
           "still_frac": float(still.mean()),
           "segments": rows, "error_floor_mm": {str(k): v for k, v in floor.items()},
           "_ident": {"P": P_grid, "Rg": np.transpose(Rg_grid, (1, 2, 0)),
                      "R_bs": R_bs, "a": a_grid, "dt": dt},
           "_seg_data": seg_data}
    if rows:
        for stage in STAGES:
            e = [r["eps_mm"][stage] for r in rows if stage in r["eps_mm"]]
            if e:
                out[f"eps_{stage}"] = {"median_mm": float(np.median(e)),
                                       "p95_mm": float(np.percentile(e, 95)),
                                       "max_mm": float(np.max(e))}
        # 눈금을 0.5 s 부터 끊는다. 프로빙의 짧은 재배치(0.5~1.5 s)가 이 QC 의
        # 사용가능 영역인데, 1 s 부터 끊으면 그 영역이 한 칸에 뭉개진다.
        out["by_duration"] = _bucket(rows, "seconds", (0.5, 1.0, 2.0, 4.0, 8.0))
        out["by_move_duration"] = _bucket(rows, "move_seconds", (0.5, 1.0, 2.0, 4.0, 8.0))
        out["by_distance"] = _bucket(rows, "dist_mm", (25.0, 50.0, 100.0, 200.0))
    return out


def _nanmed(v):
    """전부 nan 이면 nan. np.nanmedian 은 그때 경고를 뱉는다 — 상대오차는 이동
    거리가 0 인 구간(순수 회전)에서 정의되지 않으므로 정상적으로 생기는 일이다."""
    v = np.asarray(v, float)
    return float(np.median(v[np.isfinite(v)])) if np.any(np.isfinite(v)) else float("nan")


def _bucket(rows, key, edges):
    """구간을 길이/거리로 묶은 표. '몇 초를 움직이면 몇 mm 틀리는가' 의 축이다."""
    out = []
    lo = 0.0
    for hi in list(edges) + [float("inf")]:
        sel = [r for r in rows if lo <= r[key] < hi]
        if sel:
            out.append({"lo": lo, "hi": hi, "n": len(sel),
                        "dist_med_mm": float(np.median([r["dist_mm"] for r in sel])),
                        **{f"eps_{k}_med_mm": float(np.median([r["eps_mm"][k] for r in sel]))
                           for k in STAGES if k in sel[0]["eps_mm"]},
                        "eps_C_p95_mm": float(np.percentile([r["eps_mm"]["C"] for r in sel], 95)),
                        "eps_rel_C_med_pct": _nanmed([r["eps_rel_pct"]["C"] for r in sel]),
                        "eps_rel_D_med_pct": (_nanmed([r["eps_rel_pct"]["D"] for r in sel])
                                              if "D" in sel[0]["eps_mm"] else float("nan"))})
        lo = hi
    return out


# ------------------------------------------------------------------ 판정
def verdict(res):
    """통과 기준(reference.PASS) 대조. 모자란 것은 '미측정' 으로 남긴다 —
    측정 못 한 것을 통과로 적으면 그 표는 거짓말이 된다."""
    P = ref.PASS
    v = []

    def add(name, value, limit, cmp="<", unit=""):
        if value is None or not np.isfinite(value):
            v.append({"name": name, "value": None, "limit": limit, "unit": unit,
                      "pass": None, "note": "미측정"})
            return
        ok = value < limit if cmp == "<" else value > limit
        v.append({"name": name, "value": float(value), "limit": limit, "unit": unit,
                  "pass": bool(ok)})

    al = res.get("alignment", {})
    add("정렬 잔차 RMS", al.get("resid_rms_deg"), P["rot_static_resid_rms_deg"], "<", "deg")
    rot = res.get("rotation", {}).get(res.get("primary_source", "chip"), {})
    geo = rot.get("geodesic", {})
    add("동적 각오차 RMS (지연보정)", geo.get("rms_delay_corrected_deg"),
        P["rot_dynamic_rms_deg"], "<", "deg")
    # 이제 부호가 있다. 기준은 **크기**에 건다 — 어느 쪽으로 벌어지든 두 스트림이
    # 그만큼 어긋나 있다는 뜻이고, 부호는 어느 쪽이 앞서는지만 말한다. 부호 그대로
    # 비교하면 음수가 자동으로 '통과' 가 되어 거짓 통과가 다시 생긴다.
    lag_ms = geo.get("lag_ms")
    add("회전 실효 지연 |값|",
        abs(lag_ms) if lag_ms is not None and np.isfinite(lag_ms) else None,
        P["rot_latency_ms"], "<", "ms")
    corr = [rot.get("channels", {}).get(c, {}).get("corr") for c in ANGLE_CHANNELS]
    corr = [c for c in corr if c is not None and np.isfinite(c)]
    add("상호상관 최소", min(corr) if corr else None, P["rot_corr_min"], ">", "")
    # 프로브가 거의 연직이면 **침투축 둘레 회전(spin)이 곧 heading** 이다.
    # 자력계가 로봇 근처에서 흔들리면 여기가 가장 먼저, 가장 크게 틀어진다.
    add("heading(spin) 드리프트", abs(rot.get("channels", {}).get("spin", {})
                                    .get("drift_deg_per_min", np.nan)),
        P["yaw_drift_deg_per_min"], "<", "deg/min")

    # --- 병진 --------------------------------------------------------
    # 판정은 **정지 구간 오차 바닥**에만 건다. 같은 칩·같은 파이프라인의 실측
    # (imu_bench §9) 이 있으므로 '이 리그가 성한가' 를 물을 수 있는 유일한 칸이다.
    # 이동 구간은 판정하지 않는다 — reference.REPORT_ONLY 참조.
    tr = res.get("translation", {}).get(res.get("primary_block", "probe"), {})
    floor = tr.get("error_floor_mm", {}) or {}
    ratios = []
    for W, rowref in ref.DISPLACEMENT_FLOOR_MM.items():
        got = (floor.get(str(W)) or floor.get(W) or {}).get("C")
        if got is not None and np.isfinite(got) and rowref.get("C"):
            ratios.append(got / rowref["C"])
    add("변위 오차 바닥 / §9 실측 (최대 배수)",
        max(ratios) if ratios else None, P["disp_floor_ratio"], "<", "배")

    # 아래 둘은 **참고**다. 값은 싣되 통과/불합격을 찍지 않는다.
    def report(name, value, unit, note):
        v.append({"name": name, "unit": unit, "limit": None, "pass": None,
                  "note": note,
                  "value": (float(value) if value is not None and np.isfinite(value)
                            else None)})

    for W in (0.5, 1.0, 2.0):
        b = next((r for r in tr.get("by_duration", []) if r["lo"] <= W < r["hi"]), None)
        report(f"[참고] 변위 오차 {W:g} s 창 (C)", b["eps_C_med_mm"] if b else None,
               "mm", "판정 없음 (용도 미확정)" if b else "미측정")
    # 상대오차는 **검증 블록의 D 단계**로 본다 — 모형을 푼 블록에서 재면
    # 자기평가가 되기 때문이다.
    vb = res.get("validation_block")
    vtr = res.get("translation", {}).get(vb, {}) if vb else {}
    far = [r for r in vtr.get("segments", []) if r["dist_mm"] >= 80.0]
    stage = "D" if far and "D" in far[0]["eps_rel_pct"] else "C"
    report(f"[참고] 100 mm 이동 상대오차 ({stage}, 검증)",
           float(np.nanmedian([r["eps_rel_pct"][stage] for r in far])) if far else None,
           "%", "판정 없음 (용도 미확정)" if far else "미측정")
    return v


# ------------------------------------------------------------------ 출력
def fmt(x, n=1, w=8):
    return f"{'—':>{w}}" if x is None or not np.isfinite(x) else f"{x:>{w}.{n}f}"


def print_report(res):
    R = ref
    print("\n" + "=" * 78)
    print("IMU 추적 QC — FR5 teleop 초음파 프로빙")
    print("=" * 78)

    print("\n1/6  시간축 감사   (틀어진 시간축으로 낸 오차는 오차가 아니다)")
    print(f"  {'블록':<12}{'GT n':>8}{'GT Hz':>8}{'IMU n':>8}{'IMU Hz':>8}"
          f"{'skew ppm':>10}{'잔차 p95':>10}{'짝비율':>8}{'겹침 s':>9}")
    for b, d in res["timebase"].items():
        print(f"  {b:<12}{d['gt_n']:>8d}{d['gt_hz']:>8.1f}{d['imu_n']:>8d}{d['imu_hz']:>8.1f}"
              f"{fmt(d.get('skew_ppm'), 0, 10)}{fmt(d.get('resid_p95_ms'), 2, 10)}"
              f"{100 * d['paired_frac']:>7.0f}%{fmt(d.get('overlap_s'), 1, 9)}")
    print(f"  기준: |skew| < {R.TIMEBASE['skew_ppm_abs_max']:.0f} ppm, "
          f"잔차 p95 {R.TIMEBASE['resid_p95_ms']:.2f} ms (Exp-Latency §3.2)")
    for b, d in res["timebase"].items():
        ov = d.get("overlap_s", float("nan"))
        if np.isfinite(ov) and ov < 5.0:
            print(f"  [경고] {b}: GT 와 IMU 가 겹치는 구간이 {ov:.1f} s 다 —"
                  " 이 블록은 분석에서 **통째로 빠진다.**")
            print("         한쪽 로거만 다시 딴 채 옛 파일이 남았을 때 이렇게 된다"
                  " (log_robot 은 pose 를 못 받으면 저장을 건너뛰고도 0 으로 끝난다).")
            print(f"         그 블록을 다시 딸 것:  python3 run_session.py --blocks {b}")
    for b, d in res["timebase"].items():
        if d["gt_hz"] < 60:
            print(f"  [주의] {b}: GT {d['gt_hz']:.0f} Hz. 한 샘플이 "
                  f"{1000 / d['gt_hz']:.0f} ms 라 지연을 그보다 잘게 못 읽는다 —"
                  " rates.status_publish_hz 를 올려서 다시 딸 것")

    al = res.get("alignment")
    if al:
        print("\n2/6  IMU world <-> 로봇 base 고정회전   (정지 자세에서만 푼다)")
        print(f"  {'지표':<22}{'값':>10}   기준 (Exp-Latency §3.3)")
        print(f"  {'정지 자세 수':<22}{al['n']:>10d}   {R.ALIGNMENT['n_poses']} "
              f"(최소 {R.ALIGNMENT['n_poses_min']})")
        print(f"  {'잔차 RMS [deg]':<22}{al['resid_rms_deg']:>10.2f}   "
              f"{R.ALIGNMENT['resid_rms_deg']:.2f}  (< {R.ALIGNMENT['resid_rms_pass_deg']:.0f} 이면 정상)")
        print(f"  {'잔차 최대 [deg]':<22}{al['resid_max_deg']:>10.2f}   "
              f"{R.ALIGNMENT['resid_max_deg']:.2f}")
        print(f"  {'관측도':<22}{al['observability']:>10.3f}   "
              f"{R.ALIGNMENT['observability']:.3f}")
        if al["observability"] < 0.5 * R.ALIGNMENT["observability"]:
            print("  [주의] 관측도가 낮다 — align 자세가 한 축으로만 기울었다."
                  " 두 축을 동시에 기울인 자세를 넣어 다시 딸 것")

    print("\n3/6  회전   (지연 -> 지연보정 후 각오차 -> 드리프트)")
    for src, rot in res["rotation"].items():
        if rot is None:
            continue
        tag = {"chip": "칩 RV (9축)", "host9": "호스트 (9축)", "host6": "호스트 (6축, 자력계 없음)"}[src]
        g = rot["geodesic"]
        print(f"\n  [{tag}]  블록 {rot['block']}  {rot['span_s']:.0f} s")
        print(f"  {'채널':<10}{'지연[ms]':>10}{'상관':>8}{'진폭[deg]':>11}"
              f"{'오차RMS':>9}{'보정후':>9}{'드리프트':>11}   기준(Exp-Latency)")
        for ch in ANGLE_CHANNELS:
            c = rot["channels"][ch]
            rr = R.latency_row(ch)
            note = (f"{rr['eff_ms']} ms / corr {rr['corr']}  [{REF_CHANNEL[ch]}]"
                    if rr else "기준 대응 없음 (기울기 성분)")
            print(f"  {ch:<10}{fmt(c['latency_ms'], 1, 10)}{fmt(c['corr'], 3, 8)}"
                  f"{fmt(c['amp_gt_deg'], 1, 11)}{fmt(c['err_rms_deg'], 2, 9)}"
                  f"{fmt(c['err_rms_delay_corrected_deg'], 2, 9)}"
                  f"{fmt(c['drift_deg_per_min'], 2, 11)}   {note}")
        print(f"  {'회전 전체':<10}{fmt(g['lag_ms'], 1, 10)}{'':>8}{'':>11}"
              f"{fmt(g['rms_deg'], 2, 9)}{fmt(g['rms_delay_corrected_deg'], 2, 9)}")
        sp = rot["channels"]["tip_x"].get("spectrum", {})
        if sp:
            print(f"  이 블록의 운동 스펙트럼: 중앙 {sp['f_median_hz']:.2f} Hz, "
                  f"90 % 아래 {sp['f_p90_hz']:.2f} Hz "
                  "— 실효 지연은 이 스펙트럼에 딸린 값이다")

    lk = res.get("lockin") or {}
    if lk:
        print("\n3b/6 록인 (스크립트 자극)   지연이 평평하면 순수지연, 우하향이면 1차 지체")
        for block, d in lk.items():
            if d is None:
                continue
            print(f"\n  [{block}] 채널 {d['channel']}")
            print(f"  {'f [Hz]':>8}{'GT 진폭':>10}{'센서 진폭':>11}{'이득':>8}"
                  f"{'지연[ms]':>10}{'잔차비':>8}   기준 지연/이득")
            key = {"tip_x": "depression", "tip_y": "depression",
                   "tilt": "depression", "spin": "roll"}.get(d["channel"])
            base = R.LOCKIN_MS.get(key, {})
            gain = R.LOCKIN_GAIN.get(key, {})
            for r in d["segments"]:
                b = base.get(r["freq_hz"])
                gg = gain.get(r["freq_hz"])
                note = f"{b} ms / {gg}" if b else ""
                # 잔차비 > 0.5 = 그 주파수에서 신호가 잡음에 묻혔다. 그 줄의
                # 지연은 아무 뜻이 없으므로 별표로 못 박는다.
                mark = "*" if not np.isfinite(r["resid_sn"]) or r["resid_sn"] > 0.5 else " "
                print(f" {mark}{r['freq_hz']:>8.2f}{fmt(r['amp_gt'], 2, 10)}{fmt(r['amp_sn'], 2, 11)}"
                      f"{fmt(r['gain'], 3, 8)}{fmt(r['latency_ms'], 0, 10)}"
                      f"{fmt(r['resid_sn'], 2, 8)}   {note}")
            f = d["fit"]
            if f.get("ok"):
                print(f"    -> L = {f['L'] * 1000:.0f} ms (순수지연), T = {f['T'] * 1000:.0f} ms"
                      f" (1차 지체), L+T = {f['dc_group_delay'] * 1000:.0f} ms")
                print("       IMU 는 T ~ 0 이어야 정상이다 (대역폭 손실 없음)."
                      "  기준 depression L 14 / T 13, roll L 1 / T 23 ms")

    print("\n4/6  병진   (IMU 는 병진을 직접 재지 못한다 — ZUPT 구간별로 본다)")
    mdl = res.get("model")
    if mdl:
        print(f"\n  [가속도 모형]  a_corr = C a + d, 지렛대 r 을 함께 동정"
              f"  (적합 블록 {mdl['fit_block']})")
        print(f"    스케일 [%]        {['%+.2f' % v for v in mdl['scale_pct']]}"
              f"     기준: 보정 전 {R.SCALE_ERROR_PCT['uncalibrated']:+.2f} %,"
              f" 보정 후 {R.SCALE_ERROR_PCT['calibrated']:+.2f} %")
        print(f"    비직교 [deg]      {mdl['nonorthogonality_deg']:+.3f}")
        print(f"    바이어스 [m/s^2]  {np.round(mdl['bias_ms2'], 4).tolist()}")
        print(f"    지렛대 r [mm]     {np.round(mdl['lever_mm'], 1).tolist()}"
              f"  (|r| {mdl['lever_norm_mm']:.1f} mm)")
        print(f"    가속도 잔차       {mdl['accel_resid_before_ms2']:.4f}"
              f" -> {mdl['accel_resid_after_ms2']:.4f} m/s^2   (n={mdl['n']})")
        print("    r 은 회전이 있어야, C 는 여러 자세가 있어야 관측된다. 잔차가 안 줄면"
              " 그 조건이 안 채워진 것이다.")
    for block, tr in res["translation"].items():
        if tr is None:
            continue
        tag = ("적합 블록 — D 는 자기평가" if tr.get("model_is_fit_block")
               else ("검증 블록" if block == res.get("validation_block") else ""))
        print(f"\n  [{block}]  정지 {tr['n_holds']} 개 / 이동 구간 {tr['n_segments']} 개"
              f"  (정지 비율 {100 * tr['still_frac']:.0f} %)"
              + (f"   << {tag}" if tag else ""))
        if tr.get("lever_arm_used_mm"):
            print(f"  지렛대 {np.round(tr['lever_arm_used_mm'], 1).tolist()} mm"
                  f" ({tr['lever_arm_source']}) · 지연 보정 {tr['lag_used_ms']:.0f} ms")
        if np.isfinite(tr.get("gravity_ms2", np.nan)):
            err = 100.0 * (tr["gravity_ms2"] / qc.G0 - 1.0)
            print(f"  영점 중력 {tr['gravity_ms2']:.4f} m/s^2 (스케일 오차 {err:+.2f} %)"
                  f"  [{tr['zero_from']}]")
        if tr.get("by_duration"):
            print(f"  {'창 [s]':<12}{'n':>4}{'거리':>9}{'A 무보정':>11}{'B 영점':>9}"
                  f"{'C +ZUPT':>10}{'D 캘리브':>10}{'C p95':>9}{'D 상대':>9}")
            for b in tr["by_duration"]:
                hi = "inf" if not np.isfinite(b["hi"]) else f"{b['hi']:.0f}"
                print(f"  {b['lo']:.0f}~{hi:<8}{b['n']:>4d}{b['dist_med_mm']:>8.0f}mm"
                      f"{b['eps_A_med_mm']:>11.1f}{b['eps_B_med_mm']:>9.2f}"
                      f"{b['eps_C_med_mm']:>10.2f}{fmt(b.get('eps_D_med_mm'), 2, 10)}"
                      f"{b['eps_C_p95_mm']:>9.2f}{fmt(b.get('eps_rel_D_med_pct'), 1, 8)}%")
        fl = tr.get("error_floor_mm") or {}
        if fl:
            print("\n  오차 바닥 (정지 구간에 같은 파이프라인)   기준: imu_bench QC_PLAN §9")
            print(f"  {'창 [s]':>8}{'A':>10}{'B':>9}{'C':>9}{'D':>9}    기준 A / B / C")
            for key in sorted(fl, key=float):
                d = fl[key]
                b = R.DISPLACEMENT_FLOOR_MM.get(float(key))
                note = f"{b['A']:.0f} / {b['B']:.2f} / {b['C']:.2f}" if b else ""
                print(f"  {float(key):>8.1f}{d['A']:>10.1f}{d['B']:>9.2f}{d['C']:>9.2f}"
                      f"{fmt(d.get('D'), 2, 9)}    {note}")

    print("\n5/6  캘리브레이션 검증   적합과 검증을 나누지 않은 숫자는 아무것도 말하지 않는다")
    vb = res.get("validation_block")
    if mdl and vb:
        for blk in (mdl["fit_block"].split("+")[-1], vb):
            tr = res["translation"].get(blk)
            if not tr:
                continue
            label = "적합(자기평가)" if blk in mdl["fit_block"] else "검증"
            rows = [r for r in tr.get("segments", []) if "D" in r["eps_mm"]]
            if not rows:
                continue
            c = np.array([r["eps_mm"]["C"] for r in rows])
            d = np.array([r["eps_mm"]["D"] for r in rows])
            print(f"  {label:<12} {blk:<10} n={len(rows):>3d}"
                  f"   C {np.median(c):>7.2f} mm  ->  D {np.median(d):>7.2f} mm"
                  f"   (p95 {np.percentile(d, 95):.2f})")
        ft = res["translation"].get(mdl["fit_block"].split("+")[-1], {})
        vt = res["translation"].get(vb, {})
        fe = [r["eps_mm"]["D"] for r in ft.get("segments", []) if "D" in r["eps_mm"]]
        ve = [r["eps_mm"]["D"] for r in vt.get("segments", []) if "D" in r["eps_mm"]]
        if fe and ve and np.median(ve) > 2.0 * np.median(fe):
            print("  [주의] 검증이 적합보다 두 배 넘게 나쁘다 — 과적합이다."
                  " 자세 다양성을 늘리거나 lam 을 키울 것")
    else:
        print("  모형 또는 검증 블록이 없어 하지 못했다.")

    print("\n6/6  판정")
    print(f"  {'항목':<26}{'값':>10}{'기준':>10}   결과")
    for v in res["verdict"]:
        lim = "—" if v["limit"] is None else f"{v['limit']:.2f}"
        if v["pass"] is None:
            mark = v.get("note") or "미측정"
        else:
            mark = "통과" if v["pass"] else "**불합격**"
        print(f"  {v['name']:<34}{fmt(v['value'], 2, 10)}{lim:>10}   {mark}")
    print("\n  이동 구간 변위는 판정하지 않는다 — 시술 중 실제 이동량을 확언할 수"
          " 없어 어떤 문턱도 임의값이 된다.")
    print("  대신 §4 의 '창 [s]' 표가 곧 답이다: 구간이 짧을수록 급격히 정확해진다"
          " (오차 ~ a·T^2/8).")
    print("\n  게이팅:")
    for b, g in res["gating"].items():
        mg = g.get("magnetic") or {}
        mtxt = ""
        if mg.get("raw"):
            mtxt = (f"  복각산포 {mg['raw']['dip_sd_deg']:.1f}°"
                    f"(원시)/{(mg.get('cal') or {}).get('dip_sd_deg', float('nan')):.1f}°(보정)")
        print(f"    {b:<12} CRC {g['crc_errors']:>3d}  누락 {g['seq_gaps']:>3d}"
              f"  영점정지 {'예' if g['zero_still'] else '아니오'}"
              f"  후퇴 {g['retreat_events']:>2d}  gimbal {100 * g['gimbal_frac']:>3.0f} %{mtxt}")

    # 자기 교란 — 센서에 고정된 왜곡을 **전부** 담은 모형으로도 설명이 되는가
    mm_ = res.get("mag_model") or {}
    if mm_:
        print("\n  자기 모형  m = K R h + b   (K 가 소프트아이언·축오차·스케일·핸드아이"
              "회전을 전부 흡수한다)")
        print(f"    {'블록':<12}{'자세':>4}{'잔차':>9}{'잡음':>8}{'배수':>7}"
              f"{'방향오차':>9}{'장':>8}{'하드아이언':>10}{'자세다양성':>11}")
        for b, r in mm_.items():
            print(f"    {b:<12}{r['n_holds']:>4}{r['resid_uT']:>7.2f}µT"
                  f"{r['noise_uT']:>7.2f}{r['resid_over_noise']:>7.1f}"
                  f"{r['dir_err_deg']:>8.2f}°{r['field_uT']:>7.1f}"
                  f"{r['hard_iron_uT']:>9.1f}{r['pose_diversity']:>11.2f}")
        pooled = mm_.get("(전체 합침)")
        per = [r for b, r in mm_.items() if b != "(전체 합침)"]
        if pooled and per:
            mx = max(r["resid_over_noise"] for r in per)
            if pooled["resid_over_noise"] > 1.3 * mx:
                print(f"    블록별로는 잔차가 잡음의 {mx:.1f} 배인데 합치면"
                      f" {pooled['resid_over_noise']:.1f} 배로 뛴다 —")
                print("    한 블록 안에서는 센서 고정 모형이 맞고, **블록 사이에서**"
                      " 무언가 달라진다는 뜻이다.")
                print("    후보는 둘: 팔 위치(=장이 다르다) 또는 칩 보정 상태가"
                      " 그 사이에 바뀐 것. 아래 두 줄이 가른다.")
        worst_b, worst = max(mm_.items(), key=lambda kv: kv[1]["resid_over_noise"])
        w = mm_[worst_b]
        if w["resid_over_noise"] > 3.0:
            print(f"    잔차가 잡음의 {w['resid_over_noise']:.1f} 배다 ({worst_b}) —"
                  " 센서에 고정된 어떤 왜곡으로도 설명이 안 된다.")
            print("    **다만 이것만으로 '환경이 변한다' 를 단정할 수 없다.** 칩 보정"
                  " 미수렴, 자세 다양성 부족,")
            print("    GT 자세 오차도 같은 모양의 잔차를 만든다. 가르는 것은 아래 두"
                  " 줄이다 — 칩 보정 상태와,")
            print("    자세를 고정하고 위치만 옮긴 쌍(mag_map). 그 둘이 없으면 여기서"
                  " 결론을 내지 말 것.")
        else:
            print("    잔차가 잡음 수준이다 — 센서 고정 왜곡으로 설명된다."
                  " 자세 다양성이 낮으면(< 0.3) 해가 유일하지 않으니")
            print("    그 경우 '설명된다' 가 '맞다' 는 뜻은 아니다.")

    mp = {b: r for b, r in (res.get("mag_position") or {}).items()
          if r.get("n_pairs")}
    if mp:
        print("\n  자세 고정 / 위치만 이동  (모형 없는 판정 — 센서 고정 왜곡이라면"
              " m 이 같아야 한다)")
        print(f"    {'블록':<12}{'쌍':>4}{'위치차 최대':>11}{'Δm 중앙':>10}"
              f"{'잡음':>8}{'배수':>7}{'Δ방향':>8}")
        for b, r in mp.items():
            print(f"    {b:<12}{r['n_pairs']:>4}{r['dp_max_mm']:>9.0f}mm"
                  f"{r['dm_median_uT']:>9.2f}µT{r['noise_uT']:>7.2f}"
                  f"{r['dm_over_noise']:>7.1f}{r['dm_deg_median']:>7.1f}°")
        worst = max(mp.values(), key=lambda r: r["dm_over_noise"])
        thin = max(r["n_pairs"] for r in mp.values()) < 10
        if worst["dm_over_noise"] > 3.0:
            print("    자세가 같은데 m 이 잡음의 몇 배로 다르다 — **장이 위치에 따라"
                  " 변한다.** 보정으로 못 고친다.")
        elif worst["dm_over_noise"] > 1.5:
            print("    차이가 잡음 위에 있지만 크지 않다. 위치를 더 벌려"
                  " (300 mm 이상) 다시 볼 것.")
        else:
            print("    자세가 같으면 m 도 같다 — 장은 위치에 안 따라간다."
                  " 자세오차의 원인은 보정 쪽에서 찾을 것.")
        if thin:
            print("    [주의] 쌍이 10 개 미만이다. 우연히 얻은 쌍으로는 결론이 약하다 —"
                  " mag_map 블록을 딸 것:")
            print("           python3 run_session.py --blocks mag_map")
    else:
        print("\n  [주의] 자세 고정 / 위치 이동 쌍이 없어 **모형 없는 판정을 못 했다.**")
        print("         '장이 변한 것' 과 '보정이 안 된 것' 은 대처가 정반대다."
              "  python3 run_session.py --blocks mag_map")

    cal_rows = [(b, g["chip_cal"]) for b, g in res["gating"].items() if g.get("chip_cal")]
    if cal_rows:
        print("\n  칩 보정 상태 (BNO085 자체 평가, 0~3):")
        for b, c in cal_rows:
            print(f"    {b:<12} " + "  ".join(
                f"{k} {v['median']:.0f}(만점 {100 * v['frac_full']:.0f}%)"
                for k, v in c.items()))
        magf = [c["mag"]["frac_full"] for _, c in cal_rows if "mag" in c]
        if magf and max(magf) < 0.5:
            print("    [주의] 자력계가 보정 3 에 도달한 시간이 절반도 안 된다."
                  " 이 상태의 9 축 결과로는")
            print("           '환경이 교란됐다' 를 주장할 수 없다 —"
                  " 보정이 수렴을 안 한 것과 구별되지 않는다.")
    else:
        print("\n  [주의] 칩 보정 상태(cal_status)가 기록에 없다."
              " 자기 교란 판정이 반쪽이 된다 —")
        print("         자세오차가 '환경' 탓인지 '칩 보정 미수렴' 탓인지 못 가른다."
              " 로거를 갱신하고 다시 딸 것.")

    # 적용 중인 보정이 **일을 거꾸로 하고 있지 않은가.** 보정 후 복각 산포가
    # 원시보다 크면 그 mag_cal 은 무효다 (구를 조금만 훑고 맞춘 해가 이렇게 된다).
    worse = [(b, mg["raw"]["dip_sd_deg"], mg["cal"]["dip_sd_deg"])
             for b, g in res["gating"].items()
             for mg in [g.get("magnetic") or {}]
             if mg.get("raw") and mg.get("cal")
             and mg["cal"]["dip_sd_deg"] > mg["raw"]["dip_sd_deg"] * 1.2]
    if worse:
        print(f"\n  [주의] 적용 중인 mag_cal 이 복각 산포를 **키우고 있다** "
              f"({len(worse)}/{len(res['gating'])} 블록):")
        for b, r_, c_ in worse:
            print(f"           {b:<12} 원시 {r_:.1f}° -> 보정 후 {c_:.1f}°")
        print("           보정이 무효라는 뜻이다. mag_calib 의 residual_pct 와"
              " coverage_deg 를 볼 것 —")
        print("           구를 조금만 훑고 맞춘 해가 이렇게 된다."
              " 로봇 옆 실제 자기환경에서 다시 뜰 것.")
    lag_ms = ((res.get("rotation", {}).get(res.get("primary_source", "chip")) or {})
              .get("geodesic", {}) or {}).get("lag_ms", float("nan"))
    if np.isfinite(lag_ms) and lag_ms < 0:
        print(f"\n  [읽는 법] 실효 지연이 음수다 ({lag_ms:+.1f} ms) — IMU 가 로봇 상태"
              " 스트림보다 **앞선다.**")
        print("            센서가 빠른 것이 아니라 GT 가 늦은 것이다: 컨트롤러 UDP"
              " 상태 패키지가 PC 에 닿는 몫이")
        print("            IMU 자신의 지연보다 크면 이렇게 보인다. 이 리그에서 IMU"
              " 지연의 절대값은 이 방식으로 못 읽는다 —")
        print("            읽으려면 GT 쪽 전송 지연을 따로 재거나, sweep 블록으로"
              " L/T 를 분리해야 한다.")

    print(f"\n  {qc.ROBOT_STATE_LAG_NOTE}")
    print(f"  기준값 출처: {R.SOURCE['latency']}")
    print(f"              {R.SOURCE['displacement']}")


# ------------------------------------------------------------------ main
def analyse(sources=("chip",), r_cad=None, primary="probe"):
    program = qc.load_json(qc.PROGRAM_JSON, {}) or {}
    blocks = [b for b in qc.list_blocks(qc.DIR_ROBOT)]
    if not blocks:
        raise SystemExit(f"기록이 없다: {qc.DIR_ROBOT}")

    res = {"blocks": blocks, "sources": list(sources), "primary_block": primary,
           "primary_source": sources[0], "timebase": {}, "gating": {},
           "mag_position": {}, "mag_model": {},
           "rotation": {}, "translation": {}, "lockin": {}, "reference": ref.as_dict()}
    loaded = {}
    for b in blocks:
        gt, imu = load_block(b, sources)
        if gt is None:
            continue
        loaded[b] = (gt, imu)
        res["timebase"][b] = {
            "gt_n": gt["audit"]["n"], "gt_hz": gt["audit"].get("rate_hz", float("nan")),
            "gt_jitter_p95_ms": gt["audit"].get("jitter_p95_ms", float("nan")),
            "imu_n": imu["audit"]["n"], "imu_hz": imu["audit"].get("rate_hz", float("nan")),
            "paired_frac": gt["paired_frac"],
            "skew_ppm": float(imu["attrs"].get("clock_skew_ppm", np.nan)),
            "resid_p95_ms": float(imu["attrs"].get("clock_resid_p95_ms", np.nan)),
            # 두 로거가 **같은 시간대**를 담고 있나. 한쪽만 다시 딴 채 옛 파일이
            # 남으면 여기가 음수가 된다. 그 블록은 아래 분석에서 통째로 조용히
            # 빠지므로(겹치는 격자가 없다) 감사에서 잡지 못하면 아무도 모른다.
            "overlap_s": float(min(gt["t"][-1], imu["t"][-1])
                               - max(gt["t"][0], imu["t"][0])),
        }
        res["gating"][b] = {
            "crc_errors": int(imu["attrs"].get("crc_errors", 0)),
            "seq_gaps": int(imu["attrs"].get("seq_gaps", 0)),
            "zero_still": bool(imu["attrs"].get("zero_still", False)),
            "mag_cal": bool(imu["attrs"].get("mag_cal", False)),
            "retreat_events": int(gt["attrs"].get("retreat_events", 0)),
            "gimbal_frac": float(1.0 - np.mean(gt["spin_valid"] & gt["tip_valid"])),
            "magnetic": magnetic_disturbance(imu),        # 싼 지표만
            "chip_cal": chip_cal_status(imu),
        }

    # --- 정렬: align 블록 우선, 없으면 primary 블록의 정지 구간 ---------
    src0 = sources[0]
    align_block = "align" if "align" in loaded else primary
    if align_block not in loaded:
        raise SystemExit("정렬을 풀 블록이 없다 (align 또는 probe)")
    gt_a, imu_a = loaded[align_block]
    # 보정으로 못 지우는 몫은 여기서 한 번만 잰다 — 자세가 다양한 블록이라야
    # 물음이 성립하고, 그 블록이 곧 정렬을 푸는 블록이다.
    deep = magnetic_disturbance(imu_a, deep=True)
    if deep and align_block in res["gating"]:
        res["gating"][align_block]["magnetic"] = deep
    # 회전을 아는 쪽이 엄격하게 낫다 — 불변량 전용 적합은 조건수가 나쁘다.
    res["mag_model"] = {}
    for b, (g2, i2) in loaded.items():
        r = mag_model_residual([(g2, i2)])
        if r:
            res["mag_model"][b] = r
    # 블록을 **합쳐서** 한 모형으로 푼다. 센서에 고정된 왜곡이라면 블록이 바뀌어도
    # 같은 (K, b, h) 여야 하므로, 합친 잔차가 블록별 잔차와 비슷해야 한다.
    # 합쳤을 때만 뛰면, 블록 사이에서 무언가가 달라진 것이다 (위치, 또는 칩 보정).
    if len(loaded) > 1:
        r = mag_model_residual(list(loaded.values()))
        if r:
            res["mag_model"]["(전체 합침)"] = r
    # 모형 없는 판정. mag_map 블록이 이걸 위해 설계돼 있지만, 다른 블록에서도
    # 조건에 맞는 쌍이 우연히 나오면 쓴다 (그때는 쌍 수가 적다는 것을 같이 낸다).
    res["mag_position"] = {}
    for b, (g2, i2) in loaded.items():
        r = mag_position_test(g2, i2)
        if r:
            res["mag_position"][b] = r
    A, B, info = alignment(gt_a, imu_a, src0)
    info["block"] = align_block
    res["alignment"] = info
    qc.dump_json(qc.ALIGN_JSON, {"A": A.tolist(), "B": B.tolist(), **info})

    AB = {src0: (A, B)}
    for s in sources[1:]:
        try:
            AB[s] = alignment(gt_a, imu_a, s)[:2]
        except SystemExit:
            AB[s] = (A, B)

    # --- 회전 ----------------------------------------------------------
    for s in sources:
        blk = primary if primary in loaded else align_block
        gt, imu = loaded[blk]
        ab = AB[s]
        if s in PER_BLOCK_ALIGN:
            try:
                ab = alignment(gt, imu, s)[:2]
            except SystemExit:
                pass
        rot = analyse_rotation(gt, imu, s, *ab, blk,
                               settle_s=REFUSE_SETTLE_S if s != "chip" else 0.0)
        if rot:
            rot.pop("_grid", None); rot.pop("_sn", None); rot.pop("_lags", None)
        res["rotation"][s] = rot

    # --- 록인 (스크립트 자극 블록이 있으면) ------------------------------
    for b in loaded:
        if b.startswith("sweep_"):
            gt, imu = loaded[b]
            res["lockin"][b] = analyse_lockin(b, gt, imu, src0, *AB[src0], program)

    # --- 병진: 먼저 모형 없이 한 번 돌려 격자를 얻고, primary 로 모형을 푼다 ---
    rot0 = res["rotation"].get(src0) or {}
    lag = (rot0.get("geodesic", {}) or {}).get("lag_ms", 0.0) / 1000.0
    lag = lag if np.isfinite(lag) else 0.0
    res["lag_used_ms"] = 1000.0 * lag

    fit_block = primary if primary in loaded else align_block
    # 동정에는 align 도 같이 넣는다. C·d 는 자세 다양성에서, r 은 회전에서
    # 관측되는데 그 둘을 한 블록이 다 주지 못한다 (identify_accel_model 참조).
    fit_blocks = [b for b in ("align", fit_block) if b in loaded]
    idents = []
    for b in fit_blocks:
        g2, i2 = loaded[b]
        seed = analyse_translation(g2, i2, src0, *AB[src0], b, r_cad=r_cad, lag_s=lag)
        if seed:
            idents.append(seed["_ident"])
    model = identify_accel_model(idents, r0=r_cad) if idents else None
    if model:
        model["fit_block"] = "+".join(fit_blocks)
    res["model"] = model

    for b in loaded:
        if b == "align":
            continue
        gt, imu = loaded[b]
        tr = analyse_translation(gt, imu, src0, *AB[src0], b,
                                 r_cad=r_cad, lag_s=lag, model=model)
        if tr:
            tr.pop("_ident", None)
            # D 단계는 모형을 **푼 블록에서는 자기평가**다. 그 사실을 데이터에 남긴다.
            tr["model_is_fit_block"] = bool(model and b in fit_blocks)
            res["translation"][b] = tr

    val_block = "probe_cal" if "probe_cal" in res["translation"] else None
    res["validation_block"] = val_block
    if model and not val_block:
        print("  [주의] probe_cal 블록이 없어 캘리브레이션의 **검증을 못 한다.**"
              " D 단계는 자기평가일 뿐이니 그렇게 읽을 것.")
    for tr in res["translation"].values():
        tr.pop("_seg_data", None)

    res["verdict"] = verdict(res)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="보존된 run 또는 raw_data 경로")
    ap.add_argument("--sources", default="chip",
                    help="자세 소스: chip,host9,host6 (쉼표 구분). "
                         "host* 는 원시값으로 오프라인 재퓨전이라 느리다")
    ap.add_argument("--block", default="probe", help="주 분석 블록")
    ap.add_argument("--lever-arm-mm", default=None,
                    help="CAD 지렛대 r [mm], 'x,y,z' (tool 프레임). 없으면 회귀값을 쓴다")
    ap.add_argument("--json", default=None, help="결과 JSON 경로 (기본 meta/qc_result.json)")
    args = ap.parse_args()

    if args.run:
        qc.use_raw_dir(qc.resolve_run(args.run))
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    r_cad = None
    if args.lever_arm_mm:
        r_cad = np.array([float(v) for v in args.lever_arm_mm.split(",")]) / 1000.0

    res = analyse(sources=sources, r_cad=r_cad, primary=args.block)
    print_report(res)
    path = qc.dump_json(args.json or qc.RESULT_JSON, res)
    print(f"\n  -> {path}")
    bad = [v for v in res["verdict"] if v["pass"] is False]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
