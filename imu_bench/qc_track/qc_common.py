#!/usr/bin/env python3
"""qc_track 공통 규약 — 경로, 프레임, 시계 정합, 표 입출력, 정지 판정.

이 QC 가 재는 것
----------------
FR5 플랜지에 BNO085 를 달고 **teleop 으로 초음파 프로빙하듯 움직이는 동안**,
IMU 가 회전과 병진을 로봇 FK(=GT) 대비 얼마나 정확히 따라가는가.

  · 회전   GT = tl_cur_pos 의 자세            vs  BNO085 RV / 호스트 퓨전
  · 병진   GT = tl_cur_pos 의 위치 (+지렛대)   vs  가속도 이중적분 (+ZUPT)

Surgilogger QC(Exp-Latency, 2026-07-12) 와의 관계
-------------------------------------------------
프레임 규약·시계 정합·정렬 해법(X = A·Y·B)·록인/상호상관은 **그쪽과 같은 것**을
쓴다. 숫자를 나란히 놓고 읽어야 하기 때문이다 (reference.py 가 그쪽 값을 들고
있고 분석기가 결과에 같이 박는다). 달라진 것은 셋이다.

  1. GT 소스가 FR3 tip_pose 200 Hz 가 아니라 FR5 ``ee_wrt_base`` 다.
     기본 30 Hz 이므로 QC 때는 ``rates.status_publish_hz`` 를 올려서 띄운다.
  2. 삽입깊이(ToF) 채널이 없다. 대신 **병진** 채널이 들어온다 — IMU 는 병진을
     직접 재지 못하므로 이중적분 + ZUPT 이고, 평가 방식이 회전과 다르다.
  3. 자극이 스크립트가 아니라 **사람 손(teleop)** 이다. 그래서 상호상관 실효
     지연은 "그 운동의 스펙트럼에 딸린 값" 이라는 단서가 회전 채널에도 붙는다.
     L/T 분리가 필요하면 excite_twist.py 로 스크립트 자극 블록을 따로 붙인다.

프레임 규약
-----------
로봇 tool 프레임은 probe.yaml §tool 의 정의를 따른다 — **+z 가 조직 침투 방향**,
+x 가 트랜스듀서 배열 방향. ``allow_missing_tool: true`` 인 동안 tool 은 J6
플랜지 그 자체다 (us_servo_node 가 tl_cur_pos 를 그대로 낸다). 즉 이 QC 가 재는
GT 자세는 **플랜지 자세**이고, IMU 는 그 플랜지에 고정된 강체다.

Surgilogger 쪽은 샤프트가 tip->핸들 +z 라 ``shaft = -R[:,2]`` 였다. 여기는 그
부호 반전이 **없다** — 프로브 침투축이 곧 +z 다. 같은 이름의 함수라도 부호가
다르므로 그쪽 코드를 그대로 복사해 오지 말 것. (실기에서 축 부호가 조용히
뒤집히면 tilt 와 roll 이 통째로 틀린 채 실험이 끝난다 — Exp-Latency §7.2.)
"""
import json
import os
import signal
import time

import numpy as np

G0 = 9.80665

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
HOST = os.path.join(BENCH, "host")

RAW = os.path.join(HERE, "raw_data")
DIR_ROBOT = os.path.join(RAW, "robot")
DIR_IMU = os.path.join(RAW, "imu")
DIR_META = os.path.join(RAW, "meta")

PROGRAM_JSON = os.path.join(DIR_META, "program.json")
ALIGN_JSON = os.path.join(DIR_META, "alignment.json")
RESULT_JSON = os.path.join(DIR_META, "qc_result.json")
TRIALS_JSONL = os.path.join(DIR_META, "trials.jsonl")
CURRENT_TRIAL = os.path.join(DIR_META, "current_trial.json")

# --- 로봇 인터페이스 ------------------------------------------------------
#
# fr5_control/config/probe.yaml 의 robot.name 과 같아야 한다. 바꾸려면 로거의
# --robot 인자를 쓴다 (여기 상수를 고치지 말 것 — 기본값의 출처가 흐려진다).
ROBOT = "fr5_right"


def topics(robot=ROBOT):
    ns = f"/{robot}"
    return {
        "pose":    f"{ns}/ee_wrt_base",             # geometry_msgs/Pose  (헤더 없음!)
        "joints":  f"{ns}/joint_states",            # sensor_msgs/JointState (헤더 있음)
        "wrench":  f"{ns}/wrench",
        "twist":   f"{ns}/desired_twist",
        "track_err": "/diag/twist_tracking_error",
        "retreat":   "/diag/retreating",
    }


# GT 시각을 어떻게 정하나
# ----------------------
# ``ee_wrt_base`` 는 geometry_msgs/Pose 라 **타임스탬프가 없다.** 그런데
# us_servo_node._publish_status() 는 pose 와 joint_states 를 같은 콜백에서 같은
# ``stamp`` 로 낸다. 그래서 로거는 pose 를 받은 pc_ts 와 가장 가까운 joint_states
# 의 헤더 시각을 그 pose 의 시각으로 삼는다 — 구독자 쪽 스케줄링 지터가 GT
# 시간축에서 빠진다. 짝을 못 찾으면 pc_ts 를 그대로 쓰고 그 사실을 남긴다.
PAIR_TOL_S = 0.020


# 정직하게 밝혀 둘 것 -------------------------------------------------------
#
# 이 QC 가 내는 지연은 **"로봇 상태 스트림 대비" 지연**이다. 컨트롤러 UDP 상태
# 패키지(tl_cur_pos)가 PC 에 닿기까지의 몫은 GT 쪽에 이미 실려 있고 여기서
# 지울 방법이 없다. 그 몫은 세 회전 채널에 **똑같이** 실리므로 채널 사이의
# 차이는 정확하고, 절대값에는 그만큼의 공통 오프셋이 있다.
ROBOT_STATE_LAG_NOTE = (
    "지연은 로봇 상태 스트림(tl_cur_pos) 기준이다. 컨트롤러->PC 몫은 공통 오프셋으로 남는다."
)


# --------------------------------------------------------------- 회전 유틸
def quat_xyzw_to_R(q):
    """(...,4) xyzw -> (...,3,3). 정규화는 여기서 한다."""
    q = np.asarray(q, float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_wxyz_to_R(q):
    """BNO085 / 호스트 퓨전은 w,x,y,z 순으로 준다."""
    q = np.asarray(q, float)
    return quat_xyzw_to_R(np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], -1))


def project_SO3(M):
    """가장 가까운 회전행렬. 반사(det<0)는 뒤집어 막는다."""
    U, _, Vt = np.linalg.svd(np.asarray(M, float))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def geodesic_deg(Ra, Rb):
    """두 회전 사이 각도[deg]. 배치 입력."""
    Ra, Rb = np.asarray(Ra, float), np.asarray(Rb, float)
    E = np.einsum("...ij,...kj->...ik", Ra, Rb)
    tr = np.trace(E, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def so3_log_deg(Ra, Rb):
    """Rb^T Ra 의 회전벡터 [deg,3]. 어느 축에서 틀리는지를 보려고 쓴다."""
    Ra, Rb = np.asarray(Ra, float), np.asarray(Rb, float)
    E = np.einsum("...ji,...jk->...ik", Rb, Ra)          # Rb^T Ra
    tr = np.trace(E, axis1=-2, axis2=-1)
    th = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    v = np.stack([E[..., 2, 1] - E[..., 1, 2],
                  E[..., 0, 2] - E[..., 2, 0],
                  E[..., 1, 0] - E[..., 0, 1]], -1)
    s = np.sin(th)
    # th->0 에서 v/(2 sin th) -> v/2 다. 0 나눔을 피하려고 갈라 쓴다.
    scale = np.where(s > 1e-8, th / np.maximum(2 * s, 1e-12), 0.5)
    return np.degrees(v * scale[..., None])


# ------------------------------------------------------- 프로브 각도 (화면 값)
UP = np.array([0.0, 0.0, 1.0])


def probe_axis(R):
    """프로브 침투 방향 = tool **+z** (probe.yaml §tool).

    Surgilogger 쪽 shaft_axis 는 -z 였다. 부호가 반대라는 것을 여기 한 번만
    적어 두고, 축 부호는 register_pose.py 의 일치 검사가 실기에서 다시 본다.
    """
    return np.asarray(R, float)[..., :, 2]


def tilt_deg(u):
    """침투축이 **연직 아래(-z)** 에서 몇 도 벗어났나. 프로브가 수직이면 0.

    Surgilogger 의 '하방각(depression)' 과 **정확히 여각**이다:

        depression = asin(-u_z),   tilt = acos(-u_z),   depression + tilt = 90 deg

    상수 오프셋과 부호만 다르므로 지연·이득·오차 크기는 그대로 비교된다.
    그런데도 여기서 여각을 쓰는 이유: 프로빙 자세는 프로브를 거의 수직으로
    세우므로 depression 이 90 도 근처에 붙어 **변화가 두 자릿수 작게 보인다.**
    tilt 는 0 근처에서 움직여 눈금이 맞는다.
    """
    u = np.asarray(u, float)
    return np.degrees(np.arccos(np.clip(-u[..., 2], -1.0, 1.0)))


def tip_xy_deg(u):
    """기울기를 두 성분으로. (tip_x, tip_y) [deg].

        tip_x = atan2(u_x, -u_z)      base +x 쪽으로 기운 각
        tip_y = atan2(u_y, -u_z)      base +y 쪽으로 기운 각

    '기울기 크기 + 기울어진 방위' 로 쪼개지 않는 이유: 수직에 가까우면 방위가
    **정의되지 않는다** (0/0). 프로빙은 바로 그 근처에서 논다. 두 성분은 어디서도
    무너지지 않고, 둘 다 0 근처의 작은 각이라 지연 추정에 그대로 쓸 수 있다.
    """
    u = np.asarray(u, float)
    dn = -u[..., 2]
    return (np.degrees(np.arctan2(u[..., 0], dn)),
            np.degrees(np.arctan2(u[..., 1], dn)))


def tip_valid(u, min_down=0.1):
    """침투축이 아래를 향하고 있나. 옆이나 위를 보면 tip_x/tip_y 가 감긴다."""
    return -np.asarray(u, float)[..., 2] > min_down


def spin_deg(R):
    """침투축 둘레의 회전. 기준선은 **base +x 를 축에 수직한 평면에 투영**한 것.

    Surgilogger 는 이 기준선을 지평선(위쪽 +z 의 투영)으로 잡았다. 그쪽은 도구가
    트로카에 비스듬히 꽂혀 있어 그래도 됐다. 여기서는 안 된다 — 프로브는 거의
    연직이라 UP 이 축과 나란해지고 **투영이 0 벡터가 된다** (gimbal). 기준선을
    base +x 로 바꾸면 그 특이점이 '프로브가 수평으로 +-x 를 정확히 겨눌 때' 로
    옮겨 가고, 그 자세는 프로빙에서 안 나온다.

    기준선이 다르므로 이 값의 **절대치는** Surgilogger 의 roll 과 다르다. 같은
    것은 '축 둘레 회전' 이라는 물리량이고, 지연·이득·오차는 그대로 비교된다
    (기준선 차이는 상수 오프셋이다).
    """
    R = np.asarray(R, float)
    u = probe_axis(R)
    base_x = np.zeros(u.shape); base_x[..., 0] = 1.0
    ref = base_x - u * np.sum(u * base_x, axis=-1, keepdims=True)
    n = np.linalg.norm(ref, axis=-1, keepdims=True)
    fallback = np.zeros(u.shape); fallback[..., 1] = 1.0
    ref = np.where(n > 1e-6, ref / np.maximum(n, 1e-12), fallback)
    xt = R[..., :, 0]
    xt = xt - u * np.sum(u * xt, axis=-1, keepdims=True)
    xt = xt / np.maximum(np.linalg.norm(xt, axis=-1, keepdims=True), 1e-12)
    s = np.sum(np.cross(ref, xt) * u, axis=-1)
    c = np.sum(ref * xt, axis=-1)
    return np.degrees(np.arctan2(s, c))


SPIN_GIMBAL = 0.95          # |u . x_hat| 가 이보다 크면 spin 기준선을 못 믿는다


def spin_valid(u):
    return np.abs(np.asarray(u, float)[..., 0]) < SPIN_GIMBAL


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(np.asarray(a, float))))


# ------------------------------------------------------------------ 시계 정합
def fit_clock(device_t, pc_t, iters=6):
    """장치 시계 -> PC 시계 선형 정합. pc ~= a*device + b. **하단 포락선**.

    최소자승이 아닌 이유: pc_ts = 진짜도착시각 + 지연 이고 지연은 항상 0 이상이라
    잔차가 한쪽으로 쏠린다. 선 아래 점만 남기며 다시 맞추면 포락선으로 수렴한다.
    시리얼 read 한 번이 여러 프레임을 물고 오므로 특히 중요하다.
    """
    d = np.asarray(device_t, float)
    t = np.asarray(pc_t, float)
    ok = np.isfinite(d) & np.isfinite(t)
    d, t = d[ok], t[ok]
    if d.size < 10 or np.ptp(d) <= 0:
        return np.nan, np.nan, {"n": int(d.size), "ok": False}
    keep = np.ones(d.size, bool)
    a = b = np.nan
    for _ in range(iters):
        a, b = np.polyfit(d[keep], t[keep], 1)
        r = t - (a * d + b)
        nxt = r <= np.median(r[keep])
        if nxt.sum() < 10:
            break
        keep = nxt
    r = t - (a * d + b)
    return float(a), float(b), {
        "n": int(d.size), "ok": True,
        "skew_ppm": float((a - 1.0) * 1e6),
        "resid_med_ms": float(np.median(r) * 1e3),
        "resid_p95_ms": float(np.percentile(r, 95) * 1e3),
        "resid_max_ms": float(r.max() * 1e3),
        "span_s": float(np.ptp(d)),
    }


def apply_clock(device_t, a, b):
    return a * np.asarray(device_t, float) + b


def rate_audit(t, name=""):
    """레이트·지터·구멍. 시간축 감사는 **다른 무엇보다 먼저** 본다 —
    시간축이 틀어진 데이터로 낸 오차는 오차가 아니라 정렬 실패다."""
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    if t.size < 3:
        return {"name": name, "n": int(t.size), "ok": False}
    dt = np.diff(t)
    dt = dt[dt > 0]
    med = float(np.median(dt))
    return {"name": name, "n": int(t.size), "ok": True,
            "span_s": float(t[-1] - t[0]),
            "rate_hz": float(1.0 / med) if med > 0 else float("nan"),
            "jitter_p95_ms": float(np.percentile(np.abs(dt - med), 95) * 1e3),
            "gap_max_ms": float(dt.max() * 1e3),
            "gaps_over_3x": int(np.sum(dt > 3 * med))}


# ------------------------------------------------------------------ 정지 판정
#
# host/zero_ref.py 의 게이트와 같은 기준이다. ZUPT 구간을 여기서 자르므로
# 값이 두 군데로 갈라지면 "영점은 통과했는데 ZUPT 는 아니다" 같은 모순이 생긴다.
STILL_GYRO_SD = 0.005       # rad/s
STILL_GYRO_MEAN = 0.010     # rad/s — |w| 평균. 아주 느린 등속 회전을 거른다
STILL_ACCEL_SD = 0.15       # m/s^2


def still_mask(t, gyr, acc, win_s=0.30, dt=None):
    """샘플별 정지 여부. 이동창 안의 자이로·가속도 산포로 판정한다.

    창을 0.3 s 로 잡은 이유: 프로빙 중의 자연스러운 멈춤이 대개 0.5~2 s 라
    그보다 짧아야 구간의 양 끝을 물지 않는다. 반대로 너무 짧으면 사인 운동의
    변곡점(속도 0, 가속도 최대)을 정지로 오인한다 — 그래서 **가속도 산포**를
    같이 본다. 변곡점은 자이로가 0 이어도 가속도가 크게 흔들린다.

    ``dt`` 를 주면 창 폭을 그 값으로 고정한다. 안 주면 받은 데이터의 중앙 간격에서
    가져오는데, 그러면 **같은 신호라도 어느 구간을 넘겨받았느냐에 따라 창이 한두
    샘플 달라진다** (실측 75~77). 조각으로 나눠 이어붙이며 세는 쪽(monitor.py)은
    그 흔들림 때문에 앞뒤가 서로 다른 창으로 계산되므로 반드시 고정해서 부른다.
    """
    t = np.asarray(t, float)
    g = np.asarray(gyr, float)
    a = np.asarray(acc, float)
    n = t.size
    if n < 8:
        return np.zeros(n, bool)
    dt = float(np.median(np.diff(t))) if dt is None else float(dt)
    k = max(3, int(round(win_s / max(dt, 1e-6))))
    out = np.zeros(n, bool)
    gm = np.linalg.norm(g, axis=1)
    for i in range(n):
        lo, hi = max(0, i - k // 2), min(n, i + k // 2 + 1)
        gs = g[lo:hi]; as_ = a[lo:hi]
        if hi - lo < 3:
            continue
        out[i] = (gs.std(axis=0).max() < STILL_GYRO_SD
                  and gm[lo:hi].mean() < STILL_GYRO_MEAN
                  and as_.std(axis=0).max() < STILL_ACCEL_SD)
    return out


def segments_from_mask(t, mask, min_s):
    """mask 가 참인 연속 구간 [(t0,t1,i0,i1), ...]. min_s 보다 짧으면 버린다."""
    t = np.asarray(t, float)
    m = np.asarray(mask, bool)
    out = []
    i = 0
    while i < m.size:
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < m.size and m[j + 1]:
            j += 1
        if t[j] - t[i] >= min_s:
            out.append((float(t[i]), float(t[j]), int(i), int(j)))
        i = j + 1
    return out


# --------------------------------------------------------------- 표 저장/적재
try:
    import h5py as _h5py
    HAVE_H5PY = True
except Exception:                       # noqa: BLE001 -- ABI 불일치는 ImportError 가 아니다
    _h5py = None
    HAVE_H5PY = False


def save_table(path_noext, cols, attrs=None):
    """cols: {이름: ndarray}. h5py 가 있으면 .h5, 없으면 .npz.

    rclpy 쪽(시스템 python)과 분석 쪽의 h5py 가지가 갈릴 수 있어 양쪽을 다
    읽는다. Surgilogger QC 와 같은 관례다.
    """
    attrs = attrs or {}
    os.makedirs(os.path.dirname(os.path.abspath(path_noext)), exist_ok=True)
    if HAVE_H5PY:
        path = path_noext + ".h5"
        with _h5py.File(path, "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            for k, v in cols.items():
                f.create_dataset(k, data=np.asarray(v), compression="gzip",
                                 compression_opts=4)
        return path
    path = path_noext + ".npz"
    payload = {k.replace("/", "__"): np.asarray(v) for k, v in cols.items()}
    payload["__attrs__"] = np.array(json.dumps(attrs, default=float))
    np.savez_compressed(path, **payload)
    return path


def load_table(path_noext):
    """save_table 이 쓴 것을 읽는다. (cols, attrs). 없으면 (None, None)."""
    h5 = path_noext + ".h5"
    if os.path.exists(h5):
        if not HAVE_H5PY:
            raise SystemExit(f"{h5} 를 읽으려면 h5py 가 필요하다")
        with _h5py.File(h5, "r") as f:
            return ({k: f[k][:] for k in _walk(f)}, dict(f.attrs))
    npz = path_noext + ".npz"
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=False)
        attrs = json.loads(str(z["__attrs__"])) if "__attrs__" in z else {}
        cols, bad = {}, []
        for k in z.files:
            if k == "__attrs__":
                continue
            try:
                cols[k.replace("__", "/")] = z[k]
            except Exception:                                   # noqa: BLE001
                bad.append(k.replace("__", "/"))
        if bad:
            attrs["__corrupt_columns__"] = bad
            print(f"  [경고] {os.path.basename(npz)}: 열 {bad} 가 깨져 건너뛴다.")
        return cols, attrs
    return None, None


def _walk(g, prefix=""):
    out = []
    for k in g:
        item = g[k]
        name = f"{prefix}{k}"
        if hasattr(item, "keys"):
            out += _walk(item, name + "/")
        else:
            out.append(name)
    return out


def list_blocks(directory):
    if not os.path.isdir(directory):
        return []
    seen = []
    for f in sorted(os.listdir(directory)):
        base, ext = os.path.splitext(f)
        if ext in (".h5", ".npz") and base not in seen:
            seen.append(base)
    return seen


# --------------------------------------------------------------------- 경로
def ensure_dirs():
    for d in (DIR_ROBOT, DIR_IMU, DIR_META):
        os.makedirs(d, exist_ok=True)


def use_raw_dir(path):
    """raw_data 의 위치를 바꾼다 — 보존된 run 을 **같은 분석기로** 다시 보려고.

    경로를 쓰는 쪽은 전부 qc.DIR_* 를 접근 시점에 읽는다. import 직후에 부를 것.
    """
    global RAW, DIR_ROBOT, DIR_IMU, DIR_META
    global PROGRAM_JSON, ALIGN_JSON, RESULT_JSON, TRIALS_JSONL, CURRENT_TRIAL
    RAW = os.path.abspath(path)
    DIR_ROBOT = os.path.join(RAW, "robot")
    DIR_IMU = os.path.join(RAW, "imu")
    DIR_META = os.path.join(RAW, "meta")
    PROGRAM_JSON = os.path.join(DIR_META, "program.json")
    ALIGN_JSON = os.path.join(DIR_META, "alignment.json")
    RESULT_JSON = os.path.join(DIR_META, "qc_result.json")
    TRIALS_JSONL = os.path.join(DIR_META, "trials.jsonl")
    CURRENT_TRIAL = os.path.join(DIR_META, "current_trial.json")
    return RAW


def resolve_run(name):
    """run 이름 또는 경로 -> raw_data 디렉토리."""
    for cand in (name, os.path.join(HERE, name)):
        if os.path.isdir(os.path.join(cand, "raw_data")):
            return os.path.join(os.path.abspath(cand), "raw_data")
        if os.path.isdir(os.path.join(cand, "meta")):
            return os.path.abspath(cand)
    runs = os.path.join(HERE, "runs")
    if os.path.isdir(runs):
        hit = sorted(d for d in os.listdir(runs) if d.startswith(name))
        if len(hit) == 1:
            return os.path.join(runs, hit[0], "raw_data")
        if len(hit) > 1:
            raise SystemExit(f"'{name}' 에 맞는 run 이 여럿이다: {hit}")
    raise SystemExit(f"run 을 못 찾겠다: {name}")


# ------------------------------------------------------------------ 종료 처리
def install_shutdown_handlers():
    """SIGINT/SIGTERM 을 KeyboardInterrupt 로. 백그라운드에서도 끝낼 수 있게.

    bash 가 비대화형 셸에서 띄운 자식은 SIGINT 처분이 SIG_IGN 이 되고, CPython 은
    이미 SIG_IGN 인 SIGINT 를 자기 핸들러로 덮지 않는다. 그러면 러너의 kill -INT
    가 무동작이 되어 wait 이 영영 안 돌아온다. rclpy.init() **뒤에** 부를 것.
    """
    def _raise(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _raise)
        except (ValueError, OSError):
            pass


# --------------------------------------------------------------------- trial
def write_current_trial(block, extra=None):
    """레코더가 파일명을 짓는 데 쓴다. START 보내기 **전에** 써 둘 것."""
    os.makedirs(DIR_META, exist_ok=True)
    d = {"trial_id": block, "block": block, "written_unix": time.time()}
    if extra:
        d.update(extra)
    tmp = CURRENT_TRIAL + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, CURRENT_TRIAL)
    return d


def read_current_trial():
    try:
        with open(CURRENT_TRIAL) as f:
            return json.load(f)
    except Exception:                                          # noqa: BLE001
        return {"trial_id": f"unknown_{int(time.time())}", "block": "?"}


def append_trial(record):
    os.makedirs(DIR_META, exist_ok=True)
    with open(TRIALS_JSONL, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:                                          # noqa: BLE001
        return default


def dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
    os.replace(tmp, path)
    return path
