"""영점 캘리브레이션 — 로깅 시작 시 '움직이지 않은 상태'를 기준으로 잡는다.

무엇을 재고 왜 필요한가:

* **기준 자세 `q0`** — 이후 자세를 "처음 대비 얼마나 돌았는가"로 읽기 위한 원점.
  절대 자세는 지구 프레임(중력+지자기) 기준이라 그 자체로는 로봇 base 와 무관하다.

* **지구 프레임 가속도 바이어스 `b_E`** — 변위 파이프라인의 2 단계에서 빼는 값
  (QC_PLAN §3.6). 정지 시 지구 프레임 가속도는 `(0, 0, g)` 여야 하는데 실제로는
  가속도계 스케일 오차 때문에 `g` 가 어긋나고 수평 성분도 0 이 아니다. 그 실측값을
  통째로 빼는 것이 **가장 큰 위치 오차원 하나를 없애는 유일한 방법**이다.
  공칭 중력만 빼면 1 초에 112 mm 가 틀어진다 (QC_PLAN §2.6).

* **정지 여부 게이트** — 움직이는 동안 잡은 영점은 조용히 쓰레기가 된다. 자이로·가속도
  산포로 판정하고, 실패하면 값을 쓰지 않고 알린다. QC 팀이 hand-eye 수집에서
  "FCI 가 꺼진 채로 동일 포즈 30 개를 저장하고도 에러 없이 끝난다"고 경고한 것과 같은
  종류의 실패다 — **조용히 실패하지 않게** 만드는 것이 요점이다.
"""

import math

import numpy as np

from fusion import quat_conj, quat_mul, quat_to_euler_deg, quat_to_matrix

G0 = 9.80665

# 정지 판정 임계 — 손으로 들고 있어도 통과하지 않을 만큼 빡빡하게.
STILL_GYRO_SD = 0.005      # rad/s
STILL_GYRO_MEAN = 0.005    # rad/s
STILL_ACCEL_SD = 0.15      # m/s^2


class ZeroReference:
    """영점 한 벌. 로그 메타데이터로 그대로 직렬화된다."""

    def __init__(self, q0, b_E, gyro_bias, quality, quat_convention):
        self.q0 = np.asarray(q0, dtype=float)
        self.b_E = np.asarray(b_E, dtype=float)
        self.gyro_bias = np.asarray(gyro_bias, dtype=float)
        self.quality = dict(quality)
        self.quat_convention = quat_convention

    # ---- 적용 ----------------------------------------------------------
    def relative_quat(self, q):
        """기준 자세 대비 상대 회전. '처음 자세에서 얼마나 돌았는가'."""
        return quat_mul(quat_conj(self.q0), np.asarray(q, dtype=float))

    def relative_euler_deg(self, q):
        return quat_to_euler_deg(self.relative_quat(q))

    def linear_accel(self, q, acc):
        """센서 가속도를 지구 프레임으로 돌리고 바이어스(중력 포함)를 뺀다."""
        R = quat_to_matrix(np.asarray(q, dtype=float))
        f_E = R @ np.asarray(acc, dtype=float) if self.quat_convention == "R" else R.T @ acc
        return f_E - self.b_E

    # ---- 직렬화 --------------------------------------------------------
    def to_dict(self):
        return {
            "q0": self.q0.tolist(),
            "accel_bias_earth": self.b_E.tolist(),
            "gravity_measured": float(np.linalg.norm(self.b_E)),
            "gravity_scale_error_pct": float(100.0 * (np.linalg.norm(self.b_E) / G0 - 1.0)),
            "gyro_bias": self.gyro_bias.tolist(),
            "quat_convention": self.quat_convention,
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, d):
        if not d:
            return None
        return cls(d["q0"], d["accel_bias_earth"], d["gyro_bias"],
                   d.get("quality", {}), d.get("quat_convention", "R"))

    # ---- 사람이 읽는 보고 ----------------------------------------------
    def report(self):
        q = self.quality
        r, p, y = quat_to_euler_deg(self.q0)
        g = np.linalg.norm(self.b_E)
        tilt = math.degrees(math.atan2(np.linalg.norm(self.b_E[:2]), abs(self.b_E[2])))
        lines = [
            "  기준 자세 q0      : [%+.4f %+.4f %+.4f %+.4f]  rpy=(%+7.2f %+7.2f %+7.2f)"
            % (*self.q0, r, p, y),
            "  지구프레임 바이어스: (%+.4f %+.4f %+.4f) m/s^2" % tuple(self.b_E),
            "  측정 중력         : %.4f m/s^2   (표준 9.80665, 스케일 오차 %+.2f %%)"
            % (g, 100.0 * (g / G0 - 1.0)),
            "  중력벡터 기울기   : %.3f deg   (0 에 가까워야 자세 추정이 맞은 것)" % tilt,
            "  자이로 잔류 바이어스: (%+.2e %+.2e %+.2e) rad/s" % tuple(self.gyro_bias),
            "  정지 판정         : %s   (gyro σ %.5f, mean %.5f / accel σ %.4f, n=%d, %.1f s)"
            % ("통과" if q.get("still") else "**실패**",
               q.get("gyro_sd", float("nan")), q.get("gyro_mean_abs", float("nan")),
               q.get("accel_sd", float("nan")), q.get("n", 0), q.get("duration", 0.0)),
        ]
        if not q.get("still"):
            lines.append("  ! 영점 수집 중 센서가 움직였습니다. 가만히 둔 채 다시 잡으세요.")
        return "\n".join(lines)


def _earth_convention(quat, acc):
    """쿼터니언 규약을 데이터로 정한다 — 정지 시 지구 z 가 +g 가 되는 쪽.

    문서를 믿지 않는 이유: 규약을 반대로 잡으면 중력이 안 빠지고 오차가 100 배가 되는데,
    그 증상은 '센서가 나쁘다'로 오인되기 딱 좋다.
    """
    R = np.array([quat_to_matrix(q) for q in quat])
    fwd = np.einsum("nij,nj->ni", R, acc)
    inv = np.einsum("nji,nj->ni", R, acc)
    score = lambda v: abs(v[:, 2].mean() - G0) + np.abs(v[:, :2].mean(0)).sum()
    return ("R", fwd) if score(fwd) <= score(inv) else ("R.T", inv)


def measure(samples):
    """capture 로 모은 샘플에서 영점을 계산한다.

    samples: [(t, acc(3), gyr(3), quat(4)), ...]  — quat 은 칩 RV
    반환: ZeroReference (정지 판정 실패해도 값은 채워서 돌려준다. 쓸지는 호출자가 정한다)
    """
    if len(samples) < 20:
        raise ValueError("영점 계산에 샘플이 부족합니다 (%d 개)" % len(samples))

    t = np.array([s[0] for s in samples])
    A = np.array([s[1] for s in samples])
    G = np.array([s[2] for s in samples])
    Q = np.array([s[3] for s in samples])

    conv, f_E = _earth_convention(Q, A)

    gyro_sd = float(np.linalg.norm(G.std(0)))
    gyro_mean_abs = float(np.linalg.norm(G.mean(0)))
    accel_sd = float(np.linalg.norm(A.std(0)))
    still = (gyro_sd < STILL_GYRO_SD and gyro_mean_abs < STILL_GYRO_MEAN
             and accel_sd < STILL_ACCEL_SD)

    # 기준 자세는 평균 쿼터니언 대신 중앙 샘플을 쓴다 — 쿼터니언 산술평균은
    # 정지 상태에서도 부호 뒤집힘에 취약하고, 이득이 없다.
    q0 = Q[len(Q) // 2]
    q0 = q0 / np.linalg.norm(q0)

    quality = {
        "still": bool(still),
        "gyro_sd": gyro_sd,
        "gyro_mean_abs": gyro_mean_abs,
        "accel_sd": accel_sd,
        "n": int(len(samples)),
        "duration": float(t[-1] - t[0]),
        "thresholds": {"gyro_sd": STILL_GYRO_SD, "gyro_mean_abs": STILL_GYRO_MEAN,
                       "accel_sd": STILL_ACCEL_SD},
    }
    return ZeroReference(q0, f_E.mean(0), G.mean(0), quality, conv)
