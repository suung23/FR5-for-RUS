#!/usr/bin/env python3
"""정지-정지 구간 변위를 IMU 로 얼마나 정확히 잴 수 있는가 — 오차 바닥 실측.

    python3 displacement_check.py logs/imu_*.csv

**질문**: 로봇 끝단이 처음 위치에서 얼마나 움직였는지를 IMU 가 얼마나 잘 감지하는가.

가속도를 지구 프레임으로 돌려 중력을 뺀 뒤 이중적분하면 변위가 나온다. 문제는 남은
가속도 오차가 t^2 로 증폭된다는 것이고, 이 스크립트는 **실제로 안 움직인 구간**에 대해
그 파이프라인을 그대로 돌려 "0 이 나와야 하는데 얼마가 나오는지" 를 잰다. 그게 오차 바닥이고,
실제 이동 구간의 오차는 여기에 스케일 오차 항이 더해진 값이다.

세 가지 보정 단계를 나란히 비교한다 (QC 팀의 Model A~D ablation 과 같은 취지):

  A. 무보정        중력만 공칭값 9.80665 로 뺀다
  B. 바이어스 보정  정지 구간에서 잰 지구프레임 가속도 평균을 뺀다 (스케일 오차까지 흡수)
  C. B + ZUPT     구간 양 끝이 정지임을 알고 속도 드리프트를 선형 제거한다

C 가 실용 상한이다. 로봇 point-to-point 이동은 정지에서 출발해 정지로 끝나므로 ZUPT 를
쓸 수 있고, **상수 가속도 바이어스는 ZUPT 가 원리적으로 완전히 제거한다.** 남는 것은
이동 중 자세가 바뀌면서 중력 투영이 함께 바뀌는 성분과 랜덤워크다.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import zero_ref as zr              # noqa: E402
from fusion import quat_to_matrix  # noqa: E402

G0 = 9.80665


def earth_accel(quat, acc):
    """센서 프레임 가속도를 지구 프레임으로 돌린다.

    쿼터니언 규약(센서->지구인가 지구->센서인가)을 문서에 의존하지 않고 **데이터로**
    정한다: 정지 상태에서 지구프레임 가속도가 (0,0,+g) 에 가까워지는 쪽이 맞는 것이다.
    규약을 잘못 잡으면 중력이 안 빠지고 오차가 100 배로 나오는데, 그건 센서 탓이 아니다.
    """
    R = np.array([quat_to_matrix(q) for q in quat])
    fwd = np.einsum("nij,nj->ni", R, acc)
    inv = np.einsum("nji,nj->ni", R, acc)
    # z 성분이 +g 에 가깝고 수평 성분이 작은 쪽을 고른다
    score = lambda v: abs(v[:, 2].mean() - G0) + np.abs(v[:, :2].mean(0)).sum()
    if score(fwd) <= score(inv):
        return fwd, "R (센서->지구)"
    return inv, "R.T (지구->센서의 역)"


def integrate(a_lin, dt, zupt):
    """가속도 -> 변위. zupt 면 구간 끝 속도가 0 이라는 조건으로 선형 디드리프트."""
    v = np.cumsum(a_lin, axis=0) * dt
    if zupt and len(v) > 1:
        ramp = np.linspace(0.0, 1.0, len(v))[:, None]
        v = v - v[-1] * ramp
    return np.cumsum(v, axis=0) * dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--windows", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
                    help="평가할 구간 길이 [s]")
    ap.add_argument("--calib", type=float, default=10.0,
                    help="바이어스 추정에 쓸 앞부분 길이 [s]")
    ap.add_argument("--still-thresh", type=float, default=0.01)
    ap.add_argument("--ignore-zero", action="store_true",
                    help="로그에 기록된 영점을 무시하고 이 구간에서 다시 추정한다")
    args = ap.parse_args()

    d = np.genfromtxt(args.log, delimiter=",", names=True)
    t = d["pc_unix"] - d["pc_unix"][0]
    A = np.vstack([d["ax"], d["ay"], d["az"]]).T
    G = np.vstack([d["gx"], d["gy"], d["gz"]]).T
    Q = np.vstack([d["chip_qw"], d["chip_qx"], d["chip_qy"], d["chip_qz"]]).T
    if np.all(np.isnan(Q)):
        raise SystemExit("칩 RV 쿼터니언이 로그에 없습니다.")

    # 로그 첫머리에는 칩 RV 가 아직 도착하지 않은 행이 있다 (빈 칸 -> NaN).
    # 그대로 두면 NaN 이 적분 전체로 번져 결과가 통째로 NaN 이 된다.
    ok = np.isfinite(Q).all(1) & np.isfinite(A).all(1) & np.isfinite(G).all(1)
    n_drop = int((~ok).sum())
    if n_drop:
        print("유효하지 않은 행 %d 개 제거 (칩 RV 미도착 등)" % n_drop)
    t, A, G, Q = t[ok], A[ok], G[ok], Q[ok]
    if len(t) < 100:
        raise SystemExit("유효 샘플이 부족합니다 (%d 개)" % len(t))

    fs = len(t) / (t[-1] - t[0])

    # 가장 긴 정지 구간
    still = np.linalg.norm(G, axis=1) < args.still_thresh
    idx = np.flatnonzero(np.diff(np.r_[0, still.view(np.int8), 0]))
    s, e = idx[::2], idx[1::2]
    k = int(np.argmax(e - s))
    s, e = int(s[k]), int(e[k])
    dur = t[e - 1] - t[s]
    print("로그        : %s" % args.log)
    print("정지 구간   : %.1f ~ %.1f s (%.1f s, %d 샘플, %.1f Hz)"
          % (t[s], t[e - 1], dur, e - s, fs))
    if dur < max(args.windows) + args.calib:
        print("  ! 정지 구간이 짧아 긴 창은 건너뜁니다 (필요 %.0f s)"
              % (max(args.windows) + args.calib))

    A, Q = A[s:e], Q[s:e]
    aE, conv = earth_accel(Q, A)
    print("쿼터니언 규약: %s  ->  정지 시 지구프레임 가속도 = (%+.3f %+.3f %+.3f) m/s^2"
          % (conv, *aE.mean(0)))

    # 로깅 시작 시 잡아 둔 영점이 있으면 그걸 쓴다 — 그게 실제 운용 조건이다.
    # 없으면(또는 --ignore-zero) 이 구간 앞부분에서 추정한다.
    meta_path = args.log.rsplit(".", 1)[0] + ".meta.json"
    logged = None
    if not args.ignore_zero and os.path.exists(meta_path):
        with open(meta_path) as f:
            logged = zr.ZeroReference.from_dict(json.load(f).get("zero_ref"))

    nc = int(args.calib * fs)
    if logged is not None:
        bias = logged.b_E
        src = "로그에 기록된 영점 (still=%s)" % logged.quality.get("still")
    else:
        bias = aE[:nc].mean(0)
        src = "이 구간 앞 %.0f s 로 추정 (로그에 영점 없음)" % args.calib
    print("바이어스      : (%+.4f %+.4f %+.4f) m/s^2   |b| = %.4f"
          % (*bias, np.linalg.norm(bias)))
    print("  출처        : %s" % src)
    print("  * 공칭 중력만 뺐을 때 남는 z 잔차 %.3f m/s^2 = 가속도계 스케일 오차"
          % (aE[:nc, 2].mean() - G0))

    dt = 1.0 / fs
    variants = [
        ("A 무보정 (공칭 중력만)", aE - np.array([0.0, 0.0, G0]), False),
        ("B + 영점 바이어스 보정",  aE - bias,                    False),
        ("C + ZUPT (정지->정지)",   aE - bias,                    True),
    ]

    print("\n실제로는 **안 움직인** 구간이므로 아래 값은 전부 오차다 (0 이 나와야 정답).")
    print("구간마다 시작점을 옮겨 가며 여러 번 평가한 뒤 중앙값 / 최악값을 낸다.\n")
    print("  %-24s %8s %12s %12s" % ("보정 단계", "창 [s]", "3D 오차 중앙", "3D 오차 최악"))
    print("  " + "-" * 60)

    for name, a_lin, zupt in variants:
        for W in args.windows:
            n = int(W * fs)
            if n < 5 or (len(a_lin) - nc) < n * 2:
                continue
            starts = np.arange(nc, len(a_lin) - n, max(n // 2, 1))
            errs = []
            for i in starts:
                p = integrate(a_lin[i:i + n], dt, zupt)
                errs.append(np.linalg.norm(p[-1]))
            if not errs:
                continue
            errs = np.array(errs)
            print("  %-24s %8.1f %10.4g m %10.4g m"
                  % (name if W == args.windows[0] else "", W,
                     np.median(errs), errs.max()))
        print()

    print("읽는 법")
    print("  * A 는 스케일 오차가 t^2 로 증폭된 것이다 — 로봇 GT 없이도 예측되는 값이다.")
    print("  * B 는 정지 구간에서 바이어스를 재서 뺀 것. 온도·자세가 변하면 다시 어긋난다.")
    print("  * C 가 실용 상한이다. 이보다 잘 나올 수는 없고, 실제 이동에서는 여기에")
    print("    가속도계 스케일 오차 x 이동거리 (약 2.3 %) 가 더해진다.")
    print("  * 즉 100 mm 이동을 2 s 에 하면 예상 오차 = C(2 s) + 0.023 x 100 mm 수준이다.")


if __name__ == "__main__":
    main()
