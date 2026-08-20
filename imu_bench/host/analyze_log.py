#!/usr/bin/env python3
"""로그 한 개를 QC 지표로 환산한다. **외부 GT 가 필요 없는 항목만** 다룬다.

    python3 analyze_log.py ../logs/imu_*.csv

내는 것 (QC_PLAN 의 Exp 1 / §7.0 / §10 에 대응):
  1. 시간축 감사   — rate, 지터, 구멍, 디바이스시계 vs PC시계 정합. **이게 먼저다.**
                     시간축이 틀어진 데이터로 낸 노이즈는 노이즈가 아니라 정렬 오차다.
  2. 정적 안정성   — 자세 SD 와 드리프트를 **대역폭별로** 낸다. 대역폭 없는 "SD < X" 는
                     합격도 불합격도 마음대로 만들 수 있다 (QC_PLAN §10 대전제).
  3. Allan 편차    — 각도 랜덤워크(ARW)와 바이어스 안정도. 얼마나 오래 적분할 수 있는지가
                     여기서 나온다. 정지 로그만 있으면 되고 GT 가 필요 없다.
  4. 자기환경 감사 — |m| 과 자기복각의 안정도. 로봇 근처는 자성 환경이라 yaw 가 통째로
                     못 쓰게 될 수 있다 (QC_PLAN §7.1 이 미리 경고한 항목). 이게 나빠지면
                     9축을 버리고 6축으로 가야 한다는 뜻이므로, 융합 설계를 바꾼다.
  5. 위치 예산     — IMU 단독 사장추측이 얼마나 가는지. 관측이 아니라 **전파된 예산**이다.

정지 구간은 자이로 크기로 자동 판정한다. --window 로 직접 지를 수도 있다.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

G0 = 9.80665


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    if d.ndim == 0 or len(d) < 10:
        raise SystemExit("샘플이 너무 적습니다: %s" % path)
    meta_path = path.rsplit(".", 1)[0] + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return d, meta


def find_still(gyro, fs, min_s=5.0, thresh=0.01):
    """자이로 크기가 임계 아래인 가장 긴 연속 구간. 없으면 (None, None)."""
    still = np.linalg.norm(gyro, axis=1) < thresh
    idx = np.flatnonzero(np.diff(np.r_[0, still.view(np.int8), 0]))
    if len(idx) < 2:
        return None, None
    starts, ends = idx[::2], idx[1::2]
    k = int(np.argmax(ends - starts))
    if (ends[k] - starts[k]) / fs < min_s:
        return None, None
    return int(starts[k]), int(ends[k])


def block_sd(x, sec, fs):
    """sec 초 블록평균의 표준편차 — '대역폭을 밝힌 σ'."""
    n = max(1, int(sec * fs))
    m = len(x) // n
    return float(x[:m * n].reshape(m, n).mean(1).std()) if m >= 3 else float("nan")


def allan_dev(x, fs, n_tau=25):
    """Overlapping Allan deviation. 반환 (tau[s], sigma[x 단위])."""
    N = len(x)
    taus, sigmas = [], []
    for m in np.unique(np.geomspace(1, N // 5, n_tau).astype(int)):
        if m < 1 or N // m < 3:
            continue
        k = N // m
        blocks = x[:k * m].reshape(k, m).mean(1)
        diffs = np.diff(blocks)
        sigmas.append(float(np.sqrt(0.5 * np.mean(diffs ** 2))))
        taus.append(m / fs)
    return np.array(taus), np.array(sigmas)


def section(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def audit_timebase(d, fs):
    section("1/5  시간축 감사   — 이게 먼저다. 틀어진 시간축의 노이즈는 정렬 오차다")
    t = d["pc_unix"]
    dt = np.diff(t)
    dur = t[-1] - t[0]
    print("  샘플 %d,  길이 %.1f s,  실효 rate %.1f Hz" % (len(t), dur, len(t) / dur))
    print("  간격 중앙 %.2f ms,  p95 %.2f ms,  최대 %.1f ms"
          % (np.median(dt) * 1e3, np.percentile(dt, 95) * 1e3, dt.max() * 1e3))
    holes = int((dt > 3 * np.median(dt)).sum())
    print("  구멍(기대주기 3배 초과) : %d 개%s" % (holes, "" if holes == 0 else "   <-- 확인 필요"))

    # 디바이스 µs 시계 vs PC 시계. pc_ts = 진짜도착 + 지연 이고 지연은 항상 0 이상이라
    # 최소자승이 아니라 하단 포락선을 맞춘다 (QC_PLAN §8).
    dev = d["dev_us"] * 1e-6
    dev = dev - dev[0]
    pc = t - t[0]
    resid = pc - dev
    lo = np.percentile(resid, 1)          # 하단 포락선 대용
    keep = resid < np.percentile(resid, 20)
    slope, icept = np.polyfit(dev[keep], pc[keep], 1)
    r2 = pc - (slope * dev + icept)
    print("  디바이스시계 정합 : skew %+.1f ppm,  잔차 p95 %.2f ms  (하단 포락선 기준)"
          % ((slope - 1) * 1e6, np.percentile(np.abs(r2 - lo), 95) * 1e3))
    print("  * skew 는 정합으로 지워지지만 **고정 지연은 안 지워진다** — 그건 가진 프로브로만 잰다")


def static_stability(d, s, e, fs):
    section("2/5  정적 안정성   — 대역폭을 밝힌 σ (QC_PLAN §10 대전제)")
    seg = slice(s, e)
    t = d["pc_unix"][seg]
    dur = t[-1] - t[0]
    print("  정지 구간 %.1f s (%d 샘플)" % (dur, e - s))

    print("\n  %-22s %11s %11s %11s" % ("", "샘플당", "0.2 s 평균", "1 s 평균"))
    rows = [("host roll [deg]", d["host_roll"][seg]),
            ("host pitch [deg]", d["host_pitch"][seg]),
            ("host yaw [deg]", d["host_yaw"][seg])]
    if not np.all(np.isnan(d["chip_roll"][seg])):
        rows += [("chip roll [deg]", d["chip_roll"][seg]),
                 ("chip pitch [deg]", d["chip_pitch"][seg]),
                 ("chip yaw [deg]", d["chip_yaw"][seg])]
    rows += [("accel z [m/s²]", d["az"][seg]), ("gyro x [rad/s]", d["gx"][seg])]
    for nm, x in rows:
        x = np.asarray(x, dtype=float)
        print("  %-22s %11.5f %11.5f %11.5f"
              % (nm, x.std(), block_sd(x, 0.2, fs), block_sd(x, 1.0, fs)))

    print("\n  드리프트 (마지막 5 s 평균 - 처음 5 s 평균, %.1f s 창):" % dur)
    n5 = int(5 * fs)
    for nm in ("host_roll", "host_pitch", "host_yaw"):
        x = np.asarray(d[nm][seg], dtype=float)
        if len(x) < 2 * n5:
            continue
        drift = x[-n5:].mean() - x[:n5].mean()
        print("    %-12s %+7.3f deg   (%+.3f deg/min 환산)"
              % (nm, drift, drift / dur * 60))
    print("  * 30 s 창 드리프트를 2배 해 mm/min 을 만들면 노이즈가 크다. 진짜 rate 는")
    print("    10 분 이상 정지 로그로 따로 재야 한다 (QC_PLAN §7.2 추가 권장 항목).")


def allan(d, s, e, fs):
    section("3/5  Allan 편차   — 얼마나 오래 적분할 수 있나 (GT 불필요)")
    print("  %-10s %12s %12s %14s" % ("축", "ARW", "바이어스 안정도", "최적 τ"))
    for ax in ("gx", "gy", "gz"):
        x = np.asarray(d[ax][s:e], dtype=float)
        tau, sig = allan_dev(x, fs)
        if len(tau) < 3:
            continue
        # ARW: 기울기 -1/2 구간 -> sigma(1s) 를 deg/sqrt(hr) 로
        i1 = int(np.argmin(np.abs(tau - 1.0)))
        arw = np.rad2deg(sig[i1]) * 60.0          # deg/sqrt(hr)
        k = int(np.argmin(sig))
        bi = np.rad2deg(sig[k]) * 3600.0          # deg/hr
        print("  %-10s %9.4f°/√hr %9.3f °/hr %12.1f s" % (ax, arw, bi, tau[k]))
    print("  * 바이어스 안정도가 최소가 되는 τ 보다 오래 자유적분하면 오차가 다시 커진다.")
    print("    BNO085 는 칩이 온라인으로 바이어스를 빼므로 이 값이 매우 작게 나올 수 있다 —")
    print("    그건 센서가 좋다기보다 **이미 보정이 걸린 출력**이라는 뜻이다.")


def magnetic(d, s, e):
    section("4/5  자기환경 감사   — yaw 를 믿을 수 있나 (QC_PLAN §7.1 경고 항목)")
    mx, my, mz = (np.asarray(d[k][s:e], dtype=float) for k in ("mx_raw", "my_raw", "mz_raw"))
    if np.all(np.isnan(mx)):
        print("  MAG 레코드가 없습니다.")
        return
    m = np.vstack([mx, my, mz]).T
    norm = np.linalg.norm(m, axis=1)
    a = np.vstack([np.asarray(d[k][s:e], dtype=float) for k in ("ax", "ay", "az")]).T
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    incl = np.rad2deg(np.arcsin(np.clip(np.sum(an * m, axis=1) / norm, -1, 1)))

    print("  |m|      = %.2f ± %.2f µT       (지구 자기장은 보통 25~65 µT)"
          % (norm.mean(), norm.std()))
    print("  자기복각  = %+.2f ± %.2f deg      (한국 위도에서 대략 -53 deg)"
          % (incl.mean(), incl.std()))
    verdict = []
    if not (20.0 <= norm.mean() <= 70.0):
        verdict.append("|m| 이 지구 자기장 범위 밖 — 하드아이언 또는 강한 외부 자원")
    if norm.std() / max(norm.mean(), 1e-9) > 0.05:
        verdict.append("정지 중인데 |m| 이 흔들림 — 주변 자기장이 시간에 따라 변함")
    if abs(incl.std()) > 3.0:
        verdict.append("복각이 흔들림 — 자력계 기준이 불안정")
    if verdict:
        for v in verdict:
            print("  [경고] %s" % v)
        print("  -> yaw 를 자력계로 묶을 수 없으면 **9축을 버리고 6축(--no-mag)** 으로 간다.")
        print("     그러면 roll/pitch 는 절대값으로 남고 yaw 만 상대값이 된다.")
    else:
        print("  판정: 자기환경 양호 — 9축 MARG 로 yaw 를 절대값으로 쓸 수 있다.")


def position_budget(d, s, e, fs):
    section("5/5  위치 사장추측 예산   — 관측이 아니라 전파된 예산이다")
    a = np.vstack([np.asarray(d[k][s:e], dtype=float) for k in ("ax", "ay", "az")]).T
    g_meas = np.linalg.norm(a, axis=1).mean()
    scale_err = abs(g_meas - G0)
    print("  정지 시 |a| = %.4f m/s²  vs  9.80665  ->  스케일 오차 %+.2f %% (잔차 %.3f m/s²)"
          % (g_meas, 100 * (g_meas / G0 - 1), scale_err))
    noise1s = block_sd(a[:, 2], 1.0, fs)

    rows = [("가속도계 스케일 잔차", scale_err),
            ("자세오차 1.0° -> 중력누설", G0 * np.sin(np.deg2rad(1.0))),
            ("자세오차 0.1° -> 중력누설", G0 * np.sin(np.deg2rad(0.1))),
            ("accel 백색잡음 (1 s 평균)", noise1s)]
    print("\n  위치오차 = ½·a_err·t²")
    print("  %-28s %9s %8s %8s %8s %8s" % ("오차원", "a[m/s²]", "0.5 s", "1 s", "3 s", "10 s"))
    for nm, av in rows:
        print("  %-28s %9.4f %8s %8s %8s %8s"
              % (nm, av, *["%.3g m" % (0.5 * av * T * T) for T in (0.5, 1, 3, 10)]))
    print("\n  * IMU 단독으로는 위치가 관측되지 않는다. 외부 관측원(RCM 구속·ToF·비전) 없이는")
    print("    이 표가 곧 성능 상한이고, 실험을 더 해도 이 사실만 재확인한다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="imu_*.csv")
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"),
                    help="정지 구간을 자동판정 대신 직접 지정 [s]")
    ap.add_argument("--still-thresh", type=float, default=0.01,
                    help="정지 판정 자이로 임계 [rad/s]")
    args = ap.parse_args()

    d, meta = load(args.log)
    t = d["pc_unix"]
    fs = len(t) / (t[-1] - t[0])
    print("로그    : %s" % args.log)
    if meta:
        print("세션    : board=%s  use_mag=%s  mag_cal=%s"
              % (meta.get("board"), meta.get("use_mag"),
                 "있음" if meta.get("mag_cal") else "없음"))

    audit_timebase(d, fs)

    G = np.vstack([np.asarray(d[k], dtype=float) for k in ("gx", "gy", "gz")]).T
    if args.window:
        rel = t - t[0]
        s = int(np.searchsorted(rel, args.window[0]))
        e = int(np.searchsorted(rel, args.window[1]))
    else:
        s, e = find_still(G, fs, thresh=args.still_thresh)
    if s is None:
        print("\n정지 구간(>=5 s)을 찾지 못했습니다. --window 로 지정하거나,")
        print("센서를 가만히 둔 로그를 하나 받아 주세요.")
        return
    print("\n정지 구간 : %.1f ~ %.1f s (%.1f s)"
          % (t[s] - t[0], t[e - 1] - t[0], t[e - 1] - t[s]))

    static_stability(d, s, e, fs)
    allan(d, s, e, fs)
    magnetic(d, s, e)
    position_budget(d, s, e, fs)
    print()


if __name__ == "__main__":
    main()
