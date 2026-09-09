#!/usr/bin/env python3
"""세션 하나를 학습 없이 점검한다 — 데이터가 들어오면 **가장 먼저** 돌리는 스크립트.

    python scripts/inspect_session.py <session_dir> [--set imu.still_gyro_sd=0.02 ...] [--plot out.png] [--latency]

보고 항목
  1. 시간축: US fps (⏳ 실측 — 여기서 잰 값이 timing.obs_frames 를 정한다), IMU Hz,
     dev_us↔pc_unix 시계 적합 (기울기·잔차), 프레임당 IMU 표본 수
  2. 정지 판정: 이동창 자이로/가속도 산포의 분위수 → 임계가 데이터 분포 어디에 놓이는지
  3. 분절: 정지 구간 수·길이, 이동 구간 수·길이, ZUPT 전 잔류 속도 (라벨 신뢰도)
  4. 라벨: 축별 순변위 분포, σ 분포, chunk 가 이동을 다 덮는 비율
  5. (옵션) --plot: 자이로/가속도 노름 + 정지 마스크 + 구간 앵커 그림
  6. (옵션) --latency: 영상 평균강도 급변 ↔ 가속도 스파이크 상호상관으로 timing.us_latency_s 추정 (§7.1).
     프로브를 젤에서 급히 떼었다 붙이는 사건이 10–20 회 있는 세션에서 돌린다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import add_config_args, config_from_args

from rus_policy.calibration import estimate_us_latency
from rus_policy.imu_labels import _rolling_mean_std, _window_bounds, label_session
from rus_policy.session import load_session


def pct(x, qs=(5, 25, 50, 75, 95, 99)):
    x = np.asarray(x, float)
    return {q: float(np.percentile(x, q)) for q in qs} if x.size else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("session_dir")
    parser.add_argument("--plot", default=None, help="PNG 로 저장")
    parser.add_argument("--latency", action="store_true", help="US 지연 추정 (프레임 전부 읽는다)")
    parser.add_argument("--dither-hz", type=float, default=0.5, help="m 재산정용 면외 dither 주파수 (§1.3)")
    args = parser.parse_args()
    cfg = config_from_args(args)

    s = load_session(args.session_dir)
    imu = s.imu
    print(f"세션 {s.name}: {s.summary()}")
    a, b = imu.clock
    resid = imu.t_pc - imu.dev_to_pc(imu.t_dev)
    print(f"  시계 적합 pc = {a:.3f} + {b:.9f}·dev   잔차 p95 {np.percentile(np.abs(resid), 95) * 1e3:.2f} ms  "
          f"skew {(b - 1) * 1e6:.1f} ppm")
    if s.n_frames > 1:
        fd = np.diff(s.frame_t_pc)
        print(f"  US 프레임 간격 중앙값 {np.median(fd) * 1e3:.1f} ms  p95 {np.percentile(fd, 95) * 1e3:.1f} ms  "
              f"최대 {fd.max() * 1e3:.1f} ms  → 프레임당 IMU 표본 ≈ {np.median(fd) * imu.rate_hz:.1f}")
        inside = (s.frame_t_pc >= imu.t_pc[0]) & (s.frame_t_pc <= imu.t_pc[-1])
        print(f"  IMU 시간 범위 안의 프레임 {inside.sum()}/{s.n_frames}")
        fps = s.us_fps
        m_needed = int(np.ceil(fps / args.dither_hz)) if np.isfinite(fps) and fps > 0 else None
        print(f"  ⏳ US fps 실측 {fps:.2f} → m = ceil(f_us / f_dither={args.dither_hz}) = {m_needed}"
              f"   (설정 timing.obs_frames = {cfg.timing.obs_frames}"
              f"{' ← 갱신 필요' if m_needed is not None and m_needed != cfg.timing.obs_frames else ' ✓'})")

    if args.latency and s.n_frames > 4:
        fr = np.asarray(s.frames)
        mean_i = fr.reshape(fr.shape[0], -1).mean(axis=1) / 255.0
        est = estimate_us_latency(s.frame_t_pc, mean_i, imu.t_pc, imu.acc)
        verdict = ("신뢰" if est.peak_corr > 0.15 and est.second_ratio > 1.5 and est.n_imu_events >= 5
                   else "애매 — 사건이 적거나 피크가 약함")
        print(f"
US 지연 추정: {est.latency_s * 1e3:+.0f} ms (양수 = 영상이 늦다)  피크 상관 {est.peak_corr:.3f}  "
              f"2위 대비 {est.second_ratio:.2f}  사건 영상 {est.n_image_events} / IMU {est.n_imu_events}  → {verdict}")
        print(f"  → --set timing.us_latency_s={est.latency_s:.3f}   (설정 현재값 {cfg.timing.us_latency_s})")

    # 정지 판정 분포
    dt = float(np.median(np.diff(imu.t_dev)))
    k = max(3, int(round(cfg.imu.still_win_s / dt)))
    lo, hi = _window_bounds(imu.n, k)
    _, g_sd = _rolling_mean_std(imu.gyr, lo, hi)
    gm_mean, _ = _rolling_mean_std(np.linalg.norm(imu.gyr, axis=1)[:, None], lo, hi)
    _, a_sd = _rolling_mean_std(imu.acc, lo, hi)
    print(f"\n정지 판정 (창 {cfg.imu.still_win_s} s = {k} 표본). 분위수 [5,25,50,75,95,99]:")
    for name, val, thr in (("gyro sd max [rad/s]", g_sd.max(1), cfg.imu.still_gyro_sd),
                           ("|gyro| mean [rad/s]", gm_mean[:, 0], cfg.imu.still_gyro_mean),
                           ("accel sd max [m/s2]", a_sd.max(1), cfg.imu.still_accel_sd)):
        p = pct(val)
        below = float(np.mean(val < thr))
        print(f"  {name:22s} " + " ".join(f"{p[q]:.4f}" for q in (5, 25, 50, 75, 95, 99))
              + f"   임계 {thr}  → 통과 {below * 100:.1f} %")

    labels = label_session(imu, cfg.timing, cfg.imu, cfg.labels,
                           convention=(s.zero_ref or {}).get("quat_convention"))
    d = labels.diagnostics
    print(f"\n분절: 정지 {d['n_still_segments']} 구간 (정지 비율 {d['still_fraction'] * 100:.1f} %), "
          f"이동 {d['n_move_segments']} 구간, 규약 {d['convention']}"
          f"{' (모호 — zero_ref 권장)' if d.get('convention_ambiguous') else ''}, "
          f"정지 중 |a_E| 평균 {d['gravity_check_m_s2']:.3f} m/s² (g=9.807)")
    if labels.stills:
        sl = [imu.t_dev[e] - imu.t_dev[b_] for b_, e in labels.stills]
        print(f"  정지 길이 [s]: " + " ".join(f"{v:.2f}" for v in pct(sl).values()))
    if labels.labels:
        L = labels.labels
        mv = [l.move_s for l in L]
        vend = [l.diagnostics["zupt_v_end_raw_mm_s"] for l in L]
        cover = np.mean([l.diagnostics["chunk_covers_move"] for l in L])
        P = np.array([l.P[-1] for l in L])
        sig = np.array([l.sigma_net for l in L])
        print(f"  이동 길이 [s]: " + " ".join(f"{v:.2f}" for v in pct(mv).values()))
        print(f"  ZUPT 전 잔류 속도 [mm/s]: " + " ".join(f"{v:.1f}" for v in pct(vend).values())
              + "   (수십 mm/s 이상이면 정지 임계가 너무 느슨하거나 바이어스가 큼)")
        print(f"  chunk({cfg.timing.chunk_horizon_s:.1f} s) 가 이동을 다 덮는 비율 {cover * 100:.0f} %")
        for j, name in enumerate(("Δx [mm]", "Δy [mm]", "Δθz [deg]")):
            print(f"  {name:10s} |중앙| {np.median(np.abs(P[:, j])):.2f}  p95 {np.percentile(np.abs(P[:, j]), 95):.2f}  "
                  f"σ 중앙 {np.median(sig[:, j]):.2f}  SNR 중앙 {np.median(np.abs(P[:, j]) / sig[:, j]):.1f}")
    else:
        print("  이동 구간이 없습니다 — 임계를 느슨하게 하거나 프로토콜(정지→이동→정지)을 확인하십시오.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = imu.t_dev - imu.t_dev[0]
        fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        ax[0].plot(t, np.linalg.norm(imu.gyr, axis=1), lw=0.5)
        ax[0].set_ylabel("|ω| [rad/s]")
        ax[1].plot(t, np.linalg.norm(imu.acc, axis=1) - 9.80665, lw=0.5)
        ax[1].set_ylabel("|a|−g [m/s²]")
        ax[2].fill_between(t, 0, labels.still.astype(float), step="mid", alpha=0.4, label="still")
        for l in labels.labels:
            ax[2].axvline(l.t_anchor_dev - imu.t_dev[0], color="g", lw=0.8)
            ax[2].axvline(l.t_end_dev - imu.t_dev[0], color="r", lw=0.8)
        ax[2].set_ylabel("still / anchors")
        ax[2].set_xlabel("t [s]")
        ax[2].legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"그림: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
