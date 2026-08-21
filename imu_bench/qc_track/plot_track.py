#!/usr/bin/env python3
"""한 장 요약 그림 — 행 = 채널, 색 = 신호의 출처.

    python3 plot_track.py
    python3 plot_track.py --run raw_data_sim --block probe

기준 보고서(Exp-Latency §5)와 **같은 읽는 법**을 쓴다.

  파랑  로봇 GT        주황  IMU        청록  IMU 를 제 지연만큼 당긴 것

청록이 파랑에 겹치면 차이의 정체가 오로지 시간이동이라는 뜻이고, 겹치지 않고
남는 부분이 대역폭 손실 또는 진짜 자세 오차다. 가운데 열은 **세 행이 같은 창**
이라 서로 다른 지연을 같은 자로 잰다.

마지막 행은 회전이 아니라 **병진**이다. 회전은 시계열로 겹쳐 그릴 수 있지만
병진은 그럴 수 없다 — IMU 변위는 구간 안에서만 뜻이 있기 때문이다. 그래서
"창 길이 -> 오차" 로 그리고, imu_bench 오차 바닥을 같이 얹는다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "host"))

import analyze_track as at                                      # noqa: E402
import qc_common as qc                                          # noqa: E402
import reference as ref                                         # noqa: E402

BLUE, ORANGE, TEAL, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#999999"

# 한글 폰트가 없는 기계에서 그리면 라벨이 전부 두부(口)가 된다. 그림이 깨진 것을
# 그림 문제로 오인하기 딱 좋으므로, 폰트가 없으면 **라벨을 영어로 바꾼다.**
_HANGUL_FONTS = ("NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR",
                 "Noto Sans CJK JP", "Noto Sans CJK SC", "Malgun Gothic", "AppleGothic")
KO = True


def pick_font():
    global KO
    have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for fam in _HANGUL_FONTS:
        if fam in have:
            matplotlib.rcParams["font.family"] = fam
            matplotlib.rcParams["axes.unicode_minus"] = False
            KO = True
            return fam
    KO = False
    matplotlib.rcParams["axes.unicode_minus"] = False
    print("  [주의] 한글 폰트가 없다 — 그림 라벨을 영어로 낸다."
          "  (sudo apt install fonts-nanum 이면 한글로 나온다)")
    return None


def T(ko, en):
    return ko if KO else en
ROWS = ("tip_x", "tip_y", "spin")
LABEL_KO = {"tip_x": "기울기 x [deg]", "tip_y": "기울기 y [deg]", "spin": "축둘레 회전 [deg]"}
LABEL_EN = {"tip_x": "tip x [deg]", "tip_y": "tip y [deg]", "spin": "spin about axis [deg]"}


def draw(res, block="probe", src="chip", zoom_s=2.5, out=None):
    gt, imu = at.load_block(block, (src,))
    if gt is None:
        raise SystemExit(f"블록 {block} 의 기록이 없다")
    # **정렬을 분석기와 똑같이 고른다.** 그림과 표가 다른 정렬을 쓰면 둘 다 못
    # 믿는다. 실제로 alignment.json (= 주 소스 chip 의 해) 을 host6 에 씌워
    # spin 이 260 deg 어긋난 그림이 나왔고, 같은 데이터로 분석기 표는 상관
    # 1.000 / RMS 1.95 deg 를 내고 있었다.
    #
    # 순서도 분석기와 같다 — (1) 이 소스로 align 블록에서 풀고, (2) PER_BLOCK_ALIGN
    # 소스면 이 블록 자신의 정지 자세로 다시 풀어 덮어쓴다. 6 축은 heading 이
    # 원리적으로 관측되지 않아 블록마다 요 원점이 다르기 때문이다.
    A = B = None
    try:
        gt_a, imu_a = at.load_block("align", (src,))
        if gt_a is not None:
            A, B = at.alignment(gt_a, imu_a, src)[:2]
    except SystemExit:
        A = B = None
    if A is None:                       # align 블록이 없거나 못 풀렸다
        al = qc.load_json(qc.ALIGN_JSON)
        if not al or "A" not in al:
            raise SystemExit(f"정렬 결과가 없다: {qc.ALIGN_JSON}")
        print(f"  [주의] align 블록에서 {src} 정렬을 못 풀어 alignment.json 을 쓴다"
              " — 그 파일은 **주 소스의 해**라 이 소스에 맞지 않을 수 있다.")
        A, B = np.array(al["A"]), np.array(al["B"])
    if src in at.PER_BLOCK_ALIGN:
        try:
            A, B = at.alignment(gt, imu, src)[:2]
        except SystemExit:
            pass                        # 분석기와 같이 align 블록 해로 남는다
    sn = at.add_channels(imu, src, A, B)

    dt = 1.0 / at.RESAMPLE_HZ
    lo = max(gt["t"][0], imu["t"][0]); hi = min(gt["t"][-1], imu["t"][-1])
    grid = np.arange(lo, hi, dt)
    t0 = grid[0]

    rot = (res.get("rotation") or {}).get(src) or {}
    lock = res.get("lockin") or {}
    tr = (res.get("translation") or {}).get(block) or {}

    fig, ax = plt.subplots(4, 3, figsize=(15.5, 11.0),
                           gridspec_kw={"width_ratios": [2.1, 1.0, 1.0]})
    fig.suptitle(T("IMU 추적 QC — FR5 teleop 프로빙   블록 ", "IMU tracking QC — FR5 teleop probing   block ")
                 + f"'{block}'  ·  {src}",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")

    # 가운데 열이 세 행 모두 같은 창이도록, 움직임이 큰 자리를 고른다
    probe_ch = at.resample(gt["t"], gt["tip_x"], grid)
    k = int(np.nanargmax(np.abs(np.gradient(np.nan_to_num(probe_ch)))))
    zlo = grid[max(0, k - int(zoom_s / dt / 2))]
    zhi = zlo + zoom_s

    for i, ch in enumerate(ROWS):
        g = at.resample(gt["t"], gt[ch], grid)
        s = at.resample(imu["t"], sn[ch], grid)
        c = rot.get("channels", {}).get(ch, {})
        lag = c.get("latency_ms", np.nan) / 1000.0
        sc = at.shift_apply(s, dt, lag) if np.isfinite(lag) else np.full_like(s, np.nan)

        a0 = ax[i][0]
        a0.plot(grid - t0, g, color=BLUE, lw=1.4, label=T("로봇 GT", "robot GT"))
        a0.plot(grid - t0, s, color=ORANGE, lw=1.1, label="IMU")
        a0.plot(grid - t0, sc, color=TEAL, lw=1.0, ls="--",
                label=T("IMU − 측정지연", "IMU − measured lag"))
        a0.axvspan(zlo - t0, zhi - t0, color=GREY, alpha=0.25, lw=0)
        a0.set_ylabel((LABEL_KO if KO else LABEL_EN)[ch])
        rr = ref.latency_row(ch)
        title = (T("지연 ", "lag ") + f"{c.get('latency_ms', float('nan')):.0f} ms   "
                 + T("상관 ", "corr ") + f"{c.get('corr', float('nan')):.3f}   "
                 + T("보정후 RMS ", "RMS after shift ")
                 + f"{c.get('err_rms_delay_corrected_deg', float('nan')):.2f} deg")
        if rr:
            title += T("      기준 ", "      ref ") + f"{rr['eff_ms']} ms / {rr['corr']}"
        a0.set_title(title, fontsize=9, loc="left")
        if i == 0:
            a0.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
            a0.text(0.005, 1.28, T("회색 띠 = 오른쪽 확대 구간", "grey band = zoom window"),
                    transform=a0.transAxes,
                    fontsize=8, color=GREY)

        a1 = ax[i][1]
        m = (grid >= zlo) & (grid <= zhi)
        a1.plot(grid[m] - t0, g[m], color=BLUE, lw=1.8)
        a1.plot(grid[m] - t0, s[m], color=ORANGE, lw=1.4)
        a1.plot(grid[m] - t0, sc[m], color=TEAL, lw=1.2, ls="--")
        a1.set_title(T("확대 — 세 행 모두 같은 창", "zoom — same window in all rows")
                     if i == 0 else "", fontsize=9, loc="left")

        a2 = ax[i][2]
        blk = {"tip_x": "sweep_tilt", "tip_y": "sweep_tilt", "spin": "sweep_roll"}[ch]
        d = lock.get(blk)
        if d and d.get("channel") == ch:
            f = [r["freq_hz"] for r in d["segments"]]
            y = [r["latency_ms"] for r in d["segments"]]
            a2.semilogx(f, y, "o-", color=ORANGE, label=T("이번 측정", "this run"))
            key = {"tip_x": "depression", "tip_y": "depression", "spin": "roll"}[ch]
            rb = ref.LOCKIN_MS.get(key, {})
            a2.semilogx(list(rb), list(rb.values()), "s--", color=GREY, label=T("기준", "reference"))
            fit = d.get("fit") or {}
            if fit.get("ok"):
                a2.set_title(f"L {fit['L'] * 1000:.0f} + T {fit['T'] * 1000:.0f} ms",
                             fontsize=9, loc="left")
            a2.set_xlabel(T("자극 주파수 [Hz]", "excitation freq [Hz]"))
            a2.set_ylabel(T("지연 [ms]", "lag [ms]"))
            if i == 0:
                a2.legend(fontsize=8, frameon=False)
        else:
            e = sc - g
            e = e[np.isfinite(e)]
            if e.size:
                a2.hist(e, bins=60, color=TEAL, alpha=0.85)
            a2.set_xlabel(T("지연보정 후 오차 [deg]", "error after lag shift [deg]"))
            a2.set_title(T("스크립트 자극 없음 — 오차 분포", "no scripted sweep — error histogram")
                         if i == 0 else "",
                         fontsize=9, loc="left")

    # --- 4 행: 병진 ------------------------------------------------------
    a0, a1, a2 = ax[3]
    rows = tr.get("segments", [])
    if rows:
        d = [r["dist_mm"] for r in rows]
        for stage, color, mark in (("C", TEAL, "o"), ("D", BLUE, "s")):  # noqa: B007
            e = [r["eps_mm"].get(stage, np.nan) for r in rows]
            if np.any(np.isfinite(e)):
                a0.scatter(d, e, s=26, color=color, marker=mark,
                           label=f"{stage} " + (at.STAGE_LABEL[stage] if KO else ""))
        a0.set_xlabel(T("구간 이동거리 [mm]", "segment distance [mm]"))
        a0.set_ylabel(T("변위 오차 [mm]", "displacement error [mm]"))
        a0.set_yscale("log"); a0.legend(fontsize=8, frameon=False)
        a0.set_title(T("ZUPT 구간별 변위 오차   n=", "per-ZUPT-segment error   n=") + str(len(rows)),
                     fontsize=9, loc="left")

    fl = tr.get("error_floor_mm") or {}
    if fl:
        W = sorted(float(k) for k in fl)
        for stage, color in (("A", "#d62728"), ("B", ORANGE), ("C", TEAL)):
            y = [fl[str(k) if str(k) in fl else str(int(k))].get(stage, np.nan) for k in W]
            a1.loglog(W, y, "o-", color=color, label=stage)
        rw = sorted(ref.DISPLACEMENT_FLOOR_MM)
        a1.loglog(rw, [ref.DISPLACEMENT_FLOOR_MM[k]["C"] for k in rw], "s--",
                  color=GREY, label=T("기준 C", "ref C"))
        a1.set_xlabel(T("창 길이 [s]", "window [s]"))
        a1.set_ylabel(T("오차 바닥 [mm]", "error floor [mm]"))
        a1.legend(fontsize=8, frameon=False, ncol=2)
        a1.set_title(T("정지 구간 오차 바닥", "error floor on still segments"), fontsize=9, loc="left")

    v = res.get("verdict", [])
    a2.axis("off")
    a2.text(0, 1.0, T("판정", "verdict"), fontweight="bold", fontsize=10, va="top")
    # monospace 로 정렬하지 않는다 — 시스템 mono 폰트에는 한글이 없어 두부가 된다.
    # 열을 따로 그려서 맞춘다.
    for j, item in enumerate(v):
        mark = ("-" if item["pass"] is None
                else (T("통과", "PASS") if item["pass"] else T("불합격", "FAIL")))
        val = T("미측정", "n/a") if item["value"] is None else f"{item['value']:.2f}"
        # 참고 행(용도 미확정이라 문턱이 없는 것)은 limit 이 None 이다. 예전에는
        # 여기서 그대로 포맷해 TypeError 로 그림 전체가 안 나왔다.
        lim = "—" if item.get("limit") is None else f"{item['limit']:g}"
        y = 0.90 - 0.098 * j
        col = "#d62728" if item["pass"] is False else "black"
        a2.text(0.00, y, item["name"][:24], fontsize=8, va="top", color=col)
        a2.text(0.80, y, f"{val} / {lim}", fontsize=8, va="top",
                ha="right", color=col)
        a2.text(1.00, y, mark, fontsize=8, va="top", ha="right", color=col)

    for row in ax:
        for a in row:
            a.grid(alpha=0.25, lw=0.5)
    ax[2][0].set_xlabel(T("시간 [s]  (블록 시작으로부터)", "time [s]  (from block start)"))
    ax[2][1].set_xlabel(T("시간 [s]", "time [s]"))
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    out = out or os.path.join(qc.DIR_META, "analysis", "qc_track")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out + ext, dpi=130)
    plt.close(fig)
    return out + ".png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--block", default="probe")
    ap.add_argument("--source", default="chip")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.run:
        qc.use_raw_dir(qc.resolve_run(args.run))
    res = qc.load_json(qc.RESULT_JSON)
    if res is None:
        raise SystemExit(f"먼저 analyze_track.py 를 돌릴 것 ({qc.RESULT_JSON} 이 없다)")
    pick_font()
    print("  ->", draw(res, args.block, args.source, out=args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
