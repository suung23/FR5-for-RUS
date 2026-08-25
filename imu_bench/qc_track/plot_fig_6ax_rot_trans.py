#!/usr/bin/env python3
"""학회 발표용 그림 — 6 축 IMU 한 대의 **회전 오차와 병진 오차** 한 장.

    python3 plot_fig_6ax_rot_trans.py

plot_track.py 의 진단용 배치를 학회용으로 다시 짠 것이다. 달라진 점만 적는다.

  · 자력계를 안 쓰는 host6 소스 하나만 그린다 (6 축 vs 9 축 비교는 다른 그림).
  · 판정 표는 뺀다 — 발표에서는 오차 자체를 보여주는 그림만 남긴다.
  · 라벨은 전부 영어, 색은 하늘색–회색 한 계열.

숫자와 시계열은 전부 raw_data 와 QC 결과 JSON 에서 읽는다. 손으로 적은 값은 없다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.gridspec import GridSpec                        # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "host"))

import analyze_track as at                                      # noqa: E402
import qc_common as qc                                          # noqa: E402
import reference as ref                                         # noqa: E402

# 하늘색–회색 한 계열. 회색 = 기준(로봇 GT), 하늘색 = IMU.
SKY_LT, SKY, SKY_DK = "#A7CFE6", "#4FA3D1", "#2E6E96"
GREY_LT, GREY, GREY_DK = "#C7CCD1", "#9AA1A9", "#666C74"
INK, SUB = "#1a1f26", "#5a6068"

SRC, BLOCK = "host6", "probe"
ROWS = ("tip_x", "tip_y", "spin")
YLAB = {"tip_x": "Tilt x [deg]", "tip_y": "Tilt y [deg]",
        "spin": "Axial rotation [deg]"}
OUT = os.path.join(_HERE, "report", "figures", "fig_6axis_rotation_translation")


def latin_font() -> str:
    """본문용 산세리프. 한글은 안 쓰지만 그리는 PC 에 있는 것을 골라 쓴다."""
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for fam in ("Arial", "Helvetica", "Inter", "Source Sans Pro", "Liberation Sans",
                "DejaVu Sans"):
        if fam in have:
            return fam
    return "DejaVu Sans"


def style(ax, *, grid_axis="both"):
    ax.grid(alpha=0.28, lw=0.5, color="#c3c8ce", axis=grid_axis)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8b9198")
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(labelsize=9, colors=INK, length=3.2, width=0.9)


def tag(ax, t, dx=-0.085, dy=1.06):
    ax.text(dx, dy, t, transform=ax.transAxes, fontsize=12, fontweight="bold",
            color=INK, ha="left", va="baseline")


def head(ax, t):
    ax.set_title(t, fontsize=9.5, loc="left", color=SUB, pad=5)


def solve_alignment(gt, imu):
    """분석기와 같은 순서로 정렬을 푼다 (plot_track.py 와 동일한 이유)."""
    A = B = None
    try:
        gt_a, imu_a = at.load_block("align", (SRC,))
        if gt_a is not None:
            A, B = at.alignment(gt_a, imu_a, SRC)[:2]
    except SystemExit:
        A = B = None
    if A is None:
        al = qc.load_json(qc.ALIGN_JSON)
        if not al or "A" not in al:
            raise SystemExit(f"정렬 결과가 없다: {qc.ALIGN_JSON}")
        A, B = np.array(al["A"]), np.array(al["B"])
    if SRC in at.PER_BLOCK_ALIGN:                # 6 축은 블록마다 요 원점이 다르다
        try:
            A, B = at.alignment(gt, imu, SRC)[:2]
        except SystemExit:
            pass
    return A, B


def build(res, out_base, zoom_s=2.5):
    gt, imu = at.load_block(BLOCK, (SRC,))
    if gt is None:
        raise SystemExit(f"블록 {BLOCK} 의 기록이 없다")
    A, B = solve_alignment(gt, imu)
    sn = at.add_channels(imu, SRC, A, B)

    dt = 1.0 / at.RESAMPLE_HZ
    lo = max(gt["t"][0], imu["t"][0]); hi = min(gt["t"][-1], imu["t"][-1])
    grid = np.arange(lo, hi, dt); t0 = grid[0]
    rot = res["rotation"][SRC]
    tr = res["translation"][BLOCK]
    fl = res["translation"]["still"]["error_floor_mm"]

    # 세 행이 같은 창을 보도록, 움직임이 가장 큰 자리를 고른다
    k = int(np.nanargmax(np.abs(np.gradient(
        np.nan_to_num(at.resample(gt["t"], gt["tip_x"], grid))))))
    zlo = grid[max(0, k - int(zoom_s / dt / 2))]; zhi = zlo + zoom_s

    fig = plt.figure(figsize=(15.0, 10.6))
    fig.patch.set_facecolor("white")
    gs = GridSpec(4, 3, figure=fig, width_ratios=[2.15, 1.0, 1.0],
                  height_ratios=[1.0, 1.0, 1.0, 1.22],
                  left=0.062, right=0.985, top=0.905, bottom=0.062,
                  wspace=0.26, hspace=0.46)

    # ---- 회전: 행 = 채널 ---------------------------------------------------
    for i, ch in enumerate(ROWS):
        g = at.resample(gt["t"], gt[ch], grid)
        s = at.resample(imu["t"], sn[ch], grid)
        c = rot["channels"][ch]
        lag = c["latency_ms"] / 1000.0
        sc = at.shift_apply(s, dt, lag) if np.isfinite(lag) else np.full_like(s, np.nan)

        a0 = fig.add_subplot(gs[i, 0])
        a0.plot(grid - t0, g, color=GREY, lw=2.2, label="Robot ground truth", zorder=2)
        a0.plot(grid - t0, s, color=SKY, lw=1.1, label="6-axis IMU", zorder=3)
        a0.axvspan(zlo - t0, zhi - t0, color=GREY_LT, alpha=0.55, lw=0, zorder=0)
        a0.set_ylabel(YLAB[ch], fontsize=10, color=INK)
        head(a0, f"latency {c['latency_ms']:+.0f} ms   ·   r = {c['corr']:.3f}   ·   "
                 f"RMS after latency correction {c['err_rms_delay_corrected_deg']:.2f}°")
        style(a0)
        if i == 0:
            a0.legend(loc="upper right", ncol=2, fontsize=9, frameon=False)
            a0.text(zhi - t0, a0.get_ylim()[0], "  zoom window of (b)", fontsize=8.5,
                    color=SUB, va="bottom", ha="left")
            tag(a0, "(a)")

        a1 = fig.add_subplot(gs[i, 1])
        m = (grid >= zlo) & (grid <= zhi)
        a1.plot(grid[m] - t0, g[m], color=GREY, lw=2.4)
        a1.plot(grid[m] - t0, s[m], color=SKY, lw=1.5)
        a1.plot(grid[m] - t0, sc[m], color=SKY_DK, lw=1.3, ls=(0, (4, 2.5)))
        style(a1)
        if i == 0:
            a1.plot([], [], color=SKY_DK, lw=1.3, ls=(0, (4, 2.5)),
                    label="IMU, latency-corrected")
            a1.legend(loc="lower right", fontsize=8.5, frameon=False)
            tag(a1, "(b)", dx=-0.20)

        a2 = fig.add_subplot(gs[i, 2])
        e = (sc - g)[np.isfinite(sc - g)]
        a2.hist(e, bins=60, color=SKY_LT, edgecolor=SKY_DK, lw=0.35, zorder=3)
        a2.axvline(0.0, color=GREY_DK, ls=(0, (4, 3)), lw=1.1, zorder=4)
        a2.set_ylabel("count", fontsize=9.5, color=INK)
        style(a2)
        rms = c["err_rms_delay_corrected_deg"]; p95 = c["err_p95_delay_corrected_deg"]
        # 오차 덩어리 반대쪽 구석에 적는다 — 히스토그램·0 선과 겹치지 않게
        right = np.nanmedian(e) < 0.5 * (a2.get_xlim()[0] + a2.get_xlim()[1])
        a2.text(0.925 if right else 0.045, 0.95, f"RMS {rms:.2f}°\np95 {p95:.2f}°",
                transform=a2.transAxes, ha="right" if right else "left", va="top",
                fontsize=9, color=INK,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))
        if i == 0:
            tag(a2, "(c)", dx=-0.22)

        if i == len(ROWS) - 1:
            a0.set_xlabel("Time [s]   (from block start)", fontsize=10, color=INK)
            a1.set_xlabel("Time [s]", fontsize=10, color=INK)
            a2.set_xlabel("Attitude error after latency correction [deg]",
                          fontsize=10, color=INK)

    # ---- 병진 (d) 구간별 변위 오차 ------------------------------------------
    d0 = fig.add_subplot(gs[3, 0])
    seg = tr["segments"]
    dist = [r["dist_mm"] for r in seg]
    for stage, color, mark, lab in (("C", SKY, "o", "C: zero-bias + ZUPT"),
                                    ("D", GREY_DK, "s", "D: C + calibration")):
        d0.scatter(dist, [r["eps_mm"][stage] for r in seg], s=34, marker=mark,
                   facecolor=color, edgecolor="white", lw=0.6, zorder=3, label=lab)
    d0.set_yscale("log")
    d0.set_xlabel("Segment travel distance [mm]", fontsize=10, color=INK)
    d0.set_ylabel("Displacement error [mm]", fontsize=10, color=INK)
    d0.legend(fontsize=9, frameon=False, loc="lower right")
    head(d0, f"per-ZUPT-segment displacement error   ·   n = {len(seg)} segments")
    style(d0)
    tag(d0, "(d)")

    # ---- 병진 (e) 처리 단계별 오차 바닥 -------------------------------------
    d1 = fig.add_subplot(gs[3, 1])
    FK = sorted(fl, key=float)                   # 키가 "0.5"/"1.0" 이라 f"{w:g}" 로 못 되돌린다
    W = [float(k) for k in FK]
    for stage, color, lab in (("A", GREY_LT, "A: raw integration"),
                              ("B", SKY_LT, "B: + zero-bias"),
                              ("C", SKY_DK, "C: + ZUPT")):
        d1.loglog(W, [fl[k][stage] for k in FK], "o-", color=color, lw=1.8,
                  ms=5.5, label=lab, zorder=3)
    d1.set_xlabel("Integration window T [s]", fontsize=10, color=INK)
    d1.set_ylabel("Error floor [mm]", fontsize=10, color=INK)
    d1.legend(fontsize=8.5, frameon=False, loc="upper left")
    head(d1, "error floor on still segments")
    style(d1)
    tag(d1, "(e)", dx=-0.24)

    # ---- 병진 (f) 실제 이동 vs 오차 바닥 ------------------------------------
    d2 = fig.add_subplot(gs[3, 2])
    fy = [fl[k]["C"] for k in FK]
    d2.loglog(W, fy, "o-", color=GREY, lw=1.8, ms=5.5, zorder=3,
              label="error floor (still)")
    MOVED_MM = 5.0                       # 실제로 움직인 칸만 '이동' 으로 본다
    mv = [(float(np.sqrt(b["lo"] * b["hi"])), b["eps_C_med_mm"])
          for b in tr["by_duration"] if b["dist_med_mm"] >= MOVED_MM]
    d2.loglog([m[0] for m in mv], [m[1] for m in mv], "s-", color=SKY_DK, lw=2.0,
              ms=6.5, zorder=4, label="actual motion (median)")
    tt = np.array([W[0] * 0.85, W[-1] * 1.2])
    d2.loglog(tt, fy[0] * (tt / W[0]) ** 2, ls=(0, (2, 2.5)), color=GREY_DK, lw=1.1,
              zorder=2, label=r"$T^{2}$ slope")
    d2.set_xlabel("Segment duration T [s]", fontsize=10, color=INK)
    d2.set_ylabel("Displacement error [mm]", fontsize=10, color=INK)
    d2.legend(fontsize=8.5, frameon=False, loc="upper left")
    head(d2, "motion error vs. error floor  (stage C)")
    style(d2)
    tag(d2, "(f)", dx=-0.24)

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    png, svg, pdf = out_base + ".png", out_base + ".svg", out_base + ".pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    try:                                          # 논문 규격: PNG 는 RGB
        from PIL import Image
        Image.open(png).convert("RGB").save(png, dpi=(300, 300))
    except Exception:                             # noqa: BLE001
        pass
    for p in (png, svg, pdf):
        print(f"  -> {p}")
    plt.close(fig)


def main() -> int:
    plt.rcParams["font.family"] = [latin_font()]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["pdf.fonttype"] = 42
    res = json.load(open(os.path.join(qc.DIR_META, f"qc_{SRC}.json")))
    build(res, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
