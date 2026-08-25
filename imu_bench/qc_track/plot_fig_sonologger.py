#!/usr/bin/env python3
"""학회용 그림 — Sonologger 데이터 수집 원리.

    python3 plot_fig_sonologger.py

세 단(段)으로 읽는다.

  (a) 계측 프로브를 만든 순서 — 실측 → CAD → 프린트 쉘 → IMU 장착 (사진 4 장)
  (b) 수집 구조 — 두 스트림이 각자 다른 경로로 와서 **호스트 수신 시각(pc_unix)**
      하나로 묶인다
  (c) 시간축 — 레이트가 다른 두 스트림이 같은 축에 어떻게 놓이는지

숫자·경로·파일 이름은 전부 host/us_imu_collect.py 에 적힌 실제 값이다.
사진은 report/figures/assets/ 의 실물 사진이고, 초음파 프레임도 실제 수집 프레임이다.
"""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch,  # noqa: E402
                                Polygon, Ellipse)
from PIL import Image                                           # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(_HERE, "report", "figures", "assets")
OUT = os.path.join(_HERE, "report", "figures", "fig_sonologger_acquisition")

INK, SUB = "#1a1f26", "#5a6068"
GREY_FILL, GREY_EDGE = "#F1F3F5", "#A5ABB2"
BLUE_FILL, BLUE_EDGE = "#E6F0F8", "#2E6E96"
ORNG_FILL, ORNG_EDGE = "#FDF1E5", "#C9711F"
BLUE, ORANGE, GREY = "#2E6E96", "#D9822B", "#8A9199"


def font():
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for fam in ("Arial", "Helvetica", "Inter", "Liberation Sans", "DejaVu Sans"):
        if fam in have:
            return fam
    return "DejaVu Sans"


def load(name, crop=None, rotate=0):
    """assets/ 사진을 열어 관심영역만 남긴다. crop 은 (x0, y0, x1, y1) 비율."""
    im = Image.open(os.path.join(ASSETS, name))
    if im.mode in ("RGBA", "LA", "P"):                 # 투명 배경을 흰색으로
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if crop:
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h),
                      int(crop[2] * w), int(crop[3] * h)))
    if rotate:
        im = im.rotate(rotate, expand=True)
    return np.asarray(im)


def photo_ax(fig, box, img, title, sub=None, frame=True):
    """[x0, y0, x1, y1] (figure fraction) 안에 사진을 비율 유지로 앉힌다."""
    x0, y0, x1, y1 = box
    W, H = fig.get_size_inches()
    bw, bh = (x1 - x0) * W, (y1 - y0) * H
    ar = img.shape[1] / img.shape[0]
    w, h = (bw, bw / ar) if bw / ar <= bh else (bh * ar, bh)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax = fig.add_axes([cx - w / 2 / W, cy - h / 2 / H, w / W, h / H])
    ax.imshow(img, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(frame)
        sp.set_color(GREY_EDGE); sp.set_linewidth(0.9)
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, fontweight="bold", pad=6)
    if sub:
        ax.text(0.5, -0.055, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=SUB)
    return ax


def box(ax, x, y, w, h, fill, edge, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=fill, ec=edge, lw=lw, zorder=2))


def arrow(ax, p0, p1, color=GREY, lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                 zorder=4))


def txt(ax, x, y, s, size=9.5, color=INK, weight="normal", ha="left", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            zorder=5)


def monitor_icon(ax, x, y, w, h):
    """임상용 초음파 모니터 도식 — 베젤·화면·받침. x 와 y 의 눈금이 달라 비율을
    눈으로 맞춘다 (이 축은 등축이 아니다)."""
    ax.add_patch(FancyBboxPatch((x, y + 0.22 * h), w, 0.78 * h,
                                boxstyle="round,pad=0,rounding_size=0.3",
                                fc="#2B2F34", ec="#2B2F34", lw=0.8, zorder=4))
    sx, sy = x + 0.09 * w, y + 0.30 * h
    sw, sh = 0.82 * w, 0.62 * h
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="square,pad=0",
                                fc="#0A0A0A", ec="none", zorder=5))
    fan = [(sx + 0.44 * sw, sy + sh), (sx + 0.56 * sw, sy + sh),
           (sx + 0.80 * sw, sy + 0.18 * sh), (sx + 0.20 * sw, sy + 0.18 * sh)]
    ax.add_patch(Polygon(fan, closed=True, fc="#8C9298", ec="none", zorder=6))
    ax.add_patch(Ellipse((sx + 0.5 * sw, sy + 0.46 * sh), 0.30 * sw, 0.28 * sh,
                         fc="#0A0A0A", ec="#E8E8E8", lw=0.7, zorder=7))
    ax.add_patch(FancyBboxPatch((x + 0.42 * w, y + 0.08 * h), 0.16 * w, 0.16 * h,
                                boxstyle="square,pad=0", fc="#6E7378", ec="none",
                                zorder=4))
    ax.add_patch(FancyBboxPatch((x + 0.28 * w, y + 0.02 * h), 0.44 * w, 0.07 * h,
                                boxstyle="square,pad=0", fc="#6E7378", ec="none",
                                zorder=4))


# ---------------------------------------------------------------- (b) 구조도
def draw_architecture(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    yb, yo = 74, 26                       # IMU 레인 / US 레인 중심
    hb = 30

    # 센서
    box(ax, 18, yb - hb / 2, 24, hb, BLUE_FILL, BLUE_EDGE)
    txt(ax, 20, yb + 10, "BNO085 IMU (9-axis)", 11, INK, "bold")
    txt(ax, 20, yb + 2.5, "accelerometer + gyroscope   ~200 Hz", 9.5, INK)
    txt(ax, 20, yb - 3.5, "rotation vector + magnetometer   ~100 Hz", 9.5, INK)
    txt(ax, 20, yb - 10, "host fusion uses 6 axes (magnetometer off)", 9, SUB)

    box(ax, 18, yo - hb / 2, 18, hb, ORNG_FILL, ORNG_EDGE)
    txt(ax, 19.2, yo + 9, "GE 4C-RS convex array probe", 10, INK, "bold")
    txt(ax, 19.2, yo + 1.5, "Connected to GE Voluson P6", 9.2, INK)
    txt(ax, 19.2, yo - 4.5, "Ultrasound display output", 9.2, INK)

    # 전송 경로
    box(ax, 47, yb - 9, 19, 18, "white", BLUE_EDGE)
    txt(ax, 48.5, yb + 3.4, "USB serial", 10.5, INK, "bold")
    txt(ax, 48.5, yb - 2.2, "/dev/ttyACM0", 9.5, INK)
    txt(ax, 48.5, yb - 6.6, "115 200 baud", 9, SUB)

    box(ax, 39, yo - hb / 2, 15, hb, "white", ORNG_EDGE)
    txt(ax, 40.1, yo + 12, "GE Voluson P6 monitor", 10, INK, "bold")
    txt(ax, 40.1, yo + 7, "Probe image display", 9.0, INK)
    txt(ax, 40.1, yo + 2.6, "HDMI video output", 9.0, INK)
    monitor_icon(ax, 43.0, yo - 13.5, 7.0, 13.0)

    box(ax, 57, yo - 13, 13, 26, "white", ORNG_EDGE)
    txt(ax, 58.1, yo + 9, "HDMI capture", 10, INK, "bold")
    txt(ax, 58.1, yo + 3, "HDMI input → USB video", 8.8, INK)
    txt(ax, 58.1, yo - 2.5, "Captured display frames", 8.8, INK)
    txt(ax, 58.1, yo - 7.5, "~8 fps", 8.8, SUB)

    ax.plot([0, 10], [50, 50], color=GREY, lw=1.6, zorder=3)
    ax.plot([10, 10], [yo, yb], color=GREY, lw=1.6, zorder=3)
    arrow(ax, (10, yb), (18, yb), GREY)
    arrow(ax, (10, yo), (18, yo), GREY)
    arrow(ax, (42, yb), (47, yb), BLUE)
    arrow(ax, (36, yo), (39, yo), ORANGE)
    arrow(ax, (54, yo), (57, yo), ORANGE)

    # 호스트 타임스탬프 — 두 레인을 하나로
    box(ax, 73, 8, 12, 84, GREY_FILL, GREY_EDGE)
    ax.text(79.0, 50, "host receive stamp\n" + r"$t \leftarrow$ time.time()"
            + "\n" + r"$\rightarrow$  pc_unix",
            fontsize=10.5, color=INK, fontweight="bold", ha="center", va="center",
            zorder=5, linespacing=1.6)
    arrow(ax, (66, yb), (73, yb), BLUE)
    arrow(ax, (70, yo), (73, yo), ORANGE)

    # 세션 저장소
    box(ax, 87, 8, 13, 84, "white", GREY_EDGE)
    txt(ax, 93.5, 86, "session store", 10.5, INK, "bold", ha="center")
    rows = [(80, "imu_<stamp>.csv", BLUE),
            (73.5, "imu_<stamp>.meta.json", BLUE),
            (64, "us_frames.bin", ORANGE),
            (57.5, "us_index.csv", ORANGE),
            (48, "session.meta.json", GREY)]
    for y, name, c in rows:
        ax.plot([88.4, 88.4], [y - 2.4, y + 2.4], color=c, lw=2.6, zorder=5,
                solid_capstyle="butt")
        txt(ax, 89.4, y, name, 9, INK)
    txt(ax, 89.4, 41, "us_index.csv columns", 8.8, SUB)
    txt(ax, 89.4, 36.5, "pc_unix, us_seq,", 8.8, INK)
    txt(ax, 89.4, 32.5, "frame_id, byte_offset", 8.8, INK)
    ax.plot([88.4, 99], [26, 26], color=GREY_EDGE, lw=0.8, zorder=5)
    txt(ax, 89.4, 20, "join key", 9, SUB)
    txt(ax, 89.4, 15, "pc_unix", 10.5, INK, "bold")
    arrow(ax, (85, 50), (87, 50), GREY)

    # 범례
    ax.plot([18, 22], [98, 98], color=BLUE, lw=2.4, zorder=5)
    txt(ax, 23, 98, "pose stream", 9.5, SUB)
    ax.plot([38, 42], [98, 98], color=ORANGE, lw=2.4, zorder=5)
    txt(ax, 43, 98, "image stream", 9.5, SUB)


# ---------------------------------------------------------------- (c) 시간축
def draw_timeline(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    x0, x1 = 8, 88
    ax.plot([x0, x1], [36, 36], color=INK, lw=1.2, zorder=3)
    ax.annotate("", xy=(x1 + 6, 36), xytext=(x1, 36),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    txt(ax, x1 + 7, 36, "pc_unix", 10, INK, "bold")

    # IMU 200 Hz — 눈금은 표시용으로 솎았다
    ticks = np.linspace(x0, x1, 77)
    ax.vlines(ticks, 40, 58, color=BLUE, lw=1.0, zorder=3)
    txt(ax, x0, 70, "IMU samples   ·   accel + gyro ~200 Hz   "
                    "(ticks thinned for display)", 9.5, INK)
    txt(ax, x0 - 2, 49, "IMU", 9, BLUE_EDGE, "bold", ha="right")

    # US 8 fps
    fx = np.linspace(x0 + 2.5, x1 - 2.5, 8)
    for x in fx:
        ax.add_patch(FancyBboxPatch((x - 1.8, 14), 3.6, 10,
                                    boxstyle="round,pad=0,rounding_size=0.7",
                                    fc=ORNG_FILL, ec=ORNG_EDGE, lw=1.1, zorder=3))
        ax.plot([x, x], [24, 36], color=ORNG_EDGE, lw=0.9, zorder=3)
    txt(ax, x0 - 2, 19, "US", 9, ORNG_EDGE, "bold", ha="right")
    txt(ax, x0, 4, "display frames   ·   ~8 fps   ·   timestamped on host arrival",
        9.5, INK)
    return fx


def build():
    fig = plt.figure(figsize=(16.6, 10.4))
    fig.patch.set_facecolor("white")
    W, H = fig.get_size_inches()

    # ---------------- (a) 계측 프로브 제작 -------------------------------
    fig.text(0.035, 0.928, "(a)", fontsize=12.5, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(0.062, 0.928, "Instrumented probe build", fontsize=12.5,
             fontweight="bold", color=INK, ha="left", va="center")

    # 배경을 지우지 않은 원본 사진을 쓴다 (관심영역만 크롭)
    cells = [
        ("photo_probe_survey.png", (0.27, 0.03, 1.00, 0.72), 0,
         "1  Dimensional survey",
         "GE 4C-RS convex array · steel rule + digital protractor"),
        ("photo_probe_cad.png", (0.26, 0.12, 0.98, 0.81), 0,
         "2  CAD reconstruction", "Fusion 360 mesh · ultrasound_probe v5"),
        ("photo_probe_shell.png", (0.14, 0.05, 0.86, 0.70), 90,
         "3  Printed clamp shell", "hinged split shell + sensor bracket"),
        ("photo_imu_mount.png", (0.14, 0.08, 0.86, 0.92), 0,
         "4  IMU on bracket", "BNO085 breakout (GY-BN008X), I²C to MCU"),
    ]
    x_lo, x_hi, gap = 0.035, 0.985, 0.020
    cw = (x_hi - x_lo - 3 * gap) / 4
    for i, (name, crop, rot, title, sub) in enumerate(cells):
        cx0 = x_lo + i * (cw + gap)
        # 제목·설명은 그림 좌표에 고정한다 — 사진마다 크기가 달라도 줄이 맞아야 한다
        photo_ax(fig, [cx0, 0.663, cx0 + cw, 0.872], load(name, crop, rot), None)
        fig.text(cx0 + cw / 2, 0.897, title, fontsize=10.5, color=INK,
                 fontweight="bold", ha="center", va="center")
        fig.text(cx0 + cw / 2, 0.646, sub, fontsize=9, color=SUB,
                 ha="center", va="center")
        if i < 3:
            fig.patches.append(FancyArrowPatch(
                (cx0 + cw + 0.0035, 0.768), (cx0 + cw + gap - 0.0035, 0.768),
                transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
                color=GREY, lw=1.6, shrinkA=0, shrinkB=0, zorder=6))

    # ---------------- (b) 수집 구조 --------------------------------------
    fig.text(0.035, 0.598, "(b)", fontsize=12.5, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(0.062, 0.598, "Acquisition architecture — two paths, one clock",
             fontsize=12.5, fontweight="bold", color=INK, ha="left", va="center")

    fr5 = load("photo_fr5_arm.png", (0.02, 0.02, 0.98, 1.00))
    ax_fr5 = photo_ax(fig, [0.035, 0.265, 0.135, 0.560], fr5, None, frame=False)
    ax_fr5.text(0.5, -0.045, "FR5 cobot  ·  192.168.58.3", transform=ax_fr5.transAxes,
                ha="center", va="top", fontsize=9.5, color=INK, fontweight="bold")
    ax_fr5.text(0.5, -0.095, "instrumented probe from (a) on the flange",
                transform=ax_fr5.transAxes, ha="center", va="top",
                fontsize=9, color=SUB)

    ax_b = fig.add_axes([0.150, 0.245, 0.835, 0.330]); draw_architecture(ax_b)
    fig.lines.append(plt.Line2D([0.138, 0.150], [0.410, 0.410],
                                transform=fig.transFigure, color=GREY, lw=1.6,
                                zorder=6))

    # ---------------- (c) 시간축 -----------------------------------------
    fig.text(0.035, 0.196, "(c)", fontsize=12.5, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(0.062, 0.196, "Common time base", fontsize=12.5, fontweight="bold",
             color=INK, ha="left", va="center")

    ax_c = fig.add_axes([0.035, 0.045, 0.575, 0.145]); fx = draw_timeline(ax_c)

    us = load("us_frame_example.png")
    ax_us = photo_ax(fig, [0.640, 0.052, 0.752, 0.180], us, None)
    ax_us.text(0.5, -0.06, "captured display frame",
               transform=ax_us.transAxes, ha="center", va="top", fontsize=9, color=SUB)
    fig.patches.append(FancyArrowPatch(
        (0.035 + 0.575 * fx[-1] / 100, 0.045 + 0.145 * 0.24), (0.637, 0.115),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=13,
        color=ORNG_EDGE, lw=1.3, shrinkA=2, shrinkB=2, zorder=6,
        connectionstyle="arc3,rad=-0.22"))

    fig.text(0.790, 0.183, "Arrival-time alignment", fontsize=10,
             color=INK, ha="left", va="top", fontweight="bold")
    fig.text(0.790, 0.152,
             "IMU:  USB serial\n"
             "Ultrasound:  monitor display → HDMI capture\n"
             "Host timestamps both streams at arrival\n"
             "Display and capture latency measured separately",
             fontsize=9.2, color=SUB, ha="left", va="top", linespacing=1.7)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    png, svg, pdf = OUT + ".png", OUT + ".svg", OUT + ".pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    try:
        Image.open(png).convert("RGB").save(png, dpi=(300, 300))
    except Exception:                                  # noqa: BLE001
        pass
    for p in (png, svg, pdf):
        print(f"  -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams["font.family"] = [font()]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["pdf.fonttype"] = 42
    build()
