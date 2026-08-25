#!/usr/bin/env python3
"""학회용 그림 — 지령 twist 가 프로브–조직 운동에 어떻게 작용하는가.

    python3 plot_fig_twist_axes.py

근거는 docs/IMAGE_SERVOING_MATH.md §2·§4 와 DESIGN_NOTES §8.3 이다. 영상면 Π 를
불변으로 두는 부분군(면내)과 평면을 떠나보내는 여집합(면외)으로 6 축을 가르고,
축마다 화면에 무엇이 보이는지와 누가 그 축을 쥐는지를 적는다.

작은 그림은 전부 **모니터 한 대 + 검은 화면 + 흰 윤곽선**이다. 실제 영상이 아니라
도식이며(해부 증거를 지어내지 않는다), 방광 윤곽은 원이 아닌 닫힌 불규칙 곡선이다.
"""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch,  # noqa: E402
                                Polygon, Rectangle)

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "report", "figures", "fig_twist_probe_tissue")

INK, SUB = "#141A21", "#5A6068"
EDGE, GROUP_FILL, GHOST = "#4F5459", "#F5F5F5", "#A8AEB6"
BEZEL, STAND = "#2B2F34", "#6E7378"
TILE_BG, LINE, TRACE, MOVE = "#0A0A0A", "#FFFFFF", "#9AA0A6", "#E4E4E4"

FIG_W, FIG_H = 16.4, 9.6
AX = [0.022, 0.016, 0.958, 0.968]


def font():
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for fam in ("Arial", "Helvetica", "Inter", "Liberation Sans", "DejaVu Sans"):
        if fam in have:
            return fam
    return "DejaVu Sans"


def fig_xy(x, y):
    return AX[0] + AX[2] * x / 100.0, AX[1] + AX[3] * y / 100.0


def sub_ax(fig, dbox, xlim, ylim):
    x0, y0 = fig_xy(dbox[0], dbox[1])
    x1, y1 = fig_xy(dbox[2], dbox[3])
    a = fig.add_axes([x0, y0, x1 - x0, y1 - y0])
    a.set_aspect("equal"); a.axis("off")
    a.set_xlim(0, xlim); a.set_ylim(0, ylim)
    return a


def box(ax, x0, y0, x1, y1, fc="white", ec=EDGE, lw=1.0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="square,pad=0",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def t(ax, x, y, s, size=8.8, color=INK, weight="normal", ha="left", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            zorder=5)


def arrow(ax, p0, p1, color=INK, lw=1.5, rad=0.0, ms=10):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=6,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------- 화면과 윤곽선
def contour(cx, cy, rx, ry, rot=0.0, shape=0):
    """방광 단면 윤곽 — 원이 아니라 완만하게 불규칙한 닫힌 곡선.

    `shape` 가 다르면 다른 단면이다. 난수를 쓰지 않아 다시 그려도 같은 모양이다.
    """
    th = np.linspace(0, 2 * np.pi, 240)
    ph = 0.7 * shape
    r = (1.0 + 0.085 * np.sin(3 * th + ph) + 0.055 * np.cos(2 * th - 0.6 * ph)
         + 0.030 * np.sin(5 * th + 1.3 * ph))
    x, y = rx * r * np.cos(th), ry * r * np.sin(th)
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    return cx + c * x - s * y, cy + s * x + c * y


def monitor(a, x0, y0, w, h, stand=True):
    """모니터 한 대. 얇은 베젤 + 검은 화면 + 작은 받침. 화면 사각형을 돌려준다."""
    a.add_patch(Rectangle((x0, y0), w, h, fc=BEZEL, ec=BEZEL, lw=0.8, zorder=3))
    b = 0.055 * h + 0.035 * w
    a.add_patch(Rectangle((x0 + b, y0 + b), w - 2 * b, h - 2 * b, fc=TILE_BG,
                          ec="none", zorder=4))
    if stand:
        a.add_patch(Rectangle((x0 + w / 2 - 0.055 * w, y0 - 0.16 * h), 0.11 * w,
                              0.16 * h, fc=STAND, ec="none", zorder=2))
        a.add_patch(Rectangle((x0 + w / 2 - 0.19 * w, y0 - 0.23 * h), 0.38 * w,
                              0.075 * h, fc=STAND, ec="none", zorder=2))
    return x0 + b, y0 + b, w - 2 * b, h - 2 * b


def section(a, cx, cy, rx, ry, rot=0.0, shape=0, ghost=False):
    x, y = contour(cx, cy, rx, ry, rot, shape)
    if ghost:
        a.plot(x, y, color=TRACE, lw=1.0, ls=(0, (3, 2.2)), zorder=5)
    else:
        a.plot(x, y, color=LINE, lw=1.7, zorder=6)


def draw_cell(a, kind):
    """면내 3 축 · 면외 3 축의 화면. 면내는 같은 단면, 면외는 다른 단면."""
    sx, sy, sw, sh = monitor(a, 2.2, 0.62, 5.6, 2.90)
    cx, cy = sx + sw / 2, sy + sh / 2
    top = sy + sh - 0.24

    if kind == "v_x":                       # 같은 단면이 옆으로
        section(a, cx - 1.15, cy - 0.12, 0.92, 0.50, ghost=True)
        section(a, cx + 1.15, cy - 0.12, 0.92, 0.50)
        arrow(a, (cx - 0.70, top), (cx + 0.70, top), color=MOVE)
    elif kind == "v_z":                     # 같은 단면이 아래로
        section(a, cx, cy + 0.42, 0.92, 0.36, ghost=True)
        section(a, cx, cy - 0.42, 0.92, 0.36)
        arrow(a, (cx + 1.85, cy + 0.62), (cx + 1.85, cy - 0.62), color=MOVE)
    elif kind == "w_y":                     # 같은 단면이 화면 안에서 회전
        section(a, cx, cy - 0.12, 1.00, 0.48, ghost=True)
        section(a, cx, cy - 0.12, 1.00, 0.48, rot=34)
        arrow(a, (cx - 1.45, cy + 0.62), (cx + 1.45, cy + 0.66), color=MOVE, rad=-0.30)
    elif kind == "v_y":                     # 다른 단면
        section(a, cx, cy - 0.12, 1.00, 0.55, ghost=True)
        section(a, cx + 0.05, cy - 0.12, 0.66, 0.38, shape=2)
        arrow(a, (cx + 1.45, top - 0.05), (cx + 2.05, top - 0.55), color=MOVE)
    elif kind == "w_x":                     # 기울어 잘린 단면
        section(a, cx, cy - 0.12, 1.00, 0.55, ghost=True)
        section(a, cx + 0.08, cy - 0.16, 1.28, 0.40, rot=-13, shape=3)
        arrow(a, (cx + 1.75, cy + 0.72), (cx + 1.95, cy - 0.78), color=MOVE, rad=-0.28)
    else:                                   # ω_z — 빔축 둘레 회전
        section(a, cx, cy - 0.12, 1.00, 0.55, ghost=True)
        section(a, cx, cy - 0.12, 0.88, 0.64, rot=38, shape=4)
        arrow(a, (cx - 1.50, cy + 0.60), (cx + 1.50, cy + 0.60), color=MOVE, rad=-0.30)


# ------------------------------------------------- 프로브 좌표계 스케치
def draw_frame(a):
    a.add_patch(Rectangle((0.3, 0.4), 8.4, 4.2, fc="#FAFAFA", ec="none", zorder=1))
    a.plot([0.3, 8.7], [4.6, 4.6], color=SUB, lw=1.1, zorder=2)
    a.text(8.6, 4.18, "tissue", fontsize=8.2, color=SUB, ha="right", va="top")

    a.add_patch(Rectangle((3.6, 4.6), 2.0, 1.7, fc="white", ec=INK, lw=1.2, zorder=4))
    a.add_patch(Rectangle((3.85, 6.3), 1.5, 0.35, fc="white", ec=INK, lw=1.2, zorder=4))
    a.text(4.6, 5.42, "probe", fontsize=8.0, color=INK, ha="center", va="center",
           zorder=5)

    fan = np.array([[3.85, 4.6], [5.35, 4.6], [7.05, 1.25], [2.15, 1.25]])
    a.add_patch(Polygon(fan, closed=True, fc="none", ec=SUB, lw=1.0,
                        ls=(0, (3, 2.2)), zorder=3))
    a.text(4.6, 0.75, r"$\Pi = \{\, y_{probe} = 0 \,\}$", fontsize=8.8, color=INK,
           ha="center", va="center", zorder=5)

    o = np.array([4.6, 4.75])
    arrow(a, tuple(o), tuple(o + np.array([2.0, 0.0])), color=INK, ms=9)
    arrow(a, tuple(o), tuple(o + np.array([-1.35, 0.85])), color=INK, ms=9)
    arrow(a, (4.6, 4.6), (4.6, 3.15), color=INK, ms=9)
    a.text(6.75, 4.77, r"$+x$", fontsize=8.4, color=INK, ha="left", va="center")
    a.text(3.15, 5.72, r"$+y$", fontsize=8.4, color=INK, ha="right", va="center")
    a.text(4.85, 3.35, r"$+z$", fontsize=8.4, color=INK, ha="left", va="center")

    arrow(a, (7.2, 2.9), (10.0, 2.9), color=GHOST, lw=1.2, ms=9)
    sx, sy, sw, sh = monitor(a, 10.7, 1.45, 5.6, 3.3)
    section(a, sx + sw / 2, sy + sh / 2, 0.98, 0.66)
    a.text(13.5, 0.30, "ultrasound view", fontsize=8.2, color=SUB, ha="center",
           va="center")


def build():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes(AX); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---------------- (a) 좌표계와 분해 ---------------------------------
    box(ax, 0.5, 78.5, 99.5, 99.5, fc=GROUP_FILL)
    t(ax, 1.6, 97.4, "(a)", 11.0, INK, "bold")
    t(ax, 4.6, 97.4, "Probe frame / image plane", 11.0, INK, "bold")
    a = sub_ax(fig, (1.4, 79.4, 30.5, 95.9), 18.0, 7.0); draw_frame(a)

    t(ax, 34.0, 94.2, "Probe-frame twist", 9.4, INK, "bold")
    ax.text(34.0, 90.8, r"$V_{probe} = (\,v_x,\ v_y,\ v_z,\ \omega_x,\ \omega_y,"
            r"\ \omega_z\,)$", fontsize=10.0, color=INK, ha="left", va="center",
            zorder=5)
    ax.plot([34.0, 62.0], [88.2, 88.2], color=GHOST, lw=0.8, zorder=3)
    ax.text(34.0, 85.2, r"$\mathrm{Stab}(\Pi) = \mathrm{span}\{\,v_x,\ v_z,"
            r"\ \omega_y\,\}$", fontsize=9.8, color=INK, ha="left", va="center",
            zorder=5)
    ax.text(34.0, 82.0, r"$\mathrm{Off}(\Pi) = \mathrm{span}\{\,v_y,\ \omega_x,"
            r"\ \omega_z\,\}$", fontsize=9.8, color=INK, ha="left", va="center",
            zorder=5)

    t(ax, 70.0, 94.2, "Motion classes", 9.4, INK, "bold")
    t(ax, 70.0, 90.8, "In-plane:  same section", 9.2)
    t(ax, 70.0, 87.6, "Out-of-plane:  new section", 9.2)
    t(ax, 70.0, 84.4, "Metric depth", 9.2)

    # ---------------- (b) 면내 · (c) 면외 --------------------------------
    groups = [
        dict(y0=42.5, y1=76.5, tag="(b)   In-plane",
             rhs=r"$\mathrm{Stab}(\Pi) = \mathrm{span}\{v_x, v_z, \omega_y\}$",
             cells=[("$v_x$", "Lateral", "v_x", r"$\dot x_g = -\,v_x - \omega_y z_g$",
                     "Image policy", "Analytic"),
                    ("$v_z$", "Normal press", "v_z",
                     r"$\dot z_g = -\,v_z + \omega_y x_g$", "Admittance",
                     "Force control"),
                    ("$\\omega_y$", "In-plane rotation", "w_y",
                     r"$\dot\theta = \omega_y$", "Analytic",
                     r"$M_y \rightarrow 0$")]),
        dict(y0=7.5, y1=40.5, tag="(c)   Out-of-plane",
             rhs=r"$\mathrm{Off}(\Pi) = \mathrm{span}\{v_y, \omega_x, \omega_z\}$",
             cells=[("$v_y$", "Elevation", "v_y", "New section", "Image policy",
                     "No 1st-order gain"),
                    ("$\\omega_x$", "Tilt", "w_x", "Plane tilt", "Admittance",
                     r"$M_x \rightarrow 0$"),
                    ("$\\omega_z$", "Spin", "w_z", "Beam-axis spin", "Image policy",
                     r"Axis $\parallel v_y$")]),
    ]
    for g in groups:
        y0, y1 = g["y0"], g["y1"]
        box(ax, 0.5, y0, 99.5, y1)
        t(ax, 1.6, y1 - 2.2, g["tag"], 11.0, INK, "bold")
        ax.text(98.4, y1 - 2.2, g["rhs"], fontsize=9.6, color=INK, ha="right",
                va="center", zorder=5)
        for i, (sym, name, kind, eq, src, note) in enumerate(g["cells"]):
            x0 = 1.6 + i * 32.9
            cx = x0 + 15.0
            t(ax, cx, y1 - 6.0, sym, 12.5, INK, "bold", ha="center")
            t(ax, cx, y1 - 9.0, name, 9.4, INK, ha="center")
            a = sub_ax(fig, (x0, y0 + 8.6, x0 + 30.0, y1 - 10.6), 10.0, 3.9)
            draw_cell(a, kind)
            ax.text(cx, y0 + 6.6, eq, fontsize=9.8, color=INK, ha="center",
                    va="center", zorder=5)
            t(ax, cx, y0 + 4.0, src, 9.2, INK, "bold", ha="center")
            t(ax, cx, y0 + 1.8, note, 8.6, SUB, ha="center")

    # ---------------- (d) 축 배분 ---------------------------------------
    box(ax, 0.5, 0.5, 99.5, 5.5, fc=GROUP_FILL)
    t(ax, 1.6, 3.0, "(d)", 10.2, INK, "bold")
    t(ax, 4.6, 3.0, "Axis allocation", 10.2, INK, "bold")
    ax.text(24.0, 3.0, r"$V_{probe} = [\ v_x^{img},\ v_y^{img},\ v_z^{adm},"
            r"\ \omega_x^{adm},\ \omega_y^{adm},\ \omega_z^{img}\ ]$",
            fontsize=10.0, color=INK, ha="left", va="center", zorder=5)
    t(ax, 62.0, 3.0, "Image:  ~5 Hz (ZOH)   ·   Force:  100 Hz", 9.2, SUB)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    png, svg, pdf = OUT + ".png", OUT + ".svg", OUT + ".pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    try:
        from PIL import Image
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
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    build()
