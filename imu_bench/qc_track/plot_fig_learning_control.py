#!/usr/bin/env python3
"""학회용 그림 — Sonologger 데이터가 어떻게 학습되어 제어 로직이 되는가.

    python3 plot_fig_learning_control.py

DESIGN_NOTES.md §3 '전체 아키텍처' 의 ASCII 도식을 그대로 옮긴 것이다. 왼쪽은
오프라인 학습, 오른쪽은 온라인 제어 스택(느린 루프 → 빠른 루프)이고 둘을 잇는
것은 학습 가중치 θ 하나다. 수치·기호·경고는 설계 노트에 적힌 값 그대로다.

사진 셋만 실물이다 — 초음파 프레임, Slim U-Net 예측 프레임, FR5.
"""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from PIL import Image                                           # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(_HERE, "report", "figures", "assets")
OUT = os.path.join(_HERE, "report", "figures", "fig_learning_to_control")

INK, SUB = "#141A21", "#5A6068"
NAVY, NAVY_DK = "#1F3A5C", "#12253C"
NAVY_TINT = "#EAF0F6"
GREY_EDGE, GREY_FILL, GREY = "#A8AEB6", "#F2F4F6", "#8A9199"
# 모듈 상자 — 직각·무채색. 흰 채움 = 처리/제어 블록, 아주 옅은 회색 = 묶음용.
BOX_EDGE, BOX_GROUP, BOX_SUB = "#4F5459", "#F5F5F5", "#7C8187"

FIG_W, FIG_H = 16.8, 11.2
AX = [0.028, 0.022, 0.944, 0.906]        # 본 축의 그림 내 위치


def font():
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for fam in ("Arial", "Helvetica", "Inter", "Liberation Sans", "DejaVu Sans"):
        if fam in have:
            return fam
    return "DejaVu Sans"


def fig_xy(x, y):
    """축 데이터 좌표(0..100) → 그림 좌표. 사진을 정확한 칸에 앉히는 데 쓴다."""
    return AX[0] + AX[2] * x / 100.0, AX[1] + AX[3] * y / 100.0


def photo(fig, dbox, name, border=True):
    x0, y0 = fig_xy(dbox[0], dbox[1])
    x1, y1 = fig_xy(dbox[2], dbox[3])
    im = Image.open(os.path.join(ASSETS, name))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    img = np.asarray(im.convert("RGB"))
    bw, bh = (x1 - x0) * FIG_W, (y1 - y0) * FIG_H
    ar = img.shape[1] / img.shape[0]
    w, h = (bw, bw / ar) if bw / ar <= bh else (bh * ar, bh)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax = fig.add_axes([cx - w / 2 / FIG_W, cy - h / 2 / FIG_H, w / FIG_W, h / FIG_H])
    ax.imshow(img, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(border); sp.set_color(GREY_EDGE); sp.set_linewidth(0.8)
    return ax


# --------------------------------------------------------------- 그리기 도구
def _text_width(txt, size, weight):
    """텍스트 폭을 데이터 단위로 잰다 — 굵은 주파수 뒤에 제목을 정확히 붙이려고."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    prop = FontProperties(family=plt.rcParams["font.family"][0], weight=weight,
                          size=size)
    w_pt = TextPath((0, 0), txt, size=size, prop=prop).get_extents().width
    return w_pt / (AX[2] * FIG_W * 72.0 / 100.0)


def block(ax, x0, y0, x1, y1, *, freq=None, title=None, online=True, lw=1.0):
    """모듈 상자 하나. 직각 모서리, 중성 회색 테두리, 흰색(또는 옅은 회색) 채움."""
    fc = "white" if online else BOX_GROUP
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="square,pad=0",
                                fc=fc, ec=BOX_EDGE, lw=lw, zorder=2))
    tx, ty = x0 + 1.2, y1 - 1.85
    if freq:
        ax.text(tx, ty, freq, fontsize=10.4, color=INK, fontweight="bold",
                ha="left", va="center", zorder=4)
        tx += _text_width(freq, 10.4, "bold") + 1.1
    if title:
        ax.text(tx, ty, title, fontsize=10.4, color=INK,
                fontweight="bold" if not freq else "normal",
                ha="left", va="center", zorder=4)


def t(ax, x, y, s, size=8.6, color=INK, weight="normal", ha="left", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            zorder=5)


def flow(ax, p0, p1, label=None, dashed=False, color=NAVY, lw=1.5, rad=0.0,
         lab_dx=0.6, lab_ha="left"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=6,
                                 linestyle=(0, (4, 2.5)) if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + lab_dx, (p0[1] + p1[1]) / 2, label, fontsize=8.4,
                color=SUB, ha=lab_ha, va="center", zorder=7,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))


def warn(ax, x, y, s, size=8.2):
    ax.text(x, y, "▲  " + s, fontsize=size, color=NAVY_DK, ha="left", va="center",
            zorder=5, style="italic")


# ------------------------------------------------------------------- 본 그림
def build():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes(AX); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    L0, L1 = 0.5, 46.5                       # 왼쪽 열
    R0, R1 = 53.5, 94.2                      # 오른쪽 열

    fig.text(0.028, 0.906, "(a)", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.049, 0.906, "Offline — learning from Sonologger sessions",
             fontsize=12.5, fontweight="bold", color=INK)
    x_r, _ = fig_xy(R0, 0)
    fig.text(x_r, 0.906, "(b)", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(x_r + 0.021, 0.906, "Online — control stack on the robot",
             fontsize=12.5, fontweight="bold", color=INK)

    # ---------------- (a) 오프라인 -------------------------------------
    # 1. 두 입력
    block(ax, L0, 84.0, 22.6, 96.0, title="Ultrasound stream", online=False)
    t(ax, 10.6, 91.0, "256 × 256  @  8 fps", 8.8)
    t(ax, 10.6, 88.4, "raw candidate frames", 8.4, SUB)
    photo(fig, (L0 + 0.7, 84.8, 9.6, 90.4), "us_frame_example.png")

    block(ax, 24.4, 84.0, L1, 96.0, title="Probe trajectory", online=False)
    t(ax, 25.4, 91.0, "IMU  ~200 Hz      or      robot FK  125 Hz", 8.8)
    t(ax, 25.4, 88.4, "pose of the probe during each session", 8.4, SUB)

    # 2. 시계 하나로 조인
    block(ax, L0, 76.0, L1, 82.4, title="Joined on a single clock", online=False)
    t(ax, 24.0, 78.4, "pc_unix  —  host receive time of both streams", 8.8,
      ha="center")
    flow(ax, (11.3, 84.0), (11.3, 82.4), color=GREY)
    flow(ax, (35.5, 84.0), (35.5, 82.4), color=GREY)

    # 3. ZUPT 괄호로 분절
    block(ax, L0, 63.5, L1, 74.4, title="Segmented into ZUPT brackets", online=False)
    bx0, bx1, by = 2.0, 24.0, 66.8
    ax.plot([bx0, bx1], [by, by], color=GREY_EDGE, lw=1.0, zorder=3)
    spans = [(2.0, 5.0, "still\n0.3 s", "white"), (5.0, 20.5, "move\n1 – 1.5 s", "#E4E4E4"),
             (20.5, 24.0, "still\n0.3 s", "white")]
    for a, b, lab, fc in spans:
        ax.add_patch(FancyBboxPatch((a, by), b - a, 2.6, boxstyle="square,pad=0",
                                    fc=fc, ec=BOX_SUB, lw=0.9, zorder=3))
        ax.text((a + b) / 2, by + 1.3, lab, fontsize=7.8, color=INK, ha="center",
                va="center", zorder=4, linespacing=1.35)
    t(ax, 25.4, 70.4, "one bracket  =  one action chunk", 8.8)
    t(ax, 25.4, 68.0, "net displacement is observable;", 8.4, SUB)
    t(ax, 25.4, 65.9, "per-step velocity labels are not", 8.4, SUB)

    # 4. 관측 / 라벨
    block(ax, L0, 42.5, 22.6, 61.8, title="Observation  $o_t$", online=False)
    for i, s in enumerate(["16 raw frames  $I_{t-15..t}$",
                           "ControlState  ·  $Q_{seg}$",
                           "wrench  ·  roll · pitch",
                           "$\\tilde{V}_{t-1}$ achieved motion",
                           "$\\bar{F}_n^*$  ·  saturation flags"]):
        t(ax, 10.4, 56.6 - 2.4 * i, s, 8.4)
    photo(fig, (L0 + 0.7, 45.6, 9.7, 55.6), "unet_seg_example.png")
    t(ax, L0 + 1.2, 44.0, "Slim U-Net  ·  fill = prediction, cyan = GT", 7.6, SUB)

    block(ax, 24.4, 42.5, L1, 61.8, title="Label  $A_t$", online=False)
    t(ax, 25.4, 57.4, "net displacement  $P_k$", 8.8)
    t(ax, 44.6, 57.4, "high weight", 8.2, SUB, ha="right")
    t(ax, 25.4, 54.6, "trajectory shape  $R_i$", 8.8)
    t(ax, 44.6, 54.6, "low weight", 8.2, SUB, ha="right")
    ax.plot([25.4, 44.6], [52.6, 52.6], color=GREY_EDGE, lw=0.8, zorder=3)
    t(ax, 25.4, 50.4, "one bracket  →  one action chunk,  k = 8", 8.6)
    flow(ax, (11.3, 63.5), (11.3, 61.8), color=GREY)
    flow(ax, (35.5, 63.5), (35.5, 61.8), color=GREY)

    # 5. 학습
    block(ax, L0, 22.0, L1, 40.5, title="ACT CVAE  +  $\\hat{Q}$ head  +  "
          "$\\hat{F}_n$ head", online=False)
    t(ax, L0 + 1.2, 35.4, "ACT predicts action chunks, image quality, and contact "
      "force.", 8.6, SUB)
    ax.text(L0 + 1.2, 30.6,
            r"$\mathcal{L} = \mathcal{L}_{\mathrm{action}}"
            r" + \lambda_Q\,\mathcal{L}_{\mathrm{quality}}"
            r" + \lambda_F\,\mathcal{L}_{\mathrm{force}}"
            r" + \lambda_c\,\mathcal{L}_{\mathrm{feasible}}"
            r" + \lambda_r\,\mathcal{L}_{\mathrm{risk}} + \beta\,\mathrm{KL}$",
            fontsize=9.6, color=INK, ha="left", va="center", zorder=5)
    flow(ax, (11.3, 42.5), (11.3, 40.5), color=GREY)
    flow(ax, (35.5, 42.5), (35.5, 40.5), color=GREY)

    # θ — 학습에서 정책으로
    ax.add_patch(FancyArrowPatch((L1, 32.0), (R0, 51.0), arrowstyle="-|>",
                                 mutation_scale=15, color=NAVY, lw=2.2, shrinkA=0,
                                 shrinkB=0, zorder=6,
                                 connectionstyle="arc3,rad=-0.14"))
    ax.text(49.9, 43.5, "θ", fontsize=13, color=NAVY, fontweight="bold",
            ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="circle,pad=0.30", fc="white", ec=NAVY, lw=1.2))
    ax.text(49.9, 39.4, "learned\nweights", fontsize=8.2, color=SUB, ha="center",
            va="center", zorder=7, linespacing=1.4)

    # ---------------- (b) 온라인 ---------------------------------------
    # R1 센싱
    block(ax, R0, 85.0, R1, 96.0, freq="30 Hz", title="Sensing  ·  us_frame_node")
    t(ax, R0 + 1.2, 91.0, "/us/image     ~8 fps", 8.4, SUB)
    t(ax, R0 + 1.2, 88.6, "quality_raw_node   classical, segmentation-free", 8.6)
    t(ax, R1 - 1.2, 88.6, "→  $Q_{raw}$", 8.6, ha="right")
    t(ax, R0 + 1.2, 86.3, "perception_node   Slim U-Net → ControlState", 8.6)
    t(ax, R1 - 1.2, 86.3, "→  $Q_{seg}$ · center error", 8.6, ha="right")

    # R2 힘 setpoint
    block(ax, R0, 72.0, R1, 83.0, freq="0.2 Hz", title="Force setpoint  ·  "
          "force_search_node")
    t(ax, R0 + 1.2, 78.0, "Coarse-to-fine search with small background dithering.",
      8.6, SUB)
    ax.text(R0 + 1.2, 74.6, r"$F_n^* = \arg\max_F\ \bar{Q}_{raw}(F)$",
            fontsize=10.2, color=INK, ha="left", va="center", zorder=5)

    # R3 상태머신
    block(ax, R0, 60.0, R1, 70.0, freq="20 Hz", title="Supervisor")
    st = ["SEARCH", "STAGE1A", "STAGE1B", "TRACK"]
    sx = R0 + 1.4
    for i, s in enumerate(st):
        w = 7.0
        ax.add_patch(FancyBboxPatch((sx, 63.9), w, 2.5, boxstyle="square,pad=0",
                                    fc="white", ec=BOX_SUB, lw=0.9, zorder=3))
        ax.text(sx + w / 2, 65.15, s, fontsize=7.8, color=INK, fontweight="bold",
                ha="center", va="center", zorder=4)
        if i < len(st) - 1:
            flow(ax, (sx + w, 65.15), (sx + w + 1.4, 65.15), lw=1.1)
        sx += w + 1.4
    ax.add_patch(FancyArrowPatch((sx - 1.4, 65.15), (sx, 65.15), arrowstyle="<|-|>",
                                 mutation_scale=10, color=GREY, lw=1.2, shrinkA=0,
                                 shrinkB=0, zorder=6))
    ax.add_patch(FancyBboxPatch((sx, 63.9), 8.0, 2.5, boxstyle="square,pad=0",
                                fc=BOX_GROUP, ec=BOX_SUB, lw=0.9, zorder=3))
    ax.text(sx + 4.0, 65.15, "RECOVER", fontsize=7.8, color=INK, fontweight="bold",
            ha="center", va="center", zorder=4)
    t(ax, R0 + 1.2, 62.0, "an invalid control state or a sustained loss of image "
      "quality sends the stack back to search or recovery", 8.2, SUB)

    # R4 영상축
    block(ax, R0, 44.0, R1, 58.0, freq="~5 Hz", title="Image axis  ·  policy_node  +  "
          "offset_filter")
    t(ax, R0 + 1.2, 53.2, "ACT selects an image-improving action chunk from raw "
      "ultrasound observations.", 8.6, SUB)
    ax.text(R0 + 1.2, 49.2, r"$V_i^* = \pi_\theta(o_t)$", fontsize=10.2, color=INK,
            ha="left", va="center", zorder=5)

    # R5 힘축 + QP
    block(ax, R0, 24.5, R1, 42.0, freq="100 Hz", title="Force axis + arbitration  ·  "
          "admittance_node  +  QP")
    t(ax, R0 + 1.2, 38.2, "QP arbitration enforces joint-rate, force, and safety "
      "constraints.", 8.6, SUB)
    ax.text(R0 + 1.2, 34.0,
            r"$v_z = \mathrm{clamp}\left((F_n^*-F_n)/B_z,\ \pm v_{max}\right)$",
            fontsize=10.2, color=INK, ha="left", va="center", zorder=5)
    ax.text(R0 + 1.2, 29.4,
            r"$v^* = \arg\min_v\ \|W(s(v)-s^*)\|^2$"
            r"$\quad$ subject to safety constraints",
            fontsize=10.2, color=INK, ha="left", va="center", zorder=5)

    # R6 서보
    block(ax, R0, 15.5, R1, 22.5, freq="125 Hz", title="Servo  ·  servo_node")
    t(ax, R0 + 1.2, 18.7, "ServoJ execution with a watchdog.", 8.6, SUB)
    ax.text(R0 + 1.2, 16.6, r"$q \leftarrow q + \dot q \cdot dt$", fontsize=10.2,
            color=INK, ha="left", va="center", zorder=5)

    # R7 하드웨어
    block(ax, R0, 2.0, R1, 13.5, title="Hardware")
    photo(fig, (R0 + 1.0, 2.6, R0 + 10.0, 11.4), "photo_fr5_arm.png", border=False)
    t(ax, R0 + 12.0, 9.4, "FR5 right arm  +  instrumented probe  +  PX6D F/T", 9.2,
      weight="bold")
    t(ax, R0 + 12.0, 7.0, "skin contact is the nominal state, not a fault", 8.6, SUB)

    # 층 사이 화살표
    for y0, y1, lab in ((85.0, 83.0, "$Q_{raw}$ · $Q_{seg}$ · ControlState"),
                        (72.0, 70.0, "$F_n^*$   (policy sees dither-free $\\bar F_n^*$)"),
                        (60.0, 58.0, "mode · image-axis gate"),
                        (44.0, 42.0, "$V_i^* = (v_x, v_y, \\omega_z)$   ZOH 20×"),
                        (24.5, 22.5, "$\\dot q$"),
                        (15.5, 13.5, "")):
        flow(ax, (R0 + 8.0, y0), (R0 + 8.0, y1))
        if lab:
            t(ax, R0 + 9.4, (y0 + y1) / 2, lab, 8.4, SUB)

    # 되먹임 — 오른쪽 여백을 타고 올라간다
    for gx, y_from, y_to, lab in ((96.4, 7.8, 33.0, "wrench"),
                                  (98.6, 7.8, 50.5, "$\\tilde V$  achieved motion")):
        ax.plot([R1, gx], [y_from, y_from], color=GREY, lw=1.3, ls=(0, (4, 2.5)),
                zorder=3)
        ax.plot([gx, gx], [y_from, y_to], color=GREY, lw=1.3, ls=(0, (4, 2.5)),
                zorder=3)
        flow(ax, (gx, y_to), (R1, y_to), dashed=True, color=GREY, lw=1.3)
        ax.text(gx, (y_from + y_to) / 2, lab, fontsize=8.2, color=SUB, rotation=90,
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))

    # 범례
    lx, ly = L0 + 0.4, 16.5
    ax.add_patch(FancyBboxPatch((lx, ly - 9.4), 45.0, 12.2, boxstyle="square,pad=0",
                                fc="white", ec=BOX_EDGE, lw=1.0, zorder=2))
    t(ax, lx + 1.2, ly + 1.0, "How to read", 9.6, INK, "bold")
    rows = [("solid navy", "command / data flow, slow loop → fast loop", NAVY, "-"),
            ("dashed grey", "feedback measured on the robot", GREY, (0, (4, 2.5))),
            ("thick navy", "training-only path — weights θ, computed offline", NAVY, "-")]
    for i, (name, desc, c, ls) in enumerate(rows):
        y = ly - 1.6 - 2.5 * i
        ax.plot([lx + 1.4, lx + 6.4], [y, y], color=c, lw=2.4 if i == 2 else 1.5,
                ls=ls, zorder=4)
        t(ax, lx + 7.4, y, desc, 8.5, SUB)

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
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    build()
