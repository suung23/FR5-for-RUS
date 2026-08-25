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

INK, SUB = "#000000", "#000000"      # 그림 안 글자는 전부 검정 — 위계는 굵기·크기로
NAVY, NAVY_DK = "#1F3A5C", "#12253C"
NAVY_TINT = "#EAF0F6"
GREY_EDGE, GREY_FILL, GREY = "#A8AEB6", "#F2F4F6", "#8A9199"
# 모듈 상자 — 직각·무채색. 흰 채움 = 처리/제어 블록, 아주 옅은 회색 = 묶음용.
BOX_EDGE, BOX_GROUP, BOX_SUB = "#4F5459", "#F5F5F5", "#7C8187"

FIG_W, FIG_H = 16.4, 10.2
AX = [0.022, 0.016, 0.958, 0.966]        # 본 축의 그림 내 위치


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

    L0, L1 = 0.5, 45.5                       # 왼쪽 열
    R0, R1 = 53.5, 92.5                      # 오른쪽 열

    t(ax, L0, 99.0, "(a)", 12.0, INK, "bold")
    t(ax, L0 + 3.4, 99.0, "Offline — learning from Sonologger sessions", 12.0, INK,
      "bold")
    t(ax, R0, 99.0, "(b)", 12.0, INK, "bold")
    t(ax, R0 + 3.4, 99.0, "Online — control stack on the robot", 12.0, INK, "bold")

    # ---------------- (a) 오프라인 -------------------------------------
    # 1. 두 입력
    block(ax, L0, 84.0, 22.2, 97.0, title="Ultrasound stream", online=False)
    t(ax, 14.0, 92.4, "256 × 256  @  8 fps", 8.8)
    t(ax, 14.0, 89.8, "raw candidate frames", 8.4, SUB)
    photo(fig, (L0 + 0.8, 84.8, 12.6, 94.2), "us_frame_wide.png")

    block(ax, 23.4, 84.0, L1, 97.0, title="Probe trajectory", online=False)
    t(ax, 25.6, 92.4, "IMU  ~200 Hz      or      robot FK  125 Hz", 8.8)
    t(ax, 25.6, 89.8, "pose of the probe during each session", 8.4, SUB)

    # 2. 시계 하나로 조인
    block(ax, L0, 77.5, L1, 82.5, title="Joined on a single clock", online=False)
    t(ax, 24.0, 78.7, "pc_unix  —  host receive time of both streams", 8.8,
      ha="center")
    flow(ax, (11.4, 84.0), (11.4, 82.5), color=GREY)
    flow(ax, (34.6, 84.0), (34.6, 82.5), color=GREY)

    # 3. ZUPT 괄호로 분절
    block(ax, L0, 64.0, L1, 76.0, title="Segmented into ZUPT brackets", online=False)
    by = 67.5
    spans = [(2.0, 5.0, "still\n0.3 s", "white"),
             (5.0, 20.5, "move\n1 – 1.5 s", "#E4E4E4"),
             (20.5, 24.0, "still\n0.3 s", "white")]
    for a, b, lab, fc in spans:
        ax.add_patch(FancyBboxPatch((a, by), b - a, 2.6, boxstyle="square,pad=0",
                                    fc=fc, ec=BOX_SUB, lw=0.9, zorder=3))
        ax.text((a + b) / 2, by + 1.3, lab, fontsize=7.8, color=INK, ha="center",
                va="center", zorder=4, linespacing=1.35)
    t(ax, 25.6, 71.6, "one bracket  =  one action chunk", 8.8)
    t(ax, 25.6, 69.2, "net displacement is observable;", 8.4, SUB)
    t(ax, 25.6, 67.1, "per-step velocity labels are not", 8.4, SUB)
    flow(ax, (11.4, 77.5), (11.4, 76.0), color=GREY)
    flow(ax, (34.6, 77.5), (34.6, 76.0), color=GREY)

    # 4. 관측 / 라벨
    block(ax, L0, 34.0, 22.2, 62.5, title="Observation  $o_t$", online=False)
    for i, s in enumerate(["16 raw frames  $I_{t-15..t}$",
                           "ControlState  ·  $Q_{seg}$",
                           "wrench  ·  roll · pitch",
                           "$\\tilde{V}_{t-1}$ achieved motion",
                           "$\\bar{F}_n^*$  ·  saturation flags"]):
        t(ax, 14.2, 57.6 - 2.6 * i, s, 8.4)
    photo(fig, (L0 + 0.8, 36.4, 13.0, 57.0), "us_frame_wide_seg.png")
    t(ax, L0 + 1.0, 35.0, "Slim U-Net  ·  fill = prediction, cyan = GT", 7.6, SUB)

    block(ax, 23.4, 34.0, L1, 62.5, title="Label  $A_t$", online=False)
    t(ax, 25.6, 57.6, "net displacement  $P_k$", 8.8)
    t(ax, 44.3, 57.6, "high weight", 8.2, SUB, ha="right")
    t(ax, 25.6, 54.8, "trajectory shape  $R_i$", 8.8)
    t(ax, 44.3, 54.8, "low weight", 8.2, SUB, ha="right")
    ax.plot([25.6, 45.8], [52.6, 52.6], color=GREY_EDGE, lw=0.8, zorder=3)
    t(ax, 25.6, 50.4, "one bracket  →  one action chunk,  k = 8", 8.6)
    flow(ax, (11.4, 64.0), (11.4, 62.5), color=GREY)
    flow(ax, (34.6, 64.0), (34.6, 62.5), color=GREY)

    # 5. 학습
    block(ax, L0, 12.5, L1, 32.5, title="ACT CVAE  +  $\\hat{Q}$ head  +  "
          "$\\hat{F}_n$ head", online=False)
    t(ax, L0 + 1.2, 27.4, "ACT predicts action chunks, image quality, and contact "
      "force.", 8.6, SUB)
    ax.text(L0 + 1.2, 22.4,
            r"$\mathcal{L} = \mathcal{L}_{\mathrm{action}}"
            r" + \lambda_Q\,\mathcal{L}_{\mathrm{quality}}"
            r" + \lambda_F\,\mathcal{L}_{\mathrm{force}}"
            r" + \lambda_c\,\mathcal{L}_{\mathrm{feasible}}"
            r" + \lambda_r\,\mathcal{L}_{\mathrm{risk}} + \beta\,\mathrm{KL}$",
            fontsize=9.6, color=INK, ha="left", va="center", zorder=5)
    t(ax, L0 + 1.2, 17.6, "chunk  k = 8  ·  heads predict $\\hat{Q}$ and "
      "$\\hat{F}_n$ alongside the action", 8.4, SUB)
    flow(ax, (11.4, 34.0), (11.4, 32.5), color=GREY)
    flow(ax, (34.6, 34.0), (34.6, 32.5), color=GREY)

    # θ — 학습에서 정책으로
    ax.add_patch(FancyArrowPatch((L1, 22.0), (R0, 53.0), arrowstyle="-|>",
                                 mutation_scale=14, color=NAVY, lw=2.2, shrinkA=0,
                                 shrinkB=0, zorder=6,
                                 connectionstyle="arc3,rad=-0.16"))
    ax.text(49.5, 27.5, "learned weights  θ", fontsize=8.4, color=INK, ha="center",
            va="center", zorder=7,
            bbox=dict(boxstyle="square,pad=0.28", fc="white", ec="none"))

    # ---------------- (b) 온라인 ---------------------------------------
    block(ax, R0, 86.0, R1, 97.0, freq="30 Hz", title="Sensing  ·  us_frame_node")
    t(ax, R0 + 1.2, 92.0, "US image topic     ~8 fps", 8.4, SUB)
    t(ax, R0 + 1.2, 89.6, "quality_raw_node   classical, segmentation-free", 8.6)
    t(ax, 76.0, 89.6, "→   $Q_{raw}$", 8.6)
    t(ax, R0 + 1.2, 87.3, "perception_node   Slim U-Net → ControlState", 8.6)
    t(ax, 76.0, 87.3, "→   $Q_{seg}$ · center error", 8.6)

    block(ax, R0, 73.5, R1, 84.5, freq="0.2 Hz", title="Force setpoint  ·  "
          "force_search_node")
    t(ax, R0 + 1.2, 79.4, "Coarse-to-fine search with small background dithering.",
      8.6, SUB)
    ax.text(R0 + 1.2, 76.0, r"$F_n^* = \arg\max_F\ \bar{Q}_{raw}(F)$",
            fontsize=10.2, color=INK, ha="left", va="center", zorder=5)

    block(ax, R0, 63.0, R1, 72.0, freq="20 Hz", title="Supervisor")
    st = ["SEARCH", "STAGE1A", "STAGE1B", "TRACK"]
    w, gap = 5.9, 1.40
    sx = R0 + 1.8
    for i, s in enumerate(st):
        ax.add_patch(FancyBboxPatch((sx, 66.0), w, 2.4, boxstyle="square,pad=0",
                                    fc="white", ec=BOX_SUB, lw=0.9, zorder=3))
        ax.text(sx + w / 2, 67.2, s, fontsize=7.4, color=INK, fontweight="bold",
                ha="center", va="center", zorder=4)
        if i < len(st) - 1:
            flow(ax, (sx + w, 67.2), (sx + w + gap, 67.2), lw=1.1)
        sx += w + gap
    ax.add_patch(FancyArrowPatch((sx - gap, 67.2), (sx, 67.2), arrowstyle="<|-|>",
                                 mutation_scale=10, color=GREY, lw=1.2, shrinkA=0,
                                 shrinkB=0, zorder=6))
    ax.add_patch(FancyBboxPatch((sx, 66.0), w, 2.4, boxstyle="square,pad=0",
                                fc=BOX_GROUP, ec=BOX_SUB, lw=0.9, zorder=3))
    ax.text(sx + w / 2, 67.2, "RECOVER", fontsize=7.4, color=INK, fontweight="bold",
            ha="center", va="center", zorder=4)
    t(ax, R0 + 1.2, 64.2, "an invalid control state or a sustained loss of image "
      "quality sends the stack back to search or recovery", 8.2, SUB)

    block(ax, R0, 47.5, R1, 61.5, freq="~5 Hz", title="Image axis  ·  policy_node  +  "
          "offset_filter")
    t(ax, R0 + 1.2, 56.6, "ACT selects an image-improving action chunk from raw "
      "ultrasound observations.", 8.6, SUB)
    ax.text(R0 + 1.2, 52.4, r"$V_i^* = \pi_\theta(o_t)$", fontsize=10.2, color=INK,
            ha="left", va="center", zorder=5)

    block(ax, R0, 30.5, R1, 46.0, freq="100 Hz", title="Force axis + arbitration  ·  "
          "admittance_node  +  QP")
    t(ax, R0 + 1.2, 41.2, "QP arbitration enforces joint-rate, force, and safety "
      "constraints.", 8.6, SUB)
    ax.text(R0 + 1.2, 37.4,
            r"$v_z = \mathrm{clamp}\left((F_n^*-F_n)/B_z,\ \pm v_{max}\right)$",
            fontsize=10.2, color=INK, ha="left", va="center", zorder=5)
    ax.text(R0 + 1.2, 33.2,
            r"$v^* = \arg\min_v\ \|W(s(v)-s^*)\|^2$"
            r"$\quad$ subject to safety constraints",
            fontsize=10.2, color=INK, ha="left", va="center", zorder=5)

    block(ax, R0, 22.0, R1, 29.0, freq="125 Hz", title="Servo  ·  servo_node")
    t(ax, R0 + 1.2, 25.2, "ServoJ execution with a watchdog.", 8.6, SUB)
    ax.text(R0 + 1.2, 23.2, r"$q \leftarrow q + \dot q \cdot dt$", fontsize=10.2,
            color=INK, ha="left", va="center", zorder=5)

    block(ax, R0, 8.5, R1, 20.5, title="Hardware")
    photo(fig, (R0 + 1.0, 9.2, R0 + 8.6, 17.4), "photo_fr5_arm.png", border=False)
    t(ax, R0 + 11.0, 15.0, "FR5 right arm  +  instrumented probe  +  PX6D force–torque sensor", 9.2,
      weight="bold")
    t(ax, R0 + 11.0, 12.6, "skin contact is the nominal state, not a fault", 8.6, SUB)

    # 층 사이 화살표
    for y0, y1, lab in ((86.0, 84.5, "$Q_{raw}$ · $Q_{seg}$ · ControlState"),
                        (73.5, 72.0, "$F_n^*$   (policy sees dither-free $\\bar F_n^*$)"),
                        (63.0, 61.5, "mode · image-axis gate"),
                        (47.5, 46.0, "$V_i^* = (v_x, v_y, \\omega_z)$   ZOH 20×"),
                        (30.5, 29.0, "$\\dot q$"),
                        (22.0, 20.5, "")):
        flow(ax, (R0 + 8.0, y0), (R0 + 8.0, y1))
        if lab:
            t(ax, R0 + 9.4, (y0 + y1) / 2, lab, 8.4, SUB)

    # 되먹임 — 오른쪽 여백을 타고 올라간다
    for gx, y_from, y_to, lab in ((94.4, 14.2, 38.0, "wrench"),
                                  (96.9, 14.2, 54.0, "$\\tilde V$  achieved motion")):
        ax.plot([R1, gx], [y_from, y_from], color=GREY, lw=1.3, ls=(0, (4, 2.5)),
                zorder=3)
        ax.plot([gx, gx], [y_from, y_to], color=GREY, lw=1.3, ls=(0, (4, 2.5)),
                zorder=3)
        flow(ax, (gx, y_to), (R1, y_to), dashed=True, color=GREY, lw=1.3)
        # 라벨은 화살촉 옆에 가로로 — 세로쓰기를 쓰지 않는다
        ax.text(R1 - 1.4, y_to, lab, fontsize=8.2, color=INK, ha="right",
                va="center", zorder=7,
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none"))

    # 범례 — 하드웨어 블록 아래 오른쪽 구석에 작게
    lx0, ly0, lx1, ly1 = 70.5, 0.6, R1, 7.2
    ax.add_patch(FancyBboxPatch((lx0, ly0), lx1 - lx0, ly1 - ly0,
                                boxstyle="square,pad=0", fc="white", ec="#D5D8DC",
                                lw=0.8, zorder=2))
    t(ax, lx0 + 1.0, 5.9, "Legend", 8.6, INK, "bold")
    rows = [("data and command", NAVY, "-", 1.4),
            ("robot feedback", GREY, (0, (4, 2.5)), 1.3),
            ("training path", NAVY, "-", 2.4)]
    for i, (desc, c, ls, lw) in enumerate(rows):
        y = 4.3 - 1.35 * i
        ax.plot([lx0 + 1.0, lx0 + 5.4], [y, y], color=c, lw=lw, ls=ls, zorder=4)
        t(ax, lx0 + 6.2, y, desc, 7.8, SUB)

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
