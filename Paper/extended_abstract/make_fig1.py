#!/usr/bin/env python3
"""Figure 1 — 시스템 구성과 임상적 위치.

**검증된 경로와 미래 경로를 색으로 가른다.** 남색 실선은 이 원고가 팬텀에서 잰
힘 제어 경로이고, 회색 파선은 아직 구현·검증되지 않은 영상/학습 경로다. 한 장에
같이 그리는 이유는 독자가 둘을 섞어 읽는 것이 이 그림이 막아야 할 오해이기
때문이다 — 분리해 두면 "그림에 있으니 된 것" 이 성립하지 않는다.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch   # noqa: E402

NAVY_DK, NAVY, NAVY_LT = "#1F3A5F", "#2E6E96", "#DCE6EF"
GREY_DK, GREY, GREY_LT = "#666C74", "#9AA1A9", "#EDEFF1"
INK, SUB = "#1A1F26", "#5A6068"

W, H = 17.4, 7.6          # cm 비례


def box(ax, x, y, w, h, title, lines, *, future=False, fill=None):
    edge = GREY if future else NAVY_DK
    face = fill if fill else (GREY_LT if future else NAVY_LT)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.12",
        linewidth=1.0 if not future else 0.9, edgecolor=edge, facecolor=face,
        linestyle="--" if future else "-", zorder=2))
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="top", fontsize=7.4,
            weight="bold", color=GREY_DK if future else INK, zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.62 - i * 0.27, line, ha="center", va="top",
                fontsize=6.4, color=SUB, zorder=3)


def arrow(ax, a, b, *, future=False, rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=9,
        linewidth=0.9 if future else lw,
        color=GREY if future else NAVY_DK,
        linestyle=(0, (4, 3)) if future else "-",
        connectionstyle=f"arc3,rad={rad}", zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(W / 2.54, H / 2.54))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # ── 임상 문맥 ─────────────────────────────────────────────────────────
    ax.text(0.15, H - 0.20, "Clinical context", fontsize=7.4, weight="bold",
            color=INK, va="top")
    box(ax, 0.15, H - 2.15, 3.05, 1.55, "HoLEP morcellation",
        ["suprapubic view", "bladder wall vs", "morcellator tip"], fill="#FFFFFF")

    # ── 검증된 힘 제어 경로 ───────────────────────────────────────────────
    ax.text(3.85, H - 0.20, "Validated here \u2014 contact-force regulation",
            fontsize=7.4, weight="bold", color=NAVY_DK, va="top")
    y = H - 2.15
    box(ax, 3.85, y, 3.05, 1.55, "Probe replica",
        ["3D-scanned from the", "clinical GE 4C-RS", "on the FR5 flange"])
    box(ax, 7.30, y, 3.05, 1.55, "PX6D F/T sensor",
        ["six-axis, 1 kHz", "gravity and payload", "compensated"])
    box(ax, 10.75, y, 3.05, 1.55, "Admittance loop",
        ["penetration axis only", "100 Hz on \u2016F\u2016", "10 mm/s, 0.2 rad/s"])
    box(ax, 14.20, y, 2.65, 1.55, "FR5 arm",
        ["6-DoF cobot", "servo 125 Hz", "probe placement"])

    for a, b in ((3.20, 3.85), (6.90, 7.30), (10.35, 10.75), (13.80, 14.20)):
        arrow(ax, (a, y + 0.77), (b, y + 0.77))

    # 되먹임 — 팔이 움직이면 접촉이 바뀌고 그것을 센서가 다시 본다
    arrow(ax, (15.5, y - 0.02), (8.85, y - 0.62), rad=-0.10)
    ax.text(12.4, y - 1.20, "contact force fed back each sample", fontsize=6.2,
            color=NAVY_DK, ha="center", style="italic")

    # 안전 감시자 — 학습·영상과 무관하게 독립
    box(ax, 3.85, y - 2.62, 13.0, 1.05,
        "Safety supervisor \u2014 independent of image and learning",
        ["warning 4.5 N   \u00b7   hard limit 5.0 N with forced retreat   "
         "\u00b7   command watchdogs"])
    arrow(ax, (14.6, y - 1.57), (14.6, y - 0.02))

    # ── 미래 경로 ─────────────────────────────────────────────────────────
    yf = 0.25
    ax.text(0.15, yf + 1.78, "Not validated here \u2014 future image-guided and "
            "learned components", fontsize=7.4, weight="bold", color=GREY_DK,
            va="bottom")
    box(ax, 0.15, yf, 3.05, 1.55, "Sonologger",
        ["ultrasound video and", "IMU orientation on", "one host clock"],
        future=True, fill="#FFFFFF")
    box(ax, 3.85, yf, 3.05, 1.55, "Bladder-lumen",
        ["segmentation", "not integrated with", "the control stack"], future=True)
    box(ax, 7.30, yf, 3.05, 1.55, "Image-guided",
        ["orientation control", "not implemented", "in the control stack"],
        future=True)
    box(ax, 10.75, yf, 3.05, 1.55, "Learned policy",
        ["from clinician", "demonstrations", "none trained"], future=True)
    for a, b in ((3.20, 3.85), (6.90, 7.30), (10.35, 10.75)):
        arrow(ax, (a, yf + 0.77), (b, yf + 0.77), future=True)
    arrow(ax, (13.80, yf + 0.77), (14.20, yf + 0.77), future=True)
    ax.text(14.35, yf + 0.77, "to the arm,\nalongside the\nforce loop",
            fontsize=6.2, color=GREY_DK, va="center", linespacing=1.5)

    fig.tight_layout(pad=0.12)
    for suffix in (".png", ".pdf"):
        fig.savefig(os.path.join(_HERE, "figures", "fig1_architecture" + suffix),
                    dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    print("figures/fig1_architecture.png")


if __name__ == "__main__":
    main()
