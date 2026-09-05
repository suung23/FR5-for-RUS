#!/usr/bin/env python3
"""Overview figure — the guidance architecture from frame to two readers.

    python3 Unet_seg/scripts/lateral_instruction_overview.py

Manuscript Figure 1. Three stages left to right — segmentation, quality
judgment, corrective action — ending in the two readers of the same
instruction: the human operator (language, displayed or spoken) and the
robotic probe holder (action token, translated to a twist by lookup).

The human-operator panel prefers a reference illustration if one exists at
``Paper/figures/assets/clinician.png`` (drop the approved artwork there and
re-run); otherwise it falls back to the built-in vector clinician.

Style follows imu_bench/qc_track/plot_fig_learning_control.py: rectangular
achromatic boxes, photographs are the only full-colour elements, numbers read
from lateral_instruction.json so the figure cannot drift from the analysis.
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import (                                 # noqa: E402
    FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle,
)
from PIL import Image                                            # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
FIGDIR = os.path.join(REPO, "Unet_seg", "experiments", "lateral_instruction")
ASSETS = os.path.join(REPO, "imu_bench", "qc_track", "report", "figures", "assets")
CLINICIAN_ART = os.path.join(REPO, "Paper", "figures", "assets", "clinician.png")
#: 사용자가 승인한 완성 일러스트가 있으면 그 파일을 그대로 fig_overview 로 쓴다.
FINAL_ART = os.path.join(REPO, "Paper", "figures", "assets", "fig_overview_final.png")
OUT = os.path.join(FIGDIR, "fig_overview")

S = json.load(open(os.path.join(FIGDIR, "lateral_instruction.json")))
SIGMA = S["sigma_px"]
HOLD_PX = S["thresholds"]["5%"]["px"]
HOLD_K = S["thresholds"]["5%"]["k"]

INK = "#000000"
NAVY = "#1F3A5C"
GREY_EDGE, GREY = "#A8AEB6", "#8A9199"
BOX_EDGE, BOX_GROUP = "#4F5459", "#F5F5F5"
SKIN = "#F2D8C3"
SUB = "#3A3F45"
HAIR = "#16294A"
MASK = "#A8D4F0"
GLOVE = "#3E8E7E"
DRAPE = "#B9DDE9"
OUTLINE = "#1F3A5C"

FIG_W, FIG_H = 16.4, 5.6
AX = [0.004, 0.006, 0.992, 0.986]

#: Vertical stretch that makes shapes look square on this non-square canvas.
YS = (FIG_W / 100.0) / (FIG_H / 100.0)


def font():
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for fam in ("Arial", "Helvetica", "Inter", "Liberation Sans", "DejaVu Sans"):
        if fam in have:
            return fam
    return "DejaVu Sans"


def fig_xy(x, y):
    return AX[0] + AX[2] * x / 100.0, AX[1] + AX[3] * y / 100.0


def photo(fig, dbox, path, border=True, anchor="c"):
    """Place an image inside dbox, aspect kept. anchor: c=centre, n=top."""
    x0, y0 = fig_xy(dbox[0], dbox[1])
    x1, y1 = fig_xy(dbox[2], dbox[3])
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    img = np.asarray(im.convert("RGB"))
    bw, bh = (x1 - x0) * FIG_W, (y1 - y0) * FIG_H
    ar = img.shape[1] / img.shape[0]
    w, h = (bw, bw / ar) if bw / ar <= bh else (bh * ar, bh)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2 if anchor == "c" else y1 - h / FIG_H / 2
    ax = fig.add_axes([cx - w / 2 / FIG_W, cy - h / 2 / FIG_H, w / FIG_W, h / FIG_H])
    ax.imshow(img, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(border); sp.set_color(GREY_EDGE); sp.set_linewidth(0.8)
    return ax


#: 그림 안 캡션 전부 검정, 크기 1.5배 (2026-09-06 사용자 지시 — 인쇄 판독성).
CAPTION_SCALE = 1.5


def t(ax, x, y, s, size, color=INK, weight="normal", ha="left", va="top",
      style="normal"):
    ax.text(x, y, s, fontsize=size * CAPTION_SCALE, color=INK, fontweight=weight,
            ha=ha, va=va, fontstyle=style, linespacing=1.3, zorder=12)


def box(ax, x0, y0, x1, y1, fill="white", edge=BOX_EDGE, lw=1.1, ls="-"):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fill,
                           edgecolor=edge, linewidth=lw, linestyle=ls,
                           zorder=2))


def arrow(ax, p0, p1, color=NAVY, lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=4,
        shrinkA=1.5, shrinkB=1.5))


def _ell(ax, cx, cy, rx, ry_vis, fill, edge="none", lw=0.8, z=6):
    ax.add_patch(matplotlib.patches.Ellipse(
        (cx, cy), 2 * rx, 2 * ry_vis * YS, facecolor=fill, edgecolor=edge,
        linewidth=lw, zorder=z))


def _poly(ax, pts, fill, edge="none", lw=0.9, z=6):
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=edge,
                         linewidth=lw, zorder=z))


def scanning_clinician(ax, x0, y0, sc=1.0):
    """Flat, faceless clinician scanning a draped patient with a probe.

    Vector fallback for the reference illustration (white coat over navy
    scrubs, hair in a bun, surgical mask, teal gloves, probe on the exposed
    abdomen). ``(x0, y0)`` anchors the lower-left of a ~13 × 11 unit scene.
    """
    def X(dx):
        return x0 + dx * sc

    def Y(dy):
        return y0 + dy * YS * sc

    def E(cx, cy, rx, ry, fill, **kw):
        _ell(ax, cx, cy, rx * sc, ry * sc, fill, **kw)

    BED = "#8FB9CE"      # examination table
    BED_TOP = "#A9CDDD"
    GOWN = "#CDE3EE"     # patient drape / gown

    # ---- examination bed: horizontal table the patient reclines on -----
    _poly(ax, [(X(3.0), Y(0.0)), (X(13.2), Y(0.0)), (X(13.2), Y(2.4)),
               (X(3.0), Y(2.4))], BED, edge=OUTLINE, lw=0.9, z=4)
    _poly(ax, [(X(3.0), Y(2.1)), (X(13.2), Y(2.1)), (X(13.2), Y(2.4)),
               (X(3.0), Y(2.4))], BED_TOP, z=5)

    # ---- patient: reclining, head on a pillow at the right ------------
    # body + gown draped along the table, exposed abdomen at the centre
    _poly(ax, [(X(5.4), Y(2.4)), (X(12.4), Y(2.4)), (X(12.4), Y(3.9)),
               (X(10.8), Y(4.4)), (X(9.0), Y(4.0)), (X(7.4), Y(3.7)),
               (X(6.0), Y(3.4))], GOWN, edge=OUTLINE, lw=0.8, z=6)
    E(X(11.7), Y(3.5), 1.5, 0.95, "white", edge=OUTLINE, lw=0.7, z=6)  # pillow
    E(X(11.6), Y(4.4), 0.95, 0.9, SKIN, edge=OUTLINE, lw=0.8, z=7)     # head
    E(X(11.75), Y(4.95), 1.05, 0.6, HAIR, z=8)                          # hair
    E(X(7.35), Y(3.5), 1.5, 0.62, SKIN, edge=OUTLINE, lw=0.7, z=7)     # abdomen

    # ---- clinician: seated at the left, leaning toward the patient ----
    # white coat torso
    _poly(ax, [(X(0.6), Y(1.2)), (X(1.1), Y(6.6)), (X(2.0), Y(7.5)),
               (X(4.6), Y(7.5)), (X(5.3), Y(6.7)), (X(5.2), Y(2.2)),
               (X(4.2), Y(1.2))], "white", edge=OUTLINE, lw=1.0, z=8)
    # scrub V-neck under the coat
    _poly(ax, [(X(2.35), Y(7.4)), (X(4.05), Y(7.4)), (X(3.2), Y(5.2))],
          NAVY, z=9)
    _poly(ax, [(X(2.35), Y(7.4)), (X(3.15), Y(5.5)), (X(2.1), Y(6.6))],
          "white", edge=OUTLINE, lw=0.7, z=10)
    _poly(ax, [(X(4.05), Y(7.4)), (X(3.25), Y(5.5)), (X(4.3), Y(6.6))],
          "white", edge=OUTLINE, lw=0.7, z=10)

    # both arms sweeping from the shoulders down to the probe hand
    _poly(ax, [(X(1.5), Y(6.3)), (X(2.5), Y(6.7)), (X(6.9), Y(3.9)),
               (X(6.2), Y(3.1))], "white", edge=OUTLINE, lw=1.0, z=9)
    _poly(ax, [(X(4.4), Y(6.9)), (X(5.2), Y(7.0)), (X(7.2), Y(4.2)),
               (X(6.5), Y(3.5))], "white", edge=OUTLINE, lw=1.0, z=9)
    E(X(6.75), Y(3.75), 0.62, 0.5, GLOVE, edge=OUTLINE, lw=0.8, z=11)
    E(X(6.35), Y(3.35), 0.5, 0.42, GLOVE, edge=OUTLINE, lw=0.8, z=11)

    # probe in the gloved hand, resting on the abdomen
    _poly(ax, [(X(6.55), Y(2.9)), (X(7.35), Y(2.9)), (X(7.15), Y(3.9)),
               (X(6.75), Y(3.9))], "white", edge=OUTLINE, lw=0.8, z=10)

    # head: dark hair with a bun, skin, blue surgical mask (faceless)
    E(X(2.75), Y(9.0), 1.45, 1.5, HAIR, z=9)
    E(X(1.5), Y(9.35), 0.62, 0.62, HAIR, z=9)          # bun
    E(X(2.95), Y(8.75), 1.15, 1.2, SKIN, z=10)
    E(X(2.8), Y(9.65), 1.32, 0.72, HAIR, z=11)         # fringe
    E(X(3.12), Y(8.2), 0.95, 0.66, MASK, edge=OUTLINE, lw=0.7, z=11)
    ax.add_patch(Rectangle((X(2.35), Y(7.25)), 1.05 * sc, 0.7 * YS * sc,
                           facecolor=SKIN, edgecolor="none", zorder=8))


def main():
    if os.path.exists(FINAL_ART):
        import shutil
        shutil.copy2(FINAL_ART, OUT + ".png")
        print("wrote", OUT + ".png", "(from approved artwork)")
        return
    plt.rcParams["font.family"] = font()
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(AX)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")

    # 스테이지 박스는 내용에 맞춰 잘라낸다 — 상하 여백은 사실상 없다.
    Y0, Y1 = 24.0, 90.0
    TTL = Y1 - 2.2

    # ---- stage 1 · segmentation ----------------------------------------
    box(ax, 1.0, Y0, 22.5, Y1, fill="white")
    t(ax, 2.2, TTL, "1", 11.5, NAVY, "bold")
    t(ax, 4.8, TTL, "Segmentation", 10.0, INK, "bold")
    photo(fig, (2.0, 52.0, 21.7, 82.5),
          os.path.join(ASSETS, "us_frame_wide_seg.png"), anchor="n")
    t(ax, 2.2, 50.0,
      "U-Net lumen mask on the B-mode\nbladder frame, 256 × 256", 7.8, INK)
    t(ax, 2.2, 38.5,
      "mask → centroid $\\hat{c}$\nimage → contrast, centering,\n"
      "boundary sharpness", 7.8, SUB)

    # ---- stage 2 · quality judgment ------------------------------------
    box(ax, 25.0, Y0, 46.5, Y1, fill="white")
    t(ax, 26.2, TTL, "2", 11.5, NAVY, "bold")
    t(ax, 28.8, TTL, "Quality judgment", 10.0, INK, "bold")
    t(ax, 26.2, 80.5,
      "$Q$ — weighted geometric mean\nof image and mask sub-scores", 8.2, INK)
    t(ax, 26.2, 66.5,
      "attained range 0.64\n→ 47 distinguishable states", 7.8, SUB)
    t(ax, 26.2, 52.5,
      "92.4% of the range is read\noff the ultrasound image", 7.8, SUB)
    t(ax, 26.2, 38.5,
      "decides whether the frame can\ninstruct, and what it falls\nshort of",
      7.8, SUB, style="italic")

    # ---- stage 3 · corrective action -----------------------------------
    box(ax, 49.0, Y0, 72.5, Y1, fill="white")
    t(ax, 50.2, TTL, "3", 11.5, NAVY, "bold")
    t(ax, 52.8, TTL, "Corrective action", 10.0, INK, "bold")
    t(ax, 50.2, 80.5,
      "$\\hat{e} = A - \\hat{c}$   lateral, in-plane", 8.6, INK)
    t(ax, 50.2, 70.5,
      "contrast ≤ 0 → check bladder filling\n"
      f"|$\\hat{{e}}$| < {HOLD_K:.2f}σ = {HOLD_PX:.1f} px → hold\n"
      "otherwise → move left / move right", 7.8, INK)
    t(ax, 50.2, 52.5,
      "the word names the direction;\n$\\hat{e}$ travels beside it in pixels,\n"
      "unrounded", 7.8, SUB)
    t(ax, 50.2, 35.5,
      f"both boundaries derived, not tuned\nσ = {SIGMA:.2f} px centroid noise",
      7.8, SUB, style="italic")

    # ---- output token ---------------------------------------------------
    t(ax, 75.9, 57.0, "word + $\\hat{e}$", 8.0, NAVY, "bold", ha="center",
      va="center")

    # ---- reader A · human operator --------------------------------------
    box(ax, 79.5, 52.0, 99.0, 98.0, fill=BOX_GROUP)
    t(ax, 80.7, 96.2, "Human operator", 9.4, INK, "bold")
    t(ax, 80.7, 91.8, "language, displayed or spoken", 7.4, SUB)
    if os.path.exists(CLINICIAN_ART):
        photo(fig, (80.0, 53.5, 90.5, 88.0), CLINICIAN_ART, border=False)
    else:
        scanning_clinician(ax, 80.0, 53.5, sc=0.85)
    bx0, by0, bx1, by1 = 91.0, 62.0, 98.4, 80.0
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor="white", edgecolor=NAVY,
                                linewidth=1.0, zorder=10))
    ax.add_patch(Polygon([(bx0 - 0.1, by0 + 14.6), (bx0 - 2.6, by0 + 11.2),
                          (bx0 - 0.1, by0 + 9.0)], closed=True,
                         facecolor="white", edgecolor=NAVY, linewidth=1.0,
                         zorder=11))
    ax.add_patch(Rectangle((bx0 + 0.02, by0 + 8.6), 0.5, 6.6,
                           facecolor="white", edgecolor="white", zorder=11))
    t(ax, (bx0 + bx1) / 2, by0 + 15.6, "“move left”", 8.4, NAVY, "bold",
      ha="center")
    t(ax, (bx0 + bx1) / 2, by0 + 8.0, "$\\hat{e}$ = 12 px", 7.2, SUB,
      ha="center")

    # ---- reader B · robotic probe holder --------------------------------
    box(ax, 79.5, 2.0, 99.0, 48.0, fill="white", ls=(0, (5, 3)))
    t(ax, 80.7, 46.0, "Robotic probe holder", 9.4, INK, "bold")
    t(ax, 80.7, 41.4, "action token, translated by lookup", 7.4, SUB)
    photo(fig, (80.0, 4.0, 87.6, 33.5),
          os.path.join(ASSETS, "photo_fr5_arm.png"), border=False)
    t(ax, 88.0, 33.5,
      "$\\langle$move left, $\\hat{e}\\,\\rangle$\n"
      "→ $v_x = -k\\,\\hat{e}$, saturated,\nother axes zero", 7.4, INK)
    t(ax, 88.4, 15.5, "hold → zero twist\ncheck filling → gate", 7.0, SUB)
    t(ax, 88.0, 6.0, "execution: §4.1, design only", 6.2, GREY, style="italic")

    # ---- arrows ---------------------------------------------------------
    mid = (Y0 + Y1) / 2
    arrow(ax, (22.5, mid), (25.0, mid))
    arrow(ax, (46.5, mid), (49.0, mid))
    arrow(ax, (72.5, mid + 4.0), (79.5, 73.0), rad=-0.20)
    arrow(ax, (72.5, mid - 4.0), (79.5, 27.0), rad=0.20)

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, facecolor="white")
    print("wrote", OUT + ".png")


if __name__ == "__main__":
    main()
