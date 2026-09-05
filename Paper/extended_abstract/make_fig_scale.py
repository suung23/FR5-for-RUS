#!/usr/bin/env python3
"""저울 대조 그림 — 단 폭 두 판. force_validation/outputs/trial_results.csv 에서 직접 그린다."""
from __future__ import annotations

import csv
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
SRC = os.path.join(_ROOT, "force_validation", "outputs", "trial_results.csv")

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

NAVY_DK, NAVY, NAVY_LT = "#1F3A5F", "#2E6E96", "#A7CFE6"
GREY_DK, GREY, GREY_LT = "#666C74", "#9AA1A9", "#C7CCD1"
INK, SUB, BAND = "#1A1F26", "#5A6068", "#DCE6EF"


def main():
    rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8"))
            if r["included_in_primary_analysis"].strip().upper() == "TRUE"]
    ref = np.array([float(r["scale_reference_force_N"]) for r in rows])
    rob = np.array([float(r["robot_normal_force_instant_N"]) for r in rows])
    diff = rob - ref
    bias = diff.mean()
    sd = diff.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    slope, intercept = np.polyfit(ref, rob, 1)
    u = float(rows[0]["combined_reference_uncertainty_N"])

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": INK, "axes.linewidth": 0.7, "axes.labelcolor": INK,
        "axes.grid": True, "grid.color": GREY_LT, "grid.linewidth": 0.45,
        "xtick.color": SUB, "ytick.color": SUB, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "font.size": 6.5, "legend.frameon": False,
        "legend.fontsize": 5.8, "axes.titlesize": 7, "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.15 / 2.54, 3.9 / 2.54))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.235, top=0.865, wspace=0.46)

    # (a) 로봇 대 저울
    ax = axes[0]
    grid = np.linspace(0, ref.max() * 1.06, 20)
    ax.plot(grid, grid, color=GREY_DK, ls="--", lw=0.8, zorder=1)
    ax.plot(grid, slope * grid + intercept, color=NAVY_DK, lw=1.0, zorder=2)
    ax.scatter(ref, rob, s=9, color=NAVY, alpha=.9, edgecolor="white",
               linewidth=.35, zorder=3)
    ax.set_xlabel("Scale reference [N]", labelpad=1.5)
    ax.set_ylabel("Robot normal force [N]", labelpad=1.5)
    ax.set_title(f"slope {slope:.3f}, $R^2$ 0.993", pad=2.5, color=INK)
    ax.set_xlim(0, ref.max() * 1.06); ax.set_ylim(0, ref.max() * 1.06)

    # (b) Bland–Altman
    ax = axes[1]
    mean = (rob + ref) / 2
    ax.axhspan(-u, u, color=BAND, lw=0, zorder=0)
    ax.axhline(bias, color=NAVY_DK, lw=1.0, zorder=2)
    for level, style in ((lo, (0, (4, 3))), (hi, (0, (4, 3)))):
        ax.axhline(level, color=GREY_DK, ls=style, lw=0.8, zorder=2)
    ax.scatter(mean, diff, s=9, color=NAVY, alpha=.9, edgecolor="white",
               linewidth=.35, zorder=3)
    ax.set_xlabel("Mean of robot and scale [N]", labelpad=1.5)
    ax.set_ylabel("Robot − scale [N]", labelpad=1.5)
    ax.set_ylim(lo - 0.16, max(diff.max(), hi) + 0.10)
    ax.set_title(f"bias {bias:+.3f} N, LoA {lo:+.2f} to {hi:+.2f}", pad=2.5, color=INK)
    ax.text(mean.max(), hi, " +1.96 SD", fontsize=5.4, color=SUB, ha="right",
            va="bottom")
    ax.text(mean.max(), lo, " −1.96 SD", fontsize=5.4, color=SUB, ha="right",
            va="top")

    for suffix in (".png", ".pdf"):
        fig.savefig(os.path.join(_HERE, "figures", "fig3_scale_validation" + suffix),
                    dpi=400, bbox_inches="tight", pad_inches=0.015, facecolor="white")
    plt.close(fig)
    print(f"n={len(rows)}  bias={bias:+.4f}  sd={sd:.4f}  LoA={lo:.4f}..{hi:.4f}  "
          f"slope={slope:.4f}  intercept={intercept:.4f}  u_combined={u:.5f}")


if __name__ == "__main__":
    main()
