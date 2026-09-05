#!/usr/bin/env python3
"""Figure 2 — 힘 유지 검증. 캡처 로그에서 직접 그린다, 손으로 적은 값 없음."""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "force_hold_validation"))

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from fh.analysis import load_run, settling_point                 # noqa: E402

NAVY_DK, NAVY, NAVY_LT = "#1F3A5F", "#2E6E96", "#A7CFE6"
GREY_DK, GREY, GREY_LT = "#666C74", "#9AA1A9", "#C7CCD1"
INK, SUB, BAND = "#1A1F26", "#5A6068", "#DCE6EF"
RUNS = os.path.join(_ROOT, "force_hold_validation", "runs")
HOLD = ("A_t0p5", "A_t2p0", "A_t3p0", "A_t4p0")


def load(label):
    return load_run(os.path.join(RUNS, f"{label}_samples.csv"))


def main():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "axes.grid": True, "grid.color": GREY_LT, "grid.linewidth": 0.5,
        "xtick.color": SUB, "ytick.color": SUB, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "font.size": 7.5, "legend.frameon": False,
        "legend.fontsize": 6.8, "axes.titlesize": 8, "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(17.4 / 2.54, 5.6 / 2.54))
    gs = fig.add_gridspec(1, 5, wspace=0.95, left=0.045, right=0.995,
                          bottom=0.16, top=0.88)

    # (a) 대역별 유지 — 네 설정점을 한 축에 겹치면 스케일이 뭉개지므로 작은 판 넷.
    inner = gs[0, :3].subgridspec(1, 4, wspace=0.55)
    for i, label in enumerate(HOLD):
        ax = fig.add_subplot(inner[0, i])
        run = load(label)
        mask = run.probing()
        t = run.t[mask] - run.t[mask][0]
        f = run.force[mask]
        point = settling_point(run)
        ax.axhspan(run.target - run.band, run.target + run.band, color=BAND,
                   lw=0, zorder=0)
        ax.axhline(run.target, color=GREY_DK, ls="--", lw=0.8, zorder=2)
        ax.plot(t, f, color=NAVY, lw=0.45, zorder=3)
        ax.set_xlim(0, 60)
        span = max(run.band * 1.7, 0.36)
        ax.set_ylim(point - span, run.target + run.band + span * 0.2)
        ax.set_title(f"{run.target:.1f} N", color=INK, pad=3)
        ax.set_xticks([0, 30, 60])
        if i == 0:
            ax.set_ylabel("Contact force ‖F‖ [N]")
        if i == 1:
            ax.set_xlabel("Time [s]", labelpad=1)
    fig.text(0.165, 0.955, "(a)  60 s holds — shaded band is the deadband, "
             "dashed line the setpoint", fontsize=7.6, weight="bold", color=INK,
             ha="center")

    # (b) 교란과 복귀 — 경고·한계선을 같은 축에 둔다.
    ax = fig.add_subplot(gs[0, 3:])
    run = load("C_limit")
    mask = run.probing()
    t = run.t[mask] - run.t[mask][0]
    f = run.force[mask]
    warn = float(run.meta["warn_contact_force_n"])
    limit = float(run.meta["max_contact_force_n"])
    top = run.target + run.band
    ax.axhspan(run.target - run.band, top, color=BAND, lw=0, zorder=0)
    ax.fill_between(t, top, f, where=f > top, color=NAVY_LT, lw=0, zorder=1)
    ax.plot(t, f, color=NAVY, lw=0.45, zorder=3)
    ax.axhline(limit, color=INK, ls="--", lw=1.1, zorder=4)
    ax.axhline(warn, color=GREY_DK, ls="-.", lw=0.9, zorder=4)
    ax.set_ylim(min(f.min(), run.target - run.band) - 0.25, limit + 0.30)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Contact force ‖F‖ [N]")
    ax.text(t[-1] * 0.99, limit, "force limit 5.0 N ", fontsize=6.8, color=INK,
            ha="right", va="bottom")
    ax.text(t[-1] * 0.99, warn, "warning 4.5 N ", fontsize=6.8, color=SUB,
            ha="right", va="bottom")
    ax.text(t[-1] * 0.99, top, "deadband top ", fontsize=6.8, color=NAVY_DK,
            ha="right", va="bottom")
    fig.text(0.80, 0.955, "(b)  Repeated rapid injections at the 3.0 N setpoint",
             fontsize=7.6, weight="bold", color=INK, ha="center")

    for suffix in (".png", ".pdf"):
        fig.savefig(os.path.join(_HERE, "figures", "fig2_force_validation" + suffix),
                    dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("figures/fig2_force_validation.png")


if __name__ == "__main__":
    main()
