#!/usr/bin/env python3
"""검증 리포트 그림. 숫자는 전부 캡처에서 읽고 손으로 적은 값은 없다."""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib.pyplot as plt                                  # noqa: E402
from fh.analysis import (excursions, hold_metrics, load_run,      # noqa: E402
                         loop_state, settling_point)
from report.style import (BAND, GREY, GREY_DK, GREY_LT, INK, NAVY,  # noqa: E402
                          NAVY_DK, NAVY_LT, SUB, save, setup)

RUNS = os.path.join(os.path.dirname(_HERE), "runs")
HOLD = ("A_t0p5", "A_t2p0", "A_t3p0", "A_t4p0")
DIST = ("B_t0p5", "B_t3p0_pose", "C_limit")

#: 표·범례용 이름. 내부 라벨은 파일 이름이지 독자가 읽을 것이 아니다.
NAMES = {"B_t0p5": "Disturbance · 0.5 N",
         "B_t3p0_pose": "Disturbance · 3.0 N",
         "C_limit": "Hard press · 3.0 N"}



def load(label):
    return load_run(os.path.join(RUNS, f"{label}_samples.csv"))


# ── Fig 1 ──────────────────────────────────────────────────────────────────
def fig_hold_traces():
    runs = [load(l) for l in HOLD]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.3), sharey=False)
    for ax, run in zip(axes, runs):
        mask = run.probing()
        t = run.t[mask] - run.t[mask][0]
        f = run.force[mask]
        point = settling_point(run)
        ax.axhspan(run.target - run.band, run.target + run.band,
                   color=BAND, lw=0, zorder=0)
        ax.axhline(run.target, color=GREY_DK, ls="--", lw=0.9, zorder=2)
        ax.axhline(point, color=NAVY_DK, ls=":", lw=1.0, zorder=2)
        ax.plot(t, f, color=NAVY, lw=0.5, zorder=3)
        ax.set_xlim(0, 60)
        span = max(run.band * 1.6, 0.35)
        ax.set_ylim(point - span, run.target + run.band + span * 0.25)
        ax.set_title(f"setpoint {run.target:.1f} N   band ±{run.band:.2f}", color=INK)
        ax.set_xlabel("Time [s]")
    axes[0].set_ylabel("Contact force ‖F‖ [N]")
    handles = [plt.Line2D([], [], color=NAVY, lw=1.2),
               plt.Line2D([], [], color=NAVY_DK, ls=":", lw=1.2),
               plt.Line2D([], [], color=GREY_DK, ls="--", lw=1.2),
               plt.Rectangle((0, 0), 1, 1, color=BAND)]
    fig.legend(handles, ["Measured", "Operating point", "Setpoint", "Deadband"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return save(fig, "fig1_hold_traces")


# ── Fig 2 ──────────────────────────────────────────────────────────────────
def fig_hold_quality():
    rows = []
    for label in HOLD:
        run = load(label)
        m = hold_metrics(run)
        rows.append((run.target, run.band, m["sd_n"], m["error_vs_settling_n"],
                     m["sd_n"] / run.target * 100))
    target = np.array([r[0] for r in rows])
    sd = np.array([r[2] for r in rows])
    err = np.array([r[3] for r in rows])
    rel = np.array([r[4] for r in rows])
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.5))
    axes[0].bar(x, sd, width=0.55, color=NAVY, edgecolor="white", lw=0.6)
    for i, v in enumerate(sd):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    axes[0].set_ylabel("Hold SD [N]")
    axes[0].set_ylim(0, sd.max() * 1.35)

    axes[1].bar(x, err, width=0.55, color=NAVY_LT, edgecolor=NAVY, lw=0.7)
    axes[1].axhline(0, color=INK, lw=0.8)
    for i, v in enumerate(err):
        axes[1].text(i, v, f"{v:+.3f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
    axes[1].set_ylabel("Offset from operating point [N]")
    axes[1].set_ylim(min(err.min() * 1.6, -0.02), max(err.max() * 1.5, 0.02))

    axes[2].plot(x, rel, "o-", color=NAVY, ms=5, lw=1.2)
    for i, v in enumerate(rel):
        axes[2].text(i, v, f" {v:.1f}%", fontsize=7.5, color=SUB, va="center")
    axes[2].set_ylabel("SD / setpoint [%]")
    axes[2].set_ylim(0, rel.max() * 1.3)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.1f} N" for t in target])
        ax.set_xlabel("Setpoint")
    fig.tight_layout()
    return save(fig, "fig2_hold_quality")


# ── Fig 3 ──────────────────────────────────────────────────────────────────
def fig_disturbance():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    shades = {"B_t0p5": NAVY_LT, "B_t3p0_pose": NAVY, "C_limit": NAVY_DK}
    for label in DIST:
        run = load(label)
        marks = [t for t, k in run.events if k == "i"]
        for onset in marks:
            w = (run.t >= onset - 0.5) & (run.t <= onset + 6.0)
            if w.sum() < 20:
                continue
            base = run.force[(run.t >= onset - 0.3) & (run.t < onset)].mean()
            axes[0].plot(run.t[w] - onset, run.force[w] - base,
                         color=shades[label], lw=0.6, alpha=0.75)
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].axvline(0, color=GREY_DK, ls=":", lw=0.9)
    axes[0].set_xlabel("Time from 150 mL injection [s]")
    axes[0].set_ylabel("Force change [N]")
    axes[0].legend([plt.Line2D([], [], color=shades[l], lw=1.4) for l in DIST],
                   [NAMES[l] for l in DIST], loc="upper right")

    allex = []
    for label in DIST:
        allex.extend(excursions(load(label)))
    labels = sorted({e["run"] for e in allex})
    rng = np.random.default_rng(0)
    for i, label in enumerate(labels):
        d = [e["duration_s"] for e in allex if e["run"] == label]
        axes[1].scatter(np.full(len(d), i) + rng.uniform(-.09, .09, len(d)), d,
                        s=22, color=NAVY, alpha=.85, edgecolor="white", lw=.6, zorder=3)
        axes[1].plot([i - .26, i + .26], [np.median(d)] * 2, color=INK, lw=1.6, zorder=4)
    worst = max(e["duration_s"] for e in allex)
    axes[1].axhline(worst, color=GREY_DK, ls="--", lw=0.9)
    axes[1].text(len(labels) - 0.55, worst, f"max {worst:.2f} s", fontsize=7.5,
                 color=SUB, va="bottom", ha="right")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels([NAMES[l] for l in labels], fontsize=7, rotation=12,
                            ha="right")
    axes[1].set_ylabel("Return to band [s]")
    axes[1].set_ylim(0, worst * 1.25)
    fig.tight_layout()
    return save(fig, "fig3_disturbance_response")


# ── Fig 4 ──────────────────────────────────────────────────────────────────
def fig_exposure():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    shades = {"B_t0p5": NAVY_LT, "B_t3p0_pose": NAVY, "C_limit": NAVY_DK}
    warn = limit = None
    for label in DIST:
        run = load(label)
        warn = float(run.meta["warn_contact_force_n"])
        limit = float(run.meta["max_contact_force_n"])
        mask = run.probing()
        t, f = run.t[mask], run.force[mask]
        top = run.target + run.band
        levels = np.linspace(top, limit, 40)
        longest = []
        for lv in levels:
            above = f > lv
            best, i = 0.0, 0
            while i < len(above):
                if not above[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(above) and above[j + 1]:
                    j += 1
                best = max(best, float(t[j] - t[i]))
                i = j + 1
            longest.append(best)
        axes[0].plot(levels, longest, color=shades[label], lw=1.4)
    axes[0].axvline(warn, color=GREY_DK, ls="-.", lw=1.0)
    axes[0].axvline(limit, color=INK, ls="--", lw=1.2)
    mid_y = axes[0].get_ylim()[1] * 0.45
    axes[0].text(warn, mid_y, "Warn ", fontsize=7.5, color=SUB, ha="right",
                 va="center", rotation=90)
    axes[0].text(limit, mid_y, "Limit ", fontsize=7.5, color=INK, ha="right",
                 va="center", rotation=90)
    axes[0].set_xlabel("Force threshold [N]")
    axes[0].set_ylabel("Longest continuous time above [s]")
    axes[0].legend([plt.Line2D([], [], color=shades[l], lw=1.4) for l in DIST],
                   [NAMES[l] for l in DIST], loc="center left")

    run = load("C_limit")
    mask = run.probing()
    t = run.t[mask] - run.t[mask][0]
    f = run.force[mask]
    top = run.target + run.band
    axes[1].axhline(limit, color=INK, ls="--", lw=1.2)
    axes[1].axhline(warn, color=GREY_DK, ls="-.", lw=1.0)
    axes[1].axhline(top, color=NAVY_DK, ls=":", lw=1.0)
    axes[1].fill_between(t, top, f, where=f > top, color=NAVY_LT, lw=0)
    axes[1].plot(t, f, color=NAVY, lw=0.5)
    axes[1].set_ylim(min(f.min(), top) - 0.3, limit + 0.25)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Contact force ‖F‖ [N]")
    axes[1].text(t[-1], limit, "Limit ", fontsize=7.5, color=INK, ha="right", va="bottom")
    axes[1].text(t[-1], warn, "Warn ", fontsize=7.5, color=SUB, ha="right", va="bottom")
    fig.tight_layout()
    return save(fig, "fig4_force_exposure")


# ── Fig 5 ──────────────────────────────────────────────────────────────────
def fig_travel():
    from fh.analysis import axial_travel
    run = load("B_t3p0_pose")
    mask = run.probing() & np.isfinite(run.quat).all(axis=1) & \
        np.isfinite(run.tip).all(axis=1)
    idx = np.where(mask)[0]
    t = run.t[idx] - run.t[idx][0]
    f = run.force[idx]
    # 기준은 자세가 실제로 들어온 첫 표본이다. 캡처 첫 순간은 자세가 아직 없어
    # NaN 이고, 그것을 기준으로 삼으면 변위가 통째로 0 이 된다.
    axis = np.array([2 * (run.quat[idx[0], 0] * run.quat[idx[0], 2]
                          + run.quat[idx[0], 1] * run.quat[idx[0], 3]),
                     2 * (run.quat[idx[0], 1] * run.quat[idx[0], 2]
                          - run.quat[idx[0], 0] * run.quat[idx[0], 3]),
                     1 - 2 * (run.quat[idx[0], 0] ** 2 + run.quat[idx[0], 1] ** 2)])
    axis /= np.linalg.norm(axis)
    travel = (run.tip[idx] - run.tip[idx[0]]) @ axis * 1000.0

    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.axhspan(run.target - run.band, run.target + run.band, color=BAND, lw=0, zorder=0)
    ax.plot(t, f, color=NAVY, lw=0.6, zorder=3)
    ax.set_ylabel("Contact force ‖F‖ [N]", color=NAVY_DK)
    ax.set_xlabel("Time [s]")
    ax.set_ylim(f.min() - 0.2, f.max() + 0.2)

    twin = ax.twinx()
    twin.plot(t, travel, color=GREY_DK, lw=1.0, zorder=2)
    twin.set_ylabel("Probe axial travel [mm]", color=GREY_DK)
    twin.grid(False)
    for onset, key in run.events:
        if key in ("i", "w"):
            ax.axvline(onset - run.t[idx][0], color=GREY, ls=":", lw=0.7, zorder=1)
    ax.legend([plt.Line2D([], [], color=NAVY, lw=1.4),
               plt.Line2D([], [], color=GREY_DK, lw=1.4)],
              ["Contact force", "Probe travel"], loc="upper left")
    fig.tight_layout()
    return save(fig, "fig5_probe_travel")


def main():
    setup()
    made = []
    for fn in (fig_hold_traces, fig_hold_quality, fig_disturbance,
               fig_exposure, fig_travel):
        made += fn()
    for path in made:
        print(os.path.relpath(path, os.path.dirname(_HERE)))


if __name__ == "__main__":
    main()
