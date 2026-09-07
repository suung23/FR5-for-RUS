#!/usr/bin/env python3
"""Print-size redraws of the force-hold figures, for the two-page abstract.

``force_hold_validation/fh/plotting.py`` draws for the analysis report: eight
rows of panels, 7.2 in wide, read on screen. Scaled into a two-column abstract
those figures land near 9 cm wide and their tick labels fall under 5 pt, which
does not survive print. These redraws use the same CSVs — nothing is recomputed
here — at the size they will actually be printed at, so the type in the figure
matches the type around it.

Matplotlib has no Times New Roman on this machine; Nimbus Roman is the
metric-compatible substitute and is what the figures are set in.
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "force_hold_validation"))
from fh.analysis import load_run     # noqa: E402

DATA = os.path.join(REPO, "force_hold_validation", "outputs_pooled")
RUNS = os.path.join(REPO, "force_hold_validation", "runs_lap2")

#: One text-block width, and short enough that two of them plus two tables fit
#: the page budget [in].
WIDE, TALL = 5.35, 1.32
DARK, GRAY, LIGHT = "#1a1a1a", "#6f6f6f", "#d9d9d9"


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "lines.linewidth": 0.8,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def read(name: str, directory: str = DATA) -> list:
    with open(os.path.join(directory, name), newline="") as handle:
        return list(csv.DictReader(handle))


def hold_figure(path: str, settle_s: float = 3.0, seconds: float = 27.0,
                every: int = 40) -> str:
    """(a) every hold as error from its setpoint, (b) the phantom's F(z) line."""
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(WIDE, TALL),
                                gridspec_kw={"width_ratios": [1.55, 1.0]})

    # -- (a) the holds themselves -------------------------------------------
    ax = axes[0]
    ax.axhspan(-0.05, 0.05, color=LIGHT, lw=0, zorder=0)
    ax.axhline(0.0, color=GRAY, lw=0.5, zorder=1)
    paths = sorted(glob.glob(os.path.join(RUNS, "A_*_samples.csv")))
    for index, samples in enumerate(paths):
        run = load_run(samples)
        mask = run.probing()
        if not mask.any():
            continue
        start = run.t[mask][0] + settle_s
        mask &= (run.t >= start) & (run.t <= start + seconds)
        shade = str(0.72 - 0.09 * index)          # light 0.5 N → dark 4.0 N
        ax.plot(run.t[mask][::every] - start, (run.force[mask] - run.target)[::every],
                color=shade, lw=0.55, zorder=2)
    ax.set_xlim(0, seconds)
    ax.set_ylim(-0.17, 0.17)
    ax.set_xlabel("time in hold [s]")
    ax.set_ylabel("|F| − setpoint [N]")
    ax.annotate("eight setpoints, 0.5 → 4.0 N (light → dark); shaded: ±0.05 N deadband",
                xy=(0.5, 0.985), xycoords="axes fraction", fontsize=5.6, color=DARK,
                ha="center", va="top",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))

    # -- (b) the phantom, measured by the holds themselves ------------------
    ax = axes[1]
    curve = read("phantom_stiffness_curve.csv")
    fit = read("phantom_stiffness.csv")[0]
    depth = np.array([float(r["depth_mm"]) for r in curve])
    force = np.array([float(r["force_n"]) for r in curve])
    k = float(fit["k_n_per_mm"])
    intercept = float(fit["intercept_n"])
    span = np.linspace(depth.min() - 0.6, depth.max() + 0.6, 2)
    ax.plot(span, intercept + k * span, color=DARK, lw=0.7, zorder=2)
    ax.plot(depth, force, "o", ms=2.6, color=GRAY, mec="white", mew=0.4, zorder=3)
    ax.set_xlabel("indentation past the 0.5 N equilibrium [mm]")
    ax.set_ylabel("|F| [N]")
    ax.annotate(f"k = {k:.3f} N/mm\nR² = {float(fit['r_squared']):.3f}",
                xy=(0.04, 0.96), xycoords="axes fraction", fontsize=6.0,
                va="top", color=DARK)

    for ax in axes:
        ax.grid(True, color="#ececec", lw=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    figure.tight_layout(pad=0.15, w_pad=0.9)
    figure.savefig(path)
    plt.close(figure)
    return path


def placebo_figure(path: str, excluded=(4.0,), p_value=float("nan")) -> str:
    """(a) peak excursion per band, both arms; (b) the two arms' distributions."""
    _style()
    events = read("disturbance_events.csv")
    figure, axes = plt.subplots(1, 2, figsize=(WIDE, TALL),
                                gridspec_kw={"width_ratios": [1.35, 1.0]})

    def peaks(letter, target=None):
        return np.array([abs(float(r["peak_error_n"])) for r in events
                         if r["run"][:1] == letter
                         and (target is None or float(r["target_n"]) == target)])

    targets = sorted({float(r["target_n"]) for r in events})
    arms = (("B", "force hold on", DARK, "o"), ("C", "off (placebo)", GRAY, "s"))

    # -- (a) band by band ----------------------------------------------------
    ax = axes[0]
    for target in excluded:
        ax.axvspan(target - 0.24, target + 0.24, color=LIGHT, lw=0, zorder=0)
    keep = [abs(float(r["peak_error_n"])) for r in events
            if float(r["target_n"]) not in set(excluded)]
    top = max(keep) * 1.18
    for index, (letter, label, color, marker) in enumerate(arms):
        offset = (index - 0.5) * 0.17
        for target in targets:
            values = peaks(letter, target)
            if not values.size:
                continue
            x = target + offset
            median = float(np.median(values))
            ax.vlines(x, np.percentile(values, 25), np.percentile(values, 75),
                      color=color, lw=0.8, zorder=2)
            if median > top:                      # off the axis, kept on it
                ax.plot([x], [top * 0.97], marker, color=color, ms=2.6, mec="white",
                        mew=0.4, clip_on=False, zorder=4)
                ax.annotate(f"{median:.1f}", xy=(x, top * 0.97),
                            xytext=(-2 if offset < 0 else 2, 4),
                            textcoords="offset points", fontsize=5.4, color=color,
                            ha="right" if offset < 0 else "left")
                continue
            ax.plot([x], [median], marker, color=color, ms=2.6, mec="white", mew=0.4,
                    ls="none", zorder=3, label=label if target == targets[0] else None)
    ax.set_ylim(0, top)
    ax.set_xticks(targets)
    ax.set_xticklabels([f"{t:.1f}" for t in targets])
    ax.set_xlabel("setpoint [N]")
    ax.set_ylabel("peak deviation from setpoint [N]")
    ax.annotate("force limit reached,\nnot open-loop", xy=(min(excluded) - 0.32, top * 0.30),
                fontsize=5.4, color=GRAY, ha="right", va="center")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.1,
              labelspacing=0.25)

    # -- (b) both arms whole -------------------------------------------------
    ax = axes[1]
    for letter, label, color, marker in arms:
        values = np.sort(peaks(letter))
        fraction = np.arange(1, values.size + 1) / values.size
        ax.step(values, fraction, where="post", color=color, lw=0.9)
        ax.plot([float(np.median(values))], [0.5], marker, color=color, ms=2.6,
                mec="white", mew=0.4)
    ax.set_xlabel("peak deviation from setpoint [N], all bands")
    ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1.03)
    ax.annotate(f"rank-sum two-sided\np = {p_value:.2f}\nn = {peaks('B').size} per arm",
                xy=(0.96, 0.10), xycoords="axes fraction", fontsize=5.8,
                ha="right", va="bottom", color=DARK)

    for ax in axes:
        ax.grid(True, color="#ececec", lw=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    figure.tight_layout(pad=0.15, w_pad=0.9)
    figure.savefig(path)
    plt.close(figure)
    return path


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    print(hold_figure(os.path.join(out, "fig_fh_hold.png")))
    print(placebo_figure(os.path.join(out, "fig_fh_placebo.png")))
