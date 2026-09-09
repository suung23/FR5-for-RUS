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
#: Photographs of the bench (2026-09-07), stored rotated and downscaled.
PHOTOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "photos")

#: One text-block width, and short enough that two of them plus two tables fit
#: the page budget [in].
WIDE, TALL = 5.35, 1.32
DARK, GRAY, LIGHT = "#1a1a1a", "#6f6f6f", "#d9d9d9"


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                       "DejaVu Serif"],
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


def _band_panel(ax, events, field, arms, targets, excluded, ylabel, p_value=None,
                legend=False, p_xy=(0.03, 0.96), p_ha="left"):
    """Median and IQR of |field| per band, both arms side by side."""
    for target in excluded:
        ax.axvspan(target - 0.24, target + 0.24, color=LIGHT, lw=0, zorder=0)

    def values(letter, target=None):
        return np.array([abs(float(r[field])) for r in events
                         if r["run"][:1] == letter and r[field] not in ("", "nan")
                         and (target is None or float(r["target_n"]) == target)])

    keep = [abs(float(r[field])) for r in events
            if float(r["target_n"]) not in set(excluded) and r[field] not in ("", "nan")]
    top = max(keep) * 1.18
    for index, (letter, label, color, marker) in enumerate(arms):
        offset = (index - 0.5) * 0.17
        for target in targets:
            v = values(letter, target)
            if not v.size:
                continue
            x = target + offset
            median = float(np.median(v))
            ax.vlines(x, np.percentile(v, 25), np.percentile(v, 75), color=color, lw=0.8,
                      zorder=2)
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
    ax.set_ylabel(ylabel)
    if p_value is not None:
        text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        ax.annotate(f"rank-sum two-sided\n{text}", xy=p_xy, xycoords="axes fraction",
                    fontsize=5.6, ha=p_ha, va="top", color=DARK)
    if legend:
        ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.1,
                  labelspacing=0.25)


def placebo_figure(path: str, excluded=(4.0,), p_travel=float("nan"),
                   p_end=float("nan"), **_ignored) -> str:
    """The placebo comparison as what the loop does, not what it fails to do:
    (a) how far the probe moved per syringe step, (b) how much deviation from the
    setpoint was left when the next step arrived. Both are band by band, median
    and IQR, force hold on against off. The peak itself is in Table 2."""
    _style()
    events = read("disturbance_events.csv")
    figure, axes = plt.subplots(1, 2, figsize=(WIDE, TALL))
    targets = sorted({float(r["target_n"]) for r in events})
    arms = (("B", "force hold on", DARK, "o"), ("C", "off (placebo)", GRAY, "s"))

    _band_panel(axes[0], events, "peak_travel_mm", arms, targets, excluded,
                "probe travel per step, |peak| [mm]", p_value=p_travel, legend=True,
                p_xy=(0.52, 0.96), p_ha="center")
    _band_panel(axes[1], events, "end_error_n", arms, targets, excluded,
                "deviation left at step end [N]", p_value=p_end)
    axes[0].annotate("force limit reached,\nnot open-loop",
                     xy=(min(excluded) - 0.32, axes[0].get_ylim()[1] * 0.12),
                     fontsize=5.4, color=GRAY, ha="right", va="center")

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


# ---------------------------------------------------------------- overview
#: (box patch, text) pairs from the last overview_figure() call — a test hook so
#: a checker can confirm every boxed label stays inside its box.
OVERVIEW_BOXED = []


def overview_figure(path: str, S: dict | None = None) -> str:
    """One figure for the whole argument, left to right: the clinical need
    (a suprapubic probe held on a moving abdomen during morcellation), the
    control that answers it (one-axis admittance with a safety layer), and the
    experiment that tests it (a syringe-driven phantom, three arms). Symbols,
    marks and deployed values are explained in the figure's legend in the paper,
    not in the figure — no abbreviations in the drawing itself. ``S`` is accepted
    for interface stability; nothing in the drawing depends on it.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Polygon, Rectangle
    _style()
    S = S or {}
    ORANGE = "#b8501a"
    OVERVIEW_BOXED.clear()
    figure = plt.figure(figsize=(WIDE, 2.3))
    gs = figure.add_gridspec(1, 3, wspace=0.05, left=0.005, right=0.995, top=0.995,
                             bottom=0.005)
    panels = [figure.add_subplot(gs[0, i]) for i in range(3)]
    for ax in panels:
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(ax, x, y, w, h, text, fc="white", ec=DARK, lw=0.6, fs=5.4, ls="-", color=DARK):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                               fc=fc, ec=ec, lw=lw, linestyle=ls)
        ax.add_patch(patch)
        label = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                        color=color, linespacing=1.15)
        OVERVIEW_BOXED.append((patch, label))
        return patch, label

    def arrow(ax, p0, p1, color=DARK, lw=0.7, style="-|>", ms=5):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color,
                                     lw=lw, shrinkA=0, shrinkB=0))

    def title(ax, text):
        ax.text(1, 99.5, text, ha="left", va="top", fontsize=6.4, fontweight="bold", color=DARK)

    # ---- (a) the clinical need ------------------------------------------------
    ax = panels[0]
    title(ax, "(a) Clinical need")
    ax.text(50, 88, "too little force: coupling lost\ntoo much: deformation, injury",
            ha="center", va="center", fontsize=5.2, color=DARK, linespacing=1.15)
    xs = np.linspace(8, 92, 60)
    wall_top = 53 + 3 * np.sin((xs - 8) / 84 * np.pi)
    ax.fill_between(xs, wall_top - 9, wall_top, color=LIGHT, lw=0)
    ax.plot(xs, wall_top, color=DARK, lw=0.7); ax.plot(xs, wall_top - 9, color=GRAY, lw=0.5)
    ax.text(91, 42, "abdominal wall", ha="right", va="top", fontsize=5.0, color=DARK)
    ax.add_patch(Ellipse((50, 30), 44, 22, fc="white", ec=DARK, lw=0.7))
    ax.text(50, 33, "bladder", ha="center", va="center", fontsize=5.6, color=DARK)
    ax.plot([50, 50], [8, 24], color=DARK, lw=1.4)                    # morcellator
    ax.plot([46, 54], [24, 24], color=DARK, lw=0.9)
    ax.text(53, 13, "morcellator", ha="left", va="center", fontsize=5.0, color=DARK)
    px = 50; ptop = float(np.interp(px, xs, wall_top))
    ax.add_patch(Polygon([[px - 9, ptop + 1], [px + 9, ptop + 1], [px + 6, ptop + 12],
                          [px - 6, ptop + 12]], closed=True, fc=DARK, ec=DARK, lw=0.5))
    ax.add_patch(Rectangle((px - 3, ptop + 12), 6, 14, fc=DARK, ec=DARK, lw=0.5))
    ax.text(px + 8, ptop + 18, "robot-held\nprobe", ha="left", va="center", fontsize=5.0,
            color=DARK, linespacing=1.1)
    arrow(ax, (px - 16, ptop + 24), (px - 16, ptop + 3), lw=0.9, ms=6)
    ax.text(px - 18, ptop + 14, "F", ha="right", va="center", fontsize=6.5, style="italic")
    for x0 in (22, 78):
        y0 = float(np.interp(x0, xs, wall_top))
        arrow(ax, (x0, y0 - 4), (x0, y0 + 8), color=GRAY, lw=0.7, style="<|-|>", ms=4)
    ax.text(50, 3, "wall moves: breathing, bladder filling", ha="center", va="center",
            fontsize=5.0, color=DARK)

    # ---- (b) the control ------------------------------------------------------
    ax = panels[1]
    title(ax, "(b) Control")
    box(ax, 22, 75, 72, 16, "safety layer, both arms\n≥ 4.5 N: no advance · ≥ 5.0 N: retreat",
        ec=ORANGE, ls="--", fs=5.0)
    box(ax, 3, 54, 21, 14, "setpoint\n$F_{ref}$")
    box(ax, 33, 54, 31, 14, "admittance\n$v_z = e / B_z$")
    box(ax, 72, 54, 26, 14, "inverse\nkinematics")
    box(ax, 72, 24, 26, 14, "contact\n$F = k\\,x$")
    box(ax, 33, 24, 31, 14, "force sensor\n‖F‖, 1 kHz", fs=5.2)
    arrow(ax, (24, 61), (33, 61)); ax.text(28.5, 64, "e", ha="center", va="bottom", fontsize=6, style="italic")
    arrow(ax, (64, 61), (72, 61)); ax.text(68, 64, "$v_z$", ha="center", va="bottom", fontsize=6)
    arrow(ax, (85, 54), (85, 38)); ax.text(87, 46, "robot", ha="left", va="center", fontsize=5.0, color=DARK)
    arrow(ax, (72, 31), (64, 31)); ax.text(68, 34, "F", ha="center", va="bottom", fontsize=6, style="italic")
    arrow(ax, (33, 31), (13.5, 31)); arrow(ax, (13.5, 31), (13.5, 54))
    ax.text(11, 43, "−", ha="right", va="center", fontsize=7)
    arrow(ax, (85, 8), (85, 24), color=GRAY, lw=0.8)
    ax.text(85, 6, "surface motion", ha="center", va="top", fontsize=5.2, color=DARK)
    ax.text(3, 10, "τ = $B_z / k$ ≈ 12 s", ha="left", va="center", fontsize=5.4, color=DARK)

    # ---- (c) the experiment ---------------------------------------------------
    ax = panels[2]
    title(ax, "(c) Bench test")
    xs = np.linspace(34, 96, 50)
    top = 50 + 3 * np.sin((xs - 34) / 62 * 2 * np.pi)
    ax.fill_between(xs, 28, top, color=LIGHT, lw=0)
    ax.plot(xs, top, color=DARK, lw=0.7)
    ax.plot([34, 34, 96, 96], [top[0], 28, 28, top[-1]], color=GRAY, lw=0.5)
    ax.text(65, 36, "water-filled\nabdominal phantom", ha="center", va="center", fontsize=5.2,
            color=DARK, linespacing=1.1)
    px = 65; ptop = float(np.interp(px, xs, top))
    ax.add_patch(Polygon([[px - 8, ptop + 1], [px + 8, ptop + 1], [px + 5, ptop + 10],
                          [px - 5, ptop + 10]], closed=True, fc=DARK, ec=DARK, lw=0.5))
    ax.add_patch(Rectangle((px - 2.5, ptop + 10), 5, 14, fc=DARK, ec=DARK, lw=0.5))
    ax.text(px + 7, ptop + 17, "robot-held\nprobe replica", ha="left", va="center", fontsize=5.0,
            linespacing=1.1)
    ax.add_patch(Rectangle((10, 30), 18, 7, fc="white", ec=DARK, lw=0.7))    # syringe
    ax.plot([6, 10], [33.5, 33.5], color=DARK, lw=1.0)
    ax.plot([28, 34], [33.5, 33.5], color=DARK, lw=0.7)
    ax.text(19, 27, "syringe", ha="center", va="top", fontsize=5.0)
    arrow(ax, (19, 47), (19, 41), color=GRAY, lw=0.7, style="<|-|>", ms=4)
    ax.text(19, 49, "inject / withdraw\nevery 4.6 s", ha="center", va="bottom", fontsize=5.0,
            color=DARK, linespacing=1.1)
    arrow(ax, (44, top[8] - 3), (44, top[8] + 7), color=GRAY, lw=0.7, style="<|-|>", ms=4)
    y = 20
    for letter, text in (("A", "undisturbed hold"),
                         ("B", "syringe steps, force hold on"),
                         ("C", "same steps, force hold off (placebo)")):
        ax.text(2, y, letter, ha="left", va="center", fontsize=5.4, fontweight="bold")
        ax.text(8, y, text, ha="left", va="center", fontsize=5.0)
        y -= 7

    figure.savefig(path)
    plt.close(figure)
    return path


# ------------------------------------------------------------- setup photo
#: Crops as fractions of each photo (left, top, right, bottom). The overview
#: loses the ceiling; the close-up keeps the arm, the probe on the phantom and
#: the syringe scale, and drops the mannequin legs below the syringe.
SETUP_CROPS = {
    "fh_setup_overview.jpg": (0.00, 0.085, 1.00, 1.00),
    "fh_setup_closeup.jpg": (0.13, 0.015, 0.83, 0.78),
}


def setup_figure(path: str, height_in: float = 2.5, gap_in: float = 0.06,
                 dpi: int = 300) -> str:
    """Two bench photographs side by side at one height: (a) the bench, (b) the
    probe on the phantom. Composed with PIL rather than matplotlib so the JPEG
    pixels are not resampled twice."""
    from PIL import Image, ImageDraw, ImageFont

    panels = []
    for name, crop in SETUP_CROPS.items():
        image = Image.open(os.path.join(PHOTOS, name)).convert("RGB")
        w, h = image.size
        image = image.crop((int(crop[0] * w), int(crop[1] * h),
                            int(crop[2] * w), int(crop[3] * h)))
        target_h = int(round(height_in * dpi))
        image = image.resize((int(round(image.width * target_h / image.height)), target_h),
                             Image.LANCZOS)
        panels.append(image)

    gap = int(round(gap_in * dpi))
    width = sum(p.width for p in panels) + gap * (len(panels) - 1)
    sheet = Image.new("RGB", (width, panels[0].height), "white")
    x = 0
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("times.ttf", int(round(8.0 / 72 * dpi)))
    except OSError:
        font = ImageFont.load_default()
    for index, panel in enumerate(panels):
        sheet.paste(panel, (x, 0))
        label = f"({'ab'[index]})"
        pad = int(round(0.03 * dpi))
        box = draw.textbbox((0, 0), label, font=font)
        draw.rectangle((x + pad, pad, x + pad + box[2] - box[0] + 2 * pad,
                        pad + box[3] - box[1] + 2 * pad), fill="white")
        draw.text((x + 2 * pad, 2 * pad - box[1]), label, fill=DARK, font=font)
        x += panel.width + gap
    sheet.save(path, quality=90, optimize=True, dpi=(dpi, dpi))
    return path

