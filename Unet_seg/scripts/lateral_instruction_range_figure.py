#!/usr/bin/env python3
"""Figure 3 of the lateral-instruction manuscript: the attained range of Q.

(a) The interval Q actually attains against its nominal domain [0, 1].  (b) The interval each active sub-score attains and its
share of the resulting range.

The numbers reproduce Tables 2-3 of the manuscript (attained intervals, widths,
shares).  The central-80% *endpoints* are not tabulated anywhere; the values
below were recovered from the previous rendering of this figure and normalised
to the tabulated widths (0.450 and 0.123).  Replace them from the Q-value dump
if it is ever re-exported.

Style: black text throughout; dark navy for the image-reading terms and the
reformulated aggregation, grey for everything else.  Panel letters sit above
the axes so they cannot cover the plots.
"""
from __future__ import annotations

from pathlib import Path

NAVY = "#1f3864"
NAVY_PALE = "#ccd5e3"
GREY = "#9e9e9e"
GREY_PALE = "#e3e3e3"
INK = "#000000"

#: (label, attained interval, central 80%, pale colour, solid colour), top row first.
AGGREGATIONS = (
    ("$Q$\n(weighted geometric,\nimage-first)", (0.14, 0.78), (0.270, 0.720), NAVY_PALE, NAVY),
)

#: (name, attained interval, share of the range of Q, colour), Table 2 order.
SUB_SCORES = (
    ("lumen contrast", (0.000, 1.000), "47.6%", NAVY),
    ("lumen centering", (0.001, 0.999), "44.8%", NAVY),
    ("boundary sharpness", (0.237, 0.451), "4.4%", GREY),
    ("mask completeness", (0.250, 1.000), "3.2%", GREY),
    ("segmentation confidence", (0.992, 0.999), "0.0%", GREY),
)


def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "text.color": INK,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def build():
    import matplotlib.pyplot as plt

    _style()
    figure, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.55),
                                  gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- (a) attained range of Q under the two aggregations ----------
    from matplotlib.patches import Patch

    step = 1.3
    for row, (label, attained, central, pale, solid) in enumerate(AGGREGATIONS):
        y = step * (len(AGGREGATIONS) - 1 - row)
        a.barh(y, attained[1] - attained[0], left=attained[0], height=0.46,
               color=pale, zorder=2)
        a.barh(y, central[1] - central[0], left=central[0], height=0.46,
               color=solid, zorder=3)
        for value in attained:
            a.text(value, y - 0.33, f"{value:.2f}", ha="center", va="top",
                   fontsize=8, color=INK, clip_on=False)
        a.text((attained[0] + attained[1]) / 2, y + 0.36,
               f"width {attained[1] - attained[0]:.2f}", ha="center", va="bottom",
               fontsize=8.5, fontweight="bold", color=INK)
    a.set_yticks([0.0])
    a.set_yticklabels([entry[0] for entry in AGGREGATIONS], fontsize=8.5)
    a.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    a.set_xlim(0.0, 1.0)
    a.set_ylim(-0.72, 0.95)
    a.tick_params(labelsize=8)
    a.set_xlabel("$Q$", fontsize=9)
    a.legend(handles=[Patch(facecolor=GREY_PALE, edgecolor="#bbbbbb",
                            label="attained range"),
                      Patch(facecolor=GREY, label="central 80%")],
             loc="lower left", fontsize=7.5, frameon=False,
             handlelength=1.4, handletextpad=0.6, labelspacing=0.35,
             borderaxespad=0.2)
    a.text(-0.02, 1.03, "(a)", transform=a.transAxes, ha="right", va="bottom",
           fontsize=10, fontweight="bold", color=INK)

    # ---- (b) attained interval and share per sub-score ---------------
    rows = len(SUB_SCORES)
    for row, (name, interval, share, colour) in enumerate(SUB_SCORES):
        y = rows - 1 - row
        b.barh(y, max(interval[1] - interval[0], 0.004), left=interval[0],
               height=0.52, color=colour, zorder=2)
        b.text(1.035, y, share, ha="left", va="center", fontsize=8.5, color=INK,
               fontweight="bold" if colour == NAVY else "normal", clip_on=False)
    b.set_yticks(range(rows))
    b.set_yticklabels([entry[0] for entry in reversed(SUB_SCORES)], fontsize=8.5)
    b.set_xlim(0.0, 1.0)
    b.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    b.tick_params(labelsize=8)
    b.set_xlabel("attained interval of the sub-score", fontsize=9)
    b.text(1.0, 1.03, "share of $Q$ range $\\rightarrow$", transform=b.transAxes,
           ha="right", va="bottom", fontsize=7.8, color=INK)
    b.text(-0.02, 1.03, "(b)", transform=b.transAxes, ha="right", va="bottom",
           fontsize=10, fontweight="bold", color=INK)

    figure.tight_layout(w_pad=3.2)
    return figure


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(__file__).resolve().parents[1] / "experiments" / "lateral_instruction"
    output.mkdir(parents=True, exist_ok=True)
    figure = build()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"fig_range.{suffix}", dpi=300,
                       facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output / 'fig_range.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
