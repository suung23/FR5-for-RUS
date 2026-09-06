#!/usr/bin/env python3
"""Merged poster Figure 1: the architecture diagram over the action-token
specification table, composited into a single image.

    python3 Unet_seg/scripts/poster_overview_table.py

For the two-page poster abstract, the framework overview (Figure 1) and the
action-token specification (Table 1) of the full paper are combined into one
figure. The architecture artwork sits on top; the four action states, their
triggers, safety rationale, and deterministic interpretation are drawn beneath
it as a compact table in the same serif family as the manuscript body.
"""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from PIL import Image                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Paper", "figures")
INK = "#000000"

COLS = ["Action state", "Trigger", "Safety rationale", "Control interpretation"]
ROWS = [
    ["check-filling", "contrast ≤ 0", "image failure, not a pose error",
     "gate: no lateral motion"],
    ["move-left / move-right", "contrast > 0, |ê| ≥ 8.1 px",
     "direction reliable above noise", "vₓ = −k ê, other axes zero"],
    ["hold", "contrast > 0, |ê| < 8.1 px", "direction within noise",
     "deadband: zero lateral twist"],
]
COL_W = [0.20, 0.24, 0.30, 0.26]


def font():
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for fam in ("Times New Roman", "Times", "DejaVu Serif"):
        if fam in have:
            return fam
    return "DejaVu Serif"


def main() -> int:
    plt.rcParams["font.family"] = font()
    plt.rcParams["mathtext.fontset"] = "stix"

    arch = Image.open(os.path.join(FIGDIR, "fig_overview.png")).convert("RGB")
    ar = arch.width / arch.height           # ~2.9 : 1

    FIG_W = 9.2                              # inches
    arch_h = FIG_W / ar
    table_h = 1.55
    gap = 0.12
    FIG_H = arch_h + gap + table_h

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    ax_img = fig.add_axes([0.0, (gap + table_h) / FIG_H, 1.0, arch_h / FIG_H])
    ax_img.imshow(np.asarray(arch), interpolation="lanczos")
    ax_img.axis("off")

    ax_tab = fig.add_axes([0.02, 0.0, 0.96, table_h / FIG_H])
    ax_tab.axis("off")
    tbl = ax_tab.table(cellText=ROWS, colLabels=COLS, colWidths=COL_W,
                       cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1.0, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#b0b0b0")
        cell.set_linewidth(0.6)
        cell.get_text().set_color(INK)
        if r == 0:
            cell.get_text().set_fontweight("bold")
            cell.set_facecolor("#f2f2f2")
        if c == 0 and r > 0:
            cell.get_text().set_fontstyle("italic")

    out = os.path.join(FIGDIR, "fig_poster_overview.png")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight",
                pad_inches=0.03)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
