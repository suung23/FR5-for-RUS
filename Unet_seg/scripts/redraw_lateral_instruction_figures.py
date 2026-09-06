#!/usr/bin/env python3
"""Redraw the lateral-instruction figures from cached artifacts, without the model.

fig_axes uses only the measured sector mask (configs/roi/pfus_sector_256.npy);
fig_agreement and fig_validity use frames.csv and lateral_instruction.json;
fig_range is pure constants (see lateral_instruction_range_figure.py).  Only
fig_examples needs the checkpoint and dataset and is NOT redrawn here — rerun
lateral_instruction_figures.py on the training machine after style changes.

    python scripts/redraw_lateral_instruction_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

UNET = Path(__file__).resolve().parents[1]
OUTPUT = UNET / "experiments" / "lateral_instruction"      # cached artifacts (json/csv)
FIGURES = UNET.parent / "Paper" / "figures"               # every paper figure lives here


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import lateral_instruction as analysis
    import lateral_instruction_figures as drawn
    import lateral_instruction_range_figure as range_figure

    roi = np.load(UNET / "configs" / "roi" / "pfus_sector_256.npy") > 0
    stats = json.loads((OUTPUT / "lateral_instruction.json").read_text())
    with (OUTPUT / "frames.csv").open(newline="") as handle:
        rows = [{**r, "truth_px": float(r["truth_px"]),
                 "instruction_px": float(r["instruction_px"])}
                for r in csv.DictReader(handle)]

    jobs = (
        (drawn.figure_axes_and_instruction(roi), "fig_axes", 300),
        (range_figure.build(), "fig_range", 300),
        (analysis.figure_agreement(rows, stats), "fig_agreement", 220),
        (analysis.figure_validity(stats), "fig_validity", 220),
    )
    for figure, stem, dpi in jobs:
        for suffix in ("png", "pdf"):
            figure.savefig(FIGURES / f"{stem}.{suffix}", dpi=dpi,
                           facecolor="white", bbox_inches="tight")
        plt.close(figure)
        print("wrote", FIGURES / f"{stem}.png")
    print("skipped fig_examples (needs the checkpoint; rerun "
          "lateral_instruction_figures.py on the training machine)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
