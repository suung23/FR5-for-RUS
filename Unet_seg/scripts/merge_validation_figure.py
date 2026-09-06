#!/usr/bin/env python3
"""Merge the two validation panels into one figure for the compact paper.

    python3 Unet_seg/scripts/merge_validation_figure.py

Figure 3 of the conference version stacks (a) the estimated-vs-ground-truth
agreement panel and (b) the wrong-direction validity curve into a single
image, so the five-page draft uses one validation figure instead of two.
Both source PNGs are produced by redraw_lateral_instruction_figures.py; this
script only composites them, so it must be re-run after those regenerate.
"""
from __future__ import annotations

import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Paper", "figures")
GAP = 26          # white gutter between the two panels (px)
PAD = 12          # outer white padding (px)


def main() -> int:
    top = Image.open(os.path.join(FIGDIR, "fig_agreement.png")).convert("RGB")
    bot = Image.open(os.path.join(FIGDIR, "fig_validity.png")).convert("RGB")
    # scale both to a common width (the wider of the two)
    width = max(top.width, bot.width)

    def fit(im):
        if im.width == width:
            return im
        h = round(im.height * width / im.width)
        return im.resize((width, h), Image.LANCZOS)

    top, bot = fit(top), fit(bot)
    canvas = Image.new("RGB", (width + 2 * PAD,
                               top.height + bot.height + GAP + 2 * PAD), "white")
    canvas.paste(top, (PAD, PAD))
    canvas.paste(bot, (PAD, PAD + top.height + GAP))
    out = os.path.join(FIGDIR, "fig_validation.png")
    canvas.save(out, dpi=(300, 300))
    print("wrote", out, canvas.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
