#!/usr/bin/env python3
"""Two-page KOSMI poster abstract, derived from the five-page manuscript.

    python3 Paper/build_poster_abstract.py

Same KOSMI 연제논문 format as the full paper (single-column front matter, two
-column body, 신명조 / Times New Roman, 10 pt, 장평 95% / 자간 −5%). This is a
minimal poster version: Figure 1 of the full paper and its action-token table
are merged into one figure; the beam-axis B-mode figure and the quantitative
results table are kept; everything is compressed to fit two pages including
references. Authors and affiliations are the finalised set (do not edit).

Writes "<paper title>_poster.docx" (the submission file name); the five-page file is left
untouched.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_lateral_instruction_manuscript as base   # noqa: E402
from build_lateral_instruction_manuscript import (     # noqa: E402
    para, head, equation, wide_figure, table, _span, columns, style_doc,
    page_numbers, S, VOCAB, SIGMA, TH, SEG_DOMAIN, OVERALL, PAPER,
)
from docx import Document                              # noqa: E402
from docx.enum.section import WD_SECTION               # noqa: E402
from docx.shared import Pt                             # noqa: E402

#: File name chosen by the user for submission (the paper title + "_poster").
OUT = os.path.join(PAPER, "An Explainable Action-Token Generation Framework Based on "
                   "Bladder Ultrasound Segmentation and Image Quality Assessment_poster.docx")

# ---- finalised author / affiliation block (do not edit) --------------------
AUTHORS = ("++Seong Jeong++^1,2,3^, Minsung Kim^2,3,4^, Dongho Yee^2,3,4^, "
           "Yechan Seo^1,2,3^, Juahn Oh^1,2,5^, Hyoun-Joong Kong^1,5,6,*^")
AFFILS = (
    "^1^ Department of Medicine, Seoul National University, Seoul, Republic of Korea",
    "^2^ Department of Transdisciplinary Medicine, Seoul National University Hospital, Seoul, Republic of Korea",
    "^3^ Rosota Inc., Seoul, Republic of Korea",
    "^4^ Department of Mechanical Engineering, Seoul National University, Seoul, Republic of Korea",
    "^5^ Eulji University College of Medicine, Daejeon, Republic of Korea",
    "^6^ ICMIT, Seoul National University Hospital, Seoul, Republic of Korea",
)


def front_matter(doc):
    para(doc, "An Explainable Action-Token Generation Framework Based on Bladder "
              "Ultrasound Segmentation and Image Quality Assessment",
         size=12.0, bold=True, align="center", after=3)
    para(doc, AUTHORS, size=10.0, align="center", after=3)
    for line in AFFILS:
        para(doc, line, size=8.0, align="center", after=0.4)
    para(doc, "*Corresponding Author", size=8.0, italic=True, align="center", after=0.4)
    doc.add_paragraph().paragraph_format.space_after = Pt(3)

    _abs = para(doc,
         "**Abstract:** "
         "During morcellation in holmium laser enucleation of the prostate (HoLEP), a "
         "suprapubic probe must stay over the bladder lumen; automating that view needs "
         "an explicit, auditable representation of image state, which a mask alone is "
         "not. We convert segmentation and image-quality assessment into a "
         "safety-constrained, machine-readable action token: a U-Net segments the lumen, "
         "a transparent quality function *Q* summarises the frame, and a discrete action "
         "state, one of *move-left*, *move-right*, *hold*, or *check-filling*, is paired "
         "with a continuous lateral displacement *ê*. The vocabulary is bounded by what "
         "*Q* distinguishes: against frame-to-frame jitter its attained range "
         "[0.14, 0.78] separates 47 distinguishable states. Both boundaries are derived, "
         "not tuned: an anechoicity gate assigns *check-filling* and an uncertainty "
         f"deadband assigns *hold* below {TH['5%']['k']:.2f}σ = {TH['5%']['px']:.1f} px "
         f"of centroid noise, σ = {SIGMA:.2f} px. On {S['n_frames']:,} laterally "
         f"translated held-out frames the action state is "
         f"{VOCAB[3]['accuracy'] * 100:.1f}% correct with "
         f"{VOCAB[3]['direction_error'] * 100:.2f}% direction error and displacement "
         f"follows slope {OVERALL['slope']:.3f}; graded magnitude classes drop accuracy "
         f"to {VOCAB[5]['accuracy'] * 100:.1f}% and {VOCAB[7]['accuracy'] * 100:.1f}%. "
         "The token layer targets HoLEP probe guidance; closed-loop validation remains "
         "future work.",
         size=10.0)
    _abs.paragraph_format.line_spacing = 1.2
    para(doc, "**Keywords:** bladder ultrasound segmentation, image quality assessment, "
              "explainable artificial intelligence, action token, robotic probe control, "
              "HoLEP morcellation", size=10.0, align="left", before=1, after=2)
    doc.add_paragraph().paragraph_format.space_after = Pt(3)


def body(doc):
    head(doc, "I. Introduction")
    para(doc,
         "During the morcellation phase of holmium laser enucleation of the prostate "
         "(HoLEP), a suprapubic ultrasound probe must be kept over the bladder lumen [1]. "
         "Deep networks segment the bladder accurately [3, 4] and guidance systems steer "
         "toward a standard view [5], but a segmentation mask is not directly usable by a "
         "downstream controller, and end-to-end control hides why a motion is issued and is "
         "hard to audit under segmentation uncertainty. We "
         "insert an explainable, safety-constrained layer that turns image-derived state "
         "into a structured action token for deterministic downstream decoding (Figure 1).")

    head(doc, "II. Methods")
    para(doc,
         "A U-Net [2] segments the bladder lumen on 256 × 256 B-mode frames from PFUS1 "
         "[6, 7], a transperineal pelvic-floor dataset (not intraoperative HoLEP imaging); "
         "measurements use two patient-disjoint held-out sets (1,661 and 1,292 frames). An "
         "interpretable image state (lumen contrast, centring, boundary sharpness over the "
         "insonified sector) feeds a transparent weighted geometric-mean quality score,")
    equation(doc, "Q = exp( Σ w~i~ ln s~i~ / Σ w~i~ )", "1")
    para(doc,
         "whose smallest term dominates. At frame *t* the framework emits a structured "
         "action token")
    equation(doc, "τ~t~ = ( a~t~, ê~t~, Q~t~ )", "2")
    para(doc,
         "with *a*~t~ the discrete action state, *ê*~t~ = *A* − *ĉ*~t~ the continuous "
         "lateral displacement (beam axis *A* minus predicted lumen-centroid abscissa "
         "*ĉ*; Figure 2), and *Q*~t~ the quality score. A fixed, auditable rule decodes "
         "the token downstream with no learned policy. Two derived, untuned boundaries fix "
         "the states: a **physical gate** (a non-anechoic lumen cannot be a filled "
         "bladder → *check-filling*) and an **uncertainty deadband** (offsets below "
         f"{TH['5%']['k']:.2f}σ = {TH['5%']['px']:.1f} px of the σ = {SIGMA:.2f} px "
         "centroid noise cannot be signed → *hold*). The token is exercised on laterally "
         "translated frames that carry the lumen across the beam axis so that *e* changes "
         "sign.")
    wide_figure(doc, 2, "fig_axes.png",
                "(a) Only lateral sliding *v*~x~ leaves the imaging plane invariant among "
                "the plane-preserving axes. (b) The lateral displacement on one B-mode "
                "frame: *A* is the beam axis, *c* the lumen centroid, *e* = *A* − *c*.",
                8.8)

    head(doc, "III. Results")
    para(doc,
         "In the distended-bladder working domain Dice reaches "
         f"{SEG_DOMAIN['val']['dice']['mean']:.3f} / "
         f"{SEG_DOMAIN['test']['dice']['mean']:.3f} (val / test). *Q* is defined on "
         "[0, 1] but attains only [0.14, 0.78] and, against a frame-to-frame jitter of "
         "0.0095, separates 47 distinguishable states within its central operating range "
         "(Table 1). On "
         f"{S['n_frames']:,} laterally translated held-out frames the discrete action "
         f"state is classified at {VOCAB[3]['accuracy'] * 100:.1f}% "
         f"({VOCAB[3]['direction_error'] * 100:.2f}% direction error) and the continuous "
         f"displacement at slope {OVERALL['slope']:.3f} (median "
         f"{OVERALL['abs_error_median']:.2f} px); wrong directions occur only near the "
         "axis and follow the Gaussian tail Φ(−|*e*|/σ), licensing the derived "
         f"{TH['5%']['px']:.1f} px *hold* deadband. Quantising magnitude into graded "
         f"classes lowers accuracy to {VOCAB[5]['accuracy'] * 100:.1f}% and "
         f"{VOCAB[7]['accuracy'] * 100:.1f}%, so displacement is kept continuous.")
    seg_rows = [
        ["Segmentation Dice, working domain (val / test)",
         f"{SEG_DOMAIN['val']['dice']['mean']:.3f} / {SEG_DOMAIN['test']['dice']['mean']:.3f}"],
        ["Attained *Q* range (width; central 80%)", "0.64; 0.45"],
        ["Distinguishable states (full; central 80%)", "67; 47"],
        [f"Action-state accuracy ({VOCAB[3]['size']} states); direction error",
         f"{VOCAB[3]['accuracy'] * 100:.1f}%; {VOCAB[3]['direction_error'] * 100:.2f}%"],
        ["Displacement estimate (slope; median error)",
         f"{OVERALL['slope']:.3f}; {OVERALL['abs_error_median']:.2f} px"],
        [f"Graded-magnitude accuracy ({VOCAB[5]['size']}; {VOCAB[7]['size']} states)",
         f"{VOCAB[5]['accuracy'] * 100:.1f}%; {VOCAB[7]['accuracy'] * 100:.1f}%"],
    ]
    table(doc, 1, "Main quantitative results on the held-out and laterally translated "
                  "frames.",
          ["Metric", "Value"], seg_rows, widths=[5.2, 2.6], size=7.5)

    head(doc, "IV. Conclusion")
    para(doc,
         "An explainable, quality-aware action-token layer bridges bladder ultrasound "
         "segmentation and structured downstream action handling, its *hold* and "
         "*check-filling* states providing an uncertainty deadband and a physics-based "
         "gate that a downstream system can interpret deterministically. The study "
         "validates image-to-token generation, not closed-loop control or clinical "
         "deployment; re-estimating the constants on intraoperative suprapubic HoLEP "
         "imaging and evaluating downstream integration remain future work.")

    head(doc, "References")
    refs = [
        "1. T. Jang, H.-J. Kong, C. Baek, J. Kim, M. S. Choo, S.-J. Oh. Effect of "
        "Self-Training Using Virtual Reality Head-Mounted Display Simulator on the "
        "Acquisition of Holmium Laser Enucleation of the Prostate Surgical Skills. "
        "International Neurourology Journal 2024;28(2):138–146.",
        "2. O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for "
        "Biomedical Image Segmentation. In: MICCAI 2015, LNCS 9351, Springer, 2015, "
        "pp. 234–241.",
        "3. M. Saini, Y. Jiang, T. Gangopadhyay, D. P. Rosen, A. Alizad, M. Fatemi. "
        "BWS-Net: An Optimal Deep Learning Architecture for the Anterior Bladder Wall "
        "Segmentation using Ultrasound Imaging. IEEE J. Biomed. Health Inform. 2026. "
        "doi:10.1109/JBHI.2026.3675965.",
        "4. Z. Song, M. Asiedu, S. Wang, et al. Memory-efficient low-compute "
        "segmentation algorithms for bladder-monitoring smart ultrasound devices. "
        "Scientific Reports 2023;13:16450.",
        "5. H.-L. Hsu, M. Zahiri, G. Y. Li, et al. Active guidance in ultrasound bladder "
        "scanning using reinforcement learning. Scientific Reports 2026;16:5273.",
        "6. D. Solís-Martín, J. A. Sainz, J. Galán-Páez, J. Borrego-Díaz, "
        "J. A. García-Mejido. PFUS1: Premier pelvic floor ultrasound segmentation "
        "dataset. Data in Brief 2026;64:112346.",
        "7. J. A. García-Mejido, D. Solís-Martín, M. Martín-Morán, et al. Applicability "
        "of deep learning to dynamically identify the different organs of the pelvic "
        "floor in the midsagittal plane. Int. Urogynecol. J. 2024;35(12):2285–2293.",
        "8. Trinkler, Dietrich. Ultrasound of the Urinary Bladder. In: EFSUMB Course "
        "Book, European Federation of Societies for Ultrasound in Medicine and Biology, "
        "2019.",
    ]
    for r in refs:
        para(doc, r, size=9.0, after=0.6)
    para(doc, "[TO BE COMPLETED] HoLEP-morcellation, visual-servoing and IQA "
              "citations to be added before submission.", size=9.0, italic=True)


def build():
    doc = Document()
    style_doc(doc)
    page_numbers(doc)
    columns(doc.sections[0], 1)
    front_matter(doc)
    # merged Figure 1 (architecture + action-token specification) spans the page
    wide_figure(doc, 1, "fig_poster_overview.png",
                "The action-token generation framework and its four action states. An "
                "ultrasound frame is segmented (U-Net), summarised by the image-quality "
                "function *Q*, and emitted as a structured token (*a*, *ê*, *Q*) decoded "
                "by a fixed downstream rule (top). The table lists each state's trigger, "
                "safety rationale, and deterministic interpretation; the dashed downstream "
                "execution is a potential future integration, not validated.", 13.8)
    body(doc)

    doc.save(OUT)
    print("wrote", OUT)
    print("embedded", len(base.USED_FIGURES), "figures from", base.FIGDIR)


if __name__ == "__main__":
    build()
