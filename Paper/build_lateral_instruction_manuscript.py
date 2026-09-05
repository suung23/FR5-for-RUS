#!/usr/bin/env python3
"""Lateral-instruction manuscript (.docx) — text, figures and tables.

    python3 docs/build_lateral_instruction_manuscript.py
    soffice --headless --convert-to pdf Lateral_Instruction_Manuscript.docx

Format follows the HoLEP contact-force extended abstract: A4, two columns,
Times New Roman, numbered tables and figures with captions beneath. Every number
in the text is read from experiments/lateral_instruction/lateral_instruction.json
rather than typed, so the manuscript cannot drift from the analysis that
produced it.
"""
from __future__ import annotations

import json
import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO, "Unet_seg", "experiments", "lateral_instruction")
STATS = json.load(open(os.path.join(FIGDIR, "lateral_instruction.json")))
#: Segmentation metrics of the same checkpoint on the same frame sets, from the
#: volume-gated audit; the manuscript's Table 2 is read from here, not typed.
AUDIT = json.load(open(os.path.join(REPO, "Unet_seg", "experiments", "quality_retro",
                                    "audit", "audit.json")))
OUT = os.path.join(REPO, "Lateral_Instruction_Manuscript.docx")

BODY_PT = 9.7
SERIF = "Times New Roman"
RULE = "808080"

S = STATS
SEG_ALL = {s: next(e for e in AUDIT["sweep"][s] if e["floor"] == 0.0)
           for s in ("val", "test")}
SEG_DOMAIN = AUDIT["segmentation"]
FLOOR = AUDIT["min_gt_area_ratio"]
OVERALL, BY = S["overall"], S["by_split"]
SIGMA = S["sigma_px"]
NAT = S["natural_poses"]
CURVE = [c for c in S["validity_curve"] if c["n"]]
TH = S["thresholds"]


# --------------------------------------------------------------- formatting
def style_doc(doc):
    st = doc.styles["Normal"]
    st.font.name = SERIF
    st.font.size = Pt(BODY_PT)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), SERIF)
    pf = st.paragraph_format
    pf.space_before = pf.space_after = Pt(0)
    pf.line_spacing = 1.06
    for s in doc.sections:
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.top_margin = s.bottom_margin = Cm(1.9)
        s.left_margin = s.right_margin = Cm(1.8)
        s.header_distance = s.footer_distance = Cm(1.0)


def columns(section, count, space_cm=0.7):
    """Set the column count on a section; python-docx has no API for it."""
    cols = section._sectPr.xpath("./w:cols")[0]
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), str(int(space_cm * 567)))


def page_numbers(doc):
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    r.font.name, r.font.size = SERIF, Pt(9)
    for kind, txt in (("fldChar", "begin"), ("instrText", " PAGE "), ("fldChar", "end")):
        el = OxmlElement("w:" + kind)
        if kind == "fldChar":
            el.set(qn("w:fldCharType"), txt)
        else:
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        r._r.append(el)


def rich(p, text, *, size=BODY_PT, bold=False, italic=False, color=None):
    """Split '**bold**', '*italic*', '~sub~', '^sup^' into runs."""
    import re
    for tok in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*|~[^~]+?~|\^[^^]+?\^)", text):
        if not tok:
            continue
        b, i, sub, sup = bold, italic, False, False
        if tok.startswith("**") and tok.endswith("**"):
            tok, b = tok[2:-2], True
        elif tok.startswith("*") and tok.endswith("*"):
            tok, i = tok[1:-1], True
        elif tok.startswith("~") and tok.endswith("~"):
            tok, sub = tok[1:-1], True
        elif tok.startswith("^") and tok.endswith("^"):
            tok, sup = tok[1:-1], True
        r = p.add_run(tok)
        r.font.name, r.font.size = SERIF, Pt(size)
        r.font.bold, r.font.italic = b, i
        r.font.subscript, r.font.superscript = sub, sup
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
    return p


def para(doc, text, *, size=BODY_PT, bold=False, italic=False, align="just",
         before=0, after=3.2, indent=0.0, color=None):
    p = doc.add_paragraph()
    p.alignment = {"just": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    pf = p.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.left_indent = Cm(indent)
    rich(p, text, size=size, bold=bold, italic=italic, color=color)
    return p


def head(doc, text, level=1):
    p = para(doc, text, size={1: 10.0, 2: 9.4}[level], bold=True, align="left",
             before=6.5 if level == 1 else 4.5, after=2.0)
    p.paragraph_format.keep_with_next = True
    return p


def equation(doc, text, tag):
    """Equation flush left, its number right-aligned exactly on the column edge."""
    from docx.enum.text import WD_TAB_ALIGNMENT
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = pf.space_after = Pt(4)
    pf.tab_stops.add_tab_stop(Cm(COL_CM), WD_TAB_ALIGNMENT.RIGHT)
    rich(p, text, size=BODY_PT, italic=True)
    p.add_run("\t")
    rich(p, "(%s)" % tag, size=BODY_PT)
    return p


def figure(doc, number, filename, caption, width_cm):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(os.path.join(FIGDIR, filename), width=Cm(width_cm))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.space_after = Pt(6)
    rich(c, "**Figure %d.** %s" % (number, caption), size=7.9)


#: Text width of one body column [cm]: (page - margins - gutter) / 2.
COL_CM = (21.0 - 2 * 1.8 - 0.7) / 2


def _fix_layout(t, widths):
    """Pin the table to the column width.

    ``autofit = False`` alone is not enough: without an explicit ``tblW`` and a
    fixed layout algorithm, Writer re-flows the table to its content and lets it
    run past the column edge.
    """
    tbl_pr = t._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)
    total = OxmlElement("w:tblW")
    total.set(qn("w:w"), str(int(sum(widths) * 567)))
    total.set(qn("w:type"), "dxa")
    tbl_pr.append(total)
    grid = t._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for cell, width in zip(grid.findall(qn("w:gridCol")), widths):
            cell.set(qn("w:w"), str(int(width * 567)))
    for row in t.rows:
        for index, width in enumerate(widths):
            row.cells[index].width = Cm(width)


def _scale(widths, full_width=False):
    """Scale a width recipe to fill one column, or the whole text block."""
    target = (21.0 - 2 * 1.8 - 0.05) if full_width else (COL_CM - 0.05)
    factor = target / sum(widths)
    return [w * factor for w in widths]


def _span(doc, count):
    """Continuous section break that changes the column count in place."""
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.9)
    section.left_margin = section.right_margin = Cm(1.8)
    columns(section, count)
    return section


def wide_figure(doc, number, filename, caption, width_cm=17.0):
    """A figure spanning both columns.

    The wide panels are drawn at a 2.5:1 aspect; squeezed into one 8.2 cm column
    their axis labels fall below 5 pt. Spanning the page is the difference
    between a figure a reviewer can read and one they cannot.
    """
    _span(doc, 1)
    figure(doc, number, filename, caption, width_cm)
    _span(doc, 2)


def wide_table(doc, *args, **kwargs):
    """A table spanning both columns, for row sets a column cannot hold."""
    _span(doc, 1)
    kwargs["full_width"] = True
    table(doc, *args, **kwargs)
    _span(doc, 2)


def table(doc, number, caption, header, rows, widths=None, size=7.8, full_width=False):
    """Caption above, then a plain ruled table in the reference's style."""
    c = doc.add_paragraph()
    c.paragraph_format.space_before = Pt(6)
    c.paragraph_format.space_after = Pt(2)
    c.paragraph_format.keep_with_next = True
    rich(c, "**Table %d.** %s" % (number, caption), size=7.9)

    t = doc.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for index, text in enumerate(header):
        cell = t.rows[0].cells[index]
        cell.paragraphs[0].paragraph_format.space_after = Pt(0.5)
        rich(cell.paragraphs[0], text, size=size, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for index, text in enumerate(row):
            cells[index].paragraphs[0].paragraph_format.space_after = Pt(0.5)
            rich(cells[index].paragraphs[0], str(text), size=size)
    _fix_layout(t, _scale(widths or [1] * len(header), full_width))
    _rule_table(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(3)
    return t


def _rule_table(t):
    """Horizontal rules only: a header line and hairlines between rows."""
    for r_index, row in enumerate(t.rows):
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "bottom"):
                el = OxmlElement("w:" + edge)
                heavy = (r_index == 0 and edge == "top") or (r_index == 0 and edge == "bottom")
                el.set(qn("w:val"), "single")
                el.set(qn("w:sz"), "8" if heavy else "4")
                el.set(qn("w:color"), "000000" if heavy else RULE)
                borders.append(el)
            for edge in ("left", "right"):
                el = OxmlElement("w:" + edge)
                el.set(qn("w:val"), "nil")
                borders.append(el)
            tc_pr.append(borders)


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


# --------------------------------------------------------------------- body
def front_matter(doc):
    para(doc, "From Segmentation to Instruction: Language-Level Operator Guidance Built on "
              "an Image-Quality Function in Transabdominal Bladder Ultrasound for HoLEP "
              "Morcellation",
         size=13.5, bold=True, align="center", after=7)
    para(doc, "Seong Jeong^1, 2, 5^, Minsung Kim^1,2,4^, Dongho Yee^1,2,4^, "
              "Yechan Seo^1,2,5^, Juahn Oh^1,2,3^, Hyoun-Joong Kong^1,2,5,*^",
         size=10.0, align="center", after=6)
    for line in (
        "^1^ Department of Transdisciplinary Medicine, Seoul National University Hospital, Seoul, Republic of Korea",
        "^2^ Rosota Inc., Seoul, Republic of Korea",
        "^3^ Eulji University College of Medicine, Daejeon, Republic of Korea",
        "^4^ Department of Mechanical Engineering, Seoul National University, Seoul, Republic of Korea",
        "^5^ Department of Medicine, Seoul National University, Seoul, Republic of Korea",
    ):
        para(doc, line, size=8.4, align="left", after=0.6)
    doc.add_paragraph().paragraph_format.space_after = Pt(3)

    para(doc, "Abstract", size=9.8, bold=True, align="left", after=2)
    para(doc,
         "During the morcellation phase of holmium laser enucleation of the prostate "
         "(HoLEP) an operator holds a suprapubic probe and must keep the bladder lumen in "
         "view. We present a pipeline that turns each transabdominal frame into guidance "
         "for that operator: a U-Net segments the bladder lumen, a transparent "
         "image-quality function *Q* judges the frame from the segmentation and the image, "
         "and the judgment is emitted as a **language instruction** naming the next "
         "corrective movement — *move left*, *move right*, *hold*, or *check bladder "
         "filling*. Language is the deliberate output format: a human operator can follow "
         "the words as displayed or spoken, and each word carries a defined condition and "
         "axis, so a robotic probe holder can translate it into a twist command by lookup. "
         "An instruction is only as fine as the function issuing it, so we first measure "
         "what *Q* can judge. As originally specified *Q* attains [0.70, 1.00], 30% of its "
         "nominal domain, and resolves only 9 levels above its own frame-to-frame noise — "
         "too few to ground an instruction set; reformulated, it attains [0.14, 0.78] and "
         "resolves 47 levels. The vocabulary is then derived rather than chosen: one "
         "boundary is physical (a lumen that is not anechoic cannot be a filled bladder — "
         "*check bladder filling*), the other statistical (a lateral offset below "
         f"{TH['5%']['k']:.2f}σ of the segmentation's centroid noise, σ = {SIGMA:.2f} px, "
         "cannot be signed reliably — *hold*). On "
         f"{S['n_frames']:,} laterally displaced held-out frames the three-word vocabulary "
         "is emitted correctly on 85.0% with a 0.10% direction-error rate, and a finer "
         "vocabulary is unsupported: five words drop accuracy to 76.0%, seven to 71.0%. We "
         "close by outlining how the same judgments would enter robot control — gating, "
         "twist scaling, and a noise-floor stopping band — without validating that loop "
         "here.",
         size=8.9)
    para(doc, "**Keywords** — bladder ultrasound segmentation, image-quality function, "
              "operator guidance, language instruction, attained range",
         size=8.9, before=3)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def introduction(doc):
    head(doc, "1. Introduction")
    para(doc,
         "During the morcellation phase of holmium laser enucleation of the prostate (HoLEP) "
         "an assistant holds a suprapubic probe and keeps the bladder lumen in view. Keeping "
         "it there is a skill of the kind HoLEP training is built to teach [1]: the lumen "
         "drifts as the bladder volume changes, and deciding "
         "which way to slide the probe — or recognising that the image has degraded for a "
         "reason no probe motion can fix — currently rests on the operator's judgment alone. "
         "The image itself carries the information that judgment needs, and for any "
         "operator, human or robot alike, the missing piece is not a better segmentation "
         "mask but the path from a mask to something the operator can be told to do.")
    para(doc,
         "This paper builds that path as a three-stage pipeline: **segmentation** — a U-Net "
         "delineates the bladder lumen on each transabdominal frame; **quality judgment** — "
         "a transparent function *Q*, computed from the segmentation and the image, decides "
         "whether and how the frame falls short; and **instruction** — the judgment is "
         "emitted as one of a small set of words naming the operator's next corrective "
         "movement. The output is language by design. To a human operator the words are "
         "guidance that can be displayed or spoken; to a robotic probe holder each word, "
         "carrying a defined condition and axis, translates deterministically into a twist "
         "command. The paper validates the pipeline up to the emitted instruction and "
         "deliberately stops short of closing a robot loop; Section 4.1 records how the "
         "same judgments would enter one.")
    para(doc,
         "The central technical question is what instruction set *Q* can honestly support. "
         "The property that decides it is the function's **attained range** — the interval "
         "of values it actually takes on real frames — and not its correlation with any "
         "accuracy metric. A score that is formally defined on [0, 1] but occupies a band "
         "narrower than its own frame-to-frame noise cannot separate any two decisions, "
         "however well it ranks. Range divided by noise is the number of levels available, "
         "and that number is an upper bound on the size of an honest instruction "
         "vocabulary: Sections 3.2 and 3.3 measure it, and Section 3.4 derives the "
         "vocabulary from it.")


def methods_section(doc):
    head(doc, "2. Methods")
    head(doc, "2.1 Segmentation and the Image State", 2)
    para(doc,
         "A U-Net [2] takes a 256 × 256 transabdominal B-mode frame and produces a lumen "
         "probability map, thresholded at 0.5 and reduced to its largest connected component "
         "with interior holes filled. Frames come from PFUS1, a retrospective transabdominal "
         "bladder dataset; the model was trained on a patient-disjoint split and was not "
         "retrained or re-tuned for this study. All measurements below are made on the two "
         "held-out frame sets (1,661 and 1,292 frames) that share no patient with training "
         "or with each other; the segmentation performance on those frames is reported in "
         "Section 3.1.")
    para(doc,
         "From the mask an image state is computed — centroid, area, boundary statistics, "
         "and lumen-to-surround intensity contrast. Every ratio is taken over the insonified "
         "sector rather than the image rectangle; the sector is not recorded in the frame "
         "metadata and was measured from the training images, covering 66.4% of the frame "
         "and agreeing with held-out frames at an intersection-over-union of at least 0.964. "
         "Without it the frame grabber's crop enters every area ratio and a mask cut by the "
         "sector edge registers as touching no edge.")


    head(doc, "2.2 The Quality Function", 2)
    para(doc,
         "*Q* is a weighted mean of sub-scores, each normalised to [0, 1], every term and "
         "weight logged so that a value can be traced back to its components. Two "
         "aggregations are compared below: the arithmetic mean of the original "
         "specification, and a weighted geometric mean.")
    equation(doc, "Q = exp( Σ w~i~ ln s~i~ / Σ w~i~ )", "1")
    para(doc,
         "The aggregation is not a cosmetic choice. With eight terms an arithmetic mean caps "
         "the effect of one collapsed sub-score at its weight share, so a frame whose lumen "
         "contrast has vanished entirely still scores near the top of the range. A geometric "
         "mean is dominated by its smallest term, which is the shape a decision needs.")
    table(doc, 1, "Sub-scores of *Q* and what each reads. Only the first three consult the "
                  "ultrasound image; the rest restate the mask or its history.",
          ["Sub-score", "Reads", "Definition"],
          [["lumen contrast", "image", "(ring − lumen) / (ring + lumen), normalised"],
           ["lumen centering", "image", "exp(−offset to the darkness centroid / 0.25)"],
           ["boundary sharpness", "prob. map", "exp(−mean boundary entropy / 0.30)"],
           ["mask completeness", "mask", "plateau on the lumen area ratio"],
           ["segmentation confidence", "prob. map", "mean |2p − 1| over the sector"],
           ["border penalty, component quality", "mask", "mask geometry (weight 0 here)"],
           ["temporal terms (3)", "mask history", "agreement with the previous mask"]],
          widths=[2.5, 1.5, 4.0], size=7.5)
    para(doc,
         "The weights are set by rule, not fitted. A pre-declared tuning protocol — "
         "candidate weightings scored on the validation cohort by their AUROC for "
         "Dice ≥ 0.80 within bladder-volume bands, ties broken by dynamic range, the test "
         "cohort untouched — was run and then set aside for cause: with eleven patients "
         "per split, every candidate's bootstrapped confidence interval on the selection "
         "metric spanned most of the unit interval, so this cohort cannot rank weightings "
         "and a fitted weight would report sampling noise. The weights are therefore "
         "tiered, and equal within a tier: the three terms that read the ultrasound image "
         "share the top weight (1.5 each, 4.5 of the 6.0 total), the mask-side domain "
         "check keeps a minority 0.5, and terms whose measured behaviour disqualifies them "
         "— a border penalty duplicating a hard limit the validity gate already enforces, "
         "a component-quality term that returns one value — are set to zero rather than "
         "down-weighted, because a consistently inverted or saturated term is not a "
         "tuning question. `segmentation confidence` keeps its inherited weight of 1.0; "
         "what that buys is measured rather than assumed, in Section 3.2.")
    para(doc,
         "The three temporal terms are switched off for this study. The frames are static "
         "examinations in which nothing moved in response to the image, so frame-to-frame "
         "agreement measures the smoothness of a recording rather than the stability of a "
         "control loop; scoring it would credit *Q* for a property the setting has not been "
         "asked to produce.")


    head(doc, "2.3 Range, Noise, and Resolvable Levels", 2)
    para(doc,
         "The first property measured is the **attained range** — the interval of values "
         "*Q* actually takes, read over all 2,953 held-out frames and set against its "
         "nominal domain [0, 1]. Because the geometric mean is additive in the logarithm, "
         "the attained range decomposes exactly: each sub-score contributes its weight "
         "share times the log-width of the interval it attains, so every part of the range "
         "is traceable to one term.")
    para(doc,
         "A range is only useful relative to the noise on it. These are static "
         "examinations, so the change in *Q* between adjacent frames is almost entirely "
         "measurement noise, and its standard deviation is a direct estimate of the "
         "function's own jitter. Two frames whose scores differ by less than that jitter "
         "cannot be ordered — the difference is as likely noise as signal — so thresholds "
         "spaced closer than one jitter step do not define distinct decisions; they hand "
         "the same frame different labels on different frames. Dividing the attained "
         "range by the jitter therefore gives the number of levels a reader of *Q* can "
         "distinguish; the central 80% of the distribution — the operating region a "
         "threshold would actually be placed in — is reported alongside the full range.")

    head(doc, "2.4 The Instruction Axis", 2)
    para(doc,
         "The instruction is issued on the one image axis that leaves the imaging plane "
         "invariant. Of the probe's six axes only *v*~x~, *v*~z~ and *ω*~y~ preserve the "
         "plane; of the three the image policy holds, only lateral sliding *v*~x~ does, and "
         "only it presents a measurable vector error rather than a scalar to be searched "
         "(Figure 1a). Write the instruction as *ê* = *A* − *ĉ*, with *A* the beam axis and "
         "*ĉ* the predicted lumen centroid abscissa (Figure 1b).")
    wide_figure(doc, 1, "fig_axes.png",
                "(a) Only *v*~x~ leaves the imaging plane invariant among the three axes the "
                "image policy holds. (b) The lateral instruction on one frame: *A* is the "
                "beam axis, *c* the lumen centroid abscissa, *e* = *A* − *c*.", 14.0)

    head(doc, "2.5 Validation by Lateral Translation", 2)
    para(doc,
         "One property of the frames prevents the direction from being validated on them as "
         "recorded: in all 2,953 the lumen lies on the same side of the beam axis, so *e* "
         "never changes sign and a constant output would score 99.3%. The sonographer holds "
         "a diagnostic view rather than centring the lumen on the axis, which is a property "
         "of how the frames were acquired and not of the function.")
    para(doc,
         "The instruction is therefore exercised on laterally translated frames. Lateral "
         "translation is the one probe motion single-plane frames can reproduce faithfully, "
         "and it carries the lumen across the axis so that *e* changes sign; each frame is "
         "shifted by the amount that places *e* on a prescribed target, sampled densely near "
         "the crossing and sparsely away from it. Fabricated margin is filled by edge "
         "replication, never black, which every intensity term would read as anechoic lumen. "
         f"Across the sweep the segmentation is unaffected — Dice holds at "
         f"{S['dice_under_translation']['median']:.3f} — and {S['n_frames']:,} frames "
         "result, 43% with *e* < 0.")


def results_section(doc):
    head(doc, "3. Results")
    head(doc, "3.1 Segmentation Performance", 2)
    para(doc,
         "Table 2 reports the segmentation stage on the two held-out frame sets, both "
         "unfiltered and restricted to the distended-bladder working domain — frames "
         f"whose annotated lumen occupies at least {FLOOR * 100:.1f}% of the sector, the "
         "state morcellation irrigation is meant to maintain. Within that domain the "
         "lateral centroid error, the quantity the instruction is built on, has a median "
         "near 4 px. The tail below the domain is the same population the *check bladder "
         "filling* branch of Section 3.4 addresses: segmentation degrades exactly where "
         "the instruction refuses to steer.")
    seg_rows = []
    for split in ("val", "test"):
        full = SEG_ALL[split]
        seg_rows.append([
            split + ", all frames", f"{full['n_frames']:,} / {full['n_patients']}",
            f"{full['dice']['mean']:.3f} ± {full['dice']['sd']:.3f}",
            f"{full['dice']['median']:.3f}", "—", "—"])
    for split in ("val", "test"):
        dom = SEG_DOMAIN[split]
        seg_rows.append([
            split + ", working domain", f"{dom['n_frames']:,} / {dom['n_patients']}",
            f"{dom['dice']['mean']:.3f} ± {dom['dice']['sd']:.3f}",
            f"{dom['dice']['median']:.3f}",
            f"{dom['iou']['mean']:.3f} ± {dom['iou']['sd']:.3f}",
            f"{dom['centroid_error_px']['median']:.2f}"])
    table(doc, 2, "Segmentation performance of the U-Net [2] on the held-out frame sets. "
                  "The working domain keeps frames whose annotated lumen area is at least "
                  f"{FLOOR * 100:.1f}% of the sector; IoU and the lateral centroid error "
                  "(median, px at 256 × 256) are computed there.",
          ["Frame set", "Frames / pts", "Dice", "Dice med.", "IoU", "Centr. err."],
          seg_rows, widths=[1.9, 1.2, 1.5, 1.0, 1.5, 1.0], size=7.5)

    head(doc, "3.2 The Attained Range of *Q*", 2)
    para(doc,
         "*Q* is defined on [0, 1]. On 2,953 held-out frames the original specification "
         "attains [0.70, 1.00] — a width of 0.30, or 30% of the nominal domain — and its "
         "central 80% spans only 0.12. Reformulated with the geometric aggregation and "
         "weighted toward the terms that read the image, it attains [0.14, 0.78], a width "
         "of 0.64, with a central 80% of 0.45 (Figure 2a).")
    para(doc,
         "The decomposition of Section 2.3 attributes **92.4% of the range to two "
         "sub-scores** (Figure 2b, Table 3), and both are terms that consult the "
         "ultrasound image rather than the mask.")
    wide_figure(doc, 2, "fig_range.png",
                "(a) The interval *Q* actually attains under the two aggregations, against "
                "its nominal domain [0, 1]. (b) The interval each active sub-score attains "
                "and its share of the resulting range. `segmentation confidence` occupies a "
                "band 0.007 wide and contributes nothing.", 15.4)
    table(doc, 3, "Attained interval of each active sub-score and its share of the range of "
                  "*Q*. Shares follow from the geometric aggregation and sum to 100%.",
          ["Sub-score", "Weight", "Attained interval", "Width", "Share"],
          [["lumen contrast", "1.5", "0.000 – 1.000", "1.000", "47.6%"],
           ["lumen centering", "1.5", "0.001 – 0.999", "0.997", "44.8%"],
           ["boundary sharpness", "1.5", "0.237 – 0.451", "0.214", "4.4%"],
           ["mask completeness", "0.5", "0.250 – 1.000", "0.750", "3.2%"],
           ["segmentation confidence", "1.0", "0.992 – 0.999", "**0.007**", "**0.0%**"]],
          widths=[2.4, 1.0, 2.0, 1.0, 1.0], size=7.5)
    para(doc,
         "The last row is the point of the section. `segmentation confidence` is not a weak "
         "term that could be rescued by more weight: it returns essentially one value, so it "
         "has no range, and a function with no range cannot participate in any decision. It "
         "carries a weight of 1.0 while contributing 0.0% of what *Q* can express. "
         "Two further terms — border penalty and component quality — behave the same way and "
         "are already at weight 0.")


    head(doc, "3.3 Resolvable Levels", 2)
    para(doc,
         "Table 4 divides the attained range by the frame-to-frame jitter measured on the "
         "same frames (Section 2.3).")
    table(doc, 4, "Resolution of *Q*: attained range over frame-to-frame jitter. The central "
                  "80% figure is the operating region a threshold would actually be placed "
                  "in.",
          ["Aggregation", "Range", "Jitter", "Levels", "Central 80%", "Levels"],
          [["arithmetic, 8 terms", "0.296", "0.0144", "21", "0.123", "**9**"],
           ["geometric, image-first", "0.635", "0.0095", "67", "0.450", "**47**"]],
          widths=[2.4, 1.0, 1.1, 1.0, 1.4, 1.0], size=7.5)
    para(doc,
         "Nine levels is the quantitative statement of a failure that had previously been "
         "described only qualitatively: with the original specification, no threshold below "
         "0.75 rejected a single frame out of 2,953, and the whole usable operating region "
         "lay between 0.85 and 0.95. Any instruction set finer than nine states would have "
         "been reporting noise. The reformulated function has room for 47.")


    head(doc, "3.4 The Derived Vocabulary", 2)
    para(doc,
         "Two boundaries partition the range, and neither is a fitted hyperparameter.")
    para(doc,
         "**A physical boundary.** A urine-filled lumen is anechoic, so a non-positive "
         "lumen-to-surround contrast means the delineated region cannot be one. On 13.1% of "
         "frames the contrast is non-positive; those frames have Dice 0.665 against 0.840 "
         "elsewhere, and what sets them apart is the delineated region itself — 1.85 times "
         "brighter and a third the area, the signature of an under-distended bladder rather "
         "than of a poorly placed probe. The remedy is not a lateral slide; it is upstream, "
         "in bladder filling, and the instruction says so instead of commanding a movement "
         "that cannot help.")
    para(doc,
         "**A statistical boundary.** The difference *δ* = *ê* − *e* is the segmentation's "
         "lateral centroid error, independent of where the lumen sits, with "
         f"*σ* = {SIGMA:.2f} px and a bias of {S['delta_mean_px']:+.2f} px. The instruction "
         "points the wrong way exactly when *δ* opposes *e* and exceeds it, so")
    equation(doc, "P(wrong direction) = Φ(−|e| / σ)", "2")
    para(doc,
         f"and the instruction is withheld below {TH['5%']['k']:.2f}*σ* = "
         f"{TH['5%']['px']:.1f} px, where the modelled error rate reaches 5%. Below that "
         "boundary a controller that keeps acting is following its own segmentation noise; "
         "**this is a stopping condition, not a defect.**")
    table(doc, 5, "The instruction vocabulary. Both boundaries are derived — one from the "
                  "physics of an anechoic lumen, one from the segmentation's centroid noise "
                  "— and neither is tuned.",
          ["Instruction", "Condition", "Acts on"],
          [["*check bladder filling*", "contrast ≤ 0", "not the probe — filling"],
           ["*move left* / *move right*", f"contrast > 0 and |*ê*| ≥ {TH['5%']['px']:.1f} px",
            "lateral axis *v*~x~"],
           ["*hold*", f"contrast > 0 and |*ê*| < {TH['5%']['px']:.1f} px", "nothing — converged"]],
          widths=[2.5, 3.0, 2.3], size=7.5)
    para(doc,
         "Vocabulary size is bounded by the same noise. A magnitude class narrower than 2*σ* "
         "cannot be assigned reliably even at its centre, and the data bear this out: adding "
         "graded magnitudes costs accuracy without buying direction (Table 6). Three words "
         "is what the function supports.")
    table(doc, 6, "Accuracy of the emitted instruction against the instruction the ground "
                  "truth implies, over the controlled sweep of Section 2.5. Direction "
                  "errors stay negligible; it is the magnitude classes that fail.",
          ["Vocabulary", "Size", "Accuracy", "Direction error"],
          [["move / hold", "2", "85.1%", "0.00%"],
           ["**move left / move right / hold**", "**3**", "**85.0%**", "**0.10%**"],
           ["+ two magnitude grades", "5", "76.0%", "0.11%"],
           ["+ three magnitude grades", "7", "71.0%", "0.11%"]],
          widths=[3.2, 0.9, 1.4, 1.6], size=7.5)
    para(doc,
         "Each word is a complete instruction to two different readers. To the human "
         "operator the vocabulary is displayed or spoken as it stands: it names a "
         "direction rather than a number, which is the register in which probe handling is "
         "actually taught and corrected. To a robotic probe holder each word is a symbol "
         "with a defined condition and axis, so translation into a twist command is a "
         "lookup, not an inference (Table 7). The language layer is thus the interface "
         "between the two settings: the same judgment drives either operator, and nothing "
         "about the pipeline changes when the reader does.")
    table(doc, 7, "One vocabulary, two readers. The mapping from word to action is fixed "
                  "in advance on both sides; no model sits between the instruction and its "
                  "execution.",
          ["Instruction", "Human operator", "Robotic probe holder"],
          [["*check bladder filling*", "pause; reassess filling before scanning on",
            "zero twist; hand control back to the supervisor"],
           ["*move left* / *move right*", "slide the probe laterally as stated",
            "*v*~x~ = −*k ê*, saturated; other axes zero"],
           ["*hold*", "keep the current pose",
            "zero lateral twist; no correction commanded"]],
          widths=[2.2, 2.8, 2.8], size=7.5)


    head(doc, "3.5 Accuracy of the Emitted Instruction", 2)
    para(doc,
         "Figure 3 shows one frame of the sweep at three lateral displacements, including "
         "one inside the band where the direction is no longer trustworthy.")
    wide_figure(doc, 3, "fig_examples.png",
                "One frame at three lateral displacements. Cyan is the annotation, orange "
                "dashed the prediction, white dashed the beam axis, the arrow the "
                "instruction. At *e* = +2.1 px the instruction has collapsed to +0.8 px: "
                "the loop has entered the band where Equation (2) says the direction is no "
                "longer reliable.", 15.4)
    para(doc,
         f"The instruction agrees with the truth in {pct(OVERALL['sign_agreement'])} of "
         f"frames, with a median magnitude error of {OVERALL['abs_error_median']:.2f} px, a "
         f"regression slope of {OVERALL['slope']:.3f} and *r* = "
         f"{OVERALL['pearson_r']:.3f}. A slope this close to unity means the instruction is "
         "correctly scaled and not merely directional. The two held-out frame sets agree to "
         "within 0.7 px of median error.")
    wide_figure(doc, 4, "fig_agreement.png",
                "Instructed against ground-truth lateral displacement, one panel per "
                "held-out frame set. Shaded quadrants are wrong-direction outcomes; every "
                "one lies within a few pixels of the axis.", 15.2)
    table(doc, 8, "Agreement between the emitted instruction and the truth.",
          ["Frame set", "n", "Sign agr.", "|err| med.", "|err| p90", "Slope", "*r*"],
          [[k, f"{v['n']:,}", pct(v["sign_agreement"]), f"{v['abs_error_median']:.2f}",
            f"{v['abs_error_p90']:.2f}", f"{v['slope']:.3f}", f"{v['pearson_r']:.3f}"]
           for k, v in (("val", BY["val"]), ("test", BY["test"]), ("pooled", OVERALL))],
          widths=[1.4, 1.0, 1.3, 1.2, 1.2, 1.0, 0.9], size=7.5)
    para(doc,
         "**The result is not that percentage but the shape of its failure.** Wrong "
         "directions occur only near the axis — their median required displacement is "
         "7.9 px against 15.8 px for correct ones — and their rate follows Equation (2) "
         "with *σ* fitted once to the centroid error and not to this curve (Figure 5, "
         "Table 9). Agreement across two decades of error rate is what licenses deriving "
         "the vocabulary boundary from the model rather than reading it off the data.")
    wide_figure(doc, 5, "fig_validity.png",
                "Wrong-direction rate against required displacement. The curve is Equation "
                "(2); circles are observed rates with frame counts. Dashed verticals mark "
                "where the modelled rate reaches 5% and 1%.", 15.2)
    table(doc, 9, "Observed and modelled wrong-direction rates. The model is conservative in "
                  "the largest bins, where it predicts rates below what a few hundred frames "
                  "resolve.",
          ["|*e*| [px]", "n", "Observed", "Model"],
          [[f"{c['low']}–{c['high']}" if c["high"] < 999 else f"> {c['low']}",
            f"{c['n']:,}", pct(c["observed_sign_error"]), pct(c["modelled_sign_error"])]
           for c in CURVE],
          widths=[1.7, 1.3, 1.9, 1.7])
    para(doc,
         "The constants are stable against annotation quality. A model-blind label audit "
         "flags four of the frame sets' patients as carrying regions that cannot be lumen; "
         f"recomputing without them moves *σ* from {SIGMA:.2f} to 4.67 px (−5%), sign "
         "agreement from 93.7% to 94.1%, and the 5% boundary from "
         f"{TH['5%']['px']:.1f} to 7.7 px.")


def robot_section(doc):
    head(doc, "4.1 Reflecting the Quality Judgment in Robot Control", 2)
    para(doc,
         "The pipeline validated above ends at the words. The words, however, were shaped "
         "so that a robotic probe holder can consume them, and the quality judgment enters "
         "that loop at two points. Neither is validated here; this section records the "
         "design, not a result.")
    para(doc,
         "**Gating.** *check bladder filling* suspends image-based motion outright: when "
         "the lumen contrast is non-positive the segmentation cannot be delineating a "
         "filled bladder, and no twist is preferable to a twist computed from a mask that "
         "physics rejects. Because instructions are issued per frame, one degraded frame "
         "gates one command, not the session. Beyond the words, *Q* itself serves as a "
         "supervisory scalar: with 47 resolvable levels it has room to scale the commanded "
         "speed down as quality falls — something the original function, at nine levels, "
         "could not have supported honestly.")
    para(doc,
         "**Scaling and stopping.** *move left* and *move right* carry the magnitude *ê*, "
         "so the robot realisation is proportional rather than bang-bang: "
         "*v*~x~ = −*k ê*, saturated, with every other axis commanded to zero. The "
         f"{TH['5%']['k']:.2f}*σ* boundary of Section 3.4 becomes a deadband: inside it the "
         "commanded twist is zero, because Equation (2) says the sign of any correction "
         "would be noise. A controller that keeps acting below that floor is servoing its "
         "own segmentation error; *hold* is a stopping condition the robot inherits for "
         "free.")
    para(doc,
         "The instruction acts only on *v*~x~, the one axis that leaves the imaging plane "
         "invariant, so nothing the image loop commands can rotate the anatomy out of its "
         "own view. Closing and validating this loop — its latency, and *Q* measured while "
         "the probe actually moves — is deliberately outside this paper's scope.")


def discussion_section(doc):
    head(doc, "4. Discussion")
    para(doc,
         "Reporting an image-quality function by its correlation with a segmentation metric "
         "answers the wrong question. Correlation says how the score orders frames; the "
         "attained range says how many decisions can be taken from it, and the two come "
         "apart. The function studied here ordered frames respectably in its original form "
         "while supporting nine distinguishable states — fewer than the instructions a "
         "controller would want to issue — and the diagnosis was invisible to any accuracy "
         "figure. Range, its decomposition, and range-over-noise are cheap to compute and "
         "we suggest reporting them for any score intended to instruct an operator or "
         "drive a device.")
    para(doc,
         "The decomposition also gives a concrete design rule. A term contributing 0.0% of "
         "the range is not underweighted; it is degenerate, and the fix is to measure it "
         "somewhere the decision is hard or to remove it. Here the two terms that consult "
         "the ultrasound image carry 92.4% of the range while three mask-geometry and "
         "probability-map terms carry almost none.")

def limitations_section(doc):
    head(doc, "4.2 Limitations", 2)
    for item in (
        "No real probe motion is validated. The frames are static examinations whose lumen "
        "centroid moves 0.45 px between samples, below the segmentation's own centroid "
        "error, so lateral displacement is simulated and the instruction has not been closed "
        "into a loop with a person or a robot.",
        "Two of the image policy's three axes are untouched: *v*~y~ and *ω*~z~ leave the "
        "imaging plane, present only a scalar quality, and cannot be simulated from "
        "single-plane frames.",
        "The target pose — lumen on the beam axis — is a design choice for the robotic "
        "setting. The archived frames were not acquired toward it, so the proportion of "
        "frames falling in each instruction band describes the frames, not the loop's "
        "steady state.",
        "No pixel spacing accompanies the frames, so σ and every derived boundary are stated "
        "at 256 × 256 and must be rescaled before they mean millimetres.",
        "σ is treated as one constant. Per-frame estimation would tighten the stopping "
        "boundary where the segmentation is confident and loosen it where it is not.",
        "The *check bladder filling* branch is supported by 13.1% of frames from four "
        "patients; whether the cause is under-distension or annotation is not separable "
        "here [TO BE CONFIRMED: IRB approval number, scanner model, annotation protocol].",
    ):
        para(doc, "— " + item, size=8.6, indent=0.35, after=1.8)


def references_section(doc):
    head(doc, "References")
    para(doc,
         "[1] T. Jang, H.-J. Kong, C. Baek, J. Kim, M. S. Choo, S.-J. Oh. Effect of "
         "Self-Training Using Virtual Reality Head-Mounted Display Simulator on the "
         "Acquisition of Holmium Laser Enucleation of the Prostate Surgical Skills. "
         "International Neurourology Journal 2024;28(2):138–146.", size=8.6, after=2)
    para(doc,
         "[2] O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for "
         "Biomedical Image Segmentation. In: Medical Image Computing and Computer-Assisted "
         "Intervention (MICCAI 2015), LNCS 9351, Springer, 2015, pp. 234–241.",
         size=8.6, after=2)
    para(doc,
         "[TO BE COMPLETED] Citations for HoLEP morcellation, ultrasound visual servoing "
         "and image-quality assessment are to be added before submission.",
         size=8.6, italic=True)


def build():
    doc = Document()
    style_doc(doc)
    page_numbers(doc)
    columns(doc.sections[0], 1)
    front_matter(doc)

    body = doc.add_section(WD_SECTION.CONTINUOUS)
    style_doc(doc)
    columns(body, 2)

    introduction(doc)
    methods_section(doc)
    results_section(doc)
    discussion_section(doc)
    robot_section(doc)
    limitations_section(doc)
    references_section(doc)

    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
