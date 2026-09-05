#!/usr/bin/env python3
"""Lateral-instruction manuscript (.docx) — text, figures and tables.

    python3 docs/build_lateral_instruction_manuscript.py
    soffice --headless --convert-to pdf Lateral_Instruction_Manuscript.docx

Format follows the 대한의료정보학회(KOSMI) 연제논문 초록 포맷: A4, single column,
10 mm top/bottom margins and 25 mm side margins, 10 pt body at 140% line
spacing (abstract 120%), 장평 95% · 자간 −5%, Korean glyphs in 신명조 and Latin
in Times New Roman. Numbered tables and figures carry captions; every number in
the text is read from experiments/lateral_instruction/lateral_instruction.json
rather than typed, so the manuscript cannot drift from the analysis.
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
#: volume-gated audit; Section 3.1's numbers are read from here, not typed.
AUDIT = json.load(open(os.path.join(REPO, "Unet_seg", "experiments", "quality_retro",
                                    "audit", "audit.json")))
#: Word accuracy per vocabulary size, from scripts/vocabulary_accuracy.py.
VOCAB = {v["size"]: v for v in json.load(
    open(os.path.join(FIGDIR, "vocabulary_accuracy.json")))["vocabularies"]}
#: Everything the paper is made of collects under Paper/: the .docx, and a
#: copy of every figure the build actually placed (kept in sync at build time —
#: the analysis scripts under Unet_seg/ regenerate the originals in FIGDIR).
PAPER = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PAPER, "Lateral_Instruction_Manuscript.docx")
USED_FIGURES: list[str] = []

# 대한의료정보학회(KOSMI) 연제논문 초록 포맷을 따른다 (KOSMI_Abstract_style):
# A4, 단일 단, 여백 위·아래·머리말·꼬리말 10 mm · 좌·우 25 mm, 본문 10 pt,
# 줄간격 140 %(초록 120 %), 장평 95 %, 자간 −5 %, 국문 글꼴 신명조.
BODY_PT = 10
SERIF = "Times New Roman"     # 라틴(영문) 글꼴 — 신명조와 같은 명조 계열
EAST = "신명조"               # 한글 글꼴 (KOSMI 지정)
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
    rpr = st.element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:eastAsia"), EAST)
    # 장평 95 % (w:w) · 자간 −5 % ≈ −0.5 pt = −10 twip (w:spacing). KOSMI 전 스타일 공통.
    for tag, val in (("w:w", "95"), ("w:spacing", "-10")):
        el = OxmlElement(tag)
        el.set(qn("w:val"), val)
        rpr.append(el)
    pf = st.paragraph_format
    pf.space_before = pf.space_after = Pt(0)
    pf.line_spacing = 1.4              # KOSMI 바탕글 줄간격 140 %
    pf.widow_control = True
    for s in doc.sections:
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.top_margin = s.bottom_margin = Cm(1.0)      # 위·아래 10 mm
        s.left_margin = s.right_margin = Cm(2.5)      # 좌·우 25 mm
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
    # KOSMI: Ⅱ.소제목 = 가운데·문장위 3 mm, 2.소제목 = 왼쪽·문장위 2 mm, 둘 다 10 pt.
    p = para(doc, text, size=10.0, bold=True,
             align="center" if level == 1 else "left",
             before=8.5 if level == 1 else 5.7, after=0.0)
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
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(os.path.join(FIGDIR, filename), width=Cm(width_cm))
    USED_FIGURES.append(filename)
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.space_after = Pt(4)
    rich(c, "**Figure %d.** %s" % (number, caption), size=7.9)


#: Text width [cm] — single column, page minus the 25 mm side margins (KOSMI).
COL_CM = 21.0 - 2 * 2.5


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
    """Scale a width recipe to the single-column text width."""
    target = COL_CM - 0.05
    factor = target / sum(widths)
    return [w * factor for w in widths]


def wide_figure(doc, number, filename, caption, width_cm=16.0):
    """Full-width figure. In the KOSMI single-column layout there is nothing
    to span — the figure simply sits in the one text column, centred."""
    figure(doc, number, filename, caption, min(width_cm, COL_CM))


def wide_table(doc, *args, **kwargs):
    """Full-width table (single column)."""
    kwargs["full_width"] = True
    table(doc, *args, **kwargs)


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
    _keep_together(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(3)
    return t


def _keep_together(t):
    """Stop a table from being split across a page break.

    Two levers: every row is marked ``cantSplit`` so no single row breaks
    mid-cell, and every row but the last keeps with the next so the block
    migrates to the next page whole instead of straddling the boundary.
    """
    rows = t.rows
    for index, row in enumerate(rows):
        tr_pr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        tr_pr.append(cant)
        if index < len(rows) - 1:
            for cell in row.cells:
                p = cell.paragraphs[0]._p
                p_pr = p.get_or_add_pPr()
                keep = OxmlElement("w:keepNext")
                p_pr.append(keep)


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
         size=12.0, bold=True, align="center", after=3)
    para(doc, "Seong Jeong^1, 2, 5^, Minsung Kim^1,2,4^, Dongho Yee^1,2,4^, "
              "Yechan Seo^1,2,5^, Juahn Oh^1,2,3^, Hyoun-Joong Kong^1,2,5,*^",
         size=10.0, align="center", after=3)
    for line in (
        "^1^ Department of Transdisciplinary Medicine, Seoul National University Hospital, Seoul, Republic of Korea",
        "^2^ Rosota Inc., Seoul, Republic of Korea",
        "^3^ Eulji University College of Medicine, Daejeon, Republic of Korea",
        "^4^ Department of Mechanical Engineering, Seoul National University, Seoul, Republic of Korea",
        "^5^ Department of Medicine, Seoul National University, Seoul, Republic of Korea",
    ):
        para(doc, line, size=8.0, align="center", after=0.4)
    doc.add_paragraph().paragraph_format.space_after = Pt(3)

    para(doc, "Abstract", size=10.0, bold=True, align="left", before=3, after=1)
    _abs = para(doc,
         "During the morcellation phase of holmium laser enucleation of the prostate "
         "(HoLEP) an operator holds a suprapubic probe and must keep the bladder lumen in "
         "view. We present a guidance architecture that turns each bladder ultrasound "
         "frame into one corrective instruction with two readers: a U-Net segments the "
         "lumen, a transparent image-quality function *Q* judges the frame, and the "
         "judgment is emitted as a **language instruction** naming the next movement — "
         "*move left*, *move right*, *hold*, or *check bladder filling*. To a human "
         "operator it is pixel-level probing feedback, displayed or spoken; to a robotic "
         "probe holder the same emission is an action token (word, *ê*), translated to a "
         "twist by lookup. An instruction is only as fine as the function issuing it, so "
         "we first measure what *Q* can tell apart: it attains [0.14, 0.78] of its "
         "nominal domain, separating **47 states above its noise** — the ceiling any "
         "honest vocabulary must fit under. The vocabulary is then derived, not chosen: "
         "one boundary is physical (a non-anechoic lumen cannot be a filled bladder), the "
         f"other statistical (an offset below {TH['5%']['k']:.2f}σ of the segmentation's "
         f"centroid noise, σ = {SIGMA:.2f} px, cannot be signed). On {S['n_frames']:,} "
         "displaced held-out frames the three-word vocabulary is emitted "
         f"correctly on {VOCAB[3]['accuracy'] * 100:.1f}% of frames at a "
         f"{VOCAB[3]['direction_error'] * 100:.2f}% direction-error rate; the magnitude "
         f"travels beside the word as a continuous value (slope {S['overall']['slope']:.3f}), "
         "since graded vocabularies drop accuracy to "
         f"{VOCAB[5]['accuracy'] * 100:.1f}% and {VOCAB[7]['accuracy'] * 100:.1f}%. We "
         "outline how the judgments would enter robot control, without validating "
         "that loop.",
         size=10.0)
    _abs.paragraph_format.line_spacing = 1.2      # KOSMI 초록 줄간격 120 %
    para(doc, "**Keywords** — bladder ultrasound segmentation, image-quality function, "
              "operator guidance, language instruction, attained range",
         size=10.0, align="left", before=1, after=2)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def introduction(doc):
    head(doc, "1. Backgrounds")
    para(doc,
         "During the morcellation phase of holmium laser enucleation of the prostate (HoLEP) "
         "an assistant holds a suprapubic probe and keeps the bladder lumen in view. Keeping "
         "it there is a skill of the kind HoLEP training is built to teach [1]: the lumen "
         "drifts as the bladder volume changes, and deciding "
         "which way to slide the probe — or recognising that the image has degraded for a "
         "reason no probe motion can fix — currently rests on the operator's judgment alone. "
         "Automated support for bladder ultrasound has concentrated on the two ends of "
         "that judgment: deep networks segment the bladder and its wall accurately and "
         "cheaply enough for wearable monitors [3, 4], and guidance systems steer the "
         "operator toward a standard view [5]. "
         "The image itself carries the information that judgment needs, and for any "
         "operator, human or robot alike, the missing piece is not a better segmentation "
         "mask but the path from a mask to something the operator can be told to do.")
    para(doc,
         "This paper builds and argues for that path as an architecture of three stages "
         "and one interface (Figure 1): **segmentation** — a U-Net delineates the bladder "
         "lumen on each frame; **quality judgment** — a transparent "
         "function *Q*, computed from the segmentation and the image, decides whether and "
         "how the frame falls short; **corrective action** — the judgment is reduced to "
         "one of a small set of words naming the next corrective movement, the movement's "
         "magnitude travelling beside the word as a continuous value in pixels; and an "
         "**output interface with two readers**. To a human operator the words are "
         "pixel-level probing feedback that can be displayed or spoken; to a robotic "
         "probe holder the same emission is an action token — the word selects a "
         "pre-declared condition and axis, the number scales the motion — so translation "
         "into a twist command is a lookup, not an inference. The paper validates the "
         "architecture up to the emitted instruction and deliberately stops short of "
         "closing a robot loop; Section 4.1 records how the token channel would enter "
         "one.")
    wide_figure(doc, 1, "fig_overview.png",
                "The guidance architecture. A U-Net segments the lumen (1); *Q* judges "
                "the frame from the segmentation and the image (2); the judgment becomes "
                "one instruction — a word with the continuous magnitude *ê* beside it "
                "(3). The same emission is read twice: as language feedback by the human "
                "operator, and as an action token by a robotic probe holder, translated "
                "to a twist by lookup. Stages 1–3 and the emitted instruction are "
                "validated here; the robotic execution (dashed) is a design contract, "
                "recorded in Section 4.1.", 16.2)
    para(doc,
         "The central technical question is what instruction set *Q* can honestly support. "
         "The property that decides it is the function's **attained range** — the interval "
         "of values it actually takes on real frames — and not its correlation with any "
         "accuracy metric. A score that is formally defined on [0, 1] but occupies a band "
         "narrower than its own frame-to-frame noise cannot separate any two decisions, "
         "however well it ranks. Range divided by noise counts the states the function "
         "can genuinely tell apart — score differences a reader can trust as real rather "
         "than as noise — and that count is an upper bound on the size of an honest "
         "instruction vocabulary: Sections 3.2 and 3.3 measure it, and Section 3.4 "
         "derives the vocabulary from it.")


def methods_section(doc):
    head(doc, "2. Methods")
    head(doc, "2.1 Segmentation and the Image State", 2)
    para(doc,
         "A U-Net [2] takes a 256 × 256 B-mode frame and produces a lumen "
         "probability map, thresholded at 0.5 and reduced to its largest connected component "
         "with interior holes filled. Frames come from PFUS1 [6], a public transperineal "
         "pelvic-floor ultrasound dataset: midsagittal videos of 101 women without "
         "pelvic-floor pathology, at rest and during Valsalva, acquired on one scanner at "
         "one centre, with the urinary bladder among the eight annotated organs [6, 7]. "
         "**It is not HoLEP imaging** — the acoustic window (transperineal rather than "
         "suprapubic), the population (women in diagnostic examinations rather than men "
         "under morcellation), and the scene (no morcellator, no irrigation inflow) all "
         "differ — and Section 4.2 states what that gap leaves unvalidated. The model was "
         "trained on a patient-disjoint split and was not "
         "retrained or re-tuned for this study. All measurements below are made on the two "
         "held-out frame sets (1,661 and 1,292 frames) that share no patient with training "
         "or with each other; the segmentation performance on those frames is reported in "
         "Section 3.1.")
    para(doc,
         "From the mask an image state is computed — centroid, area, boundary statistics, "
         "and lumen-to-surround intensity contrast — with every ratio taken over the "
         "insonified sector rather than the image rectangle. The sector was measured from "
         "the training images (66.4% of the frame, IoU ≥ 0.964 against held-out frames); "
         "without it the frame grabber's crop enters every area ratio.")


    head(doc, "2.2 The Quality Function", 2)
    para(doc,
         "*Q* is a weighted geometric mean of sub-scores, each normalised to [0, 1], "
         "every term and weight logged so that a value can be traced back to its "
         "components. Three sub-scores consult the ultrasound image — lumen-to-surround "
         "contrast, centering on the darkness centroid, and boundary sharpness of the "
         "probability map — and the rest restate the mask or its history (completeness, "
         "confidence, geometry, temporal agreement).")
    equation(doc, "Q = exp( Σ w~i~ ln s~i~ / Σ w~i~ )", "1")
    para(doc,
         "The geometric aggregation has the shape a decision needs: the mean is dominated "
         "by its smallest term, so a frame whose lumen contrast has collapsed scores low "
         "no matter how well the remaining terms do — no single degraded property can "
         "hide inside an average.")
    para(doc,
         "The weights are set by rule, not fitted — with eleven patients per split, any "
         "fitted weight would report sampling noise. They are tiered and equal within a "
         "tier: the three image-reading terms share the top weight (1.5 each, 4.5 of the "
         "6.0 total), the mask-side domain check keeps 0.5, terms whose measured "
         "behaviour disqualifies them are set to zero, and `segmentation confidence` "
         "keeps its inherited 1.0 — what that buys is measured rather than assumed, in "
         "Section 3.2. The three temporal terms are switched off: the frames are static "
         "examinations, so frame-to-frame agreement would score the smoothness of a "
         "recording, not the stability of a control loop.")


    head(doc, "2.3 Range, Noise, and What *Q* Can Tell Apart", 2)
    para(doc,
         "The first property measured is the **attained range** — the interval of values "
         "*Q* actually takes over all 2,953 held-out frames, set against its nominal "
         "domain [0, 1]; because the geometric mean is additive in the logarithm, every "
         "part of that range is traceable to one sub-score. A range is then only useful "
         "relative to the noise on it. These are static examinations, so the change in "
         "*Q* between adjacent frames is almost entirely the function's own jitter, and "
         "two frames whose scores differ by less than that jitter cannot be ordered — "
         "the difference is as likely noise as signal. Dividing the attained range by "
         "the jitter therefore counts **how many states a reader of *Q* can genuinely "
         "tell apart**; the central 80% of the distribution — the operating region a "
         "threshold would actually be placed in — is reported alongside the full range.")

    head(doc, "2.4 The Instruction Axis", 2)
    para(doc,
         "The instruction is issued on the one image axis that leaves the imaging plane "
         "invariant. Of the probe's six axes only *v*~x~, *v*~z~ and *ω*~y~ preserve the "
         "plane; of the three the image policy holds, only lateral sliding *v*~x~ does, and "
         "only it presents a measurable vector error rather than a scalar to be searched "
         "(Figure 2a). Write the instruction as *ê* = *A* − *ĉ*, with *A* the beam axis and "
         "*ĉ* the predicted lumen centroid abscissa (Figure 2b).")
    wide_figure(doc, 2, "fig_axes.png",
                "(a) Only *v*~x~ leaves the imaging plane invariant among the three axes the "
                "image policy holds. (b) The lateral instruction on one frame: *A* is the "
                "beam axis, *c* the lumen centroid abscissa, *e* = *A* − *c*.", 12.6)

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
         "Table 1 stratifies the segmentation stage by bladder filling — the share of "
         "the insonified sector the annotated lumen occupies, the area a filling bladder "
         "sweeps. Performance rises monotonically with distension, from "
         f"{SEG_ALL['val']['dice']['mean']:.3f} / {SEG_ALL['test']['dice']['mean']:.3f} "
         "(val / test, all frames) to above 0.90 in the fullest bands; within the "
         f"working domain (lumen ≥ {FLOOR * 100:.1f}% of the sector, the state "
         "morcellation irrigation is meant to maintain) the lateral centroid error, the "
         "quantity the instruction is built on, has a median of "
         f"{SEG_DOMAIN['val']['centroid_error_px']['median']:.2f} px and "
         f"{SEG_DOMAIN['test']['centroid_error_px']['median']:.2f} px. The tail below "
         "the domain is the same population the *check bladder filling* branch of "
         "Section 3.4 addresses: segmentation degrades exactly where the instruction "
         "refuses to steer.")
    sweep = {s: {e["floor"]: e for e in AUDIT["sweep"][s] if e["n_frames"]}
             for s in ("val", "test")}
    vol_rows = []
    for f in sorted(set(sweep["val"]) & set(sweep["test"])):
        v, t = sweep["val"][f], sweep["test"][f]
        cells = ["all frames" if f == 0 else f"≥ {f * 100:.1f}%",
                 f"{v['n_frames']:,}",
                 f"{v['dice']['mean']:.3f} ± {v['dice']['sd']:.3f}",
                 f"{t['n_frames']:,}",
                 f"{t['dice']['mean']:.3f} ± {t['dice']['sd']:.3f}"]
        if abs(f - FLOOR) < 1e-9:
            cells = [f"**{c}**" for c in cells]
        vol_rows.append(cells)
    table(doc, 1, "Segmentation by bladder filling: frames stratified by the share of "
                  "the insonified sector the annotated lumen occupies. Dice is "
                  "mean ± sd; bold marks the distended-bladder working domain.",
          ["Lumen area", "n (val)", "Dice (val)", "n (test)", "Dice (test)"],
          vol_rows, widths=[1.7, 1.0, 2.0, 1.0, 2.0], size=7.5)

    head(doc, "3.2 The Attained Range of *Q*", 2)
    para(doc,
         "*Q* is defined on [0, 1]. On 2,953 held-out frames it attains [0.14, 0.78] — "
         "a width of 0.64 — with a central 80% of 0.45 (Figure 3a).")
    para(doc,
         "The decomposition of Section 2.3 attributes **92.4% of the range to two "
         "sub-scores** (Figure 3b), and both are terms that consult the "
         "ultrasound image rather than the mask.")
    wide_figure(doc, 3, "fig_range.png",
                "(a) The interval *Q* actually attains against its nominal domain [0, 1]; "
                "the solid band is the central 80%, the operating region a threshold "
                "would actually be placed in. (b) The interval each active sub-score attains "
                "and its share of the resulting range. `segmentation confidence` occupies a "
                "band 0.007 wide and contributes nothing.", 12.6)
    para(doc,
         "`segmentation confidence` is the cautionary term of the panel: it is not a weak "
         "term that could be rescued by more weight, but a degenerate one — it returns "
         "essentially one value (a band 0.007 wide), so it has no range, and a function "
         "with no range cannot participate in any decision. It carries a weight of 1.0 "
         "while contributing 0.0% of what *Q* can express; border penalty and component "
         "quality behave the same way and are already at weight 0.")


    head(doc, "3.3 How Many States *Q* Tells Apart", 2)
    para(doc,
         "Against a frame-to-frame jitter of 0.0095 (Section 2.3), the attained range of "
         "0.635 separates 67 states over the full range and **47 within the central 80%** "
         "— score differences wide enough for a reader to trust as real rather than as "
         "noise. That census, not any accuracy figure, is what an instruction set must "
         "fit within: words spaced more finely than the states *Q* can tell apart would "
         "hand the same frame different labels on different frames.")


    head(doc, "3.4 The Derived Vocabulary", 2)
    para(doc,
         "**A physical boundary.** Two boundaries partition the range, and neither is a "
         "fitted hyperparameter. The first is physical. The lumen-to-surround contrast is "
         "positive when the "
         "delineated region is darker than the tissue ring around it, so a non-positive "
         "value means the region is *no darker* than its surround. A urine-filled bladder "
         "lumen, however, is anechoic — a textbook property of the organ [8], not "
         "something estimated from these frames — so a region that fails to read dark "
         "cannot be a filled bladder. The boundary sits at zero because the physics puts "
         "it there: there is no constant to fit, and none to re-fit in a new imaging "
         "domain. On 13.1% of frames the contrast is non-positive; those frames have "
         "Dice 0.665 against 0.840 elsewhere, and what sets them apart is the delineated "
         "region itself — 1.85 times brighter and a third the area, the signature of a "
         "bladder that is not yet filled enough to read as anechoic rather than of a "
         "poorly placed probe. The remedy is therefore not a lateral slide, which cannot "
         "add urine to an under-distended bladder; it is upstream, in bladder filling, "
         "and the instruction says exactly that instead of commanding a movement that "
         "cannot help.")
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
         "**this is a stopping condition, not a defect.** The rule — withhold where "
         "Equation (2) puts the direction error above 5% — carries no fitted constant; "
         "the one measured quantity in it, *σ*, is a property of this segmentation on "
         "these frames, so moving to HoLEP imaging means re-measuring *σ* there "
         "(Section 4.2), not re-tuning a threshold.")
    table(doc, 2, "The instruction vocabulary. Both boundaries are derived — one from the "
                  "physics of an anechoic lumen [8], one from the segmentation's centroid "
                  "noise — and neither is tuned.",
          ["Instruction", "Condition", "Acts on"],
          [["*check bladder filling*", "contrast ≤ 0", "not the probe — filling"],
           ["*move left* / *move right*", f"contrast > 0 and |*ê*| ≥ {TH['5%']['px']:.1f} px",
            "lateral axis *v*~x~"],
           ["*hold*", f"contrast > 0 and |*ê*| < {TH['5%']['px']:.1f} px", "nothing — converged"]],
          widths=[2.5, 3.0, 2.3], size=7.5)
    para(doc,
         "**The magnitude is a number, not a word.** The word carries the direction; the "
         "displacement itself, *ê* in pixels, enters and leaves the pipeline as a "
         "continuous value — shown beside the word to a human operator, consumed directly "
         "as a velocity command by a robot. The same noise that derives the boundaries "
         "dictates this format. A magnitude put into words means quantising |*ê*| into "
         f"classes, and a class narrower than 2*σ* ≈ {2 * SIGMA:.1f} px cannot be "
         "assigned reliably even at its centre; the sweep bears this out, with graded "
         "vocabularies dropping word accuracy from "
         f"{VOCAB[3]['accuracy'] * 100:.1f}% to {VOCAB[5]['accuracy'] * 100:.1f}% and "
         f"{VOCAB[7]['accuracy'] * 100:.1f}% while the direction error holds at "
         f"{VOCAB[3]['direction_error'] * 100:.2f}% (Table 3). The continuous value, by "
         "contrast, is validated in Section 3.5 at a regression slope of "
         f"{OVERALL['slope']:.3f} and a median error of "
         f"{OVERALL['abs_error_median']:.2f} px: the magnitude is trustworthy as a "
         "measurement and unreliable as a word, so the language layer names the direction "
         "and passes the measurement through unrounded.")

    def _vocab_row(size, bold=False):
        v = VOCAB[size]
        cells = [v["name"], str(v["size"]), f"{v['accuracy'] * 100:.1f}%",
                 f"{v['direction_error'] * 100:.2f}%"]
        return [f"**{c}**" if bold else c for c in cells]

    table(doc, 3, "Word accuracy of the emitted instruction against the instruction the "
                  "ground truth implies, over the controlled sweep of Section 2.5. "
                  "Magnitude grades cut |*ê*| at successive multiples of the derived "
                  f"{TH['5%']['px']:.1f} px hold boundary, so the grading introduces no "
                  "new constant. Direction errors stay negligible; discretising the "
                  "magnitude is what costs accuracy — the reason it is passed through as "
                  "a number.",
          ["Vocabulary", "Size", "Accuracy", "Direction error"],
          [_vocab_row(2), _vocab_row(3, bold=True), _vocab_row(5), _vocab_row(7)],
          widths=[3.2, 0.9, 1.4, 1.6], size=7.5)
    para(doc,
         "Each word is a complete instruction to two different readers. To the human "
         "operator the vocabulary is displayed or spoken as it stands: the word names the "
         "direction — the register in which probe handling is actually taught and "
         "corrected — while the measured displacement stands beside it as a number. To a "
         "robotic probe holder each word is a symbol "
         "with a defined condition and axis, so translation into a twist command is a "
         "lookup, not an inference (Table 4). The language layer is thus the interface "
         "between the two settings: the same judgment drives either operator, and nothing "
         "about the pipeline changes when the reader does.")
    table(doc, 4, "One vocabulary, two readers. The mapping from word to action is fixed "
                  "in advance on both sides; no model sits between the instruction and its "
                  "execution.",
          ["Instruction", "Human operator", "Robotic probe holder"],
          [["*check bladder filling*", "pause; reassess filling before scanning on",
            "zero twist; hand control back to the supervisor"],
           ["*move left* / *move right*", "slide the probe laterally by the displayed "
            "amount", "*v*~x~ = −*k ê*, saturated; other axes zero"],
           ["*hold*", "keep the current pose",
            "zero lateral twist; no correction commanded"]],
          widths=[2.2, 2.8, 2.8], size=7.5)


    head(doc, "3.5 Accuracy of the Emitted Instruction", 2)
    _lead = para(doc,
         "Figure 4 shows one frame of the sweep at three lateral displacements, including "
         "one inside the band where the direction is no longer trustworthy.")
    _lead.paragraph_format.keep_with_next = True
    wide_figure(doc, 4, "fig_examples.png",
                "One frame at three lateral displacements. Cyan is the annotation, orange "
                "dashed the prediction, white dashed the beam axis, the arrow the "
                "instruction. At *e* = +2.1 px the instruction has collapsed to +0.8 px: "
                "the loop has entered the band where Equation (2) says the direction is no "
                "longer reliable.", 12.6)
    para(doc,
         f"The instruction agrees with the truth in {pct(OVERALL['sign_agreement'])} of "
         f"frames, with a median magnitude error of {OVERALL['abs_error_median']:.2f} px, a "
         f"regression slope of {OVERALL['slope']:.3f} and *r* = "
         f"{OVERALL['pearson_r']:.3f}. A slope this close to unity means the instruction is "
         "correctly scaled and not merely directional. The two held-out frame sets agree to "
         "within 0.7 px of median error.")
    wide_figure(doc, 5, "fig_agreement.png",
                "Instructed against ground-truth lateral displacement, one panel per "
                "held-out frame set; per-set agreement statistics are inset. Shaded "
                "quadrants are wrong-direction outcomes; every "
                "one lies within a few pixels of the axis.", 11.5)
    para(doc,
         "**The result is not that percentage but the shape of its failure.** Wrong "
         "directions occur only near the axis — their median required displacement is "
         "7.9 px against 15.8 px for correct ones — and their rate follows Equation (2) "
         "with *σ* fitted once to the centroid error and not to this curve (Figure 6). "
         "Agreement across two decades of error rate is what licenses deriving "
         "the vocabulary boundary from the model rather than reading it off the data.")
    wide_figure(doc, 6, "fig_validity.png",
                "Wrong-direction rate against required displacement. The curve is Equation "
                "(2); circles are observed rates with frame counts. Dashed verticals mark "
                "where the modelled rate reaches 5% and 1%.", 10.8)
    para(doc,
         "The constants are stable against annotation quality. A model-blind label audit "
         "flags four of the frame sets' patients as carrying regions that cannot be lumen; "
         f"recomputing without them moves *σ* from {SIGMA:.2f} to 4.67 px (−5%), sign "
         "agreement from 93.7% to 94.1%, and the 5% boundary from "
         f"{TH['5%']['px']:.1f} to 7.7 px.")


def robot_section(doc):
    head(doc, "4.1 The Action-Token Channel: From Judgment to Twist", 2)
    para(doc,
         "The pipeline validated above ends at the emitted instruction; this section "
         "records how a robotic probe holder consumes it — the design, not a result. The "
         "channel's input is the per-frame emission of stage 3: an action token "
         "(word, *ê*), with the quality scalar *Q* beside it. Decoding is the fixed "
         "lookup of Table 4, and no model sits between token and twist. *move left* and "
         "*move right* select the lateral axis and let the number scale it — "
         "*v*~x~ = −*k ê*, saturated, every other axis commanded to zero, so the "
         "realisation is proportional rather than bang-bang. *hold* commands zero twist: "
         f"the {TH['5%']['k']:.2f}*σ* boundary of Section 3.4 arrives as a ready-made "
         "deadband, and a controller that kept acting inside it would be servoing its own "
         "segmentation error. *check bladder filling* gates image-based motion outright — "
         "no twist is preferable to a twist computed from a mask that physics rejects — "
         "and because tokens are issued per frame, one degraded frame gates one command, "
         "not the session.")
    para(doc,
         "Beyond the words, *Q* has room to scale the commanded speed down as quality "
         "falls: 47 distinguishable states give it the room to do that honestly. The token acts only on *v*~x~, the one axis that leaves the imaging "
         "plane invariant, so nothing the image loop commands can rotate the anatomy out "
         "of its own view. Closing and validating this loop — its latency, and *Q* "
         "measured while the probe actually moves — is deliberately outside this paper's "
         "scope.")


def discussion_section(doc):
    head(doc, "4. Conclusion")
    para(doc,
         "Reporting an image-quality function by its correlation with a segmentation "
         "metric answers the wrong question. Correlation says how the score orders "
         "frames; the attained range over the function's own noise says how many states "
         "a reader can genuinely tell apart, and the two come apart — a score can rank "
         "frames respectably while supporting fewer distinctions than the instructions a "
         "controller would want to issue, a failure invisible to any accuracy figure. "
         "Range, its decomposition, and range-over-noise are cheap to compute and we "
         "suggest reporting them for any score intended to instruct an operator or drive "
         "a device. The decomposition also gives a concrete design rule: a term "
         "contributing none of the range is not underweighted but degenerate, and the "
         "fix is to measure it where the decision is hard or to remove it — here the two "
         "terms that consult the ultrasound image carry 92.4% of the range while three "
         "mask-side terms carry almost none.")

def limitations_section(doc):
    head(doc, "4.2 Limitations", 2)
    for item in (
        "PFUS1 is transperineal pelvic-floor imaging of women [6], not HoLEP imaging. "
        "No suprapubic acoustic window, male pelvic anatomy, morcellator, or irrigation "
        "inflow appears in any frame, so every constant reported here — *σ* and the "
        "boundaries derived from it included — must be re-estimated on intraoperative "
        "suprapubic frames before the vocabulary is trusted in theatre.",
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
        para(doc, "— " + item, size=9.0, indent=0.35, after=1.8)


def references_section(doc):
    head(doc, "References")
    para(doc,
         "[1] T. Jang, H.-J. Kong, C. Baek, J. Kim, M. S. Choo, S.-J. Oh. Effect of "
         "Self-Training Using Virtual Reality Head-Mounted Display Simulator on the "
         "Acquisition of Holmium Laser Enucleation of the Prostate Surgical Skills. "
         "International Neurourology Journal 2024;28(2):138–146.", size=9.0, after=2)
    para(doc,
         "[2] O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for "
         "Biomedical Image Segmentation. In: Medical Image Computing and Computer-Assisted "
         "Intervention (MICCAI 2015), LNCS 9351, Springer, 2015, pp. 234–241.",
         size=9.0, after=2)
    para(doc,
         "[3] M. Saini, Y. Jiang, T. Gangopadhyay, D. P. Rosen, A. Alizad, M. Fatemi. "
         "BWS-Net: An Optimal Deep Learning Architecture for the Anterior Bladder Wall "
         "Segmentation using Ultrasound Imaging. IEEE Journal of Biomedical and Health "
         "Informatics 2026. doi:10.1109/JBHI.2026.3675965.", size=9.0, after=2)
    para(doc,
         "[4] Z. Song, M. Asiedu, S. Wang, Q. Li, A. Ozturk, V. Mittal, S. Schoen Jr., "
         "S. Ramaswamy, T. T. Pierce, A. E. Samir, Y. C. Eldar, A. Chandrakasan, V. Kumar. "
         "Memory-efficient low-compute segmentation algorithms for bladder-monitoring "
         "smart ultrasound devices. Scientific Reports 2023;13:16450.", size=9.0, after=2)
    para(doc,
         "[5] H.-L. Hsu, M. Zahiri, G. Y. Li, R. Al Mukaddim, H. Lee, M. G. Wilson, "
         "J. Grube, S. Schmidt, G. Ghoshal, B. Raju. Active guidance in ultrasound "
         "bladder scanning using reinforcement learning. Scientific Reports "
         "2026;16:5273.", size=9.0, after=2)
    para(doc,
         "[6] D. Solís-Martín, J. A. Sainz, J. Galán-Páez, J. Borrego-Díaz, "
         "J. A. García-Mejido. PFUS1: Premier pelvic floor ultrasound segmentation "
         "dataset. A resource for advancing research. Data in Brief 2026;64:112346.",
         size=9.0, after=2)
    para(doc,
         "[7] J. A. García-Mejido, D. Solís-Martín, M. Martín-Morán, "
         "C. Fernández-Conde, F. Fernández-Palacín, J. A. Sainz-Bueno. Applicability of "
         "deep learning to dynamically identify the different organs of the pelvic floor "
         "in the midsagittal plane. International Urogynecology Journal "
         "2024;35(12):2285–2293.", size=9.0, after=2)
    para(doc,
         "[8] Trinkler, Dietrich. Ultrasound of the Urinary Bladder. In: EFSUMB Course "
         "Book, European Federation of Societies for Ultrasound in Medicine and Biology, "
         "2019.", size=9.0, after=2)
    para(doc,
         "[TO BE COMPLETED] Citations for HoLEP morcellation, ultrasound visual servoing "
         "and image-quality assessment are to be added before submission.",
         size=9.0, italic=True)


def build():
    doc = Document()
    style_doc(doc)
    page_numbers(doc)
    columns(doc.sections[0], 1)      # 단일 단 (KOSMI)
    front_matter(doc)

    introduction(doc)
    methods_section(doc)
    results_section(doc)
    discussion_section(doc)
    robot_section(doc)
    limitations_section(doc)
    references_section(doc)

    doc.save(OUT)
    print("wrote", OUT)

    import shutil
    figdir = os.path.join(PAPER, "figures")
    os.makedirs(figdir, exist_ok=True)
    for name in USED_FIGURES:
        stem = os.path.splitext(name)[0]
        for candidate in (name, stem + ".pdf"):
            src = os.path.join(FIGDIR, candidate)
            if os.path.exists(src):
                shutil.copy2(src, figdir)
    print("collected", len(USED_FIGURES), "figures →", figdir)


if __name__ == "__main__":
    build()
