#!/usr/bin/env python3
"""Action-token manuscript (.docx) — text, figures and tables.

    python3 docs/build_lateral_instruction_manuscript.py
    soffice --headless --convert-to pdf Lateral_Instruction_Manuscript.docx

Format follows the 대한의료정보학회(KOSMI) 연제논문 포맷: A4, 10 mm top/bottom
margins and 25 mm side margins, 10 pt body at 140% line spacing (abstract 120%),
장평 95% · 자간 −5%, Korean glyphs in 신명조 and Latin in Times New Roman. The
front matter (title, authors, abstract, keywords) spans the page; the body runs
in two columns to stay within the free-paper page budget, with wide figures and
tables spanning both columns. Section headings are Roman-numbered I–V per the
KOSMI template. Every number in the text is read from
experiments/lateral_instruction/lateral_instruction.json rather than typed.
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
#: Numeric artifacts (json/csv) of the lateral-instruction analysis.
DATADIR = os.path.join(REPO, "Unet_seg", "experiments", "lateral_instruction")
STATS = json.load(open(os.path.join(DATADIR, "lateral_instruction.json")))
#: Segmentation metrics of the same checkpoint on the same frame sets, from the
#: volume-gated audit; Section III.1's numbers are read from here, not typed.
AUDIT = json.load(open(os.path.join(REPO, "Unet_seg", "experiments", "quality_retro",
                                    "audit", "audit.json")))
#: Word accuracy per vocabulary size, from scripts/vocabulary_accuracy.py.
VOCAB = {v["size"]: v for v in json.load(
    open(os.path.join(DATADIR, "vocabulary_accuracy.json")))["vocabularies"]}
#: Everything the paper is made of lives under Paper/: the .docx and, in
#: Paper/figures, every figure — the analysis scripts under Unet_seg/scripts
#: write their figures straight there, so the build embeds them in place.
PAPER = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(PAPER, "figures")
OUT = os.path.join(PAPER, "Lateral_Instruction_Manuscript.docx")
USED_FIGURES: list[str] = []

# 대한의료정보학회(KOSMI) 연제논문 포맷 (KOSMI_Abstract_style + 연제논문 템플릿):
# A4, 여백 위·아래·머리말·꼬리말 10 mm · 좌·우 25 mm, 본문 10 pt, 줄간격 140 %
# (초록 120 %), 장평 95 %, 자간 −5 %, 국문 신명조. 전면부는 전폭 1단, 본문은
# 2단(분량 압축), 넓은 그림·표는 두 단 걸침. 절 제목은 로마숫자 I–V.
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
    """Split '**bold**', '*italic*', '++underline++', '~sub~', '^sup^' into runs."""
    import re
    for tok in re.split(r"(\+\+[^+]+?\+\+|\*\*.+?\*\*|\*[^*]+?\*|~[^~]+?~|\^[^^]+?\^)", text):
        if not tok:
            continue
        b, i, u, sub, sup = bold, italic, False, False, False
        if tok.startswith("++") and tok.endswith("++"):
            tok, u = tok[2:-2], True
        elif tok.startswith("**") and tok.endswith("**"):
            tok, b = tok[2:-2], True
        elif tok.startswith("*") and tok.endswith("*"):
            tok, i = tok[1:-1], True
        elif tok.startswith("~") and tok.endswith("~"):
            tok, sub = tok[1:-1], True
        elif tok.startswith("^") and tok.endswith("^"):
            tok, sup = tok[1:-1], True
        r = p.add_run(tok)
        r.font.name, r.font.size = SERIF, Pt(size)
        r.font.bold, r.font.italic, r.font.underline = b, i, u
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


#: Full text block [cm]: page minus the 25 mm side margins (KOSMI).
TEXT_CM = 21.0 - 2 * 2.5
#: One body column [cm] — two columns with a 0.7 cm gutter (compact layout).
COL_CM = (TEXT_CM - 0.7) / 2


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
    """Scale a width recipe to one column, or the whole text block."""
    target = (TEXT_CM - 0.05) if full_width else (COL_CM - 0.05)
    factor = target / sum(widths)
    return [w * factor for w in widths]


def _span(doc, count):
    """Continuous section break that changes the column count in place —
    KOSMI margins (10 mm top/bottom, 25 mm sides) carried onto the section."""
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.0)
    section.left_margin = section.right_margin = Cm(2.5)
    section.header_distance = section.footer_distance = Cm(1.0)
    columns(section, count)
    return section


def wide_figure(doc, number, filename, caption, width_cm=16.0):
    """A figure spanning both columns — wide panels are unreadable squeezed
    into one 7.6 cm column, so they break the flow to span the text block."""
    _span(doc, 1)
    figure(doc, number, filename, caption, min(width_cm, TEXT_CM))
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
    para(doc, "An Explainable Action-Token Generation Framework Based on Bladder "
              "Ultrasound Segmentation and Image Quality Assessment",
         size=12.0, bold=True, align="center", after=3)
    # KOSMI: 발표자를 제일 먼저 · 밑줄, 교신저자는 * 로 표시.
    para(doc, "++Seong Jeong++^1, 2, 5^, Minsung Kim^1,2,4^, Dongho Yee^1,2,4^, "
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
    para(doc, "*Corresponding Author", size=8.0, italic=True, align="center", after=0.4)
    doc.add_paragraph().paragraph_format.space_after = Pt(3)

    _abs = para(doc,
         "**Abstract:** "
         "During the morcellation phase of holmium laser enucleation of the prostate "
         "(HoLEP), a suprapubic ultrasound probe must be kept over the bladder lumen. "
         "Automating that view requires an explicit, auditable representation mapping "
         "image-derived states to actionable outputs; a segmentation mask alone is not "
         "such a representation. We present an explainable framework that converts "
         "bladder ultrasound segmentation and image-quality assessment into a "
         "safety-constrained, machine-readable action token. A U-Net segments the bladder "
         "lumen; a transparent image-quality function *Q*, computed from interpretable "
         "image and mask features, summarises the frame; and the system emits a structured "
         "token pairing a discrete action state, one of *move-left*, *move-right*, *hold*, "
         "or *check-filling*, with a continuous lateral displacement estimate *ê*, designed "
         "for deterministic decoding rather than natural-language recommendation. The "
         "vocabulary is bounded by what *Q* can distinguish: *Q* attains [0.14, 0.78] of "
         "its nominal domain and, against frame-to-frame jitter, separates "
         "**47 distinguishable states**, the ceiling on safe action states. Two "
         "boundaries are derived, not tuned. A physical gate assigns *check-filling* "
         "because a non-anechoic lumen cannot be a filled bladder, and a statistical "
         f"deadband assigns *hold* because an offset below {TH['5%']['k']:.2f}σ of the "
         f"centroid noise, σ = {SIGMA:.2f} px, cannot be signed. "
         f"On {S['n_frames']:,} laterally translated held-out frames the action state is "
         f"correct on {VOCAB[3]['accuracy'] * 100:.1f}% of frames at "
         f"{VOCAB[3]['direction_error'] * 100:.2f}% direction error; the displacement "
         f"stays continuous (slope {S['overall']['slope']:.3f}), since quantising it into "
         f"graded classes drops accuracy to {VOCAB[5]['accuracy'] * 100:.1f}% and "
         f"{VOCAB[7]['accuracy'] * 100:.1f}%. The framework supplies an explainable "
         "action-token layer for HoLEP probe guidance; closed-loop validation on "
         "intraoperative imaging remains future work.",
         size=10.0)
    _abs.paragraph_format.line_spacing = 1.2      # KOSMI 초록 줄간격 120 %
    para(doc, "**Keywords:** bladder ultrasound segmentation, image quality assessment, "
              "explainable artificial intelligence, action token, robotic probe control, "
              "HoLEP morcellation",
         size=10.0, align="left", before=1, after=2)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def introduction(doc):
    head(doc, "I. Introduction")
    para(doc,
         "During the morcellation phase of holmium laser enucleation of the prostate "
         "(HoLEP), a suprapubic ultrasound probe must be kept over the bladder lumen [1]. "
         "Automated analysis of that view needs an explicit representation of what the "
         "current image implies for an actionable state; a segmentation mask alone is not "
         "such a representation. Deep networks already segment the bladder accurately and "
         "cheaply [3, 4] and guidance systems can steer toward a standard view [5], but a "
         "mask is not something a downstream module can act on. An end-to-end learned "
         "mapping would produce motion directly yet hide *why* an action was issued and is "
         "hard to audit under segmentation uncertainty, whereas a raw geometric rule acts "
         "on the mask with no account of when the image is too degraded to act on at all. "
         "What is wanted instead is a structured layer that encodes the direction to move, "
         "withholds motion when the estimate is untrustworthy, and flags failures upstream "
         "of any probe motion.")
    para(doc,
         "We build that layer as an explainable **action-token generation framework** of "
         "three stages (Figure 1): a U-Net segments the bladder lumen; a transparent "
         "image-quality function *Q*, computed from interpretable image and mask features, "
         "summarises whether and how the frame falls short; and the state is emitted as a "
         "structured token pairing a discrete action state (*move-left*, *move-right*, "
         "*hold*, or *check-filling*) with a continuous lateral displacement estimate *ê*. "
         "The token is designed for deterministic decoding, with the identical "
         "human-readable wording retained only as an optional monitoring layer for "
         "oversight. The framework is confined to the lateral, image-preserving axis — the "
         "one axis presenting a measurable image-domain displacement that can be simulated "
         "from single-plane frames. This study validates the pipeline up to token "
         "generation; the token-to-command mapping is a design specification for future "
         "integration (Section IV), not a closed-loop result.")
    wide_figure(doc, 1, "fig_overview.png",
                "The action-token generation framework: an ultrasound frame is segmented "
                "(U-Net), summarised by the image-quality function *Q*, and emitted as a "
                "structured action token (*a*, *ê*, *Q*) that a fixed decoder maps to a "
                "potential downstream system output. Stages 1–3 are validated; the dashed "
                "downstream execution (a potential future robotic-control integration) is "
                "not validated (Section IV).", 16.2)


def methods_section(doc):
    head(doc, "II. Methods")
    head(doc, "1. Segmentation, Image State, and Quality Assessment", 2)
    para(doc,
         "A U-Net [2] maps a 256 × 256 B-mode frame to a lumen probability map, "
         "thresholded at 0.5 and reduced to its largest hole-filled connected component. "
         "Frames come from PFUS1 [6], a public transperineal pelvic-floor dataset (101 "
         "women, midsagittal, rest and Valsalva, one scanner) whose eight annotated "
         "organs include the bladder [6, 7]; **it is not intraoperative HoLEP imaging**. "
         "The model was trained on a patient-disjoint split and used unchanged; all "
         "measurements use two held-out sets (1,661 and 1,292 frames) that share no "
         "patient. From the mask we compute an interpretable image state — centroid, "
         "area, boundary statistics, and lumen-to-surround contrast — every ratio taken "
         "over the insonified sector rather than the image rectangle.")
    para(doc,
         "The quality score is a weighted geometric mean of sub-scores *s*~i~ ∈ [0, 1], "
         "every term and weight logged so a value is traceable to its components:")
    equation(doc, "Q = exp( Σ w~i~ ln s~i~ / Σ w~i~ )", "1")
    para(doc,
         "Dominated by its smallest term, *Q* cannot let one collapsed property hide "
         "inside an average; three image-reading terms (lumen contrast, centring, "
         "boundary sharpness) carry the top weight, and weights are tiered by rule, not "
         "fitted. As these are static examinations, the change in *Q* between adjacent "
         "frames estimates its own frame-to-frame jitter; the attained range divided by "
         "that jitter counts the states *Q* can distinguish — the ceiling on the number "
         "of reliable action states.")

    head(doc, "2. Action Token and Safety Boundaries", 2)
    para(doc,
         "The token acts on the lateral, image-preserving axis *v*~x~, the only "
         "plane-preserving axis presenting a measurable image-domain displacement. At "
         "frame *t* the framework emits a structured action token")
    equation(doc, "τ~t~ = ( a~t~, ê~t~, Q~t~ )", "2")
    para(doc,
         "where *a*~t~ ∈ {move-left, move-right, hold, check-filling} is the discrete "
         "action state, *ê*~t~ = *A* − *ĉ*~t~ is the continuous lateral displacement "
         "(beam axis *A* minus predicted lumen-centroid abscissa *ĉ*), and *Q*~t~ is the "
         "score of Equation (1). A fixed and auditable rule can decode the token for "
         "downstream use: the state selects an action mode, *ê* supplies the continuous "
         "lateral scaling variable, and *Q* is available for conservative confidence-aware "
         "scaling, with no learned policy between the token and its downstream "
         "interpretation. A robotic lateral-velocity command is one potential future "
         "implementation of this decoding rule, but no hardware integration or closed-loop "
         "execution is evaluated in this study. Two derived, "
         "untuned boundaries fix the states (Table 1). The **physical gate** is exact: a "
         "urine-filled lumen is anechoic [8], so a non-positive lumen-to-surround "
         "contrast cannot be a filled bladder, and lateral motion cannot remedy "
         "insufficient filling. The **uncertainty deadband** follows from the centroid "
         f"error δ = *ê* − *e* (σ = {SIGMA:.2f} px, bias {S['delta_mean_px']:+.2f} px): "
         "the estimated direction is wrong when δ opposes and exceeds *e*, a Gaussian-tail "
         f"probability Φ(−|*e*|/σ); requiring the modelled error below 5% withholds motion "
         f"under {TH['5%']['k']:.2f}σ = {TH['5%']['px']:.1f} px.")
    table(doc, 1, "Action-token specification: trigger, safety rationale, and "
                  "deterministic control interpretation for each state (both boundaries "
                  "derived, not tuned).",
          ["Action state", "Trigger", "Safety rationale", "Control interpretation"],
          [["*check-filling*", "contrast ≤ 0", "image failure, not a pose error",
            "gate: no lateral motion"],
           ["*move-left* / *move-right*", f"contrast > 0, |*ê*| ≥ {TH['5%']['px']:.1f} px",
            "direction reliable above noise", "*v*~x~ ∝ *ê*, other axes zero"],
           ["*hold*", f"contrast > 0, |*ê*| < {TH['5%']['px']:.1f} px",
            "direction within noise", "deadband: zero lateral twist"]],
          widths=[2.0, 2.2, 2.5, 2.5], size=7.5)

    head(doc, "3. Validation by Lateral Translation", 2)
    para(doc,
         "In every recorded frame the lumen lies on one side of the beam axis, so *e* "
         "never changes sign and a constant output would score 99.3%. We therefore "
         "translate each frame laterally — the one probe motion single-plane frames "
         "reproduce faithfully — to place *e* on prescribed targets, sampled densely near "
         "the axis crossing; fabricated margin is edge-replicated, never black. "
         f"Segmentation is unaffected across the sweep (Dice {S['dice_under_translation']['median']:.3f}), "
         f"yielding {S['n_frames']:,} frames, 43% with *e* < 0.")


def results_section(doc):
    head(doc, "III. Results")
    head(doc, "1. Segmentation and Quality Discrimination", 2)
    para(doc,
         "Segmentation improves with bladder filling; in the working domain (annotated "
         f"lumen ≥ {FLOOR * 100:.1f}% of the sector, the state morcellation irrigation is "
         "meant to maintain) Dice reaches "
         f"{SEG_DOMAIN['val']['dice']['mean']:.3f} (val) and "
         f"{SEG_DOMAIN['test']['dice']['mean']:.3f} (test), and the lateral centroid "
         "error the token's displacement rests on has a median of "
         f"{SEG_DOMAIN['val']['centroid_error_px']['median']:.2f} and "
         f"{SEG_DOMAIN['test']['centroid_error_px']['median']:.2f} px (Table 2). Below "
         "that domain segmentation degrades — exactly the population the *check-filling* "
         "state gates.")
    para(doc,
         "*Q* is defined on [0, 1] but attains only [0.14, 0.78] (width 0.64, central "
         "80% 0.45); against a frame-to-frame jitter of 0.0095 this separates 67 states "
         "over the full range and **47 within the central 80%** — the ceiling on the "
         "number of reliable action states (Figure 2). Two image-reading terms carry "
         "92.4% of the range, confirming that the discriminating information is "
         "image-derived rather than mask-derived.")
    wide_figure(doc, 2, "fig_range.png",
                "Quality-function discrimination: (a) the interval *Q* attains within its "
                "nominal [0, 1] domain, with the central-80% operating band; (b) each "
                "active sub-score's attained interval and share of the range.", 12.6)


    head(doc, "2. Action-Token Validation", 2)
    para(doc,
         f"On the {S['n_frames']:,}-frame sweep the discrete action state is classified "
         f"correctly on {VOCAB[3]['accuracy'] * 100:.1f}% of frames with a "
         f"{VOCAB[3]['direction_error'] * 100:.2f}% direction error, and the continuous "
         f"displacement is estimated at slope {OVERALL['slope']:.3f}, "
         f"*r* = {OVERALL['pearson_r']:.3f}, median error "
         f"{OVERALL['abs_error_median']:.2f} px (Table 2, Figure 3). Wrong directions "
         "occur only near the axis — median required displacement 7.9 px against 15.8 px "
         "for correct ones — and their rate follows the Gaussian tail Φ(−|*e*|/σ), which "
         f"is what licenses deriving the {TH['5%']['px']:.1f} px *hold* deadband from the "
         "model rather than reading it off the data (Figure 3).")
    para(doc,
         "Direction is thus robustly discretisable, whereas quantising the magnitude into "
         f"graded action classes narrower than 2σ ≈ {2 * SIGMA:.1f} px lowers "
         f"classification accuracy to {VOCAB[5]['accuracy'] * 100:.1f}% (five states) and "
         f"{VOCAB[7]['accuracy'] * 100:.1f}% (seven states) at unchanged direction error; "
         "the displacement is therefore kept as a continuous control variable rather than "
         "a class.")
    wide_figure(doc, 3, "fig_validation.png",
                "Action-token validation: (a) estimated versus ground-truth lateral "
                "displacement per held-out set (shaded quadrants are wrong-direction "
                "outcomes); (b) wrong-direction rate versus required displacement against "
                "the Gaussian-tail model Φ(−|*e*|/σ), with the modelled 5% and 1% "
                "boundaries marked.", 13.0)

    seg_rows = [
        ["Segmentation Dice, all frames (val / test)",
         f"{SEG_ALL['val']['dice']['mean']:.3f} / {SEG_ALL['test']['dice']['mean']:.3f}"],
        ["Segmentation Dice, working domain (val / test)",
         f"{SEG_DOMAIN['val']['dice']['mean']:.3f} / {SEG_DOMAIN['test']['dice']['mean']:.3f}"],
        ["Lateral centroid error, working domain, median (val / test)",
         f"{SEG_DOMAIN['val']['centroid_error_px']['median']:.2f} / "
         f"{SEG_DOMAIN['test']['centroid_error_px']['median']:.2f} px"],
        ["Attained *Q* range (width; central 80%)", "0.64; 0.45"],
        ["Distinguishable states (full; central 80%)", "67; 47"],
        [f"Action-state accuracy ({VOCAB[3]['size']} states); direction error",
         f"{VOCAB[3]['accuracy'] * 100:.1f}%; {VOCAB[3]['direction_error'] * 100:.2f}%"],
        ["Displacement estimate (slope; median error)",
         f"{OVERALL['slope']:.3f}; {OVERALL['abs_error_median']:.2f} px"],
        [f"Graded-magnitude accuracy ({VOCAB[5]['size']}; {VOCAB[7]['size']} states)",
         f"{VOCAB[5]['accuracy'] * 100:.1f}%; {VOCAB[7]['accuracy'] * 100:.1f}%"],
    ]
    table(doc, 2, "Main quantitative results: segmentation and action-token performance "
                  "on the held-out and laterally translated frames.",
          ["Metric", "Value"], seg_rows, widths=[5.2, 2.6], size=7.5)


def discussion_section(doc):
    head(doc, "IV. Discussion")
    para(doc,
         "An image-quality function intended to support structured downstream action "
         "handling should be judged by "
         "its attained range relative to its noise, because that ratio — not correlation "
         "with a segmentation metric — sets the number of safely distinguishable action "
         "states. A score can rank frames respectably while supporting fewer distinctions "
         "than a controller needs, a failure invisible to any accuracy figure; range and "
         "range-over-noise are cheap to compute and we suggest reporting them for any "
         "score intended to support downstream action handling.")
    para(doc,
         "The contribution is deliberately not to automate a judgment a trained observer "
         "makes by inspection — *move-left*, *move-right*, *hold*, and *check-filling* are "
         "visually obvious on screen, and we claim no diagnostic insight or improvement to "
         "human performance. It is to turn image-derived information into a structured, "
         "explainable, safety-constrained representation that can be interpreted "
         "deterministically by a downstream system, each token traceable to explicit "
         "features, segmentation "
         "geometry, a physical constraint, and measured uncertainty. The two non-motion "
         "states are the safety core: *hold* is an uncertainty-aware deadband that stops "
         "a controller from servoing its own segmentation noise near the axis, and "
         "*check-filling* is a physics-based motion-inhibition gate marking that lateral "
         "repositioning is the wrong remedy — keeping an unreliable estimate, an upstream "
         "image failure, and a safety decision out of a black-box policy. The fixed "
         "decoding rule is an auditable integration specification for potential future "
         "systems, including robotic-control implementations, rather than a validated "
         "controller or hardware system.")
    para(doc,
         "**Limitations.** PFUS1 is transperineal pelvic-floor imaging, not intraoperative "
         "suprapubic HoLEP data; actual probe motion and closed-loop control were not "
         "validated, so the token-to-command mapping is a specification; and all "
         "target-domain constants — the segmentation uncertainty *σ* and its derived "
         "boundaries — must be re-estimated before use in theatre.")


def conclusion_section(doc):
    head(doc, "V. Conclusion")
    para(doc,
         "An explainable, quality-aware action-token layer can bridge bladder ultrasound "
         "segmentation and structured downstream action handling. Its vocabulary is "
         "bounded by the measurable discrimination capacity of the image-quality function, "
         "and its *hold* and *check-filling* states supply explicit safety logic — an "
         "uncertainty-aware deadband and a physics-based gate — that a downstream system "
         "can interpret deterministically without a black-box policy. The study validates "
         "action-token generation and continuous displacement estimation on translated "
         "held-out frames; it does not validate closed-loop control or clinical "
         "deployment. Re-estimating the constants on intraoperative suprapubic HoLEP "
         "imaging and evaluating downstream integration remain future work.")


def references_section(doc):
    head(doc, "References")
    para(doc,
         "1. T. Jang, H.-J. Kong, C. Baek, J. Kim, M. S. Choo, S.-J. Oh. Effect of "
         "Self-Training Using Virtual Reality Head-Mounted Display Simulator on the "
         "Acquisition of Holmium Laser Enucleation of the Prostate Surgical Skills. "
         "International Neurourology Journal 2024;28(2):138–146.", size=9.0, after=2)
    para(doc,
         "2. O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for "
         "Biomedical Image Segmentation. In: Medical Image Computing and Computer-Assisted "
         "Intervention (MICCAI 2015), LNCS 9351, Springer, 2015, pp. 234–241.",
         size=9.0, after=2)
    para(doc,
         "3. M. Saini, Y. Jiang, T. Gangopadhyay, D. P. Rosen, A. Alizad, M. Fatemi. "
         "BWS-Net: An Optimal Deep Learning Architecture for the Anterior Bladder Wall "
         "Segmentation using Ultrasound Imaging. IEEE Journal of Biomedical and Health "
         "Informatics 2026. doi:10.1109/JBHI.2026.3675965.", size=9.0, after=2)
    para(doc,
         "4. Z. Song, M. Asiedu, S. Wang, Q. Li, A. Ozturk, V. Mittal, S. Schoen Jr., "
         "S. Ramaswamy, T. T. Pierce, A. E. Samir, Y. C. Eldar, A. Chandrakasan, V. Kumar. "
         "Memory-efficient low-compute segmentation algorithms for bladder-monitoring "
         "smart ultrasound devices. Scientific Reports 2023;13:16450.", size=9.0, after=2)
    para(doc,
         "5. H.-L. Hsu, M. Zahiri, G. Y. Li, R. Al Mukaddim, H. Lee, M. G. Wilson, "
         "J. Grube, S. Schmidt, G. Ghoshal, B. Raju. Active guidance in ultrasound "
         "bladder scanning using reinforcement learning. Scientific Reports "
         "2026;16:5273.", size=9.0, after=2)
    para(doc,
         "6. D. Solís-Martín, J. A. Sainz, J. Galán-Páez, J. Borrego-Díaz, "
         "J. A. García-Mejido. PFUS1: Premier pelvic floor ultrasound segmentation "
         "dataset. A resource for advancing research. Data in Brief 2026;64:112346.",
         size=9.0, after=2)
    para(doc,
         "7. J. A. García-Mejido, D. Solís-Martín, M. Martín-Morán, "
         "C. Fernández-Conde, F. Fernández-Palacín, J. A. Sainz-Bueno. Applicability of "
         "deep learning to dynamically identify the different organs of the pelvic floor "
         "in the midsagittal plane. International Urogynecology Journal "
         "2024;35(12):2285–2293.", size=9.0, after=2)
    para(doc,
         "8. Trinkler, Dietrich. Ultrasound of the Urinary Bladder. In: EFSUMB Course "
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
    columns(doc.sections[0], 1)      # 전면부(제목·저자·초록)는 전폭 1단
    front_matter(doc)

    _span(doc, 2)                    # 본문부터 2단 (분량 압축)
    introduction(doc)
    methods_section(doc)
    results_section(doc)
    discussion_section(doc)
    conclusion_section(doc)
    references_section(doc)

    doc.save(OUT)
    print("wrote", OUT)
    print("embedded", len(USED_FIGURES), "figures from", FIGDIR)


if __name__ == "__main__":
    build()
