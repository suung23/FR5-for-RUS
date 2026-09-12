#!/usr/bin/env python3
"""main.tex → ICRA_draft.docx.

이 저장소의 다른 원고는 본문이 파이썬 안에 있다 (`build_sonologger_manuscript.py`).
이 초안은 반대로 **본문이 main.tex 하나에만** 있고, 이 스크립트는 그것을 워드로 옮기기만
한다. 그래야 tex 와 docx 가 갈라지지 않는다. 고칠 곳은 언제나 main.tex 다.

    python3 Paper/icra_bladder/build_docx.py            # → ICRA_draft.docx
    soffice --headless --convert-to pdf ICRA_draft.docx # 필요하면 PDF

일반 LaTeX 변환기가 아니다. main.tex 가 쓰는 매크로만 안다 (siunitx 몇 개, cite, ref,
figure, tabular, itemize/enumerate, equation). 새 매크로를 tex 에 쓰면 여기도 늘려야
한다 — 모르는 명령은 조용히 지우지 않고 그대로 남겨 눈에 띄게 둔다.

워드로 옮기면서 **의도적으로 달라지는 것 둘**:
  * 단 구성. tex 는 2단이고 docx 는 1단이다. 편집용이지 제출용이 아니다.
  * 그림 배치. 부동체가 없으므로 본문 흐름의 그 자리에 들어간다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

HERE = Path(__file__).resolve().parent
FONT = "Times New Roman"

#: 통짜 그림의 최대 폭. 지면을 꽉 채우면 그 그림이 들어갈 자리를 못 찾아
#: 페이지 하나가 통째로 비는 일이 생긴다.
WIDE_MAX_IN = 5.6

#: 2단 조판. tex 와 같은 폭감각으로 읽히게 맞춘다 — 한 단이 약 3.4 in 다.
MARGIN_IN = 0.6
GUTTER_IN = 0.25
BODY_PT = 9.0
CAPTION_PT = 8.0


def page_geometry(section) -> None:
    section.left_margin = section.right_margin = Inches(MARGIN_IN)
    section.top_margin = section.bottom_margin = Inches(MARGIN_IN)


def column_width_in(section) -> float:
    total = section.page_width.inches - 2 * MARGIN_IN
    return (total - GUTTER_IN) / 2


def set_columns(section, n: int) -> None:
    """단 수를 정한다. python-docx 에 API 가 없어 sectPr 을 직접 만진다."""
    cols = section._sectPr.xpath("./w:cols")[0]
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(int(GUTTER_IN * 1440)))
    cols.set(qn("w:equalWidth"), "1")


def one_column_break(doc):
    """통짜 그림용. 1단 구간을 열고, 그 구간을 돌려준다."""
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    page_geometry(section)
    set_columns(section, 1)
    return section


def two_column_break(doc):
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    page_geometry(section)
    set_columns(section, 2)
    return section

# siunitx 단위 → 사람이 읽는 표기. 긴 것부터 맞춘다.
UNITS = [
    ("\\newton\\second\\per\\metre", "N·s/m"),
    ("\\newton\\per\\milli\\metre", "N/mm"),
    ("\\milli\\metre\\per\\second", "mm/s"),
    ("\\degree\\per\\minute", "°/min"),
    ("\\degree\\per\\second", "°/s"),
    ("\\radian\\per\\second", "rad/s"),
    ("\\metre\\per\\second", "m/s"),
    ("\\milli\\second", "ms"),
    ("\\micro\\metre", "µm"),
    ("\\milli\\metre", "mm"),
    ("\\milli\\litre", "mL"),
    ("\\kilo\\hertz", "kHz"),
    ("\\per\\newton", "/N"),
    ("\\giga MAC", "GMAC"),
    ("\\newton", "N"),
    ("\\second", "s"),
    ("\\minute", "min"),
    ("\\hertz", "Hz"),
    ("\\degree", "°"),
    ("\\percent", "%"),
    ("\\gram", "g"),
    ("\\metre", "m"),
]


def _unit(tex: str) -> str:
    out = tex
    for src, dst in UNITS:
        out = out.replace(src, dst)
    return out.strip()


def strip_comments(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        # 이스케이프되지 않은 %부터 줄 끝까지가 주석이다.
        cut = re.sub(r"(?<!\\)%.*$", "", line)
        lines.append(cut)
    return "\n".join(lines)


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
#: label -> 사람이 읽는 번호. build() 가 본문을 훑어 채운다. 비어 있으면 라벨 이름이
#: 그대로 나오는데, 그것이 "Fig. overview" 같은 흉한 참조의 정체였다.
LABELS: dict[str, str] = {}


def number_labels(body: str) -> None:
    """섹션·그림·표·수식 번호를 등장 순서대로 매긴다."""
    LABELS.clear()
    sec = fig = tab = eq = 0
    pattern = re.compile(
        r"\\section\{|\\begin\{figure\*?\}|\\begin\{table\}|"
        r"\\begin\{equation\}|\\label\{([^}]*)\}")
    current = {"sec": "", "fig": "", "tab": "", "eq": ""}
    for m in pattern.finditer(body):
        token = m.group(0)
        if token.startswith("\\section"):
            sec += 1
            current["sec"] = ROMAN[sec - 1] if sec <= len(ROMAN) else str(sec)
        elif token.startswith("\\begin{figure"):
            fig += 1
            current["fig"] = str(fig)
        elif token.startswith("\\begin{table"):
            tab += 1
            current["tab"] = str(tab)
        elif token.startswith("\\begin{equation"):
            eq += 1
            current["eq"] = str(eq)
        elif m.group(1):
            label = m.group(1)
            kind = label.split(":", 1)[0]
            if kind in current and current[kind]:
                LABELS[label] = current[kind]


def inline(tex: str, cites: list[str]) -> str:
    """문단 하나를 워드에 넣을 평문으로. 굵게/기울임은 표시를 남기고 뒤에서 처리한다."""
    t = tex

    # 수치와 단위
    t = re.sub(r"\\SIrange\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}",
               lambda m: f"{m.group(1)}–{m.group(2)} {_unit(m.group(3))}", t)
    t = re.sub(r"\\SI\{([^}]*)\}\{([^}]*)\}",
               lambda m: f"{m.group(1).replace(chr(92) + 'pm', '±').strip()} {_unit(m.group(2))}", t)
    t = re.sub(r"\\numrange\{([^}]*)\}\{([^}]*)\}", r"\1–\2", t)
    t = re.sub(r"\\num\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\si\{([^}]*)\}", lambda m: _unit(m.group(1)), t)
    t = t.replace("\\,", " ").replace("\\%", "%").replace("~", " ")

    # 인용: 나오는 순서대로 번호를 매긴다
    def cite(m):
        nums = []
        for key in [k.strip() for k in m.group(1).split(",")]:
            if key not in cites:
                cites.append(key)
            nums.append(str(cites.index(key) + 1))
        return "[" + ", ".join(nums) + "]"

    t = re.sub(r"\\cite\{([^}]*)\}", cite, t)
    t = re.sub(r"\\(?:eq)?ref\{([^}]*)\}",
               lambda m: LABELS.get(m.group(1), m.group(1).split(":", 1)[-1]), t)

    # 강조: 워드에서 실제 굵게/기울임으로 만들 자리를 표시해 둔다
    t = re.sub(r"\\textbf\{([^{}]*)\}", lambda m: "\x01" + m.group(1) + "\x01", t)
    t = re.sub(r"\\emph\{([^{}]*)\}", lambda m: "\x02" + m.group(1) + "\x02", t)
    t = re.sub(r"\\texttt\{([^{}]*)\}", lambda m: m.group(1), t)
    t = re.sub(r"\\TODO\{([^{}]*)\}", lambda m: "\x03[TODO: " + m.group(1) + "]\x03", t)

    # 수식은 평문으로 (초안 편집용이라 이미지로 만들지 않는다)
    t = re.sub(r"\$([^$]*)\$", lambda m: math_text(m.group(1)), t)

    t = t.replace("R^2", "R²").replace("$", "")
    t = t.replace("``", "\u201c").replace("''", "\u201d")
    t = t.replace("\\&", "&").replace("\\_", "_")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def math_text(tex: str) -> str:
    t = tex
    for src, dst in [("\\hat Q", "Q̂"), ("\\rho", "ρ"), ("\\gamma", "γ"), ("\\sigma", "σ"),
                     ("\\tau_{us}", "τ_us"), ("\\tau", "τ"), ("\\lambda_Q", "λ_Q"),
                     ("\\lambda", "λ"), ("\\beta_{KL}", "β_KL"), ("\\beta", "β"),
                     ("\\pm", "±"), ("\\le", "≤"), ("\\ge", "≥"), ("\\times", "×"),
                     ("\\mathrm", ""), ("\\mathbb{R}", "R"), ("\\in", "∈"),
                     ("\\Delta", "Δ"), ("\\Omega", "Ω"), ("\\varphi", "φ"),
                     ("\\alpha", "α"), ("\\theta", "θ"), ("\\omega", "ω"),
                     ("\\emph", ""), ("\\left", ""), ("\\right", ""), ("\\!", ""),
                     ("\\big", ""), ("\\Big", ""), ("\\;", " "), ("\\,", " "),
                     ("\\approx", "≈"), ("\\qquad", "    "), ("\\quad", "  "),
                     ("\\cdot", "·"), ("\\sim", "~"), ("\\ldots", "…"),
                     ("\\operatorname", ""), ("\\arg\\max", "argmax"),
                     ("\\hat", ""), ("\\mathbf", "")]:
        t = t.replace(src, dst)
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\sum_\{([^{}]*)\}\^\{([^{}]*)\}", r"Σ(\1..\2)", t)
    t = re.sub(r"\\sum", "Σ", t)
    t = re.sub(r"[{}]", "", t)
    t = t.replace("^\\star", "*").replace("\\star", "*")
    return re.sub(r"\s+", " ", t).strip()


def add_runs(par, text: str) -> None:
    """\x01 굵게, \x02 기울임, \x03 TODO(붉게)."""
    tokens = re.split(r"([\x01\x02\x03])", text)
    bold = italic = todo = False
    for tok in tokens:
        if tok == "\x01":
            bold = not bold
        elif tok == "\x02":
            italic = not italic
        elif tok == "\x03":
            todo = not todo
        elif tok:
            run = par.add_run(tok)
            run.font.name = FONT
            run.bold = bold or todo
            run.italic = italic
            if todo:
                run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)


def emit_paragraph(doc, text: str, style: str | None = None,
                   align=None, size: float = BODY_PT, space_after: float = 3.0):
    par = doc.add_paragraph(style=style)
    add_runs(par, text)
    if align is not None:
        par.alignment = align
    par.paragraph_format.space_after = Pt(space_after)
    for run in par.runs:
        run.font.size = Pt(size)
    return par


def emit_figure(doc, block: str, cites: list[str], reserve_mm: float = 0.0) -> None:
    """그림 하나. 통짜(figure*)는 1단 구간으로 빠져나갔다 돌아온다.

    아직 그리지 않은 그림은 tex 의 ``\\figspace`` 만큼 **빈칸을 그대로 비워 둔다** —
    분량을 재려면 자리가 실제로 잡혀 있어야 한다.
    """
    wide = "figure*" in block
    imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", block)
    cap = re.search(r"\\caption\{(.*?)\}\s*\\label", block, re.S)
    section = doc.sections[-1]
    col = column_width_in(section)
    full = section.page_width.inches - 2 * MARGIN_IN

    if wide:
        one_column_break(doc)
    if imgs:
        paths = [HERE / name for name in imgs]
        paths = [p for p in paths if p.exists()]
        if paths:
            span = min(full, WIDE_MAX_IN) if wide else col
            each = (span / len(paths)) - 0.06 if wide else col
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.space_after = Pt(2)
            for path in paths:
                par.add_run().add_picture(str(path), width=Inches(each))
    elif reserve_mm:
        blank = doc.add_paragraph()
        blank.paragraph_format.space_after = Pt(2)
        run = blank.add_run("[FIGURE TO BE DRAWN]")
        run.font.name = FONT
        run.font.size = Pt(CAPTION_PT)
        run.bold = True
        blank.alignment = WD_ALIGN_PARAGRAPH.CENTER
        # 예약한 높이만큼 빈 줄을 둔다 (한 줄 ≈ 4.2 mm at 9.5 pt)
        for _ in range(max(1, int(reserve_mm / 4.2))):
            spacer = doc.add_paragraph()
            spacer.paragraph_format.space_after = Pt(0)
            spacer.add_run(" ").font.size = Pt(CAPTION_PT)
    if cap:
        emit_paragraph(doc, inline(cap.group(1), cites), size=CAPTION_PT,
                       align=WD_ALIGN_PARAGRAPH.LEFT, space_after=8.0)
    if wide:
        two_column_break(doc)


def emit_table(doc, block: str, cites: list[str]) -> None:
    cap = re.search(r"\\caption\{(.*?)\}\s*\\label", block, re.S)
    if cap:
        emit_paragraph(doc, inline(cap.group(1), cites), size=9.0, space_after=3.0)
    body = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", block, re.S)
    if not body:
        return
    rows = []
    for raw in body.group(1).split("\\\\"):
        line = raw
        for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
            line = line.replace(rule, "")
        line = line.strip()
        if not line:
            continue
        rows.append([inline(c, cites) for c in line.split("&")])
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.style = "Table Grid"
    # 표는 **단 폭 안에** 들어가야 한다. 자동 맞춤에 맡기면 워드가 단을 넘겨 잘라 낸다.
    table.autofit = False
    width = Inches((column_width_in(doc.sections[-1]) - 0.05) / ncol)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            target = table.cell(i, j)
            target.width = width
            par = target.paragraphs[0]
            par.paragraph_format.space_after = Pt(0)
            add_runs(par, cell)
            for run in par.runs:
                run.font.size = Pt(7.0)
                run.bold = run.bold or i == 0
    for row in table.rows:              # 셀 폭은 행마다 다시 써 줘야 먹는다
        for j, cell in enumerate(row.cells):
            cell.width = width
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def bib_entries(keys: list[str]) -> list[str]:
    """refs.bib 를 순서대로 평문 항목으로. 서지 스타일이 아니라 초안용 나열이다."""
    text = (HERE / "refs.bib").read_text(encoding="utf-8")
    out = []
    for n, key in enumerate(keys, 1):
        m = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)\n\}", text, re.S)
        if not m:
            out.append(f"[{n}] {key} — NOT FOUND in refs.bib")
            continue
        fields = dict(re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", m.group(1), re.S))
        def clean(raw: str) -> str:
            out = raw
            # LaTeX 악센트. 저자 이름이 Garc\'ia 처럼 새어 나오면 그건 오식이다.
            for src, dst in [("\\\"o", "ö"), ("\\\"u", "ü"), ("\\\"a", "ä"),
                             ("\\'a", "á"), ("\\'e", "é"), ("\\'i", "í"),
                             ("\\'o", "ó"), ("\\'u", "ú"), ("\\'n", "ń"),
                             ("\\~n", "ñ"), ("\\c c", "ç"), ("\\i", "i")]:
                out = out.replace(src, dst)
            out = out.replace("{", "").replace("}", "").replace("\\", "")
            return re.sub(r"\s+", " ", out).strip()
        # 저자가 넷 이상이면 "et al." — IEEE 관례이기도 하고, 초안에서 참고문헌이
        # 페이지 하나를 더 먹는 것을 막는다. 전체 목록은 refs.bib 에 그대로 있다.
        authors = [clean(a) for a in re.split(r"\s+and\s+", fields.get("author", "")) if a.strip()]
        if len(authors) > 3:
            authors = [authors[0], "et al."]
        bits = [", ".join(authors), clean(fields.get("title", ""))]
        venue = fields.get("journal") or fields.get("booktitle") or fields.get("publisher") or ""
        if venue:
            bits.append(clean(venue))
        for f in ("volume", "number", "pages", "year"):
            if fields.get(f):
                bits.append(clean(fields[f]))
        if fields.get("doi"):
            bits.append("doi:" + clean(fields["doi"]))
        if fields.get("note"):
            bits.append("[" + clean(fields["note"]) + "]")
        out.append(f"[{n}] " + ". ".join(b for b in bits if b) + ".")
    return out


def build() -> Path:
    tex = strip_comments((HERE / "main.tex").read_text(encoding="utf-8"))
    body = tex[tex.index("\\begin{document}"):tex.index("\\end{document}")]
    number_labels(body)
    cites: list[str] = []

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_PT)
    normal.paragraph_format.space_after = Pt(3)
    page_geometry(doc.sections[0])

    # 제목과 초록은 tex 와 같이 단을 가로지른다.
    set_columns(doc.sections[0], 1)

    title = re.search(r"\\title\{(.*?)\}\s*\n\s*\n?\\author", tex, re.S)
    if title:
        emit_paragraph(doc, inline(title.group(1), cites),
                       align=WD_ALIGN_PARAGRAPH.CENTER, size=15.0, space_after=4.0)
    emit_paragraph(doc, "Anonymous Authors", align=WD_ALIGN_PARAGRAPH.CENTER,
                   size=11.0, space_after=12.0)

    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if abstract:
        emit_paragraph(doc, "\x01Abstract\x01", size=11.0, space_after=3.0)
        emit_paragraph(doc, inline(abstract.group(1), cites), size=BODY_PT, space_after=8.0)
        body = body[body.index("\\end{abstract}") + len("\\end{abstract}"):]
    two_column_break(doc)

    # 블록 단위로 훑는다: 그림/표 환경과 그 밖의 문단들
    pattern = re.compile(
        r"(\\begin\{figure\*?\}.*?\\end\{figure\*?\}"
        r"|\\begin\{table\}.*?\\end\{table\}"
        r"|\\begin\{itemize\}.*?\\end\{itemize\}"
        r"|\\begin\{enumerate\}.*?\\end\{enumerate\}"
        r"|\\begin\{equation\}.*?\\end\{equation\})", re.S)
    pos = 0
    for m in pattern.finditer(body):
        emit_text(doc, body[pos:m.start()], cites)
        block = m.group(1)
        if block.startswith("\\begin{figure"):
            space = re.search(r"\\figspace\{([0-9.]+)mm\}", block)
            emit_figure(doc, block, cites,
                        reserve_mm=float(space.group(1)) if space else 0.0)
        elif block.startswith("\\begin{table"):
            emit_table(doc, block, cites)
        elif block.startswith("\\begin{equation"):
            inner = block.split("}", 1)[1].rsplit("\\end", 1)[0]
            inner = re.sub(r"\\label\{[^}]*\}", "", inner)
            emit_paragraph(doc, math_text(inner), align=WD_ALIGN_PARAGRAPH.CENTER,
                           size=10.0, space_after=8.0)
        else:
            for item in re.findall(r"\\item\s+(.*?)(?=\\item|\\end\{)", block, re.S):
                emit_paragraph(doc, inline(item, cites),
                               style="List Bullet" if "itemize" in block else "List Number",
                               size=10.0, space_after=3.0)
        pos = m.end()
    emit_text(doc, body[pos:], cites)

    emit_paragraph(doc, "\x01References\x01", size=12.0, space_after=4.0)
    for line in bib_entries(cites):
        emit_paragraph(doc, line, size=7.5, space_after=0.5)

    out = HERE / "ICRA_draft.docx"
    doc.save(out)
    return out


def match_braced(text: str, command: str):
    """``\\command{...}`` 를 중괄호 짝을 세어 잡는다.

    제목 안에 ``$Q_\\mathrm{seg}$`` 처럼 중괄호가 들어 있어서 비탐욕 정규식이
    첫 ``}`` 에서 끊겼다 — 소제목이 두 조각으로 찢어져 나왔다.
    """
    head = "\\" + command + "{"
    if not text.startswith(head):
        return None
    depth, i = 1, len(head)
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth:
        return None
    return text[len(head):i - 1], i


def emit_text(doc, chunk: str, cites: list[str]) -> None:
    chunk = re.sub(r"\\(?:maketitle|balance|centering|bibliographystyle\{[^}]*\}"
                   r"|bibliography\{[^}]*\}|begin\{document\}|figspace\{[^}]*\})", "", chunk)
    chunk = re.sub(r"\\label\{[^}]*\}", "", chunk)
    for para in re.split(r"\n\s*\n", chunk):
        para = para.strip()
        if not para:
            continue
        sec = match_braced(para, "section")
        sub = match_braced(para, "subsection")
        if sec:
            emit_paragraph(doc, "\x01" + inline(sec[0], cites) + "\x01",
                           size=11.0, space_after=3.0)
            rest = para[sec[1]:].strip()
            if rest:
                emit_paragraph(doc, inline(rest, cites))
            continue
        if sub:
            emit_paragraph(doc, "\x02" + inline(sub[0], cites) + "\x02",
                           size=10.0, space_after=2.0)
            rest = para[sub[1]:].strip()
            if rest:
                emit_paragraph(doc, inline(rest, cites))
            continue
        emit_paragraph(doc, inline(para, cites))


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
    sys.exit(0)
