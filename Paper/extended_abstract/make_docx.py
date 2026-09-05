#!/usr/bin/env python3
"""확장 초록 -> Word(.docx). 2 단 학회 지면, 그림은 단을 가로지른다.

숫자는 본문에 적혀 있지만 **모두 force_hold_validation 산출물과 대조된 값** 이고,
그 대응은 FIGURE_TABLE_SOURCE_MAP.md 가 들고 있다.
"""
from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

from docx import Document                                        # noqa: E402
from docx.enum.section import WD_SECTION                         # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT                   # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH                    # noqa: E402
from docx.oxml import OxmlElement                                # noqa: E402
from docx.oxml.ns import qn                                      # noqa: E402
from docx.shared import Cm, Pt, RGBColor                         # noqa: E402

OUT = os.path.join(_HERE, "HoLEP_Force_Control_Extended_Abstract.docx")
FIGS = os.path.join(_HERE, "figures")
SONO_FIG_PATH = os.path.join(_HERE, "figures", "fig3_sonologger.png")

INK = RGBColor(0x1A, 0x1F, 0x26)
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
SUB = RGBColor(0x5A, 0x60, 0x68)
HEAD_FILL = "E8EEF4"
RULE = "C7CCD1"
SANS = "Calibri"

COL_W = 8.15                     # 단 폭 [cm] — A4, 여백 1.8, 단간 0.7
FULL_W = 17.4                    # 단을 가로지르는 폭 [cm]


# ── 저수준 서식 ───────────────────────────────────────────────────────────
def _cols(section, n: int, space_cm: float = 0.7) -> None:
    """단 수를 지정한다. python-docx 에 API 가 없어 XML 로 직접 쓴다."""
    cols = section._sectPr.xpath("./w:cols")[0]
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(int(space_cm * 567)))
    cols.set(qn("w:equalWidth"), "1")


def _shade(cell, hexcolor: str) -> None:
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hexcolor)
    cell._tc.get_or_add_tcPr().append(el)


def _borders(table) -> None:
    """가로선만. 세로 격자는 좁은 단에서 숫자를 더 읽기 어렵게 만든다."""
    borders = OxmlElement("w:tblBorders")
    for edge, sz, col in (("top", "8", "1A1F26"), ("bottom", "8", "1A1F26"),
                          ("insideH", "4", RULE)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), sz)
        el.set(qn("w:color"), col)
        borders.append(el)
    for edge in ("left", "right", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        borders.append(el)
    table._tbl.tblPr.append(borders)


def _tight_cells(table, top=8, bottom=8, side=68) -> None:
    """셀 안쪽 여백 [twips]. 기본값은 좁은 지면에서 행마다 낭비가 쌓인다."""
    mar = OxmlElement("w:tblCellMar")
    for edge, val in (("top", top), ("bottom", bottom),
                      ("left", side), ("right", side)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    table._tbl.tblPr.append(mar)


def _fixed(table, widths_cm) -> None:
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    table._tbl.tblPr.append(layout)
    grid = table._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        table._tbl.remove(grid)
    grid = OxmlElement("w:tblGrid")
    for cm in widths_cm:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(int(cm * 567)))
        grid.append(col)
    table._tbl.insert(list(table._tbl).index(table._tbl.tblPr) + 1, grid)
    for row in table.rows:
        rowpr = row._tr.get_or_add_trPr()
        rowpr.append(OxmlElement("w:cantSplit"))
        for i, cell in enumerate(row.cells):
            cell.width = Cm(widths_cm[i])


def _run(par, text, *, bold=False, italic=False, size=9.0, color=INK, font=SANS):
    r = par.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.size = Pt(size)
    r.font.name = font
    r.font.color.rgb = color
    return r


def _rich(par, text, *, size=9.0, color=INK):
    """``**굵게**`` 와 ``` `코드` ``` 만 처리한다."""
    for piece in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not piece:
            continue
        if piece.startswith("**"):
            _run(par, piece.strip("*"), bold=True, size=size, color=color)
        elif piece.startswith("`"):
            _run(par, piece.strip("`"), size=size - 0.5, color=SUB, font="Consolas")
        else:
            _run(par, piece, size=size, color=color)


def heading(doc, text, *, size=10.5, before=8, after=3):
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(before)
    par.paragraph_format.space_after = Pt(after)
    par.paragraph_format.keep_with_next = True
    _run(par, text, bold=True, size=size, color=NAVY)
    return par


def body(doc, text, *, size=8.6, after=4, justify=True):
    par = doc.add_paragraph()
    par.paragraph_format.space_after = Pt(after)
    par.paragraph_format.line_spacing = 1.02
    if justify:
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _rich(par, text, size=size)
    return par


def bullet(doc, text, *, size=8.8):
    par = doc.add_paragraph(style="List Bullet")
    par.paragraph_format.space_after = Pt(2)
    par.paragraph_format.line_spacing = 1.05
    _rich(par, text, size=size)
    return par


def caption(doc, text, *, width=FULL_W, before=3, after=8):
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(before)
    par.paragraph_format.space_after = Pt(after)
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _rich(par, text, size=8.0, color=SUB)
    return par


def figure(doc, path, width_cm):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(2)
    par.paragraph_format.space_after = Pt(1)
    par.paragraph_format.keep_with_next = True
    par.add_run().add_picture(path, width=Cm(width_cm))
    return par


def table(doc, header, rows, widths, *, align_right=()):
    t = doc.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _borders(t)
    for i, name in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        par = cell.paragraphs[0]
        par.paragraph_format.space_after = Pt(1)
        par.alignment = (WD_ALIGN_PARAGRAPH.RIGHT if i in align_right
                         else WD_ALIGN_PARAGRAPH.LEFT)
        _run(par, name, bold=True, size=7.8, color=NAVY)
        _shade(cell, HEAD_FILL)
    for row in rows:
        cells = t.add_row().cells
        for i, text in enumerate(row):
            par = cells[i].paragraphs[0]
            par.paragraph_format.space_after = Pt(1)
            par.alignment = (WD_ALIGN_PARAGRAPH.RIGHT if i in align_right
                             else WD_ALIGN_PARAGRAPH.LEFT)
            bold = text.startswith("**") and text.endswith("**")
            _run(par, text.strip("*"), bold=bold, size=8.0)
    _fixed(t, widths)
    _tight_cells(t)
    return t


def span_section(doc, n_cols: int):
    """단 수가 다른 새 구역을 이어 붙인다 (연속 구역 나누기)."""
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    _cols(section, n_cols)
    return section


# ── 본문 ─────────────────────────────────────────────────────────────────
TITLE = ("Contact-Force Regulation for Robotic Transabdominal Ultrasound in "
         "HoLEP Morcellation: Bench Validation on a Volume-Changing Phantom")

AUTHORS = "[TO BE CONFIRMED: author list, affiliations, corresponding author]"

CLINICAL = [
    "During the morcellation phase of holmium laser enucleation of the prostate "
    "(HoLEP), a rotating morcellator works inside a distended bladder. Keeping "
    "the bladder adequately filled and knowing where the morcellator sits "
    "relative to the bladder wall are what protect the mucosa, and transabdominal "
    "ultrasound is the bedside means of seeing both. In the workflow that "
    "motivates this work, an assistant holds a suprapubic probe and keeps the "
    "bladder lumen in view for as long as morcellation lasts.",

    "That task is narrow but unrelenting. It is not a diagnostic sweep: the "
    "region is already known, and the work is to hold one plane while the "
    "abdominal wall moves with irrigation inflow and outflow, so that the lumen "
    "does not leave the image and acoustic coupling does not lapse. Holding it "
    "occupies a pair of hands for the duration, and image steadiness tracks the "
    "assistant's attention. **A robot that maintains probe contact could return "
    "those hands to the operation without touching surgical judgement**, which is "
    "the narrow question this work addresses. What follows is a benchtop system "
    "and a phantom experiment, intended to establish whether the contact-force "
    "channel is steady and bounded enough to carry forward — not to demonstrate "
    "clinical use. No patient data were collected and no part of morcellation is "
    "automated.",
]

PLATFORM = [
    "A six-degree-of-freedom collaborative arm (FR5) carries the probe on its "
    "flange behind a six-axis force–torque sensor (PaXini PX6D) read at 1 kHz "
    "over a dedicated USB link. The probe mounted for these experiments is a "
    "**3D-scanned replica of the clinical GE 4C-RS convex probe**, reproducing "
    "its contact geometry and mass; it does not image, and no ultrasound was "
    "acquired during the force experiments reported here.",

    "Wrenches are compensated for gravity and tool payload and referred to the "
    "probe contact point, and the controller acts on a single scalar — the "
    "contact-force magnitude ‖F‖ = √(Fx²+Fy²+Fz²) in the probe frame, signed by "
    "the normal component. Using the magnitude rather than the normal force "
    "matters at the forces of interest: a probe that lands slightly tilted puts "
    "much of the contact into shear, and a normal-force threshold would then "
    "read the contact as absent.",

    "An admittance loop runs at 100 Hz on the probe penetration axis alone, with "
    "servo output at 125 Hz. Once contact is detected the arm drops to contact "
    "velocity limits of 10 mm/s and 0.2 rad/s. A supervisor holds a soft warning "
    "level of 4.5 N and a hard limit of 5.0 N at which the arm is driven back, "
    "together with command watchdogs that retreat the probe if operator input or "
    "robot state stops arriving. **This supervisor and the force loop contain no "
    "learned or image-derived component**; the force channel is deterministic and "
    "runs whether or not anything is looking at an image.",

    "The measurement chain itself was checked against an external reference. The "
    "probe was pressed onto an electronic scale of 1 g resolution at 19 poses "
    "spanning 0.9–9.8 N, and the compensated probe normal force compared with the "
    "scale reading. Unloaded multi-pose residuals gave the gravity-compensation "
    "error separately.",
]

VALIDATION = [
    "Force holding was measured on an abdominal phantom whose internal volume is "
    "set by a 500 mL syringe [TO BE CONFIRMED: phantom make and model]. Injecting "
    "and withdrawing fluid lifts and lowers the surface under the probe, which is "
    "the bench analogue of an abdominal wall moving with irrigation.",

    "Contact probing was engaged at a fixed 0.4 N entry threshold in every run, so "
    "each setpoint received the same approach transient and the bands remain "
    "comparable. Four setpoints were held for 60 s each: 0.5, 2.0, 3.0 and 4.0 N. "
    "The deadband was ±0.15 N at 0.5 N and ±0.50 N at the others. Because the "
    "regulator commands zero velocity inside the deadband, the force settles where "
    "it first enters the band rather than at the setpoint; that entry force is "
    "reported below as the **operating point**, and tracking is measured against "
    "it. The first 3 s after entry are excluded as approach transient.",

    "Disturbance runs repeated 150 mL injections and withdrawals at the same "
    "setpoints, each syringe action timestamped at its onset. A separate run drove "
    "the force up deliberately with rapid injections to probe the upper end. Every "
    "wrench sample the controller acted on was recorded at sensor rate and no "
    "signal was filtered before analysis.",
]

RESULTS = [
    "Across the four setpoints the held force had a standard deviation of "
    "0.022–0.065 N and departed from its operating point by no more than 0.10 N. "
    "Standard deviation is independent of band width and is therefore the quantity "
    "that compares setpoints; it shows no trend across the tested range. In "
    "relative terms the loop is tighter at higher forces — 1.0–1.4 % of setpoint "
    "above 2 N against 13.0 % at 0.5 N — because the noise floor is set by the "
    "sensor rather than by the controller.",

    "Twenty-two excursions above the deadband were recorded across two disturbance "
    "runs and one deliberate hard-press run. All of them returned inside the band; "
    "the slowest took 1.39 s, with per-run medians of 0.19–1.09 s. The highest "
    "force seen anywhere in the campaign was 4.41 N, and no run reached the 4.5 N "
    "warning level, leaving 0.59 N of margin to the hard limit at the worst point.",

    "Flange pose logged beside force shows how the disturbance was absorbed: over "
    "one disturbance run the probe travelled 2.29 mm along its own penetration "
    "axis while the force stayed within 3.0 ± 0.5 N. **The force was held by "
    "moving the arm, not by tolerating the load** — a distinction that a "
    "force-only record cannot make.",

    "Against the electronic scale the compensated force tracked the reference "
    "with slope 1.019 (95 % CI 0.975–1.064) and R² 0.993, at a mean bias of "
    "−0.293 N and RMSE 0.366 N — 3.7 % of the 9.85 N full scale — with "
    "Bland–Altman limits of agreement of −0.734 to +0.148 N. The reference is "
    "itself known only to 0.182 N, a term dominated by gravity compensation "
    "rather than by the scale (0.003 N). **Absolute accuracy is therefore an "
    "order coarser than the short-term stability above.** The loop repeats a "
    "force far more finely than it knows that force in absolute terms, and a "
    "setpoint chosen for tissue should be read against the accuracy figure.",
]

SONOLOGGER = [
    "Sonologger is a separate acquisition platform for the data a future "
    "image-guided or learned controller would need. A printed clamp shell fits "
    "the clinical GE 4C-RS probe and carries a BNO085 inertial measurement unit; "
    "the scanner's display output is captured over HDMI, the IMU streams over USB "
    "serial, and the host timestamps both on one clock.",

    "In bench sessions Sonologger recorded 776 ultrasound frames over 96.9 s "
    "(8.00 fps) alongside 24,471 inertial records (252 Hz) on that shared clock. "
    "Against robot forward-kinematics ground truth during teleoperated "
    "probing-like motion, latency-corrected angular RMS error was 1.14°, 0.73° and "
    "1.95° for tilt x, tilt y and axial rotation, with 0.23°/min heading drift. "
    "**Sonologger records relative orientation from a stationary reference — not "
    "translation, and not absolute SE(3) pose.** No policy has been trained on "
    "its recordings; learning from demonstration is future work, not a result.",
]

LIMITS = (
    "**Limitations.** Phantom and bench only, with no clinical data [TO BE "
    "CONFIRMED: IRB status]; a water-filled phantom is more linear and less "
    "viscoelastic than an abdomen, so these recovery times are a lower bound. "
    "Peak force stayed 0.59 N below the limit, leaving forced retreat untested at "
    "its threshold. Deadband width differed at 0.5 N, so standard deviation is "
    "the comparable quantity. No image-guided orientation control, segmentation "
    "or learned policy is part of the validated system."
)


T1_HEAD = ("Component / parameter", "Specification", "Role")
T1_ROWS = [
    ("Robot arm", "FR5, 6-DoF collaborative", "Probe positioning"),
    ("Mounted probe", "3D-scanned GE 4C-RS replica", "Contact geometry (non-imaging)"),
    ("Force sensor", "PX6D, six-axis, 1 kHz", "Contact-force measurement"),
    ("Controlled quantity", "contact-force magnitude ‖F‖", "Gravity- and payload-compensated, referred to the contact point"),
    ("Admittance controller", "100 Hz, penetration axis", "Contact-force regulation (servo output 125 Hz)"),
    ("Linear velocity limit", "10 mm/s", "Contact safety"),
    ("Angular velocity limit", "0.2 rad/s", "Contact safety"),
    ("Warning level", "4.5 N", "Supervisory warning, no advance"),
    ("Force limit", "5.0 N", "Forced retreat threshold"),
]

T2_HEAD = ("Metric", "Result", "Interpretation")
T2_ROWS = [
    ("Tested force range", "0.5–4.0 N", "Bench validation range"),
    ("Held-force SD", "0.022–0.065 N", "Short-term force stability"),
    ("Deviation from operating point", "≤ 0.10 N", "Operating-point stability"),
    ("Band excursions recovered", "22 of 22", "All returned inside the deadband"),
    ("Slowest recovery", "1.39 s", "After the tested disturbance"),
    ("Peak force observed", "4.41 N", "Below the 4.5 N warning level"),
    ("Probe-axis travel", "2.29 mm", "Compensation for surface displacement"),
    ("Scale-referenced bias", "−0.293 N", "Robot reads low against the scale (n = 19)"),
    ("Scale-referenced RMSE", "0.366 N (3.7 % FS)", "Absolute accuracy of the force channel"),
    ("Sonologger frames / IMU records", "776 / 24,471", "Synchronized bench acquisition"),
    ("Angular RMS error", "1.14°, 0.73°, 1.95°", "Tilt x, tilt y, axial rotation (Sonologger)"),
]

CAP1 = ("**Figure 1.** System architecture and clinical role. **Solid dark-blue: "
        "the pathway validated in this work.** **Light-grey dashed: future "
        "components — bladder-lumen segmentation, image-guided orientation "
        "control and any learned policy — none of which is present in the robot "
        "control stack.**")

CAP2 = ("**Figure 2.** Force holding on the volume-changing phantom, from the "
        "recorded wrench logs. **(a)** 60 s holds at four setpoints; shaded band "
        "is the commanded deadband, dashed line the setpoint. **(b)** Hard-press "
        "run at 3.0 N: nine rapid injections, each returning inside the band; "
        "peak 4.41 N against the 5.0 N limit.")

CAP_SCALE = ("**Figure 3.** Compensated probe normal force against an electronic "
             "scale, n = 19 over 0.9–9.8 N. Shaded band is the combined reference "
             "uncertainty (0.182 N), which the gravity-compensation residual "
             "dominates.")

CAP4 = ("**Figure 4.** Sonologger acquisition — two paths, one host clock. "
        "Latency-corrected angular RMS error against robot forward kinematics was "
        "1.14°, 0.73° and 1.95°. **The IMU gives relative orientation — not "
        "translation and not absolute SE(3) pose.**")


def build() -> None:
    doc = Document()
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = Cm(1.8)
    s.top_margin = s.bottom_margin = Cm(1.8)
    doc.styles["Normal"].font.name = SANS
    doc.styles["Normal"].font.size = Pt(9)

    # ── 표제 (단 하나) ────────────────────────────────────────────────────
    _cols(s, 1)
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(4)
    _run(par, TITLE, bold=True, size=15, color=NAVY)
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(9)
    _run(par, AUTHORS, size=8.5, color=SUB, italic=True)

    # ── 2 단: 임상 동기 + 플랫폼 + 표 1 ──────────────────────────────────
    span_section(doc, 2)
    heading(doc, "1  Clinical Motivation", before=0)
    for p in CLINICAL:
        body(doc, p)
    heading(doc, "2  Robotic Ultrasound Platform")
    for p in PLATFORM:
        body(doc, p)

    # ── 전폭: 표 1 + 그림 1 ──────────────────────────────────────────────
    span_section(doc, 1)
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(2)
    cap.paragraph_format.space_after = Pt(2)
    cap.paragraph_format.keep_with_next = True
    _rich(cap, "**Table 1.** Robot system and safety-control specifications. "
          "Every entry is read from the running control stack or its "
          "configuration.", size=8.0, color=SUB)
    table(doc, T1_HEAD, T1_ROWS, (4.3, 5.6, 7.5))
    figure(doc, os.path.join(FIGS, "fig1_architecture.png"), 14.8)
    caption(doc, CAP1)

    # ── 2 단: 팬텀 검증 + 결과 + 표 2 ────────────────────────────────────
    span_section(doc, 2)
    heading(doc, "3  Phantom Force-Holding Validation", before=0)
    for p in VALIDATION:
        body(doc, p)
    heading(doc, "4  Results")
    for i, p in enumerate(RESULTS):
        body(doc, p)
        if i == len(RESULTS) - 1:
            # 저울 그림은 단 폭으로 둔다 — 전폭으로 빼면 지면을 세 배로 먹는다.
            figure(doc, os.path.join(FIGS, "fig3_scale_validation.png"), COL_W)
            caption(doc, CAP_SCALE, after=5)

    # ── 전폭: 그림 2 + 표 2 ──────────────────────────────────────────────
    span_section(doc, 1)
    figure(doc, os.path.join(FIGS, "fig2_force_validation.png"), 15.0)
    caption(doc, CAP2, after=6)
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(2)
    cap.paragraph_format.space_after = Pt(2)
    cap.paragraph_format.keep_with_next = True
    _rich(cap, "**Table 2.** Quantitative validation summary. All entries are "
          "bench measurements on the phantom; none is a clinical-safety claim.",
          size=8.0, color=SUB)
    table(doc, T2_HEAD, T2_ROWS, (5.4, 5.0, 7.0))
    figure(doc, SONO_FIG_PATH, 9.7)
    caption(doc, CAP4, after=4)

    # ── 2 단: Sonologger + 결론 ──────────────────────────────────────────
    span_section(doc, 2)
    heading(doc, "5  Sonologger and Future Learning Data")
    for p in SONOLOGGER:
        body(doc, p)
    heading(doc, "6  Conclusion and Limitations")
    body(doc, "On a volume-changing phantom the platform held 0.5–4.0 N with a "
              "standard deviation of 0.022–0.065 N, stayed within 0.10 N of its "
              "operating point, and recovered within 1.39 s after every tested "
              "disturbance, peak force remaining 0.59 N below the limit. The "
              "force channel achieving this is deterministic and independent of "
              "image and learning.")
    body(doc, LIMITS)

    doc.save(OUT)
    print(os.path.relpath(OUT, _ROOT))



if __name__ == "__main__":
    build()
