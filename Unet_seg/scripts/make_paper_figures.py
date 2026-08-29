#!/usr/bin/env python3
"""Publication figures for the robotic-ultrasound bladder-segmentation paper.

Two independent figures, each written to a vector PDF and a 600 dpi PNG:

``figure1_acquisition_domain_shift``
    A schematic contrasting the transabdominal acquisition the FR5 probing loop
    targets with the transperineal acquisition PFUS1 was recorded under. It
    exists to make the domain gap legible: different probe path, different field
    of view, different anatomical target.

``figure2_pfus1_segmentation_examples``
    The best, median and worst predictions of ``exp_seed43/best.pt`` over the
    2,953 labelled test+val frames, rendered from the source PNGs and the model's
    own masks -- never from a screenshot.

Typography: Arial/Helvetica are requested first and fall back to Liberation Sans
(Arial metrics) or Nimbus Sans (a Helvetica clone), which is what a Linux box
actually has. ``pdf.fonttype=42`` keeps PDF text as embedded TrueType rather than
Type 3, which several journals reject.

Example:
    python scripts/make_paper_figures.py --output-dir figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

logger = logging.getLogger("figures")

# --- shared style ---------------------------------------------------------
SANS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
INK = "#000000"          # all figure text is solid black
INK_SOFT = "#000000"
LEADER = "#6B6B6B"       # leader lines stay gray; only text is black

# Figure 1 palette. Anatomy carries the saturated, conventional illustration
# colours -- skin, ivory bone, dusky muscle, urine-filled bladder -- while the
# beam stays a restrained technical steel blue, because it is an annotation
# drawn over the anatomy rather than a structure of its own. Keeping those two
# roles in different colour families is what stops the panel reading as a flat
# pastel infographic.
SKIN_FILL, SKIN_EDGE = "#EAB98F", "#A9724A"        # outer skin rind
TISSUE_FILL, TISSUE_EDGE = "#F4DECC", "#C89C7C"    # subcutaneous / soft tissue
FLOOR_FILL, FLOOR_EDGE = "#BC8071", "#8A5142"      # levator plate (muscle)
BLADDER_FILL, BLADDER_EDGE = "#A8C6D2", "#2C5B6E"  # urine-filled lumen, muscular wall
BONE_FILL, BONE_EDGE = "#F0E7D2", "#AF9B78"        # bone is ivory, not gray
RECTUM_COLOR = "#A9705B"
BEAM_EDGE, BEAM_FILL = "#2B4A6B", "#2B4A6B"
PROBE_FILL, PROBE_EDGE = "#FFFFFF", "#2B2B2B"

# Figure 2 palette (as specified)
GT_COLOR = "#16B9D4"
PRED_COLOR = "#E17C32"
TP_COLOR = "#2D6F9F"
FN_COLOR = "#4FA65B"
FP_COLOR = "#D34A4A"
OVERLAY_ALPHA = 0.40


def apply_paper_style() -> None:
    """Set rcParams shared by both figures: white ground, sans text, vector PDF."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": SANS,
            "text.color": INK,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 120,
        }
    )


def save_both(figure, output_dir: Path, stem: str, dpi: int = 600) -> tuple[Path, Path]:
    """Write ``stem.pdf`` (vector) and ``stem.png`` (``dpi``) side by side."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path, png_path = output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    figure.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    import matplotlib.pyplot as plt

    plt.close(figure)
    return pdf_path, png_path


# ===========================================================================
# Figure 1 -- acquisition domain shift
# ===========================================================================
#
# Both panels draw the SAME simplified midsagittal pelvis in the same
# coordinates, so the only differences a reader sees are the ones that matter:
# where the probe sits, which way the beam points, and how much of the bladder
# the field of view contains. Orientation is anterior-left, superior-up.
#
# Anatomy check the figure has to survive: the suprapubic window puts the
# transducer on the lower anterior abdominal wall above a distended bladder and
# angles the beam postero-caudally, while the transperineal window puts it on the
# perineum and angles the beam cranially. The two approach paths are therefore
# opposed in their vertical component, which is what the figure must show.

# Outline control points, walked from the upper abdomen down the anterior wall,
# across the perineum and back up the posterior wall. The two top points sit
# above the axes limit on purpose, so the body is clipped by the panel edge and
# reads as continuing cranially instead of ending in a drawn cut line.
_OUTLINE = [
    (30.0, 106.0), (25.6, 90.0), (21.9, 76.0), (21.2, 63.0), (23.7, 52.0),
    (28.7, 43.4), (35.6, 36.9), (43.0, 32.8),
    (51.5, 29.8), (60.5, 29.2), (68.8, 30.9),
    (76.0, 36.6), (81.5, 47.0), (84.6, 60.0), (85.0, 76.0), (83.8, 90.0),
    (82.5, 106.0),
]
#: Fractions of the outline that face anterior and inferior -- the two surfaces
#: a transducer can be seated on.
_ANTERIOR_SPAN = (0.05, 0.46)
_INFERIOR_SPAN = (0.47, 0.66)


def _catmull_rom(points: Sequence[tuple[float, float]], per_segment: int = 28,
                 closed: bool = False) -> np.ndarray:
    """Smooth curve through every control point, with no optional dependency.

    scipy is not installed on the training box, and the previous implementation
    silently fell back to linear interpolation there -- which is why the
    silhouette rendered as a faceted polygon. A Catmull-Rom spline is a few lines
    of numpy, always available, and passes through its control points, which is
    what hand-placed anatomy needs.
    """
    pts = np.asarray(points, dtype=float)
    count = len(pts)
    if count < 3:
        return pts
    indices = np.arange(count)
    out = []
    span = count if closed else count - 1
    for i in range(span):
        p0 = pts[indices[(i - 1) % count]] if closed else pts[max(i - 1, 0)]
        p1 = pts[indices[i % count]]
        p2 = pts[indices[(i + 1) % count]]
        p3 = pts[indices[(i + 2) % count]] if closed else pts[min(i + 2, count - 1)]
        t = np.linspace(0.0, 1.0, per_segment, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    curve = np.vstack(out)
    return curve if closed else np.vstack([curve, pts[-1]])


def _smooth(points: Sequence[tuple[float, float]], samples: int = 0) -> np.ndarray:
    """Smooth an open run of control points."""
    return _catmull_rom(points)


def _body_polygon() -> np.ndarray:
    """Corner-free body outline shared by both panels."""
    return _catmull_rom(_OUTLINE, per_segment=34)


def _outline_span(span: tuple[float, float]) -> np.ndarray:
    """The portion of the outline a probe may be seated on."""
    curve = _body_polygon()
    lo, hi = (int(round(f * len(curve))) for f in span)
    return curve[lo:hi]


def _offset_inward(polygon: np.ndarray, distance: float) -> np.ndarray:
    """Offset a curve by ``distance`` along its inward normal.

    Scaling toward the centroid -- the obvious shortcut -- gives a rind that is
    thick where the outline is far from the centroid and thin where it is close.
    A true normal offset keeps the skin an even thickness the whole way round,
    which is the difference between a drawn body and a stretched one.
    """
    tangents = np.gradient(polygon, axis=0)
    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.where(lengths == 0, 1.0, lengths))
    centroid = polygon.mean(axis=0)
    if np.sum(normals * (centroid - polygon)) < 0:
        normals = -normals
    return polygon + distance * normals


def _surface_pose(curve: np.ndarray, target: tuple[float, float]) -> tuple[np.ndarray, float]:
    """Contact point on ``curve`` nearest ``target``, and the inward normal angle.

    Placing the transducer by solving for the surface rather than by eye is what
    keeps the probe face flush with the skin in both panels; a floating probe is
    the first thing a reader notices.

    Returns:
        ``(contact_xy, inward_normal_degrees)``.
    """
    distances = np.linalg.norm(curve - np.asarray(target, dtype=float), axis=1)
    index = int(np.argmin(distances))
    lo, hi = max(index - 3, 0), min(index + 3, len(curve) - 1)
    tangent = curve[hi] - curve[lo]
    normal = np.array([tangent[1], -tangent[0]], dtype=float)
    centroid = curve.mean(axis=0)
    if np.dot(normal, centroid - curve[index]) < 0:  # point it into the body
        normal = -normal
    return curve[index], float(np.degrees(np.arctan2(normal[1], normal[0])))


def _sector(face: tuple[float, float], axis_deg: float, half_angle_deg: float,
            depth: float, apex_setback: float = 12.0) -> np.ndarray:
    """Curvilinear field of view starting at the transducer face.

    The virtual apex sits ``apex_setback`` behind the face, which is what gives a
    curved-array sector its characteristic near-field arc instead of a triangle
    that appears to originate from a point on the skin.
    """
    axis = np.radians(axis_deg)
    apex = np.array(face, dtype=float) - apex_setback * np.array([np.cos(axis), np.sin(axis)])
    angles = np.radians(np.linspace(axis_deg - half_angle_deg, axis_deg + half_angle_deg, 120))
    inner = apex + apex_setback * np.column_stack([np.cos(angles), np.sin(angles)])
    outer = apex + (apex_setback + depth) * np.column_stack([np.cos(angles), np.sin(angles)])
    return np.vstack([inner, outer[::-1]])


def _probe(ax, face: tuple[float, float], axis_deg: float,
           length: float = 16.0, thickness: float = 6.4) -> None:
    """Transducer sitting outside the skin with its emitting face at ``face``."""
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import Affine2D

    # Local frame: +y is the beam direction, so the body occupies negative y and
    # the face lands exactly on the contact point.
    patch = FancyBboxPatch(
        (-length / 2, -thickness), length, thickness,
        boxstyle="round,pad=0,rounding_size=2.6",
        facecolor=PROBE_FILL, edgecolor=PROBE_EDGE, linewidth=1.1, zorder=6,
        joinstyle="round", capstyle="round",
    )
    rot = Affine2D().rotate_deg(axis_deg - 90).translate(*face)
    patch.set_transform(rot + ax.transData)
    ax.add_patch(patch)
    face_line = rot.transform(np.array([[-length / 2 + 2.2, 0.0], [length / 2 - 2.2, 0.0]]))
    ax.plot(face_line[:, 0], face_line[:, 1], color=PROBE_EDGE, linewidth=2.4,
            solid_capstyle="round", zorder=7)


def _label(ax, text: str, xy: tuple[float, float], xytext: tuple[float, float],
           ha: str = "left", size: float = 7.4) -> None:
    """Structure label with a hairline leader."""
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va="center", fontsize=size, color=INK,
                arrowprops=dict(arrowstyle="-", color=LEADER, linewidth=0.6,
                                shrinkA=1.0, shrinkB=1.5),
                annotation_clip=False, zorder=8)


def _orientation_key(ax, x: float = 95.0, y: float = 93.0, arm: float = 3.8) -> None:
    """Small anterior/posterior, superior/inferior cross in light gray."""
    ax.plot([x, x], [y - arm, y + arm], color="#BBBBBB", linewidth=0.6, zorder=8)
    ax.plot([x - arm, x + arm], [y, y], color="#BBBBBB", linewidth=0.6, zorder=8)
    for text, dx, dy, ha, va in (("S", 0, arm + 0.8, "center", "bottom"),
                                 ("I", 0, -arm - 0.8, "center", "top"),
                                 ("A", -arm - 0.8, 0, "right", "center"),
                                 ("P", arm + 0.8, 0, "left", "center")):
        ax.text(x + dx, y + dy, text, fontsize=5.8, color=INK, ha=ha, va=va, zorder=8)


def _draw_anatomy(ax, bladder: tuple[float, float, float, float]) -> tuple:
    """Soft tissue, pelvic floor, pubis, rectum, bladder, urethra.

    Args:
        bladder: ``(cx, cy, width, height)``. This is the one structure that
            legitimately differs between panels: a transabdominal bladder scan is
            performed on a distended bladder, a pelvic-floor study is not.

    Returns:
        ``(body_patch, bladder_neck_xy, urethra_midpoint_xy)``. The body patch is
        handed back so the beam can be clipped to it -- ultrasound does not
        propagate through air, and a sector spilling past the skin is the kind of
        error a reviewer of an imaging paper notices immediately.
    """
    from matplotlib.patches import Ellipse, Polygon

    outline = _body_polygon()
    body = Polygon(outline, closed=True, facecolor=SKIN_FILL,
                   edgecolor=SKIN_EDGE, linewidth=1.1, zorder=1,
                   joinstyle="round")
    ax.add_patch(body)
    # Inner soft tissue, leaving the outer band reading as skin.
    ax.add_patch(Polygon(_offset_inward(outline, 2.6), closed=True, facecolor=TISSUE_FILL,
                         edgecolor=TISSUE_EDGE, linewidth=0.7, zorder=1.2,
                         joinstyle="round"))

    # Levator plate and rectum are drawn as round-capped strokes rather than
    # polygon bands: a polygon cannot round its own end caps, and blunt ends read
    # as a brick laid across the pelvis. Each is stroked twice, a darker wider
    # pass under a lighter narrower one, which gives an outlined band whose ends
    # are properly finished.
    floor = _smooth([(45.0, 37.4), (52.5, 34.3), (61.0, 33.7), (68.4, 35.4)])
    for width, color, order in ((9.0, FLOOR_EDGE, 2.0), (7.2, FLOOR_FILL, 2.1)):
        ax.plot(floor[:, 0], floor[:, 1], color=color, linewidth=width,
                solid_capstyle="round", solid_joinstyle="round", zorder=order)

    rectum = _smooth([(67.4, 55.0), (68.8, 48.0), (67.2, 42.6), (65.2, 39.0)])
    for width, color, order in ((3.6, "#8A5142", 2.4), (2.4, RECTUM_COLOR, 2.5)):
        ax.plot(rectum[:, 0], rectum[:, 1], color=color, linewidth=width,
                solid_capstyle="round", zorder=order)

    ax.add_patch(Ellipse((38.8, 40.5), 9.6, 12.0, angle=28, facecolor=BONE_FILL,
                         edgecolor=BONE_EDGE, linewidth=0.9, zorder=4))

    cx, cy, w, h = bladder
    ax.add_patch(Ellipse((cx, cy), w, h, facecolor=BLADDER_FILL, edgecolor=BLADDER_EDGE,
                         linewidth=1.7, zorder=3))

    # The urethra leaves the neck and runs antero-caudally through the levator
    # plate. It stops AT the plate rather than at the skin: drawn to the skin, a
    # spline through four points overshoots the perineal surface, and a urethra
    # hanging outside the body is worse than one that ends where it is crossed.
    neck = (cx - w * 0.22, cy - h * 0.48)
    urethra = _smooth([neck, (neck[0] - 1.2, neck[1] - 3.2),
                       (neck[0] - 2.0, neck[1] - 6.2), (46.6, 35.2)], 60)
    for width, color, order in ((3.4, BLADDER_EDGE, 4.8), (2.0, BLADDER_FILL, 4.9)):
        line, = ax.plot(urethra[:, 0], urethra[:, 1], color=color, linewidth=width,
                        solid_capstyle="round", zorder=order)
        line.set_clip_path(body)
    return body, neck, tuple(urethra[len(urethra) // 2])


def _beam(ax, face, axis_deg, half_angle, depth, clip_to=None):
    """Draw the sector twice: a wash inside, a restrained outline on top.

    Args:
        clip_to: Patch the sector is clipped to, normally the soft-tissue body,
            so the field of view stops at the skin.
    """
    from matplotlib.patches import Polygon

    polygon = _sector(face, axis_deg, half_angle, depth)
    wash = Polygon(polygon, closed=True, facecolor=BEAM_FILL, alpha=0.10,
                   edgecolor="none", zorder=4.6)
    outline = Polygon(polygon, closed=True, facecolor="none", edgecolor=BEAM_EDGE,
                      linewidth=1.0, zorder=5.6, joinstyle="round")
    for patch in (wash, outline):
        ax.add_patch(patch)
        if clip_to is not None:
            patch.set_clip_path(clip_to)
    return polygon


def build_figure1():
    """Two-panel acquisition schematic. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.28, 4.15))
    figure.subplots_adjust(left=0.004, right=0.996, top=0.918, bottom=0.175, wspace=0.03)

    for ax in (left, right):
        ax.set_xlim(1, 101)
        ax.set_ylim(16, 100)
        ax.set_aspect("equal")
        ax.axis("off")

    anterior = _outline_span(_ANTERIOR_SPAN)
    inferior = _outline_span(_INFERIOR_SPAN)

    # ---------------- Panel A -- transabdominal, distended bladder ----------
    body_a, _, _ = _draw_anatomy(left, bladder=(55.5, 62.0, 41.0, 29.0))
    face_a, normal_a = _surface_pose(anterior, (22.5, 79.0))
    axis_a = normal_a - 30.0            # angled postero-caudally over the pubis
    # Depth reaches the far bladder wall measured from the virtual apex, not from
    # the face: the apex sits `apex_setback` outside the skin, so a depth chosen
    # by eye leaves the sector cutting through the organ it is meant to contain.
    _beam(left, face_a, axis_a, 30.0, 58.0, clip_to=body_a)
    _probe(left, face_a, axis_a)

    left.set_title("Transabdominal ultrasound (HoLEP-relevant)",
                   fontsize=8.8, fontweight="bold", color=INK, pad=8)
    left.text(0.0, 1.045, "A", transform=left.transAxes, fontsize=11.5,
              fontweight="bold", color=INK, va="bottom", ha="left")
    _orientation_key(left)

    _label(left, "Abdominal wall", (24.6, 88.0), (37.0, 95.0))
    _label(left, "Ultrasound beam", (46.0, 79.0), (62.0, 92.0))
    _label(left, "Urinary bladder", (66.0, 62.0), (90.0, 72.0))
    _label(left, "Pubic bone", (35.4, 44.6), (12.0, 50.0), ha="right")

    # ---------------- Panel B -- transperineal, pelvic-floor plane ----------
    body_b, _, _ = _draw_anatomy(right, bladder=(55.5, 57.5, 30.0, 21.0))
    face_b, normal_b = _surface_pose(inferior, (52.0, 29.4))
    axis_b = normal_b + 4.0             # cranially, very slightly posterior
    # Narrower and shallower than panel A on purpose: the pelvic-floor study
    # images the neck, urethra and levator plate, not the bladder dome.
    _beam(right, face_b, axis_b, 25.0, 26.0, clip_to=body_b)
    _probe(right, face_b, axis_b)

    right.set_title("Transperineal ultrasound (PFUS1)",
                    fontsize=8.8, fontweight="bold", color=INK, pad=8)
    right.text(0.0, 1.045, "B", transform=right.transAxes, fontsize=11.5,
               fontweight="bold", color=INK, va="bottom", ha="left")
    _orientation_key(right)

    neck_b = (55.5 - 30.0 * 0.22, 57.5 - 21.0 * 0.48)
    _label(right, "Perineal probe", face_b, (24.0, 20.5), ha="right")
    _label(right, "Ultrasound beam", (68.5, 47.0), (92.0, 42.0))
    _label(right, "Bladder neck", neck_b, (90.0, 58.0))
    _label(right, "Pubic bone", (35.4, 44.6), (12.0, 50.0), ha="right")
    _label(right, "Urethra", (45.6, 39.0), (16.0, 32.0), ha="right")
    _label(right, "Pelvic floor", (60.0, 34.6), (92.0, 27.5))

    # ---------------- panel annotations ----------------
    for ax, text in ((left, "Broad transabdominal view of the distended bladder"),
                     (right, "Midsagittal pelvic-floor view during Valsalva")):
        ax.text(0.5, -0.035, text, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.6, color=INK_SOFT)

    # ---------------- bracket spanning both panels ----------------
    x0, x1, y = 0.08, 0.92, 0.092
    figure.add_artist(plt.Line2D([x0, x1], [y, y], color=INK_SOFT, linewidth=0.8,
                                 transform=figure.transFigure))
    for x in (x0, x1):
        figure.add_artist(plt.Line2D([x, x], [y, y + 0.016], color=INK_SOFT,
                                     linewidth=0.8, transform=figure.transFigure))
    figure.text(0.5, y - 0.014, "Acquisition and anatomical domain shift", ha="center",
                va="top", fontsize=8.2, fontweight="bold", color=INK)
    figure.text(0.5, y - 0.068,
                "Different probe path, field of view, anatomical target, and image appearance",
                ha="center", va="top", fontsize=7.8, color=INK_SOFT)
    return figure


# ===========================================================================
# Figure 2 -- PFUS1 segmentation examples
# ===========================================================================

#: Cases and the metrics reported for them. Dice and the pixel counts are the
#: evaluation-protocol values, computed at the 256 x 256 network resolution by
#: scripts/evaluate.py -- the same numbers a results table quotes. The panels
#: are rendered from the native-resolution frame and the model's native-grid
#: mask, which is where the visual quality is; that rendering scores within
#: 0.005 Dice of the protocol value (0.9593 / 0.8502 / 0.3639).
FIG2_CASES = [
    ("Best", "P041", 1, 0.9597),
    ("Median", "P012", 59, 0.8480),
    ("Worst", "P000", 70, 0.3588),
]


def extract_figure2_data(output: Path, config_path: str = "configs/exp_seed43.yaml",
                         checkpoint: str = "checkpoints/exp_seed43/best.pt",
                         dataset_root: str = "/home/rosotauser/datasets/pfus") -> Path:
    """Re-run the model on the three cases and cache frames, masks and metrics.

    The figure is rendered from the model's own output, never from a screenshot,
    so this step is the figure's provenance. Two predictions are produced per
    case: one on the 256 x 256 grid, which reproduces the evaluation protocol and
    is asserted against the recorded metrics, and one restored to the native
    frame grid, which is what the panels display.

    Raises:
        RuntimeError: If a reproduced 256 x 256 Dice disagrees with the value in
            :data:`FIG2_CASES` -- a silent mismatch would put a number in the
            figure that the checkpoint does not actually produce.
    """
    from rus_perception.utils.config import load_config
    from rus_perception.control.features import FeatureExtractionConfig
    from rus_perception.inference.predictor import Predictor, PredictorConfig
    from rus_perception.data.io import load_grayscale, load_mask, resize_image, resize_mask
    from rus_perception.metrics.spatial import compute_frame_metrics

    config = load_config(config_path)
    feature_config = FeatureExtractionConfig.from_dict(
        {"postprocess": config.section("postprocess"), **config.section("control")}
    )

    def predictor(restore: bool) -> "Predictor":
        return Predictor.from_checkpoint(
            checkpoint, model_config=config.section("model"),
            predictor_config=PredictorConfig(
                input_size=(256, 256),
                intensity_normalization=str(config.get("data.intensity_normalization")),
                device=str(config.get("train.device", "auto")).replace("auto", "cuda"),
                restore_original_size=restore,
            ),
            feature_config=feature_config,
        )

    at_256, at_native = predictor(False), predictor(True)
    store, provenance = {}, []

    for role, patient, frame, expected_dice in FIG2_CASES:
        image_path = f"raw/data/{patient}/frame_{frame:03d}.png"
        mask_path = f"masks/{patient}/frame_{frame:03d}.png"
        image = load_grayscale(f"{dataset_root}/{image_path}")
        truth = load_mask(f"{dataset_root}/{mask_path}")

        state = at_256.predict_control_state(resize_image(image, (256, 256)))
        metrics = compute_frame_metrics(
            prediction=np.asarray(state.binary_mask, dtype=np.float32),
            target=resize_mask(truth, (256, 256)), frame_id=f"{patient}_{frame}",
            patient_id=patient, sequence_id="seq0", frame_index=frame,
            compute_hd95=False, spacing=1.0,
        ).to_dict()
        if abs(float(metrics["dice"]) - expected_dice) > 5e-4:
            raise RuntimeError(
                f"{patient} frame {frame}: reproduced Dice {float(metrics['dice']):.4f} "
                f"does not match the recorded {expected_dice:.4f}. Refusing to render a "
                "figure labelled with a number the checkpoint no longer produces."
            )

        native = at_native.predict_control_state(image)
        store[f"{role}_img"] = image.astype(np.float32)
        store[f"{role}_gt"] = truth.astype(np.uint8)
        store[f"{role}_pred"] = np.asarray(native.binary_mask, dtype=np.uint8)
        provenance.append({"role": role, "image": image_path, "mask": mask_path,
                           "dice_protocol_256": round(float(metrics["dice"]), 4)})

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **store)
    logger.info("Figure 2 source: %s", json.dumps(provenance))
    return output


def _mask_to_path(mask: np.ndarray):
    """Vector :class:`~matplotlib.path.Path` of a binary mask, holes included.

    Filling the error map with vector polygons rather than an RGBA raster is what
    keeps the PDF overlay resolution-independent. ``RETR_CCOMP`` gives outer
    boundaries and holes as separate contours, which map onto a single Path with
    even-odd style subpaths.

    Returns:
        The Path, or ``None`` when the mask is empty.
    """
    import cv2
    from matplotlib.path import Path as MplPath

    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    vertices, codes = [], []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        vertices.extend([tuple(points[0])] + [tuple(p) for p in points[1:]] + [tuple(points[0])])
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1) + [MplPath.CLOSEPOLY])
    if not vertices:
        return None
    return MplPath(np.asarray(vertices, dtype=float), codes)


def _draw_contours(ax, mask: np.ndarray, color: str, linewidth: float,
                   dashes: Optional[tuple] = None) -> None:
    """Stroke every boundary of a binary mask as a vector polyline."""
    import cv2

    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        closed = np.vstack([points, points[:1]])
        line, = ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=linewidth,
                        solid_joinstyle="round", zorder=4)
        if dashes:
            line.set_dashes(list(dashes))


def _centre_on_canvas(array: np.ndarray, shape: tuple[int, int], fill: float = 0.0) -> np.ndarray:
    """Centre ``array`` in a ``shape`` canvas, so every row shares one pixel scale.

    The three frames were acquired at different matrix sizes. Rescaling them to a
    common panel size would silently zoom one case relative to another; padding
    keeps one pixel equal to one pixel in all nine panels, and the padding is
    black, which is what lies outside an ultrasound sector anyway.
    """
    canvas = np.full(shape, fill, dtype=array.dtype)
    top = (shape[0] - array.shape[0]) // 2
    left = (shape[1] - array.shape[1]) // 2
    canvas[top:top + array.shape[0], left:left + array.shape[1]] = array
    return canvas


def build_figure2(data_path: Path):
    """Best / median / worst prediction panels. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch, Patch

    apply_paper_style()
    store = np.load(data_path)

    heights = [store[f"{role}_img"].shape[0] for role, *_ in FIG2_CASES]
    widths = [store[f"{role}_img"].shape[1] for role, *_ in FIG2_CASES]
    canvas_shape = (max(heights), max(widths))

    figure = plt.figure(figsize=(7.4, 5.05))
    grid = GridSpec(3, 4, figure=figure, width_ratios=[0.60, 1, 1, 1],
                    wspace=0.035, hspace=0.045,
                    left=0.004, right=0.988, top=0.905, bottom=0.012)

    column_titles = ["Input ultrasound", "Ground truth and prediction", "Error map"]

    for row, (role, patient, frame, dice) in enumerate(FIG2_CASES):
        image = _centre_on_canvas(store[f"{role}_img"], canvas_shape, 0.0)
        gt = _centre_on_canvas(store[f"{role}_gt"], canvas_shape, 0)
        pred = _centre_on_canvas(store[f"{role}_pred"], canvas_shape, 0)

        gt_bool, pred_bool = gt > 0.5, pred > 0.5
        regions = (
            (np.logical_and(gt_bool, pred_bool), TP_COLOR),
            (np.logical_and(gt_bool, ~pred_bool), FN_COLOR),
            (np.logical_and(~gt_bool, pred_bool), FP_COLOR),
        )

        # --- row label ---
        label_ax = figure.add_subplot(grid[row, 0])
        label_ax.axis("off")
        label_ax.text(0.97, 0.60, role, ha="right", va="bottom", fontsize=9.6,
                      fontweight="bold", color=INK, transform=label_ax.transAxes)
        label_ax.text(0.97, 0.52, f"{patient} \u00b7 frame {frame:03d}", ha="right", va="top",
                      fontsize=7.0, color=INK_SOFT, transform=label_ax.transAxes)
        label_ax.text(0.97, 0.35, f"Dice {dice:.3f}", ha="right", va="top",
                      fontsize=7.0, color=INK_SOFT, transform=label_ax.transAxes)

        for column in range(3):
            ax = figure.add_subplot(grid[row, column + 1])
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="antialiased")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if column == 1:
                _draw_contours(ax, gt, GT_COLOR, 1.5)
                _draw_contours(ax, pred, PRED_COLOR, 1.9, dashes=(3.6, 2.2))
            elif column == 2:
                for region, color in regions:
                    path = _mask_to_path(region)
                    if path is not None:
                        ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none",
                                               alpha=OVERLAY_ALPHA, zorder=4))

            if row == 0:
                ax.set_title(column_titles[column], fontsize=8.4, fontweight="bold",
                             color=INK, pad=5)
                if column == 1:
                    ax.legend(
                        handles=[Line2D([], [], color=GT_COLOR, linewidth=1.6, label="Ground truth"),
                                 Line2D([], [], color=PRED_COLOR, linewidth=1.9,
                                        dashes=(3.0, 1.8), label="Prediction")],
                        loc="upper left", fontsize=6.2, labelcolor="white", frameon=False,
                        handlelength=1.9, handletextpad=0.5, borderpad=0.35,
                        labelspacing=0.34,
                    )
                elif column == 2:
                    ax.legend(
                        handles=[Patch(facecolor=TP_COLOR, edgecolor="none", label="True positive"),
                                 Patch(facecolor=FN_COLOR, edgecolor="none", label="False negative"),
                                 Patch(facecolor=FP_COLOR, edgecolor="none", label="False positive")],
                        loc="upper left", fontsize=6.2, labelcolor="white", frameon=False,
                        handlelength=1.1, handleheight=0.9, handletextpad=0.5,
                        borderpad=0.35, labelspacing=0.34,
                    )
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--only", choices=["1", "2"], default=None)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--figure2-data", default=None,
                        help="npz produced by the Figure 2 extraction step")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(args.output_dir)
    if args.only in (None, "1"):
        paths = save_both(build_figure1(), out, "figure1_acquisition_domain_shift", args.dpi)
        logger.info("Figure 1 -> %s , %s", *paths)
    if args.only in (None, "2"):
        data_path = Path(args.figure2_data or (out / "_figure2_source.npz"))
        if not data_path.is_file():
            extract_figure2_data(data_path)
        paths = save_both(build_figure2(data_path), out,
                          "figure2_pfus1_segmentation_examples", args.dpi)
        logger.info("Figure 2 -> %s , %s", *paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
