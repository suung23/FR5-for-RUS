"""Figures for training curves and quality-score trust validation.

Design decisions, so they are not re-litigated per figure:

**Palette.** The two domain colours are the ones ``docs/architecture.html``
already uses across the project: warm for the force domain (``Q_raw``), cool for
the image domain (``Q_seg``). They are separated in *lightness* as well as hue
(L* ~42 vs ~27) so the distinction survives greyscale printing. Status colours
(good / bad) are reserved and never reused as a series colour. The 5-slot
categorical ramp used by the loss chart is a separate, validated palette --
adjacent pairs clear CVD ΔE 18 and normal-vision ΔE 19, with green never
adjacent to orange or red.

**Light only, deliberately.** ``docs/export_figures.py`` states the rule for this
project: exported raster figures are always light, because a printed or pasted
figure must not depend on the reader's OS theme. These follow it.

**English labels.** Axis and legend text is English while the surrounding
documentation is Korean. Matplotlib needs a font *file* for Korean glyphs, and
the training machine that regenerates these will not reliably have one -- a
figure that renders as boxes on the box where it is produced is worse than one
in plain technical English. The Korean explanation lives in the caption.

**No dual axes anywhere.** Where two quantities have different units they get
stacked panels sharing an x-axis, never a second y-scale.

Raises:
    ImportError: On import, if matplotlib is unavailable. It is an optional
        extra (``pip install -e ".[viz]"``); nothing on the real-time path needs
        it, so it is not a core dependency.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "Plotting requires matplotlib. Install it with: pip install -e \".[viz]\""
    ) from exc

from ..metrics.trust import ForceResponseReport, TrustReport

__all__ = [
    "PALETTE",
    "CATEGORICAL",
    "apply_style",
    "plot_loss_decomposition",
    "plot_validation_curves",
    "plot_run_comparison",
    "plot_temporal_diagnostics",
    "plot_optimization_health",
    "plot_quality_vs_accuracy",
    "plot_risk_coverage",
    "plot_reliability",
    "plot_roc",
    "plot_component_attribution",
    "plot_reason_codes",
    "plot_per_patient",
    "plot_force_response",
    "plot_latency",
    "save",
]

#: Domain and status colours, shared with the project's SVG figures.
PALETTE: dict[str, str] = {
    "force": "#8A5A00",      # Q_raw / force domain -- warm, L* 42
    "force_fill": "#E8E1CE",
    "image": "#17455C",      # Q_seg / image domain -- cool, L* 27
    "image_fill": "#DDE6EB",
    "good": "#1F5130",
    "bad": "#8C1D1D",
    "bad_fill": "#F7EDED",
    "ink": "#111111",
    "ink_2": "#3D3D3B",
    "ink_3": "#6E6E6B",
    "rule": "#BFBFBC",
    "rule_2": "#E2E2DF",
    "surface": "#FFFFFF",
    "surface_2": "#F1F0ED",
}

#: Fixed-order categorical ramp. Assign by position; never cycle, never generate
#: a sixth hue. Validated: adjacent CVD ΔE >= 18, normal-vision ΔE >= 19.
CATEGORICAL: tuple[str, ...] = ("#C46A00", "#1B6E9E", "#2F8A3C", "#7B52B5", "#B22222")

_LINE = 1.5
_THIN = 0.8
_MARKER = 6.0


def apply_style() -> None:
    """Set the shared rcParams: recessive grid and axes, thin marks, no clutter."""
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "savefig.facecolor": PALETTE["surface"],
            "font.family": ["DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.labelcolor": PALETTE["ink_2"],
            "axes.edgecolor": PALETTE["rule"],
            "axes.linewidth": _THIN,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["rule_2"],
            "grid.linewidth": 0.6,
            "xtick.color": PALETTE["ink_3"],
            "ytick.color": PALETTE["ink_3"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": _LINE,
            "lines.markersize": _MARKER,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def save(figure: "Figure", path: str | Path) -> Path:
    """Write a figure and close it, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def _series(history: Sequence[Mapping[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """``(epochs, values)`` for one history key, skipping records that lack it."""
    epochs, values = [], []
    for index, record in enumerate(history):
        if key not in record:
            continue
        value = record[key]
        if value is None or not math.isfinite(float(value)):
            continue
        epochs.append(float(record.get("epoch", index + 1)))
        values.append(float(value))
    return epochs, values


def _label_line_end(ax: "matplotlib.axes.Axes", x, y, text: str, color: str) -> None:
    """Direct-label the last point of a line -- identity without a legend lookup."""
    if not x:
        return
    ax.annotate(
        text,
        xy=(x[-1], y[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=color,
        fontweight="medium",
        annotation_clip=False,
    )


def _note(ax: "matplotlib.axes.Axes", text: str, loc: str = "lower right") -> None:
    """A small stat block in a corner, in ink -- never in a series colour."""
    positions = {
        "lower right": (0.98, 0.03, "right", "bottom"),
        "lower left": (0.02, 0.03, "left", "bottom"),
        "upper left": (0.02, 0.97, "left", "top"),
        "upper right": (0.98, 0.97, "right", "top"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        color=PALETTE["ink_2"],
        family="monospace",
        linespacing=1.5,
    )


# ===========================================================================
# A. training curves -- from history.json
# ===========================================================================


def plot_loss_decomposition(history: Sequence[Mapping[str, Any]]) -> "Figure":
    """Every loss term separately, spatial and temporal in their own panels.

    Two panels rather than one axes with five lines: the temporal terms are
    scaled by a warm-up/ramp factor that the spatial ones are not, so overlaying
    them invites reading a crossing as meaningful when it is a schedule artefact.
    The ramp is shaded so the reader can see when the temporal term was even on.
    """
    apply_style()
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.2, 5.4), sharex=True, gridspec_kw={"hspace": 0.28}
    )

    spatial = [
        ("loss/bce", "BCE", CATEGORICAL[0]),
        ("loss/dice", "Dice", CATEGORICAL[1]),
        ("loss/jaccard", "Jaccard", CATEGORICAL[2]),
        ("loss/spatial_total", "spatial total", PALETTE["ink_2"]),
    ]
    for key, label, color in spatial:
        x, y = _series(history, key)
        if not x:
            continue
        style = "--" if key == "loss/spatial_total" else "-"
        top.plot(x, y, style, color=color, label=label)
        _label_line_end(top, x, y, label, color)
    top.set_title("Spatial loss terms")
    top.set_ylabel("loss")
    if top.lines:
        top.legend(loc="upper right", ncols=2)

    temporal = [
        ("loss/temporal_pixel", "temporal pixel", CATEGORICAL[3]),
        ("loss/temporal_control", "temporal control", CATEGORICAL[4]),
    ]
    drawn = False
    for key, label, color in temporal:
        x, y = _series(history, key)
        if not x:
            continue
        drawn = True
        bottom.plot(x, y, "-", color=color, label=label)
        _label_line_end(bottom, x, y, label, color)

    ramp_x, ramp_y = _series(history, "temporal/weight_factor")
    if ramp_x:
        twin_scale = bottom.get_ylim()[1] if drawn else 1.0
        bottom.fill_between(
            ramp_x,
            0,
            [v * twin_scale for v in ramp_y],
            color=PALETTE["rule_2"],
            alpha=0.55,
            linewidth=0,
            zorder=0,
            label="warm-up / ramp",
        )
    bottom.set_title("Temporal loss terms" + ("" if drawn else " -- disabled in this run"))
    bottom.set_ylabel("loss")
    bottom.set_xlabel("epoch")
    if bottom.lines or ramp_x:
        bottom.legend(loc="upper right", ncols=2)

    return figure


def plot_validation_curves(history: Sequence[Mapping[str, Any]]) -> "Figure":
    """Validation accuracy, stability and the selection score that picks the model.

    All three are bounded in ``[0, 1]``, so they share one axis honestly. The
    chosen epoch is marked: model selection never uses training loss, and seeing
    *which* epoch won next to the curves is how a silent selection bug surfaces.
    """
    apply_style()
    figure, ax = plt.subplots(figsize=(7.2, 4.0))

    for key, label, color, style in (
        ("val/dice", "val Dice", PALETTE["image"], "-"),
        ("val/temporal_iou", "val temporal IoU", PALETTE["force"], "-"),
        ("val/selection_score", "selection score", PALETTE["ink"], "--"),
    ):
        x, y = _series(history, key)
        if not x:
            continue
        ax.plot(x, y, style, color=color, label=label)
        _label_line_end(ax, x, y, label, color)

    x, y = _series(history, "val/selection_score")
    if x:
        best = int(np.argmax(y))
        ax.axvline(x[best], color=PALETTE["good"], linewidth=_THIN, linestyle=":")
        ax.plot([x[best]], [y[best]], "o", color=PALETTE["good"], zorder=5)
        ax.annotate(
            f"best  epoch {x[best]:.0f}\nscore {y[best]:.4f}",
            xy=(x[best], y[best]),
            xytext=(6, -18),
            textcoords="offset points",
            fontsize=8,
            color=PALETTE["good"],
        )

    ax.set_title("Validation -- accuracy, stability, and what selected the model")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    return figure


def _label_ends_decollided(
    ax: "matplotlib.axes.Axes",
    entries: Sequence[tuple[float, str, str]],
    min_gap: float = 0.062,
) -> None:
    """Direct-label several lines at the right edge without letting labels overlap.

    :func:`_label_line_end` anchors a label to its own last point, which is right
    until two lines finish close together and the labels land on top of each
    other. This variant places every label in the same pass: it converts the end
    values to axes fractions, pushes any pair closer than ``min_gap`` apart, and
    annotates at the adjusted heights. The label may then sit slightly off its
    line, which is the cheaper error -- an unreadable label carries no identity
    at all.

    Args:
        ax: Axes to annotate; its y-limits must already be final.
        entries: ``(y_value, text, colour)`` per line, in any order.
        min_gap: Minimum separation in axes fractions.
    """
    if not entries:
        return
    low, high = ax.get_ylim()
    span = (high - low) or 1.0
    placed = sorted(
        ((float(y) - low) / span, text, color) for y, text, color in entries
    )
    for index in range(1, len(placed)):
        previous, current = placed[index - 1][0], placed[index][0]
        if current - previous < min_gap:
            placed[index] = (previous + min_gap, placed[index][1], placed[index][2])
    for fraction, text, color in placed:
        ax.annotate(
            text,
            xy=(1.015, min(max(fraction, 0.0), 1.0)),
            xycoords="axes fraction",
            va="center",
            fontsize=8,
            color=color,
            annotation_clip=False,
        )


def plot_run_comparison(
    runs: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> "Figure":
    """Compare several runs' histories: what generalised, and what was memorised.

    Three stacked panels sharing the epoch axis. They are stacked rather than
    overlaid because validation Dice, training loss and their gap have different
    units and ranges; a second y-scale would make the comparison a matter of
    where the axes happened to be pinned.

    The panels answer three different questions:

    ``val Dice``
        What the run is actually worth. Read the *shape*, not only the peak: a
        curve that is flat from the first epoch says the schedule is not what
        limits the score.
    ``train loss``
        How completely the run fitted its own training set. On a log axis,
        because it falls by more than an order of magnitude.
    ``val loss - train loss``
        The generalisation gap, which is the memorisation the top panel hides.
        A run whose val Dice is flat while this climbs is not improving; it is
        memorising.

    Epochs are plotted 1-based. ``history.json`` stores a 0-based ``epoch``
    field, and reporting "epoch 0" for the first epoch invites off-by-one
    mistakes when these figures sit next to prose.

    Args:
        runs: ``(label, history)`` pairs. Colours are taken from
            :data:`CATEGORICAL` by position, so a run keeps its colour when
            another is added -- never cycled, never generated.

    Returns:
        The figure, for :func:`save`. Per-run summary statistics are deliberately
        not drawn inside the panels; they belong in the caption or a table, where
        they cannot collide with the curves.

    Raises:
        ValueError: If ``runs`` is empty or has more entries than
            :data:`CATEGORICAL` has slots.
    """
    if not runs:
        raise ValueError("plot_run_comparison needs at least one (label, history) pair.")
    if len(runs) > len(CATEGORICAL):
        raise ValueError(
            f"{len(runs)} runs exceed the {len(CATEGORICAL)}-slot categorical ramp. "
            "Facet into small multiples rather than generating a new hue."
        )

    apply_style()
    figure, (top, mid, bottom) = plt.subplots(
        3, 1, figsize=(8.4, 8.6), sharex=True,
        gridspec_kw={"hspace": 0.30, "height_ratios": [1.25, 1, 1]},
    )

    ends: dict[Any, list[tuple[float, str, str]]] = {top: [], mid: [], bottom: []}
    handles: list[Any] = []

    for index, (label, history) in enumerate(runs):
        color = CATEGORICAL[index]

        epochs, dice = _series(history, "val/dice")
        if epochs:
            x = [e + 1 for e in epochs]
            line, = top.plot(x, dice, "-", color=color, label=label)
            handles.append(line)
            best = int(np.argmax(dice))
            top.plot([x[best]], [dice[best]], "o", color=color, zorder=5,
                     markeredgecolor=PALETTE["surface"], markeredgewidth=1.5)
            ends[top].append((dice[-1], label, color))

        epochs_loss, train = _series(history, "loss/total")
        if epochs_loss:
            mid.plot([e + 1 for e in epochs_loss], train, "-", color=color)
            ends[mid].append((train[-1], label, color))

        epochs_val, val_loss = _series(history, "val/loss")
        if epochs_loss and epochs_val:
            span = min(len(train), len(val_loss))
            gap = [val_loss[i] - train[i] for i in range(span)]
            bottom.plot([e + 1 for e in epochs_val[:span]], gap, "-", color=color)
            ends[bottom].append((gap[-1], label, color))

    top.set_title("What generalised -- validation Dice   (o marks the selected epoch)")
    top.set_ylabel("val Dice")

    mid.set_title("What was memorised -- training loss")
    mid.set_ylabel("train loss")
    mid.set_yscale("log")

    bottom.set_title("The gap -- val loss minus train loss")
    bottom.set_ylabel("val - train loss")
    bottom.set_xlabel("epoch")
    bottom.axhline(0.0, color=PALETTE["rule"], linewidth=_THIN, linestyle=":")

    for ax in (top, mid, bottom):
        # Room at the right for the direct labels, which live outside the axes.
        ax.set_xlim(left=0.5)
    figure.subplots_adjust(right=0.74)
    for ax, entries in ends.items():
        _label_ends_decollided(ax, entries)

    if len(handles) > 1:
        figure.legend(
            handles=handles, loc="lower center", ncols=min(len(handles), 3),
            bbox_to_anchor=(0.42, -0.015), frameon=False,
        )
    return figure


def plot_temporal_diagnostics(history: Sequence[Mapping[str, Any]]) -> "Figure":
    """Whether the optical flow was usable, and how many pairs were thrown away.

    A collapsing reliable-pixel ratio means the temporal loss quietly stopped
    supervising anything -- the run still trains, the curves still look fine, and
    the temporal term contributes nothing. This is the panel that catches it.
    """
    apply_style()
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"hspace": 0.3, "height_ratios": [2, 1]}
    )

    for key, label, color in (
        ("temporal/reliable_pixel_ratio", "reliable pixels", PALETTE["image"]),
        ("temporal/weight_factor", "ramp weight", PALETTE["ink_3"]),
    ):
        x, y = _series(history, key)
        if not x:
            continue
        top.plot(x, y, "-", color=color, label=label)
        _label_line_end(top, x, y, label, color)
    top.axhline(0.10, color=PALETTE["bad"], linewidth=_THIN, linestyle=":")
    top.annotate(
        "min_reliable_ratio",
        xy=(0.01, 0.10),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=7.5,
        color=PALETTE["bad"],
    )
    top.set_ylim(0, 1)
    top.set_ylabel("fraction")
    top.set_title("Optical-flow health")
    if top.lines:
        top.legend(loc="lower right")

    x, y = _series(history, "temporal/skipped_pairs")
    if x:
        bottom.bar(x, y, color=PALETTE["rule"], width=0.7, linewidth=0)
    bottom.set_ylabel("skipped pairs")
    bottom.set_xlabel("epoch")
    return figure


def plot_optimization_health(history: Sequence[Mapping[str, Any]]) -> "Figure":
    """Gradient norm and learning rate, on separate log panels.

    Different units, so two panels rather than a second y-axis. Spikes in the
    gradient norm that coincide with a loss plateau are the signature of the
    non-finite batches the trainer skips silently.
    """
    apply_style()
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw={"hspace": 0.3}
    )

    x, y = _series(history, "train/grad_norm")
    if x:
        top.plot(x, y, "-", color=PALETTE["force"])
        top.set_yscale("log")
    top.set_ylabel("grad norm")
    top.set_title("Optimisation health")

    x, y = _series(history, "train/lr")
    if x:
        bottom.plot(x, y, "-", color=PALETTE["ink_2"])
        bottom.set_yscale("log")
    bottom.set_ylabel("learning rate")
    bottom.set_xlabel("epoch")

    x, y = _series(history, "train/nonfinite_batches")
    if x and any(v > 0 for v in y):
        for xi, yi in zip(x, y):
            if yi > 0:
                top.axvline(xi, color=PALETTE["bad"], linewidth=_THIN, alpha=0.5)
        _note(top, "vertical marks = epochs with\nnon-finite batches skipped", "upper left")
    return figure


# ===========================================================================
# B. trust validation -- from TrustReport
# ===========================================================================


def plot_quality_vs_accuracy(
    report: TrustReport,
    quality: Sequence[float],
    accuracy: Sequence[float],
    valid: Optional[Sequence[bool]] = None,
    gate_threshold: Optional[float] = None,
) -> "Figure":
    """**The headline figure.** Every frame as a point, split into four quadrants.

    The upper-left region -- high quality, low accuracy -- is the only dangerous
    one: the perception layer vouched for a mask that was wrong, so the
    controller acted with no warning. Its point count is the number the whole
    validation exists to drive to zero. Everything else is either correct or
    merely conservative, and conservatism costs throughput, not safety.
    """
    apply_style()
    figure, ax = plt.subplots(figsize=(6.0, 5.6))

    x = np.asarray(quality, dtype=float)
    y = np.asarray(accuracy, dtype=float)
    floor = report.accuracy_floor
    cut = gate_threshold if gate_threshold is not None else report.operating.threshold

    if cut is not None:
        ax.add_patch(
            Rectangle(
                (cut, 0.0), 1.0 - cut, floor,
                facecolor=PALETTE["bad_fill"], edgecolor="none", zorder=0,
            )
        )
        ax.axvline(cut, color=PALETTE["ink_3"], linewidth=_THIN, linestyle="--")
    ax.axhline(floor, color=PALETTE["ink_3"], linewidth=_THIN, linestyle="--")

    accepted = np.asarray(valid, dtype=bool) if valid is not None else np.ones_like(x, dtype=bool)
    good = y >= floor
    for mask, color, label in (
        (accepted & good, PALETTE["good"], "trusted, good"),
        (accepted & ~good, PALETTE["bad"], "TRUSTED, BAD"),
        (~accepted & good, PALETTE["ink_3"], "rejected, good"),
        (~accepted & ~good, PALETTE["rule"], "rejected, bad"),
    ):
        if not mask.any():
            continue
        ax.scatter(
            x[mask], y[mask],
            s=14, c=color, alpha=0.55, linewidths=0.4,
            edgecolors=PALETTE["surface"], label=f"{label}  n={int(mask.sum())}", zorder=3,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{report.name}  (quality score)")
    ax.set_ylabel("Dice against ground truth")
    ax.set_title(f"Does {report.name} know when it is wrong?")

    rho = report.agreement.spearman
    lines = [
        f"rho   {rho:.3f}" if rho is not None else "rho   n/a",
        f"AUROC {report.detection.auroc:.3f}" if report.detection.auroc is not None else "AUROC n/a",
    ]
    bad_rate = report.gate.trusted_bad_rate
    if bad_rate is not None:
        lines.append(f"bad   {100.0 * bad_rate:.1f}%")
    _note(ax, "\n".join(lines), "lower right")
    ax.legend(loc="upper left", markerscale=1.4)
    return figure


def plot_risk_coverage(report: TrustReport) -> "Figure":
    """How much accuracy is bought by declining to act, and at what coverage.

    Read it right-to-left: at coverage 1.0 the controller acts on everything and
    inherits the model's raw error rate; moving left it acts less often and is
    wrong less often. The marked point is the largest coverage that still meets
    the safety budget -- the operating point the supervisor should be built to.
    """
    apply_style()
    figure, ax = plt.subplots(figsize=(6.4, 4.4))

    points = report.risk.points
    if points:
        coverage = [p.coverage for p in points]
        bad = [p.bad_rate for p in points]
        ax.plot(coverage, bad, "-", color=PALETTE["image"])
        ax.fill_between(coverage, 0, bad, color=PALETTE["image_fill"], alpha=0.6, linewidth=0)

    target = report.operating.target_bad_rate
    ax.axhline(target, color=PALETTE["bad"], linewidth=_THIN, linestyle=":")
    ax.annotate(
        f"safety budget  {100.0 * target:.0f}%",
        xy=(0.02, target),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8,
        color=PALETTE["bad"],
    )

    if report.operating.feasible:
        ax.plot(
            [report.operating.coverage], [report.operating.achieved_bad_rate],
            "o", color=PALETTE["good"], zorder=5,
        )
        ax.annotate(
            f"operating point\ncoverage {100.0 * report.operating.coverage:.0f}%"
            f"   {report.name} >= {report.operating.threshold:.3f}",
            xy=(report.operating.coverage, report.operating.achieved_bad_rate),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=8,
            color=PALETTE["good"],
        )
    else:
        _note(ax, "NO threshold meets\nthe safety budget", "upper left")

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("coverage -- fraction of frames acted on")
    ax.set_ylabel(f"fraction below Dice {report.accuracy_floor:.2f}")
    ax.set_title("Selective risk: what caution buys")
    if report.risk.aurc is not None:
        _note(ax, f"AURC {report.risk.aurc:.3f}", "lower right")
    return figure


def plot_reliability(report: TrustReport) -> "Figure":
    """Binned quality against the Dice those bins actually delivered.

    The diagonal is where the score would read directly as expected Dice. Q was
    never built to be a probability, so distance from it is informative rather
    than damning -- but the *sequence* must rise. A bin that delivers less than
    the bin below it means the score is not merely mis-scaled but misleading, and
    no rescaling repairs that.
    """
    apply_style()
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(5.6, 5.4), sharex=True,
        gridspec_kw={"hspace": 0.12, "height_ratios": [3, 1]},
    )

    bins = report.calibration.bins
    if bins:
        qx = [b.mean_quality for b in bins]
        ay = [b.mean_accuracy for b in bins]
        err = [b.std_accuracy for b in bins]
        top.plot([0, 1], [0, 1], "--", color=PALETTE["rule"], linewidth=_THIN, zorder=1)
        top.errorbar(
            qx, ay, yerr=err, fmt="o-", color=PALETTE["image"],
            ecolor=PALETTE["rule"], elinewidth=1.0, capsize=3, zorder=3,
        )
        bottom.bar([b.mean_quality for b in bins], [b.n for b in bins],
                   width=0.045, color=PALETTE["rule"], linewidth=0)

    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.set_ylabel("mean Dice delivered")
    top.set_title("Reliability -- does a higher score deliver a better mask?")
    bottom.set_ylabel("frames")
    bottom.set_xlabel(f"{report.name} bin mean")

    lines = []
    if report.calibration.ece is not None:
        lines.append(f"ECE      {report.calibration.ece:.3f}")
    if report.calibration.monotone_fraction is not None:
        lines.append(f"monotone {100.0 * report.calibration.monotone_fraction:.0f}%")
    if lines:
        _note(top, "\n".join(lines), "upper left")
    return figure


def plot_roc(report: TrustReport, gate_point: bool = True) -> "Figure":
    """The score's discriminative power, with the shipped gate marked on it.

    Separating the two matters: a curve well above the diagonal with the gate
    sitting far below it means the score is fine and the *thresholds* are
    mis-set -- a config change. A flat curve means no threshold can help, and
    the weights are what need work.
    """
    apply_style()
    figure, ax = plt.subplots(figsize=(5.2, 5.0))

    ax.plot([0, 1], [0, 1], "--", color=PALETTE["rule"], linewidth=_THIN)
    roc = report.detection.roc
    if roc:
        ax.plot([p.x for p in roc], [p.y for p in roc], "-", color=PALETTE["image"])
        ax.fill_between([p.x for p in roc], 0, [p.y for p in roc],
                        color=PALETTE["image_fill"], alpha=0.5, linewidth=0)

    if gate_point:
        gate = report.gate
        if gate.recall is not None and gate.specificity is not None:
            ax.plot([1.0 - gate.specificity], [gate.recall], "D",
                    color=PALETTE["bad"], markersize=7, zorder=5)
            ax.annotate(
                "shipped gate\n(valid_for_control)",
                xy=(1.0 - gate.specificity, gate.recall),
                xytext=(10, -20), textcoords="offset points",
                fontsize=8, color=PALETTE["bad"],
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false-positive rate -- bad frames trusted")
    ax.set_ylabel("true-positive rate -- good frames used")
    ax.set_title(f"{report.name} as a detector of Dice >= {report.accuracy_floor:.2f}")
    if report.detection.auroc is not None:
        _note(ax, f"AUROC {report.detection.auroc:.3f}", "lower right")
    return figure


def plot_component_attribution(report: TrustReport) -> "Figure":
    """What each sub-score contributes, measured -- the evidence for the weights.

    Bars are the change in the aggregate's rank agreement when that component is
    removed. A bar to the left of zero means the aggregate gets *better* without
    the component: its weight belongs at zero, not merely lower. Dots are each
    component's own correlation with accuracy, which can be strong even where the
    bar is small if another component already carries the same information.
    """
    apply_style()
    components = [c for c in report.components if c.delta_spearman is not None]
    if not components:
        figure, ax = plt.subplots(figsize=(6.4, 2.4))
        ax.text(0.5, 0.5, "no component attribution available",
                ha="center", va="center", color=PALETTE["ink_3"], transform=ax.transAxes)
        ax.set_axis_off()
        return figure

    components = sorted(components, key=lambda c: c.delta_spearman)
    names = [c.name for c in components]
    deltas = [c.delta_spearman for c in components]
    solos = [c.spearman if c.spearman is not None else 0.0 for c in components]
    y = np.arange(len(components))

    figure, ax = plt.subplots(figsize=(7.0, 0.42 * len(components) + 2.0))
    colors = [PALETTE["bad"] if d < 0 else PALETTE["image"] for d in deltas]
    ax.barh(y, deltas, height=0.55, color=colors, linewidth=0, zorder=3)
    ax.scatter(solos, y, s=34, facecolors=PALETTE["surface"],
               edgecolors=PALETTE["ink_2"], linewidths=1.1, zorder=4, label="own rho vs Dice")
    ax.axvline(0, color=PALETTE["ink_3"], linewidth=_THIN)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{c.name}   w={c.weight:g}" for c in components], fontsize=8, family="monospace"
    )
    ax.set_xlabel("bar: drop in aggregate rho when removed      dot: own rho vs Dice")
    ax.set_title("Which sub-scores earn their weight?")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _note(ax, "bar left of zero =\nremoving it improves Q", "upper left")
    return figure


def plot_reason_codes(report: TrustReport) -> "Figure":
    """Per reason code: the Dice of the frames it fires on versus the rest.

    Each code maps to a distinct supervisor recovery action, so a code whose two
    markers sit on top of each other is not a harmless extra check -- it is a
    state transition taken for frames that were no worse than average.
    """
    apply_style()
    stats = report.reasons
    if not stats:
        figure, ax = plt.subplots(figsize=(6.4, 2.4))
        ax.text(0.5, 0.5, "no rejection reasons fired",
                ha="center", va="center", color=PALETTE["ink_3"], transform=ax.transAxes)
        ax.set_axis_off()
        return figure

    stats = sorted(stats, key=lambda s: (s.mean_accuracy_when_fired or 0.0))
    y = np.arange(len(stats))
    figure, ax = plt.subplots(figsize=(7.2, 0.42 * len(stats) + 2.2))

    for index, stat in enumerate(stats):
        fired = stat.mean_accuracy_when_fired or 0.0
        silent = stat.mean_accuracy_when_silent
        if silent is not None:
            ax.plot([fired, silent], [index, index], "-", color=PALETTE["rule"], linewidth=2.0, zorder=2)
            ax.plot([silent], [index], "o", color=PALETTE["ink_3"], markersize=6, zorder=3)
        ax.plot([fired], [index], "o", color=PALETTE["bad"], markersize=7, zorder=4)

    ax.axvline(report.accuracy_floor, color=PALETTE["ink_3"], linewidth=_THIN, linestyle="--")
    ax.annotate(
        f"floor {report.accuracy_floor:.2f}",
        xy=(report.accuracy_floor, len(stats) - 0.4),
        xytext=(4, 0), textcoords="offset points",
        fontsize=8, color=PALETTE["ink_3"],
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{s.reason}   n={s.n_fired}  lift={s.lift:.2f}" if s.lift is not None
         else f"{s.reason}   n={s.n_fired}" for s in stats],
        fontsize=8, family="monospace",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean Dice     red = when the code fired,  grey = when it did not")
    ax.set_title("Does each rejection reason mark genuinely bad frames?")
    ax.grid(axis="y", visible=False)
    return figure


def plot_per_patient(
    accuracy_by_patient: Mapping[str, Sequence[float]],
    accuracy_floor: float = 0.70,
) -> "Figure":
    """Dice per patient, every frame shown.

    Pooling hides the failure mode that matters most for a patient-level claim:
    strong performance on four subjects and collapse on the fifth averages to
    something respectable. Splits in this repository are patient-level for the
    same reason.
    """
    apply_style()
    patients = sorted(accuracy_by_patient)
    figure, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(patients) + 2.0), 4.2))

    rng = np.random.default_rng(0)
    for index, patient in enumerate(patients):
        values = np.asarray(list(accuracy_by_patient[patient]), dtype=float)
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.16, 0.16, values.size)
        below = values < accuracy_floor
        ax.scatter(index + jitter[~below], values[~below], s=10,
                   c=PALETTE["image"], alpha=0.45, linewidths=0, zorder=3)
        ax.scatter(index + jitter[below], values[below], s=10,
                   c=PALETTE["bad"], alpha=0.6, linewidths=0, zorder=4)
        median = float(np.median(values))
        ax.plot([index - 0.3, index + 0.3], [median, median],
                "-", color=PALETTE["ink"], linewidth=2.0, zorder=5)
        ax.annotate(f"n={values.size}", xy=(index, 0.02), ha="center",
                    fontsize=7.5, color=PALETTE["ink_3"])

    ax.axhline(accuracy_floor, color=PALETTE["ink_3"], linewidth=_THIN, linestyle="--")
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels(patients, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice")
    ax.set_title("Per patient -- pooling hides the one that fails")
    ax.grid(axis="x", visible=False)
    return figure


def plot_force_response(report: ForceResponseReport, name: str = "Q_raw") -> "Figure":
    """``Q`` against contact force: the experiment Stage 1 stands or falls on.

    The force search assumes this curve has one optimum it can climb toward. If
    the response is flat, or turns more than once outside the noise band, the
    search is optimising nothing -- so the verdict printed on the figure is the
    finding, not the curve's prettiness. ``F*`` is deliberately the *smallest*
    force statistically indistinguishable from the peak, not the peak itself.
    """
    apply_style()
    figure, ax = plt.subplots(figsize=(6.6, 4.4))

    measured = [level for level in report.levels if level.mean is not None]
    missing = [level for level in report.levels if level.mean is None]

    if measured:
        forces = [level.force for level in measured]
        means = [level.mean for level in measured]
        sems = [level.sem or 0.0 for level in measured]
        peak = max(means)
        ax.axhspan(peak - report.epsilon, peak, color=PALETTE["force_fill"],
                   alpha=0.6, linewidth=0, zorder=0)
        ax.annotate(
            f"within eps={report.epsilon:g} of the peak",
            xy=(0.02, peak - report.epsilon / 2),
            xycoords=("axes fraction", "data"),
            fontsize=7.5, color=PALETTE["force"], va="center",
        )
        ax.errorbar(forces, means, yerr=sems, fmt="o-", color=PALETTE["force"],
                    ecolor=PALETTE["rule"], elinewidth=1.0, capsize=3, zorder=3)

    for level in missing:
        ax.axvline(level.force, color=PALETTE["bad"], linewidth=_THIN, linestyle=":", alpha=0.7)
        ax.annotate("measurement\nfailure", xy=(level.force, 0.02),
                    xycoords=("data", "axes fraction"), ha="center",
                    fontsize=7, color=PALETTE["bad"])

    if report.f_star is not None:
        ax.axvline(report.f_star, color=PALETTE["good"], linewidth=1.2)
        ax.annotate(f"F* = {report.f_star:g} N", xy=(report.f_star, 0.94),
                    xycoords=("data", "axes fraction"), xytext=(5, 0),
                    textcoords="offset points", fontsize=8.5,
                    color=PALETTE["good"], fontweight="bold")
    if report.argmax_force is not None and report.argmax_force != report.f_star:
        ax.axvline(report.argmax_force, color=PALETTE["ink_3"], linewidth=_THIN, linestyle="--")
        ax.annotate("argmax", xy=(report.argmax_force, 0.86),
                    xycoords=("data", "axes fraction"), xytext=(5, 0),
                    textcoords="offset points", fontsize=8, color=PALETTE["ink_3"])

    verdict = "UNIMODAL" if report.is_unimodal else "NOT UNIMODAL"
    lines = [f"verdict   {verdict}"]
    if report.spearman_with_force is not None:
        lines.append(f"rho vs F  {report.spearman_with_force:+.3f}")
    if report.sign_changes is not None:
        lines.append(f"turns     {report.sign_changes}")
    _note(ax, "\n".join(lines), "lower right")

    ax.set_xlabel("contact force  F_n  [N]")
    ax.set_ylabel(f"mean {name} over the hold window")
    ax.set_title(f"{name} against contact force -- is there anything to climb?")
    return figure


def plot_latency(
    samples_ms: Mapping[str, Sequence[float]], budget_ms: float = 1000.0 / 30.0
) -> "Figure":
    """Per-stage latency against the frame budget.

    The budget line is the point of the figure: a mean comfortably under it says
    little if the tail crosses it, because a control loop misses the frame it was
    late for, not the average frame.
    """
    apply_style()
    figure, ax = plt.subplots(figsize=(6.8, 4.2))

    for index, (label, values) in enumerate(sorted(samples_ms.items())):
        array = np.asarray(list(values), dtype=float)
        if array.size == 0:
            continue
        color = CATEGORICAL[index % len(CATEGORICAL)]
        ax.hist(array, bins=40, histtype="step", linewidth=_LINE, color=color, label=label)
        ax.axvline(float(np.percentile(array, 95)), color=color,
                   linewidth=_THIN, linestyle=":", alpha=0.8)

    ax.axvline(budget_ms, color=PALETTE["bad"], linewidth=1.4)
    ax.annotate(
        f"frame budget {budget_ms:.1f} ms",
        xy=(budget_ms, 0.96), xycoords=("data", "axes fraction"),
        xytext=(5, 0), textcoords="offset points",
        fontsize=8, color=PALETTE["bad"], va="top",
    )
    ax.set_xlabel("latency [ms]      dotted = p95")
    ax.set_ylabel("frames")
    ax.set_title("Latency against the 30 Hz frame budget")
    ax.legend(loc="upper right")
    return figure
