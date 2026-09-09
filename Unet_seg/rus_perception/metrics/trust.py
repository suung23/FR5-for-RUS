"""Is the quality score *trustworthy*? -- validation of Q_seg and Q_raw.

The other metric modules answer "how accurate is the segmentation" (:mod:`spatial`)
and "how stable is it over time" (:mod:`temporal`). Neither answers the question
a controller actually has to act on:

    When the perception layer says a frame is good, is it good?

That is a different quantity from accuracy. A model with mediocre Dice but an
honest quality score is *safe* -- the controller declines the bad frames. A model
with excellent mean Dice and a quality score that fails to separate its own
failures is *dangerous*, because nothing warns the controller before it acts on a
wrong mask. Mean Dice cannot distinguish those two systems. Everything here can.

Five questions, five families of function:

======================  ================================================  ==========================
question                function                                          headline number
======================  ================================================  ==========================
does Q track accuracy?  :func:`rank_agreement`                            Spearman rho
does the gate work?     :func:`gate_confusion`                            ``trusted_bad_rate``
where should we cut?    :func:`risk_coverage`, :func:`operating_point`    coverage at a target risk
does Q read as Dice?    :func:`calibration`                               ECE, bin monotonicity
which sub-scores earn   :func:`component_attribution`                     per-component rho / delta
their weight?
======================  ================================================  ==========================

Plus two that stand apart:

* :func:`reason_code_report` -- does each rejection reason fire on genuinely bad
  frames, or is it noise? A reason that fires constantly with no accuracy
  difference is a recovery action the supervisor will take for nothing.
* :func:`force_response` -- ``Q_raw`` has **no ground truth**: there is no
  annotation for "well coupled". Its validation is therefore structural, not
  comparative: does it respond monotonically (or unimodally) to contact force?
  If not, the Stage 1 force search is optimising noise.

Everything is NumPy-only and side-effect free. No plotting, no I/O: the report
objects are plain dataclasses with ``to_dict()``, and
``scripts/plot_report.py`` draws them.

Warning:
    ``accuracy_floor`` -- the Dice above which a frame counts as "good enough to
    act on" -- is a **decision, not a measurement**. The default of 0.70 is a
    placeholder. It belongs to the control requirement (how wrong may a mask be
    before the probe moves the wrong way?) and must be set deliberately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "DEFAULT_ACCURACY_FLOOR",
    "FrameOutcome",
    "AgreementReport",
    "GateConfusion",
    "CurvePoint",
    "DetectionReport",
    "RiskCoveragePoint",
    "RiskCoverageReport",
    "CalibrationBin",
    "CalibrationReport",
    "ComponentAttribution",
    "ReasonCodeStat",
    "ForceLevelSummary",
    "ForceResponseReport",
    "OperatingPoint",
    "TrustReport",
    "spearman",
    "pearson",
    "rank_agreement",
    "gate_confusion",
    "detection_report",
    "risk_coverage",
    "calibration",
    "component_attribution",
    "reason_code_report",
    "asymmetric_margin",
    "unimodal_regression",
    "force_response",
    "operating_point",
    "build_trust_report",
    "group_by",
]

#: Dice at or above which a frame counts as usable for control. A DECISION.
DEFAULT_ACCURACY_FLOOR = 0.70

_EPS = 1e-12


# ---------------------------------------------------------------------------
# input record
# ---------------------------------------------------------------------------


@dataclass
class FrameOutcome:
    """What one frame produced, and what it was actually worth.

    Attributes:
        quality: The score under test (``control_quality_score`` for Q_seg,
            ``score`` for Q_raw). ``None`` means *not measured* -- the frame is
            excluded from every statistic and counted separately, never treated
            as a zero.
        valid: The gate's decision (``valid_for_control``, or
            ``usable_for_contact_search``).
        accuracy: Ground-truth agreement for this frame, normally Dice.
            ``None`` for an unlabeled frame; excluded and counted, never zero.
        components: Sub-scores that produced ``quality``, for attribution.
        reasons: Rejection reasons this frame emitted.
        patient_id: Grouping key. Aggregates that ignore it overstate confidence,
            because frames within a sweep are highly correlated.
        sequence_id: Grouping key within a patient.
        frame_index: Chronological position, for timeline plots.
    """

    quality: Optional[float] = None
    valid: bool = False
    accuracy: Optional[float] = None
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    patient_id: str = ""
    sequence_id: str = ""
    frame_index: int = 0

    @property
    def is_scored(self) -> bool:
        """True when both a quality score and a ground-truth accuracy exist."""
        return (
            self.quality is not None
            and self.accuracy is not None
            and math.isfinite(float(self.quality))
            and math.isfinite(float(self.accuracy))
        )


def _paired(outcomes: Sequence[FrameOutcome]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(quality, accuracy, valid)`` for frames that have both values."""
    usable = [o for o in outcomes if o.is_scored]
    if not usable:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, np.zeros(0, dtype=bool)
    quality = np.asarray([float(o.quality) for o in usable], dtype=np.float64)
    accuracy = np.asarray([float(o.accuracy) for o in usable], dtype=np.float64)
    valid = np.asarray([bool(o.valid) for o in usable], dtype=bool)
    return quality, accuracy, valid


def group_by(
    outcomes: Sequence[FrameOutcome], key: str = "patient_id"
) -> dict[str, list[FrameOutcome]]:
    """Split outcomes by ``patient_id`` (default) or ``sequence_id``.

    Frames within one sweep are highly correlated, so a pooled statistic over all
    frames reports a smaller spread than the system will show on the next
    patient. Report per patient as well as pooled, always.
    """
    if key not in ("patient_id", "sequence_id"):
        raise ValueError(f"group_by key must be patient_id or sequence_id, got {key!r}.")
    groups: dict[str, list[FrameOutcome]] = {}
    for outcome in outcomes:
        groups.setdefault(str(getattr(outcome, key)), []).append(outcome)
    return groups


# ---------------------------------------------------------------------------
# 1. does Q track accuracy?
# ---------------------------------------------------------------------------


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared -- the tie handling Spearman requires."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Pearson correlation, or ``None`` when it is undefined (n < 2 or no spread)."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or b.size != a.size:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denominator < _EPS:
        return None
    return float((a * b).sum() / denominator)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation, tie-corrected.

    Rank correlation, not Pearson, is the primary statistic here: a quality score
    is used by *thresholding and ordering* frames, never by reading its value as
    a physical quantity. Whether Q is linear in Dice does not matter; whether it
    orders frames the same way does.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or b.size != a.size:
        return None
    return pearson(_rankdata(a), _rankdata(b))


@dataclass
class AgreementReport:
    """How well the quality score orders frames by their realised accuracy."""

    n: int
    spearman: Optional[float]
    pearson: Optional[float]
    n_unscored: int = 0
    n_unlabeled: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "spearman": self.spearman,
            "pearson": self.pearson,
            "n_unscored": self.n_unscored,
            "n_unlabeled": self.n_unlabeled,
        }


def rank_agreement(outcomes: Sequence[FrameOutcome]) -> AgreementReport:
    """Correlate the quality score against ground-truth accuracy.

    A high Spearman rho means the score can be thresholded usefully. A rho near
    zero means the score carries no information about accuracy, and no choice of
    threshold will make the gate work -- the weights, not the threshold, are what
    needs fixing.
    """
    quality, accuracy, _ = _paired(outcomes)
    return AgreementReport(
        n=int(quality.size),
        spearman=spearman(quality, accuracy),
        pearson=pearson(quality, accuracy),
        n_unscored=sum(1 for o in outcomes if o.quality is None),
        n_unlabeled=sum(1 for o in outcomes if o.accuracy is None),
    )


# ---------------------------------------------------------------------------
# 2. does the gate work?
# ---------------------------------------------------------------------------


@dataclass
class GateConfusion:
    """The four outcomes of a binary gate, named by what they cost.

    ``trusted_bad`` is the only one that can hurt a patient: the perception layer
    said the frame was usable and it was not, so the controller acted on a wrong
    mask with no warning. Every other cell is either correct or merely
    conservative.
    """

    accuracy_floor: float
    trusted_good: int = 0
    trusted_bad: int = 0
    rejected_bad: int = 0
    rejected_good: int = 0

    @property
    def n(self) -> int:
        return self.trusted_good + self.trusted_bad + self.rejected_bad + self.rejected_good

    @property
    def coverage(self) -> Optional[float]:
        """Fraction of frames the gate accepts -- how often the controller can act."""
        return None if self.n == 0 else (self.trusted_good + self.trusted_bad) / self.n

    @property
    def trusted_bad_rate(self) -> Optional[float]:
        """**The headline number.** Of the frames we trusted, what fraction were wrong?"""
        accepted = self.trusted_good + self.trusted_bad
        return None if accepted == 0 else self.trusted_bad / accepted

    @property
    def rejected_good_rate(self) -> Optional[float]:
        """Of the frames we rejected, what fraction were actually fine? The cost of caution."""
        rejected = self.rejected_good + self.rejected_bad
        return None if rejected == 0 else self.rejected_good / rejected

    @property
    def recall(self) -> Optional[float]:
        """Of all usable frames, how many did the gate let through?"""
        good = self.trusted_good + self.rejected_good
        return None if good == 0 else self.trusted_good / good

    @property
    def specificity(self) -> Optional[float]:
        """Of all unusable frames, how many did the gate catch?"""
        bad = self.trusted_bad + self.rejected_bad
        return None if bad == 0 else self.rejected_bad / bad

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy_floor": self.accuracy_floor,
            "n": self.n,
            "trusted_good": self.trusted_good,
            "trusted_bad": self.trusted_bad,
            "rejected_bad": self.rejected_bad,
            "rejected_good": self.rejected_good,
            "coverage": self.coverage,
            "trusted_bad_rate": self.trusted_bad_rate,
            "rejected_good_rate": self.rejected_good_rate,
            "recall": self.recall,
            "specificity": self.specificity,
        }


def gate_confusion(
    outcomes: Sequence[FrameOutcome], accuracy_floor: float = DEFAULT_ACCURACY_FLOOR
) -> GateConfusion:
    """Score the binary validity gate against ground truth."""
    _, accuracy, valid = _paired(outcomes)
    good = accuracy >= accuracy_floor
    return GateConfusion(
        accuracy_floor=float(accuracy_floor),
        trusted_good=int(np.count_nonzero(valid & good)),
        trusted_bad=int(np.count_nonzero(valid & ~good)),
        rejected_bad=int(np.count_nonzero(~valid & ~good)),
        rejected_good=int(np.count_nonzero(~valid & good)),
    )


# ---------------------------------------------------------------------------
# 3. threshold sweeps: ROC, PR, risk-coverage, operating point
# ---------------------------------------------------------------------------


@dataclass
class CurvePoint:
    """One threshold on a sweep."""

    threshold: float
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"threshold": self.threshold, "x": self.x, "y": self.y}


@dataclass
class DetectionReport:
    """The quality score treated as a detector of "this frame is usable"."""

    accuracy_floor: float
    n: int
    n_positive: int
    auroc: Optional[float]
    average_precision: Optional[float]
    roc: list[CurvePoint] = field(default_factory=list)
    precision_recall: list[CurvePoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy_floor": self.accuracy_floor,
            "n": self.n,
            "n_positive": self.n_positive,
            "auroc": self.auroc,
            "average_precision": self.average_precision,
            "roc": [p.to_dict() for p in self.roc],
            "precision_recall": [p.to_dict() for p in self.precision_recall],
        }


def _sweep_thresholds(score: np.ndarray) -> np.ndarray:
    """Descending unique scores: every threshold that changes the accepted set."""
    return np.unique(score)[::-1]


def detection_report(
    outcomes: Sequence[FrameOutcome], accuracy_floor: float = DEFAULT_ACCURACY_FLOOR
) -> DetectionReport:
    """ROC and precision-recall for "does Q > theta identify a usable frame?".

    AUROC is threshold-free, so it separates *the score's discriminative power*
    from *where the current gate happens to sit*. A high AUROC with a bad
    :func:`gate_confusion` means the score is fine and the thresholds are wrong;
    a low AUROC means no threshold can save it.
    """
    quality, accuracy, _ = _paired(outcomes)
    positive = accuracy >= accuracy_floor
    n = int(quality.size)
    n_positive = int(np.count_nonzero(positive))

    if n == 0 or n_positive == 0 or n_positive == n:
        # With only one class present, neither curve is defined. Say so rather
        # than emitting a 0.5 that reads like a measured result.
        return DetectionReport(
            accuracy_floor=float(accuracy_floor),
            n=n,
            n_positive=n_positive,
            auroc=None,
            average_precision=None,
        )

    roc: list[CurvePoint] = []
    pr: list[CurvePoint] = []
    n_negative = n - n_positive

    for threshold in _sweep_thresholds(quality):
        accepted = quality >= threshold
        tp = int(np.count_nonzero(accepted & positive))
        fp = int(np.count_nonzero(accepted & ~positive))
        tpr = tp / n_positive
        fpr = fp / n_negative
        roc.append(CurvePoint(threshold=float(threshold), x=fpr, y=tpr))
        precision = tp / max(tp + fp, 1)
        pr.append(CurvePoint(threshold=float(threshold), x=tpr, y=precision))

    # AUROC via the rank statistic: equivalent to the trapezoid area and exact
    # under ties, which a curve-integration would get subtly wrong.
    ranks = _rankdata(quality)
    rank_sum = float(ranks[positive].sum())
    auroc = (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)

    # Average precision: sum of precision at each threshold weighted by the
    # recall it gains, which is the standard step-wise integral.
    average_precision = 0.0
    previous_recall = 0.0
    for point in pr:
        average_precision += point.y * (point.x - previous_recall)
        previous_recall = point.x

    return DetectionReport(
        accuracy_floor=float(accuracy_floor),
        n=n,
        n_positive=n_positive,
        auroc=float(auroc),
        average_precision=float(average_precision),
        roc=roc,
        precision_recall=pr,
    )


@dataclass
class RiskCoveragePoint:
    """Accepting the top ``coverage`` fraction by quality: what do we get?"""

    threshold: float
    coverage: float
    mean_accuracy: float
    bad_rate: float
    n_accepted: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "coverage": self.coverage,
            "mean_accuracy": self.mean_accuracy,
            "bad_rate": self.bad_rate,
            "n_accepted": self.n_accepted,
        }


@dataclass
class RiskCoverageReport:
    """Selective prediction: the accuracy bought by declining to act.

    This is the curve the controller is actually designed against. It answers
    "if the supervisor only acts on frames the perception layer likes, how often
    can it act, and how wrong is it when it does?" -- which is precisely the
    trade-off between throughput and safety.
    """

    accuracy_floor: float
    points: list[RiskCoveragePoint] = field(default_factory=list)
    aurc: Optional[float] = None
    full_coverage_bad_rate: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy_floor": self.accuracy_floor,
            "aurc": self.aurc,
            "full_coverage_bad_rate": self.full_coverage_bad_rate,
            "points": [p.to_dict() for p in self.points],
        }


def risk_coverage(
    outcomes: Sequence[FrameOutcome], accuracy_floor: float = DEFAULT_ACCURACY_FLOOR
) -> RiskCoverageReport:
    """Sweep the acceptance threshold from "accept nothing" to "accept everything"."""
    quality, accuracy, _ = _paired(outcomes)
    if quality.size == 0:
        return RiskCoverageReport(accuracy_floor=float(accuracy_floor))

    n = int(quality.size)
    order = np.argsort(-quality, kind="mergesort")
    sorted_accuracy = accuracy[order]
    sorted_quality = quality[order]

    cumulative_accuracy = np.cumsum(sorted_accuracy)
    cumulative_bad = np.cumsum((sorted_accuracy < accuracy_floor).astype(np.float64))
    counts = np.arange(1, n + 1, dtype=np.float64)

    points = [
        RiskCoveragePoint(
            threshold=float(sorted_quality[i]),
            coverage=float(counts[i] / n),
            mean_accuracy=float(cumulative_accuracy[i] / counts[i]),
            bad_rate=float(cumulative_bad[i] / counts[i]),
            n_accepted=int(counts[i]),
        )
        for i in range(n)
    ]

    # AURC over the bad-rate curve: lower is better, 0 means every accepted frame
    # was good at every coverage.
    coverage = np.asarray([p.coverage for p in points])
    bad_rate = np.asarray([p.bad_rate for p in points])
    aurc = float(np.trapezoid(bad_rate, coverage)) if n > 1 else float(bad_rate[0])

    return RiskCoverageReport(
        accuracy_floor=float(accuracy_floor),
        points=points,
        aurc=aurc,
        full_coverage_bad_rate=float(bad_rate[-1]),
    )


@dataclass
class OperatingPoint:
    """A concrete threshold recommendation, with its cost stated."""

    target_bad_rate: float
    accuracy_floor: float
    threshold: Optional[float]
    coverage: Optional[float]
    achieved_bad_rate: Optional[float]
    mean_accuracy: Optional[float]
    n_accepted: int = 0
    feasible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_bad_rate": self.target_bad_rate,
            "accuracy_floor": self.accuracy_floor,
            "threshold": self.threshold,
            "coverage": self.coverage,
            "achieved_bad_rate": self.achieved_bad_rate,
            "mean_accuracy": self.mean_accuracy,
            "n_accepted": self.n_accepted,
            "feasible": self.feasible,
        }


def operating_point(
    outcomes: Sequence[FrameOutcome],
    target_bad_rate: float = 0.02,
    accuracy_floor: float = DEFAULT_ACCURACY_FLOOR,
    min_coverage: float = 0.0,
) -> OperatingPoint:
    """Largest coverage whose accepted set stays within ``target_bad_rate``.

    Safety first, throughput second: the search maximises how often the
    controller may act *subject to* the trusted-bad rate staying under budget,
    rather than trading the two off against each other with a single score.

    Args:
        outcomes: Frames with quality and ground truth.
        target_bad_rate: Maximum tolerable fraction of accepted frames that turn
            out to be below ``accuracy_floor``. A safety budget, not a fit
            parameter.
        accuracy_floor: Dice defining "good enough to act on".
        min_coverage: Reject an operating point that accepts less than this
            fraction of frames -- a gate that never opens is not a solution.

    Returns:
        An :class:`OperatingPoint`; ``feasible=False`` when no threshold meets
        the budget, which is a real and reportable answer.
    """
    report = risk_coverage(outcomes, accuracy_floor)
    feasible = [
        p
        for p in report.points
        if p.bad_rate <= target_bad_rate and p.coverage >= min_coverage
    ]
    if not feasible:
        return OperatingPoint(
            target_bad_rate=float(target_bad_rate),
            accuracy_floor=float(accuracy_floor),
            threshold=None,
            coverage=None,
            achieved_bad_rate=None,
            mean_accuracy=None,
            feasible=False,
        )
    best = max(feasible, key=lambda p: p.coverage)
    return OperatingPoint(
        target_bad_rate=float(target_bad_rate),
        accuracy_floor=float(accuracy_floor),
        threshold=best.threshold,
        coverage=best.coverage,
        achieved_bad_rate=best.bad_rate,
        mean_accuracy=best.mean_accuracy,
        n_accepted=best.n_accepted,
        feasible=True,
    )


# ---------------------------------------------------------------------------
# 4. does Q read as accuracy?
# ---------------------------------------------------------------------------


@dataclass
class CalibrationBin:
    """One bin of the reliability curve."""

    lower: float
    upper: float
    n: int
    mean_quality: float
    mean_accuracy: float
    std_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "n": self.n,
            "mean_quality": self.mean_quality,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
        }


@dataclass
class CalibrationReport:
    """How closely the quality score reads as the Dice it will deliver.

    Note:
        ``Q`` is **not** a probability of correctness and was never built to be
        one, so a non-zero ECE is not by itself a defect. What matters more is
        ``monotone_fraction``: if bins of increasing Q do not deliver increasing
        Dice, the score is not merely mis-scaled, it is misleading, and no
        recalibration fixes that.
    """

    bins: list[CalibrationBin] = field(default_factory=list)
    ece: Optional[float] = None
    mce: Optional[float] = None
    monotone_fraction: Optional[float] = None
    bin_spearman: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece": self.ece,
            "mce": self.mce,
            "monotone_fraction": self.monotone_fraction,
            "bin_spearman": self.bin_spearman,
            "bins": [b.to_dict() for b in self.bins],
        }


def calibration(
    outcomes: Sequence[FrameOutcome],
    n_bins: int = 10,
    strategy: str = "quantile",
    min_bin_count: int = 5,
) -> CalibrationReport:
    """Bin frames by quality and compare mean quality with mean accuracy.

    Args:
        outcomes: Frames with quality and ground truth.
        n_bins: Number of bins.
        strategy: ``quantile`` (equal counts, the default -- quality scores
            cluster and equal-width bins leave most of them nearly empty) or
            ``uniform`` (equal width).
        min_bin_count: Bins smaller than this are dropped from ECE and
            monotonicity rather than contributing a mean over three frames.

    Raises:
        ValueError: On an unknown ``strategy`` or ``n_bins < 2``.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}.")
    if strategy not in ("quantile", "uniform"):
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}.")

    quality, accuracy, _ = _paired(outcomes)
    if quality.size == 0:
        return CalibrationReport()

    if strategy == "quantile":
        edges = np.unique(np.quantile(quality, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        edges = np.linspace(float(quality.min()), float(quality.max()), n_bins + 1)
    if edges.size < 2:
        return CalibrationReport()

    bins: list[CalibrationBin] = []
    for i in range(edges.size - 1):
        lower, upper = float(edges[i]), float(edges[i + 1])
        last = i == edges.size - 2
        selected = (quality >= lower) & ((quality <= upper) if last else (quality < upper))
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                n=count,
                mean_quality=float(quality[selected].mean()),
                mean_accuracy=float(accuracy[selected].mean()),
                std_accuracy=float(accuracy[selected].std()),
            )
        )

    usable = [b for b in bins if b.n >= min_bin_count]
    if not usable:
        return CalibrationReport(bins=bins)

    total = sum(b.n for b in usable)
    gaps = [abs(b.mean_quality - b.mean_accuracy) for b in usable]
    ece = sum(b.n * g for b, g in zip(usable, gaps)) / total
    mce = max(gaps)

    if len(usable) >= 2:
        steps = [
            usable[i + 1].mean_accuracy >= usable[i].mean_accuracy
            for i in range(len(usable) - 1)
        ]
        monotone_fraction = sum(steps) / len(steps)
        bin_spearman = spearman(
            [b.mean_quality for b in usable], [b.mean_accuracy for b in usable]
        )
    else:
        monotone_fraction = None
        bin_spearman = None

    return CalibrationReport(
        bins=bins,
        ece=float(ece),
        mce=float(mce),
        monotone_fraction=monotone_fraction,
        bin_spearman=bin_spearman,
    )


# ---------------------------------------------------------------------------
# 5. which sub-scores earn their weight?
# ---------------------------------------------------------------------------


@dataclass
class ComponentAttribution:
    """One sub-score's contribution, measured rather than assumed."""

    name: str
    n: int
    spearman: Optional[float]
    weight: float
    leave_one_out_spearman: Optional[float] = None
    delta_spearman: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "spearman": self.spearman,
            "weight": self.weight,
            "leave_one_out_spearman": self.leave_one_out_spearman,
            "delta_spearman": self.delta_spearman,
        }


def _weighted_mean(components: Mapping[str, float], weights: Mapping[str, float]) -> Optional[float]:
    active = {k: float(weights.get(k, 0.0)) for k in components if weights.get(k, 0.0) > 0}
    total = sum(active.values())
    if total <= 0:
        return None
    return sum(float(components[k]) * w for k, w in active.items()) / total


def component_attribution(
    outcomes: Sequence[FrameOutcome], weights: Mapping[str, float]
) -> list[ComponentAttribution]:
    """Measure what each sub-score actually contributes to ordering frames.

    Two numbers per component:

    * ``spearman`` -- how well that sub-score *alone* orders frames by accuracy.
    * ``delta_spearman`` -- how much the **aggregate** score's rank agreement
      drops when this component's weight is set to zero. Negative means removing
      it makes the aggregate *better*: the component is actively harmful and its
      weight should not merely be reduced.

    This is the evidence that replaces the hand-chosen weights. Until it exists,
    every weight in ``control.quality`` is a guess -- including the ones this
    repository ships.

    Args:
        outcomes: Frames carrying ``components`` and ground-truth accuracy.
        weights: The weights the aggregate currently uses.

    Returns:
        One entry per component seen, ordered by ``|spearman|`` descending.
    """
    labeled = [o for o in outcomes if o.accuracy is not None and o.components]
    if not labeled:
        return []

    accuracy = np.asarray([float(o.accuracy) for o in labeled], dtype=np.float64)
    names = sorted({name for o in labeled for name in o.components})

    baseline_scores, baseline_accuracy = [], []
    for outcome, value in zip(labeled, accuracy):
        aggregate = _weighted_mean(outcome.components, weights)
        if aggregate is not None:
            baseline_scores.append(aggregate)
            baseline_accuracy.append(value)
    baseline = spearman(baseline_scores, baseline_accuracy)

    results: list[ComponentAttribution] = []
    for name in names:
        present = [i for i, o in enumerate(labeled) if name in o.components]
        solo = spearman(
            [float(labeled[i].components[name]) for i in present],
            [float(accuracy[i]) for i in present],
        )

        reduced_weights = {k: (0.0 if k == name else float(v)) for k, v in weights.items()}
        reduced_scores, reduced_accuracy = [], []
        for outcome, value in zip(labeled, accuracy):
            aggregate = _weighted_mean(outcome.components, reduced_weights)
            if aggregate is not None:
                reduced_scores.append(aggregate)
                reduced_accuracy.append(value)
        loo = spearman(reduced_scores, reduced_accuracy)

        delta = None if (baseline is None or loo is None) else float(baseline - loo)
        results.append(
            ComponentAttribution(
                name=name,
                n=len(present),
                spearman=solo,
                weight=float(weights.get(name, 0.0)),
                leave_one_out_spearman=loo,
                delta_spearman=delta,
            )
        )

    results.sort(key=lambda c: (-abs(c.spearman) if c.spearman is not None else 0.0, c.name))
    return results


# ---------------------------------------------------------------------------
# 6. do the reason codes mean anything?
# ---------------------------------------------------------------------------


@dataclass
class ReasonCodeStat:
    """How often one rejection reason fires, and whether it was right to."""

    reason: str
    n_fired: int
    fire_rate: float
    mean_accuracy_when_fired: Optional[float]
    mean_accuracy_when_silent: Optional[float]
    precision: Optional[float]
    lift: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "n_fired": self.n_fired,
            "fire_rate": self.fire_rate,
            "mean_accuracy_when_fired": self.mean_accuracy_when_fired,
            "mean_accuracy_when_silent": self.mean_accuracy_when_silent,
            "precision": self.precision,
            "lift": self.lift,
        }


def reason_code_report(
    outcomes: Sequence[FrameOutcome], accuracy_floor: float = DEFAULT_ACCURACY_FLOOR
) -> list[ReasonCodeStat]:
    """Per reason code: how often it fires and whether the frame was really bad.

    Each reason maps to a distinct supervisor recovery action, so a reason that
    fires on frames no worse than average is not a harmless extra check -- it is
    a state transition taken for nothing. ``lift`` below 1.0 marks exactly that:
    the frames it fires on are no more likely to be bad than a random frame.
    """
    labeled = [o for o in outcomes if o.accuracy is not None]
    if not labeled:
        return []

    accuracy = np.asarray([float(o.accuracy) for o in labeled], dtype=np.float64)
    base_bad_rate = float(np.count_nonzero(accuracy < accuracy_floor)) / len(labeled)
    codes = sorted({code for o in labeled for code in o.reasons})

    stats: list[ReasonCodeStat] = []
    for code in codes:
        fired = np.asarray([code in o.reasons for o in labeled], dtype=bool)
        n_fired = int(np.count_nonzero(fired))
        if n_fired == 0:
            continue
        fired_accuracy = accuracy[fired]
        silent_accuracy = accuracy[~fired]
        precision = float(np.count_nonzero(fired_accuracy < accuracy_floor)) / n_fired
        stats.append(
            ReasonCodeStat(
                reason=code,
                n_fired=n_fired,
                fire_rate=n_fired / len(labeled),
                mean_accuracy_when_fired=float(fired_accuracy.mean()),
                mean_accuracy_when_silent=(
                    float(silent_accuracy.mean()) if silent_accuracy.size else None
                ),
                precision=precision,
                lift=(precision / base_bad_rate) if base_bad_rate > _EPS else None,
            )
        )

    stats.sort(key=lambda s: -s.n_fired)
    return stats


# ---------------------------------------------------------------------------
# 7. Q_raw has no ground truth: validate its shape instead
# ---------------------------------------------------------------------------


def _normal_cdf(t: float) -> float:
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def _normal_pdf(t: float) -> float:
    return math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)


def asymmetric_margin(sigma: float, w_minus_over_plus: float = 5.0) -> float:
    """Safety margin above the left edge of the tie set, for asymmetric cost.

    The Stage 1 rule picks the *smallest* force whose quality ties with the
    maximum. Taken literally, that leaves zero margin on the side where an
    error means **losing contact** -- and losing contact costs far more than
    pressing slightly harder (DESIGN_NOTES §5.3, §7.1 correction of
    2026-08-25). So the setpoint is moved inward from the edge by the ``u >= 0``
    that minimises the expected asymmetric quadratic cost under a Gaussian
    error ``n ~ N(0, sigma^2)`` on the true edge position:

    .. math::

        u^* = \\arg\\min_{u \\ge 0}\\;
        \\mathbb{E}\\big[w_-\\,((-u-n)^+)^2 + w_+\\,((u+n)^+)^2\\big]

    which is the root of
    ``w_+ * E[(u+n)^+] = w_- * E[(-u-n)^+]`` with
    ``E[(u+n)^+] = u*Phi(u/sigma) + sigma*phi(u/sigma)`` and
    ``E[(-u-n)^+] = -u*Phi(-u/sigma) + sigma*phi(u/sigma)``, found by bisection
    on ``t = u/sigma`` in ``[0, 5]``.

    Args:
        sigma: Standard deviation of the edge-position error, in newtons.
            Typically the force-grid spacing or the spread of ``f_left`` over
            repeated searches. ``sigma <= 0`` returns ``0``.
        w_minus_over_plus: Cost ratio ``w_- / w_+`` of undershooting (losing
            contact) to overshooting (extra pressure). ``5`` is the PROVISIONAL
            design value; ``<= 1`` gives no margin.

    Returns:
        The margin in the units of ``sigma``: ``0.636 sigma`` for a ratio of 5,
        ``0.436 sigma`` for 3, ``0.901 sigma`` for 10.
    """
    sigma = float(sigma)
    ratio = float(w_minus_over_plus)
    if sigma <= 0.0 or not math.isfinite(sigma):
        return 0.0
    if ratio < 0.0:
        raise ValueError(f"w_minus_over_plus must be >= 0, got {ratio}.")

    def imbalance(t: float) -> float:
        # Positive when overshoot cost already dominates: push u no further.
        over = t * _normal_cdf(t) + _normal_pdf(t)
        under = -t * _normal_cdf(-t) + _normal_pdf(t)
        return over - ratio * under

    low, high = 0.0, 5.0
    if imbalance(low) >= 0.0:
        return 0.0
    if imbalance(high) <= 0.0:
        return high * sigma
    for _ in range(100):
        mid = 0.5 * (low + high)
        if imbalance(mid) < 0.0:
            low = mid
        else:
            high = mid
        if high - low < 1e-12:
            break
    return 0.5 * (low + high) * sigma


def _pava_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: the closest non-decreasing sequence."""
    blocks: list[list[float]] = []  # [weighted mean, total weight, count]
    for value, weight in zip(values, weights):
        blocks.append([float(value), float(weight), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            mean_b, weight_b, count_b = blocks.pop()
            mean_a, weight_a, count_a = blocks.pop()
            total = weight_a + weight_b
            pooled = (mean_a * weight_a + mean_b * weight_b) / total if total > 0 else 0.5 * (mean_a + mean_b)
            blocks.append([pooled, total, count_a + count_b])
    fit = np.empty(values.size, dtype=np.float64)
    start = 0
    for mean, _, count in blocks:
        stop = start + int(count)
        fit[start:stop] = mean
        start = stop
    return fit


def unimodal_regression(
    means: np.ndarray, weights: Optional[np.ndarray] = None
) -> tuple[np.ndarray, int, float]:
    """Umbrella (rise-then-fall) weighted isotonic regression.

    For every candidate peak ``p`` the left part ``means[:p+1]`` is fitted
    non-decreasing and the right part ``means[p:]`` non-increasing (both by
    weighted PAVA, the right one on the reversed sequence); the two fits share
    index ``p`` and the peak takes the larger of the two values there, which
    keeps both halves feasible. The ``p`` with the smallest weighted sum of
    squared residuals wins.

    This is the shape the force search assumes ``Q(F)`` has. How far the data
    sit from their best umbrella fit -- relative to their own noise -- is a
    direct test of that assumption, and one that, unlike counting sign
    changes, does not depend on an ad-hoc step threshold.

    Args:
        means: Per-level values in force order.
        weights: Non-negative weights of the same length; ``None`` means equal.

    Returns:
        ``(fit, peak_index, weighted_sse)``. For an empty input the fit is empty,
        the index ``0`` and the SSE ``0``.
    """
    values = np.asarray(means, dtype=np.float64).ravel()
    n = values.size
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != values.shape:
            raise ValueError(f"weights shape {w.shape} does not match means {values.shape}.")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
    if n == 0:
        return values.copy(), 0, 0.0

    best_fit = values.copy()
    best_peak = 0
    best_sse = math.inf
    for p in range(n):
        left = _pava_increasing(values[: p + 1], w[: p + 1])
        right = _pava_increasing(values[p:][::-1], w[p:][::-1])[::-1]
        fit = np.empty(n, dtype=np.float64)
        fit[:p] = left[:p]
        fit[p + 1 :] = right[1:]
        fit[p] = max(left[p], right[0])
        sse = float(np.sum(w * (values - fit) ** 2))
        if sse < best_sse - 1e-15:
            best_fit, best_peak, best_sse = fit, p, sse
    return best_fit, best_peak, best_sse


def _lag1_autocorrelation(values: np.ndarray) -> float:
    """Lag-1 autocorrelation of a hold-window sample sequence; 0 when undefined."""
    if values.size < 3:
        return 0.0
    centred = values - values.mean()
    denominator = float(np.dot(centred, centred))
    if denominator < _EPS:
        return 0.0
    rho = float(np.dot(centred[:-1], centred[1:]) / denominator)
    return max(-1.0, min(1.0, rho))


@dataclass
class ForceLevelSummary:
    """Q averaged over the hold window at one force level.

    Attributes:
        rho: Lag-1 autocorrelation of the usable samples in hold-window order;
            ``0`` when fewer than three samples or no spread.
        n_eff: Effective sample count ``n (1 - rho) / (1 + rho)`` clipped to
            ``[2, n]`` -- consecutive frames of a hold window are not
            independent, and ``sem`` is computed against this count, not ``n``.
    """

    force: float
    n: int
    mean: Optional[float]
    std: Optional[float]
    sem: Optional[float]
    valid_fraction: float = 1.0
    rho: Optional[float] = None
    n_eff: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "force": self.force,
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "sem": self.sem,
            "valid_fraction": self.valid_fraction,
            "rho": self.rho,
            "n_eff": self.n_eff,
        }


@dataclass
class ForceResponseReport:
    """Does Q respond to contact force in a way a search can climb?

    The Stage 1 force search assumes ``Q(F)`` has a single well-defined optimum.
    If the response is flat, noisy, or multi-modal, the search is optimising
    nothing and the whole stage needs rethinking -- so this is the experiment
    that decides whether Stage 1 is viable, not a nice-to-have diagnostic.

    Attributes:
        f_left: Smallest force in the tie set with the maximum -- the literal
            "smallest force" of the Stage 1 rule.
        margin_n: Safety margin added to ``f_left`` (see :func:`asymmetric_margin`).
        f_star: ``f_left + margin_n``, the recommended setpoint.
        tie_rule: ``"epsilon"`` (fixed quality tolerance) or ``"welch"``
            (tolerance scaled by the two levels' standard errors).
        unimodal_sse_ratio: Residual of the best umbrella fit per level,
            divided by the mean squared standard error -- about ``1`` when the
            data are unimodal up to noise, large when they are not.
        unimodal_fit: The umbrella fit itself, one value per measured level.
        unimodal_peak_force: Force at the umbrella fit's peak.
    """

    levels: list[ForceLevelSummary] = field(default_factory=list)
    spearman_with_force: Optional[float] = None
    monotone_fraction: Optional[float] = None
    sign_changes: Optional[int] = None
    is_unimodal: Optional[bool] = None
    argmax_force: Optional[float] = None
    f_star: Optional[float] = None
    epsilon: float = 0.03
    f_left: Optional[float] = None
    margin_n: float = 0.0
    tie_rule: str = "epsilon"
    z_alpha: Optional[float] = None
    unimodal_sse_ratio: Optional[float] = None
    unimodal_ratio_max: float = 2.0
    unimodal_fit: Optional[list[float]] = None
    unimodal_peak_force: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spearman_with_force": self.spearman_with_force,
            "monotone_fraction": self.monotone_fraction,
            "sign_changes": self.sign_changes,
            "is_unimodal": self.is_unimodal,
            "argmax_force": self.argmax_force,
            "f_star": self.f_star,
            "epsilon": self.epsilon,
            "f_left": self.f_left,
            "margin_n": self.margin_n,
            "tie_rule": self.tie_rule,
            "z_alpha": self.z_alpha,
            "unimodal_sse_ratio": self.unimodal_sse_ratio,
            "unimodal_ratio_max": self.unimodal_ratio_max,
            "unimodal_fit": self.unimodal_fit,
            "unimodal_peak_force": self.unimodal_peak_force,
            "levels": [level.to_dict() for level in self.levels],
        }


def force_response(
    samples_by_force: Mapping[float, Sequence[Optional[float]]],
    epsilon: float = 0.03,
    min_valid_fraction: float = 0.60,
    *,
    margin_n: float = 0.0,
    z_alpha: Optional[float] = None,
    autocorr_correct: bool = True,
    unimodal_ratio_max: float = 2.0,
) -> ForceResponseReport:
    """Summarise ``Q`` against contact force and test the shape the search assumes.

    The 2026-09-08 revision of DESIGN_NOTES §7.1/§7.3 changed four things about
    how the setpoint is read off the response, all of which are keyword-only
    here so that a legacy call keeps its exact former behaviour:

    * **Autocorrelation.** The frames of a 1 s hold window are not independent
      samples: tissue relaxes and speckle drifts slowly, so ``std/sqrt(n)``
      overstates how well a level's mean is known. Each level's lag-1
      autocorrelation ``rho`` is estimated and ``sem`` uses
      ``n_eff = n (1 - rho) / (1 + rho)`` (clipped to ``[2, n]``) instead of
      ``n``. ``autocorr_correct=False`` restores ``n``.
    * **Tie rule.** With ``z_alpha=None`` a level ties with the maximum when
      ``mean >= peak - epsilon`` (legacy). With ``z_alpha`` set the tolerance is
      Welch-type, ``z_alpha * sqrt(sem^2 + sem_peak^2)``, so "statistically
      indistinguishable from the best" is measured against the data's own
      noise rather than a fixed 0.03.
    * **Margin.** ``f_left`` is the smallest force in the tie set; the
      recommended setpoint is ``f_star = f_left + margin_n``. The literal
      left edge leaves no room on the side where an error means losing
      contact, and under a ``w_-/w_+ = 5`` asymmetric cost the optimum sits
      ``0.636 sigma`` inside it (:func:`asymmetric_margin`). With the default
      ``margin_n=0`` the two coincide, as before. The caller keeps ``f_star``
      inside the safe force range; nothing here clamps it.
    * **Unimodality.** ``sign_changes`` (turns larger than ``epsilon``) is
      still reported, but when every measured level has a positive ``sem`` the
      verdict comes from :func:`unimodal_regression`: the best rise-then-fall
      fit's residual per level, divided by the mean ``sem^2``, is about ``1``
      for a unimodal response and grows without bound for a multimodal one;
      ``is_unimodal`` is ``ratio <= unimodal_ratio_max``. When the ratio is
      undefined (a level with zero spread) the legacy ``sign_changes <= 1``
      rule decides.

    Args:
        samples_by_force: ``{force_newtons: [q, q, None, ...]}`` -- every sample
            collected during that level's hold window, in time order. ``None``
            entries are unmeasured frames and are counted, not zeroed.
        epsilon: Quality difference treated as a tie under the legacy rule, and
            the step size below which a turn is noise for ``sign_changes``.
        min_valid_fraction: A level with fewer usable samples than this is a
            **measurement failure**, reported with ``mean=None`` and excluded
            from the shape tests, rather than contributing a mean over a handful
            of frames.
        margin_n: Newtons added to ``f_left`` to obtain ``f_star``.
        z_alpha: One-sided normal quantile for the Welch tie rule (``1.64`` for
            alpha = 0.05); ``None`` keeps the ``epsilon`` rule.
        autocorr_correct: Use ``n_eff`` rather than ``n`` in ``sem``.
        unimodal_ratio_max: Largest umbrella-fit SSE ratio still called unimodal.

    Returns:
        A :class:`ForceResponseReport`. ``f_star`` follows the Stage 1 rule --
        the *smallest* force whose mean ties with the maximum, not the argmax,
        plus the margin -- so the choice is biased toward less pressure on the
        patient without sitting on the edge of contact loss.
    """
    levels: list[ForceLevelSummary] = []
    for force in sorted(samples_by_force):
        raw = list(samples_by_force[force])
        usable = [
            float(v) for v in raw if v is not None and math.isfinite(float(v))
        ]
        valid_fraction = (len(usable) / len(raw)) if raw else 0.0
        if not usable or valid_fraction < min_valid_fraction:
            levels.append(
                ForceLevelSummary(
                    force=float(force),
                    n=len(usable),
                    mean=None,
                    std=None,
                    sem=None,
                    valid_fraction=valid_fraction,
                )
            )
            continue
        array = np.asarray(usable, dtype=np.float64)
        n = int(array.size)
        std = float(array.std(ddof=1)) if n > 1 else 0.0
        rho = _lag1_autocorrelation(array)
        if autocorr_correct and n >= 3 and rho > -1.0:
            n_eff = float(min(max(n * (1.0 - rho) / (1.0 + rho), 2.0), float(n)))
        else:
            n_eff = float(n)
        levels.append(
            ForceLevelSummary(
                force=float(force),
                n=n,
                mean=float(array.mean()),
                std=std,
                sem=(std / math.sqrt(n_eff)) if n > 1 else 0.0,
                valid_fraction=valid_fraction,
                rho=rho,
                n_eff=n_eff,
            )
        )

    measured = [level for level in levels if level.mean is not None]
    if len(measured) < 2:
        return ForceResponseReport(
            levels=levels,
            epsilon=float(epsilon),
            margin_n=float(margin_n),
            tie_rule="epsilon" if z_alpha is None else "welch",
            z_alpha=None if z_alpha is None else float(z_alpha),
            unimodal_ratio_max=float(unimodal_ratio_max),
        )

    forces = np.asarray([level.force for level in measured], dtype=np.float64)
    means = np.asarray([float(level.mean) for level in measured], dtype=np.float64)
    sems = np.asarray([float(level.sem) for level in measured], dtype=np.float64)

    differences = np.diff(means)
    monotone_fraction = float(np.count_nonzero(differences >= 0) / differences.size)

    # Legacy shape test: the difference sequence changes sign at most once
    # (rises then falls). Steps smaller than epsilon are noise, not turns.
    significant = differences[np.abs(differences) > epsilon]
    signs = np.sign(significant)
    sign_changes = int(np.count_nonzero(np.diff(signs) != 0)) if signs.size > 1 else 0

    # Umbrella fit: how far are the level means from the nearest rise-then-fall
    # sequence, relative to their own standard errors?
    fit, peak_index, sse = unimodal_regression(means)
    if np.all(sems > 0.0):
        unimodal_sse_ratio: Optional[float] = float((sse / means.size) / float(np.mean(sems**2)))
        is_unimodal = bool(unimodal_sse_ratio <= unimodal_ratio_max)
    else:
        unimodal_sse_ratio = None
        is_unimodal = bool(sign_changes <= 1)

    # Tie set with the maximum, then the smallest force in it.
    argmax = int(np.argmax(means))
    peak = float(means[argmax])
    if z_alpha is None:
        tie = means >= peak - epsilon
        tie_rule = "epsilon"
    else:
        tolerance = float(z_alpha) * np.sqrt(sems**2 + sems[argmax] ** 2)
        tie = means >= peak - tolerance
        tie_rule = "welch"
    f_left = float(forces[tie].min())

    return ForceResponseReport(
        levels=levels,
        spearman_with_force=spearman(forces, means),
        monotone_fraction=monotone_fraction,
        sign_changes=sign_changes,
        is_unimodal=is_unimodal,
        argmax_force=float(forces[argmax]),
        f_star=f_left + float(margin_n),
        epsilon=float(epsilon),
        f_left=f_left,
        margin_n=float(margin_n),
        tie_rule=tie_rule,
        z_alpha=None if z_alpha is None else float(z_alpha),
        unimodal_sse_ratio=unimodal_sse_ratio,
        unimodal_ratio_max=float(unimodal_ratio_max),
        unimodal_fit=[float(v) for v in fit],
        unimodal_peak_force=float(forces[peak_index]),
    )


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


@dataclass
class TrustReport:
    """Everything :func:`build_trust_report` computed, in one serialisable object."""

    name: str
    accuracy_floor: float
    agreement: AgreementReport
    gate: GateConfusion
    detection: DetectionReport
    risk: RiskCoverageReport
    calibration: CalibrationReport
    operating: OperatingPoint
    components: list[ComponentAttribution] = field(default_factory=list)
    reasons: list[ReasonCodeStat] = field(default_factory=list)
    per_patient: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "accuracy_floor": self.accuracy_floor,
            "agreement": self.agreement.to_dict(),
            "gate": self.gate.to_dict(),
            "detection": self.detection.to_dict(),
            "risk_coverage": self.risk.to_dict(),
            "calibration": self.calibration.to_dict(),
            "operating_point": self.operating.to_dict(),
            "components": [c.to_dict() for c in self.components],
            "reason_codes": [r.to_dict() for r in self.reasons],
            "per_patient": self.per_patient,
        }

    def summary_lines(self) -> list[str]:
        """A few lines a human can read without opening the JSON."""
        rho = self.agreement.spearman
        lines = [
            f"[{self.name}]  n={self.agreement.n} labeled frames, "
            f"floor Dice>={self.accuracy_floor:.2f}",
            f"  rank agreement   rho = {rho:.3f}" if rho is not None else
            "  rank agreement   rho = n/a",
            f"  gate             trusted-bad {_pct(self.gate.trusted_bad_rate)} "
            f"at coverage {_pct(self.gate.coverage)}",
            f"  discrimination   AUROC = {_num(self.detection.auroc)}   "
            f"AURC = {_num(self.risk.aurc)}",
        ]
        if self.operating.feasible:
            lines.append(
                f"  operating point  Q >= {self.operating.threshold:.3f} -> "
                f"coverage {_pct(self.operating.coverage)}, "
                f"bad {_pct(self.operating.achieved_bad_rate)}"
            )
        else:
            lines.append(
                f"  operating point  NONE meets a {_pct(self.operating.target_bad_rate)} "
                "trusted-bad budget"
            )
        return lines


def _num(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def build_trust_report(
    outcomes: Sequence[FrameOutcome],
    name: str = "Q_seg",
    accuracy_floor: float = DEFAULT_ACCURACY_FLOOR,
    weights: Optional[Mapping[str, float]] = None,
    target_bad_rate: float = 0.02,
    n_bins: int = 10,
) -> TrustReport:
    """Run every trust analysis over one set of frames.

    Args:
        outcomes: Per-frame quality, gate decision and ground-truth accuracy.
        name: Which score this is, for the report header.
        accuracy_floor: Dice defining "good enough to act on". A decision.
        weights: Sub-score weights, enabling :func:`component_attribution`.
        target_bad_rate: Safety budget for :func:`operating_point`.
        n_bins: Calibration bins.
    """
    report = TrustReport(
        name=name,
        accuracy_floor=float(accuracy_floor),
        agreement=rank_agreement(outcomes),
        gate=gate_confusion(outcomes, accuracy_floor),
        detection=detection_report(outcomes, accuracy_floor),
        risk=risk_coverage(outcomes, accuracy_floor),
        calibration=calibration(outcomes, n_bins=n_bins),
        operating=operating_point(outcomes, target_bad_rate, accuracy_floor),
        components=component_attribution(outcomes, weights) if weights else [],
        reasons=reason_code_report(outcomes, accuracy_floor),
    )

    # Per patient as well as pooled: frames within a sweep are correlated, so a
    # pooled number hides "works on four patients, fails on the fifth".
    for patient, frames in sorted(group_by(outcomes, "patient_id").items()):
        if not patient:
            continue
        report.per_patient[patient] = {
            "agreement": rank_agreement(frames).to_dict(),
            "gate": gate_confusion(frames, accuracy_floor).to_dict(),
        }
    return report
