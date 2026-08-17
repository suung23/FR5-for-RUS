"""Segmentation-independent raw B-mode image quality (``Q_raw``).

This is the **second** of the two image-quality functions in this repository,
and the counterpart to :mod:`rus_perception.control.quality`:

===============  ===========================  ==================================
score            derived from                 what the controller optimises with it
===============  ===========================  ==================================
``Q_raw``        the raw B-mode frame only    the force axes (``z``, ``rx``, ``ry``)
``Q_seg``        the segmentation output      the in-plane axes (``x``, ``y``, ``rz``)
===============  ===========================  ==================================

Both live here for one reason: they share the acquisition geometry. The
fan/sector ROI mask, the depth-band split and the A-line sampling convention are
properties of *the ultrasound machine*, not of either score, and defining them
in two repositories guarantees they drift apart.

``Q_raw`` exists because every quantity in a :class:`~rus_perception.control.state.ControlState`
presupposes that the bladder was found. At the start of a contact search it has
not been, ``valid_for_control`` is ``False`` and ``Q_seg`` is undefined -- there
is nothing to optimise. ``Q_raw`` answers the strictly earlier question: *is the
probe acoustically coupled to the tissue at all?* It uses no network, no mask
and no temporal history, which is also why it is more robust in exactly the
regime where the network is least trustworthy.

Status:
    **Skeleton.** The structure, the configuration schema and the reason codes
    are final; every numeric default below is provisional and marked ``PROVISIONAL``.
    They cannot be fixed until the ultrasound image geometry is known (probe
    type, depth scale, fan ROI). None of the four sub-scores has been shown to
    be monotone or unimodal in contact force -- that is an experiment, not an
    assumption, and until it is run ``Q_raw`` must not be trusted as a search
    objective.

Warning:
    Like ``Q_seg``, this is a transparent heuristic, not a validated measure of
    diagnostic image quality, and it emits no robot command of any kind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

__all__ = [
    "RAW_QUALITY_COMPONENT_NAMES",
    "RAW_REJECTION_REASONS",
    "RawQualityConfig",
    "RawQualityResult",
    "ScanGeometryError",
    "compute_raw_quality",
    "sample_a_lines",
]

#: Every sub-score name ``Q_raw`` can emit, in a fixed order, so that a CSV
#: column set stays stable across frames. Mirrors ``QUALITY_COMPONENT_NAMES``.
RAW_QUALITY_COMPONENT_NAMES: tuple[str, ...] = (
    "near_field_echo",
    "contact_continuity",
    "total_echo_energy",
    "shadow_penalty",
)

#: Machine-readable reasons a frame is unusable as a *contact* measurement.
#:
#: These are deliberately disjoint from
#: :data:`rus_perception.control.validity.REJECTION_REASONS`, which describes
#: ways a *segmentation* is unusable. A supervisor consumes the union of the two
#: sets; nothing here ever appears in a ``ControlState.rejection_reasons``.
RAW_REJECTION_REASONS: tuple[str, ...] = (
    "no_contact",
    "poor_acoustic_coupling",
    "excessive_shadowing",
    "low_near_field_echo",
)

_EPS = 1e-8


class ScanGeometryError(NotImplementedError):
    """Raised for a scan geometry whose A-line sampling is not implemented."""


@dataclass
class RawQualityConfig:
    """Weights, band definitions and thresholds of ``Q_raw``.

    Every value is a **PROVISIONAL** default. The depth fractions, the darkness
    threshold and the reference intensity all depend on the ultrasound machine's
    depth scale, gain, TGC and dynamic range, none of which is known yet.

    Attributes:
        weights: Weight per sub-score; ``0`` removes a component.
        scan_geometry: ``"linear"`` -- image columns are A-lines. ``"sector"``
            (curvilinear/phased, scan-converted) is **not implemented**: after
            scan conversion an A-line is a ray from the apex, not a column, and
            treating columns as A-lines would silently mix depths.
        near_field_fraction: Shallowest fraction of the depth range forming the
            near-field band. The probe face is at the top of the image.
        far_field_fraction: Deepest fraction of the depth range forming the
            far-field band, used for the shadow test.
        near_field_reference: Near-field mean intensity that scores ``1.0``.
        dark_intensity_threshold: Near-field mean below which an A-line counts
            as *not coupled* (an air gap kills the whole line).
        shadow_relative_threshold: An A-line is shadowed when its far-field mean
            falls below this fraction of the frame's unshadowed far-field level
            (see ``shadow_reference_percentile``). Relative rather than absolute
            so the test survives a machine-side gain or TGC change.
        shadow_reference_percentile: Percentile of the per-A-line far-field
            means taken as "what an unshadowed line looks like in this frame".
            A median (50) would be wrong: once more than half the lines are
            shadowed the median *is* the shadow level and the test reports
            nothing. A high percentile stays correct until nearly every line is
            shadowed, which is the case the absolute floor below catches.
        shadow_absolute_floor: Backstop. An A-line also counts as shadowed when
            its far-field mean falls below this absolute intensity. A purely
            relative test cannot see *uniform* shadowing -- if every line dies
            equally there is no unshadowed reference left -- so this floor is
            the only thing that catches total far-field loss. It is the one
            gain-dependent threshold here; keep it low enough to mean "no echo
            returned at all", not "a dim preset".
        target_mean_intensity: ROI mean intensity considered ideal.
        intensity_tolerance: Half-width of the plateau around
            ``target_mean_intensity``.
        min_valid_a_lines: Fewer usable A-lines than this and the frame yields
            no score at all, rather than a score computed from a sliver.
        max_dark_a_line_ratio: Above this dark-A-line ratio the frame is
            rejected with ``poor_acoustic_coupling``.
        max_shadowed_a_line_ratio: Above this shadowed-A-line ratio the frame is
            rejected with ``excessive_shadowing``.
        min_near_field_echo: Below this ``near_field_echo`` sub-score the frame
            is rejected with ``low_near_field_echo``.
        no_contact_intensity: ROI mean below which the probe is considered to be
            in air entirely (``no_contact``).
    """

    weights: dict[str, float] = field(
        default_factory=lambda: {
            # PROVISIONAL. contact_continuity carries the most weight because a
            # dead A-line is the least ambiguous evidence of an air gap; every
            # other sub-score can be depressed by anatomy alone.
            "near_field_echo": 1.0,
            "contact_continuity": 1.5,
            "total_echo_energy": 0.5,
            "shadow_penalty": 1.0,
        }
    )

    scan_geometry: Literal["linear", "sector"] = "linear"  # PROVISIONAL

    # -- band definitions (PROVISIONAL; see README "Benchmark status") --------
    near_field_fraction: float = 0.15
    far_field_fraction: float = 0.35

    # -- normalisation (PROVISIONAL; depends on gain/TGC/dynamic range) ------
    near_field_reference: float = 0.45
    dark_intensity_threshold: float = 0.10
    shadow_relative_threshold: float = 0.25
    shadow_reference_percentile: float = 90.0
    shadow_absolute_floor: float = 0.02
    target_mean_intensity: float = 0.30
    intensity_tolerance: float = 0.12

    # -- gate thresholds (PROVISIONAL) ---------------------------------------
    min_valid_a_lines: int = 16
    max_dark_a_line_ratio: float = 0.35
    max_shadowed_a_line_ratio: float = 0.50
    min_near_field_echo: float = 0.25
    no_contact_intensity: float = 0.05

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.weights.values()):
            raise ValueError(f"Raw quality weights must be non-negative, got {self.weights}.")
        if sum(self.weights.values()) <= 0:
            raise ValueError("At least one raw quality weight must be positive.")
        if self.scan_geometry not in ("linear", "sector"):
            raise ValueError(
                f"scan_geometry must be 'linear' or 'sector', got {self.scan_geometry!r}."
            )
        for name in ("near_field_fraction", "far_field_fraction"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}.")
        if self.near_field_fraction + self.far_field_fraction > 1.0:
            raise ValueError(
                "near_field_fraction + far_field_fraction must not exceed 1.0; the "
                f"bands would overlap ({self.near_field_fraction} + {self.far_field_fraction})."
            )
        for name in (
            "near_field_reference",
            "target_mean_intensity",
            "intensity_tolerance",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}.")
        if self.min_valid_a_lines < 1:
            raise ValueError(f"min_valid_a_lines must be >= 1, got {self.min_valid_a_lines}.")
        if not 50.0 <= self.shadow_reference_percentile <= 100.0:
            raise ValueError(
                "shadow_reference_percentile must be in [50, 100] -- below 50 the "
                "reference would be taken from the shadowed lines themselves, got "
                f"{self.shadow_reference_percentile}."
            )
        if self.shadow_absolute_floor < 0:
            raise ValueError(
                f"shadow_absolute_floor must be >= 0, got {self.shadow_absolute_floor}."
            )

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "RawQualityConfig":
        """Build from a config mapping, rejecting unknown keys."""
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown raw quality config key(s): {sorted(unknown)}. "
                f"Known keys: {sorted(known)}"
            )
        if "weights" in data and data["weights"] is not None:
            base = cls().weights
            unknown_weights = set(data["weights"]) - set(base)
            if unknown_weights:
                raise ValueError(
                    f"Unknown raw quality weight(s): {sorted(unknown_weights)}. "
                    f"Known: {sorted(base)}"
                )
            base.update({k: float(v) for k, v in data["weights"].items()})
            data["weights"] = base
        return cls(**data)


@dataclass
class RawQualityResult:
    """``Q_raw`` together with every sub-score and diagnostic that produced it.

    Attributes:
        score: The weighted mean in ``[0, 1]``, or ``None`` when the frame
            yielded too few usable A-lines to score at all. ``None`` means
            *not measured*, never *bad*.
        components: Sub-score values actually used.
        weights: Weight applied to each of those sub-scores.
        rejection_reasons: Codes from :data:`RAW_REJECTION_REASONS`; empty when
            the frame is usable as a contact measurement.
        dark_a_line_ratio: Fraction of A-lines whose near field is dark.
        shadowed_a_line_ratio: Fraction of A-lines whose far field collapsed.
        near_field_mean: Mean intensity of the near-field band inside the ROI.
        far_field_mean: Mean intensity of the far-field band inside the ROI.
        roi_mean: Mean intensity over the whole ROI.
        a_line_count: Number of A-lines that had enough ROI support to be used.
    """

    score: Optional[float]
    components: dict[str, float]
    weights: dict[str, float]
    rejection_reasons: list[str] = field(default_factory=list)
    dark_a_line_ratio: Optional[float] = None
    shadowed_a_line_ratio: Optional[float] = None
    near_field_mean: Optional[float] = None
    far_field_mean: Optional[float] = None
    roi_mean: Optional[float] = None
    a_line_count: int = 0

    @property
    def usable_for_contact_search(self) -> bool:
        """True when a score was produced and nothing rejected the frame.

        This is an *observation quality* verdict, exactly like
        ``valid_for_control``. It is not a command, and it says nothing about
        what the controller should do next.
        """
        return self.score is not None and not self.rejection_reasons

    def explain(self) -> str:
        """Render the weighted-mean computation as readable text."""
        head = "not measured" if self.score is None else f"{self.score:.4f}"
        lines = [f"raw_quality_score = {head} (weighted mean of:)"]
        for name in RAW_QUALITY_COMPONENT_NAMES:
            if name not in self.components:
                continue
            weight = self.weights.get(name, 0.0)
            lines.append(f"  {name:<22} score={self.components[name]:.4f} weight={weight:.3f}")
        if self.rejection_reasons:
            lines.append(f"  rejected: {', '.join(self.rejection_reasons)}")
        return "\n".join(lines)


def _plateau(value: float, target: float, tolerance: float) -> float:
    """1 inside ``target +/- tolerance``, decaying smoothly outside it.

    Deliberately identical to ``quality._plateau`` so the two quality functions
    normalise the same way; kept local so neither module depends on the other.
    """
    distance = abs(value - target)
    if distance <= tolerance:
        return 1.0
    scale = max(tolerance, 1e-6)
    return float(min(1.0, max(0.0, math.exp(-(distance - tolerance) / scale))))


def sample_a_lines(
    image: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    scan_geometry: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the image resampled into ``depth x a_line`` form, plus its support.

    An **A-line** is one acoustic ray: a single transducer element's echo as a
    function of depth. Every ``Q_raw`` sub-score is defined per A-line because
    that is the unit in which acoustic coupling fails -- an air gap kills one
    ray completely rather than dimming the whole image.

    Args:
        image: ``H x W`` B-mode frame in ``[0, 1]``, probe face at row 0.
        roi_mask: Optional ``H x W`` boolean mask of the imaged sector. Pixels
            outside it are excluded from every statistic.
        scan_geometry: ``"linear"`` -- columns are A-lines, so the image is
            already in ``depth x a_line`` form and is returned as-is.
            ``"sector"`` raises :class:`ScanGeometryError`.

    Returns:
        ``(samples, support)``: the ``depth x a_line`` intensities, and a
        boolean array of the same shape marking which entries are inside the ROI.

    Raises:
        ValueError: If ``image`` is not 2-D or the mask shape disagrees.
        ScanGeometryError: For ``scan_geometry="sector"``.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"image must be 2-D H x W, got shape {array.shape}.")

    if scan_geometry == "sector":
        raise ScanGeometryError(
            "Sector/curvilinear A-line sampling is not implemented. After scan "
            "conversion an A-line is a ray from the virtual apex, not an image "
            "column, so treating columns as A-lines would average across "
            "different depths. Implementing it needs the apex position, the "
            "sweep angle and the depth scale from the ultrasound machine."
        )
    if scan_geometry != "linear":
        raise ValueError(f"Unknown scan_geometry {scan_geometry!r}.")

    if roi_mask is None:
        support = np.ones_like(array, dtype=bool)
    else:
        support = np.asarray(roi_mask).astype(bool)
        if support.shape != array.shape:
            raise ValueError(
                f"roi_mask shape {support.shape} does not match image {array.shape}."
            )
    return array, support


def _band_means(
    samples: np.ndarray, support: np.ndarray, start: int, stop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-A-line mean over depth rows ``[start, stop)``, and its support count."""
    band = samples[start:stop, :]
    band_support = support[start:stop, :]
    counts = band_support.sum(axis=0)
    totals = np.where(band_support, band, 0.0).sum(axis=0)
    means = np.divide(totals, np.maximum(counts, 1), dtype=np.float32)
    return means.astype(np.float32), counts


def compute_raw_quality(
    image: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    config: Optional[RawQualityConfig] = None,
) -> RawQualityResult:
    """Compute ``Q_raw`` from a single raw B-mode frame.

    Uses no network, no segmentation mask and no previous frame, by design: it
    has to work in the regime where the segmentation does not yet produce
    anything usable.

    Args:
        image: ``H x W`` grayscale B-mode frame in ``[0, 1]``, probe face at
            row 0 and depth increasing downward.
        roi_mask: Optional ``H x W`` boolean mask of the imaged sector.
        config: Weights, bands and thresholds.

    Returns:
        A :class:`RawQualityResult`. ``score`` is ``None`` -- *not measured*,
        never ``0.0`` -- when too few A-lines had ROI support.

    Raises:
        ValueError: If ``image`` is not 2-D or leaves ``[0, 1]``.
        ScanGeometryError: For an unimplemented scan geometry.
    """
    config = config or RawQualityConfig()
    samples, support = sample_a_lines(image, roi_mask, config.scan_geometry)

    if samples.size and (
        float(np.nanmin(samples)) < -1e-6 or float(np.nanmax(samples)) > 1.0 + 1e-6
    ):
        raise ValueError(
            "image must be normalized to [0, 1] before computing Q_raw; got range "
            f"[{float(np.nanmin(samples)):.4f}, {float(np.nanmax(samples)):.4f}]."
        )

    depth = samples.shape[0]
    near_stop = max(1, int(round(depth * config.near_field_fraction)))
    far_start = min(depth - 1, depth - max(1, int(round(depth * config.far_field_fraction))))

    near_means, near_counts = _band_means(samples, support, 0, near_stop)
    far_means, far_counts = _band_means(samples, support, far_start, depth)

    # An A-line is usable only where BOTH bands have ROI support: the near field
    # decides coupling and the far field decides shadowing, and a line missing
    # either would contribute an unbalanced vote.
    usable = (near_counts > 0) & (far_counts > 0)
    a_line_count = int(usable.sum())

    roi_total = float(np.where(support, samples, 0.0).sum())
    roi_count = int(support.sum())
    roi_mean = float(roi_total / roi_count) if roi_count else None

    if a_line_count < config.min_valid_a_lines:
        # Not measurable. Report the fact rather than inventing a score.
        return RawQualityResult(
            score=None,
            components={},
            weights={},
            rejection_reasons=[],
            roi_mean=roi_mean,
            a_line_count=a_line_count,
        )

    near_usable = near_means[usable]
    far_usable = far_means[usable]

    near_field_mean = float(near_usable.mean())
    far_field_mean = float(far_usable.mean())

    dark_a_line_ratio = float((near_usable < config.dark_intensity_threshold).mean())

    # Shadowing is judged RELATIVE to this frame's own unshadowed far-field
    # level, so a global gain or TGC change moves every A-line together and
    # cancels out. A purely absolute threshold would report "excessive
    # shadowing" for a correctly coupled probe on a low-gain preset.
    #
    # The reference is a HIGH PERCENTILE, not the median: with more than half
    # the lines shadowed the median is itself the shadow level and the relative
    # test goes blind exactly when shadowing is worst.
    far_reference = float(np.percentile(far_usable, config.shadow_reference_percentile))
    relative_shadow = far_usable < config.shadow_relative_threshold * far_reference
    # Backstop for *uniform* shadowing, where no unshadowed reference survives.
    absolute_shadow = far_usable < config.shadow_absolute_floor
    shadowed_a_line_ratio = float((relative_shadow | absolute_shadow).mean())

    components: dict[str, float] = {
        "near_field_echo": float(
            min(1.0, max(0.0, near_field_mean / config.near_field_reference))
        ),
        "contact_continuity": float(1.0 - dark_a_line_ratio),
        "total_echo_energy": _plateau(
            roi_mean if roi_mean is not None else 0.0,
            config.target_mean_intensity,
            config.intensity_tolerance,
        ),
        "shadow_penalty": float(1.0 - shadowed_a_line_ratio),
    }

    weights = {
        name: float(config.weights.get(name, 0.0))
        for name in components
        if config.weights.get(name, 0.0) > 0
    }
    if weights:
        total_weight = sum(weights.values())
        score = sum(components[name] * weight for name, weight in weights.items()) / total_weight
        score = float(min(1.0, max(0.0, score)))
    else:
        score = 0.0

    reasons: list[str] = []
    if roi_mean is not None and roi_mean < config.no_contact_intensity:
        reasons.append("no_contact")
    if dark_a_line_ratio > config.max_dark_a_line_ratio:
        reasons.append("poor_acoustic_coupling")
    if shadowed_a_line_ratio > config.max_shadowed_a_line_ratio:
        reasons.append("excessive_shadowing")
    if components["near_field_echo"] < config.min_near_field_echo:
        reasons.append("low_near_field_echo")

    return RawQualityResult(
        score=score,
        components=components,
        weights=weights,
        rejection_reasons=reasons,
        dark_a_line_ratio=dark_a_line_ratio,
        shadowed_a_line_ratio=shadowed_a_line_ratio,
        near_field_mean=near_field_mean,
        far_field_mean=far_field_mean,
        roi_mean=roi_mean,
        a_line_count=a_line_count,
    )