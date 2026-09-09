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
    **Structure final, numbers provisional.** The configuration schema, the
    reason codes and both A-line samplers are in place: ``linear`` returns the
    image columns, and ``sector`` (curvilinear/phased, scan-converted) is
    implemented as bin-based sampling on the same fan geometry that
    :mod:`rus_perception.control.roi` uses, so the two cannot drift apart. The
    sector sampler still needs the machine's apex position, radius range and
    sweep angle (``fan``) before it measures anything. Every numeric default
    below is provisional and marked ``PROVISIONAL``; none can be fixed until
    the ultrasound image geometry, gain and TGC are known. None of the four
    sub-scores has been shown to be monotone or unimodal in contact force --
    that is an experiment, not an assumption. Since the 2026-09-08 revision
    ``Q_raw`` is no longer asked to be a search objective at all: Stage 1a uses
    it as a *coupling gate* (:func:`coupling_gate`), and the force search runs
    on ``Q_seg`` in Stage 1b (DESIGN_NOTES §7.4).

Warning:
    Like ``Q_seg``, this is a transparent heuristic, not a validated measure of
    diagnostic image quality, and it emits no robot command of any kind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Mapping, Optional

import numpy as np

__all__ = [
    "RAW_QUALITY_COMPONENT_NAMES",
    "RAW_REJECTION_REASONS",
    "RawQualityConfig",
    "RawQualityResult",
    "ScanGeometryError",
    "compute_raw_quality",
    "coupling_gate",
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
    """Raised for a scan geometry whose A-line sampling is not implemented.

    Both shipped geometries (``linear`` and ``sector``) are implemented, so
    :func:`sample_a_lines` no longer raises this; it is kept so that callers
    written against the earlier skeleton keep importing.
    """


#: The three fan parameters ``sector`` sampling needs, in the names ``roi.py``
#: uses, so one YAML block can feed both the ROI mask and the A-line sampler.
_FAN_KEYS: tuple[str, ...] = ("apex_xy", "radius_range", "half_angle_deg")


def _fan_parameters(fan: Any) -> tuple[float, float, float, float, float]:
    """Return ``(apex_x, apex_y, r_min, r_max, half_angle_deg)`` from ``fan``.

    ``fan`` is a mapping with the keys in :data:`_FAN_KEYS`, or any object that
    carries them as attributes (a :class:`~rus_perception.control.roi.RoiConfig`
    with ``mode="fan"``). Validation mirrors ``RoiConfig.__post_init__`` so the
    same geometry is accepted and rejected in both places.

    Raises:
        ValueError: If ``fan`` is missing, incomplete or out of range.
    """
    if fan is None:
        raise ValueError(
            "scan_geometry='sector' requires the fan geometry: a mapping with "
            f"{list(_FAN_KEYS)} (the same values as roi.mode='fan'). After scan "
            "conversion an A-line is a ray from the virtual apex, not an image "
            "column, so the apex position, radius range and sweep angle must come "
            "from the ultrasound machine and cannot be guessed."
        )
    if isinstance(fan, Mapping):
        values = {key: fan.get(key) for key in _FAN_KEYS}
    else:
        values = {key: getattr(fan, key, None) for key in _FAN_KEYS}
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(
            f"fan geometry is missing {missing}; scan_geometry='sector' needs all of "
            f"{list(_FAN_KEYS)}."
        )
    apex = tuple(float(v) for v in values["apex_xy"])
    if len(apex) != 2:
        raise ValueError(f"fan.apex_xy must be (x, y) fractions, got {values['apex_xy']!r}.")
    radii = tuple(float(v) for v in values["radius_range"])
    if len(radii) != 2 or not 0.0 <= radii[0] < radii[1]:
        raise ValueError(
            f"fan.radius_range must satisfy 0 <= r_min < r_max, got {values['radius_range']!r}."
        )
    half_angle = float(values["half_angle_deg"])
    if not 0.0 < half_angle <= 90.0:
        raise ValueError(f"fan.half_angle_deg must be in (0, 90], got {half_angle}.")
    return apex[0], apex[1], radii[0], radii[1], half_angle


@dataclass
class RawQualityConfig:
    """Weights, band definitions and thresholds of ``Q_raw``.

    Every value is a **PROVISIONAL** default. The depth fractions, the darkness
    threshold and the reference intensity all depend on the ultrasound machine's
    depth scale, gain, TGC and dynamic range, none of which is known yet.

    Attributes:
        weights: Weight per sub-score; ``0`` removes a component.
        scan_geometry: ``"linear"`` -- image columns are A-lines. ``"sector"``
            (curvilinear/phased, scan-converted) -- A-lines are rays from the
            virtual apex and are resampled by binning pixels in ``(radius,
            angle)``; requires ``fan``. Treating a sector image's columns as
            A-lines would silently mix depths.
        fan: Fan geometry for ``"sector"``: a mapping with ``apex_xy`` (``x``,
            ``y`` fractions of width and height; ``y`` may be negative),
            ``radius_range`` (``r_min``, ``r_max`` fractions of the height
            measured from the apex) and ``half_angle_deg`` -- the same three
            values ``roi.mode="fan"`` takes, so one block configures both.
        n_a_lines: Number of angular bins (A-lines) for ``"sector"``.
        depth_bins: Number of radial bins for ``"sector"``. ``None`` uses one
            bin per pixel of radial extent (``round((r_max - r_min) * H)``,
            at least 8).
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
        gate_min_contact_continuity: Lowest ``contact_continuity`` at which
            :func:`coupling_gate` still reports the probe as coupled. Stricter
            than ``max_dark_a_line_ratio`` on purpose: the rejection reason
            says the frame is *unusable as a measurement*, the gate says the
            contact is *good enough to hand over to Stage 1b*.
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
    # -- sector sampling (needs the machine's fan geometry; see roi.py) -------
    fan: Optional[dict[str, Any]] = None
    n_a_lines: int = 64
    depth_bins: Optional[int] = None

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

    # -- Stage 1a coupling gate (PROVISIONAL) --------------------------------
    gate_min_contact_continuity: float = 0.8

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.weights.values()):
            raise ValueError(f"Raw quality weights must be non-negative, got {self.weights}.")
        if sum(self.weights.values()) <= 0:
            raise ValueError("At least one raw quality weight must be positive.")
        if self.scan_geometry not in ("linear", "sector"):
            raise ValueError(
                f"scan_geometry must be 'linear' or 'sector', got {self.scan_geometry!r}."
            )
        if self.scan_geometry == "sector":
            # Validate now, with the same rules as roi.py, so a bad geometry
            # fails at configuration time rather than on the first frame.
            _fan_parameters(self.fan)
        if self.n_a_lines < 1:
            raise ValueError(f"n_a_lines must be >= 1, got {self.n_a_lines}.")
        if self.depth_bins is not None and self.depth_bins < 1:
            raise ValueError(f"depth_bins must be >= 1 or None, got {self.depth_bins}.")
        if not 0.0 <= self.gate_min_contact_continuity <= 1.0:
            raise ValueError(
                "gate_min_contact_continuity must be in [0, 1], got "
                f"{self.gate_min_contact_continuity}."
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
        if "fan" in data and data["fan"] is not None:
            fan = dict(data["fan"])
            unknown_fan = set(fan) - set(_FAN_KEYS)
            if unknown_fan:
                raise ValueError(
                    f"Unknown raw quality fan key(s): {sorted(unknown_fan)}. "
                    f"Known: {list(_FAN_KEYS)} (the roi.mode='fan' parameters)."
                )
            data["fan"] = {
                key: (list(value) if isinstance(value, (list, tuple)) else value)
                for key, value in fan.items()
            }
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


@lru_cache(maxsize=8)
def _sector_cells(
    shape: tuple[int, int],
    apex_x: float,
    apex_y: float,
    r_min: float,
    r_max: float,
    half_angle_deg: float,
    n_a_lines: int,
    depth_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel ``(inside, cell)`` for one fan geometry on one frame shape.

    ``inside`` is the fan itself -- exactly :func:`roi._fan_mask` -- and ``cell``
    the flat ``depth_bin * n_a_lines + angle_bin`` index of every pixel. The
    geometry depends only on the frame shape and the fan parameters, never on
    the frame content, so it is computed once per configuration and cached:
    the real-time path pays for one boolean index and two ``bincount`` calls
    per frame.
    """
    height, width = shape
    cx = apex_x * width
    cy = apex_y * height
    r_min_px = r_min * height
    r_max_px = r_max * height
    half_angle = math.radians(half_angle_deg)

    ys, xs = np.mgrid[0:height, 0:width]
    dx = xs.astype(np.float64) + 0.5 - cx
    dy = ys.astype(np.float64) + 0.5 - cy

    radius = np.hypot(dx, dy)
    # atan2(dx, dy): signed angle away from the downward (depth) axis. Kept
    # signed here, unlike the ROI mask, because the sign is the A-line index.
    angle = np.arctan2(dx, np.maximum(dy, 1e-9))

    inside = (radius >= r_min_px) & (radius <= r_max_px)
    inside &= np.abs(angle) <= half_angle
    # Everything at or above the apex is behind the transducer face.
    inside &= dy > 0.0

    # Equal-width bins; the closed upper edge of the last bin is folded back
    # so a pixel exactly on r_max or +half_angle is not dropped.
    depth_index = np.floor((radius - r_min_px) / (r_max_px - r_min_px) * depth_bins)
    depth_index = np.clip(depth_index, 0, depth_bins - 1).astype(np.intp)
    angle_index = np.floor((angle + half_angle) / (2.0 * half_angle) * n_a_lines)
    angle_index = np.clip(angle_index, 0, n_a_lines - 1).astype(np.intp)

    cell = depth_index * n_a_lines + angle_index
    inside.setflags(write=False)
    cell.setflags(write=False)
    return inside, cell


def _sample_sector(
    array: np.ndarray,
    roi_mask: Optional[np.ndarray],
    fan: Any,
    n_a_lines: int,
    depth_bins: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a scan-converted sector image into ``depth_bins x n_a_lines`` cells."""
    apex_x, apex_y, r_min, r_max, half_angle_deg = _fan_parameters(fan)
    if n_a_lines < 1:
        raise ValueError(f"n_a_lines must be >= 1, got {n_a_lines}.")
    height, width = array.shape
    if depth_bins is None:
        depth_bins = max(8, int(round((r_max - r_min) * height)))
    elif depth_bins < 1:
        raise ValueError(f"depth_bins must be >= 1 or None, got {depth_bins}.")

    inside, cell = _sector_cells(
        (int(height), int(width)),
        apex_x,
        apex_y,
        r_min,
        r_max,
        half_angle_deg,
        int(n_a_lines),
        int(depth_bins),
    )
    keep = inside
    if roi_mask is not None:
        mask = np.asarray(roi_mask).astype(bool)
        if mask.shape != array.shape:
            raise ValueError(
                f"roi_mask shape {mask.shape} does not match image {array.shape}."
            )
        keep = inside & mask

    n_cells = int(depth_bins) * int(n_a_lines)
    flat = cell[keep]
    counts = np.bincount(flat, minlength=n_cells)
    totals = np.bincount(flat, weights=array[keep].astype(np.float64), minlength=n_cells)
    samples = (totals / np.maximum(counts, 1)).reshape(int(depth_bins), int(n_a_lines))
    support = (counts > 0).reshape(int(depth_bins), int(n_a_lines))
    return samples.astype(np.float32), support


def sample_a_lines(
    image: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    scan_geometry: str = "linear",
    *,
    fan: Any = None,
    n_a_lines: int = 64,
    depth_bins: Optional[int] = None,
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
            ``"sector"`` -- A-lines are rays from the virtual apex. Each pixel
            is assigned a radius and a signed angle in the fan geometry of
            :func:`rus_perception.control.roi.build_roi_mask` (``mode="fan"``),
            the fan is cut into ``depth_bins x n_a_lines`` equal cells in
            ``(radius, angle)``, and every cell takes the mean of its pixels. No
            interpolation: a cell with no pixel is simply unsupported, which is
            what happens near the transducer face where cells are sub-pixel.
        fan: Fan geometry for ``"sector"``: a mapping (or a ``RoiConfig``) with
            ``apex_xy`` -- ``(x, y)`` fractions of ``(W, H)``, ``y`` may be
            negative; ``radius_range`` -- ``(r_min, r_max)`` fractions of ``H``
            measured from the apex, ``r_min`` is the transducer face; and
            ``half_angle_deg``. Ignored for ``"linear"``.
        n_a_lines: Number of angular bins over ``[-half_angle, +half_angle]``
            for ``"sector"``.
        depth_bins: Number of radial bins over ``[r_min, r_max]`` for
            ``"sector"``. ``None`` uses roughly one bin per pixel of radial
            extent (``round((r_max - r_min) * H)``, at least 8).

    Returns:
        ``(samples, support)``: the ``depth x a_line`` intensities as
        ``float32``, and a boolean array of the same shape marking which
        entries carry at least one ROI pixel. Row 0 is the transducer face in
        both geometries.

    Raises:
        ValueError: If ``image`` is not 2-D, the mask shape disagrees, the
            geometry name is unknown, or ``scan_geometry="sector"`` is used
            without a complete ``fan``.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"image must be 2-D H x W, got shape {array.shape}.")

    if scan_geometry == "sector":
        return _sample_sector(array, roi_mask, fan, n_a_lines, depth_bins)
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
        ValueError: If ``image`` is not 2-D or leaves ``[0, 1]``, or the
            configured scan geometry is unusable (``sector`` without ``fan``).
    """
    config = config or RawQualityConfig()
    samples, support = sample_a_lines(
        image,
        roi_mask,
        config.scan_geometry,
        fan=config.fan,
        n_a_lines=config.n_a_lines,
        depth_bins=config.depth_bins,
    )

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


def coupling_gate(
    result: RawQualityResult, config: Optional[RawQualityConfig] = None
) -> bool:
    """Is the probe acoustically coupled? -- the Stage 1a verdict on one frame.

    Stage 1a is a **gate**, not an optimiser. It asks a yes/no question --
    *is the probe acoustically coupled to the tissue?* -- and the answer is
    what lets the contact search hand over to Stage 1b, where the search over
    force is carried out on ``Q_seg`` (DESIGN_NOTES §7.4, revised 2026-09-08).
    Nothing here climbs ``Q_raw``: its sub-scores have not been shown to be
    monotone or unimodal in force, and a gate does not need them to be. It
    needs them to separate "coupled" from "not coupled", which is the property
    they were designed around.

    The frame passes when all of the following hold:

    * ``result.score`` is not ``None`` -- the frame was measurable;
    * ``result.rejection_reasons`` is empty -- no reason code fired;
    * ``components["contact_continuity"] >= config.gate_min_contact_continuity``
      -- few enough dead A-lines (PROVISIONAL threshold);
    * ``components["near_field_echo"] >= config.min_near_field_echo`` -- the
      near field actually returns an echo.

    The last two repeat thresholds that also feed the reason codes; they are
    restated here so the gate is a single readable predicate rather than a
    property scattered over ``compute_raw_quality``.

    Args:
        result: Output of :func:`compute_raw_quality` for the frame.
        config: The thresholds; the default configuration when ``None``.

    Returns:
        ``True`` when the frame says the probe is coupled. This is an
        observation-quality verdict, not a command: what the controller does
        with a ``False`` (hold, increase force, re-seat) is decided elsewhere.
    """
    config = config or RawQualityConfig()
    if result.score is None or result.rejection_reasons:
        return False
    continuity = result.components.get("contact_continuity")
    near_field = result.components.get("near_field_echo")
    if continuity is None or near_field is None:
        return False
    return bool(
        continuity >= config.gate_min_contact_continuity
        and near_field >= config.min_near_field_echo
    )