"""Overlay rendering and bounded rolling sparklines for the real-time monitor.

All history is kept in fixed-length ``deque`` objects, so a monitoring session
that runs for hours uses constant memory: the entire video is never retained.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..control.state import ControlState

__all__ = ["StatusColors", "RollingHistory", "MonitorRenderer"]

#: Series drawn as sparklines, with the value range each is normalised against.
SPARKLINE_SERIES: tuple[tuple[str, str, tuple[float, float]], ...] = (
    ("area_ratio", "area ratio", (0.0, 0.5)),
    ("centroid_x", "centroid x", (0.0, 1.0)),
    ("centroid_y", "centroid y", (0.0, 1.0)),
    ("segmentation_confidence", "seg conf", (0.0, 1.0)),
    ("temporal_iou", "temporal IoU", (0.0, 1.0)),
    ("temporal_stability", "stability", (0.0, 1.0)),
    ("control_quality", "quality", (0.0, 1.0)),
    ("end_to_end_latency_ms", "latency ms", (0.0, 100.0)),
)


@dataclass
class StatusColors:
    """BGR colours for the three validity states."""

    valid: tuple[int, int, int] = (0, 200, 0)
    marginal: tuple[int, int, int] = (0, 200, 255)
    invalid: tuple[int, int, int] = (0, 0, 255)

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "StatusColors":
        """Build from a config mapping of BGR triples."""
        data = data or {}
        return cls(
            valid=tuple(data.get("valid", (0, 200, 0))),  # type: ignore[arg-type]
            marginal=tuple(data.get("marginal", (0, 200, 255))),  # type: ignore[arg-type]
            invalid=tuple(data.get("invalid", (0, 0, 255))),  # type: ignore[arg-type]
        )


@dataclass
class RollingHistory:
    """Bounded rolling buffers of the values plotted as sparklines.

    Args:
        maxlen: Capacity of every series. Older samples are discarded, so memory
            stays constant regardless of session length.
    """

    maxlen: int = 240
    series: dict[str, deque] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.maxlen < 2:
            raise ValueError(f"maxlen must be >= 2, got {self.maxlen}.")
        for name, _, _ in SPARKLINE_SERIES:
            self.series[name] = deque(maxlen=self.maxlen)

    def append(self, state: ControlState) -> None:
        """Record one frame's values, substituting NaN for missing measurements."""
        values = {
            "area_ratio": state.mask_area_ratio,
            "centroid_x": state.centroid_x_normalized,
            "centroid_y": state.centroid_y_normalized,
            "segmentation_confidence": state.segmentation_confidence,
            "temporal_iou": state.temporal_warped_iou,
            "temporal_stability": state.temporal_stability_score,
            "control_quality": state.control_quality_score,
            "end_to_end_latency_ms": state.end_to_end_latency_ms,
        }
        for name, value in values.items():
            self.series[name].append(float("nan") if value is None else float(value))

    def total_samples(self) -> int:
        """Number of samples currently retained in the longest series."""
        return max((len(values) for values in self.series.values()), default=0)


class MonitorRenderer:
    """Renders the four-panel monitoring view.

    Panels: the raw frame, the segmentation overlay, the probability map and a
    text/sparkline panel with the control and temporal-stability values.

    Args:
        colors: Status colours.
        history: Bounded rolling history driving the sparklines.
        marginal_quality_threshold: A frame that passes the validity gate but
            whose control-quality score is below this value is drawn yellow
            ("marginal") rather than green.
        panel_size: ``(height, width)`` of each panel.
    """

    def __init__(
        self,
        colors: Optional[StatusColors] = None,
        history: Optional[RollingHistory] = None,
        marginal_quality_threshold: float = 0.55,
        panel_size: tuple[int, int] = (360, 360),
    ) -> None:
        self.colors = colors or StatusColors()
        self.history = history or RollingHistory()
        self.marginal_quality_threshold = float(marginal_quality_threshold)
        self.panel_size = (int(panel_size[0]), int(panel_size[1]))

    # -- helpers -----------------------------------------------------------
    def status_color(self, state: ControlState) -> tuple[int, int, int]:
        """Green when valid, yellow when valid-but-marginal, red when invalid."""
        if not state.valid_for_control:
            return self.colors.invalid
        if state.control_quality_score < self.marginal_quality_threshold:
            return self.colors.marginal
        return self.colors.valid

    def status_text(self, state: ControlState) -> str:
        """``VALID`` / ``MARGINAL`` / ``INVALID``."""
        if not state.valid_for_control:
            return "INVALID"
        if state.control_quality_score < self.marginal_quality_threshold:
            return "MARGINAL"
        return "VALID"

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        import cv2

        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 3:
            return array.astype(np.uint8)
        array = np.clip(array.astype(np.float32), 0.0, 1.0)
        return cv2.cvtColor((array * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def _fit(self, image: np.ndarray) -> np.ndarray:
        import cv2

        height, width = self.panel_size
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _title(panel: np.ndarray, text: str) -> np.ndarray:
        """Draw a panel title on a dark strip so it stays readable on any image."""
        import cv2

        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (panel.shape[1], 22), (20, 20, 20), -1)
        blended = cv2.addWeighted(overlay, 0.55, panel, 0.45, 0)
        cv2.putText(blended, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1)
        return blended

    def _sparkline(
        self, values: Sequence[float], label: str, bounds: tuple[float, float], width: int, height: int
    ) -> np.ndarray:
        """Draw one labelled sparkline on a dark background."""
        import cv2

        panel = np.full((height, width, 3), 30, dtype=np.uint8)
        cv2.putText(panel, label, (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)

        finite = [v for v in values if np.isfinite(v)]
        if len(finite) >= 2:
            low, high = bounds
            observed_high = max(finite)
            if observed_high > high:  # auto-expand rather than clip the trace
                high = observed_high * 1.05
            span = max(high - low, 1e-6)
            plot_left, plot_top = 62, 2
            plot_width = max(2, width - plot_left - 4)
            plot_height = max(2, height - 4)

            points: list[tuple[int, int]] = []
            count = len(values)
            for index, value in enumerate(values):
                if not np.isfinite(value):
                    continue
                x = plot_left + int(index * plot_width / max(count - 1, 1))
                normalized = (value - low) / span
                y = plot_top + int((1.0 - min(max(normalized, 0.0), 1.0)) * plot_height)
                points.append((x, y))
            if len(points) >= 2:
                cv2.polylines(panel, [np.array(points, np.int32)], False, (0, 220, 220), 1)
            cv2.putText(
                panel, f"{finite[-1]:.3f}", (width - 52, height - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 220), 1,
            )
        else:
            cv2.putText(panel, "...", (66, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 110, 110), 1)
        return panel

    # -- panels ------------------------------------------------------------
    def overlay_panel(self, image: np.ndarray, state: ControlState) -> np.ndarray:
        """Frame with the mask contour, centroid, bounding box and image centre."""
        import cv2

        panel = self._fit(self._to_bgr(image))
        height, width = panel.shape[:2]
        color = self.status_color(state)

        if state.binary_mask is not None and state.binary_mask.any():
            mask = cv2.resize(
                state.binary_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            )
            tinted = panel.copy()
            tinted[mask > 0] = color
            panel = cv2.addWeighted(panel, 0.72, tinted, 0.28, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, color, 2)

        # Image centre: the reference point of center_error_x / center_error_y.
        cv2.drawMarker(
            panel, (width // 2, height // 2), (200, 200, 200), cv2.MARKER_CROSS, 14, 1
        )
        if state.centroid_x_normalized is not None:
            cx = int(state.centroid_x_normalized * width)
            cy = int(state.centroid_y_normalized * height)
            cv2.circle(panel, (cx, cy), 5, color, -1)
            cv2.line(panel, (width // 2, height // 2), (cx, cy), color, 1)
        if state.bounding_box is not None and state.image_width:
            sx = width / float(state.image_width)
            sy = height / float(state.image_height)
            box = state.bounding_box
            cv2.rectangle(
                panel,
                (int(box.x_min * sx), int(box.y_min * sy)),
                (int(box.x_max * sx), int(box.y_max * sy)),
                color, 1,
            )
        return self._title(panel, "segmentation")

    def probability_panel(self, state: ControlState) -> np.ndarray:
        """Probability map rendered with a perceptual colour map."""
        import cv2

        height, width = self.panel_size
        if state.probability_map is None:
            panel = np.full((height, width, 3), 40, dtype=np.uint8)
            cv2.putText(
                panel, "probability map not retained", (10, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1,
            )
            return panel
        probability = np.clip(state.probability_map, 0.0, 1.0)
        coloured = cv2.applyColorMap((probability * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        panel = self._title(self._fit(coloured), "probability / uncertainty")
        cv2.putText(
            panel, f"boundary entropy {state.mean_boundary_entropy:.3f}",
            (6, panel.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1,
        )
        return panel

    def metrics_panel(self, state: ControlState, fps: Optional[float], dropped: int) -> np.ndarray:
        """Text panel with control, temporal and latency values."""
        import cv2

        height, width = self.panel_size
        panel = np.full((height, width, 3), 25, dtype=np.uint8)
        color = self.status_color(state)

        def fmt(value: Optional[float], digits: int = 3) -> str:
            return "n/a" if value is None else f"{value:.{digits}f}"

        cv2.rectangle(panel, (0, 0), (width - 1, 26), color, -1)
        cv2.putText(
            panel, f"{self.status_text(state)}  q={state.control_quality_score:.2f}",
            (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2,
        )

        lines = [
            f"FPS            {fmt(fps, 1)}   dropped {dropped}",
            f"model latency  {state.inference_latency_ms:.1f} ms",
            f"end-to-end     {state.end_to_end_latency_ms:.1f} ms",
            "",
            f"centroid       ({fmt(state.centroid_x_normalized)}, {fmt(state.centroid_y_normalized)})",
            f"center err x   {fmt(state.center_error_x)}",
            f"center err y   {fmt(state.center_error_y)}",
            f"area ratio     {state.mask_area_ratio:.4f}",
            "",
            f"seg confidence {state.segmentation_confidence:.3f}",
            f"bnd entropy    {state.mean_boundary_entropy:.3f}",
            f"lumen contrast {fmt(state.lumen_surrounding_contrast)}",
            "",
            f"temporal IoU   {fmt(state.temporal_warped_iou)}",
            f"centroid jump  {fmt(state.normalized_centroid_jump)}",
            f"area change    {fmt(state.relative_area_change)}",
            f"stability      {fmt(state.temporal_stability_score)}",
            "",
            f"quality score  {state.control_quality_score:.3f}",
            f"valid          {state.valid_for_control}",
        ]
        y = 42
        for line in lines:
            if line:
                cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (225, 225, 225), 1)
                y += 14
            else:
                y += 7  # compact separators keep the rejection list on-panel

        if state.rejection_reasons:
            cv2.putText(panel, "rejections:", (8, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.colors.invalid, 1)
            y += 16
            for reason in state.rejection_reasons[:4]:
                cv2.putText(panel, f"  {reason}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.colors.invalid, 1)
                y += 14
            if len(state.rejection_reasons) > 4:
                cv2.putText(
                    panel, f"  (+{len(state.rejection_reasons) - 4} more)", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.colors.invalid, 1,
                )
        return panel

    def sparkline_panel(self, width: int) -> np.ndarray:
        """Stack of bounded rolling sparklines."""
        row_height = 26
        rows = [
            self._sparkline(list(self.history.series[name]), label, bounds, width, row_height)
            for name, label, bounds in SPARKLINE_SERIES
        ]
        return np.vstack(rows)

    def render(
        self,
        image: np.ndarray,
        state: ControlState,
        fps: Optional[float] = None,
        dropped: int = 0,
        update_history: bool = True,
    ) -> np.ndarray:
        """Compose the full monitoring canvas for one frame."""
        if update_history:
            self.history.append(state)

        raw = self._title(self._fit(self._to_bgr(image)), "raw ultrasound")

        top = np.hstack(
            [
                raw,
                self.overlay_panel(image, state),
                self.probability_panel(state),
                self.metrics_panel(state, fps, dropped),
            ]
        )
        return np.vstack([top, self.sparkline_panel(top.shape[1])])
