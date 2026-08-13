"""Real-time pipeline with separated capture, inference and consumption stages.

Capture runs in its own thread and pushes into a **bounded** queue. When the
queue is full the *oldest* frame is discarded rather than the newest: for a
control loop, a fresh frame is worth more than a complete history, and
accumulating a backlog would add unbounded latency between what the probe sees
and what the controller is told.

The pipeline stages are:

    capture -> [bounded queue, stale frames dropped]
            -> preprocess -> inference -> postprocess
            -> control-state extraction
            -> consumers (display, logging)

Consumers are callbacks, so display and logging never block the inference loop
structurally and can be omitted entirely in headless mode.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Any, Callable, Iterator, Optional

import numpy as np

from ..control.state import ControlState
from ..flow.base import FlowBackend
from ..metrics.latency import LatencyTracker
from .predictor import Predictor
from .sources import Frame, FrameSource

logger = logging.getLogger(__name__)

__all__ = ["RealtimeConfig", "FrameGrabber", "RealtimePipeline", "PipelineResult"]


@dataclass
class RealtimeConfig:
    """Real-time pipeline behaviour.

    Attributes:
        max_queue_size: Capacity of the capture queue. Small values keep latency
            low; 1 means "always process the newest frame available".
        drop_stale_frames: Discard the oldest queued frame when the queue is
            full. This is applied only to **live** sources (see
            :attr:`~src.inference.sources.FrameSource.is_live`): for a video
            file or an image directory every frame is still available a moment
            later, so dropping would lose data for no latency benefit. Setting
            this to ``False`` makes the capture thread block instead, trading
            latency for completeness.
        threaded_capture: Run capture in a background thread. Disable for
            deterministic tests and file-based replays.
        warmup_frames: Frames excluded from latency statistics.
        enable_temporal: Compute temporal features against the previous state.
        flow_backend: Optional backend used to align consecutive frames. Without
            one, temporal features are computed unwarped and flagged as such.
        snapshot_on_validity_change: Emit a snapshot event whenever
            ``valid_for_control`` flips.
        queue_timeout_s: How long a consumer waits for a frame before checking
            the stop flag again.
    """

    max_queue_size: int = 2
    drop_stale_frames: bool = True
    threaded_capture: bool = True
    warmup_frames: int = 5
    enable_temporal: bool = True
    flow_backend: Optional[FlowBackend] = None
    snapshot_on_validity_change: bool = False
    queue_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        if self.max_queue_size < 1:
            raise ValueError(f"max_queue_size must be >= 1, got {self.max_queue_size}.")
        if self.queue_timeout_s <= 0:
            raise ValueError(f"queue_timeout_s must be > 0, got {self.queue_timeout_s}.")
        if self.warmup_frames < 0:
            raise ValueError(f"warmup_frames must be >= 0, got {self.warmup_frames}.")


@dataclass
class PipelineResult:
    """One processed frame: the raw input, its control state and diagnostics."""

    frame: Frame
    state: ControlState
    dropped_frames: int = 0
    validity_changed: bool = False
    flow: Optional[np.ndarray] = None
    extra: dict[str, Any] = field(default_factory=dict)


class FrameGrabber:
    """Background capture thread feeding a bounded queue.

    Args:
        source: The frame source to drain.
        max_queue_size: Queue capacity.
        drop_stale: Discard the oldest queued frame when full.

    Note:
        :attr:`dropped` counts frames discarded because the consumer could not
        keep up. It is reported by the monitor so a user can see when the
        display, not the model, is the bottleneck.
    """

    def __init__(
        self, source: FrameSource, max_queue_size: int = 2, drop_stale: bool = True
    ) -> None:
        self.source = source
        self.queue: Queue = Queue(maxsize=max(1, int(max_queue_size)))
        self.drop_stale = bool(drop_stale)
        self.dropped = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._finished = threading.Event()
        self._error: Optional[BaseException] = None

    def start(self) -> "FrameGrabber":
        """Start the capture thread."""
        if self._thread is not None:
            raise RuntimeError("FrameGrabber has already been started.")
        self._thread = threading.Thread(target=self._run, name="frame-grabber", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            for frame in self.source:
                if self._stop.is_set():
                    break
                self._offer(frame)
        except BaseException as exc:  # surfaced to the consumer via `error`
            self._error = exc
            logger.exception("Frame capture failed: %s", exc)
        finally:
            self._finished.set()
            # Sentinel unblocks a consumer waiting on an empty queue.
            try:
                self.queue.put_nowait(None)
            except Full:
                pass

    def _offer(self, frame: Frame) -> None:
        """Enqueue a frame, dropping the oldest one if the queue is full."""
        while not self._stop.is_set():
            try:
                self.queue.put_nowait(frame)
                return
            except Full:
                if not self.drop_stale:
                    time.sleep(0.001)
                    continue
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except Empty:
                    pass

    def __iter__(self) -> Iterator[Frame]:
        """Yield frames until the source is exhausted or :meth:`stop` is called."""
        while True:
            if self._stop.is_set():
                break
            try:
                item = self.queue.get(timeout=0.25)
            except Empty:
                if self._finished.is_set() and self.queue.empty():
                    break
                continue
            if item is None:
                break
            yield item
        if self._error is not None:
            raise RuntimeError("Frame capture thread failed") from self._error

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the capture thread to stop and wait for it to finish."""
        self._stop.set()
        # Drain so a blocked producer can observe the stop flag.
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def queue_size(self) -> int:
        """Current number of queued frames."""
        return self.queue.qsize()


class RealtimePipeline:
    """Drives a :class:`Predictor` over a :class:`FrameSource` in real time.

    Args:
        predictor: The single-frame predictor.
        config: Pipeline behaviour.
        latency_tracker: Optional external tracker; one is created if omitted.
    """

    def __init__(
        self,
        predictor: Predictor,
        config: Optional[RealtimeConfig] = None,
        latency_tracker: Optional[LatencyTracker] = None,
    ) -> None:
        self.predictor = predictor
        self.config = config or RealtimeConfig()
        self.latency = latency_tracker or LatencyTracker(warmup=self.config.warmup_frames)
        self.previous_state: Optional[ControlState] = None
        self._previous_image: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Forget the previous frame, e.g. when switching sequence."""
        self.previous_state = None
        self._previous_image = None

    def _estimate_flow(self, current: np.ndarray) -> Optional[np.ndarray]:
        """Backward flow from the current frame into the previous one, or ``None``."""
        if (
            not self.config.enable_temporal
            or self.config.flow_backend is None
            or self._previous_image is None
        ):
            return None
        if self._previous_image.shape != current.shape:
            logger.debug("Frame size changed; skipping flow for this transition.")
            return None
        try:
            pair = self.config.flow_backend.compute(self._previous_image, current)
        except (ImportError, NotImplementedError, ValueError) as exc:
            logger.warning("Optical-flow estimation failed (%s); continuing without it.", exc)
            return None
        return pair.backward

    def process_frame(self, frame: Frame) -> PipelineResult:
        """Run the full pipeline on one frame and update the temporal state."""
        started = time.perf_counter()

        image = np.asarray(frame.image)
        if image.ndim == 3:
            image = image.mean(axis=2).astype(np.float32)

        flow = self._estimate_flow(image)
        previous = self.previous_state if self.config.enable_temporal else None

        state = self.predictor.predict_control_state(
            frame=frame.image,
            previous_state=previous,
            flow=flow,
            metadata={
                "frame_id": f"{frame.source_id}:{frame.index}" if frame.source_id else str(frame.index),
                "timestamp": frame.timestamp,
                "frame_index": frame.index,
            },
        )
        end_to_end = (time.perf_counter() - started) * 1000.0
        state.end_to_end_latency_ms = end_to_end

        self.latency.record("preprocessing", state.preprocessing_latency_ms)
        self.latency.record("inference", state.inference_latency_ms)
        self.latency.record("postprocessing", state.postprocessing_latency_ms)
        self.latency.record("control_features", state.control_feature_latency_ms)
        self.latency.record("end_to_end", end_to_end)

        validity_changed = (
            self.previous_state is not None
            and self.previous_state.valid_for_control != state.valid_for_control
        )

        self.previous_state = state
        self._previous_image = image
        return PipelineResult(
            frame=frame, state=state, validity_changed=validity_changed, flow=flow
        )

    def run(
        self,
        source: FrameSource,
        on_result: Optional[Callable[[PipelineResult], None]] = None,
        max_frames: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> list[ControlState]:
        """Process a source to completion.

        Args:
            source: Frame source. It is released before returning, including on
                error or interruption.
            on_result: Callback invoked per processed frame. Returning is enough
                to continue; raising aborts the run after releasing resources.
            max_frames: Stop after this many frames.
            stop_event: External event that requests a graceful shutdown.

        Returns:
            The list of :class:`ControlState` objects produced, in order.
        """
        states: list[ControlState] = []
        grabber: Optional[FrameGrabber] = None
        self.reset()

        try:
            if self.config.threaded_capture:
                drop_stale = self.config.drop_stale_frames and source.is_live
                if self.config.drop_stale_frames and not drop_stale:
                    logger.debug(
                        "Source %s is not live; processing every frame instead of "
                        "dropping stale ones.", source.name,
                    )
                grabber = FrameGrabber(
                    source,
                    max_queue_size=self.config.max_queue_size,
                    drop_stale=drop_stale,
                ).start()
                stream: Iterator[Frame] = iter(grabber)
            else:
                stream = iter(source)

            for count, frame in enumerate(stream):
                if stop_event is not None and stop_event.is_set():
                    logger.info("Stop requested; shutting down after %d frame(s).", count)
                    break
                result = self.process_frame(frame)
                result.dropped_frames = grabber.dropped if grabber else 0
                states.append(result.state)
                if on_result is not None:
                    on_result(result)
                if max_frames is not None and len(states) >= max_frames:
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted by user; shutting down cleanly.")
        finally:
            if grabber is not None:
                grabber.stop()
            source.release()
        return states
