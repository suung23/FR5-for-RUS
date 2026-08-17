"""Latency and deployment measurement.

Two rules make these numbers trustworthy:

#. **CUDA is synchronised** before every timestamp. Without it the measured
   "inference time" is the time to *enqueue* kernels, not to run them, which
   typically under-reports GPU latency by an order of magnitude.
#. **Warm-up iterations are discarded.** The first calls pay for lazy CUDA
   context creation, autotuning and allocator growth.

Model-only latency and end-to-end application latency are reported separately,
because a controller cares about the latter while an architecture comparison
needs the former.
"""

from __future__ import annotations

import platform
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np

__all__ = ["LatencyStatistics", "LatencyTracker", "synchronize", "benchmark_callable", "device_report"]


def synchronize(device: Any = None) -> None:
    """Synchronise the CUDA stream when ``device`` is a CUDA device.

    Safe to call unconditionally; it is a no-op on CPU.
    """
    import torch

    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@dataclass
class LatencyStatistics:
    """Summary statistics of a latency series, in milliseconds."""

    name: str
    count: int
    mean_ms: Optional[float]
    median_ms: Optional[float]
    p95_ms: Optional[float]
    min_ms: Optional[float]
    max_ms: Optional[float]
    std_ms: Optional[float]
    fps: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "std_ms": self.std_ms,
            "fps": self.fps,
        }

    def to_text(self) -> str:
        """Render one line of the latency table."""
        if self.count == 0 or self.mean_ms is None:
            return f"{self.name:<28}: no samples"
        return (
            f"{self.name:<28}: mean={self.mean_ms:7.2f} ms  median={self.median_ms:7.2f} ms  "
            f"p95={self.p95_ms:7.2f} ms  min={self.min_ms:6.2f}  max={self.max_ms:7.2f}  "
            f"fps={self.fps:6.1f}  n={self.count}"
        )


def _statistics(name: str, samples: Iterable[float]) -> LatencyStatistics:
    values = np.asarray([float(v) for v in samples], dtype=np.float64)
    if values.size == 0:
        return LatencyStatistics(name, 0, None, None, None, None, None, None, None)
    mean = float(values.mean())
    return LatencyStatistics(
        name=name,
        count=int(values.size),
        mean_ms=mean,
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        min_ms=float(values.min()),
        max_ms=float(values.max()),
        std_ms=float(values.std(ddof=0)),
        fps=float(1000.0 / mean) if mean > 0 else None,
    )


class LatencyTracker:
    """Bounded rolling latency accumulator with a warm-up phase.

    Samples are held in fixed-size deques, so a monitoring session that runs for
    hours uses constant memory.

    Args:
        warmup: Number of initial samples per stage that are discarded.
        max_samples: Ring-buffer capacity per stage.
    """

    def __init__(self, warmup: int = 10, max_samples: int = 1000) -> None:
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}.")
        if max_samples < 1:
            raise ValueError(f"max_samples must be >= 1, got {max_samples}.")
        self.warmup = int(warmup)
        self.max_samples = int(max_samples)
        self._samples: dict[str, deque[float]] = {}
        self._seen: dict[str, int] = {}

    def record(self, stage: str, milliseconds: float) -> None:
        """Record one latency sample for ``stage``, discarding warm-up samples."""
        seen = self._seen.get(stage, 0) + 1
        self._seen[stage] = seen
        if seen <= self.warmup:
            return
        self._samples.setdefault(stage, deque(maxlen=self.max_samples)).append(
            float(milliseconds)
        )

    def series(self, stage: str) -> list[float]:
        """Return the retained samples for ``stage``."""
        return list(self._samples.get(stage, ()))

    def statistics(self, stage: str) -> LatencyStatistics:
        """Return summary statistics for one stage."""
        return _statistics(stage, self._samples.get(stage, ()))

    def all_statistics(self) -> dict[str, LatencyStatistics]:
        """Return statistics for every recorded stage."""
        return {stage: self.statistics(stage) for stage in sorted(self._samples)}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of all stages."""
        return {
            "warmup": self.warmup,
            "stages": {
                stage: statistics.to_dict()
                for stage, statistics in self.all_statistics().items()
            },
        }

    def to_text(self) -> str:
        """Render the full latency table."""
        statistics = self.all_statistics()
        if not statistics:
            return "No latency samples recorded (all still within warm-up?)."
        return "\n".join(value.to_text() for value in statistics.values())


def benchmark_callable(
    function: Callable[[], Any],
    iterations: int = 100,
    warmup: int = 10,
    device: Any = None,
    name: str = "callable",
) -> LatencyStatistics:
    """Time a callable with CUDA synchronisation and a warm-up phase.

    Args:
        function: Zero-argument callable to time.
        iterations: Measured iterations.
        warmup: Discarded warm-up iterations.
        device: Device to synchronise; ``None`` synchronises the default CUDA
            device when one is available.
        name: Label for the returned statistics.

    Returns:
        A :class:`LatencyStatistics`.

    Raises:
        ValueError: If ``iterations < 1``.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}.")

    for _ in range(max(0, warmup)):
        function()
    synchronize(device)

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return _statistics(name, samples)


def device_report(device: Any = None) -> dict[str, Any]:
    """Collect device, memory and library information for a benchmark report."""
    import torch

    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device) if device is not None else "cpu",
    }
    resolved = torch.device(device) if device is not None else None
    if torch.cuda.is_available() and (resolved is None or resolved.type == "cuda"):
        index = resolved.index if resolved is not None and resolved.index is not None else 0
        properties = torch.cuda.get_device_properties(index)
        report.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_mb": properties.total_memory / (1024**2),
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated(index) / (1024**2),
                "gpu_max_memory_allocated_mb": torch.cuda.max_memory_allocated(index) / (1024**2),
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved(index) / (1024**2),
            }
        )
    else:
        report["gpu_memory_note"] = "GPU memory usage unavailable (no CUDA device in use)."
    return report
