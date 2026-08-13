"""Optical-flow backend interface and shared data structures."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = ["FlowPair", "FlowBackend", "FLOW_SCHEMA_VERSION"]

#: Bumped when the on-disk flow file layout changes incompatibly.
FLOW_SCHEMA_VERSION = 1


@dataclass
class FlowPair:
    """Forward and backward optical flow between two adjacent frames.

    Attributes:
        forward: ``H x W x 2`` field on the **previous** frame's grid pointing
            into the current frame (``previous -> current``).
        backward: ``H x W x 2`` field on the **current** frame's grid pointing
            into the previous frame (``current -> previous``). This is the field
            consumed by :func:`src.flow.warp.warp_backward`.
        reliability: Optional precomputed ``H x W`` reliability map in ``[0, 1]``.
        metadata: Free-form provenance (backend name, frame ids, direction
            convention) persisted alongside the arrays.
    """

    forward: np.ndarray
    backward: np.ndarray
    reliability: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, array in (("forward", self.forward), ("backward", self.backward)):
            if array.ndim != 3 or array.shape[2] != 2:
                raise ValueError(
                    f"{name} flow must have shape H x W x 2, got {array.shape}."
                )
        if self.forward.shape != self.backward.shape:
            raise ValueError(
                f"forward {self.forward.shape} and backward {self.backward.shape} "
                "flow must have the same shape."
            )
        if self.reliability is not None:
            if self.reliability.shape != self.forward.shape[:2]:
                raise ValueError(
                    f"reliability shape {self.reliability.shape} must match the flow "
                    f"spatial size {self.forward.shape[:2]}."
                )
            if self.reliability.min() < 0.0 or self.reliability.max() > 1.0:
                raise ValueError("reliability must be bounded in [0, 1].")

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial ``(height, width)`` of the flow fields."""
        return int(self.forward.shape[0]), int(self.forward.shape[1])


class FlowBackend(abc.ABC):
    """Abstract dense optical-flow estimator between two grayscale frames.

    Implementations must return **detached auxiliary information**: no gradients
    are propagated through the flow estimator in this version of the codebase.
    """

    #: Short identifier written into flow metadata and logs.
    name: str = "abstract"

    @abc.abstractmethod
    def compute(self, previous: np.ndarray, current: np.ndarray) -> FlowPair:
        """Estimate forward and backward flow between two frames.

        Args:
            previous: ``H x W`` grayscale image of the earlier frame.
            current: ``H x W`` grayscale image of the later frame.

        Returns:
            A :class:`FlowPair` following this module's direction convention.
        """

    @staticmethod
    def _validate_inputs(previous: np.ndarray, current: np.ndarray) -> None:
        if previous.shape != current.shape:
            raise ValueError(
                f"Frames must have identical shape, got {previous.shape} and {current.shape}."
            )
        if previous.ndim != 2:
            raise ValueError(
                f"Frames must be 2-D grayscale arrays, got {previous.ndim} dimensions."
            )

    def describe(self) -> dict[str, Any]:
        """Return metadata describing this backend's configuration."""
        return {"backend": self.name, "schema_version": FLOW_SCHEMA_VERSION}
