"""Concrete optical-flow backends.

Available backends:

``farneback``
    OpenCV's dense Farneback estimator. Cheap, no learned weights, adequate for
    the small inter-frame displacements typical of a slowly swept ultrasound
    probe.
``identity``
    Zero flow. **Debugging only** -- it asserts that nothing moved, which is
    false for real video and would make the temporal loss punish genuine motion.
``raft_small``
    ``torchvision``'s RAFT-Small, when the optional dependency is installed.
    Run under ``torch.no_grad``; gradients never flow through it.
``precomputed``
    Loads flow written by ``scripts/precompute_flow.py``; see
    :mod:`src.flow.precomputed`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .base import FlowBackend, FlowPair

logger = logging.getLogger(__name__)

__all__ = ["FarnebackFlow", "IdentityFlow", "RaftSmallFlow", "build_flow_backend"]


class FarnebackFlow(FlowBackend):
    """Dense Farneback optical flow via OpenCV.

    The backward field is obtained by calling the estimator with the frames
    swapped (``current, previous``), which yields a field defined on the current
    frame's grid -- exactly the sampling field required for backward warping.

    Args:
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags:
            Forwarded to ``cv2.calcOpticalFlowFarneback``.
    """

    name = "farneback"

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        flags: int = 0,
    ) -> None:
        self.params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=flags,
        )

    def compute(self, previous: np.ndarray, current: np.ndarray) -> FlowPair:
        """Estimate Farneback flow in both directions.

        Raises:
            ImportError: If OpenCV is not installed.
        """
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The 'farneback' flow backend requires OpenCV. "
                "Install it with: pip install opencv-python-headless"
            ) from exc

        self._validate_inputs(previous, current)
        prev_u8 = _to_uint8(previous)
        cur_u8 = _to_uint8(current)

        # previous -> current, defined on the previous frame's grid.
        forward = cv2.calcOpticalFlowFarneback(prev_u8, cur_u8, None, **self.params)
        # current -> previous, defined on the current frame's grid.
        backward = cv2.calcOpticalFlowFarneback(cur_u8, prev_u8, None, **self.params)

        return FlowPair(
            forward=np.ascontiguousarray(forward, dtype=np.float32),
            backward=np.ascontiguousarray(backward, dtype=np.float32),
            metadata={**self.describe(), "params": dict(self.params)},
        )

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "params": dict(self.params)}


class IdentityFlow(FlowBackend):
    """Zero-motion flow, for debugging and unit tests only.

    Warning:
        Using this backend for training asserts that consecutive frames are
        perfectly aligned. Real probe or anatomy motion would then be penalised
        by the temporal loss. A warning is emitted on construction.
    """

    name = "identity"

    def __init__(self, warn: bool = True) -> None:
        if warn:
            logger.warning(
                "IdentityFlow assumes zero motion between frames. It is intended for "
                "debugging and tests only; do not use it to train a temporal model."
            )

    def compute(self, previous: np.ndarray, current: np.ndarray) -> FlowPair:
        self._validate_inputs(previous, current)
        zeros = np.zeros((*previous.shape, 2), dtype=np.float32)
        return FlowPair(
            forward=zeros.copy(),
            backward=zeros.copy(),
            metadata={**self.describe(), "warning": "identity flow: zero motion assumed"},
        )


class RaftSmallFlow(FlowBackend):
    """RAFT-Small optical flow from ``torchvision``, if available.

    The model is evaluated under ``torch.no_grad`` and its output is detached:
    the temporal loss never backpropagates into the flow estimator.

    Args:
        device: Torch device string.
        num_flow_updates: RAFT refinement iterations.
        weights: ``torchvision`` weights enum name, or ``None`` for the default
            pretrained weights (downloaded on first use).
    """

    name = "raft_small"

    def __init__(
        self,
        device: str = "cpu",
        num_flow_updates: int = 12,
        weights: Optional[str] = None,
    ) -> None:
        self.device = device
        self.num_flow_updates = int(num_flow_updates)
        self._weights = weights
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'raft_small' flow backend requires torchvision with optical-flow "
                "models. Install a matching torchvision build, or use "
                "backend: farneback."
            ) from exc
        weights = (
            Raft_Small_Weights.DEFAULT
            if self._weights is None
            else Raft_Small_Weights[self._weights]
        )
        model = raft_small(weights=weights, progress=False)
        model = model.eval().to(torch.device(self.device))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        return model

    def compute(self, previous: np.ndarray, current: np.ndarray) -> FlowPair:
        import torch

        self._validate_inputs(previous, current)
        model = self._load()
        height, width = previous.shape
        if height % 8 or width % 8:
            raise ValueError(
                f"RAFT requires spatial dimensions divisible by 8, got {(height, width)}."
            )

        def to_tensor(image: np.ndarray) -> "torch.Tensor":
            arr = _to_float01(image)
            rgb = np.repeat(arr[None, None], 3, axis=1)
            # RAFT expects inputs normalized to [-1, 1].
            return torch.from_numpy(rgb).float().to(self.device) * 2.0 - 1.0

        prev_t, cur_t = to_tensor(previous), to_tensor(current)
        with torch.no_grad():
            forward = model(prev_t, cur_t, num_flow_updates=self.num_flow_updates)[-1]
            backward = model(cur_t, prev_t, num_flow_updates=self.num_flow_updates)[-1]

        return FlowPair(
            forward=forward[0].permute(1, 2, 0).cpu().numpy().astype(np.float32),
            backward=backward[0].permute(1, 2, 0).cpu().numpy().astype(np.float32),
            metadata={**self.describe(), "num_flow_updates": self.num_flow_updates},
        )


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image of any common dtype/range to ``uint8``."""
    if image.dtype == np.uint8:
        return image
    arr = image.astype(np.float32)
    if arr.max() <= 1.0 + 1e-6 and arr.min() >= -1e-6:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _to_float01(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to ``float32`` in ``[0, 1]``."""
    arr = image.astype(np.float32)
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def build_flow_backend(config: dict[str, Any]) -> FlowBackend:
    """Instantiate a flow backend from a ``flow`` config section.

    Args:
        config: Mapping with a ``backend`` key and optional backend arguments
            under ``params``.

    Returns:
        The configured :class:`~src.flow.base.FlowBackend`.

    Raises:
        KeyError: If the backend name is unknown.
    """
    backend = str(config.get("backend", "farneback"))
    params = dict(config.get("params") or {})
    if backend == "farneback":
        return FarnebackFlow(**params)
    if backend == "identity":
        return IdentityFlow(**params)
    if backend == "raft_small":
        return RaftSmallFlow(**params)
    if backend == "precomputed":
        from .precomputed import PrecomputedFlow

        return PrecomputedFlow(**params)
    raise KeyError(
        f"Unknown flow backend {backend!r}. "
        "Available: farneback, identity, raft_small, precomputed."
    )
