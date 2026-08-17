"""Reading and writing precomputed optical-flow files.

Recomputing dense optical flow every epoch is wasteful, so the recommended
workflow is::

    video sequence -> precompute forward/backward flow -> save .npz -> load

Files are ``.npz`` archives holding ``forward``, ``backward``, an optional
``reliability`` map and a JSON ``metadata`` string that records the direction
convention, the backend and the frame identifiers. Corrupted or truncated files
are detected on load rather than silently producing zeros.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import FLOW_SCHEMA_VERSION, FlowBackend, FlowPair

logger = logging.getLogger(__name__)

__all__ = ["save_flow_pair", "load_flow_pair", "is_valid_flow_file", "PrecomputedFlow"]

#: Recorded in every file so a reader can verify the warping convention.
DIRECTION_CONVENTION = (
    "forward: previous-grid, previous->current. "
    "backward: current-grid, current->previous (grid_sample sampling field)."
)


def save_flow_pair(
    path: str | Path,
    pair: FlowPair,
    frame_id: Optional[str] = None,
    previous_frame_id: Optional[str] = None,
    compress: bool = True,
) -> Path:
    """Write a :class:`FlowPair` to ``path`` as an ``.npz`` archive.

    Args:
        path: Destination file. Parent directories are created.
        pair: Flow to serialise.
        frame_id: Identifier of the current frame.
        previous_frame_id: Identifier of the previous frame.
        compress: Use ``np.savez_compressed`` (smaller, slightly slower).

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "schema_version": FLOW_SCHEMA_VERSION,
        "direction_convention": DIRECTION_CONVENTION,
        "height": int(pair.forward.shape[0]),
        "width": int(pair.forward.shape[1]),
        "frame_id": frame_id,
        "previous_frame_id": previous_frame_id,
        **pair.metadata,
    }
    arrays: dict[str, np.ndarray] = {
        "forward": pair.forward.astype(np.float32),
        "backward": pair.backward.astype(np.float32),
        "metadata": np.array(json.dumps(metadata, default=str)),
    }
    if pair.reliability is not None:
        arrays["reliability"] = pair.reliability.astype(np.float32)

    writer = np.savez_compressed if compress else np.savez
    writer(path, **arrays)
    return path


def load_flow_pair(path: str | Path, expected_shape: Optional[tuple[int, int]] = None) -> FlowPair:
    """Load a flow file written by :func:`save_flow_pair`.

    Args:
        path: File to read.
        expected_shape: Optional ``(height, width)`` the flow must match.

    Returns:
        The deserialised :class:`FlowPair`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the archive is corrupted, incomplete, has a newer schema
            version, or does not match ``expected_shape``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Flow file not found: {path}")

    missing: set[str] = set()
    metadata: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = {"forward", "backward"} - set(archive.files)
            if missing:
                # Reported after the try block, so numpy's own errors stay
                # distinguishable from this deliberate validation failure.
                forward = backward = np.empty((0, 0, 2), dtype=np.float32)
                reliability = None
            else:
                forward = np.asarray(archive["forward"], dtype=np.float32)
                backward = np.asarray(archive["backward"], dtype=np.float32)
                reliability = (
                    np.asarray(archive["reliability"], dtype=np.float32)
                    if "reliability" in archive.files
                    else None
                )
                if "metadata" in archive.files:
                    try:
                        metadata = json.loads(str(archive["metadata"].item()))
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(
                            "Flow file %s has unreadable metadata; ignoring it.", path
                        )
    except Exception as exc:
        # Truncated archives, zlib failures and non-npz files all land here.
        raise ValueError(f"Flow file {path} is corrupted and could not be read: {exc}") from exc

    if missing:
        raise ValueError(
            f"Flow file {path} is incomplete; missing array(s): {sorted(missing)}."
        )

    version = int(metadata.get("schema_version", FLOW_SCHEMA_VERSION))
    if version > FLOW_SCHEMA_VERSION:
        raise ValueError(
            f"Flow file {path} uses schema version {version}, but this code "
            f"supports at most {FLOW_SCHEMA_VERSION}. Regenerate the flow files."
        )
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise ValueError(f"Flow file {path} contains NaN or infinite values.")
    if expected_shape is not None and forward.shape[:2] != tuple(expected_shape):
        raise ValueError(
            f"Flow file {path} has spatial size {forward.shape[:2]} but "
            f"{tuple(expected_shape)} was expected."
        )

    return FlowPair(
        forward=forward, backward=backward, reliability=reliability, metadata=metadata
    )


def is_valid_flow_file(path: str | Path, expected_shape: Optional[tuple[int, int]] = None) -> bool:
    """Return ``True`` if ``path`` holds a readable, complete flow file."""
    try:
        load_flow_pair(path, expected_shape)
    except (FileNotFoundError, ValueError):
        return False
    return True


class PrecomputedFlow(FlowBackend):
    """Flow "estimator" that reads fields from disk instead of computing them.

    Args:
        root: Directory prefix prepended to relative flow paths.

    Note:
        :meth:`compute` is not supported -- this backend serves
        :meth:`load` from paths recorded in the dataset manifest.
    """

    name = "precomputed"

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)

    def compute(self, previous: np.ndarray, current: np.ndarray) -> FlowPair:
        raise NotImplementedError(
            "PrecomputedFlow does not estimate flow. Generate it first with "
            "scripts/precompute_flow.py, then reference the files from the manifest."
        )

    def load(self, path: str | Path, expected_shape: Optional[tuple[int, int]] = None) -> FlowPair:
        """Load a flow file, resolving relative paths against ``root``."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return load_flow_pair(candidate, expected_shape)
