"""Checkpoint saving and loading with full reproducibility metadata.

A checkpoint written by this repository always carries enough information to
rebuild the exact model and to trace the run that produced it: model and
optimizer state, scheduler and AMP scaler state, the epoch, the best validation
score, the complete resolved configuration, the model version, the random seed
and -- when the working tree is a git repository -- the commit hash and dirty
flag.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = ["CheckpointMetadata", "git_commit_hash", "save_checkpoint", "load_checkpoint", "checkpoint_id"]

#: Bumped when the checkpoint layout changes incompatibly.
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class CheckpointMetadata:
    """Non-tensor content of a checkpoint."""

    epoch: int
    best_metric: float
    best_metric_name: str
    model_version: str
    seed: int
    config: dict[str, Any]
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    schema_version: int = CHECKPOINT_SCHEMA_VERSION


def git_commit_hash(repo_dir: Optional[str | Path] = None) -> tuple[Optional[str], Optional[bool]]:
    """Return ``(commit_hash, is_dirty)`` for the repository, or ``(None, None)``.

    Never raises: a missing git binary or a non-repository directory simply
    yields ``None`` so training is not blocked by provenance collection.
    """
    directory = str(repo_dir or Path(__file__).resolve().parents[2])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if commit.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return commit.stdout.strip(), dirty
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        logger.debug("Could not read git metadata: %s", exc)
        return None, None


def checkpoint_id(path: str | Path) -> str:
    """Return a short, stable identifier for a checkpoint file.

    The identifier is the first 12 hex characters of the file's SHA-256 digest,
    so the same weights always yield the same id regardless of file name.
    """
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def save_checkpoint(
    path: str | Path,
    model: Any,
    metadata: CheckpointMetadata,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a checkpoint atomically.

    The file is written to a temporary name and then renamed, so an interrupted
    save cannot leave a truncated checkpoint behind.

    Args:
        path: Destination file.
        model: Model whose ``state_dict`` is saved.
        metadata: Reproducibility metadata.
        optimizer: Optional optimizer.
        scheduler: Optional LR scheduler.
        scaler: Optional AMP ``GradScaler``.
        extra: Additional JSON-friendly entries.

    Returns:
        The path written.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if metadata.git_commit is None and metadata.git_dirty is None:
        metadata.git_commit, metadata.git_dirty = git_commit_hash()

    payload: dict[str, Any] = {
        "schema_version": metadata.schema_version,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(metadata.epoch),
        "best_metric": float(metadata.best_metric),
        "best_metric_name": metadata.best_metric_name,
        "model_version": metadata.model_version,
        "seed": int(metadata.seed),
        "config": metadata.config,
        "git_commit": metadata.git_commit,
        "git_dirty": metadata.git_dirty,
    }
    if extra:
        payload.update(dict(extra))

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    logger.info(
        "Saved checkpoint %s (epoch=%d, %s=%.4f)",
        path,
        metadata.epoch,
        metadata.best_metric_name,
        metadata.best_metric,
    )
    return path


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Checkpoint file.
        map_location: ``torch.load`` device mapping.

    Returns:
        The checkpoint payload, with ``checkpoint_id`` added.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not a checkpoint produced by this repository
            or uses a newer schema version.
    """
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(
            f"{path} is not a checkpoint produced by this repository (no 'model_state' key). "
            "Legacy milesial checkpoints can be converted with "
            "src.models.remap_legacy_unet_state_dict."
        )
    version = int(payload.get("schema_version", CHECKPOINT_SCHEMA_VERSION))
    if version > CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint {path} uses schema version {version}; this code supports at "
            f"most {CHECKPOINT_SCHEMA_VERSION}."
        )
    payload["checkpoint_id"] = checkpoint_id(path)
    return payload
