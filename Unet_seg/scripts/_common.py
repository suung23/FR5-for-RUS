"""Helpers shared by the command-line entry points in ``scripts/``.

Every script imports this module first so that running ``python scripts/x.py``
from the repository root works without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging  # noqa: E402
from typing import Any, Optional  # noqa: E402

from src.data.manifest import Manifest, load_manifest  # noqa: E402
from src.data.splits import (  # noqa: E402
    apply_split,
    assert_no_patient_leakage,
    split_by_patient,
)
from src.utils.config import Config  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = [
    "REPO_ROOT",
    "prepare_manifest",
    "parse_overrides",
    "resolve_manifest_path",
]


def resolve_manifest_path(config: Config, override: Optional[str] = None) -> Path:
    """Return the manifest path from the CLI or the config.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    path = Path(override or config.require("data.manifest"))
    if not path.is_absolute():
        candidate = REPO_ROOT / path
        path = candidate if candidate.exists() else path
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Generate a synthetic one with "
            "scripts/make_synthetic_dataset.py, or point data.manifest at your dataset."
        )
    return path


def prepare_manifest(
    config: Config, manifest_override: Optional[str] = None, validate_files: bool = True
) -> Manifest:
    """Load the manifest, assign patient-level splits and audit for leakage.

    Two split strategies are supported:

    ``manifest``
        Use the ``split`` column already present in the manifest.
    ``patient_random``
        Assign splits deterministically from ``split.ratios`` and ``split.seed``,
        always at the patient level.

    Raises:
        ValueError: If ``strategy`` is ``manifest`` but no row carries a split.
        PatientLeakageError: If any patient ends up in more than one split.
    """
    manifest_path = resolve_manifest_path(config, manifest_override)
    root = config.get("data.root")
    manifest = load_manifest(manifest_path, root)
    manifest.validate_ordering()
    if validate_files:
        manifest.validate_files(check_masks=True, check_flow=False)

    strategy = str(config.get("split.strategy", "manifest"))
    if strategy == "manifest":
        if not manifest.splits:
            raise ValueError(
                f"split.strategy is 'manifest' but no row in {manifest_path} has a "
                "'split' value. Fill the column, or set split.strategy: patient_random."
            )
    elif strategy == "patient_random":
        assignment = split_by_patient(
            manifest.patients,
            config.get("split.ratios", (0.7, 0.15, 0.15)),
            int(config.get("split.seed", 42)),
        )
        manifest = apply_split(manifest, assignment)
    else:
        raise ValueError(
            f"Unknown split.strategy {strategy!r}. Expected 'manifest' or 'patient_random'."
        )

    assert_no_patient_leakage(manifest)
    logger.info(
        "Manifest %s: %d frames, %d patients, splits=%s",
        manifest_path,
        len(manifest),
        len(manifest.patients),
        manifest.splits,
    )
    return manifest


def parse_overrides(pairs: Optional[list[str]]) -> dict[str, Any]:
    """Parse ``--set key.path=value`` overrides into a nested dictionary.

    Values are parsed as YAML, so ``true``, ``3``, ``1.0e-3`` and ``[1, 2]`` all
    keep their natural types.

    Raises:
        ValueError: If an override is not in ``key=value`` form.
    """
    import yaml

    result: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override {pair!r} must be in the form key.path=value.")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        node = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result
