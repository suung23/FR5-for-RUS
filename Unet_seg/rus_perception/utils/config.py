"""YAML configuration loading, merging and startup validation.

Design rules followed throughout the repository:

* every threshold that changes behaviour lives in a config file, not in source;
* configs are validated **at startup**, so a typo fails immediately with a
  useful message instead of silently disabling a loss term;
* the fully resolved config is snapshotted into every checkpoint, so a run can
  be reproduced from its artefacts alone.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = ["Config", "load_config", "merge_dicts", "ConfigError"]

#: Sections recognised at the top level of a config file.
KNOWN_SECTIONS: tuple[str, ...] = (
    "experiment",
    "model",
    "data",
    "split",
    "augmentation",
    "loss",
    "flow",
    "train",
    "postprocess",
    "control",
    "monitor",
    "logging",
    "evaluation",
)


class ConfigError(ValueError):
    """Raised when a configuration is malformed or internally inconsistent."""


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dictionary.

    Mappings are merged key by key; every other value (including lists) is
    replaced wholesale, so a config file can override a list without having to
    reproduce its previous contents.
    """
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """A validated, nested configuration with dotted-path access.

    Args:
        data: The parsed configuration mapping.
        source: Path the config came from, recorded for provenance.
    """

    def __init__(self, data: Mapping[str, Any], source: Optional[Path] = None) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self.source = Path(source) if source is not None else None

    # -- access ------------------------------------------------------------
    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        if key not in self._data:
            raise ConfigError(
                f"Missing config section {key!r}. Present sections: {sorted(self._data)}"
            )
        return self._data[key]

    def get(self, path: str, default: Any = None) -> Any:
        """Return the value at a dotted ``path``, or ``default`` if absent.

        Example:
            ``config.get("train.optimizer.lr", 1e-3)``
        """
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        """Return the value at a dotted ``path``.

        Raises:
            ConfigError: If the path is absent.
        """
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise ConfigError(
                f"Required config value {path!r} is missing"
                + (f" in {self.source}" if self.source else "")
                + "."
            )
        return value

    def section(self, name: str, default: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Return a config section as a plain dictionary."""
        value = self._data.get(name, default if default is not None else {})
        if not isinstance(value, Mapping):
            raise ConfigError(f"Config section {name!r} must be a mapping, got {type(value).__name__}.")
        return copy.deepcopy(dict(value))

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the whole configuration."""
        return copy.deepcopy(self._data)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a new config with ``overrides`` merged in."""
        return Config(merge_dicts(self._data, overrides), self.source)

    def set(self, path: str, value: Any) -> None:
        """Set a value at a dotted ``path``, creating intermediate mappings."""
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    # -- validation --------------------------------------------------------
    def validate(self, required: Iterable[str] = ("model",)) -> "Config":
        """Validate the configuration and return ``self``.

        Args:
            required: Dotted paths that must be present.

        Raises:
            ConfigError: On unknown sections, missing required values or
                internally inconsistent settings.
        """
        unknown = [key for key in self._data if key not in KNOWN_SECTIONS]
        if unknown:
            raise ConfigError(
                f"Unknown top-level config section(s): {sorted(unknown)}. "
                f"Known sections: {list(KNOWN_SECTIONS)}"
            )
        for path in required:
            self.require(path)

        model = self.section("model")
        if model:
            from ..models.registry import available_models

            name = model.get("name")
            if name not in available_models():
                raise ConfigError(
                    f"model.name must be one of {available_models()}, got {name!r}."
                )
            input_size = model.get("input_size")
            if input_size is not None:
                if len(input_size) != 2 or any(int(v) <= 0 for v in input_size):
                    raise ConfigError(
                        f"model.input_size must be two positive integers, got {input_size!r}."
                    )

        loss = self.section("loss")
        weights = [
            float(loss.get("bce_weight", 1.0)),
            float(loss.get("dice_weight", 1.0)),
            float(loss.get("jaccard_weight", 1.0)),
        ]
        if loss and sum(weights) <= 0:
            raise ConfigError(
                "All spatial loss weights are zero; the model would receive no "
                "segmentation supervision."
            )

        temporal = loss.get("temporal") or {}
        if temporal.get("enabled") and self.get("flow.backend") in (None, "identity"):
            logger.warning(
                "Temporal loss is enabled but flow.backend is %r. Identity/absent flow "
                "assumes zero motion and will penalise genuine probe movement.",
                self.get("flow.backend"),
            )

        train = self.section("train")
        epochs = int(train.get("epochs", 1))
        if epochs < 1:
            raise ConfigError(f"train.epochs must be >= 1, got {epochs}.")
        warmup = int(temporal.get("warmup_epochs", 0) or 0)
        if temporal.get("enabled") and warmup >= epochs:
            raise ConfigError(
                f"loss.temporal.warmup_epochs ({warmup}) is not smaller than train.epochs "
                f"({epochs}); the temporal loss would never be active."
            )
        return self

    def __repr__(self) -> str:
        return f"Config(sections={sorted(self._data)}, source={self.source})"


def load_config(
    path: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
    validate: bool = True,
    required: Iterable[str] = ("model",),
) -> Config:
    """Load a YAML config file.

    Supports a single level of inheritance through a top-level ``defaults`` key
    holding a path relative to the config file.

    Args:
        path: YAML file to load.
        overrides: Mapping merged on top of the loaded config.
        validate: Run :meth:`Config.validate`.
        required: Dotted paths that must be present when validating.

    Returns:
        The loaded :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ConfigError: If the file is not a YAML mapping or fails validation.
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Loading configuration requires PyYAML. Install it with: pip install PyYAML"
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config {path} must contain a YAML mapping, got {type(data).__name__}.")

    data = dict(data)
    parent_path = data.pop("defaults", None)
    if parent_path:
        parent = load_config(path.parent / str(parent_path), validate=False)
        data = merge_dicts(parent.to_dict(), data)

    if overrides:
        data = merge_dicts(data, overrides)

    config = Config(data, path)
    if validate:
        config.validate(required)
    return config
