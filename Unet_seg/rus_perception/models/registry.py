"""Model construction from configuration dictionaries."""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from .slim_unet import SLIM_UNET_PRESETS, SlimUNet
from .standard_unet import STANDARD_UNET_PRESETS, StandardUNet

__all__ = ["build_model", "MODEL_REGISTRY", "available_models"]

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "standard_unet": StandardUNet,
    "slim_unet": SlimUNet,
}

_PRESETS: dict[str, dict[str, dict[str, object]]] = {
    "standard_unet": STANDARD_UNET_PRESETS,
    "slim_unet": SLIM_UNET_PRESETS,
}

# Config keys that describe how a model is *used* rather than how it is built.
_NON_CONSTRUCTOR_KEYS = frozenset({"name", "preset", "input_size", "version"})

# Config aliases accepted for readability in YAML files.
_KEY_ALIASES = {
    "input_channels": "in_channels",
    "output_channels": "out_channels",
}


def available_models() -> list[str]:
    """Return the sorted list of registered model names."""
    return sorted(MODEL_REGISTRY)


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build a model from a ``model`` config section.

    The section must contain ``name`` (``standard_unet`` or ``slim_unet``) and
    may contain ``preset`` plus any constructor override. ``input_channels`` and
    ``output_channels`` are accepted as aliases for ``in_channels`` and
    ``out_channels``. ``input_size`` is consumed by the data pipeline, not the
    model, and is ignored here.

    Args:
        config: Mapping describing the model.

    Returns:
        An instantiated, randomly initialised model.

    Raises:
        KeyError: If ``name`` is missing or unknown, or the preset is unknown.
        TypeError: If an override is not a valid constructor argument.
    """
    if "name" not in config:
        raise KeyError("model config requires a 'name' field (e.g. name: slim_unet).")
    name = str(config["name"])
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")

    kwargs: dict[str, Any] = {}
    preset = config.get("preset")
    if preset is not None:
        presets = _PRESETS[name]
        if str(preset) not in presets:
            raise KeyError(
                f"Unknown preset {preset!r} for model {name!r}. Available: {sorted(presets)}"
            )
        kwargs.update(presets[str(preset)])

    for key, value in config.items():
        if key in _NON_CONSTRUCTOR_KEYS:
            continue
        kwargs[_KEY_ALIASES.get(key, key)] = value

    model_cls = MODEL_REGISTRY[name]
    try:
        return model_cls(**kwargs)
    except TypeError as exc:  # pragma: no cover - exercised via tests on bad configs
        raise TypeError(
            f"Invalid arguments for {model_cls.__name__}: {exc}. "
            f"Provided keys: {sorted(kwargs)}"
        ) from exc
