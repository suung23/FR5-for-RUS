"""Segmentation architectures: Standard U-Net baseline and Slim U-Net."""

from .blocks import ConvBlock, Downsample, OutConv, UpsampleBlock
from .registry import MODEL_REGISTRY, available_models, build_model
from .report import ArchitectureReport, build_architecture_report, estimate_macs
from .slim_unet import (
    INFERRED_DETAILS,
    PAPER_PARAMETER_COUNTS,
    SLIM_UNET_PRESETS,
    SlimUNet,
)
from .standard_unet import (
    STANDARD_UNET_PRESETS,
    StandardUNet,
    remap_legacy_unet_state_dict,
)

__all__ = [
    "ConvBlock",
    "Downsample",
    "OutConv",
    "UpsampleBlock",
    "StandardUNet",
    "STANDARD_UNET_PRESETS",
    "remap_legacy_unet_state_dict",
    "SlimUNet",
    "SLIM_UNET_PRESETS",
    "INFERRED_DETAILS",
    "PAPER_PARAMETER_COUNTS",
    "build_model",
    "available_models",
    "MODEL_REGISTRY",
    "ArchitectureReport",
    "build_architecture_report",
    "estimate_macs",
]
