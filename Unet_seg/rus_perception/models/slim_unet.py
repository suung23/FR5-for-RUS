"""Slim U-Net -- unofficial, paper-based reimplementation.

Reference:
    A. Raina et al., "Slim U-Net: Efficient Anatomical Feature Preserving U-net
    Architecture for Ultrasound Image Segmentation", arXiv:2302.11524.

This is **not** the official implementation released by the paper's authors.
Details that the paper does not state explicitly were inferred; every such
inference is recorded in :data:`INFERRED_DETAILS` and reproduced in the README.

The defining property of Slim U-Net is that the contracting path performs
*fewer convolution operations* than a Standard U-Net: one 3x3 convolution per
resolution instead of two, arranged as

    3x3 conv -> BatchNorm -> ReLU -> Dropout -> 2x2 max-pool

Parameter-count reconstruction
------------------------------
The paper reports 8,635,809 trainable parameters for its Standard U-Net
baseline and 4,705,377 for Slim U-Net. Building both models with

* channel progression 32 -> 64 -> 128 -> 256 -> 512,
* 1-channel input and 1-channel (logit) output,
* biased 3x3 convolutions each followed by BatchNorm and ReLU,
* ``ConvTranspose2d(kernel_size=3, stride=2)`` decoder upsampling,
* a final 1x1 convolution,

and using *two* convolutions per stage for the baseline and *one* convolution
per stage -- in both the encoder **and** the decoder -- for Slim U-Net,
reproduces both published figures exactly. A slim encoder combined with a
double-convolution decoder yields 5,490,177 parameters and is therefore ruled
out. Both decoder variants remain constructible via ``decoder_convs`` so the
comparison can be re-run; see ``scripts/architecture_report.py``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .blocks import (
    ConvBlock,
    Downsample,
    DropoutKind,
    NormalizationName,
    OutConv,
    UpsampleBlock,
    UpsamplingName,
)

__all__ = ["SlimUNet", "SLIM_UNET_PRESETS", "INFERRED_DETAILS", "PAPER_PARAMETER_COUNTS"]

#: Trainable-parameter counts reported in the paper, used as reference targets.
PAPER_PARAMETER_COUNTS: dict[str, int] = {
    "standard_unet": 8_635_809,
    "slim_unet": 4_705_377,
}

#: Implementation choices that the paper does not specify explicitly.
INFERRED_DETAILS: tuple[str, ...] = (
    "Decoder stages use a single 3x3 convolution (like the encoder). A "
    "double-convolution decoder gives 5,490,177 parameters versus the reported "
    "4,705,377; the single-convolution decoder matches exactly.",
    "Decoder upsampling is ConvTranspose2d(kernel_size=3, stride=2, padding=1, "
    "output_padding=1). kernel_size=2 gives 3,834,977 (Slim) and 7,765,409 "
    "(Standard) parameters and matches neither reported figure.",
    "3x3 convolutions carry a bias term in addition to BatchNorm. Removing the "
    "bias gives 4,703,905 (Slim) and 8,632,865 (Standard) parameters, which no "
    "longer matches the paper.",
    "The bottleneck (deepest, 512-channel stage) is treated as a slim stage with "
    "a single convolution.",
    "Dropout is applied after ReLU at the end of each encoder stage and the "
    "bottleneck. Decoder dropout is configurable and defaults to 0.0, since the "
    "paper only describes dropout in the contracting path.",
    "Dropout is element-wise (nn.Dropout), not channel-wise (nn.Dropout2d); "
    "'spatial' is available via dropout_kind. Dropout carries no parameters, so "
    "this choice does not affect the parameter count.",
    "The output layer is a 1x1 convolution emitting raw logits; the paper's "
    "final activation is applied outside the model.",
)

#: Preset keyword arguments for :class:`SlimUNet`.
SLIM_UNET_PRESETS: dict[str, dict[str, object]] = {
    "paper": {
        "in_channels": 1,
        "out_channels": 1,
        "channels": (32, 64, 128, 256, 512),
        "encoder_convs": 1,
        "decoder_convs": 1,
        "dropout": 0.125,
        "decoder_dropout": 0.0,
        "normalization": "batch_norm",
        "conv_bias": True,
        "upsampling": "transposed_conv",
        "up_kernel_size": 3,
    },
    # Same architecture as ``paper``; the production preset differs only in the
    # input resolution used at train/deploy time (a config-level choice), so the
    # deployed network stays lightweight and export-friendly.
    "production": {
        "in_channels": 1,
        "out_channels": 1,
        "channels": (32, 64, 128, 256, 512),
        "encoder_convs": 1,
        "decoder_convs": 1,
        "dropout": 0.125,
        "decoder_dropout": 0.0,
        "normalization": "batch_norm",
        "conv_bias": True,
        "upsampling": "transposed_conv",
        "up_kernel_size": 3,
    },
}


class SlimUNet(nn.Module):
    """Unofficial paper-based Slim U-Net for binary ultrasound segmentation.

    Args:
        in_channels: Input channels; 1 for grayscale B-mode ultrasound.
        out_channels: Output channels; 1 for binary bladder-lumen logits.
        channels: Channel progression from the highest resolution to the
            bottleneck. The default ``(32, 64, 128, 256, 512)`` is the
            paper-faithful configuration and implies four down/up stages.
        encoder_convs: 3x3 convolutions per encoder stage. ``1`` is the slim
            (paper) setting.
        decoder_convs: 3x3 convolutions per decoder stage. ``1`` is the
            paper-faithful setting (see module docstring); ``2`` reproduces the
            ``SlimUNetDecoderDoubleConv`` comparison variant.
        dropout: Dropout probability for encoder stages and the bottleneck.
        decoder_dropout: Dropout probability for decoder stages.
        normalization: Normalization layer name.
        conv_bias: Whether 3x3 convolutions carry a bias term.
        upsampling: ``transposed_conv``, ``bilinear`` or ``nearest``.
        up_kernel_size: Transposed-convolution kernel (2, 3 or 4).
        dropout_kind: ``standard`` (element-wise) or ``spatial``.

    Note:
        ``forward`` returns **raw logits**; no sigmoid is applied inside the
        model so that ``BCEWithLogitsLoss`` stays numerically stable and ONNX
        export mirrors the training graph.
    """

    #: Architecture family name, written into checkpoints.
    model_name = "slim_unet"

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        encoder_convs: int = 1,
        decoder_convs: int = 1,
        dropout: float = 0.125,
        decoder_dropout: Optional[float] = 0.0,
        normalization: NormalizationName = "batch_norm",
        conv_bias: bool = True,
        upsampling: UpsamplingName = "transposed_conv",
        up_kernel_size: int = 3,
        dropout_kind: DropoutKind = "standard",
    ) -> None:
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if len(channels) < 2:
            raise ValueError(
                f"channels must contain at least two stages, got {channels!r}."
            )
        if any(c < 1 for c in channels):
            raise ValueError(f"channel widths must be positive, got {channels!r}.")
        if encoder_convs < 1 or decoder_convs < 1:
            raise ValueError(
                "encoder_convs and decoder_convs must be >= 1, got "
                f"{encoder_convs} and {decoder_convs}."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.depth = len(channels) - 1
        self.encoder_convs = encoder_convs
        self.decoder_convs = decoder_convs
        self.upsampling = upsampling
        decoder_dropout = dropout if decoder_dropout is None else decoder_dropout

        # --- Contracting path: conv -> BN -> ReLU -> Dropout, then max-pool ---
        self.encoder = nn.ModuleList(
            [
                ConvBlock(
                    in_channels if i == 0 else channels[i - 1],
                    channels[i],
                    num_convs=encoder_convs,
                    normalization=normalization,
                    conv_bias=conv_bias,
                    dropout=dropout,
                    dropout_kind=dropout_kind,
                )
                for i in range(self.depth)
            ]
        )
        self.pool = Downsample(2)
        self.bottleneck = ConvBlock(
            channels[self.depth - 1],
            channels[self.depth],
            num_convs=encoder_convs,
            normalization=normalization,
            conv_bias=conv_bias,
            dropout=dropout,
            dropout_kind=dropout_kind,
        )

        # --- Expanding path: upsample -> concat skip -> conv block ---
        decoder: list[UpsampleBlock] = []
        deep = channels[self.depth]
        for i in range(self.depth):
            skip = channels[self.depth - i - 1]
            decoder.append(
                UpsampleBlock(
                    deep_channels=deep,
                    skip_channels=skip,
                    out_channels=skip,
                    num_convs=decoder_convs,
                    upsampling=upsampling,
                    up_kernel_size=up_kernel_size,
                    normalization=normalization,
                    conv_bias=conv_bias,
                    dropout=decoder_dropout,
                    dropout_kind=dropout_kind,
                )
            )
            deep = skip
        self.decoder = nn.ModuleList(decoder)
        self.outc = OutConv(channels[0], out_channels)

        # Aliases matching the original repository's attribute names, so the
        # legacy scripts and evaluation helpers accept this model too.
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = upsampling == "bilinear"

    @classmethod
    def from_preset(cls, preset: str, **overrides: object) -> "SlimUNet":
        """Instantiate from a named preset, applying ``overrides`` on top.

        Raises:
            KeyError: If ``preset`` is unknown.
        """
        if preset not in SLIM_UNET_PRESETS:
            raise KeyError(
                f"Unknown SlimUNet preset {preset!r}. Available: {sorted(SLIM_UNET_PRESETS)}"
            )
        kwargs = dict(SLIM_UNET_PRESETS[preset])
        kwargs.update(overrides)
        return cls(**kwargs)  # type: ignore[arg-type]

    @property
    def required_size_multiple(self) -> int:
        """Spatial multiple that input height and width must satisfy."""
        return 2**self.depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``B x in_channels x H x W`` to ``B x out_channels x H x W`` logits.

        Raises:
            ValueError: If the input is not 4-D, has the wrong channel count, or
                its spatial size is not divisible by ``2 ** depth`` (which would
                make encoder and decoder resolutions inconsistent).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected a 4-D tensor B x C x H x W, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got {x.shape[1]}."
            )
        multiple = self.required_size_multiple
        if x.shape[-2] % multiple or x.shape[-1] % multiple:
            raise ValueError(
                f"Input spatial size {tuple(x.shape[-2:])} must be divisible by {multiple} "
                f"for a depth-{self.depth} SlimUNet. Resize or pad the input first."
            )

        skips: list[torch.Tensor] = []
        h = x
        for stage in self.encoder:
            h = stage(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for block, skip in zip(self.decoder, reversed(skips)):
            h = block(h, skip)
        return self.outc(h)
