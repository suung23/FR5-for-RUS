"""Building blocks shared by the Standard U-Net baseline and Slim U-Net.

The blocks in this module are deliberately parameterised so that a single
implementation can express both

* the original ``milesial/Pytorch-UNet`` Standard U-Net (bias-free 3x3
  convolutions, ``ConvTranspose2d(kernel_size=2, stride=2)`` upsampling), and
* the Standard / Slim U-Net variants described by Raina et al.
  (biased 3x3 convolutions, BatchNorm, ``ConvTranspose2d(kernel_size=3,
  stride=2)`` upsampling).

See ``docs/`` notes in the README for how the paper preset was derived.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NormalizationName = Literal["batch_norm", "instance_norm", "group_norm", "none"]
DropoutKind = Literal["standard", "spatial", "none"]
UpsamplingName = Literal["transposed_conv", "bilinear", "nearest"]

__all__ = [
    "build_normalization",
    "build_dropout",
    "ConvBlock",
    "Downsample",
    "UpsampleBlock",
    "OutConv",
    "transposed_conv_padding",
    "NormalizationName",
    "DropoutKind",
    "UpsamplingName",
]


def build_normalization(name: NormalizationName, num_channels: int, num_groups: int = 8) -> nn.Module:
    """Return a normalization layer for ``num_channels`` feature maps.

    Args:
        name: Normalization identifier.
        num_channels: Number of channels the layer normalizes.
        num_groups: Group count used only when ``name == "group_norm"``. The
            effective group count is clamped to divide ``num_channels``.

    Raises:
        ValueError: If ``name`` is not a supported normalization.
    """
    if name == "batch_norm":
        return nn.BatchNorm2d(num_channels)
    if name == "instance_norm":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if name == "group_norm":
        groups = num_groups
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if name == "none":
        return nn.Identity()
    raise ValueError(
        f"Unsupported normalization {name!r}. "
        "Expected one of: batch_norm, instance_norm, group_norm, none."
    )


def build_dropout(kind: DropoutKind, p: float) -> nn.Module:
    """Return a dropout layer, or ``nn.Identity`` when dropout is disabled.

    Args:
        kind: ``"standard"`` for element-wise dropout, ``"spatial"`` for
            channel-wise (``Dropout2d``) dropout, ``"none"`` to disable.
        p: Drop probability in ``[0, 1)``.

    Raises:
        ValueError: If ``p`` is outside ``[0, 1)`` or ``kind`` is unknown.
    """
    if not 0.0 <= p < 1.0:
        raise ValueError(f"Dropout probability must be in [0, 1), got {p}.")
    if kind == "none" or p == 0.0:
        return nn.Identity()
    if kind == "standard":
        return nn.Dropout(p)
    if kind == "spatial":
        return nn.Dropout2d(p)
    raise ValueError(f"Unsupported dropout kind {kind!r}. Expected: standard, spatial, none.")


class ConvBlock(nn.Module):
    """``num_convs`` x (3x3 conv -> norm -> ReLU) followed by a single dropout.

    ``num_convs == 2`` reproduces the classic U-Net ``DoubleConv``;
    ``num_convs == 1`` is the "slim" variant used by Slim U-Net, where the
    contracting path performs a single convolution per resolution.

    Note:
        Dropout is applied once at the end of the block, matching the
        ``conv -> BN -> ReLU -> Dropout`` ordering given in the Slim U-Net
        paper for its encoder stages.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        mid_channels: Optional[int] = None,
        normalization: NormalizationName = "batch_norm",
        conv_bias: bool = False,
        dropout: float = 0.0,
        dropout_kind: DropoutKind = "standard",
    ) -> None:
        super().__init__()
        if num_convs < 1:
            raise ValueError(f"ConvBlock requires at least one convolution, got {num_convs}.")
        mid_channels = mid_channels or out_channels

        layers: list[nn.Module] = []
        for index in range(num_convs):
            src = in_channels if index == 0 else mid_channels
            dst = out_channels if index == num_convs - 1 else mid_channels
            layers.extend(
                [
                    nn.Conv2d(src, dst, kernel_size=3, padding=1, bias=conv_bias),
                    build_normalization(normalization, dst),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(build_dropout(dropout_kind, dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample(nn.Module):
    """2x2 max pooling."""

    def __init__(self, kernel_size: int = 2) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


def transposed_conv_padding(kernel_size: int) -> tuple[int, int]:
    """Return ``(padding, output_padding)`` that make a stride-2 transpose exactly 2x.

    Raises:
        ValueError: If ``kernel_size`` is not one of 2, 3 or 4.
    """
    if kernel_size == 2:
        return 0, 0
    if kernel_size == 3:
        return 1, 1
    if kernel_size == 4:
        return 1, 0
    raise ValueError(f"up_kernel_size={kernel_size} is not supported; expected 2, 3 or 4.")


class UpsampleBlock(nn.Module):
    """Upsample by 2, concatenate the encoder skip feature, then convolve.

    Channel widths are given explicitly rather than inferred, because the two
    upsampling families consume them differently:

    * ``transposed_conv``: a learned ``ConvTranspose2d`` halves ``deep_channels``
      while doubling the spatial resolution, so the convolution block sees
      ``deep_channels // 2 + skip_channels`` inputs.
    * ``bilinear`` / ``nearest``: parameter-free interpolation keeps the channel
      count, so the convolution block sees ``deep_channels + skip_channels``
      inputs and performs the reduction itself. This mirrors the original
      ``milesial`` implementation.

    Args:
        deep_channels: Channels of the lower-resolution feature being upsampled.
        skip_channels: Channels of the encoder skip connection.
        out_channels: Channels produced by this decoder stage.
        num_convs: Convolutions in the decoder block (2 = standard, 1 = slim).
    """

    def __init__(
        self,
        deep_channels: int,
        skip_channels: int,
        out_channels: int,
        num_convs: int = 2,
        upsampling: UpsamplingName = "transposed_conv",
        up_kernel_size: int = 2,
        normalization: NormalizationName = "batch_norm",
        conv_bias: bool = False,
        dropout: float = 0.0,
        dropout_kind: DropoutKind = "standard",
    ) -> None:
        super().__init__()
        self.upsampling = upsampling

        if upsampling == "transposed_conv":
            padding, output_padding = transposed_conv_padding(up_kernel_size)
            self.up: nn.Module = nn.ConvTranspose2d(
                deep_channels,
                deep_channels // 2,
                kernel_size=up_kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
            )
            concat_channels = deep_channels // 2 + skip_channels
            mid_channels = None
        elif upsampling in ("bilinear", "nearest"):
            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear" if upsampling == "bilinear" else "nearest",
                align_corners=True if upsampling == "bilinear" else None,
            )
            concat_channels = deep_channels + skip_channels
            mid_channels = concat_channels // 2 if num_convs > 1 else None
        else:
            raise ValueError(
                f"Unsupported upsampling {upsampling!r}. "
                "Expected: transposed_conv, bilinear, nearest."
            )

        self.conv = ConvBlock(
            concat_channels,
            out_channels,
            num_convs=num_convs,
            mid_channels=mid_channels,
            normalization=normalization,
            conv_bias=conv_bias,
            dropout=dropout,
            dropout_kind=dropout_kind,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample ``x``, pad it to ``skip``'s size, concatenate and convolve."""
        x = self.up(x)
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    """Final 1x1 convolution mapping features to raw class logits."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
