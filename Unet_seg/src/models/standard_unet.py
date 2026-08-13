"""Standard U-Net baseline.

This module generalises the U-Net shipped with ``milesial/Pytorch-UNet``
(GPL-3.0, see ``LICENSE``) so that channel width, depth, convolution bias,
normalization and upsampling can be configured. With its default arguments it
is architecturally identical to the original ``unet.UNet``: same layer
sequence, same tensor shapes, same parameter count. The original class is left
untouched in :mod:`unet` and remains the reference for the legacy CLI.

Two presets are provided:

``original``
    ``base_channels=64``, bias-free convolutions, ``ConvTranspose2d(k=2)``.
    Reproduces ``unet.UNet`` exactly.
``paper``
    ``base_channels=32``, biased convolutions, ``ConvTranspose2d(k=3, s=2)``.
    Reproduces the 8,635,809-parameter Standard U-Net baseline reported by
    Raina et al. exactly (see README, "Parameter-count reconstruction").
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

__all__ = ["StandardUNet", "STANDARD_UNET_PRESETS", "remap_legacy_unet_state_dict"]

#: Preset keyword arguments for :class:`StandardUNet`.
STANDARD_UNET_PRESETS: dict[str, dict[str, object]] = {
    "original": {
        "base_channels": 64,
        "depth": 4,
        "conv_bias": False,
        "up_kernel_size": 2,
        "upsampling": "transposed_conv",
        "normalization": "batch_norm",
        "dropout": 0.0,
    },
    "paper": {
        "base_channels": 32,
        "depth": 4,
        "conv_bias": True,
        "up_kernel_size": 3,
        "upsampling": "transposed_conv",
        "normalization": "batch_norm",
        "dropout": 0.0,
    },
}


class StandardUNet(nn.Module):
    """Symmetric U-Net with two 3x3 convolutions per encoder and decoder stage.

    The contracting path is ``ConvBlock(2 convs) -> MaxPool`` repeated ``depth``
    times followed by a bottleneck ``ConvBlock``; the expanding path mirrors it
    with upsample-concat-``ConvBlock(2 convs)`` stages and a final 1x1
    convolution producing raw logits.

    Args:
        in_channels: Input channels (1 for grayscale B-mode ultrasound).
        out_channels: Output channels / classes. ``1`` means binary
            segmentation supervised with ``BCEWithLogitsLoss``.
        base_channels: Channel width of the highest-resolution stage.
        depth: Number of down/up sampling stages.
        upsampling: ``transposed_conv``, ``bilinear`` or ``nearest``.
        up_kernel_size: Kernel of the transposed convolution (2, 3 or 4).
        normalization: Normalization layer name.
        conv_bias: Whether 3x3 convolutions carry a bias term.
        dropout: Dropout probability applied at the end of every encoder block
            and the bottleneck.
        decoder_dropout: Dropout probability for decoder blocks. ``None``
            disables decoder dropout.
        dropout_kind: ``standard`` (element-wise) or ``spatial`` (channel-wise).

    Note:
        ``forward`` always returns **raw logits**. Sigmoid/softmax is applied by
        the caller during inference, metric computation and visualisation only.
    """

    #: Architecture family name, written into checkpoints.
    model_name = "standard_unet"

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        upsampling: UpsamplingName = "transposed_conv",
        up_kernel_size: int = 2,
        normalization: NormalizationName = "batch_norm",
        conv_bias: bool = False,
        dropout: float = 0.0,
        decoder_dropout: Optional[float] = None,
        dropout_kind: DropoutKind = "standard",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")
        if base_channels < 1:
            raise ValueError(f"base_channels must be >= 1, got {base_channels}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.upsampling = upsampling
        self.dropout = dropout
        decoder_dropout = dropout if decoder_dropout is None else decoder_dropout

        # ``bilinear``/``nearest`` upsampling does not reduce channels, so the
        # bottleneck is halved to keep the decoder cost comparable. This matches
        # the ``factor = 2 if bilinear else 1`` trick of the original repository.
        factor = 2 if upsampling != "transposed_conv" else 1
        channels: list[int] = [base_channels * 2**i for i in range(depth + 1)]
        channels[depth] //= factor
        self.channels: Sequence[int] = tuple(channels)

        block_kwargs = dict(
            num_convs=2,
            normalization=normalization,
            conv_bias=conv_bias,
            dropout_kind=dropout_kind,
        )

        self.inc = ConvBlock(in_channels, channels[0], dropout=dropout, **block_kwargs)
        self.pool = Downsample(2)
        self.downs = nn.ModuleList(
            [
                ConvBlock(channels[i], channels[i + 1], dropout=dropout, **block_kwargs)
                for i in range(depth)
            ]
        )

        ups: list[UpsampleBlock] = []
        deep = channels[depth]
        for i in range(depth):
            skip = channels[depth - i - 1]
            # Mirror the original: intermediate decoder stages are also halved
            # when parameter-free upsampling is used. ``deep`` tracks the actual
            # width produced by the previous decoder stage, which differs from
            # ``channels[depth - i]`` once the halving factor is active.
            out = skip // factor if i < depth - 1 else skip
            ups.append(
                UpsampleBlock(
                    deep_channels=deep,
                    skip_channels=skip,
                    out_channels=out,
                    num_convs=2,
                    upsampling=upsampling,
                    up_kernel_size=up_kernel_size,
                    normalization=normalization,
                    conv_bias=conv_bias,
                    dropout=decoder_dropout,
                    dropout_kind=dropout_kind,
                )
            )
            deep = out
        self.ups = nn.ModuleList(ups)
        self.outc = OutConv(channels[0], out_channels)

        # Backwards-compatible aliases used by the legacy scripts in the
        # repository root (``train.py``, ``predict.py``, ``evaluate.py``).
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = upsampling == "bilinear"

    @classmethod
    def from_preset(cls, preset: str, **overrides: object) -> "StandardUNet":
        """Instantiate from a named preset, applying ``overrides`` on top.

        Raises:
            KeyError: If ``preset`` is unknown.
        """
        if preset not in STANDARD_UNET_PRESETS:
            raise KeyError(
                f"Unknown StandardUNet preset {preset!r}. "
                f"Available: {sorted(STANDARD_UNET_PRESETS)}"
            )
        kwargs = dict(STANDARD_UNET_PRESETS[preset])
        kwargs.update(overrides)
        return cls(**kwargs)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``B x in_channels x H x W`` to ``B x out_channels x H x W`` logits."""
        skips: list[torch.Tensor] = []
        h = self.inc(x)
        for down in self.downs:
            skips.append(h)
            h = down(self.pool(h))
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip)
        return self.outc(h)


def remap_legacy_unet_state_dict(
    state_dict: dict[str, torch.Tensor], model: StandardUNet
) -> dict[str, torch.Tensor]:
    """Convert a legacy ``unet.UNet`` state dict to :class:`StandardUNet` keys.

    The two implementations contain the *same* parameters and buffers in the
    *same* order with the *same* shapes; only the module path names differ. The
    remap is therefore positional and is validated shape-by-shape.

    Args:
        state_dict: State dict produced by the original ``unet.UNet``. A
            ``mask_values`` entry, if present, is ignored.
        model: Target model providing the destination key order.

    Returns:
        A state dict keyed for ``model``.

    Raises:
        ValueError: If the entry count or any tensor shape does not line up,
            which means the checkpoint does not correspond to this topology.
    """
    source = {k: v for k, v in state_dict.items() if k != "mask_values"}
    target_keys = list(model.state_dict().keys())
    target_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    if len(source) != len(target_keys):
        raise ValueError(
            f"Legacy checkpoint has {len(source)} entries but {type(model).__name__} "
            f"expects {len(target_keys)}. The checkpoint does not match this topology."
        )

    remapped: dict[str, torch.Tensor] = {}
    for target_key, (source_key, tensor) in zip(target_keys, source.items()):
        if tuple(tensor.shape) != target_shapes[target_key]:
            raise ValueError(
                f"Shape mismatch remapping legacy key {source_key!r} -> {target_key!r}: "
                f"{tuple(tensor.shape)} vs {target_shapes[target_key]}."
            )
        remapped[target_key] = tensor
    return remapped
