"""Architecture reporting: layer table, parameter counts and MAC estimation.

The MAC estimate is computed with forward hooks over convolution, transposed
convolution, linear and normalization layers. It is an analytic count of
multiply-accumulate operations for a single forward pass; it is not a measured
runtime figure and is reported as such.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from .slim_unet import PAPER_PARAMETER_COUNTS

__all__ = ["LayerInfo", "ArchitectureReport", "estimate_macs", "build_architecture_report"]


@dataclass
class LayerInfo:
    """Per-layer entry of an :class:`ArchitectureReport`."""

    name: str
    type: str
    input_shape: Optional[list[int]]
    output_shape: Optional[list[int]]
    trainable_parameters: int
    total_parameters: int
    macs: Optional[int]


@dataclass
class ArchitectureReport:
    """Summary of a model's shape behaviour and parameter budget."""

    model_name: str
    model_class: str
    input_shape: list[int]
    output_shape: list[int]
    trainable_parameters: int
    total_parameters: int
    non_trainable_parameters: int
    buffer_elements: int
    estimated_macs: Optional[int]
    estimated_flops: Optional[int]
    macs_backend: str
    reference_parameter_count: Optional[int] = None
    parameter_difference_from_reference: Optional[int] = None
    layers: list[LayerInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return the report as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_table(self, max_layers: Optional[int] = None) -> str:
        """Render a fixed-width text table of the report."""
        header = (
            f"{'layer':<38} {'type':<18} {'output shape':<22} {'params':>12} {'MACs':>15}"
        )
        lines = [
            f"Model: {self.model_name} ({self.model_class})",
            f"Input : {tuple(self.input_shape)}",
            f"Output: {tuple(self.output_shape)}",
            "",
            header,
            "-" * len(header),
        ]
        layers = self.layers if max_layers is None else self.layers[:max_layers]
        for layer in layers:
            shape = str(tuple(layer.output_shape)) if layer.output_shape else "-"
            macs = f"{layer.macs:,}" if layer.macs else "-"
            lines.append(
                f"{layer.name[:38]:<38} {layer.type[:18]:<18} {shape:<22} "
                f"{layer.trainable_parameters:>12,} {macs:>15}"
            )
        if max_layers is not None and len(self.layers) > max_layers:
            lines.append(f"... ({len(self.layers) - max_layers} more layers)")
        lines += [
            "-" * len(header),
            f"Trainable parameters     : {self.trainable_parameters:,}",
            f"Total parameters         : {self.total_parameters:,}",
            f"Non-trainable parameters : {self.non_trainable_parameters:,}",
            f"Buffer elements          : {self.buffer_elements:,}",
        ]
        if self.estimated_macs is not None:
            lines.append(
                f"Estimated MACs           : {self.estimated_macs:,} "
                f"({self.estimated_macs / 1e9:.3f} G) [{self.macs_backend}]"
            )
            lines.append(
                f"Estimated FLOPs (2x MACs): {self.estimated_flops:,} "
                f"({self.estimated_flops / 1e9:.3f} G)"
            )
        else:
            lines.append(f"Estimated MACs           : unavailable [{self.macs_backend}]")
        if self.reference_parameter_count is not None:
            lines.append(
                f"Paper-reported parameters: {self.reference_parameter_count:,}"
            )
            diff = self.parameter_difference_from_reference or 0
            verdict = "exact match" if diff == 0 else f"{diff:+,} vs reference"
            lines.append(f"Difference               : {verdict}")
        return "\n".join(lines)


_MAC_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)


def _module_macs(module: nn.Module, output: torch.Tensor) -> Optional[int]:
    """Analytic multiply-accumulate count for one forward pass of ``module``."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        kernel = 1
        for k in module.kernel_size:
            kernel *= k
        spatial = 1
        for s in output.shape[2:]:
            spatial *= int(s)
        in_per_group = module.in_channels // module.groups
        return int(output.shape[0]) * module.out_channels * spatial * kernel * in_per_group
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        kernel = 1
        for k in module.kernel_size:
            kernel *= k
        spatial = 1
        for s in output.shape[2:]:
            spatial *= int(s)
        in_per_group = module.in_channels // module.groups
        # A stride-2 transpose scatters each input pixel; counting over the
        # output grid with the same kernel is the standard convention and is an
        # upper bound on the executed multiply-accumulates.
        return int(output.shape[0]) * module.out_channels * spatial * kernel * in_per_group
    if isinstance(module, nn.Linear):
        return int(output.numel()) * module.in_features
    return None


@torch.no_grad()
def estimate_macs(model: nn.Module, input_shape: Sequence[int]) -> tuple[Optional[int], str]:
    """Estimate total MACs for one forward pass.

    Args:
        model: Model to profile (moved to CPU by the caller if needed).
        input_shape: Full input shape including the batch dimension.

    Returns:
        ``(macs, backend)``. ``macs`` is ``None`` if the forward pass failed, in
        which case ``backend`` explains why.
    """
    total = 0
    handles: list[Any] = []

    def hook(module: nn.Module, _inputs: Any, output: Any) -> None:
        nonlocal total
        if isinstance(output, torch.Tensor):
            macs = _module_macs(module, output)
            if macs:
                total += macs

    for module in model.modules():
        if isinstance(module, _MAC_MODULES):
            handles.append(module.register_forward_hook(hook))
    try:
        model.eval()
        model(torch.zeros(*input_shape))
    except Exception as exc:  # pragma: no cover - reported, not raised
        for handle in handles:
            handle.remove()
        return None, f"analytic-hooks (failed: {type(exc).__name__}: {exc})"
    for handle in handles:
        handle.remove()
    return total, "analytic-hooks"


@torch.no_grad()
def build_architecture_report(
    model: nn.Module,
    input_shape: Sequence[int],
    reference_parameter_count: Optional[int] = None,
    model_name: Optional[str] = None,
) -> ArchitectureReport:
    """Run a forward pass and collect a full architecture report.

    Args:
        model: Model to report on.
        input_shape: Full input shape including batch, e.g. ``(1, 1, 128, 128)``.
        reference_parameter_count: Paper-reported count to compare against. When
            ``None``, it is looked up from :data:`PAPER_PARAMETER_COUNTS` using
            the model's ``model_name`` attribute.
        model_name: Override for the reported model name.

    Returns:
        A populated :class:`ArchitectureReport`.

    Raises:
        RuntimeError: If the forward pass fails, since a report without shapes
            would be misleading.
    """
    input_shape = list(int(s) for s in input_shape)
    name = model_name or getattr(model, "model_name", type(model).__name__)
    if reference_parameter_count is None:
        reference_parameter_count = PAPER_PARAMETER_COUNTS.get(name)

    was_training = model.training
    model.eval()

    records: dict[str, dict[str, Any]] = {}
    handles: list[Any] = []

    def make_hook(layer_name: str):
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            in_shape = (
                list(inputs[0].shape)
                if inputs and isinstance(inputs[0], torch.Tensor)
                else None
            )
            out_shape = list(output.shape) if isinstance(output, torch.Tensor) else None
            macs = (
                _module_macs(module, output)
                if isinstance(output, torch.Tensor) and isinstance(module, _MAC_MODULES)
                else None
            )
            records[layer_name] = {
                "type": type(module).__name__,
                "input_shape": in_shape,
                "output_shape": out_shape,
                "macs": macs,
            }

        return hook

    leaf_names: list[str] = []
    for layer_name, module in model.named_modules():
        if layer_name == "" or list(module.children()):
            continue  # report leaf modules only
        leaf_names.append(layer_name)
        handles.append(module.register_forward_hook(make_hook(layer_name)))

    try:
        output = model(torch.zeros(*input_shape))
    except Exception as exc:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
        raise RuntimeError(
            f"Forward pass with input shape {tuple(input_shape)} failed: {exc}"
        ) from exc
    finally:
        for handle in handles:
            handle.remove()

    modules = dict(model.named_modules())
    layers: list[LayerInfo] = []
    total_macs = 0
    for layer_name in leaf_names:
        module = modules[layer_name]
        record = records.get(layer_name, {})
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        total = sum(p.numel() for p in module.parameters(recurse=False))
        macs = record.get("macs")
        if macs:
            total_macs += int(macs)
        layers.append(
            LayerInfo(
                name=layer_name,
                type=record.get("type", type(module).__name__),
                input_shape=record.get("input_shape"),
                output_shape=record.get("output_shape"),
                trainable_parameters=trainable,
                total_parameters=total,
                macs=int(macs) if macs else None,
            )
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())

    if was_training:
        model.train()

    return ArchitectureReport(
        model_name=name,
        model_class=type(model).__name__,
        input_shape=input_shape,
        output_shape=list(output.shape),
        trainable_parameters=trainable_params,
        total_parameters=all_params,
        non_trainable_parameters=all_params - trainable_params,
        buffer_elements=buffers,
        estimated_macs=total_macs or None,
        estimated_flops=2 * total_macs if total_macs else None,
        macs_backend="analytic-hooks",
        reference_parameter_count=reference_parameter_count,
        parameter_difference_from_reference=(
            trainable_params - reference_parameter_count
            if reference_parameter_count is not None
            else None
        ),
        layers=layers,
    )
