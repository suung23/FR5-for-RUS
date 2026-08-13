"""Architecture tests: shapes, parameter counts and baseline preservation."""

from __future__ import annotations

import pytest
import torch

from src.models.blocks import transposed_conv_padding
from src.models.registry import available_models, build_model
from src.models.report import build_architecture_report
from src.models.slim_unet import PAPER_PARAMETER_COUNTS, SlimUNet
from src.models.standard_unet import StandardUNet, remap_legacy_unet_state_dict
from unet import UNet as LegacyUNet


def trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -- shapes ---------------------------------------------------------------
@pytest.mark.parametrize("size", [(128, 128), (256, 256), (64, 96)])
def test_slim_unet_output_shape_matches_input(size: tuple[int, int]) -> None:
    model = SlimUNet.from_preset("paper").eval()
    x = torch.zeros(1, 1, *size)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, *size)


@pytest.mark.parametrize("size", [(128, 128), (256, 256)])
def test_standard_unet_output_shape_matches_input(size: tuple[int, int]) -> None:
    model = StandardUNet.from_preset("paper", in_channels=1, out_channels=1).eval()
    x = torch.zeros(2, 1, *size)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, *size)


def test_slim_unet_rejects_indivisible_input_size() -> None:
    model = SlimUNet.from_preset("paper").eval()
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.zeros(1, 1, 130, 128))


def test_slim_unet_rejects_wrong_rank_and_channels() -> None:
    model = SlimUNet.from_preset("paper").eval()
    with pytest.raises(ValueError, match="4-D tensor"):
        model(torch.zeros(1, 128, 128))
    with pytest.raises(ValueError, match="input channel"):
        model(torch.zeros(1, 3, 128, 128))


def test_slim_unet_returns_logits_not_probabilities() -> None:
    """A sigmoid inside forward would confine the output to (0, 1)."""
    torch.manual_seed(0)
    model = SlimUNet.from_preset("paper").eval()
    with torch.no_grad():
        y = model(torch.randn(4, 1, 64, 64) * 5.0)
    assert y.min() < 0.0, "logits should be able to go negative"


# -- parameter counts -----------------------------------------------------
def test_paper_parameter_counts_reproduce_exactly() -> None:
    slim = build_model({"name": "slim_unet", "preset": "paper"})
    standard = build_model(
        {"name": "standard_unet", "preset": "paper", "input_channels": 1, "output_channels": 1}
    )
    assert trainable(slim) == PAPER_PARAMETER_COUNTS["slim_unet"] == 4_705_377
    assert trainable(standard) == PAPER_PARAMETER_COUNTS["standard_unet"] == 8_635_809


def test_standard_unet_has_more_parameters_than_slim_unet() -> None:
    slim = build_model({"name": "slim_unet", "preset": "paper"})
    standard = build_model(
        {"name": "standard_unet", "preset": "paper", "input_channels": 1, "output_channels": 1}
    )
    assert trainable(standard) > trainable(slim)


def test_double_conv_decoder_variant_does_not_match_the_paper() -> None:
    """The rejected decoder variant is still constructible, for comparison."""
    variant = build_model({"name": "slim_unet", "preset": "paper", "decoder_convs": 2})
    assert trainable(variant) == 5_490_177
    assert trainable(variant) != PAPER_PARAMETER_COUNTS["slim_unet"]


def test_parameter_count_is_deterministic() -> None:
    counts = {trainable(build_model({"name": "slim_unet", "preset": "paper"})) for _ in range(3)}
    assert len(counts) == 1


def test_architecture_report_contents() -> None:
    model = SlimUNet.from_preset("paper")
    report = build_architecture_report(model, (1, 1, 128, 128))
    assert report.model_name == "slim_unet"
    assert report.input_shape == [1, 1, 128, 128]
    assert report.output_shape == [1, 1, 128, 128]
    assert report.trainable_parameters == PAPER_PARAMETER_COUNTS["slim_unet"]
    assert report.parameter_difference_from_reference == 0
    assert report.estimated_macs and report.estimated_macs > 0
    assert report.estimated_flops == 2 * report.estimated_macs
    assert any(layer.type == "Conv2d" for layer in report.layers)
    assert "layers" in report.to_dict()


def test_architecture_report_survives_json_round_trip() -> None:
    import json

    report = build_architecture_report(SlimUNet.from_preset("paper"), (1, 1, 64, 64))
    assert json.loads(report.to_json())["trainable_parameters"] == 4_705_377


# -- baseline preservation -------------------------------------------------
@pytest.mark.parametrize("bilinear", [False, True])
def test_standard_unet_matches_the_original_implementation(bilinear: bool) -> None:
    """StandardUNet must stay a faithful generalisation of the original U-Net."""
    torch.manual_seed(0)
    legacy = LegacyUNet(n_channels=3, n_classes=2, bilinear=bilinear).eval()
    generalised = StandardUNet(
        in_channels=3,
        out_channels=2,
        base_channels=64,
        depth=4,
        upsampling="bilinear" if bilinear else "transposed_conv",
    ).eval()

    assert trainable(legacy) == trainable(generalised)
    generalised.load_state_dict(remap_legacy_unet_state_dict(legacy.state_dict(), generalised))

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(legacy(x), generalised(x), atol=1e-6)


def test_legacy_remap_rejects_mismatched_checkpoints() -> None:
    legacy = LegacyUNet(n_channels=3, n_classes=2)
    wrong = StandardUNet(in_channels=1, out_channels=1, base_channels=32, depth=4)
    with pytest.raises(ValueError):
        remap_legacy_unet_state_dict(legacy.state_dict(), wrong)


def test_models_expose_legacy_attribute_aliases() -> None:
    slim = SlimUNet.from_preset("paper")
    assert slim.n_channels == 1 and slim.n_classes == 1


# -- registry and configuration -------------------------------------------
def test_registry_lists_both_architectures() -> None:
    assert available_models() == ["slim_unet", "standard_unet"]


def test_build_model_rejects_unknown_name_and_preset() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        build_model({"name": "not_a_model"})
    with pytest.raises(KeyError, match="Unknown preset"):
        build_model({"name": "slim_unet", "preset": "nonexistent"})
    with pytest.raises(KeyError, match="requires a 'name'"):
        build_model({"preset": "paper"})


def test_build_model_reports_invalid_constructor_arguments() -> None:
    with pytest.raises(TypeError, match="Invalid arguments"):
        build_model({"name": "slim_unet", "preset": "paper", "base_channels": 32})


def test_configurable_fields_change_the_model() -> None:
    model = build_model(
        {
            "name": "slim_unet",
            "input_channels": 3,
            "output_channels": 2,
            "channels": [16, 32, 64],
            "dropout": 0.2,
            "normalization": "instance_norm",
            "conv_bias": False,
            "upsampling": "bilinear",
        }
    )
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 2, 32, 32)
    assert model.depth == 2


def test_normalization_and_dropout_validation() -> None:
    from src.models.blocks import build_dropout, build_normalization

    with pytest.raises(ValueError, match="Unsupported normalization"):
        build_normalization("layer_norm", 8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        build_dropout("standard", 1.5)
    assert isinstance(build_dropout("standard", 0.0), torch.nn.Identity)


@pytest.mark.parametrize("kernel,expected", [(2, (0, 0)), (3, (1, 1)), (4, (1, 0))])
def test_transposed_conv_padding(kernel: int, expected: tuple[int, int]) -> None:
    assert transposed_conv_padding(kernel) == expected


def test_transposed_conv_padding_rejects_unsupported_kernel() -> None:
    with pytest.raises(ValueError, match="not supported"):
        transposed_conv_padding(5)


def test_dropout_is_inactive_in_eval_mode() -> None:
    torch.manual_seed(0)
    model = SlimUNet.from_preset("paper", dropout=0.5).eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))
