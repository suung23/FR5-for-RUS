"""Report figures for training runs and quality-score trust validation.

Importing this package pulls in matplotlib, which is an optional extra
(``pip install -e ".[viz]"``). Nothing on the real-time control path imports it,
so a deployment without matplotlib is fully functional -- it simply cannot draw
the validation report.
"""

from .plots import (  # noqa: F401
    CATEGORICAL,
    PALETTE,
    apply_style,
    plot_component_attribution,
    plot_force_response,
    plot_latency,
    plot_loss_decomposition,
    plot_optimization_health,
    plot_per_patient,
    plot_quality_vs_accuracy,
    plot_reason_codes,
    plot_reliability,
    plot_risk_coverage,
    plot_roc,
    plot_temporal_diagnostics,
    plot_validation_curves,
    save,
)

__all__ = [
    "PALETTE",
    "CATEGORICAL",
    "apply_style",
    "save",
    "plot_loss_decomposition",
    "plot_validation_curves",
    "plot_temporal_diagnostics",
    "plot_optimization_health",
    "plot_quality_vs_accuracy",
    "plot_risk_coverage",
    "plot_reliability",
    "plot_roc",
    "plot_component_attribution",
    "plot_reason_codes",
    "plot_per_patient",
    "plot_force_response",
    "plot_latency",
]
