"""The imaged-sector ROI must reach the predictor from every entry point.

Background
----------
``control.roi`` lives on :class:`PredictorConfig`, not on
``FeatureExtractionConfig``. Every entry point built the latter from the config
mapping and the former by hand, so the ``roi`` section was accepted, validated,
and then silently dropped -- and with ``mode: full`` in force the failure was
invisible: no error, no warning, just three control features quietly measured
over the whole rectangle instead of the insonified sector.

``border_contact_ratio`` was 0.0 on every one of 1292 evaluated frames as a
result, and ``live_monitor.py`` -- the loop that drives the robot -- had the
same gap. A unit test on the feature functions cannot catch this, because each
of them handles ``roi_mask`` correctly when it is given one. What was missing
was the wiring, so that is what this file checks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Entry points that construct a Predictor and must therefore pass the ROI.
ENTRY_POINTS = ("scripts/evaluate.py", "scripts/infer.py", "scripts/live_monitor.py")


def _predictor_config_calls(source: str) -> list[ast.Call]:
    return [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PredictorConfig"
    ]


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_passes_the_roi_to_the_predictor(script: str) -> None:
    path = REPO_ROOT / script
    calls = _predictor_config_calls(path.read_text())
    assert calls, f"{script} no longer constructs a PredictorConfig; update this test."
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert "roi" in keywords, (
            f"{script} builds a PredictorConfig without roi=. The control.roi section "
            "would be parsed and then discarded, and every ROI-dependent feature "
            "(segmentation_confidence, border_contact_ratio, mask_area_ratio) would be "
            "measured over the whole frame with no error raised."
        )


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_builds_the_roi_from_the_config(script: str) -> None:
    """The ROI must come from ``control.roi``, not be hardcoded at the call site."""
    source = (REPO_ROOT / script).read_text()
    assert "RoiConfig.from_dict" in source, (
        f"{script} does not build its ROI from the config mapping."
    )
    assert 'section("control")' in source, (
        f"{script} does not read the control section the roi lives in."
    )
