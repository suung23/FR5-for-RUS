"""Tests for the PFUS preparation script.

The dataset itself is clinical data that is not shipped with this repository, so
a miniature PFUS-shaped directory is synthesised here: the point of the test is
the conversion contract (polygon -> binary mask -> manifest with patient-level
splits), not the pixels of any particular patient.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rus_perception.data.manifest import load_manifest  # noqa: E402
from rus_perception.data.splits import assert_no_patient_leakage  # noqa: E402


def _load_script():
    """Import ``scripts/prepare_pfus.py`` as a module."""
    path = REPO_ROOT / "scripts" / "prepare_pfus.py"
    spec = importlib.util.spec_from_file_location("prepare_pfus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_pfus"] = module
    spec.loader.exec_module(module)
    return module


prepare_pfus = _load_script()


def make_pfus_tree(root: Path, patients: int = 4, frames: int = 3, size=(40, 30)) -> Path:
    """Write a PFUS-shaped ``data/PXXX/frame_YYY.{png,json}`` tree.

    Each frame annotates all eight labels; the bladder is a rectangle whose
    corners are known exactly, so the rasterised area can be asserted.
    """
    data = root / "data"
    for patient in range(patients):
        directory = data / f"P{patient:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        for frame in range(frames):
            Image.new("RGB", size, (17, 17, 17)).save(directory / f"frame_{frame:03d}.png")
            annotations = []
            for label in prepare_pfus.PFUS_LABELS:
                if label == "Bladder":
                    polygon = [[10, 5], [20, 5], [20, 15], [10, 15]]
                else:
                    polygon = [[1, 1], [3, 1], [3, 3], [1, 3]]
                annotations.append({"label": label, "pol": polygon})
            (directory / f"frame_{frame:03d}.json").write_text(json.dumps(annotations))
    return data


def test_frame_number_parses_and_rejects() -> None:
    assert prepare_pfus.frame_number(Path("frame_007.json")) == 7
    assert prepare_pfus.frame_number(Path("a/b/frame_142.png")) == 142
    with pytest.raises(ValueError):
        prepare_pfus.frame_number(Path("frame_abc.json"))


def test_rasterise_polygons_fills_the_annotated_area() -> None:
    mask = np.asarray(
        prepare_pfus.rasterise_polygons([[[2, 2], [6, 2], [6, 6], [2, 6]]], (10, 10))
    )
    assert set(np.unique(mask)) <= {0, 255}
    # PIL draws the polygon inclusive of its boundary: a 4x4 box covers 5x5 px.
    assert int((mask > 127).sum()) == 25
    assert mask[0, 0] == 0 and mask[4, 4] == 255


def test_rasterise_polygons_ignores_degenerate_shapes() -> None:
    mask = np.asarray(prepare_pfus.rasterise_polygons([[[1, 1], [5, 5]]], (8, 8)))
    assert int((mask > 127).sum()) == 0


def test_only_the_requested_label_becomes_foreground(tmp_path: Path) -> None:
    source = make_pfus_tree(tmp_path / "raw", patients=1, frames=1)
    output = tmp_path / "prepared"
    result = prepare_pfus._process_patient(
        str(source / "P000"), str(output / "masks"), str(output), ("Bladder",), False
    )
    assert len(result.frames) == 1 and not result.skipped
    mask = np.asarray(Image.open(output / result.frames[0].mask_path))
    # The bladder rectangle only; the seven decoy polygons in the corner stay background.
    assert int((mask > 127).sum()) == 11 * 11
    assert mask[2, 2] == 0
    assert result.frames[0].foreground_pixels == 11 * 11
    assert result.frames[0].missing_label is False


def test_missing_label_is_reported_and_yields_an_empty_mask(tmp_path: Path) -> None:
    source = make_pfus_tree(tmp_path / "raw", patients=1, frames=1)
    frame = source / "P000" / "frame_000.json"
    annotations = [e for e in json.loads(frame.read_text()) if e["label"] != "Bladder"]
    frame.write_text(json.dumps(annotations))

    output = tmp_path / "prepared"
    result = prepare_pfus._process_patient(
        str(source / "P000"), str(output / "masks"), str(output), ("Bladder",), False
    )
    assert result.frames[0].missing_label is True
    assert result.frames[0].foreground_pixels == 0


def test_end_to_end_writes_a_manifest_with_patient_level_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_pfus_tree(tmp_path / "raw", patients=4, frames=3)
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_pfus.py",
            "--source", str(source),
            "--output-dir", str(output),
            "--labels", "Bladder",
            "--split-ratios", "0.5", "0.25", "0.25",
            "--workers", "1",
            "--qc-samples", "2",
        ],
    )
    assert prepare_pfus.main() == 0

    manifest = load_manifest(output / "manifest.csv")
    assert len(manifest) == 12
    assert manifest.patients == ["P000", "P001", "P002", "P003"]
    assert manifest.splits == ["test", "train", "val"]
    assert_no_patient_leakage(manifest)
    manifest.validate_files(check_masks=True)
    manifest.validate_ordering()

    splits = json.loads((output / "splits.json").read_text())
    assert sorted(sum(splits["assignment"].values(), [])) == manifest.patients
    assert splits["labels"] == ["Bladder"]
    assert len(list((output / "qc").glob("*.png"))) == 2

    # Every frame of a patient lands in that patient's split, and masks are binary.
    per_patient = {r.patient_id: r.split for r in manifest}
    assert all(r.split == per_patient[r.patient_id] for r in manifest)
    mask = np.asarray(Image.open(manifest.resolve(manifest[0].mask_path)))
    assert set(np.unique(mask)) <= {0, 255}


def test_rerun_reuses_existing_masks_unless_overwrite(tmp_path: Path) -> None:
    source = make_pfus_tree(tmp_path / "raw", patients=1, frames=1)
    output = tmp_path / "prepared"
    args = (str(source / "P000"), str(output / "masks"), str(output))
    first = prepare_pfus._process_patient(*args, ("Bladder",), False)
    mask_path = output / first.frames[0].mask_path

    # Corrupt the cached mask: without --overwrite it is trusted and reported as is.
    Image.new("L", Image.open(mask_path).size, 0).save(mask_path)
    reused = prepare_pfus._process_patient(*args, ("Bladder",), False)
    assert reused.frames[0].foreground_pixels == 0

    rebuilt = prepare_pfus._process_patient(*args, ("Bladder",), True)
    assert rebuilt.frames[0].foreground_pixels == 11 * 11
