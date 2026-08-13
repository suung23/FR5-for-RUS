"""Dataset tests: manifests, patient-level splits, ordering and paired augmentation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.augment import AugmentationConfig, PairedAugmentation, transform_flow_field
from src.data.image_dataset import UltrasoundFrameDataset
from src.data.manifest import Manifest, ManifestRecord, load_manifest, write_manifest
from src.data.splits import (
    PatientLeakageError,
    SplitAssignment,
    apply_split,
    assert_no_patient_leakage,
    split_by_patient,
)
from src.data.video_dataset import SequentialUltrasoundDataset, resize_flow
from src.flow.base import FlowPair
from src.flow.precomputed import save_flow_pair


def record(patient: str, sequence: str, index: int, split=None, labeled=True) -> ManifestRecord:
    return ManifestRecord(
        patient_id=patient,
        sequence_id=sequence,
        frame_index=index,
        image_path=f"images/{patient}_{sequence}_{index}.png",
        mask_path=f"masks/{patient}_{sequence}_{index}.png" if labeled else None,
        timestamp=index * 0.05,
        split=split,
    )


# -- manifest --------------------------------------------------------------
def test_manifest_orders_frames_chronologically() -> None:
    manifest = Manifest([record("P1", "S1", 3), record("P1", "S1", 1), record("P1", "S1", 2)])
    assert [r.frame_index for r in manifest] == [1, 2, 3]
    sequences = manifest.sequences()
    assert [r.frame_index for r in sequences[("P1", "S1")]] == [1, 2, 3]


def test_manifest_rejects_duplicate_frames() -> None:
    with pytest.raises(ValueError, match="Duplicate manifest entry"):
        Manifest([record("P1", "S1", 1), record("P1", "S1", 1)])


def test_manifest_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError, match="Negative frame_index"):
        Manifest([record("P1", "S1", -1)])


def test_manifest_detects_inconsistent_timestamps() -> None:
    first = record("P1", "S1", 0)
    second = ManifestRecord("P1", "S1", 1, "images/b.png", timestamp=-5.0)
    with pytest.raises(ValueError, match="timestamps that decrease"):
        Manifest([first, second]).validate_ordering()


def test_manifest_round_trip(tmp_path) -> None:
    records = [record("P1", "S1", i, split="train") for i in range(3)]
    path = write_manifest(tmp_path / "m.csv", records)
    loaded = load_manifest(path)
    assert len(loaded) == 3
    assert loaded[0].patient_id == "P1"
    assert loaded[0].timestamp == pytest.approx(0.0)
    assert loaded[1].split == "train"
    assert loaded[0].frame_id == "P1/S1/000000"


def test_manifest_requires_its_mandatory_columns(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("patient_id,frame_index\nP1,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        load_manifest(path)


def test_manifest_rejects_malformed_rows(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "patient_id,sequence_id,frame_index,image_path\nP1,S1,not_a_number,a.png\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frame_index must be an integer"):
        load_manifest(path)

    empty_image = tmp_path / "bad2.csv"
    empty_image.write_text(
        "patient_id,sequence_id,frame_index,image_path\nP1,S1,0,\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="image_path must not be empty"):
        load_manifest(empty_image)


def test_manifest_detects_missing_files(tmp_path) -> None:
    path = write_manifest(tmp_path / "m.csv", [record("P1", "S1", 0)])
    with pytest.raises(FileNotFoundError, match="missing"):
        load_manifest(path).validate_files()


def test_manifest_tracks_labeled_and_unlabeled_frames() -> None:
    manifest = Manifest(
        [record("P1", "S1", 0), record("P1", "S1", 1, labeled=False), record("P2", "S1", 0)]
    )
    statistics = manifest.statistics()
    assert statistics.num_frames == 3
    assert statistics.num_labeled == 2
    assert statistics.num_unlabeled == 1
    assert statistics.num_patients == 2
    assert statistics.num_sequences == 2
    assert len(manifest.filter(labeled_only=True)) == 2


def test_real_manifest_statistics_report(synthetic_dataset) -> None:
    _, manifest = synthetic_dataset
    statistics = manifest.statistics(compute_class_occupancy=True)
    assert statistics.num_patients == 4
    assert statistics.num_frames == 24
    assert set(statistics.frames_per_split) == {"train", "val", "test"}
    assert 0.0 < statistics.class_occupancy["foreground_ratio_mean"] < 1.0
    assert "patients" in statistics.to_text()


# -- patient-level splitting ----------------------------------------------
def test_split_is_disjoint_at_the_patient_level() -> None:
    patients = [f"P{i:02d}" for i in range(10)]
    assignment = split_by_patient(patients, (0.6, 0.2, 0.2), seed=1)
    everyone = assignment.train + assignment.val + assignment.test
    assert sorted(everyone) == sorted(patients)
    assert len(set(everyone)) == len(patients), "a patient appeared in two splits"


def test_split_is_deterministic_for_a_fixed_seed() -> None:
    patients = [f"P{i:02d}" for i in range(12)]
    first = split_by_patient(patients, (0.5, 0.25, 0.25), seed=3).as_dict()
    second = split_by_patient(patients, (0.5, 0.25, 0.25), seed=3).as_dict()
    assert first == second
    other = split_by_patient(patients, (0.5, 0.25, 0.25), seed=4).as_dict()
    assert first != other


def test_split_rejects_impossible_requests() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        split_by_patient(["P1", "P2"], (0.4, 0.3, 0.3))
    with pytest.raises(ValueError, match="empty patient list"):
        split_by_patient([])
    with pytest.raises(ValueError, match="duplicate patient"):
        split_by_patient(["P1", "P1", "P2", "P3"])
    with pytest.raises(ValueError, match="non-negative"):
        split_by_patient(["P1", "P2", "P3"], (-1.0, 1.0, 1.0))


def test_leakage_detection_flags_a_patient_in_two_splits() -> None:
    leaking = [record("P1", "S1", 0, split="train"), record("P1", "S2", 0, split="test")]
    with pytest.raises(PatientLeakageError, match="multiple splits"):
        assert_no_patient_leakage(leaking)


def test_leakage_detection_passes_for_a_clean_split() -> None:
    clean = [record("P1", "S1", 0, split="train"), record("P2", "S1", 0, split="test")]
    assert_no_patient_leakage(clean)  # must not raise


def test_apply_split_detects_leaking_assignments() -> None:
    manifest = Manifest([record("P1", "S1", 0), record("P2", "S1", 0)])
    with pytest.raises(PatientLeakageError, match="listed in both"):
        apply_split(manifest, SplitAssignment(train=["P1", "P2"], val=["P2"], test=[]))


def test_apply_split_rejects_uncovered_patients() -> None:
    manifest = Manifest([record("P1", "S1", 0), record("P2", "S1", 0)])
    with pytest.raises(ValueError, match="no split assignment"):
        apply_split(manifest, SplitAssignment(train=["P1"], val=[], test=[]))


def test_generated_dataset_has_no_leakage(synthetic_dataset) -> None:
    _, manifest = synthetic_dataset
    assert_no_patient_leakage(manifest)
    by_split = {
        split: {r.patient_id for r in manifest.filter(split=split)}
        for split in ("train", "val", "test")
    }
    assert not (by_split["train"] & by_split["val"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["val"] & by_split["test"])


# -- paired augmentation ---------------------------------------------------
def test_paired_augmentation_applies_identical_geometry_to_both_frames() -> None:
    """Independent geometry would inject artificial motion between the frames."""
    rng = np.random.default_rng(0)
    image = rng.random((32, 32)).astype(np.float32)
    mask = np.zeros((32, 32), np.float32)
    mask[8:20, 8:20] = 1.0

    augment = PairedAugmentation(
        AugmentationConfig(
            rotation_degrees=25.0,
            translate_ratio=0.1,
            horizontal_flip=1.0,
            brightness=0.0,
            contrast=0.0,
            gamma_range=(1.0, 1.0),
            gaussian_noise_std=0.0,
        ),
        seed=1,
    )
    output = augment(image, image.copy(), mask, mask.copy(), sample_seed=5)

    assert np.allclose(output.image_previous, output.image_current)
    assert np.array_equal(output.mask_previous, output.mask_current)


def test_paired_augmentation_is_deterministic_for_a_given_sample_seed() -> None:
    image = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    augment = PairedAugmentation(AugmentationConfig(), seed=7)
    first = augment(image, image, sample_seed=11)
    second = augment(image, image, sample_seed=11)
    assert np.allclose(first.image_current, second.image_current)

    different = augment(image, image, sample_seed=12)
    assert not np.allclose(first.image_current, different.image_current)


def test_disabled_augmentation_passes_data_through_untouched() -> None:
    image = np.random.default_rng(2).random((16, 16)).astype(np.float32)
    mask = (image > 0.5).astype(np.float32)
    output = PairedAugmentation(AugmentationConfig(enabled=False))(image, image, mask, mask)
    assert np.array_equal(output.image_current, image)
    assert np.array_equal(output.mask_current, mask)
    assert float(output.geometry_valid.min()) == 1.0


def test_augmented_masks_stay_binary() -> None:
    image = np.random.default_rng(3).random((32, 32)).astype(np.float32)
    mask = np.zeros((32, 32), np.float32)
    mask[10:22, 10:22] = 1.0
    output = PairedAugmentation(AugmentationConfig(rotation_degrees=30.0), seed=2)(
        image, image, mask, mask, sample_seed=1
    )
    assert set(np.unique(output.mask_current)).issubset({0.0, 1.0})


def test_augmentation_transforms_the_flow_field_with_the_geometry() -> None:
    """A flip must negate the x-component of a displacement field."""
    size = 24
    flow = np.zeros((size, size, 2), np.float32)
    flow[..., 0] = 3.0

    flip = np.array([[-1.0, 0.0, size - 1.0], [0.0, 1.0, 0.0]], np.float32)
    linear = np.array([[-1.0, 0.0], [0.0, 1.0]], np.float32)
    transformed = transform_flow_field(flow, flip, linear, (size, size))
    assert transformed[size // 2, size // 2, 0] == pytest.approx(-3.0, abs=1e-4)


def test_augmentation_config_validation() -> None:
    with pytest.raises(ValueError, match="probability"):
        AugmentationConfig(horizontal_flip=2.0)
    with pytest.raises(ValueError, match="scale_range"):
        AugmentationConfig(scale_range=(1.5, 0.5))
    with pytest.raises(ValueError, match="Unknown augmentation key"):
        AugmentationConfig.from_dict({"nope": 1})


def test_paired_augmentation_rejects_mismatched_frames() -> None:
    with pytest.raises(ValueError, match="identical shape"):
        PairedAugmentation(AugmentationConfig())(np.zeros((8, 8)), np.zeros((16, 16)))


# -- flow resizing ---------------------------------------------------------
def test_resize_flow_rescales_the_displacement_vectors() -> None:
    """Four pixels at 64 wide is two pixels at 32 wide."""
    flow = np.zeros((64, 64, 2), np.float32)
    flow[..., 0] = 4.0
    flow[..., 1] = 8.0
    resized = resize_flow(flow, (32, 32))
    assert resized.shape == (32, 32, 2)
    assert resized[16, 16, 0] == pytest.approx(2.0, abs=1e-4)
    assert resized[16, 16, 1] == pytest.approx(4.0, abs=1e-4)


# -- datasets --------------------------------------------------------------
def test_frame_dataset_yields_correctly_shaped_tensors(synthetic_dataset) -> None:
    _, manifest = synthetic_dataset
    dataset = UltrasoundFrameDataset(manifest.filter(split="train"), image_size=(64, 64))
    item = dataset[0]
    assert item["image"].shape == (1, 64, 64)
    assert item["mask"].shape == (1, 64, 64)
    assert set(torch.unique(item["mask"]).tolist()).issubset({0.0, 1.0})
    assert float(item["labeled"]) == 1.0
    assert "/" in item["frame_id"]


def test_frame_dataset_rejects_an_empty_manifest(synthetic_dataset) -> None:
    _, manifest = synthetic_dataset
    empty = manifest.filter(patients=["does_not_exist"])
    with pytest.raises(ValueError, match="empty manifest"):
        UltrasoundFrameDataset(empty)


def test_sequential_dataset_pairs_adjacent_frames(synthetic_dataset) -> None:
    _, manifest = synthetic_dataset
    dataset = SequentialUltrasoundDataset(manifest.filter(split="train"), image_size=(64, 64))
    item = dataset[0]

    for key in ("image_previous", "image_current", "mask_previous", "mask_current"):
        assert item[key].shape == (1, 64, 64)
    assert item["flow_backward"].shape == (2, 64, 64)
    assert item["frame_index"] == item["previous_frame_index"] + 1
    assert item["patient_id"] == item["previous_frame_id"].split("/")[0]


def test_sequential_dataset_never_pairs_across_patients(synthetic_dataset) -> None:
    _, manifest = synthetic_dataset
    dataset = SequentialUltrasoundDataset(manifest, image_size=(64, 64))
    for previous, current in dataset.pairs:
        assert previous.patient_id == current.patient_id
        assert previous.sequence_id == current.sequence_id
        assert previous.frame_index < current.frame_index


@pytest.mark.parametrize("interval", [1, 2, 3])
def test_sequential_dataset_honours_the_temporal_interval(synthetic_dataset, interval) -> None:
    _, manifest = synthetic_dataset
    dataset = SequentialUltrasoundDataset(manifest, image_size=(64, 64), interval=interval)
    for previous, current in dataset.pairs:
        assert current.frame_index - previous.frame_index == interval


def test_sequential_dataset_rejects_an_invalid_interval(synthetic_dataset) -> None:
    _, manifest = synthetic_dataset
    with pytest.raises(ValueError, match="interval must be >= 1"):
        SequentialUltrasoundDataset(manifest, interval=0)
    with pytest.raises(ValueError, match="No temporally adjacent pairs"):
        SequentialUltrasoundDataset(manifest, interval=100)


def test_sequential_dataset_supports_unlabeled_frames(partially_labeled_dataset) -> None:
    """Unlabeled frames must still form pairs, flagged so the spatial loss skips them."""
    _, manifest = partially_labeled_dataset
    dataset = SequentialUltrasoundDataset(
        manifest, image_size=(64, 64), require_labeled_current=False
    )
    flags = {float(dataset[i]["labeled_current"]) for i in range(len(dataset))}
    assert flags == {0.0, 1.0}, "expected both labeled and unlabeled current frames"

    labeled_only = SequentialUltrasoundDataset(
        manifest, image_size=(64, 64), require_labeled_current=True
    )
    assert len(labeled_only) < len(dataset)
    assert all(float(labeled_only[i]["labeled_current"]) == 1.0 for i in range(len(labeled_only)))


def test_sequential_dataset_reports_missing_flow_rather_than_faking_it(synthetic_dataset) -> None:
    _, manifest = synthetic_dataset
    dataset = SequentialUltrasoundDataset(manifest.filter(split="train"), image_size=(64, 64))
    item = dataset[0]
    assert float(item["flow_valid"]) == 0.0
    assert item["flow_source"] == "missing"
    assert float(item["flow_backward"].abs().max()) == 0.0

    strict = SequentialUltrasoundDataset(
        manifest.filter(split="train"), image_size=(64, 64), require_flow=True
    )
    with pytest.raises(FileNotFoundError, match="No optical flow available"):
        strict[0]


def test_sequential_dataset_loads_precomputed_flow(synthetic_dataset, tmp_path) -> None:
    manifest_path, manifest = synthetic_dataset
    train = manifest.filter(split="train")
    sequences = sorted(train.sequences().items())
    (_, records) = sequences[0]

    flow_dir = tmp_path / "flow"
    forward = np.full((64, 64, 2), 1.5, np.float32)
    updated = []
    for index, item in enumerate(train):
        flow_path = None
        if index > 0:
            flow_path = flow_dir / f"{index}.npz"
            save_flow_pair(flow_path, FlowPair(forward.copy(), -forward))
        updated.append(
            ManifestRecord(
                patient_id=item.patient_id,
                sequence_id=item.sequence_id,
                frame_index=item.frame_index,
                image_path=item.image_path,
                mask_path=item.mask_path,
                timestamp=item.timestamp,
                split=item.split,
                flow_backward_path=str(flow_path) if flow_path else None,
                flow_forward_path=str(flow_path) if flow_path else None,
            )
        )
    with_flow = Manifest(updated, train.root)
    dataset = SequentialUltrasoundDataset(with_flow, image_size=(64, 64))
    item = dataset[0]
    assert float(item["flow_valid"]) == 1.0
    assert item["flow_source"] == "precomputed"
    assert float(item["flow_backward"][0].abs().mean()) == pytest.approx(1.5, abs=0.1)


def test_augmentation_is_opt_in_for_datasets(synthetic_dataset) -> None:
    """Constructing a dataset without an augmentation config must not augment.

    Silently augmenting an evaluation set would quietly corrupt every metric
    computed from it.
    """
    _, manifest = synthetic_dataset
    plain = UltrasoundFrameDataset(manifest.filter(split="val"), image_size=(64, 64))
    assert plain.augmentation.config.enabled is False
    assert torch.allclose(plain[0]["image"], plain[0]["image"])

    augmented = UltrasoundFrameDataset(
        manifest.filter(split="val"),
        image_size=(64, 64),
        augmentation=AugmentationConfig(rotation_degrees=30.0, horizontal_flip=1.0),
    )
    assert augmented.augmentation.config.enabled is True
    assert not torch.allclose(plain[0]["image"], augmented[0]["image"])
