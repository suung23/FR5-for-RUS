#!/usr/bin/env python3
"""Shared binary-mask metrics for the PFUS1 label audit (Phases 4, 6, 8, 10).

The repository's own rus_perception.metrics.spatial covers Dice/IoU/HD95 but has
no ASSD, and its aggregation is frame-level. This module adds ASSD, relative
area error, explicit empty-mask semantics, and patient-level bootstrap CIs --
frames of one patient are not independent observations, so a frame-level CI
would be far too narrow.

Distances are in PIXELS. PFUS1 publishes no mm/px scale (configs set
evaluation.pixel_spacing: 1.0), and frame size varies by a few pixels even
within one patient, so pixel distances are not strictly comparable across
patients either. Pass `spacing` if a real scale ever becomes available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage


@dataclass
class MaskComparison:
    """One pair of binary masks. ``None`` means undefined, not zero."""

    dice: float
    iou: float
    hd95: Optional[float]
    assd: Optional[float]
    area_a: int
    area_b: int
    relative_area_error: Optional[float]
    status: str  # ok | both_empty | a_empty | b_empty

    def to_dict(self) -> dict:
        return asdict(self)


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels: foreground minus its erosion (8-connectivity)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(2, 2)
    return mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric point-to-surface distances, in pixels scaled by ``spacing``."""
    sa, sb = _surface(a), _surface(b)
    # distance_transform_edt measures distance to the nearest ZERO, so invert.
    dist_to_b = ndimage.distance_transform_edt(~sb) * spacing
    dist_to_a = ndimage.distance_transform_edt(~sa) * spacing
    return dist_to_b[sa], dist_to_a[sb]


def compare_masks(a: np.ndarray, b: np.ndarray, spacing: float = 1.0) -> MaskComparison:
    """Compare mask ``a`` against reference ``b``.

    Empty-mask convention, chosen so nothing crashes and nothing is silently
    scored as a success:

    * both empty  -> Dice = IoU = 1.0, HD95 = ASSD = 0.0, status ``both_empty``
    * one empty   -> Dice = IoU = 0.0, HD95 = ASSD = None, status ``a_empty``/``b_empty``
    * b empty     -> relative area error undefined (division by zero)
    """
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")

    area_a, area_b = int(a.sum()), int(b.sum())
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    rae = abs(area_a - area_b) / area_b if area_b else None

    if not area_a and not area_b:
        return MaskComparison(1.0, 1.0, 0.0, 0.0, 0, 0, None, "both_empty")
    if not area_a or not area_b:
        return MaskComparison(
            0.0, 0.0, None, None, area_a, area_b, rae, "a_empty" if not area_a else "b_empty"
        )

    dice = 2.0 * intersection / (area_a + area_b)
    iou = intersection / union
    d_ab, d_ba = _surface_distances(a, b, spacing)
    both = np.concatenate([d_ab, d_ba])
    hd95 = float(np.percentile(both, 95))
    assd = float(both.mean())
    return MaskComparison(float(dice), float(iou), hd95, assd, area_a, area_b, rae, "ok")


def _clean(values: Sequence[Optional[float]]) -> np.ndarray:
    return np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)


def describe(values: Sequence[Optional[float]]) -> dict:
    """mean ± SD, median [IQR], n, and the count that was undefined."""
    array = _clean(values)
    if array.size == 0:
        return {"n": 0, "n_undefined": len(values), "mean": None, "std": None,
                "median": None, "q1": None, "q3": None, "min": None, "max": None}
    return {
        "n": int(array.size),
        "n_undefined": len(values) - int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def patient_bootstrap_ci(
    values: Sequence[Optional[float]],
    patients: Sequence[str],
    n_boot: int = 10000,
    seed: int = 42,
    statistic: str = "mean",
) -> dict:
    """Cluster bootstrap over PATIENTS, not frames.

    Resamples patients with replacement, then takes every frame of each drawn
    patient. Frames within a patient are near-duplicates, so a frame-level
    bootstrap would understate the interval by roughly sqrt(frames per patient).
    """
    pairs = [(p, v) for p, v in zip(patients, values) if v is not None and np.isfinite(v)]
    if not pairs:
        return {"statistic": statistic, "point": None, "ci_low": None, "ci_high": None,
                "n_patients": 0, "n_frames": 0, "n_boot": 0}

    grouped: dict[str, list[float]] = {}
    for patient, value in pairs:
        grouped.setdefault(patient, []).append(float(value))
    keys = sorted(grouped)
    arrays = [np.array(grouped[k]) for k in keys]

    def stat(samples: list[np.ndarray]) -> float:
        # Patient-level statistic first, then across patients: one patient with
        # 200 frames must not outweigh one with 73.
        per_patient = np.array([s.mean() for s in samples])
        return float(np.median(per_patient) if statistic == "median" else per_patient.mean())

    point = stat(arrays)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    values_boot = np.array([stat([arrays[i] for i in row]) for row in draws])
    return {
        "statistic": f"patient-level {statistic}",
        "point": point,
        "ci_low": float(np.percentile(values_boot, 2.5)),
        "ci_high": float(np.percentile(values_boot, 97.5)),
        "n_patients": len(keys),
        "n_frames": len(pairs),
        "n_boot": n_boot,
    }


def summarise(values: Sequence[Optional[float]], patients: Sequence[str], seed: int = 42) -> dict:
    """describe() plus a patient-level bootstrap CI on the mean."""
    out = describe(values)
    out["patient_bootstrap_ci_95"] = patient_bootstrap_ci(values, patients, seed=seed)
    return out


def format_summary(name: str, summary: dict) -> str:
    if summary["n"] == 0:
        return f"{name:<22} n=0 (all undefined: {summary['n_undefined']})"
    ci = summary["patient_bootstrap_ci_95"]
    text = (
        f"{name:<22} {summary['mean']:.4f} ± {summary['std']:.4f}   "
        f"median {summary['median']:.4f} [{summary['q1']:.4f}, {summary['q3']:.4f}]   "
        f"n={summary['n']}"
    )
    if ci["point"] is not None:
        text += f"   patient-mean {ci['point']:.4f} 95%CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
    if summary["n_undefined"]:
        text += f"   undefined={summary['n_undefined']}"
    return text


def _self_test() -> None:
    """Sanity checks -- run with `python3 segmentation_metrics.py --self-test`."""
    a = np.zeros((64, 64), bool)
    a[20:40, 20:40] = True

    identical = compare_masks(a, a)
    assert identical.dice == 1.0 and identical.iou == 1.0, identical
    assert identical.hd95 == 0.0 and identical.assd == 0.0, identical
    assert identical.relative_area_error == 0.0

    shifted = np.zeros_like(a)
    shifted[20:40, 25:45] = True  # 5 px to the right
    cmp = compare_masks(shifted, a)
    expected_dice = 2 * (20 * 15) / (400 + 400)
    assert abs(cmp.dice - expected_dice) < 1e-9, (cmp.dice, expected_dice)
    assert 0.0 < cmp.assd <= cmp.hd95 <= 5.0 + 1e-6, cmp

    empty = np.zeros_like(a)
    assert compare_masks(empty, empty).status == "both_empty"
    assert compare_masks(empty, a).status == "a_empty"
    assert compare_masks(empty, a).hd95 is None
    assert compare_masks(a, empty).status == "b_empty"
    assert compare_masks(a, empty).relative_area_error is None

    bigger = np.zeros_like(a)
    bigger[20:40, 20:60] = True
    assert abs(compare_masks(bigger, a).relative_area_error - 1.0) < 1e-9

    # Cluster bootstrap must be wider than a naive frame-level interval.
    rng = np.random.default_rng(0)
    patients, values = [], []
    for p in range(8):
        base = rng.normal(0.8, 0.1)
        for _ in range(50):
            patients.append(f"P{p}")
            values.append(base + rng.normal(0, 0.005))
    clustered = patient_bootstrap_ci(values, patients, n_boot=2000)
    naive = patient_bootstrap_ci(values, [f"F{i}" for i in range(len(values))], n_boot=2000)
    assert (clustered["ci_high"] - clustered["ci_low"]) > 3 * (naive["ci_high"] - naive["ci_low"])

    print("segmentation_metrics self-test: OK")
    print(f"  shifted-square dice={cmp.dice:.4f} hd95={cmp.hd95:.2f}px assd={cmp.assd:.2f}px")
    print(f"  patient-cluster CI width={clustered['ci_high'] - clustered['ci_low']:.4f} "
          f"vs frame-level {naive['ci_high'] - naive['ci_low']:.4f}")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
