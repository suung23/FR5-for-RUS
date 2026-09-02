# Control-quality repair — what changed and what it bought

Same data, same model, same checkpoint (`checkpoints/exp_seed43/best.pt`).
Segmentation is untouched: max |Dice delta| is exactly 0.0 on both splits.

Reproduce:

```
python scripts/derive_roi_mask.py                       # measure the sector from train
python scripts/evaluate.py --config configs/exp_seed43_qfix.yaml \
    --checkpoint checkpoints/exp_seed43/best.pt --split test \
    --output-dir runs/exp_seed43_qfix/best_test
python scripts/tune_quality_weights.py                  # candidate sweep, val -> test
python scripts/quality_vs_accuracy.py --run-dir runs/exp_seed43_qfix/best_test \
    --config configs/exp_seed43_qfix.yaml --output-dir experiments/quality_fix/final_test
```

## Two bugs, both silent

**`control.roi` was never wired to the predictor.** The ROI lives on
`PredictorConfig`; `evaluate.py`, `infer.py` and `live_monitor.py` each built
that object by hand and omitted it, so the section was parsed, validated and
discarded. `border_contact_ratio` was 0.0 on all 1292 evaluated frames.
`live_monitor.py` — the loop that drives the robot — had the same gap.
Regression test: `tests/test_roi_wiring.py`.

**The temporal reference geometry ignored the ROI.** `features.py` passed
`roi_mask` to the current frame's geometry but not the previous frame's, so
`relative_area_change` carried a constant `frame area / ROI area` offset — 0.404
for this sector — between two identical masks. Latent while `mode: full` made
the two denominators equal; it surfaced the moment the ROI was switched on. It
flattened `area_stability` to 0.20 and fired `excessive_area_change` on frames
that had not moved. Regression test:
`test_relative_area_change_is_zero_for_an_unchanged_mask_under_an_roi`.

## The four score problems

| # | Problem | Fix | Status |
| --- | --- | --- | --- |
| 1 | 3 of 8 terms carried no information (46% of the weight) | ROI wiring; `segmentation_confidence` and `border_penalty` weights to 0 | fixed |
| 2 | Q compressed into 0.921 ± 0.056; no threshold below 0.75 rejected anything | geometric aggregation | fixed |
| 3 | Q was largely a bladder-size detector | size terms recalibrated and made one-sided | mechanism removed; benefit not measurable on 11 patients |
| 4 | High-confidence errors (P097: Q rank 2, Dice rank 8) | none that works | **not fixed** |

## Result

| config | split | AUROC | within-band | dynamic range | AUROC(centroid err) | Q | valid % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shipped | val | 0.944 | 0.808 | 0.128 | 0.683 | 0.940 ± 0.044 | 100.0 |
| shipped | test | 0.855 | 0.647 | 0.173 | 0.677 | 0.921 ± 0.056 | 98.2 |
| qfix | val | 0.940 | 0.778 | 0.321 | 0.689 | 0.886 ± 0.112 | 100.0 |
| qfix | test | **0.891** | **0.706** | **0.460** | **0.717** | 0.852 ± 0.153 | 98.7 |

Dynamic range is the only row that needs no statistics — it is a property of the
score definition, and it roughly triples. Everything else moves within the
bootstrap CI (see below), improving on test and drifting slightly on val.

## What this cohort cannot decide

`tune_quality_weights.py` bootstraps the within-band AUROC by resampling
patients. Every candidate's 95% CI spans roughly **[0.43, 0.97]**: with 11
patients per split, val cannot rank the candidates at all. The pre-declared
selection rule picked `drop_confidence_border`, which is *not* the candidate
shipped here.

The shipped configuration was therefore chosen on grounds that do not depend on
a marginal AUROC:

* `segmentation_confidence` ranks the **wrong way** on both cohorts (Spearman
  with Dice: −0.545 val, −0.344 test) and is saturated at 0.996 ± 0.001. A
  consistently inverted sign is not a tuning question.
* `border_penalty` duplicates a hard limit the validity gate already enforces
  (`max_border_contact_ratio`), and its sign disagrees between cohorts.
* `target_area_ratio` 0.15 referred to a denominator that no longer exists once
  the ROI is applied, and `area_tolerance` 0.10 was wider than the entire
  distribution — so the sub-score was pinned at 1.0 everywhere. Recalibrated to
  the val ground truth: 0.0683 ± 0.0270.
* `penalise_overfill: false` — a two-sided plateau punishes the hydro-extended
  bladder the procedure is built around. With it on, P041 (best Dice in the
  cohort, 0.931) fell from Q rank 1 to rank 11.
* Geometric aggregation — deterministic, as above.

## Problem 4 is still open

P097 keeps Q rank 3 against Dice rank 8. Its centroid error is 11.5 px and 96%
of that is a *standing bias*, which is invisible by construction to every term
in Q: `centroid_stability` measures the jump from the previous **prediction**,
so a mask that is wrong in the same place every frame scores 1.0. No ground-free
frame-level feature in the current set detects it — `mean_boundary_entropy`, the
one unused candidate, reached AUROC 0.772 on val and 0.522 on test, i.e. it does
not transfer.

Patient-level ranking also did not improve: Spearman(Q, Dice) across the 11 test
patients moved 0.764 → 0.736, and the summed rank error 16 → 20. The gains are
frame-level.

Closing this needs a measurement the pipeline does not currently make. See
`scripts/centroid_accuracy.py`, which adds it to the evaluation protocol.
