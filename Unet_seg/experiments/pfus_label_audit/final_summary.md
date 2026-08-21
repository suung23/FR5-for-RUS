# PFUS1 Bladder Segmentation — Label Quality Audit

> **Partly superseded by [FINAL_PERFORMANCE.md](FINAL_PERFORMANCE.md).** Direct
> measurement over all 110 patients contradicts two claims below: P021's label
> is **not** frozen (it tracks an image that is itself nearly static), and the
> label-contrast figures in §2 were confounded by the black letterbox being
> counted as "background". The temporal metric has since been given a single
> definition. Section 3's performance numbers stand; the ceiling analysis
> (§2, §4) was abandoned in favour of trusting the PFUS1 label.

**Question.** The model performs well but below expectation. Is that the model,
the PFUS1 annotation, train–validation leakage, or the ultrasound domain?

**Status.** Phases 0–2, 5, 6, 8, 9, 10 complete. Phases 3–4, 7 are built and
blocked on human annotation. Phases 11–13 are built and blocked on an external
dataset that is not on this machine.

**Short answer so far.** Leakage is ruled out. The domain hypothesis cannot be
tested without external data. Between the remaining two, the evidence collected
here points at **label definition** as a major contributor — but the decisive
measurement (`Dice(PFUS1, manual lumen)`) requires the expert annotation that is
now packaged and waiting.

---

## 1. Dataset split integrity

**No leakage. Verified four independent ways.**

```text
Train-Val overlap:  0    Train-Test overlap: 0    Val-Test overlap: 0
Leakage detected: NO
```

| split | patients | frames | frames/patient median [IQR] |
|---|---|---|---|
| train | 78 | 10,359 | 124.5 [98.0, 162.5] |
| val | 16 | 2,476 | 146.5 [120.0, 189.2] |
| test | 16 | 2,017 | 121.5 [100.0, 144.8] |

1. `splits.json` assignment is disjoint (110 unique patients, 0 duplicates).
2. All 14,852 rows of the manifest the training actually read: 0 patients and
   **0 sequences** span two splits.
3. `assert_no_patient_leakage()` is called on every run — completing 40 epochs
   is itself proof the check passed.
4. `splits.json` ↔ manifest: 0 mismatches.

**Each patient has exactly one sequence** (`{1: 110}`), so patient-level
splitting *is* clip-level splitting. Frames of one video cannot straddle a
split boundary. The existing split was frozen, not re-rolled — re-splitting
would invalidate every checkpoint and run in the repository.

> **Cause 3 (leakage) is eliminated.**

---

## 2. PFUS1 annotation agreement — BLOCKED

`Dice(PFUS1, manual lumen)` is the ceiling on any model trained against this
label, and it cannot be estimated from PFUS1 alone: the dataset contains no
second opinion. The audit package is ready.

**What was found without annotation:**

- **Every label is a coarse polygon.** Median 13 vertices (range 7–29) for
  `Bladder`; all eight structures are annotated on all 14,852 frames. This caps
  achievable Dice regardless of model quality, and puts the reported HD95
  (~10 px) in the same order as the annotation's own approximation error.
- **Labels are propagated, not drawn per frame.** Vertex count is *constant*
  within a patient across every frame (P021 = 11 on all 200; P000 = 29 on all
  147) — the signature of a template interpolated across keyframes.
- **P021's label is effectively frozen.** Centroid moves 0.03 px/frame (median)
  and 7.7 px total across 200 frames, while its images change as much as other
  patients'. Frame crop alone varies by ±5 px between consecutive frames, which
  exceeds the label's entire motion. The label is not tracking the image.
- **The label's appearance is inconsistent across patients.** Contrast
  (background − label interior) spans −64.7 … +45.9 over the 78 train patients:
  in many patients the "bladder" region is *brighter* than its surroundings,
  which a urine-filled lumen cannot be. Same class, opposite appearance.

**Interpretation.** `scripts/prepare_pfus.py` states in its docstring that
`--labels Bladder` "reproduces exactly the binary bladder-lumen task". Nothing
in the dataset or the code supports that. PFUS1 annotates a pelvic-floor
*organ*; the control loop needs the *lumen*. Whether these coincide is Q3, and
it is exactly what Phase 4 measures.

---

## 3. Model vs PFUS1 GT

Full test split, 2,017 frames / 16 patients, from the existing runs:

| | Standard | Slim |
|---|---|---|
| Dice mean | 0.7313 ± 0.206 | 0.6977 ± 0.269 |
| Dice median | 0.7876 | **0.8046** |
| IoU mean | 0.6099 | 0.5882 |
| HD95 median (px @256) | 10.48 | 10.16 |
| Missed-bladder rate | **0.05%** | 6.64% |
| Longest invalid run | 22 frames | 102 frames (≈3.4 s @30 fps) |

Audit subset (80 frames, patient-level bootstrap CI):

| model @ 256 | Dice | 95% CI |
|---|---|---|
| standard | 0.7533 | [0.658, 0.824] |
| slim | 0.7334 | [0.613, 0.825] |

**The interval is the story.** A patient-level bootstrap gives ±0.08 on a
16-patient test set; the Standard/Slim gap (0.02) sits well inside it. With 16
test patients, this dataset cannot resolve differences of that size.

**Two patients carry the deficit:**

| | all 16 patients | excluding P021, P043 |
|---|---|---|
| Standard | 0.7525 | **0.8074** |
| Slim | 0.7221 | 0.7985 |

---

## 4. Model vs manual lumen GT — BLOCKED

Requires Phase 3 annotation. The scoring path is implemented and self-tested
(`metrics/score_model_vs_pfus.py` already emits `reference=manual` rows the
moment `manual_masks/` is populated).

---

## 5. Boundary error analysis

ASSD was **not implemented** in the repository and was added here
(`metrics/segmentation_metrics.py`, self-tested against analytic cases).

| | Standard | Slim |
|---|---|---|
| ASSD @256 (px) | 4.63 [3.30, 6.72] | 4.99 [3.24, 7.66] |
| ASSD native (px) | 14.80 [10.36, 21.58] | 16.59 [10.28, 25.69] |
| HD95 @256 (px) | 12.32 [8.62, 17.85] | 13.39 [8.86, 19.81] |
| Relative area error | 0.249 [0.166, 0.341] | 0.284 [0.183, 0.406] |

Boundary quality is statistically indistinguishable between the two
architectures. Native-resolution distances are ~3.2× the 256 ones, matching the
resolution ratio — the two are consistent, not contradictory.

**A ~25% area error with Dice ~0.75 means the errors are boundary-band errors,
not misplacements.** For a probe servo consuming centroid and area, systematic
area bias matters more than the Dice number suggests.

---

## 6. Failure case analysis

Patient-level Dice, worst first: **P021 = 0.276, P043 = 0.461**, P000 = 0.628,
P001 = 0.693. The 15 worst audit frames are **13/15 from P021, P043 and P000**.

Stratified over the full test split (Standard, patient-level means):

| stratum | Dice |
|---|---|
| bladder small / medium / **large** | 0.690 / 0.676 / **0.876** |
| contrast low / medium / **high** | 0.656 / 0.735 / **0.856** |
| boundary weak / **moderate** / clear | 0.736 / **0.811** / 0.792 |
| shadow absent / present | 0.754 / 0.748 |
| contour regular / irregular | 0.759 / 0.720 |

**Where the error lives:** small, low-contrast bladders. Large bladders reach
0.876 and high-contrast frames 0.856 — the model is not architecturally
incapable. **Acoustic shadow is not a driver** (0.754 vs 0.748), which
contradicts the intuitive expectation and means shadow-specific fixes would be
misdirected effort.

**P021 does not fit this pattern.** Its area (p50), contrast (p85) and aspect
ratio (p82) all sit mid-distribution relative to train, and its labelled region
*is* genuinely dark (20.7 vs background 46.6). An in-distribution, correctly
dark case scoring 0.276 is not explained by any image statistic measured here —
which is why the frozen-label finding in §2 is the leading hypothesis, and why
it needs the Phase 4 measurement to confirm.

---

## 7. Temporal consistency

**The published "Warped temporal Dice" was never motion-compensated.**
`configs` set `flow.backend: precomputed`, the manifest's flow columns are
empty, so `scripts/evaluate.py` takes its `flow_backward is None` branch and
sets `warped_previous = previous_state.binary_mask`. Recomputing with real
Farneback flow over all 2,001 test transitions confirms it: Method 1 reproduces
the published 0.9646 exactly.

| | Standard | Slim |
|---|---|---|
| Method 1 — raw consecutive Dice | 0.9646 [0.943, 0.983] | 0.9536 [0.915, 0.981] |
| Method 2 — Farneback motion-compensated | 0.9713 [0.951, 0.987] | 0.9590 [0.923, 0.983] |
| Method 2, both-empty transitions excluded | 0.9713 | **0.9528** |
| Mask dropout rate | 0.0003 | 0.0450 |

**A metric pathology worth fixing.** Both-empty transitions score TC = 1.0,
i.e. a model is rewarded for *consistently producing nothing*. Slim contributes
125 such transitions — **119 of them from P021 alone, where it drops the mask on
61% of frames**. Slim's apparent P021 stability of 0.946 is that artifact. With
them removed, Slim's worst patient is P043 at 0.699.

For the FR5 control loop this is the decisive axis: Standard's dropout is 150×
lower, and dropout is signal loss, not inaccuracy.

---

## 8. Baseline comparison

| Model | Params | GFLOPs@256 | Dice | IoU | HD95 px | ASSD px | infer ms | e2e p95 ms |
|---|---|---|---|---|---|---|---|---|
| standard_unet | 8,635,809 | 32.66 | 0.7313 | 0.6099 | 10.48 | 4.63 | 1.131 | 3.198 |
| slim_unet | 4,705,377 | 21.79 | 0.6977 | 0.5882 | 10.16 | 4.99 | 0.822 | 2.876 |
| unet_plus_plus | — | — | — | — | — | — | — | not trained |
| fpn_resnet50 | — | — | — | — | — | — | — | not trained |

The two trained models already form a controlled comparison — identical split,
preprocessing, augmentation, loss and 40-epoch schedule. **U-Net++ and
FPN+ResNet50 were not trained**: the only torch in this environment is
CPU-only, the original runs took ~15 min each on an RTX 4090, and CPU training
would neither finish in reasonable time nor yield comparable inference
timings. Filling those rows under different conditions would be worse than
leaving them empty.

**Adding architectures now would be premature anyway.** With a ±0.08 CI on this
test set and an unmeasured label ceiling, a new architecture's Dice could not be
interpreted.

---

## 9. Interpretation

| Candidate cause | Verdict |
|---|---|
| **3. Train–validation leakage** | **Eliminated.** Patient- and sequence-level disjoint, four ways. |
| **2. Annotation quality / target definition** | **Strongly implicated, not yet quantified.** Coarse polygons cap Dice; labels are template-propagated; P021's label is frozen against a moving image; label appearance is contradictory across patients (contrast −65…+46). |
| **1. Model limitation** | **Partially exonerated.** 0.876 on large bladders and 0.856 on high-contrast frames; 0.807 patient-mean excluding two suspect patients. The residual small/low-contrast weakness is real but secondary. |
| **4. Ultrasound domain** | **Untestable here.** Single-domain data. Needs Phase 11–12. |

**Answers to the framing questions:**

- **Q1 (real model limitation?)** Partly. Small, low-contrast bladders are a genuine weakness. The headline gap is not primarily this.
- **Q2 (annotation uncertainty?)** Not yet a number — that is Phase 4. Structurally: 13-vertex polygons, propagated across frames, at least one patient with a static label over a moving sequence.
- **Q3 (same target?)** **Evidence says no.** PFUS1 annotates an organ polygon; many patients' labelled region is brighter than its surroundings, which a urine-filled lumen cannot be.
- **Q4 (model beats PFUS GT?)** Instrumented, awaiting manual masks.
- **Q5 (condition-specific failure?)** **Yes** — small and low-contrast bladders. **Not** shadow.
- **Q6 (patient-independent test holds?)** **Yes**, and it always was patient-independent.
- **Q7 (temporally stable?)** Standard yes (0.971 compensated, 0.03% dropout). Slim no (4.5% dropout, 61% on one patient).
- **Q8 (zero-shot external?)** Unknown. Pipeline ready, no dataset present.

---

## 10. Recommended next experiment

**Do not change the architecture yet.** Priorities 1–7 of the brief are not
complete: the label ceiling is unmeasured.

1. **Annotate the 80-frame audit subset** (`annotation_audit/images_blinded/`,
   protocol in `ANNOTATION_PROTOCOL.md`). This is the single blocking step; it
   unblocks Phases 4 and 7 and answers Q2, Q3 and Q4 at once. Everything
   downstream is already written and tested.
2. **Inspect P021 and P043 source data directly.** If their labels are frozen
   or misplaced, the honest test-set number is 0.807, not 0.752 — and fixing
   data is cheaper than fixing a model.
3. **Enable the temporal loss.** Both runs used `temporal_weight = 0.00` for all
   40 epochs, so the term never contributed. It is *self-supervised* (warped
   previous prediction vs current, no labels — `trainer.py:515-529`), so it is
   valid regardless of the label question, and Slim's consecutive dropout is
   precisely the failure mode it exists to suppress.
4. **Enable early stopping.** Standard peaked at epoch 7 of 40 and overfitted
   for 33 more. `patience=10` cuts training time to a third for the same result.
5. **Fix the temporal metric.** Both-empty transitions scoring 1.0 credits
   silence as stability; exclude them or score them 0.
6. **Fix or remove the "warped" label** in the evaluation output, or populate
   the flow columns via `scripts/precompute_flow.py`. The published number is
   uncompensated.
7. **Then**, with a known ceiling, weight small/low-contrast frames — and only
   then consider new architectures, evaluated with patient-level CIs on more
   than 16 test patients.

---

## Files

```text
config.yaml                              provenance of this audit
patient_split_seed42.json                frozen split + manifest sha256
patient_split_audit.csv                  110 patients x (split, frames)
annotation_agreement.csv                 BLOCKED -- awaits manual annotation
model_label_comparison.csv               320 rows (2 models x 80 x 2 resolutions)
temporal_consistency.csv                 32 rows (2 models x 16 patients)
temporal_consistency_transitions.csv     4,002 per-transition rows
final_summary.md                         this file

ANNOTATION_PROTOCOL.md                   lumen definition, edge cases, procedure
annotation_audit/                        80-frame package incl. blinded images
dataset_audit/                           leakage audit, per-frame features (2,017)
metrics/                                 metrics module (+ASSD), agreement, ceiling,
                                         stratification scripts
predictions/                             saved masks (npz) for both models
visualizations/                          80 overlays + worst_model_cases/
temporal_analysis/                       flow-warped consistency
baseline_comparison/                     architecture table
external_data/                           contract + zero-shot script (no data yet)
```

Reproduce: `dataset_audit/audit_splits.py` → `annotation_audit/build_audit_subset.py`
→ `predictions/run_audit_inference.py` → `metrics/score_model_vs_pfus.py` →
`metrics/error_stratification.py` → `temporal_analysis/*` →
`baseline_comparison/compare_baselines.py`. Seed 42 throughout.
