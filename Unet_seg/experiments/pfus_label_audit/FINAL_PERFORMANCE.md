# PFUS1 test-split performance — unbiased report

Ground truth is the PFUS1 label, taken at face value. The only data judgement is
a patient exclusion list, and it is derived from **label appearance alone**, never
from a model score. Both cohorts are reported so the effect of the exclusion is
visible rather than assumed.

Test split: 16 patients / 2,017 frames, patient-disjoint from train and val.
All metrics at 256×256 (the model's operating resolution). Distances in pixels —
PFUS1 publishes no mm/px scale. Aggregation: frame → patient mean → across
patients, with a **patient-level** bootstrap 95% CI (4,000 draws, seed 42).

## Headline

| metric | Standard U-Net | Slim U-Net |
|---|---|---|
| Dice | **0.7525 ± 0.173** [0.660, 0.825] | 0.7221 ± 0.240 [0.593, 0.823] |
| IoU | 0.6318 ± 0.186 [0.538, 0.712] | 0.6112 ± 0.238 [0.488, 0.714] |
| HD95 px | 13.08 ± 9.93 [8.95, 18.46] | 13.75 ± 11.63 [8.99, 20.23] |
| ASSD px | 4.77 ± 3.81 [3.32, 6.93] | 5.16 ± 4.98 [3.25, 8.00] |
| Relative area error | 0.246 ± 0.177 [0.167, 0.334] | 0.286 ± 0.251 [0.177, 0.416] |
| **Temporal Dice (v2)** | **0.9713 ± 0.038** [0.951, 0.987] | 0.9528 ± 0.076 [0.912, 0.982] |
| Absent-transition rate | 0.0000 | 0.0412 ± 0.149 |
| Track-break rate | 0.0006 | 0.0082 ± 0.026 |
| Mask-dropout rate | 0.0003 | 0.0453 ± 0.154 |

After exclusion (13 patients): Standard Dice **0.7628** [0.654, 0.848],
Slim **0.7288** [0.578, 0.848]. Every other metric moves by less than its SD.

**The two architectures are not separated by this test set.** The 0.03 Dice gap
sits inside a ±0.09 CI. What does separate them is dropout: Slim loses the mask
on 4.5% of frames against Standard's 0.03%, a 150× difference, and for a servo
loop that is signal loss rather than inaccuracy.

## Temporal consistency — one definition

Defined once in `rus_perception/metrics/temporal.py` and imported everywhere
else; the audit scripts no longer carry their own copy.

A transition is a consecutive frame pair, always compared after motion
compensation (Farneback backward flow, nearest-neighbour warp):

| current / motion-aligned previous | kind | contributes |
|---|---|---|
| non-empty / non-empty | `scored` | its actual Dice |
| non-empty / empty | `onset` | 0.0 |
| empty / non-empty | `offset` | 0.0 |
| empty / empty | `absent` | **excluded**, counted separately |

`warped_temporal_dice` is the mean over every transition except `absent`. A
sequence with nothing scorable reports `None`, never 1.0.

Two defects this replaces:

1. **Both-empty scored 1.0.** A model that emits nothing twice in a row was
   credited with perfect stability. Slim's P021 stability of 0.946 was that
   artefact — 119 of its 125 absent transitions came from that one patient.
2. **"Warped" was not warped.** `configs` set `flow.backend: precomputed`, the
   manifest's flow columns are empty, so `evaluate.py` reused the previous mask
   unchanged and still labelled the output warped. `evaluate.py` now computes
   Farneback flow when the manifest has none, and every record carries
   `motion_compensated`, so the output can no longer claim a compensation that
   did not happen (`--no-flow-fallback` opts out and reports `false`).

## Patient exclusion — what the evidence does and does not support

Two model-blind audits ran over all 110 patients.

**`dataset_audit/label_integrity.py`** — does the label track the image?
Per-transition label-centroid displacement vs. global image displacement from
phase correlation. Median ratio across patients 1.75; nothing in the test split
is anomalous.

**`dataset_audit/label_appearance.py`** — does the label sit on fluid?
Contrast between the label interior and the ring around it, inside the imaging
sector. Median +19.6, but **26/110 patients (24%) are ≤ 0**: their labelled
region is *no darker than the tissue around it*, which a urine-filled lumen
cannot be.

Exclusion rule (pre-registered, threshold at the natural zero, not tuned):
`contrast < 0` → 26 patients, of which 3 are in the test split (P000, P019, P108).

### What this changes: almost nothing

| cohort | n | Standard Dice | 95% CI |
|---|---|---|---|
| all | 16 | 0.7525 | [0.660, 0.825] |
| retained (rule) | 13 | 0.7628 | [0.654, 0.848] |
| retained − P021 | 12 | 0.8034 | [0.723, 0.860] |
| all − P021 only | 15 | 0.7843 | [0.722, 0.839] |

Per-patient Dice, worst first: **P021 0.276**, P043 0.461, P000 0.628,
P001 0.693, P019 0.737, P095 0.754, P108 0.760 … P041 0.921.

**The bad-label patients are not the bad-score patients.** The three the rule
removes score 0.628 / 0.737 / 0.760 — middling, not worst — and removing them
moves the headline by +0.010, well inside noise.

### P021 is not excluded, and the earlier justification for excluding it was wrong

A previous session recorded P021 as having a frozen label. Direct measurement
does not support it: its label moves 0.070 px per transition against an image
that moves 0.058 px — ratio 1.21, i.e. the label tracks fine and **the clip
itself is nearly static**. The appearance audit clears it too (contrast +9.6,
interior smoother than its surroundings, darker than 82% of the sector).

Neither model-blind audit finds a defect in P021's label. Dropping it would be
dropping the hardest case, and it moves the headline by +0.032 — three times
what the actual label-quality rule moves it. It stays in.

## Files

```text
final_metrics.json                     both cohorts, both models, full summaries
final_metrics_frames.csv               4,034 per-frame rows (2 models x 2,017)
metrics/final_evaluation.py            reproduces both
dataset_audit/label_integrity.{py,csv} label-vs-image motion, all 110 patients
dataset_audit/label_appearance.{py,csv} label-on-fluid test, all 110 patients
dataset_audit/excluded_patients.json   the rule and the list it produced
dataset_audit/patient_inspect.png      GT vs prediction for the suspect patients
```

## What is not covered

- **Exclusion is evaluation-only.** The 26 flagged patients (23 of them in
  train) are still in the training set. Retraining without them is the next
  experiment; a GPU is available, and the original schedule was ~15 min.
- Metrics are at 256. Native-resolution distances are ~3.2× larger.
- 16 test patients give a ±0.09 CI. No architecture comparison on this split can
  resolve less than that, whatever the point estimates say.
