# External validation data

Nothing here yet. This directory documents what an external dataset must look
like for `evaluate_external.py` to run against it, and records what was checked.

## What was searched

`/home/rosotauser/datasets/` contains only `pfus/`. No adult bladder ultrasound
dataset is present on this machine. **No download was attempted** — clinical
ultrasound datasets carry licence and IRB conditions that a script must not
assume on the user's behalf.

## Priority 1 — MGH adult bladder ultrasound

The right first external set for this project, because it matches the target
this repository actually needs:

| Property | Requirement |
|---|---|
| Anatomy | Adult urinary bladder |
| Views | Sagittal and/or transverse |
| Annotation | **Inner bladder wall** — i.e. the urine-filled lumen, not the organ |
| Modality | B-mode, any depth setting |
| Subject metadata | Subject identifier per frame, so evaluation stays patient-level |

The inner-wall annotation is the reason this dataset is priority 1: it is
annotated to the same target the FR5 control loop needs, so it tests both
domain generalisation *and* the Phase 7 label-definition question from the
outside.

## Expected layout

```text
external_data/MGH/
    images/          <subject>_<frame>.png      B-mode frames, any resolution
    masks/           <subject>_<frame>.png      binary, 0 = background, >0 = lumen
    metadata.csv
```

`metadata.csv` columns (extra columns are ignored):

```text
subject_id        required  the unit of aggregation; never split within a subject
frame_id          required  unique within subject
image_path        required  relative to external_data/<dataset>/
mask_path         required  relative to external_data/<dataset>/
view              optional  sagittal | transverse
pixel_spacing_mm  optional  if present, HD95/ASSD are reported in mm instead of px
sequence_id       optional  groups frames of one clip; enables temporal metrics
```

Any dataset matching this contract works — the loader is not MGH-specific.
Register a new one by adding a directory and passing `--dataset <name>`.

## Running it

```bash
# Phase 12 -- zero-shot, no fine-tuning, PFUS1 weights unchanged
python3 external_data/evaluate_external.py \
    --dataset MGH \
    --checkpoint ../../checkpoints/pfus_bladder/best.pt \
    --config ../../configs/pfus_bladder.yaml
```

## Phase 12/13 protocol (do not reorder)

1. **Zero-shot first.** Evaluate the PFUS1 checkpoint unchanged. This number is
   the actual domain-generalisation result and it can only be measured once —
   any fine-tuning destroys it.
2. Only then fine-tune, into a **separate** experiment directory.
3. Keep the three regimes apart and never merge their metrics:
   `PFUS1 only` · `PFUS1 → external fine-tune` · `external only`.

## Caveats that will apply on arrival

- PFUS1 is **pelvic-floor midsagittal** imaging with a fixed anatomical
  orientation, which is why `horizontal_flip` and `vertical_flip` are both 0 in
  the training config. An external set with a different probe orientation is a
  harder domain shift than the Dice gap alone will suggest.
- PFUS1 has no mm/px scale, so the model has never seen a physical size prior.
  If the external set has `pixel_spacing_mm`, report distances in mm but expect
  the area-based metrics to shift.
- `intensity_normalization: per_image` means gain/TGC differences transfer
  reasonably well. Depth-setting differences do not.
