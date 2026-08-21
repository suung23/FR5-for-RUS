# Manual Bladder-Lumen Annotation Protocol (PFUS1 audit)

Version 1.0 · seed 42 subset · 80 frames / 16 patients

## Why this exists

The PFUS1 dataset annotates a structure called **`Bladder`** as one of eight
pelvic-floor polygons. The segmentation model in this repository was trained on
that label and is used as a **bladder-lumen** detector by the FR5 ultrasound
probing control loop. Whether those two targets are the same thing has never
been verified — `scripts/prepare_pfus.py` asserts it in a docstring, but neither
the dataset's own documentation nor its polygons establish it.

This audit re-annotates 80 frames under an explicit lumen definition so the two
targets can be measured against each other.

## Target definition

```text
Target:
  Urine-filled bladder lumen — the anechoic fluid volume only.

Boundary:
  The INNER bladder wall, i.e. the fluid/wall interface.
  Where the wall is resolvable, the contour follows the luminal side of it.

Include:
  Anechoic (dark) fluid contiguous with the bladder cavity.
  Low-level internal echoes / reverberation artefact inside the fluid.

Exclude:
  Bladder wall tissue (detrusor, mucosa) — even when it is thin.
  Urethra and the bladder neck beyond the internal urethral orifice.
  Adjacent pelvic structures: uterus, vagina, rectum, pubis, levator ani.
  Acoustic shadow or enhancement outside the fluid volume.
  Any region that is dark because of shadowing rather than fluid.
```

### Edge cases

| Situation | Rule |
|---|---|
| Empty / collapsed bladder, no resolvable fluid | Annotate nothing (empty mask), set `annotation_quality = poor`, `uncertain_boundary = 1` |
| Fluid present but wall not resolvable on one side | Follow the fluid/tissue intensity transition, set `uncertain_boundary = 1` |
| Posterior acoustic enhancement below the bladder | Excluded — it is not fluid |
| Shadow crossing the lumen (bowel gas, pubic shadow) | Interpolate across the shadow along the visible contour, set `uncertain_boundary = 1` |
| Two apparently separate dark regions | Annotate only the one contiguous with the bladder cavity |
| Boundary genuinely indeterminate | Do **not** guess a plausible shape. Mark `annotation_quality = poor` and annotate only what is defensible |

**Do not force a contour to look like a bladder.** A frame recorded as
`uncertain` is a result, not a failure — the whole point of the audit is to
measure how much of PFUS1's label is inference rather than observation.

## What the annotator receives

```text
annotation_audit/images_blinded/    blind_0000.png … blind_0079.png
annotation_audit/labelme/           blind_0000.json … blind_0079.json  (empty templates)
ANNOTATION_PROTOCOL.md              this file
annotation_record.csv               one row per blinded image, to be filled in
```

**Deliberately withheld** so the re-annotation stays independent of PFUS1:
`manifest.csv`, `blind_key.csv`, `original_masks/`, `overlays/`, and every model
prediction. Filenames are shuffled with seed 42, so neither patient identity nor
frame order can be recovered from the package. Frames from the same patient are
not adjacent in the listing.

## Procedure

1. Open `blind_XXXX.png` in LabelMe (`labelme annotation_audit/images_blinded --output annotation_audit/labelme`), CVAT, or any tool that writes a polygon.
2. Draw **one polygon** per frame with label `bladder_lumen`. Leave the frame empty if no lumen is defensible.
3. Record, for each frame, in `annotation_record.csv`:
   - `annotation_quality` ∈ {`good`, `moderate`, `poor`} — confidence in the drawn contour
   - `uncertain_boundary` ∈ {0, 1} — whether any part of the contour was inferred rather than seen
   - `notes` — free text, optional
4. Do not revisit earlier frames after seeing later ones from what appears to be the same patient.

Annotation format: LabelMe JSON with `shapes[].label == "bladder_lumen"` and
`shapes[].shape_type == "polygon"`. `evaluate_annotation_agreement.py` also
accepts binary PNG masks dropped into `annotation_audit/manual_masks/` named by
`sample_id` — use whichever the tool produces.

## After annotation

```bash
python3 metrics/rasterise_manual.py          # LabelMe JSON -> manual_masks/*.png
python3 metrics/evaluate_annotation_agreement.py
python3 metrics/ceiling_analysis.py
```

Metrics are reported in **pixels**: PFUS1 publishes no mm/px scale, and frame
size varies by a few pixels between frames of the same patient.
