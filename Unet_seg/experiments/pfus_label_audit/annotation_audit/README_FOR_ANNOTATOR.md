# Read me first

You are re-annotating **urine-filled bladder lumen** on 80 anonymised pelvic-floor
ultrasound frames. Full rules: `../ANNOTATION_PROTOCOL.md`.

Short version:

- One polygon per frame, label `bladder_lumen`, following the **inner bladder wall**.
- Bladder **wall tissue is excluded**. Only the anechoic fluid volume counts.
- If no lumen is defensible, leave the frame empty — do not guess a shape.
- Fill `annotation_quality` (good/moderate/poor) and `uncertain_boundary` (0/1)
  for every frame in `annotation_record.csv`, including empty ones.

The images carry no patient or frame identifiers and are in randomised order.
This is intentional: the audit measures how much the existing dataset label
agrees with an independent reading, so please do not seek out the original
dataset annotations before finishing.
