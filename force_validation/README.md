# Probe normal-force validation against an electronic scale

Compares the normal force the robot reports at the probe tip against a reading
taken from an electronic scale at the same moment, and reports the
gravity-compensation error separately as its own uncertainty term.

The capture step **reads the existing force pipeline and nothing else**. It does
not subtract bias, apply a gravity model, or rotate into the probe frame — that
work is already done by `fr5_control`, and a second copy of it would eventually
disagree with the one the robot actually uses.

```
FR5 flange → adapter → PX6D F/T sensor → probe holder → convex probe
                              │
        raw wrench → bias → gravity → sensor-to-probe → tool offset
                              │
                    corrected probe-frame wrench   ← what this tool reads
                              │
                        F_n = F_zP
```

## Install

```bash
pip install -r requirements.txt
```

`pandas` and `seaborn` are deliberately absent. The tables here are flat CSVs of
a few hundred rows; the standard `csv` module and `numpy` handle them, and a
dependency that has to be installed before an experiment can be analysed is a
dependency that will one day block the analysis.

The capture step additionally needs a sourced ROS 2 workspace, because it
subscribes to the robot's own topics.

## 1. Capture

Start the bridge so the corrected wrench is being published, then:

```bash
source ~/FR5-for-RUS/install/setup.bash
python3 capture_normal_force.py --out-dir captures
```

Per sample:

1. Press the probe onto the scale.
2. **Wait for the scale reading to settle** — you are the one who can see it.
3. Press **space**. The instantaneous corrected normal force is written.
4. Write the scale mass down on paper.

`u` undoes the last capture, `q` saves and exits.

The recorded value is the value **at the keypress**. No smoothing replaces it.
The median, mean and SD of the surrounding 250 ms are stored beside it so you
can tell afterwards whether that instant was a quiet one — as evidence, not as a
filter.

Refusals are deliberate and are recorded rather than hidden:

| condition | what happens |
|---|---|
| Calibration invalid | capture mode will not start |
| Wrench not in the probe frame | capture mode will not start |
| Force data older than 100 ms | capture recorded with `capture_valid=FALSE` |
| Space pressed within 300 ms | ignored (auto-repeat debounce) |

Two files come out:

- `captures/robot_force_captures.csv` — one row per capture
- `captures/ground_truth_scale_template.csv` — sample IDs copied across, ready
  for the scale masses

Sample IDs are copied automatically. Typing them again is how two tables end up
describing different trials.

## 2. Enter the scale masses

Fill in `scale_mass_g` only. Set `valid_trial` to `FALSE` for any trial you know
was bad, and say why in `notes`. The reference force is

    F_scale = (m_g / 1000) × 9.80665

## 3. Gravity-compensation validation

The unloaded multi-pose calibration already recorded what it needs. Export it:

```bash
python3 export_gravity_validation.py
```

This replays the stored poses through `fr5_control.wrench_profile.compensate` —
the same function the robot uses — and writes the residual that survives
compensation. It does not measure anything new.

The working-pose zero is excluded by default. Those poses were captured before
that zero existed, and the question here is how well the gravity model fits, not
what the deployed pipeline currently outputs. Use `--include-working-tare` for
the latter.

## 4. Analyse

```bash
python3 run_force_validation_analysis.py \
  --robot-captures captures/robot_force_captures.csv \
  --scale-ground-truth captures/ground_truth_scale_template.csv \
  --gravity-validation captures/gravity_compensation_validation.csv \
  --scale-resolution-g 0.1 \
  --output-dir outputs
```

Without `--gravity-validation` the analysis still runs. It states *gravity
compensation uncertainty unavailable* and reports the robot–scale comparison
alone.

Outputs:

```
outputs/
├── trial_results.csv                       per-trial, including excluded ones
├── metrics_summary.csv
├── gravity_compensation_metrics.csv        tidy metric/unit/value/definition
├── analysis_report.md
├── gravity_compensation_report.md
├── normal_force_validation.png / .pdf      Figure 1
└── gravity_compensation_validation.png/.pdf  Figure 2
```

## The pipeline does *not* correct what this tool measured

A correction derived from the 2026-08-28 run was carried in the bridge for part
of 2026-08-31 and then **removed**. The measurement stands; what did not stand
was applying it silently, because it put the configured numbers and the real
ones on different scales — a 3.0 N hold target became 2.71 N in the units the
scale had measured, and a 2.0 N mode threshold fired at 1.71 N. On a console an
operator reads, a setpoint that means something else is worse than a known
offset.

So the residual below is what the deployed pipeline still has. It is reported,
not corrected. Before reinstating a correction, repeat the comparison under
phantom contact: nineteen axial presses onto a scale do not constrain a contact
carrying shear, and the threshold and target have to be converted with it.

## What this tool will not do

**The gravity residual is never subtracted from the captured force.** Removing
it would shrink the robot–scale residual without improving the measurement, and
the resulting number would mean nothing. It is reported as a separate systematic
error and as the `u_gravity` term of

    u_combined = √(u_scale² + u_gravity²)

which is an interpretation range, not a correction.

**No trial is dropped silently.** Every capture appears in `trial_results.csv`
with `included_in_primary_analysis` and, when excluded, the reason. Trials are
excluded only for a stated cause: a validity flag from the pipeline, a missing
scale mass, or `valid_trial=FALSE` set by you.

**No data is invented.** Missing values stay missing. Malformed input stops the
run with a message naming the file, the sample and the field.

## Per-capture gravity error

If both the captures and the validation poses carry orientation, each capture
takes the residual of the nearest validation pose in orientation space, with a
30° cut-off; captures with nothing inside that cut-off fall back to the global
value. No surface is fitted — with this many poses an interpolant would state
more than the data supports, and the specification says not to force one.

If orientation is missing on either side, the whole set's residual standard
deviation is applied to every capture. The report says which was used.

## Limitations

Gravity-compensation error is estimated from unloaded multi-pose residuals. That
estimate does not include contact-induced structural deformation, lateral
friction, dynamic motion effects, or the mechanical response uncertainty of the
electronic scale.

The report also tests the residual against pose order (Spearman). A strong
correlation there means the error is systematic rather than scatter, and the
standard deviation understates how far a single pose can be off — worth reading
before quoting `u_gravity` as a symmetric band.
