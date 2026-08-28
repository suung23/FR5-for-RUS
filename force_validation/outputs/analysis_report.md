# Probe normal-force validation against an electronic scale

Generated 2026-08-28T14:00:39+09:00

## Captures

- Captures recorded: **20**
- Included in primary analysis: **19**
- Excluded: **1**

| sample_id | reason |
|---|---|
| 2 | valid_trial=FALSE |

## Force pipeline state

The capture mode reads the final corrected probe-frame wrench from the existing pipeline. It does not re-apply bias subtraction, gravity compensation, or the sensor-to-probe transform.

- Sign convention: **negated at capture (compression positive)**
- Sensor calibration: TRUE
- Gravity compensation: TRUE
- Probe-frame transform: TRUE

## Robot versus scale accuracy

| metric | value |
|---|---|
| n | 19 |
| Mean bias | -0.2928 N |
| MAE | 0.3441 N |
| RMSE | 0.3657 N |
| Max absolute error | 0.5305 N |
| Residual SD | 0.2251 N |
| Full-scale reference | 9.846 N |
| MAE (%FS) | 3.49 % |
| RMSE (%FS) | 3.71 % |
| Max error (%FS) | 5.39 % |

### Linear regression

`F_robot = a · F_scale + b`

| term | value | 95% CI |
|---|---|---|
| slope a | 1.0194 | 0.9746 to 1.0641 |
| intercept b | -0.3727 N | -0.5871 to -0.1582 |
| R² | 0.99269 | |

### Bland–Altman

- Bias: **-0.2928 N**
- 95% limits of agreement: -0.7339 to 0.1484 N

### Repeatability

| target level | n | SD | CV |
|---|---|---|---|
| 1.42 N | 2 | 0.0153 N | 1.14 % |
| 1.83 N | 2 | 0.1506 N | 10.51 % |
| 4.31 N | 4 | 0.0473 N | 1.21 % |

## Reference-side uncertainty

- Scale resolution: 1 g (0.0098 N)
- Scale standard uncertainty `u_scale = Δ/√12`: **0.00283 N**
- Gravity-compensation uncertainty `u_gravity = σ_gravity`: **0.18214 N**
- Combined `u_combined = √(u_scale² + u_gravity²)`: **0.18216 N**

This uncertainty does **not** correct the robot–scale residual. It is the range within which the reference itself is known, and is reported alongside the residual rather than subtracted from it.

## Gravity compensation

Unloaded multi-pose validation showed a gravity-compensation residual of 0.000 ± 0.182 N (mean ± SD) along the probe normal axis, with an RMSE of 0.177 N and a 95% error range of -0.357 to 0.357 N.

- Poses: 19
- Max absolute residual: 0.3760 N
- Robust 95% interval (2.5–97.5 percentile): -0.3289 to 0.2642 N
- Per-capture gravity uncertainty applied as: **global**

Residual versus pose order: Spearman rho = +0.896 (p = 0.0000). This is a systematic component, not scatter: the residual is not independent of when the pose was taken, so the standard deviation understates what a single pose can be off by.

## Limitations

Gravity-compensation error was estimated from unloaded multi-pose residuals. This estimate does not include contact-induced structural deformation, lateral friction, dynamic motion effects, or mechanical response uncertainty of the electronic scale.
