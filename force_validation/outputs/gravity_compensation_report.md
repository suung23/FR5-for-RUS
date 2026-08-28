# Gravity compensation validation

Generated 2026-08-28T14:00:39+09:00

Unloaded multi-pose validation showed a gravity-compensation residual of 0.000 ± 0.182 N (mean ± SD) along the probe normal axis, with an RMSE of 0.177 N and a 95% error range of -0.357 to 0.357 N.

| metric | value |
|---|---|
| Validation poses | 19 |
| Bias | 0.0000 N |
| Residual SD | 0.1821 N |
| RMSE | 0.1773 N |
| Max absolute residual | 0.3760 N |
| 95% error range (mean ± 1.96 SD) | -0.3570 to 0.3570 N |
| Robust 95% (2.5–97.5 pct) | -0.3289 to 0.2642 N |
| Scale standard uncertainty | 0.00283 N |
| Combined reference uncertainty | 0.18216 N |
| Per-capture assignment | global |
| Pose-order trend (Spearman rho) | 0.896 (p = 0.0000) |

Residual versus pose order: Spearman rho = +0.896 (p = 0.0000). This is a systematic component, not scatter: the residual is not independent of when the pose was taken, so the standard deviation understates what a single pose can be off by.

The residual is reported, not removed. Subtracting it from the captured robot force would shrink the robot–scale residual without improving the measurement, and the resulting number would mean nothing.

## Limitations

Gravity-compensation error was estimated from unloaded multi-pose residuals. This estimate does not include contact-induced structural deformation, lateral friction, dynamic motion effects, or mechanical response uncertainty of the electronic scale.
