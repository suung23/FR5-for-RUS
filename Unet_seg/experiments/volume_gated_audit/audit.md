# Volume-gated segmentation and control-quality audit

- Volume floor: ground-truth bladder area ≥ **0.0574** of the ROI
- Gate: `valid_for_control` **and** Q ≥ **0.85**
- A frame counts as segmented when Dice ≥ 0.80, and as steerable when centroid error ≤ 0.020 (5.1 px at 256×256)

## 1. What the volume floor costs

| floor (GT area / ROI) | val frames | val patients | test frames | test patients | test Dice |
| --- | --- | --- | --- | --- | --- |
| 0.0000 | 1661 | 11 | 1292 | 11 | 0.807 ± 0.123 (0.848 [0.762–0.894]) |
| 0.0300 | 1431 | 10 | 995 | 10 | 0.853 ± 0.070 (0.866 [0.812–0.906]) |
| 0.0574 **←** | 994 | 7 | 396 | 6 | 0.892 ± 0.047 (0.910 [0.865–0.930]) |
| 0.0800 | 650 | 5 | 135 | 2 | 0.929 ± 0.016 (0.931 [0.921–0.938]) |
| 0.1200 | 134 | 1 | 128 | 1 | 0.931 ± 0.011 (0.932 [0.923–0.938]) |
| 0.1600 | 0 | 0 | 128 | 1 | 0.931 ± 0.011 (0.932 [0.923–0.938]) |

## 2. Segmentation on the surviving frames

| split | frames | patients | Dice | IoU | HD95 (px) | centroid error (px) |
| --- | --- | --- | --- | --- | --- | --- |
| val | 994 | 7 | 0.882 ± 0.044 (0.888 [0.851–0.913]) | 0.792 ± 0.069 (0.799 [0.741–0.840]) | 9.74 ± 3.23 (9.57 [7.39–11.91]) | 4.07 ± 2.44 (3.85 [2.29–5.12]) |
| test | 396 | 6 | 0.892 ± 0.047 (0.910 [0.865–0.930]) | 0.809 ± 0.073 (0.835 [0.762–0.868]) | 10.82 ± 5.15 (10.53 [6.76–13.73]) | 4.35 ± 3.07 (3.65 [2.38–5.29]) |

## 3. Q and its sub-scores on those frames

| term | weight | val | test | val AUROC | test AUROC | limiting share (test) |
| --- | --- | --- | --- | --- | --- | --- |
| **Q (aggregate)** | — | 0.962 ± 0.047 (0.983 [0.959–0.995]) | 0.987 ± 0.015 (0.992 [0.983–0.995]) | 0.755 | 0.598 | — |
| `segmentation_confidence` | 0.0 | 0.995 ± 0.001 | 0.995 ± 0.002 | 0.051 | 0.501 | 0% |
| `mask_completeness` | 1.0 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.500 | 0.500 | 1% |
| `lumen_contrast` | 0.5 | 0.818 ± 0.252 | 0.972 ± 0.077 | 0.719 | 0.498 | 16% |
| `border_penalty` | 0.0 | 0.870 ± 0.212 | 0.696 ± 0.275 | 0.323 | 0.131 | 0% |
| `component_quality` | 1.0 | 1.000 ± 0.001 | 1.000 ± 0.000 | 0.499 | 0.500 | 0% |
| `temporal_iou` | 1.0 | 0.984 ± 0.016 | 0.982 ± 0.016 | 0.952 | 0.701 | 1% |
| `centroid_stability` | 0.5 | 0.976 ± 0.025 | 0.971 ± 0.025 | 0.874 | 0.686 | 56% |
| `area_stability` | 0.5 | 0.978 ± 0.027 | 0.977 ± 0.027 | 0.835 | 0.481 | 26% |

## 4. Metrics that need work

Verdicts are issued only where val and test agree; where they do not, that disagreement is the finding.

| term | weight | verdict | what to do |
| --- | --- | --- | --- |
| `segmentation_confidence` | 0.0 | **SATURATED** | Spends >=90% of its mass at the ceiling on both cohorts: it cannot separate anything. Either measure it over a region where the decision is hard, or drop it and reclaim the weight. |
| `mask_completeness` | 1.0 | **REDUNDANT UNDER THIS FILTER** | Varies on the full cohort (ceiling mass 60%, 68%) but is pinned at 1.0 once the volume floor is applied: the filter already selects for what it measures. Judge it on unfiltered frames, or drop it when a volume floor is enforced upstream. |
| `lumen_contrast` | 0.5 | **UNSTABLE** | Cohorts disagree (AUROC 0.719, 0.498). Do not act on this term until a larger cohort settles it. |
| `border_penalty` | 0.0 | **INVERTED** | Ranks the wrong way on both cohorts. Reweighting cannot fix a sign; remove it from the score and keep it, if at all, as a hard validity floor. |
| `component_quality` | 1.0 | **SATURATED** | Spends >=90% of its mass at the ceiling on both cohorts: it cannot separate anything. Either measure it over a region where the decision is hard, or drop it and reclaim the weight. |
| `temporal_iou` | 1.0 | **WORKING** | Ranks consistently; keep as is. |
| `centroid_stability` | 0.5 | **WORKING** | Ranks consistently; keep as is. |
| `area_stability` | 0.5 | **UNSTABLE** | Cohorts disagree (AUROC 0.835, 0.481). Do not act on this term until a larger cohort settles it. |

## 5. What the gate lets through

**val** — 32 frames admitted with Dice < 0.80 (3.3% of everything admitted).

- Their Dice: 0.762 ± 0.024 (0.769 [0.744–0.778]); centroid error 4.32 ± 2.01 (4.18 [2.64–5.73]) px
- **28% have every sub-score above 0.8** — no reweighting of the current terms separates them
- Where the rest were held back: `lumen_contrast` ×23
- Concentrated in: P032 ×32

**test** — 19 frames admitted with Dice < 0.80 (4.8% of everything admitted).

- Their Dice: 0.759 ± 0.027 (0.763 [0.730–0.783]); centroid error 13.98 ± 2.48 (13.76 [12.80–15.13]) px
- **100% have every sub-score above 0.8** — no reweighting of the current terms separates them
- Where the rest were held back: nowhere
- Concentrated in: P097 ×19
