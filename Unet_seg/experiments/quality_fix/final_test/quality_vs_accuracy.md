# Control-quality scoring of the segmentation run

- Run: `runs/exp_seed43_qfix/best_test`
- Frames: 1292; 817 (63.2%) have Dice ≥ 0.80
- Q over the cohort: 0.717 ± 0.126 (0.761 [0.633–0.828])
- `target_area_ratio` = 0.0683 ± 0.027 (plateau 0.04–0.10 of the frame)

## Does Q track segmentation accuracy?

- Spearman rho (frame level, Q vs Dice): **0.766**
- Spearman rho (patient level, mean Q vs mean Dice): **0.782** (n = 11)
- AUROC of Q as a detector of Dice ≥ 0.80: **0.891**

### Gate operating points

| Q ≥ | frames kept | retained | Dice of accepted | Dice of rejected | precision | recall |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 1152 | 89.2% | 0.834 ± 0.089 (0.858 [0.789–0.899]) | 0.578 ± 0.122 (0.530 [0.481–0.696]) | 0.708 | 0.999 |
| 0.60 | 995 | 77.0% | 0.850 ± 0.080 (0.866 [0.823–0.906]) | 0.659 ± 0.127 (0.694 [0.537–0.764]) | 0.799 | 0.973 |
| 0.65 | 938 | 72.6% | 0.857 ± 0.070 (0.869 [0.828–0.908]) | 0.672 ± 0.132 (0.704 [0.544–0.782]) | 0.813 | 0.934 |
| 0.70 | 856 | 66.3% | 0.865 ± 0.062 (0.875 [0.844–0.911]) | 0.691 ± 0.130 (0.727 [0.595–0.791]) | 0.854 | 0.895 |
| 0.75 | 659 | 51.0% | 0.880 ± 0.053 (0.891 [0.859–0.915]) | 0.730 ± 0.128 (0.772 [0.674–0.824]) | 0.910 | 0.734 |
| 0.80 | 500 | 38.7% | 0.886 ± 0.052 (0.897 [0.865–0.922]) | 0.756 ± 0.128 (0.791 [0.696–0.854]) | 0.916 | 0.561 |
| 0.85 | 6 | 0.5% | 0.874 ± 0.060 (0.868 [0.857–0.903]) | 0.806 ± 0.123 (0.848 [0.761–0.893]) | 0.833 | 0.006 |
| 0.90 | 5 | 0.4% | 0.875 ± 0.067 (0.870 [0.854–0.914]) | 0.806 ± 0.123 (0.848 [0.761–0.893]) | 0.800 | 0.005 |

## Sub-scores

| component | weight | frames | value | rho with Dice | AUROC |
| --- | --- | --- | --- | --- | --- |
| segmentation_confidence | 0.0 | 1292 | 0.996 ± 0.001 (0.997 [0.996–0.997]) | -0.344 | 0.387 |
| mask_completeness | 1.0 | 1292 | 0.887 ± 0.212 (1.000 [0.892–1.000]) | +0.121 | 0.698 |
| lumen_contrast | 0.5 | 1292 | 0.547 ± 0.395 (0.579 [0.199–1.000]) | +0.738 | 0.888 |
| border_penalty | 0.0 | 1292 | 0.903 ± 0.208 (1.000 [1.000–1.000]) | -0.515 | 0.321 |
| component_quality | 1.0 | 1292 | 0.987 ± 0.057 (1.000 [1.000–1.000]) | +0.028 | 0.518 |
| temporal_iou | 1.0 | 1281 | 0.953 ± 0.076 (0.979 [0.954–0.989]) | +0.450 | 0.712 |
| centroid_stability | 0.5 | 1281 | 0.952 ± 0.063 (0.973 [0.945–0.985]) | +0.253 | 0.638 |
| area_stability | 0.5 | 1281 | 0.201 ± 0.033 (0.199 [0.195–0.203]) | +0.037 | 0.514 |

### Leave-one-out ablation

Change in Q's agreement with Dice when one sub-score is removed. A large drop means the term carries the signal.

| removed | rho with Dice | Δrho | AUROC | ΔAUROC |
| --- | --- | --- | --- | --- |
| _(none — full Q)_ | +0.737 | — | 0.866 | — |
| lumen_contrast | +0.544 | -0.192 | 0.749 | -0.117 |
| temporal_iou | +0.731 | -0.006 | 0.867 | +0.001 |
| area_stability | +0.737 | +0.001 | 0.867 | +0.001 |
| component_quality | +0.741 | +0.005 | 0.874 | +0.008 |
| mask_completeness | +0.741 | +0.005 | 0.880 | +0.014 |
| centroid_stability | +0.749 | +0.012 | 0.870 | +0.004 |

## Alternative weight sets (diagnostic, scored on this same run)

| weight set | rho with Dice | AUROC | Q | 5th–95th pct |
| --- | --- | --- | --- | --- |
| shipped | +0.737 | 0.866 | 0.817 ± 0.096 (0.832 [0.784–0.902]) | 0.614–0.909 |
| drop_uninformative | +0.741 | 0.874 | 0.769 ± 0.120 (0.788 [0.725–0.874]) | 0.512–0.883 |
| contrast_weighted | +0.753 | 0.891 | 0.702 ± 0.195 (0.733 [0.533–0.910]) | 0.364–0.918 |
| contrast_and_completeness | +0.749 | 0.875 | 0.717 ± 0.271 (0.780 [0.500–1.000]) | 0.182–1.000 |
| lumen_contrast_alone | +0.738 | 0.888 | 0.547 ± 0.395 (0.579 [0.199–1.000]) | 0.000–1.000 |

## Within volume bands (size held roughly constant)

| band | patients | frames | GT area px | Q | Dice | rho | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | P000, P001, P109, P108 | 544 | 1264 | 0.611 ± 0.112 (0.597 [0.496–0.710]) | 0.713 ± 0.130 (0.757 [0.650–0.812]) | +0.642 | 0.841 |
| medium | P095, P089, P012, P097 | 418 | 2302 | 0.767 ± 0.073 (0.785 [0.730–0.825]) | 0.851 ± 0.054 (0.866 [0.816–0.893]) | +0.302 | 0.580 |
| large | P085, P040, P041 | 330 | 3282 | 0.830 ± 0.018 (0.831 [0.826–0.833]) | 0.904 ± 0.034 (0.913 [0.872–0.932]) | +0.071 | -- |

## Shipped validity gate

- `valid_for_control` on **98.2%** of frames (23 rejected)
- Dice of accepted: 0.810 ± 0.120 (0.851 [0.766–0.895])
- Dice of rejected: 0.627 ± 0.136 (0.658 [0.520–0.745])

| rejection reason | frames |
| --- | --- |
| excessive_area_change | 9 |
| low_temporal_warped_iou | 8 |
| mask_area_too_small | 5 |
| fragmented_mask | 4 |

## Per patient

| patient | frames | GT area px | Q | Dice | valid | completeness | contrast | temporal IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P041 | 128 | 8984 | 0.832 ± 0.015 | 0.931 | 100% | 1.000 | 1.000 | 0.990 |
| P040 | 121 | 3135 | 0.826 ± 0.019 | 0.899 | 100% | 1.000 | 0.949 | 0.982 |
| P085 | 81 | 3052 | 0.832 ± 0.019 | 0.870 | 100% | 1.000 | 1.000 | 0.983 |
| P097 | 73 | 2857 | 0.827 ± 0.021 | 0.795 | 100% | 1.000 | 0.981 | 0.974 |
| P012 | 147 | 2336 | 0.772 ± 0.021 | 0.880 | 100% | 1.000 | 0.583 | 0.948 |
| P089 | 98 | 2168 | 0.828 ± 0.017 | 0.895 | 100% | 1.000 | 0.949 | 0.984 |
| P095 | 100 | 1788 | 0.654 ± 0.045 | 0.805 | 98% | 0.761 | 0.242 | 0.959 |
| P108 | 109 | 1578 | 0.593 ± 0.006 | 0.726 | 100% | 1.000 | 0.000 | 0.974 |
| P109 | 144 | 1484 | 0.711 ± 0.013 | 0.830 | 100% | 0.962 | 0.274 | 0.977 |
| P001 | 144 | 1119 | 0.680 ± 0.084 | 0.723 | 97% | 0.834 | 0.467 | 0.921 |
| P000 | 147 | 549 | 0.458 ± 0.038 | 0.580 | 89% | 0.369 | 0.013 | 0.841 |

## Figure cases

| role | frame | Dice | Q | valid | sub-scores |
| --- | --- | --- | --- | --- | --- |
| Best | P041 · frame 001 | 0.960 | **0.826** | yes | segmentation_c 0.99, mask_completen 1.00, component_qual 1.00, border_penalty 0.44, lumen_contrast 1.00, temporal_iou 0.98, centroid_stabi 0.96, area_stability 0.19 |
| Median | P012 · frame 059 | 0.848 | **0.780** | yes | segmentation_c 1.00, mask_completen 1.00, component_qual 1.00, border_penalty 1.00, lumen_contrast 0.56, temporal_iou 0.99, centroid_stabi 0.99, area_stability 0.20 |
| Worst | P000 · frame 070 | 0.359 | **0.484** | no | segmentation_c 1.00, mask_completen 0.26, component_qual 1.00, border_penalty 1.00, lumen_contrast 0.25, temporal_iou 0.42, centroid_stabi 0.67, area_stability 0.76 |
