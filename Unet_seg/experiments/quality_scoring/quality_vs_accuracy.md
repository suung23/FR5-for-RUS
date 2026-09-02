# Control-quality scoring of the segmentation run

- Run: `runs/exp_seed43/best_test`
- Frames: 1292; 817 (63.2%) have Dice ≥ 0.80
- Q over the cohort: 0.921 ± 0.056 (0.919 [0.891–0.968])
- `target_area_ratio` = 0.15 ± 0.1 (plateau 0.05–0.25 of the frame)

## Does Q track segmentation accuracy?

- Spearman rho (frame level, Q vs Dice): **0.746**
- Spearman rho (patient level, mean Q vs mean Dice): **0.764** (n = 11)
- AUROC of Q as a detector of Dice ≥ 0.80: **0.855**

### Gate operating points

| Q ≥ | frames kept | retained | Dice of accepted | Dice of rejected | precision | recall |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 1292 | 100.0% | 0.807 ± 0.123 (0.848 [0.762–0.894]) | -- | 0.632 | 1.000 |
| 0.60 | 1292 | 100.0% | 0.807 ± 0.123 (0.848 [0.762–0.894]) | -- | 0.632 | 1.000 |
| 0.65 | 1292 | 100.0% | 0.807 ± 0.123 (0.848 [0.762–0.894]) | -- | 0.632 | 1.000 |
| 0.70 | 1292 | 100.0% | 0.807 ± 0.123 (0.848 [0.762–0.894]) | -- | 0.632 | 1.000 |
| 0.75 | 1281 | 99.1% | 0.808 ± 0.122 (0.849 [0.764–0.894]) | 0.619 ± 0.115 (0.620 [0.570–0.679]) | 0.637 | 0.999 |
| 0.80 | 1254 | 97.1% | 0.813 ± 0.117 (0.852 [0.770–0.895]) | 0.608 ± 0.135 (0.652 [0.480–0.704]) | 0.650 | 0.998 |
| 0.85 | 1152 | 89.2% | 0.830 ± 0.097 (0.858 [0.787–0.899]) | 0.615 ± 0.143 (0.634 [0.492–0.745]) | 0.695 | 0.980 |
| 0.90 | 840 | 65.0% | 0.865 ± 0.063 (0.875 [0.844–0.911]) | 0.699 ± 0.133 (0.740 [0.606–0.797]) | 0.845 | 0.869 |

## Sub-scores

| component | weight | frames | value | rho with Dice | AUROC |
| --- | --- | --- | --- | --- | --- |
| segmentation_confidence | 1.0 | 1292 | 0.998 ± 0.001 (0.998 [0.997–0.998]) | -0.356 | 0.382 |
| mask_completeness | 1.0 | 1292 | 0.843 ± 0.101 (0.850 [0.782–0.883]) | +0.662 | 0.772 |
| lumen_contrast | 0.5 | 1292 | 0.514 ± 0.371 (0.565 [0.184–0.880]) | +0.744 | 0.874 |
| border_penalty | 1.0 | 1292 | 1.000 ± 0.000 (1.000 [1.000–1.000]) | +0.129 | 0.500 |
| component_quality | 1.0 | 1292 | 0.987 ± 0.057 (1.000 [1.000–1.000]) | +0.028 | 0.518 |
| temporal_iou | 1.0 | 1281 | 0.953 ± 0.076 (0.979 [0.954–0.989]) | +0.450 | 0.712 |
| centroid_stability | 0.5 | 1281 | 0.952 ± 0.063 (0.973 [0.945–0.985]) | +0.253 | 0.638 |
| area_stability | 0.5 | 1281 | 0.951 ± 0.087 (0.979 [0.949–0.991]) | +0.314 | 0.656 |

### Leave-one-out ablation

Change in Q's agreement with Dice when one sub-score is removed. A large drop means the term carries the signal.

| removed | rho with Dice | Δrho | AUROC | ΔAUROC |
| --- | --- | --- | --- | --- |
| _(none — full Q)_ | +0.746 | — | 0.855 | — |
| lumen_contrast | +0.632 | -0.114 | 0.762 | -0.093 |
| mask_completeness | +0.741 | -0.005 | 0.867 | +0.012 |
| border_penalty | +0.746 | -0.000 | 0.855 | -0.000 |
| segmentation_confidence | +0.747 | +0.001 | 0.855 | +0.000 |
| component_quality | +0.752 | +0.005 | 0.865 | +0.009 |
| area_stability | +0.753 | +0.007 | 0.858 | +0.002 |
| temporal_iou | +0.754 | +0.008 | 0.860 | +0.005 |
| centroid_stability | +0.758 | +0.011 | 0.861 | +0.006 |

## Alternative weight sets (diagnostic, scored on this same run)

| weight set | rho with Dice | AUROC | Q | 5th–95th pct |
| --- | --- | --- | --- | --- |
| shipped | +0.746 | 0.855 | 0.921 ± 0.056 (0.919 [0.891–0.968]) | 0.823–0.996 |
| drop_uninformative | +0.751 | 0.864 | 0.857 ± 0.101 (0.857 [0.801–0.940]) | 0.682–0.994 |
| contrast_weighted | +0.754 | 0.876 | 0.754 ± 0.178 (0.774 [0.600–0.922]) | 0.483–0.995 |
| contrast_and_completeness | +0.758 | 0.871 | 0.679 ± 0.226 (0.712 [0.471–0.879]) | 0.333–1.000 |
| lumen_contrast_alone | +0.744 | 0.874 | 0.514 ± 0.371 (0.565 [0.184–0.880]) | 0.000–1.000 |

## Within volume bands (size held roughly constant)

| band | patients | frames | GT area px | Q | Dice | rho | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | P000, P001, P109, P108 | 544 | 1264 | 0.874 ± 0.043 (0.892 [0.850–0.904]) | 0.713 ± 0.130 (0.757 [0.650–0.812]) | +0.594 | 0.784 |
| medium | P095, P089, P012, P097 | 418 | 2302 | 0.938 ± 0.038 (0.942 [0.902–0.967]) | 0.851 ± 0.054 (0.866 [0.816–0.893]) | +0.150 | 0.509 |
| large | P085, P040, P041 | 330 | 3282 | 0.977 ± 0.015 (0.971 [0.966–0.994]) | 0.904 ± 0.034 (0.913 [0.872–0.932]) | +0.516 | -- |

## Shipped validity gate

- `valid_for_control` on **98.2%** of frames (23 rejected)
- Dice of accepted: 0.811 ± 0.118 (0.851 [0.767–0.895])
- Dice of rejected: 0.561 ± 0.133 (0.496 [0.470–0.695])

| rejection reason | frames |
| --- | --- |
| mask_area_too_small | 12 |
| low_temporal_warped_iou | 8 |
| fragmented_mask | 4 |
| excessive_area_change | 1 |

## Per patient

| patient | frames | GT area px | Q | Dice | valid | completeness | contrast | temporal IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P041 | 128 | 8984 | 0.994 ± 0.003 | 0.931 | 100% | 1.000 | 1.000 | 0.990 |
| P040 | 121 | 3135 | 0.967 ± 0.006 | 0.899 | 100% | 0.932 | 0.807 | 0.982 |
| P085 | 81 | 3052 | 0.966 ± 0.007 | 0.870 | 100% | 0.884 | 0.868 | 0.983 |
| P097 | 73 | 2857 | 0.989 ± 0.007 | 0.795 | 100% | 0.996 | 0.981 | 0.974 |
| P012 | 147 | 2336 | 0.931 ± 0.017 | 0.880 | 100% | 0.864 | 0.583 | 0.948 |
| P089 | 98 | 2168 | 0.963 ± 0.006 | 0.895 | 100% | 0.849 | 0.907 | 0.984 |
| P095 | 100 | 1788 | 0.887 ± 0.018 | 0.805 | 100% | 0.759 | 0.242 | 0.959 |
| P108 | 109 | 1578 | 0.892 ± 0.007 | 0.726 | 100% | 0.854 | 0.000 | 0.974 |
| P109 | 144 | 1484 | 0.905 ± 0.006 | 0.830 | 100% | 0.799 | 0.274 | 0.977 |
| P001 | 144 | 1119 | 0.878 ± 0.043 | 0.723 | 97% | 0.779 | 0.391 | 0.921 |
| P000 | 147 | 549 | 0.828 ± 0.040 | 0.580 | 88% | 0.666 | 0.013 | 0.841 |

## Figure cases

| role | frame | Dice | Q | valid | sub-scores |
| --- | --- | --- | --- | --- | --- |
| Best | P041 · frame 001 | 0.960 | **0.991** | yes | segmentation_c 0.99, mask_completen 1.00, component_qual 1.00, border_penalty 1.00, lumen_contrast 1.00, temporal_iou 0.98, centroid_stabi 0.96, area_stability 0.96 |
| Median | P012 · frame 059 | 0.848 | **0.943** | yes | segmentation_c 1.00, mask_completen 0.88, component_qual 1.00, border_penalty 1.00, lumen_contrast 0.56, temporal_iou 0.99, centroid_stabi 0.99, area_stability 0.99 |
| Worst | P000 · frame 070 | 0.359 | **0.713** | no | segmentation_c 1.00, mask_completen 0.63, component_qual 1.00, border_penalty 1.00, lumen_contrast 0.25, temporal_iou 0.42, centroid_stabi 0.67, area_stability 0.26 |
