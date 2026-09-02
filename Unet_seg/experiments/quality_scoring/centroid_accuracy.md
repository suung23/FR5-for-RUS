# Centroid accuracy versus the centroid term in Q

- Run: `runs/exp_seed43/best_test`; 1292 frames
- `centroid_stability` in Q is `exp(-jump / 0.05)` where `jump` is the distance between this frame's predicted centroid and the previous frame's predicted centroid. **Ground truth never enters it.**
- Measured here instead: ‖predicted centroid − ground-truth centroid‖, normalized to image width at 256×256.

- Centroid error: 0.0231 ± 0.0236 (0.0165 [0.0103–0.0269]) normalized = 5.91 ± 6.03 (4.22 [2.64–6.90]) px
- 512 frames (39.6%) exceed 0.020 (5.1 px)

## Does anything in the pipeline predict centroid error?

| signal | rho with centroid error | AUROC (detect error > threshold) |
| --- | --- | --- |
| `control_quality_score` | -0.238 | 0.677 |
| `centroid_stability` | -0.201 | 0.626 |
| `dice` | -0.607 | 0.852 |

## Self-consistency is not accuracy

- Spearman rho (centroid *jump* vs centroid *error*): **+0.201**
- 89.3% of frames score `centroid_stability` ≥ 0.90
- Centroid error among those stable frames: 0.0216 ± 0.0208 (0.0158 [0.0101–0.0257])
- Centroid error among unstable frames: 0.0357 ± 0.0378 (0.0234 [0.0140–0.0349])

## Per patient

| patient | frames | GT area px | centroid error (norm) | px | bias ‖·‖ | `centroid_stability` | Q | Dice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P041 | 128 | 8984 | 0.0185 ± 0.0072 (0.0172 [0.0129–0.0226]) | 4.72 | 0.0152 | 0.973 | 0.994 | 0.931 |
| P040 | 121 | 3135 | 0.0087 ± 0.0070 (0.0059 [0.0042–0.0100]) | 2.23 | 0.0057 | 0.972 | 0.967 | 0.899 |
| P085 | 81 | 3052 | 0.0146 ± 0.0034 (0.0142 [0.0132–0.0178]) | 3.75 | 0.0142 | 0.979 | 0.966 | 0.870 |
| P097 | 73 | 2857 | 0.0448 ± 0.0167 (0.0456 [0.0293–0.0584]) | 11.47 | 0.0432 | 0.958 | 0.989 | 0.795 |
| P012 | 147 | 2336 | 0.0144 ± 0.0039 (0.0143 [0.0118–0.0168]) | 3.68 | 0.0120 | 0.935 | 0.931 | 0.880 |
| P089 | 98 | 2168 | 0.0098 ± 0.0021 (0.0097 [0.0082–0.0110]) | 2.51 | 0.0090 | 0.980 | 0.963 | 0.895 |
| P095 | 100 | 1788 | 0.0217 ± 0.0064 (0.0214 [0.0160–0.0271]) | 5.55 | 0.0192 | 0.951 | 0.887 | 0.805 |
| P108 | 109 | 1578 | 0.0348 ± 0.0098 (0.0336 [0.0255–0.0440]) | 8.91 | 0.0346 | 0.967 | 0.892 | 0.726 |
| P109 | 144 | 1484 | 0.0112 ± 0.0059 (0.0102 [0.0062–0.0155]) | 2.87 | 0.0064 | 0.972 | 0.905 | 0.830 |
| P001 | 144 | 1119 | 0.0559 ± 0.0494 (0.0286 [0.0172–0.0921]) | 14.30 | 0.0539 | 0.919 | 0.878 | 0.723 |
| P000 | 147 | 549 | 0.0222 ± 0.0134 (0.0241 [0.0083–0.0325]) | 5.70 | 0.0185 | 0.901 | 0.828 | 0.580 |

## Figure cases

| role | frame | Dice | Q | `centroid_stability` | centroid error (px) |
| --- | --- | --- | --- | --- | --- |
| Best | P041 · frame 001 | 0.960 | 0.991 | 0.962 | **1.67** |
| Median | P012 · frame 059 | 0.848 | 0.943 | 0.985 | **3.74** |
| Worst | P000 · frame 070 | 0.359 | 0.713 | 0.673 | **5.73** |
