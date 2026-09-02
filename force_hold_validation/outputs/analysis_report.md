# Constant-force hold validation

Generated 2026-09-02T16:50:01+09:00

## Runs

| run | type | target_n | samples | included | reason |
|---|---|---|---|---|---|
| A_t0p5 | hold | 0.50 | 60174 | yes |  |
| A_t1p0 | hold | 1.00 | 60164 | no | 접촉 프로빙에 한 번도 들어가지 않았다 |
| A_t1p5 | hold | 1.50 | 60166 | no | 접촉 프로빙에 한 번도 들어가지 않았다 |
| A_t2p0 | hold | 2.00 | 60170 | yes |  |
| A_t3p0 | hold | 3.00 | 60172 | yes |  |
| A_t4p0 | hold | 4.00 | 60167 | yes |  |
| B_t0p5 | disturbance | 0.50 | 87614 | yes |  |
| B_t1p0 | disturbance | 1.00 | 74762 | no | 접촉 프로빙에 한 번도 들어가지 않았다 |
| B_t1p5 | disturbance | 1.50 | 67141 | no | 접촉 프로빙에 한 번도 들어가지 않았다 |
| B_t2p0 | disturbance | 2.00 | 60499 | yes |  |
| B_t3p0_pose | disturbance | 3.00 | 56695 | yes |  |
| B_t3p0 | disturbance | 3.00 | 55331 | yes |  |
| B_t4p0 | disturbance | 4.00 | 59543 | yes |  |
| C_limit | safety | 3.00 | 91014 | yes |  |
| S_stiff | stiffness | 3.00 | 23219 | yes |  |


## Holding, by target

**품질은 SD 로 본다.** 목표 대비 평균 편차는 대부분 데드밴드가 설계대로
낸 것이지 추종 실패가 아니다 — 조절기는 밴드 안에 들어서면 `v_z = 0` 을
내고 더 가지 않으므로, **밴드 안에서는 멈춘 자리가 곧 목표다.** 목표는
도달할 지점이 아니라 그 둘레 밴드를 정의하는 값이다.

아래에서 올라오는 경우 그 운전점은 `max(진입 문턱, 목표 − 밴드)` 이고,
`error_vs_settling_n` 이 그 자리를 기준으로 한 실제 추종 오차다. SD 는
밴드 폭과 무관하므로 밴드가 서로 다른 실행끼리도 **직접 비교된다** —
이 표에서 대역 간 비교가 가능한 열은 그것 하나다.

**판정은 접촉력 ‖F‖ 로 한다** — `us_diff_ik` 가 문턱에 대는 값과 같은 스칼라이고,
`safety.max_contact_force_n` 도 같은 값에 걸린다. 법선 성분은 `fn_n` 열에
증거로만 남는다.

정착 구간(프로빙 진입 직후 3 s)은 버렸다 — 목표로 가는 과도이지 유지가 아니다.

| target_n | band_n | settling_point_n | sd_n | error_vs_settling_n | mean_error_n | seconds | in_band_pct | max_force_n |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.15 | 0.40 | 0.065 | +0.092 | -0.008 | 57.0 | 100.0 | 0.64 |
| 2.00 | 0.50 | 2.00 | 0.022 | -0.096 | -0.096 | 57.0 | 100.0 | 2.02 |
| 3.00 | 0.50 | 2.50 | 0.041 | +0.075 | -0.425 | 57.0 | 98.9 | 2.73 |
| 4.00 | 0.50 | 3.50 | 0.040 | +0.070 | -0.430 | 57.0 | 98.6 | 3.74 |


## Disturbance response

| run | target_n | direction | step_ml | peak_error_n | time_to_peak_s | settle_s | recovered |
|---|---|---|---|---|---|---|---|
| B_t0p5 | 0.50 | in | 150.0 | +0.147 | 3.24 | 0.00 | True |
| B_t0p5 | 0.50 | withdraw | 150.0 | -0.154 | 3.14 | 0.00 | True |
| B_t0p5 | 0.50 | in | 150.0 | -0.139 | 0.67 | 0.00 | True |
| B_t0p5 | 0.50 | withdraw | 150.0 | -0.122 | 2.36 | 0.00 | True |
| B_t0p5 | 0.50 | in | 150.0 | +0.141 | 1.73 | 0.00 | True |
| B_t0p5 | 0.50 | withdraw | 150.0 | +0.097 | 0.21 | 0.00 | True |
| B_t0p5 | 0.50 | in | 150.0 | +0.222 | 1.40 | 0.00 | True |
| B_t0p5 | 0.50 | withdraw | 150.0 | +0.162 | 0.00 | 0.66 | True |
| B_t0p5 | 0.50 | in | 150.0 | +0.243 | 1.44 | 0.00 | True |
| B_t0p5 | 0.50 | withdraw | 150.0 | +0.158 | 0.22 | 0.64 | True |
| B_t2p0 | 2.00 | in | 150.0 | +0.107 | 1.90 | 1.35 | True |
| B_t2p0 | 2.00 | withdraw | 150.0 | -0.270 | 3.23 | 0.00 | True |
| B_t2p0 | 2.00 | in | 150.0 | -0.239 | 0.75 | 0.00 | True |
| B_t2p0 | 2.00 | withdraw | 150.0 | -0.278 | 2.50 | 0.00 | True |
| B_t2p0 | 2.00 | in | 150.0 | -0.246 | 0.09 | 0.00 | True |
| B_t2p0 | 2.00 | withdraw | 150.0 | -0.276 | 4.34 | 0.00 | True |
| B_t2p0 | 2.00 | in | 150.0 | -0.252 | 0.65 | 0.00 | True |
| B_t2p0 | 2.00 | withdraw | 150.0 | -0.273 | 3.09 | 0.00 | True |
| B_t2p0 | 2.00 | in | 150.0 | -0.235 | 0.14 | 0.00 | True |
| B_t2p0 | 2.00 | withdraw | 150.0 | -0.255 | 5.41 | 0.00 | True |
| B_t3p0_pose | 3.00 | in | 500.0 | +0.687 | 3.05 | 0.29 | True |
| B_t3p0_pose | 3.00 | withdraw | 500.0 | -0.736 | 1.11 | 0.00 | True |
| B_t3p0_pose | 3.00 | in | 500.0 | +0.920 | 2.20 | 0.00 | True |
| B_t3p0_pose | 3.00 | withdraw | 500.0 | -0.779 | 1.28 | 0.00 | True |
| B_t3p0_pose | 3.00 | in | 500.0 | +0.894 | 2.06 | 0.00 | True |
| B_t3p0_pose | 3.00 | withdraw | 500.0 | -0.779 | 1.57 | 0.00 | True |
| B_t3p0_pose | 3.00 | in | 500.0 | +0.801 | 2.22 | 0.00 | True |
| B_t3p0_pose | 3.00 | withdraw | 500.0 | -0.713 | 1.69 | 0.00 | True |
| B_t3p0_pose | 3.00 | in | 500.0 | +0.716 | 2.63 | 0.00 | True |
| B_t3p0 | 3.00 | in | 150.0 | -0.368 | 0.96 | 0.00 | True |
| B_t3p0 | 3.00 | withdraw | 150.0 | -0.678 | 1.97 | 0.00 | True |
| B_t3p0 | 3.00 | in | 150.0 | -0.389 | 0.91 | 0.00 | True |
| B_t3p0 | 3.00 | withdraw | 150.0 | -0.621 | 1.92 | 0.00 | True |
| B_t3p0 | 3.00 | in | 150.0 | -0.323 | 0.32 | 0.00 | True |
| B_t3p0 | 3.00 | withdraw | 150.0 | -0.533 | 2.50 | 0.00 | True |
| B_t3p0 | 3.00 | in | 150.0 | -0.285 | 0.69 | 0.00 | True |
| B_t3p0 | 3.00 | withdraw | 150.0 | -0.480 | 3.74 | 0.00 | True |
| B_t3p0 | 3.00 | in | 150.0 | -0.382 | 0.14 | 0.00 | True |
| B_t3p0 | 3.00 | withdraw | 150.0 | -0.494 | 2.71 | 0.00 | True |
| B_t4p0 | 4.00 | in | 150.0 | -0.499 | 0.09 | 0.00 | True |
| B_t4p0 | 4.00 | withdraw | 150.0 | -0.750 | 1.51 | 0.00 | True |
| B_t4p0 | 4.00 | in | 150.0 | -0.286 | 0.84 | 0.00 | True |
| B_t4p0 | 4.00 | withdraw | 150.0 | -0.600 | 2.33 | 0.00 | True |
| B_t4p0 | 4.00 | in | 150.0 | -0.226 | 0.66 | 0.00 | True |
| B_t4p0 | 4.00 | withdraw | 150.0 | -0.587 | 2.77 | 0.00 | True |
| B_t4p0 | 4.00 | in | 150.0 | +0.265 | 2.28 | 0.00 | True |
| B_t4p0 | 4.00 | withdraw | 150.0 | -0.432 | 2.97 | 0.00 | True |
| B_t4p0 | 4.00 | in | 150.0 | -0.353 | 0.22 | 0.00 | True |
| B_t4p0 | 4.00 | withdraw | 150.0 | -0.447 | 2.60 | 0.00 | True |
| B_t4p0 | 4.00 | in | 150.0 | -0.385 | 0.13 | 0.00 | True |
| B_t4p0 | 4.00 | withdraw | 150.0 | -0.517 | 3.01 | 0.00 | True |


## Regulation evidence — did the robot actually do it?

"힘이 안 올랐다" 는 두 가지를 못 가른다: **로봇이 흡수했는가**, 아니면
**애초에 오를 상황이 아니었는가.** 물러난 거리가 그 둘을 가른다 — 힘이
일정한데 팔이 뒤로 갔다면, 그 변위가 로봇이 받아 낸 양이다.

**팬텀 강성 k = -56.465 N/mm** (개루프 실행에서 측정).

| run | points | k_n_per_mm | r_squared | force_span_n | travel_span_mm | reason |
|---|---|---|---|---|---|---|
| S_stiff | 4 | -56.465 | 0.8325 | 0.21 | 0.00 | — |


`counterfactual_n` = 유지한 힘 + (물러난 거리 × k). **팔이 가만히 있었다면
실렸을 힘** 이며, 이것이 안전 한계와 비교할 값이다.

| run | target_n | direction | travel_mm | held_mean_n | held_max_n | absorbed_n | counterfactual_n |
|---|---|---|---|---|---|---|---|
| B_t3p0_pose | 3.00 | in | -0.55 | 3.02 | 3.69 | -31.20 | -28.18 |
| B_t3p0_pose | 3.00 | withdraw | 1.60 | 2.49 | 2.70 | 90.22 | 92.72 |
| B_t3p0_pose | 3.00 | in | -0.79 | 3.14 | 3.92 | -44.39 | -41.25 |
| B_t3p0_pose | 3.00 | withdraw | 1.22 | 2.56 | 2.87 | 69.06 | 71.62 |
| B_t3p0_pose | 3.00 | in | -0.80 | 3.12 | 3.89 | -45.30 | -42.18 |
| B_t3p0_pose | 3.00 | withdraw | 1.03 | 2.63 | 2.84 | 57.90 | 60.53 |
| B_t3p0_pose | 3.00 | in | -0.84 | 3.24 | 3.80 | -47.15 | -43.91 |
| B_t3p0_pose | 3.00 | withdraw | 0.86 | 2.59 | 3.01 | 48.58 | 51.17 |
| B_t3p0_pose | 3.00 | in | -0.63 | 3.10 | 3.72 | -35.35 | -32.25 |


⚠️ 강성은 개루프 실행의 **한 지점 기울기**다. 팬텀이 비선형이면 외삽한 만큼
틀리고, 누른 자리가 다르면 값도 다르다. `r_squared` 와 `force_span_n` 이
그 외삽이 얼마나 먼지를 말해 준다 — 측정 구간 밖으로 크게 벗어난 반사실은
근거가 아니라 추정이다.

## Safety margin

**전 구간을 본다** — 접근 중에 넘는 것도 넘는 것이다.

| run | run_type | target_n | peak_force_n | warn_force_n | max_force_n | margin_to_limit_n | reached_warn | exceeded_limit | samples_over_limit |
|---|---|---|---|---|---|---|---|---|---|
| A_t0p5 | hold | 0.50 | 0.64 | 4.50 | 5.00 | 4.36 | False | False | 0 |
| A_t1p0 | hold | 1.00 | 1.31 | 4.50 | 5.00 | 3.69 | False | False | 0 |
| A_t1p5 | hold | 1.50 | 1.49 | 4.50 | 5.00 | 3.51 | False | False | 0 |
| A_t2p0 | hold | 2.00 | 2.03 | 4.50 | 5.00 | 2.97 | False | False | 0 |
| A_t3p0 | hold | 3.00 | 2.73 | 4.50 | 5.00 | 2.27 | False | False | 0 |
| A_t4p0 | hold | 4.00 | 3.74 | 4.50 | 5.00 | 1.26 | False | False | 0 |
| B_t0p5 | disturbance | 0.50 | 0.74 | 4.50 | 5.00 | 4.26 | False | False | 0 |
| B_t1p0 | disturbance | 1.00 | 1.33 | 4.50 | 5.00 | 3.67 | False | False | 0 |
| B_t1p5 | disturbance | 1.50 | 1.49 | 4.50 | 5.00 | 3.51 | False | False | 0 |
| B_t2p0 | disturbance | 2.00 | 2.11 | 4.50 | 5.00 | 2.89 | False | False | 0 |
| B_t3p0_pose | disturbance | 3.00 | 3.92 | 4.50 | 5.00 | 1.08 | False | False | 0 |
| B_t3p0 | disturbance | 3.00 | 3.16 | 4.50 | 5.00 | 1.84 | False | False | 0 |
| B_t4p0 | disturbance | 4.00 | 4.26 | 4.50 | 5.00 | 0.74 | False | False | 0 |
| C_limit | safety | 3.00 | 4.41 | 4.50 | 5.00 | 0.59 | False | False | 0 |
| S_stiff | stiffness | 3.00 | 0.39 | 4.50 | 5.00 | 4.61 | False | False | 0 |


## Verdict

이 실행들에서 한계를 넘은 표본은 없다. **한계가 도달 가능했는지**는 별개 질문이며, 아래 최악 힘과 한계의 거리로 판단할 것 — 거리가 크면 시험한 것은 한계가 아니라 그 아래 대역이다.

## Figures

- `fig1_force_traces.png`
- `fig1_force_traces.pdf`
- `fig2_hold_quality.png`
- `fig2_hold_quality.pdf`
- `fig3_disturbance.png`
- `fig3_disturbance.pdf`
- `fig4_safety_margin.png`
- `fig4_safety_margin.pdf`
- `fig5_regulation_evidence.png`
- `fig5_regulation_evidence.pdf`
- `fig6_excursion_recovery.png`
- `fig6_excursion_recovery.pdf`
