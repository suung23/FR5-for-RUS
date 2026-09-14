# main_rule — 결과

에피소드 15 · 자세 5 · 조건 hold, rule, search

## 조건별

| 조건 | 성공 | 찾음 | 찾기 [s] | 최장 연속 [s] | hold_q_mean | hold_good_frac | F_max [N] | >4.5 N [s] | >5 N [s] |
|---|---|---|---|---|---|---|---|---|---|
| hold | 1/5 | 1/5 | 35.4 | 0.0 | 0.188 | 0.331 | 4.77 | 5.6 | 0.0 |
| rule | 5/5 | 5/5 | 13.0 | 75.7 | 0.752 | 0.937 | 5.07 | 13.8 | 1.2 |
| search | 4/5 | 4/5 | 20.2 | 68.2 | 0.763 | 0.888 | 5.15 | 14.6 | 0.8 |

## 자세별 (짝지은 비교)

| 자세 | hold | rule | search |
|---|---|---|---|
| 1 | 실패 0.0 s · q — · ep0003_pose01_hold | 성공 75.7 s · q 0.801 · ep0001_pose01_rule | 성공 84.0 s · q 0.776 · ep0002_pose01_search |
| 2 | 성공 6.5 s · q 0.188 · ep0005_pose02_hold | 성공 97.6 s · q 0.752 · ep0006_pose02_rule | 성공 89.6 s · q 0.750 · ep0004_pose02_search |
| 3 | 실패 0.0 s · q — · ep0008_pose03_hold | 성공 71.0 s · q 0.595 · ep0007_pose03_rule | 실패 0.0 s · q — · ep0009_pose03_search |
| 4 | 실패 0.0 s · q — · ep0012_pose04_hold | 성공 115.1 s · q 0.815 · ep0010_pose04_rule | 성공 68.2 s · q 0.855 · ep0011_pose04_search |
| 5 | 실패 0.0 s · q — · ep0015_pose05_hold | 성공 46.8 s · q 0.569 · ep0013_pose05_rule | 성공 57.9 s · q 0.674 · ep0014_pose05_search |

**rule vs hold** — 짝 5 · rule 만 성공 4 · hold 만 성공 0 · 정확 이항검정 단측 p = 0.0625

> ⚠️ 짝 5 개로 도달 가능한 최소 p 는 0.0312 다. 

**search vs hold** — 짝 5 · search 만 성공 3 · hold 만 성공 0 · 정확 이항검정 단측 p = 0.1250

> ⚠️ 짝 5 개로 도달 가능한 최소 p 는 0.0312 다. 
