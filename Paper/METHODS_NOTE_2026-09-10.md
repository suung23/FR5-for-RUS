# 로봇 초음파 방광 탐색 — 방법 노트 (2026-09-10)

논문 작성용 정리. 모든 수치는 이 저장소에서 실제로 계산된 값이고, 각 절 끝에 **재현 명령**을 적었다.
⏳ 는 아직 실측/검증되지 않은 항목이다 — 논문에 단정으로 쓰면 안 되는 것들.

---

## 1. 시스템과 데이터

### 1.1 하드웨어

| | |
|---|---|
| 프로브 | Konted C10UR, Wi-Fi (프로브 AP 192.168.1.1, TCP 5002/5003) |
| 프레임 | 160 A-line × 512 깊이 표본, uint8, 81,920 B/frame, **9.98–10.4 fps** |
| 전달 형태 | **scan conversion 이전 극좌표** (행 = A-line, 행 시작 = 근거리) |
| IMU | BNO085 (XIAO nRF52840 Sense), 250–256 Hz, 회전벡터(칩 융합) + 가속도 + 자이로 |
| 공통 시계 | `pc_unix` — 수집 호스트의 `time.time()`, 두 스트림이 수신 시각으로 공유 |

프레임 배치는 처음 320 × 256 으로 잘못 기록돼 있었다. 바이트 수가 같아 오류 없이 읽히지만 A-line 하나가
두 행(얕은/깊은 절반)으로 쪼개진다. 판정 근거: **인접 라인 상관** — 320 × 256 에서 0.1–0.3, 160 × 512 에서
**0.93–0.96** (16 세션). 원시 바이트는 건드리지 않고 해석만 고쳤다.

### 1.2 수집 프로토콜

**1일차 (2026-09-09) — 연속 스캔, 20 세션 / 19,708 프레임.**
프로브를 팬텀에 대고 자유롭게 움직이며 각 세션 ~1,000 프레임(100 s). 세션 단위로 train/val/test.

**2일차 (2026-09-10) — 에피소드 모드 `find_bladder`, 248 에피소드 / 26,194 프레임.**
에피소드 하나 = 세션 하나. **방광 밖에서 시작 → 정지-이동-정지로 탐색 → 방광이 보이면 1–2 s 정지 → 저장.**
실패한 시행은 `X` 로 폐기(`discard_` 접두어, 최종 0 건). 길이 11–344 프레임(중앙 96, 6–34 s).

운동 종류는 세션 메타 `episode.motion` 에 기록된다:

| 구간 | 종류 | n |
|---|---|---|
| ep 1–79 | 병진 탐색 (translation) | 79 |
| ep 80–100 | **force_only** — 자세 고정, 접촉 힘만 변화 | 21 |
| ep 101–249 | `lateral_x` 횡이동만 | 69 |
| | `rotation_z` 회전만 | 40 |
| | `combined_mixed` 복합 | 20 |
| | `sweep_y` 면외 스윕 | 19 |

전체 275 세션 중 **268 세션을 사용**한다. 제외 7: 8월 벤치 5 (다른 프로브), `20260909_153452`
(그날 첫 세션, 자력계 미수렴 — host(Madgwick)–chip(RV) 격차 중앙 43.8° / p95 72.6°, 분당 +7.4° 발산;
나머지 세션은 격차 ~1.1°), `20260909_151006` (수집 중단으로 메타 파손).

```bash
python imu_bench/host/push_sessions.py          # 노트북 → 학습 PC (LAN, scp)
```

---

## 2. 시각 정렬

### 2.1 US 고정 지연

US 프레임은 IMU 보다 늦게 도착한다. 지연 τ_us 는 **영상 변화량과 각속도의 상호상관**으로 잰다.
프레임 t_i 의 영상 변화량과, 같은 구간을 덮는 자이로 크기의 박스 적분:

$$
\Delta I_i=\frac{1}{N}\sum_{p}\bigl|I_i(p)-I_{i-1}(p)\bigr|,
\qquad
\Omega_i(\tau)=\int_{t_i-\tau-\Delta}^{t_i-\tau}\lVert\boldsymbol\omega(s)\rVert\,ds
$$

$$
\hat\tau_{us}=\arg\max_{\tau}\ \operatorname{corr}\bigl(\Delta I_i,\ \Omega_i(\tau)\bigr)
$$

**실측 (268 세션 풀링): τ_us = +200 ms**, r = 0.817, 세션 부트스트랩 95 % CI [+190, +200],
세션별 중앙 +200 / IQR [190, 215]. 주입 테스트(−100 / +100 / +200 ms 인위 지연) 오차 0 ms.
영상 특징 4 종(전체 / 중간깊이 / 근거리 / A-line)에서 +190–200 ms 로 일치.
**장비의 frame persistence 를 포함한 유효 지연**이며 순수 전송 지연 스펙이 아니다. 실효 불확실도 ±20 ms.

적용은 한 번뿐이다: `us_index.csv` 의 원시 수신 시각에서 학습 파이프라인이 뺀다
(`dataset.py`: `frames_t = session.frame_t_pc − cfg.timing.us_latency_s`).
`sync.npz` 는 `frame_t_pc`(원시)와 `frame_t_acq`(보정)를 **따로** 들고 있어 중복 보정이 없다.

### 2.2 센서 → 프로브 회전 R_SP

IMU 는 프로브에 임의 자세로 붙어 있다. 프로브 좌표계는 **x = lateral, y = elevational, z = beam**.
전용 세션에서 프로브 x, y, z 축 둘레로 차례로 크게 회전시키고, 각 회전 구간의 자이로 표본
$W\in\mathbb R^{n\times3}$ 의 **주축**(SVD 첫 우특이벡터)을 R_SP 의 행으로 삼는다:

$$
W=U\Sigma V^\top,\qquad \mathbf u = V_{:,1}\cdot \operatorname{sign}\!\bigl(V_{:,1}^\top\textstyle\sum_i \boldsymbol\omega_i\bigr),
\qquad
R_{SP}=\operatorname{orth}\bigl[\mathbf u_x\ \mathbf u_y\ \mathbf u_z\bigr]^\top
$$

**실측 (`us_imu_20260910_140834`)**: 회전 96° / 83° / 115°,
주축 사이 각도 **93.2 / 92.7 / 97.7°** (직교 기준 90°), 투영 잔차 **4.20°**, det **+1.000**,
중력 검산 $R_{SP}\,\mathbf a_S=(+0.21,+0.68,-9.71)\ \mathrm{m/s^2}$ (빔축 아래로 둔 정지에서 $(0,0,-g)$ 기대).

$$
R_{SP}=\begin{bmatrix}0&-1&0\\0&0&-1\\1&0&0\end{bmatrix}
\qquad(x_p=-y_s,\quad y_p=-z_s,\quad z_p=+x_s)
$$

⚠️ **부호 미해결.** 축 배정(어느 센서축이 어느 프로브축인지)은 비대각 성분 |0.024| 이하로 확정적이지만,
x·y 의 **부호**는 별도 증거(2 일차 횡이동 69 세션에서 출발 가속 부호 ↔ 영상 라인 이동 부호 146/146 일치)와
반대다. 둘 다 det +1 이라 마운트 차이가 아니라 회전 방향 규약의 차이다. **정책의 lateral/elevational 부호가
뒤집혀 보이면 이 두 행의 부호를 먼저 뒤집을 것.**

```bash
python policy_learning/scripts/calibrate_sensor_to_probe.py <session> --gravity-check \
    --set imu.still_gyro_sd=0.05 --set imu.still_gyro_mean=0.08
```

---

## 3. 영상 전처리 — 극좌표 → 부채꼴 B-mode

가상 apex 기준 극좌표 $(r,\varphi)$ 를 직교 격자로 되돌린다. 기하는 벤더 뷰어 화면에서 실측:
곡률 반경 **R = 59 mm**, 반각 **α = 28°**, 깊이 **D = 220 mm**.

출력 격자 $(x,y)$ 는 apex 원점, 세로 범위 $[R\cos\alpha,\ R+D]$, 가로 반폭 $(R+D)\sin\alpha$:

$$
r=\sqrt{x^2+y^2},\qquad \varphi=\operatorname{atan2}(x,y)
$$

$$
j = \frac{r-R}{D}\,(N_s-1),\qquad
i = \frac{\varphi+\alpha}{2\alpha}\,(N_l-1),\qquad
\text{유효역}:\ R\le r\le R+D\ \wedge\ |\varphi|\le\alpha
$$

$N_l=160$, $N_s=512$. 픽셀 크기 **0.4432 mm/px**, 출력 512 × 591 → **등방 축소 + 검정 패딩으로 256²**.
2× supersampling 후 면적 평균(모아레 억제). ⏳ 라인 좌우 순서(`flip_lines`)는 미검증.

```python
# policy_learning/rus_policy/bmode.py — 학습·실행 모두 같은 변환을 쓴다
BmodeConverter(session_meta, (160, 512), out_size=256)
```

---

## 4. A-line 접촉 측정 (본 연구의 전처리 기여)

프리핸드 수집에서는 프로브가 조직에 **부분적으로만** 닿는다. 어느 각도가 실제로 결합됐는지는
부채꼴 이미지의 밝기로 판정할 수 없다 — **근거리는 트랜스듀서 링다운이라 공기 중에도 밝다**
(실측: 공기 89, 접촉 89, 깊이 표본 8–48 평균). 갈리는 곳은 중간 깊이다 (공기 6, 접촉 55).

A-line $k$ 가 결합됐는지는 중간 깊이 $\mathcal M=\{48,\dots,159\}$ 의 평균으로 판정하고,
프레임의 **접촉 비율** $\rho$ 는 결합된 라인의 비율이다:

$$
c_k=\frac{1}{|\mathcal M|}\sum_{j\in\mathcal M} I(k,j),
\qquad
\rho=\frac{1}{N_l}\sum_{k=1}^{N_l}\mathbb 1\!\left[c_k>T\right],\qquad T=20
$$

**분포 (45,902 프레임)**

| | 완전 ρ ≥ 0.90 | 부분 0.10 ≤ ρ < 0.90 | 비접촉 ρ < 0.10 |
|---|---|---|---|
| 1일차 (19,708) | 70.0 % | 27.1 % | 2.9 % |
| 에피소드 (26,194) | 43.8 % | 54.4 % | 1.8 % |

이 지표는 **분할 정확도를 단조롭게 예측한다**(§5.4). 탐색 과제(2일차)가 연속 스캔(1일차)보다 부분 접촉이
두 배 많은 것이 두 도메인의 성능 차이를 설명한다.

```bash
python Unet_seg/scripts/phantom_training_sets.py contact   # → contact_alines.npz
```

---

## 5. 방광 분할 — PFUS 사전학습 + 6 라운드 능동학습

### 5.1 사전학습과 파인튜닝

공개 **PFUS** 골반저 데이터셋으로 학습한 Standard U-Net (`pfus_bladder_edit_man`, 손보정 라벨, val
selection 0.755) 에서 **모델 텐서만** 가져와(`--init-weights`, optimizer·스케줄은 초기화) 팬텀에 파인튜닝.
입력 256², per-image 정규화, AdamW lr **2e-4** (from-scratch 의 1/5 — PFUS 특징 보존), cosine, 30–40 epoch,
batch 8. 증강: 회전 ±5°, 스케일 0.92–1.08, 이동 4 %, 밝기·대비 0.15, 감마 0.8–1.25, 가우시안 0.01,
스페클 0.02. **좌우/상하 반전은 쓰지 않는다** (라인 순서 미검증, 해부학적 좌우가 고정).

후처리: 임계 0.5 → 최대 연결요소 → 구멍 채움 (`min_component_area_ratio` 5e-4).

### 5.2 능동학습 루프 — 무엇을 라벨할지 고르는 법

라벨 예산이 작으므로 **오류와 상관되는 지표**로 큐를 정렬해야 한다. 사람이 그린 프레임의 실제 Dice 와
각 후보 지표의 상관(피어슨 r):

| 지표 | r(지표, Dice) | 판정 |
|---|---|---|
| **이웃 프레임 마스크 IoU** $\mathrm{IoU}_{nb}$ | **+0.694** | 채택 |
| **mean boundary entropy** $H_b$ | **−0.636** | 채택 |
| `segmentation_confidence` = $\overline{|2p-1|}$ | +0.507 | **기각** |
| `control_quality_score` | +0.117 | 기각 |

`segmentation_confidence` 는 **빈 예측에서 1.0 이 되는 퇴화 지표**다 — 상위 순위가 전부 프로브가 공기 중인
프레임이었다. 논문에 쓸 만한 관찰이다: *threshold-free confidence 는 abstention 이 가능한 문제에서
ranking 지표로 쓰면 안 된다.*

채택한 위험도:

$$
\mathrm{risk}=z(H_b)+z\bigl(1-\mathrm{IoU}_{nb}\bigr),
\qquad
\mathrm{IoU}_{nb}(i)=\frac{1}{|\mathcal N|}\sum_{j\in\{i-1,i+1\}}\frac{|M_i\cap M_j|}{|M_i\cup M_j|}
$$

라운드별 큐 설계:

| 라운드 | n | 표집 | 의도 |
|---|---|---|---|
| 1 | 102 | 시간상 균등, 정지 우선 | 초기 라벨 |
| 2 | 30 | risk 상위, 저녁 세션 | 라벨 없는 도메인 |
| 3 | 32 | 접촉 밴드 0.30–0.85 층화 | 부분 접촉 경계 |
| 감사 | 20 | **무작위** (순위 없음) | 편향 없는 성능 추정 |
| 5 | 40 | 접촉 밴드 층화 무작위 | 새 도메인 학습 + 홀드아웃 |
| 6 | 30 | 오탐 의심 20 + 무작위 10 | 오탐 제거 시도 |

**총 사람 라벨 234 장** (1일차 144 + 에피소드 90; 제외 세션의 15 장은 미사용).

### 5.3 자가학습과 그 함정 (재현 가능한 음성 결과)

완전 접촉 프레임의 고신뢰 의사라벨 231 장을 학습에 추가하자 Dice 는 올랐지만 **모델이 "없다"고 말하는
능력을 잃었다**. 비접촉 666 프레임에 마스크를 씌운 횟수:

| | 빈 마스크 | 비접촉 프레임 오탐 |
|---|---|---|
| 사람 라벨만 (117) | 366 | 315 |
| + 의사라벨 231 (양성만) | 45 | **631** |
| + 공기 프레임 음성 99 | 712 | **35** |

원인은 선택 편향이다 — 추가한 라벨이 전부 "마스크가 있는" 프레임이라 사전확률이 무너진다.
**의사라벨에 음성 표본을 함께 넣어야 한다**는 것이 결론이고, 이후 라운드는 모두 그렇게 했다.
또한 양성 의사라벨은 **완전 접촉 프레임에서만** 뽑는다 — 부분 접촉의 판단은 사람 라벨로만 배운다.

### 5.4 물리 기반 게이팅

모델 출력을 그대로 쓰지 않고, **측정이 존재하지 않는 곳의 예측을 지운다**. 세 규칙 모두 사람 라벨과
모순되지 않는 선에서만 적용:

1. **비접촉** ρ < 0.10 → 빈 마스크 (측정 자체가 없음)
2. **결합 영역 밖** — 마스크의 절반 이상이 결합되지 않은 A-line 각도에 놓이면 빈 마스크.
   부채꼴 픽셀 → A-line 인덱스 지도를 한 번 만들고 프레임별 결합 집합을 투영해 판정.
3. **조각** — 면적이 사람이 그린 최소 마스크(0.0101)보다 작으면(임계 0.008) 빈 마스크

적용량 (최종 모델):

| | 프레임 | 유지 | 비접촉 | 결합영역 밖 | 조각 | 최종 빈 마스크 |
|---|---|---|---|---|---|---|
| 1일차 | 19,708 | 18,914 | 540 | 0 | 110 | 1,061 (5.4 %) |
| 에피소드 | 26,194 | 23,352 | 428 | 28 | 2,386 | 6,936 (26.5 %) |

에피소드의 26.5 % 는 **무작위 감사가 말하는 실제 "방광 없음" 비율 30 %** 와 대체로 일치한다.

### 5.5 성능

지표: $\mathrm{Dice}=\dfrac{2|G\cap M|}{|G|+|M|}$, $\mathrm{IoU}=\dfrac{|G\cap M|}{|G\cup M|}$,
재현율 $\dfrac{|G\cap M|}{|G|}$, 정밀도 $\dfrac{|G\cap M|}{|M|}$. 두 마스크가 모두 비면 Dice = 1 로 둔다.

**고정 홀드아웃 49 장** (1일차 29 + 에피소드 20, 어느 라운드에서도 학습에 쓰지 않음):

| | round 3 | round 5 | **round 6 (최종)** |
|---|---|---|---|
| 1일차 (n=29) | 0.811 | 0.838 | **0.842** |
| 에피소드 (n=20) | 0.712 | 0.743 | **0.777** |
| 합계 (n=49) | 0.771 | 0.800 | **0.815** |
| 방광이 있는 프레임 (n=40) | 0.844 | 0.854 | **0.874** |
| 방광이 없는 프레임 오탐 (n=9) | 5/9 | 4/9 | 4/9 |

**방광이 있는 40 장만** (최종 모델): Dice **0.874** (중앙 0.896), IoU 0.787, 재현율 0.861, 정밀도 0.903,
**놓친 프레임 0**, 예측/사람 면적비 **0.99** (과대·과소 어느 쪽으로도 치우침 없음).

접촉 밴드별 (있는 프레임): 완전 ρ≥0.90 **0.899**, 부분 0.70–0.90 **0.820**, 부분 <0.70 **0.838**.
도메인별: 1일차 **0.904**, 에피소드 **0.810**.

**편향 없는 감사 (무작위 표집, 라벨 직전 모델을 채점):**

| 표본 | 모델 | Dice | 방광 없음 비율 | 오탐 |
|---|---|---|---|---|
| 20 장 | round 3 | 0.670 | 6/20 | 4/6 |
| 40 장 | round 5 | 0.682 | 12/40 | 6/12 |

무작위 60 장을 합치면 **방광 있음 42 장 Dice 0.778 / 놓침 0**, **방광 없음 18 장 중 10 장 오탐(56 %)**.
즉 **잔여 오차는 경계가 아니라 유무 판단에 거의 전부 몰려 있다.**

⚠️ 임계값으로는 안 고쳐진다 — 0.5–0.9 전 구간에서 오탐 4/9 로 동일했다. 확률이 포화된 **확신에 찬 오류**다.
TTA(회전±5°·스케일±5 %·감마, 7 배) 는 +0.003 Dice 에 추론 3.1 배로 기각. 시간 다수결 필터도 이득 0.

### 5.6 ROI

`segmentation_confidence` 와 `border_contact_ratio` 는 프레임 전체를 분모로 쓰면 무의미해진다
(스캔 변환이 쓰지 않는 검은 모서리에서 어떤 모델이든 확신한다). 그래서 **부채꼴 유효역을 데이터에서 실측**:
train split 세션별로 프레임 최대투영을 임계하고 과반 합의. 결과 **프레임의 53.6 %**, 세션별 IoU 최소 0.96 /
중앙 0.998 — 하나의 ROI 가 전 세션에 유효하다.

```bash
python Unet_seg/scripts/train.py --config configs/phantom_c10ur_finetune.yaml \
    --manifest data/manifest_r6.csv --init-weights checkpoints/pfus_bladder_edit_man/best.pt \
    --output-dir checkpoints/phantom_c10ur_r6 --epochs 40
python Unet_seg/scripts/phantom_training_sets.py export --run runs/phantom_c10ur_r6
```

---

## 6. 정책 학습용 데이터셋

### 6.1 정지-이동-정지 분할

프리핸드 궤적을 **정지로 끊어** 구간을 만든다. 창 $W$ = 0.30 s 안에서

$$
\text{still}(t)=\Bigl[\operatorname{sd}\lVert\boldsymbol\omega\rVert<0.02\Bigr]\wedge
\Bigl[\overline{\lVert\boldsymbol\omega\rVert}<0.03\Bigr]\wedge
\Bigl[\operatorname{sd}\lVert\mathbf a\rVert<0.25\Bigr]
$$

(rad/s, m/s²). 연속한 두 정지 사이가 이동 구간이며 **0.15 s ≤ 길이 ≤ 8.0 s** 만 채택.
검출 마스크 길이 기준은 $\min\_{still} - W$ (이동창이 정지 양 끝을 $W/2$ 씩 깎으므로).

**실측 (에피소드 50 세션 표본)**: 정지 비율 중앙 **54.2 %**, 정지-정지 간격 중 **82 % 채택**,
8 s 초과로 버린 것 **0 %**, 0.15 s 미만 18 % (회전량 0°, 손떨림이라 잃는 정보 없음).

### 6.2 라벨 — ZUPT 로 묶은 SE(2) 변위

각 이동 구간의 앵커(정지 A) 시점 프로브 프레임에서, 중력을 뺀 선형가속도를 두 번 적분하되
**끝 속도 0** 조건으로 선형 디드리프트(ZUPT)한다:

$$
\mathbf v(t)=\int_{t_0}^{t}\mathbf a_{lin},\qquad
\tilde{\mathbf v}(t)=\mathbf v(t)-\mathbf v(t_1)\frac{t-t_0}{t_1-t_0},\qquad
\mathbf p(t)=\int_{t_0}^{t}\tilde{\mathbf v}
$$

회전은 회전벡터에서 직접 얻는다. 라벨 $P\in\mathbb R^{(k+1)\times3}$ 는 정책 축
$(x\ \mathrm{mm},\ y\ \mathrm{mm},\ \theta_z\ \mathrm{deg})$ 의 chunk 격자값이고 $P[0]=0$.
6 자유도 원본 $P_6\in\mathbb R^{(k+1)\times6}$ 도 함께 저장한다.

라벨 불확실도는 적분 구간 길이 $\tau$ 에 대해

$$
\sigma_t = c_t\,\tau^{1.5}\ \ (c_t=0.7\ \mathrm{mm}),\qquad
\sigma_\theta = 0.1^\circ,\qquad
\sigma_{shape}=\max\bigl(0.3\,\lVert\text{net}\rVert,\ \sigma_t\bigr)
$$

$\tau$ 는 chunk 길이 $k/f_p$ 로 상한을 둔다. 🟡 계수는 제안값이며 튜닝 대상.

### 6.3 관측

$$
m=\Bigl\lceil f_{us}/f_{dither}\Bigr\rceil=\lceil 9.98/0.5\rceil=\mathbf{20}
$$

관측 프레임 20 장(+ 각 프레임의 21 차원 지각 상태) + 15 차원 상태 벡터.
구간당 관측 시각을 3 개(앵커 + 정지 A 안 2 개) 뽑아 증강한다 → 샘플 수 ≈ 2.41 × 구간 수.

**지각 상태 21 차원** (전부 U-Net 마스크에서 계산):
`has_mask, valid_for_control, centroid_dx, centroid_dy, area_ratio, major/minor_axis_norm,
orient_cos2/sin2, segmentation_confidence, lumen_contrast, contrast_valid, border_contact_ratio,
largest_component_ratio, boundary_entropy, quality, …`

**상태 벡터 15 차원**: `prev_dx_mm, prev_dy_mm, prev_dtheta_deg, prev_valid, prev_gap_s,
gravity_px/py/pz, F_n, M_x, M_y, F_n_star, barrier_b, delay_w, has_force`.
프리핸드 수집이라 **힘 채널은 전부 무효**(`F_valid` 0 %) — F/T 센서는 로봇 경로에서만 붙는다.

### 6.4 최종 데이터셋

```
policy_dataset.h5   5.4 GiB
  obs/frames        (4389, 20, 256, 256) uint8
  obs/state         (4389, 20, 21)  float32
  obs/vec           (4389, 15)      float32
  obs/frame_dt, frame_valid
  label/P (4389, 9, 3)   P6 (4389, 9, 6)   Q, Q_area, F (+ *_valid)
  label/sigma_net, sigma_shape
  meta/session, split, subject, source, json
```

| | |
|---|---|
| 세션 | 268 (1일차 20 + 에피소드 248) |
| 프레임 | 45,902 → 이동 구간 **1,818** → **샘플 4,389** |
| split | train 3,297 / val 576 / test 516 (**세션 단위**) |
| 관측 유효율 | 1일차 99.8 % / 에피소드 84.9 % |
| 마스크 있는 관측 | 1일차 98.1 % / 에피소드 74.5 % |

**split 의 의미**: 팬텀이 하나라 피험자가 하나다 (`allow_subject_overlap`). val/test 는
**일반화가 아니라 같은 팬텀에서의 재현성**을 잰다. 논문에서 이 점을 흐리면 안 된다.

```bash
python policy_learning/scripts/build_dataset.py --sessions data/sessions.csv --out data/policy_dataset.h5
```

---

## 7. 정책 학습 (ACT / CVAE)

5 Hz 로 $k=8$ 스텝(1.6 s) **행동 chunk** 를 예측한다. 조건부 VAE 구조로, 잠재 $z$ (차원 4) 는
디코더 쿼리에만 들어가 **누설 상한**이 되도록 설계했다.

$$
\mathcal L=\underbrace{\sum_{\text{axes}}\rho_\delta\!\left(\frac{\hat P-P}{\sigma}\right)}_{\text{Huber, }\sigma\text{ 로 정규화}}
+\beta\,D_{KL}\bigl(q(z\mid P,o)\,\Vert\,\mathcal N(0,I)\bigr)
+\lambda_Q\mathcal L_{quality}+\lambda_F\mathcal L_{force}+\lambda_s\mathcal L_{smooth}
+\lambda_{feas}\mathcal L_{feas}+\lambda_r\mathcal L_{risk}
$$

| 항 | 값 | | 학습 | 값 |
|---|---|---|---|---|
| $\beta_{KL}$ | 0.5 (워밍업 5 epoch) | | epochs | 50 |
| $\lambda_{quality}$ | 1.0 | | batch | 16 |
| $\lambda_{force}$ | 0.3 | | lr / wd | 1e-4 / 1e-4 |
| $\lambda_{smooth}$ | 0.05 | | warmup | 2 epoch |
| $\lambda_{feas}$ | 0.5 | | grad clip | 1.0 |
| $\lambda_{risk}$ | 0.2 | | $z$ 평가 표본 | 32 |
| $w_{-}/w_{+}$ | 5.0 | | 모델 | d 128, head 8, enc 2 / dec 2, ff 512 |
| Huber $\delta$ | 2σ | | 영상 인코더 | [32,64,128,256], 입력 128² |

**보고할 지표**: `val/mae_x_mm`, `val/mae_y_mm`, `val/mae_th_deg`,
`val/vy_sign_acc` (elevational 부호 정확도 — 0.5 근처면 §2.2 의 R_SP 부호 문제),
`val/mode_collapse_vy_std` (0 에 붙으면 모드 붕괴 → $\beta_{KL}$ 스윕),
`val/mode_margin`, `val/select_vy_sign_acc`.

학습에는 U-Net 이 필요 없다 — 지각 특징이 HDF5 에 이미 구워져 있다.

```bash
python policy_learning/scripts/train_policy.py --dataset policy_dataset.h5 --out runs/exp1
python policy_learning/scripts/eval_policy.py runs/exp1/best.pt --dataset policy_dataset.h5 --split test
```

---

## 8. 에피소드 분석 (탐색 과제 자체의 특성)

248 에피소드를 최종 모델로 마스킹한 결과:

- **240/248 에피소드**에서 방광이 3 프레임 이상 연속 검출 (검출률 70 %, 접촉 중앙 0.91–0.93)
- **수렴**: 앞 1/3 구간 면적 중앙 **0.0000** → 뒤 1/3 **0.031**, 248 개 중 **233 개**가 증가.
  즉 탐색 초반에는 방광이 시야에 없다가 후반에 잡힌다 — 과제 설계대로다.
- **좌우 정렬**: 이동군 마지막 1/4 구간의 좌우 중심 오차 중앙 **|0.018|** (프레임 폭 기준) — 거의 중앙.
- ⚠️ **세로는 구조적으로 못 맞춘다**: 세로 중심 오차가 이동·힘 양쪽 모두 중앙 **−0.27**.
  깊이 220 mm 설정에 표적이 근거리라 프로브를 어떻게 움직여도 화면 중앙에 오지 않는다.
  **정책의 서보 목표를 화면 좌표 중앙으로 두면 도달 불가능한 목표**가 된다 — 깊이 설정을 줄이거나
  목표를 프로브 면으로부터의 mm 로 정의해야 한다. (논문의 limitation / future work 후보)
- **force_only 21 에피소드**: 접촉 비율과 마스크 면적의 에피소드 내 상관 중앙 **r = 0.89** (20/21 양수),
  깊이와는 +0.57. 즉 **누를수록 방광이 커지고 깊어진다** — 조직 압박이라면 반대여야 한다.
  세게 누를수록 결합되는 A-line 이 늘어 더 넓고 깊은 부분이 보이는 **시야 효과**이지 변형이 아니다.
  ⚠️ **영상에서 접촉력을 추정하려면 접촉 면적을 먼저 통제해야 한다.**

---

## 9. 한계 및 미해결 (논문에 명시할 것)

| 항목 | 상태 |
|---|---|
| 팬텀 1 개 | val/test 는 재현성이지 일반화가 아니다 |
| 유무 판단 오탐 | 홀드아웃 4/9, 무작위 감사 56 % (r3). 경계는 문제 없음 |
| R_SP 부호 | 축 배정 확정, x·y **부호는 증거 두 개가 상충** (§2.2) |
| 세로 서보 목표 | 깊이 설정 때문에 화면 중앙 도달 불가 |
| 라인 좌우 순서 | `flip_lines` 미검증 |
| 힘 채널 | 프리핸드라 전부 무효 — 로봇 경로에서만 유효 |
| US 지연 | 유효 지연(frame persistence 포함), 순수 전송 지연 아님 |
| 라벨 σ 계수 | 🟡 제안값, 튜닝 안 됨 |
| `found` 표시 | 전 에피소드 `not_marked` — 성공 시점은 암묵적(마지막 프레임) |

---

## 10. 재현 요약

```bash
# 1. 세션 → 부채꼴 B-mode + 접촉
python Unet_seg/scripts/prepare_phantom_sessions.py --per-session 15
python Unet_seg/scripts/phantom_training_sets.py contact

# 2. 라벨링 (브라우저 편집기) → 학습 세트
python Unet_seg/scripts/mask_editor/server.py --data data/phantom_c10ur --queue --queue-file <queue>.csv
python Unet_seg/scripts/phantom_training_sets.py sets --run runs/<이전 추론>

# 3. U-Net 파인튜닝 → 전체 마스킹 → 물리 게이팅 내보내기
python Unet_seg/scripts/train.py --config configs/phantom_c10ur_finetune.yaml --manifest <manifest> \
    --init-weights checkpoints/pfus_bladder_edit_man/best.pt
python Unet_seg/scripts/infer.py --config configs/phantom_c10ur_finetune.yaml --checkpoint <ckpt> --output-dir runs/<run>
python Unet_seg/scripts/phantom_training_sets.py export --run runs/<run>

# 4. 정책 데이터셋 → 학습
python policy_learning/scripts/build_dataset.py --sessions data/sessions.csv --out data/policy_dataset.h5
python policy_learning/scripts/train_policy.py --dataset data/policy_dataset.h5 --out runs/exp1
```

**핵심 산출물**

| | 경로 |
|---|---|
| U-Net (최종) | `Unet_seg/checkpoints/phantom_c10ur_r6/last.pt` |
| 마스크 (1일차) | `Unet_seg/data/phantom_c10ur/masks_final/` + `manifest_masked.csv` |
| 마스크 (에피소드) | `Unet_seg/data/phantom_ep100/masks_final/` + `manifest_masked.csv` |
| 사람 라벨 234 장 | `.../phantom_c10ur/masks/`, `.../phantom_ep100/masks/` |
| 접촉 실측 | `.../contact_alines.npz` |
| 정책 데이터셋 | `policy_learning/data/policy_dataset.h5` (5.4 GiB) |
| 세션 목록 | `policy_learning/data/sessions.csv` (268 활성) |
