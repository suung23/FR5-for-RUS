# policy_learning — 초음파 영상 + IMU 시연 → ACT policy 학습 파이프라인

**범위는 Stage 2 (면내 3축) 다.** policy 가 내는 것은 `(v_x, v_y, ω_z)` 뿐이고, 힘축과 Stage 1 의
힘 탐색은 F/T + admittance 가 맡는다 (DESIGN_NOTES §7). 프리핸드 시연에서 힘 라벨을 얻을 수 없다는 것이
이 분리의 **이유**이므로, `label/F`·`Fn_star` 와 `loss.lambda_force/feas/risk` 는 쓰지 않는 확장점이고
`obs/vec` 의 wrench 7 칸은 학습·추론 양쪽에서 0 이다.

`imu_bench/host/us_imu_collect.py`(Linux, Wi-Fi 프로브) 또는 `imu_bench/host/us_imu_gui_win.py`(Windows, USB 프로브
C10UR — 벤더 뷰어 화면 캡처, 2026-09-09) 가 남기는 세션 디렉터리(US 프레임 + BNO085 IMU)만 있으면
데이터셋 빌드 → 학습 → 진단까지 바로 돌아가는 파이프라인이다. 설계 근거는 전부
[`docs/POLICY_LEARNING_MATH.md`](../docs/POLICY_LEARNING_MATH.md) 이고, 코드 주석에 절 번호를 남겼다.

> **2026-09-08 개정 (POLICY_LEARNING_MATH §10).** elevational 축의 모드 선택은 학습이 아니라 **오프셋 필터**
> (`offset_filter.py`, §3.9) 가 맡는다. ACT 는 면내 (v_x, ω_z) 와 스타일만. z 는 디코더에만 들어가고
> Q̂ 는 관측만 본다. β 는 0.5 + 워밍업, z_dim 4. elevational 라벨은 Q_seg 가 아니라 정규화 면적 `Q_area`.
> σ 는 chunk 창으로, 관측 창 증강으로 표본 ×3. **US fps 는 가정하지 않는다** — `inspect_session.py` 가 잰다.

```
세션 디렉터리 (us_imu_<stamp>/)                      ┐
   us_frames.bin + us_index.csv   (256×256 uint8, ~8 fps) │  session.py
   imu_<stamp>.csv (+meta.json)   (33열, ~200 Hz)         ┘
        │
        ├─ calibration.py  ⏳ 상수 추정: US 지연 (영상↔가속도 상호상관), R_SP (자이로 주축)      (§7.1, README §3)
        ├─ imu_labels.py   정지→이동→정지 분절, ZUPT 이중적분, 앵커 프로브 프레임으로 회전
        │                  → chunk 라벨 P̃ (k+1, 3) [x mm, y mm, θz deg] + σ_net·σ_shape (chunk 창 기준)  (§2, §5.2)
        ├─ perception.py   U-Net → ControlState → s_t (21차원), Q_seg, 액션 토큰 τ=(a, ê, Q)  (캐시)
        └─ dataset.py      샘플 = 구간 × 관측 시각:  관측 (m 프레임, s_t, Ã_{t−1}, 중력, wrench) + 라벨
                │          (P̃, Q_seg, **Q_area** = 정규화 면적) → HDF5
                │
                ├─ model.py         ACT-lite: CNN 프레임 토큰 + Transformer enc/dec, CVAE z (**디코더 쿼리에만**),
                │                   Q̂·F̂ 헤드 (Q̂ 는 관측 요약만 본다)                                   (§3)
                ├─ losses.py        이종분산 Huber(순변위+형상) + λ_Q Q̂ + λ_s smooth + β KL (워밍업)      (§5.3)
                ├─ train.py         학습 루프 + 모드 붕괴·**누설 gap**·모드 마진·부호 정확도 진단           (§6)
                ├─ offset_filter.py **elevational 오프셋 d 의 베이즈 필터** — 학습 없음. 모드 부호와 u* 를 낸다 (§3.9)
                └─ ensemble.py      실행시 chunk 혼합 — 모드 보존 ensembling, 필터 부호를 `forced_mode` 로 받는다 (§3.5)
```

## 1. 설치

```bash
pip install -r policy_learning/requirements.txt        # torch, numpy, h5py, pyyaml, pillow, opencv-python-headless
```

U-Net 지각(`perception.backend: unet`)은 `Unet_seg/rus_perception` 을 그대로 import 하며, 학습 머신의
`Unet_seg/checkpoints/exp_seed43/best.pt` 가 필요하다 (저장소에 없음). 체크포인트 없이 파이프라인만
검증하려면 `--set perception.backend=none` (Q̂ 항이 마스크되고 s_t 는 0).

## 2. 데이터가 들어오면 — 순서대로

```bash
cd policy_learning

# (0) 세션 점검 — 정지 판정 임계가 데이터 분포 어디에 놓이는지, 구간이 몇 개 잡히는지
python scripts/inspect_session.py /path/to/us_imu_20260910_101500 --plot check.png

# (1) 세션 목록 작성  data/sessions.csv
#     session_dir,subject,source,split,note
#     /data/us_imu_20260910_101500,P01,freehand,train,
#     /data/us_imu_20260910_113000,P02,freehand,val,
#     split 을 비우면 --set split.strategy=subject_random 으로 피험자 단위 자동 분할

# (2) 데이터셋 빌드 (U-Net 추론 포함, 세션별 캐시)
python scripts/build_dataset.py --sessions data/sessions.csv --out data/policy_dataset.h5

# (3) 학습
python scripts/train_policy.py --dataset data/policy_dataset.h5 --out runs/exp1
python scripts/train_policy.py --dataset data/policy_dataset.h5 --out runs/exp1_discrete --set model.head=discrete
```

출력: `runs/exp1/{config.yaml, metrics.jsonl, last.pt, best.pt}`. 학습된 모델은
`rus_policy.train.load_policy(path)` → `model.select_action(batch)` 로 chunk 를 뽑는다 (§3.3 z 후보 M 개 → Q̂ 선택).

```bash
# (4) 평가 — 실행시 선택 경로로 test split 의 MAE·부호 정확도·모드 마진, 샘플별 CSV
python scripts/eval_policy.py runs/exp1/best.pt --dataset data/policy_dataset.h5 --split test --csv runs/exp1/test.csv
```

### 하드웨어 없이 전체 경로 검증

```bash
python scripts/make_synthetic_sessions.py --out data/synthetic --n 4 --duration 60
python scripts/build_dataset.py --sessions data/synthetic/sessions.csv --out data/synthetic/dataset.h5 --set perception.backend=none
python scripts/train_policy.py --dataset data/synthetic/dataset.h5 --out runs/synthetic --set train.epochs=5
python -m pytest            # 45 tests, CPU 1–2 분
```

합성 세션은 수집기와 같은 파일 레이아웃이고 `truth.json` 에 정답이 있다. 라벨 검증: 회전 오차 0.03°,
병진 0.2–1 mm (가속도 잡음에 비례), 구간 검출률 > 95 %.

## 3. 반드시 채워야 하는 값 (⏳) — 데이터 첫날에 순서대로

| 키 | 의미 | 어디서 |
|---|---|---|
| **US fps** (`timing.obs_frames` m) | **가정하지 않는다.** `inspect_session.py` 가 프레임 간격 중앙값에서 fps 를 재고 `m = ceil(f_us / f_dither)` 를 찍어 준다 (§1.3). 현재 16 은 자리표시자 | `python scripts/inspect_session.py <session>` 첫 줄 |
| `timing.us_latency_s` | US 프레임의 고정 엔드투엔드 지연. 프레임 시각에서 뺀다 (§7.1) | `inspect_session.py <session> --latency` — 프로브를 젤에서 급히 떼었다 붙이는 사건 10–20 회가 든 세션. 영상 평균강도 급변 ↔ 가속도 스파이크 상호상관. 양수 = 영상이 늦다 |
| `imu.sensor_to_probe` | R_SP (IMU 센서 → 프로브 프레임 x lateral / y elevational / z beam). 틀리면 라벨 축이 뒤바뀐다 | `python scripts/calibrate_sensor_to_probe.py <session>` — 프로브 x, y, z 축 둘레로 차례로 크게 돌린 30 s 세션. 출력이 곧 YAML 행 |
| `imu.still_gyro_sd / still_gyro_mean / still_accel_sd` | 프리핸드 정지 판정 임계 | `inspect_session.py` 의 분위수 표. qc_common 값은 로봇용이라 손에는 너무 빡빡하다 |
| `perception.frame_transform` | candidate 프레임 방향 (scan conversion 검증 전) | `us_protocol.py` ⏳ |
| `split.allow_subject_overlap` | 피험자 1 명 파일럿(첫날 15 세션, `data/sessions.csv`) 은 세션 단위 split — 누수 검사를 경고로 낮춘다. val/test 는 같은 사람 안의 재현성만 잰다 | `--set split.allow_subject_overlap=true` |
| (자동) C10UR 극좌표 → B-mode | 320×256 극좌표 세션은 `rus_policy/bmode.py` 가 `us.fan_geometry` 로 부채꼴을 만들고 정방형 레터박스 → `frame_size`. 라인 좌우(`flip_lines`) 는 팬텀으로 검증 ⏳ | `build_dataset` / `perceive_session` 이 자동 적용, 캐시 키에 포함 |
| `paths.unet_checkpoint` | Q_seg 를 만드는 U-Net | 학습 머신 |
| `offset_filter.sigma_q` | 정규화 면적의 프레임간 잡음 (호흡·변형 포함). 필터의 사각지대를 정한다 | 정지 구간에서 `Q_area` 의 표준편차 (`inspect_session` 항목 예정) |

세션 메타에 `zero_ref` (GUI 의 Z 키) 가 있으면 그 `quat_convention` 을 쓴다. 없으면 정지 표본으로
R/Rᵀ 규약을 판정하는데, 빔이 중력과 평행하면 판정이 모호하다 (빌드 로그에 "(모호)" 표시).
**영점을 잡고 수집하는 것을 권장한다.**

## 4. 라벨의 정의 (imu_labels.py)

- 구간 = 정지 A → 이동 → 정지 B. 앵커 t_a 는 정지 A 의 끝쪽 25 % 안 (analyze_track.move_segments 와 동일).
- a_E = R_SE(q)·a_S 에서 정지 A 의 평균(중력 포함)을 빼고, 사다리꼴 이중적분 + 끝 속도 0 선형 디드리프트 (ZUPT).
  상수 바이어스는 정확히 상쇄된다 (`test_zupt_cancels_constant_bias_exactly`).
- P_P = R_SP · R_SE(q_a)ᵀ · P_E (앵커 프로브 프레임), 회전은 log(R_aᵀ R_t) 를 같은 프레임으로.
- chunk 격자 t_a + i/f_p (i = 0..k) 에서 표본화. 이동이 끝나면 값이 유지된다.
- σ_net: 프리핸드 `c·τ^1.5` (c = 0.7 mm/s^1.5, τ = 적분창), 회전 0.1°; 텔레오퍼레이션 0.1 mm / 0.05°.
  σ 가 loss 의 가중이므로 두 소스가 자동으로 올바르게 섞인다 (§5.3a).
- 힘축 (z, θx, θy) 는 `label/P6` 에 남기되 loss 에는 넣지 않는다 (§5.3b).

## 5. 관측의 정의

| 항목 | 내용 |
|---|---|
| `obs/frames` (m,H,W) | 앵커 이전 마지막 m 장. 시간 간격 `obs/frame_dt`, 패딩 `obs/frame_valid` |
| `obs/state` (m,21) | `perception.STATE_FEATURE_NAMES`: 중심 오프셋, 면적, 타원축, 신뢰도, 대비, Q, ê, 토큰 one-hot |
| `obs/vec` (15) | `dataset.OBS_VEC_NAMES`: Ã_{t−1} 실현치 (이전 구간 순변위, 현재 프레임으로 회전), 중력 방향 (roll/pitch 대용, §1.7), wrench·포화 (프리핸드 0), has_force |
| `label/Q_area` (k) | chunk 격자에서의 **정규화 마스크 면적** (세션 99 백분위 = 1). elevational 관측량 — Q_seg 의 면적 플라토(§6.3) 는 최적점 근방에서 \|d\| 에 상수라 elevational 목적으로 못 쓴다. 필터의 y_t 와 같은 정의 |
| 관측 시각 | `dataset.obs_anchor_samples` (기본 3): 앵커 외에 정지 A 안의 시각 2 개를 "지금" 으로 더 뽑는다. 프로브가 정지해 있으므로 라벨은 같다. `meta/json` 의 `obs_offset_s` |

## 6. 진단 (metrics.jsonl)

| 키 | 의미 | 경고 |
|---|---|---|
| `val/mae_x_mm, mae_y_mm, mae_th_deg` | 축별 순변위 절대오차 | |
| `val/vy_sign_acc` | Δy 부호 정확도 (Δy > σ 인 샘플) | 0.5 근처 = 이봉성을 못 풀었다 (L6) |
| `val/mode_collapse_vy_std` | z 표본 M 개의 Δy 표준편차 | 0 으로 가면 posterior collapse (§5.3g) — β 를 낮출 것 |
| `val/mode_margin` | Q̂ 로 고른 모드와 반대 부호 모드의 점수 차 | 잡음 수준이면 대칭 붕괴 (L11): dither·Ã_{t−1} 채널 점검 |
| `val/select_vy_sign_acc` | 실행시 선택 경로(select_action)의 부호 정확도 | 학습 경로보다 훨씬 낮으면 L12 |
| `val/leak_gap_mm` | `select_mae_y` (사전분포 z) − `mae_y` (사후분포 z) | σ_net 을 크게 넘으면 **누설** — 인코더가 라벨을 z 로 흘리고 디코더가 관측을 무시. β 를 올릴 것. **β 선택 규칙: gap ≤ σ_net 이면서 vy_std > 0 인 최소 β** ({0.1, 0.5, 1, 5} 스윕) |
| `train/beta` | 워밍업 중인 실효 β | `train.beta_warmup_epochs` 동안 0 → `loss.beta_kl` |

## 7. 아직 없는 것

- **필터 ↔ ACT 실행 결합.** `OffsetFilter.action()` 의 u* 를 chunk 의 v_y 로, `forced_mode()` 를 ensembler 에 넣는
  실행 루프. 지금은 각 조각과 `offset_filter.replay` (실세션 오프라인 재생) 까지다.
- **`L_rank` (§3.3-1 대안).** Q̂ 를 선택기로 유지하려면 ±d 쌍의 순위 손실이 필요한데, 그 쌍은 스윕 프로토콜
  (§10-3) 데이터에서만 나온다. 필터가 우선이므로 보류.
- **텔레오퍼레이션(FK 라벨) 로더.** `source=teleop` 는 명시적으로 `NotImplementedError`. US+FK 를 동기
  수집하는 수집기가 아직 없다 (`fr5_h5_collector.py` 는 구 복강경용). 수집기가 생기면 `label_segment` 의
  P_E/R_SE 입력만 FK 로 바꾸면 되고, σ 표와 힘 채널(`obs/vec`, `label/F`)은 이미 자리가 있다.
- **QP arbiter 와의 결합.** `select_action` + `TemporalEnsembler` 는 한 tick 의 지령까지만 낸다.
- **§8-6 의 대칭 붕괴 실험 (A6 / L11).** ⚠️ 지금 합성 세션으로는 못 한다 — 아래 "합성 데이터의 한계".

## 8. 실행시 chunk 혼합 (`ensemble.py`, §3.5 · L12)

ACT 의 표준 temporal ensembling 은 겹치는 chunk 예측의 지수가중 평균이라, 행동분포가 이봉이면
`+d` 와 `−d` 를 상쇄해 **로봇을 제자리에 세운다**. `Q̂` 모드 선택(§3.3)이 이봉성을 살리려고 넣은
것이므로 그 위에 평균을 얹으면 서로 지운다. `TemporalEnsembler` 가 §3.5 의 처방을 모드로 제공한다.

| `EnsembleConfig.mode` | 내용 |
|---|---|
| `mean` | 표준 ACT. **대조군** — 모드 소멸을 보이는 데 쓴다 |
| `cluster` | 처방 2. 평균 전에 `a_y` 부호로 클러스터링, 커밋된 모드 안에서만 평균 |
| `hysteresis` | 처방 3. cluster + 전환을 chunk 경계로 제한하고 점수 차 임계 |
| `none` | 혼합 없음 (최신 chunk 만) |

처방 1(모드 일관성 보너스)은 혼합이 아니라 **선택** 단계라 `model.select_action` 의 `gamma`·`prev_dy`
에 있다. 셋은 배타적이지 않으므로 1+2 또는 1+3 으로 겹쳐 쓰는 것이 정상이다.

**2026-09-08 — 모드의 출처는 필터다.** `step(forced_mode=필터.forced_mode())` 로 넘기면 커밋 모드가 Q̂ 점수가
아니라 필터의 MAP 부호가 된다. Q̂ 가 무정보일 때 히스테리시스가 첫 무작위 모드에 잠기는 실패 (§9-6) 가
구조적으로 사라진다. `forced_mode=0` ("필터가 아직 모름") 이면 점수 규칙으로 돌아간다.
필수 진단은 `flip_rate_hz` — 🟡 1 Hz 초과면 모드 진동이다. 이 실패는 조용하다 (로봇이 안 움직일
뿐이고 `Q_seg` 도 나빠지지 않는다).

### §8-7 (A7 / L12) 수치 실험 — 검증 완료

```bash
python scripts/exp_ensembling.py --trials 200 --ticks 400 --json a7.json
```

학습도 데이터도 필요 없다. 이봉 policy(`±d` 확률 ½)를 각 처방에 통과시켜 **유지율**
(`mean |a_y 방출| / |a_y chunk|`)과 부호 전환율을 잰다. `k = 8`, `f_p = 5 Hz`, `γ = 0.2`, `σ_Q = 1`:

| 처방 | 유지율 (p=0.5) | 전환율 Hz | 유지율 (p=0.8) |
|---|---|---|---|
| 혼합 없음 (대조) | 1.000 | 2.41 | 1.000 |
| **표준 ACT ensembling** | **0.280** | 1.12 | 0.610 |
| + 처방1 | 0.438 | 0.63 | 0.808 |
| 처방2 모드내 평균 | 1.000 | 1.12 | 1.000 |
| + 처방1 | 1.000 | 0.63 | 1.000 |
| 처방3 경계 히스테리시스 | 1.000 | 0.13 | 1.000 |
| **+ 처방1** | **1.000** | **0.02** | 1.000 |

**L12 확인.** 표준 ensembling 의 0.280 은 해석해와 일치한다 — 겹친 `k = 8` 개의 독립 `±1` 예측을
균등평균한 `E|S₈|/8 = 560/256/8 = 0.2734` (`test_mean_annihilates_bimodal_analytically` 가 고정).
즉 **모드 소멸은 구현 결함이 아니라 평균 연산자의 성질**이고, `k` 를 늘릴수록 나빠진다.

**처방 판정.** 2·3 은 진폭을 온전히 되살린다(유지율 1.000). 다만 진폭만으로는 부족하다 — 처방 2 단독은
전환율이 1.12 Hz 로 경보선 위라 매 tick 모드가 바뀌며 제자리 진동한다. **3 + 1 조합이 0.02 Hz 로
유일하게 커밋한다** (순진행 0.413 대 처방2 단독 0.086). 시나리오 B 는 처방이 **정당한 증거까지**
뭉개지 않는지 보는 대조군이고, 전부 다수 모드로 수렴하므로 통과다.

> ⚠️ 이 실험은 **A7 만** 판정한다. `Q̂` 가 애초에 모드를 고를 수 있는가(A6 / L11)는 §8-6 이고,
> 아래 한계 때문에 아직 못 한다.

### 합성 데이터의 한계 — §8-6 을 아직 못 하는 이유

`synth.py` 는 정지 구간마다 방광 오프셋을 **i.i.d. 로 새로 뽑는다**. 영상에는 슬라이스 두께로
`|y_off|` 만 나타나고 `Ã_{t−1}` 은 *이전* 목표를 가리키므로, **현재 Δy 의 부호가 관측 어디에도
없다.** 그래서 이 데이터에서 `vy_sign_acc ≈ 0.5` 는 실패가 아니라 정답이고, §3.4 가 요구한
대조군("`Ã_{t−1}` 에 진짜 이전 변위를 넣으면 정확도가 회복되는가")을 만들 수 없다.

§8-6 을 하려면 생성기가 실제 시연 프로토콜을 재현해야 한다 — 목표를 정지마다 새로 뽑지 말고
**여러 chunk 에 걸쳐 수렴**시키면 `Ã_{t−1}` 의 부호가 곧 현재 필요한 운동의 부호가 되어 대칭이
깨진다. 함께 걸리는 것 두 개:

- `perception.backend=none` 이면 `Q_valid` 가 전부 False → `l_qual = 0` → **`quality_head` 그래디언트가 0**
  이다 (학습 후 weight std 가 초기화값 `1/√(3·280)` 그대로인 것으로 확인). 합성 검증 경로가 정확히
  이 설정이라, 지금은 `select_action` 이 학습되지 않은 헤드로 후보를 고른다.
- `Q̂` 의 loss 는 **(관측, 정답 chunk) 쌍만** 본다 (`losses.py` 의 `predict_quality(…, P_lab)`).
  같은 관측에 틀린 `A` 를 붙여 낮은 `Q` 를 주는 신호가 없다. §3.3 성립조건 1 이 이것을
  "elevational 스윕이 학습 데이터에 포함되어야 한다" 로 짚어놨고 — 스윕이 목표를 지나쳐 나가는
  구간이 자연스러운 음성이 된다 — 그 데이터가 아직 없다.

## 9. 오프셋 필터 (`offset_filter.py`, §3.9 · 2026-09-08 채택)

elevational 오프셋 `d` (프로브 − 목표, +y) 를 **학습하지 않고 추정**한다. 이유 셋이 같은 뿌리다: Q̂ 는
반사실 (같은 관측에 다른 행동) 을 본 적이 없어 후보를 순위 매길 근거가 없고, Q_seg 의 면적 플라토는 최적점
근방에서 |d| 에 상수라 목적함수가 평탄하며, 앙상블 히스테리시스는 Q̂ 가 무정보일 때 첫 모드에 잠긴다.

```
예측   d_t = d_{t−1} + u_t + w,     u_t = 실현 elevational 변위 (Ã_{t−1}),  w ~ N(0, (c·τ^1.5)²)
관측   y_t = A₀·sqrt(1 − d²/R²) + v,   y = Q_area,  v ~ N(0, σ_q²)         격자 (d × R × A₀), 2,178 칸
행동   Var[d] 작음 → u* = −E[d]          이봉 → u* = −sign(d̂_major)·Δ_probe,  Δ_probe = min{Δ : |Δh| ≥ snr·σ_q}
       Δ_max 로도 못 가르면 → hold  (사각지대: |d| < snr·σ_q·R²/(2Δ_max).  σ_q = 0.01, R = 50, Δ_max = 8 → 3 mm)
```

사각지대는 면적의 2 차 형상 (dh/dd = d/R²) 이 주는 물리 한계이지 필터의 결함이 아니다 — 서보잉 §5.2 와 같은
양이고, `test_offset_filter.py` 가 그 식으로 기대치를 잡는다. **σ_q 가 설계 변수다**: 정지 구간의 `Q_area`
표준편차로 재고, 3 mm 가 임상적으로 부족하면 분해능을 올리는 길은 σ_q 를 줄이는 것 (면적 대신 장축, 프레임
평균) 뿐이다. `replay(filter, u_seq, y_seq)` 로 실세션에서 오프라인 재생해 두 봉이 실현 운동으로 접히는지
본다 (§8-6 의 대체 검증).

## 10. 수집 프로토콜 추가 — 스윕 (§10-3)

세션마다 "찾기" 가 끝난 뒤 목표를 **지나치는** 스윕을 넣는다:

```
[정지] → +δ 이동 → [정지] → −2δ 이동 → [정지] → +δ 복귀 → [정지],   δ ∈ {2, 4, 8} mm,  세션당 30 s
```

세 가지를 동시에 준다 — (1) A4 의 오프셋 공간 데이터, (2) 필터 관측 모델 h(|d|; R) 의 실측 검증, (3) Q̂ 를
선택기로 유지할 경우의 ±d 순위 손실 쌍. 이동 길이는 1–1.5 s 를 지킨다 (chunk 창 1.6 s; 빌드 로그의
"chunk 창 초과 %" 가 30 % 를 넘으면 재교육).

