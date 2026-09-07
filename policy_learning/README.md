# policy_learning — 초음파 영상 + IMU 시연 → ACT policy 학습 파이프라인

`imu_bench/host/us_imu_collect.py` 가 남기는 세션 디렉터리(US 프레임 + BNO085 IMU)만 있으면
데이터셋 빌드 → 학습 → 진단까지 바로 돌아가는 파이프라인이다. 설계 근거는 전부
[`docs/POLICY_LEARNING_MATH.md`](../docs/POLICY_LEARNING_MATH.md) 이고, 코드 주석에 절 번호를 남겼다.

```
세션 디렉터리 (us_imu_<stamp>/)                      ┐
   us_frames.bin + us_index.csv   (256×256 uint8, ~8 fps) │  session.py
   imu_<stamp>.csv (+meta.json)   (33열, ~200 Hz)         ┘
        │
        ├─ imu_labels.py   정지→이동→정지 분절, ZUPT 이중적분, 앵커 프로브 프레임으로 회전
        │                  → chunk 라벨 P̃ (k+1, 3) [x mm, y mm, θz deg] + σ_net·σ_shape      (§2, §5.2)
        ├─ perception.py   U-Net → ControlState → s_t (21차원), Q_seg, 액션 토큰 τ=(a, ê, Q)  (캐시)
        └─ dataset.py      샘플 = 구간 하나:  관측 (m 프레임, s_t, Ã_{t−1}, 중력, wrench) + 라벨 → HDF5
                │
                ├─ model.py    ACT-lite: CNN 프레임 토큰 + Transformer enc/dec, CVAE z, Q̂·F̂ 헤드  (§3)
                ├─ losses.py   이종분산 Huber(순변위+형상) + λ_Q Q̂ + λ_s smooth + β KL (+힘 항 마스크) (§5.3)
                └─ train.py    학습 루프 + 모드 붕괴·모드 마진·부호 정확도 진단                   (§6)
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
python -m pytest            # 31 tests, CPU 1–2 분
```

합성 세션은 수집기와 같은 파일 레이아웃이고 `truth.json` 에 정답이 있다. 라벨 검증: 회전 오차 0.03°,
병진 0.2–1 mm (가속도 잡음에 비례), 구간 검출률 > 95 %.

## 3. 반드시 채워야 하는 값 (⏳)

| 키 | 의미 | 어디서 |
|---|---|---|
| `timing.us_latency_s` | US 프레임의 고정 엔드투엔드 지연. 프레임 시각에서 뺀다 | §7.1, DESIGN_NOTES ⏳ 실측 |
| `imu.sensor_to_probe` | R_SP (IMU 센서 → 프로브 프레임 x lateral / y elevational / z beam) | IMU 마운트 측정. 틀리면 라벨 축이 뒤바뀐다 |
| `imu.still_gyro_sd / still_gyro_mean / still_accel_sd` | 프리핸드 정지 판정 임계 | `inspect_session.py` 의 분위수 표. qc_common 값은 로봇용이라 손에는 너무 빡빡하다 |
| `perception.frame_transform` | candidate 프레임 방향 (scan conversion 검증 전) | `us_protocol.py` ⏳ |
| `paths.unet_checkpoint` | Q_seg 를 만드는 U-Net | 학습 머신 |
| `timing.obs_frames` (m) | f_us 실측 후 재산정 (§1.3) | 8 fps 면 16 |

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

## 6. 진단 (metrics.jsonl)

| 키 | 의미 | 경고 |
|---|---|---|
| `val/mae_x_mm, mae_y_mm, mae_th_deg` | 축별 순변위 절대오차 | |
| `val/vy_sign_acc` | Δy 부호 정확도 (Δy > σ 인 샘플) | 0.5 근처 = 이봉성을 못 풀었다 (L6) |
| `val/mode_collapse_vy_std` | z 표본 M 개의 Δy 표준편차 | 0 으로 가면 posterior collapse (§5.3g) — β 를 낮출 것 |
| `val/mode_margin` | Q̂ 로 고른 모드와 반대 부호 모드의 점수 차 | 잡음 수준이면 대칭 붕괴 (L11): dither·Ã_{t−1} 채널 점검 |
| `val/select_vy_sign_acc` | 실행시 선택 경로(select_action)의 부호 정확도 | 학습 경로보다 훨씬 낮으면 L12 |

## 7. 아직 없는 것

- **텔레오퍼레이션(FK 라벨) 로더.** `source=teleop` 는 명시적으로 `NotImplementedError`. US+FK 를 동기
  수집하는 수집기가 아직 없다 (`fr5_h5_collector.py` 는 구 복강경용). 수집기가 생기면 `label_segment` 의
  P_E/R_SE 입력만 FK 로 바꾸면 되고, σ 표와 힘 채널(`obs/vec`, `label/F`)은 이미 자리가 있다.
- **실행시 통합 (QP arbiter, temporal ensembling 모드 보존, §3.5).** `select_action` 은 한 tick 의 선택까지만.
- **§8-6, §8-7 의 대칭 붕괴·ensembling 수치 실험.** 합성 세션이 이봉성(|y| 만 영상에 보임)을 재현하므로
  이 데이터로 바로 할 수 있다.
