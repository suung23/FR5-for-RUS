# 런북 — policy 불러오기와 인퍼런싱

2026-09-10 작성. 이 문서만으로 학습된 정책을 불러오고 오프라인·실시간 인퍼런싱을 돌릴 수 있다.
왜 그렇게 하는지의 근거(실측치)까지 적어 둔다 — 다시 유도하지 않게.

---

## 1. 무엇을 쓰는가

| | |
|---|---|
| 체크포인트 | **`runs/qres2/last.pt`** (λ_quality=1e2, 25 epoch 완주) |
| 행동 공간 | **6 자유도** `[x, y, z mm, θx, θy, θz deg]` — `label/P6` 전체 |
| 헤드 | CVAE (`z_dim=4`) + 잔차 분해 Q̂ |
| 학습 설정 | `beta_kl=8.0`, `beta_warmup_epochs=1`, `beta_adapt=false`, `lr=1e-3`, `epochs=25` |

> ⚠️ **`best.pt` 를 쓰지 말 것.** 이 런들의 `checkpoint_metric` 은 `select_nmae`(행동 예측 지표)인데
> 그 지표는 epoch 1~3 이후 개선되지 않아 `best.pt` 가 거의 학습되지 않은 체크포인트에 잡혀 있다.
> 목적이 Q̂ 이므로 완주한 **`last.pt`** 가 맞다.

## 2. 모델 불러오기

```python
from rus_policy.train import load_policy, to_device

model, cfg = load_policy("runs/qres2/last.pt", device="auto")   # cfg 는 체크포인트에 저장된 설정
```

`load_policy` 가 알아서 처리하는 것:

* 체크포인트에 저장된 `config` 로 모델을 조립한다. `discrete_bins` 같은 값이 나중에 바뀌어도 안전하다.
* **`quality_base` 가 없는 옛 체크포인트**(잔차 분해 도입 이전)를 감지해 `quality_residual=False` 로
  되돌린다. 이게 없으면 `from_dict` 가 현재 기본값을 채워 넣어 `state_dict` 가 어긋난다.

`ACTION_DIM` 은 모듈 상수(6)라 설정에 없다. **3 자유도 시절 체크포인트는 로딩되지 않는다** — 재학습해야 한다.

## 3. 오프라인 인퍼런싱

```bash
python3 scripts/eval_policy.py runs/qres2/last.pt \
        --dataset /data4/seong/policy_dataset.h5 --split val --csv eval.csv
```

`select_action`(사전분포 z + Q̂ 선택)으로 돈다 — **라벨을 보지 않는 실행시 경로**다. 출력의 `nmae`
는 σ 정규화 6 축 오차로 학습 로그의 `select_nmae` 와 같은 정의라 체크포인트끼리 비교된다.

> `mae_*` 를 성능으로 읽지 말 것. 학습 로그의 `mae_*` 는 **사후분포 경로**(라벨이 z 로 들어감)라
> 낙관적이다. 2026-09-10 exp1 에서 `mae_y` 2.54 mm 일 때 실제 실행 경로는 16.79 mm 였다.
> 그 차이가 `leak_gap` 이고, σ_net(≈1.06 mm)을 넘으면 누설이다.

## 4. 실시간 인퍼런싱

```bash
source ~/FR5-for-RUS/install/setup.bash
cd policy_learning

# (1) DRY-RUN — 기본값. 계산만 하고 지령하지 않는다
python3 scripts/run_policy.py runs/qres2/last.pt --image-topic /us/image

# 다른 터미널에서 시작 신호 (사용자가 시점을 지정한다)
ros2 topic pub --once /us/policy_enable std_msgs/Bool "{data: true}"

# (2) 확인이 끝나면
python3 scripts/run_policy.py runs/qres2/last.pt --execute --axes rot \
        --max-deg-s 3 --start-force 1.0 --out runs/eval_1
```

`capture_sweep.py` 와 같은 ROS 배선을 쓴다: `{ns}/desired_twist` 발행, `ee_wrt_base`·`wrench_px6d`·
`/diag/retreating` 구독, 영상은 **BEST_EFFORT**(`qos_profile_sensor_data`) 구독 — 기본 QoS 로 구독하면
DDS 가 짝을 맺지 않아 프레임이 하나도 오지 않는다.

### DRY-RUN 에서 반드시 확인할 넷

| 확인 | 기대 | 틀리면 |
|---|---|---|
| `B-mode 변환: {"mode": "scan_convert+letterbox"}` | 극좌표 160×512 자동 감지 | 전처리가 학습과 달라진다 |
| `지각 XX ms/장` | **< 125 ms** (8 fps) | 관측 창이 오래된 프레임으로 채워진다 |
| 중력 채널 `vec[5:8]` | FK 툴 프레임 = 정책 프로브 프레임(x lateral, y elevational, z beam) | 관측 분포가 어긋난다 |
| 각속도 규약 | `angular.x/y/z` = 프로브 x/y/z 둘레 rad/s | **로봇이 엉뚱한 축으로 돈다** |

마지막 둘은 코드가 검증할 수 없는 **가정**이다. `--execute` 전에 사람이 확인해야 한다.

### 안전 규약

* **빔축(z) 병진은 어떤 설정에서도 버린다** — 접촉은 admittance 몫이다.
* `--start-force`(기본 1.0 N) 미만이면 `policy_enable` 이 켜져 있어도 지령하지 않는다.
* `/diag/retreating`, 접촉 소실, 프레임 끊김에서 즉시 정지한다. 힘 한계·후퇴는 `us_diff_ik_node` 가 관리한다.
* `--max-deg-s` / `--max-mm-s` 로 속도를 묶는다. 첫 브링업은 3 °/s 권장.

### 왜 기본이 회전만인가 (`--axes rot`)

프리핸드 IMU 의 **병진 라벨은 쓸 수 없다.** `capture_sweep.py` 헤더의 실측: sweep_y 65 구간에서
ZUPT 가 제거한 누적 드리프트가 순변위의 **2.75 배**(순변위 중앙 14 mm 에 드리프트 중앙 60 mm).
결과가 학습에 그대로 나타난다.

| 축 | 출처 | 라벨 산포 | 모델 MAE | 판정 |
|---|---|---|---|---|
| x, y, z | 가속도 **이중 적분** | 14.5 / 13.4 mm | 12.2 / 7.0 / 3.6 mm | 상수 예측 수준 |
| θx, θy, θz | **AHRS 직접** | θz 0.96° | 0.55 / 0.36 / 0.66° | 상수보다 나음 |

**모델이 실제로 배운 것은 회전뿐이다.** `--axes all` 은 진단용이다.

## 5. Q̂ 이 살아 있는지 확인

```bash
python3 scripts/diag_quality.py runs/qres2/last.pt --dataset /data4/seong/policy_dataset.h5
```

이 정책의 목적은 모방이 아니라 "어떤 움직임이 어떤 영상 변화를 만드는가" 이므로, 성패는 행동 재구성
오차가 아니라 **Q̂ 이 행동에 반응하는가** 로 갈린다.

* **순위상관 ρ 가 1 순위 지표다.** 모델의 행동 기여분 vs 관측이 설명 못 한 실제 잔차의 스피어만 상관.
  무작위 가중치는 ρ≈0 이라 아래 함정에 안 걸린다.
* **민감도(행동을 흔들었을 때 출력 변화)를 단독으로 믿지 말 것.** 학습이 덜 된 모델은 무작위 가중치가
  모든 입력에 반응해 높게 나온다. 2026-09-10 에 초기 epoch 11 % → ep25 3 % 로 떨어졌다.

2026-09-10 기준값 (25 epoch, n=575, 유의 임계 |ρ| > 0.084):

| λ_quality | ρ | 판정 |
|---|---|---|
| **1e2 (qres2)** | **+0.150** | p ≈ 3e-4 — **채택** |
| 10 | +0.135 | p ≈ 0.001 |
| 1e3 | +0.106 | p ≈ 0.01 |
| 1 | +0.050 | 유의하지 않음 |

**약한 신호다.** 행동 기여 std 0.0058 vs 실제 잔차 std 0.0438 — 방향은 맞지만 폭이 13 % 다.
매 tick 옳은 것이 아니라 **평균적으로 옳은** 수준으로 기대해야 한다. 그래서 `run_policy.py` 의
후보 수 기본값이 64 이고, 모드 떨림을 막으려면 `ensemble.py` 의 `hysteresis` 모드를 얹는다.

## 6. 평가 프로토콜

팬텀 물풍선 부피를 바꾸며 영상 품질 유지를 본다. 접근은 텔레옵, 인퍼런싱 시작은 사용자가 지정
(접촉 ~1 N, 영상이 보이기 시작하는 시점).

**대조군을 반드시 같이 찍는다.**

| 조건 | 방법 |
|---|---|
| 가만히 | 텔레옵 접근 후 정지, `policy_enable` 켜지 않음 |
| 정책 | 위 (2) |
| 숙련자 (선택) | 텔레옵으로 계속 추종 — 상한선 |

`decisions.csv` 의 `Q_now` 시계열을 겹쳐 그린다. **"가만히"보다 Q 가 높게 유지되는가** 가 유일한 질문.

> 학습 데이터는 **프리핸드 접근·재위치** 동작이고 평가는 **접촉 후 서보** 구간이다. 분포가 다르다.
> 결과가 나쁠 때 정책의 한계인지 분포 밖 상황인지 구분해서 볼 것.

## 7. 열려 있는 항목

1. **`timing.us_latency_s`** 기본값을 0 → 0.200 으로 고쳐 두었으나 **데이터셋을 다시 빌드해야 반영된다**
   (`dataset.py` 가 빌드 때 프레임 시각에서 뺀다). 정책 스텝이 0.2 s 라 관측·행동이 한 스텝 어긋나 있다.
2. **`sigma_translation_coeff_mm = 0.7`** → σ_net ≈ 1.06 mm 로 병진 라벨 정밀도를 가정하는데 실측
   드리프트는 그 50 배 이상이다. `w_net = 1/σ²` 가중과 Huber δ 가 전부 이 잘못된 σ 위에 있다.
3. **Q̂ 의 행동 대비 부족.** 시연 하나당 행동이 하나뿐이라 "비슷한 영상에서 다른 움직임" 대비가 없다.
   로봇 구동 섭동(FK 가 행동을 σ 0.1 mm 로 알려준다)이 근본 해법이다. `elevational_sweep/capture_sweep.py`
   가 그 구조인데 헤더에 "학습에 쓰지 않는다" 로 명시돼 있다 — 그 배제 결정을 재검토할 것.
4. **힘 축은 학습 대상이 아니다.** `dataset.py` 가 `label/F_valid` 를 조건 없이 False 로 쓰므로
   `l_force`·`l_feas`·`l_risk` 는 그래디언트를 받은 적이 없다. 힘 제어는 별도 제어층이 담당한다 (설계대로).
