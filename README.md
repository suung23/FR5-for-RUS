# FR5-for-RUS

로봇 초음파(Robotic UltraSound) 제어 스택. 복강경 봉합용 듀얼암 시스템을 **오른팔 단일 초음파 프로브 제어**로 전환한 리포지터리입니다.

- **지각**: Slim U-Net 방광 내강 분할 + 제어용 품질함수 2종.
- **제어**: 100 Hz admittance 힘 제어 + 영상 기반 면내 스캐닝.
- **핵심**: 영상 품질을 유계 스칼라로 만들어 힘 축의 **탐색 목적함수**로 쓰고, 힘 축과 영상 축을 직교 분리해 서로 다른 대역에서 돌립니다.

| 항목 | 상태 |
|---|---|
| 제어 노드 7종 | 구현 완료, ROS 실기 검증 대기 |
| 단위 테스트 | 지각 350 + 제어 73 |
| 초음파 데이터셋 | **미도착** (`Unet_seg/data/` 비어 있음) |
| 하드웨어 | 프로브 마운트 CAD · F/T 장착 · 프레임그래버 지연 **전부 대기** |

설계 근거와 미결 항목은 [`DESIGN_NOTES.md`](DESIGN_NOTES.md), 검증 기준은 [`docs/VALIDATION_PLAN.md`](docs/VALIDATION_PLAN.md)에 있습니다.

---

## 저장소 구조

```
FR5-for-RUS/
├─ Unet_seg/                   지각 라이브러리 (ROS 무의존)
│  └─ rus_perception/
│     ├─ models/               Slim U-Net, Standard U-Net
│     ├─ control/              품질함수, 유효성 게이트, ControlState
│     ├─ losses/               공간(Dice/Jaccard/BCE) + 시간(광류 정합)
│     ├─ inference/            Predictor, 실시간 파이프라인, 프레임 소스
│     └─ metrics/              공간·시간·신뢰도 지표
│
├─ fr5_control/                ROS 2 워크스페이스
│  ├─ fr5_control/             서보 · admittance · 탐색 · 감독 · 지각 래퍼
│  ├─ fr5_ik/                  DLS 역기구학 + 추종오차 감시
│  ├─ fr5_vision/              US 프레임 취득
│  ├─ touch_teleop/            Touch 햅틱 원격조작 (C++)
│  ├─ fr5_launch/              런치
│  └─ legacy_laparoscopic/     복강경 전용 코드 (COLCON_IGNORE)
│
└─ docs/                       도식 9종 + 검증 계획
```

**경계 규칙**: `rus_perception`은 ROS를 모릅니다. 그래야 학습과 평가가 ROS 없이 돕니다.
경계를 건너는 지점은 `us_perception_node` 하나뿐입니다.

---

## 지각

### 모델

```
Slim U-Net (production preset)
├─ 입력    1 × H × W   흑백 B-mode, [0,1], H·W는 2⁴ 배수
├─ 인코더  32 → 64 → 128 → 256 → 512,  depth 4
├─ 디코더  transposed_conv, up_kernel 3
├─ 특징    단계당 conv 1회 (standard는 2회)
└─ 출력    1 × H × W   방광 내강 이진 로짓
```

| | Slim | Standard |
|---|---|---|
| 파라미터 | **4.71 M** | 31.04 M |

손실은 `BCE 1.0 + Dice 1.0 + Jaccard 1.0`이며, 시간항(광류 정합 일관성·소프트 중심)은 선택입니다.

### 품질함수 2종

분할 마스크를 그대로 쓰지 않고 **제어용 스칼라**로 환원합니다.

```
Q = Σ wᵢsᵢ / Σ wᵢ   ∈ [0,1]
```

| | 근거 | 담당 축 | 쓰이는 곳 |
|---|---|---|---|
| `Q_raw` | 고전 영상처리, **세그 비의존** | 힘 (z, rx, ry) | Stage 1a, 배경 감시 |
| `Q_seg` | 분할 기반 | 면내 (x, y, rz) | Stage 1b, policy |

`Q_raw`가 따로 필요한 이유는 Stage 1 시작 시점에 방광이 안 보여 `Q_seg`가 **정의되지 않기** 때문입니다.
점수를 낼 수 없는 프레임은 **NaN**으로 냅니다 — 0으로 내면 "품질 최악"과 "측정 불가"가 구별되지 않습니다.

두 값을 같이 읽으면 원인이 갈립니다:

| `Q_raw` | `Q_seg` | 해석 | 조치 |
|---|---|---|---|
| 정상 | 열화 | 면내 위치 문제 | policy가 처리 |
| 열화 | 열화 | **접촉 문제** | 힘 재탐색 |

---

## 제어

### 축 분담

```
z   ← admittance   F_n → F_n*      100 Hz    학습 불필요
rx  ← admittance   M_x → M*_x      100 Hz    학습 불필요
ry  ← admittance   M_y → M*_y      100 Hz    학습 불필요
x, y, rz  ← 영상 policy             5 Hz     학습 대상
```

프로브면을 표면 법선에 맞추는 것은 rx/ry 회전이고, **Mx/My가 그 오정렬을 직접 측정합니다.**
덕분에 영상 policy가 5-DoF에서 **3-DoF**로 줄고, 자세 정렬이 30 Hz 영상 루프가 아니라 100 Hz 힘 루프에서 돕니다.

policy 출력은 **속도 3 + setpoint 3**입니다. 힘 축을 속도로 지령하지 않고 setpoint만 옮기므로 대역 분리가 유지됩니다.

### 신호 경로

```
US B-mode (30 Hz)
   ├─ 고전 영상처리 → Q_raw ─┐
   └─ Slim U-Net    → Q_seg ─┤
                             ▼
              Stage 1  F* = min{F : Q̄(F) ≥ (1−ε)·max Q̄}
                             │ (F*, M*_x, M*_y)  WrenchStamped 한 묶음
                             ▼
              Admittance 100 Hz   v = (목표 − 측정) / B
                             ▼  프로브 프레임 twist 6-DoF
              DLS   q̇ = Jᵀ(JJᵀ + λ²I)⁻¹ V,  λ = 0.02
                             ▼
              ServoJ 125 Hz  →  FR5
```

**선택 규칙이 argmax가 아닙니다.** 최댓값과 통계적으로 구분되지 않는 **가장 작은 힘**을 고릅니다 — 환자 부담을 줄이는 방향으로 의도적으로 편향되어 있습니다.

### RCM이 없다는 것의 의미

복강경에서 회전 중심은 트로카에 기구학적으로 못 박혀 있었습니다. 삭제하면 회전 중심은 **툴 프레임 원점**이 됩니다.

> 프로브 접촉면이 RCM을 대체하는 회전 중심입니다. 다만 구속으로 강제되는 것이 아니라 툴 변환을 맞게 넣어야만 성립합니다.

원점이 `δ`만큼 어긋나면 `v_slip = ω × δ`의 미끄러짐이 생깁니다. `ω = 0.2 rad/s`, `δ = 20 mm`면 **4 mm/s**로 병진 예산의 40%입니다. **CAD 정확도가 회전 품질을 지배합니다.**

### 노드

| 실행파일 | 패키지 | 주기 |
|---|---|---|
| `us_servo` | `fr5_control` | 125 Hz |
| `us_admittance` | `fr5_control` | 100 Hz |
| `us_diff_ik` | `fr5_ik` | 100 Hz |
| `us_perception` | `fr5_control` | 30 Hz |
| `us_frame` | `fr5_vision` | 30 Hz |
| `us_supervisor` | `fr5_control` | 20 Hz |
| `us_force_search` | `fr5_control` | Stage 1 중에만 |
| `policy_node` | — | ❌ Phase 6 |

### 상태 머신

```
IDLE → TELEOP_APPROACH → STAGE1A_CONTACT → STAGE1B_FOCUS → STAGE2_SCAN
            ↑                    │                │              │
            └──── RETREAT ←──────┴────────────────┴──────────────┘
```

`us_admittance`가 **유일한 twist 발행자**입니다. 조작자 지령도 이 노드의 통과 모드를 거칩니다 — `TELEOP_APPROACH`에서 조작자는 z를 포함한 6축이 필요한데 z는 힘 루프 소유이기 때문입니다.

---

## 안전 envelope

| 항목 | 값 |
|---|---|
| 법선력 하드 / 소프트 | **7.0 N** / 6.0 N |
| 모멘트 한계 / policy 편향 | 0.3 / 0.1 N·m |
| 병진 / 회전 상한 | 10 mm/s / 0.2 rad/s |
| 관절속도 | 0.5 rad/s |
| 감쇠 `B_z` / `B_r` | 1000 N·s/m / 0.5 N·m·s/rad |
| setpoint 슬루 `F` / `M` | 2.0 N/s / 0.05 N·m/s |

**워치독은 2층입니다.** 후퇴는 카테시안 개념이라 관절 공간 노드가 할 수 없습니다.

```
상위  us_diff_ik    twist 0.1 s 두절  →  프로브 −z 후퇴 (5 mm/s, 상한 3 s = 15 mm)
하위  us_servo      q̇    0.1 s 두절  →  정지 (방향을 모르므로)
```

`워치독 시간 × 접근 속도 = 최악 침투 깊이`. 0.1 s × 10 mm/s = **1 mm**.

한계값이 서로 모순되면 노드가 **기동을 거부합니다** — 예컨대 `setpoint.f_normal_max_n ≠ safety.max_normal_force_n`이면 어느 쪽이 이기는지 코드를 읽어야만 알게 되기 때문입니다.

---

## 사용법

### 지각

```bash
pip install -e Unet_seg          # rus_perception 설치. ROS 노드도 이걸 임포트합니다
pytest Unet_seg/tests -q         # 350개

python Unet_seg/scripts/train.py    --config Unet_seg/configs/slim_unet_production.yaml
python Unet_seg/scripts/evaluate.py --config ...  --checkpoint ...
python Unet_seg/scripts/plot_report.py            # 검증 그림 13종
```

### 제어

```bash
cd fr5_control && colcon build --symlink-install && source install/setup.bash

# 하드웨어 없이 (mock 백엔드 + 녹화 영상)
ros2 launch fr5_launch us_phase0.launch.py perception:=true control:=true

# 실로봇
ros2 launch fr5_launch us_phase0.launch.py backend:=fairino admittance:=true teleop:=true
```

`fr5_control/fr5_control/config/probe.yaml`이 기하 · 안전 한계 · 루프 주기의 **단일 진실 공급원**입니다.
미측정 항목은 `.nan`이며, 노드는 이를 0으로 대체하지 않고 **기동을 거부**합니다 — 0으로 채우면 조용히 틀린 기하로 동작합니다.

### 문서 도식

```bash
python docs/export_figures.py    # architecture.html → SVG + 2배율 PNG 9종
```

---

## 시험

순수 로직을 ROS와 분리해 두어 하드웨어 없이 회귀를 잡습니다.

```bash
pytest Unet_seg/tests -q                                    # 350

cd fr5_control
PYTHONPATH=$PWD/fr5_control:$PWD/fr5_ik pytest \
  fr5_control/test/test_{admittance,robot_backend,force_search,supervisor}.py \
  fr5_ik/test/test_dls_solver.py -q                         # 73
```

`MockBackend`가 지령을 적분하고 합성 접촉력을 냅니다. **F/T 두절·상태 동결 같은 실패 경로를 주입할 수 있는 것**이 mock을 두는 주된 이유입니다 — 실로봇에서 일부러 일으키기 곤란한 것들입니다.

> mock은 기구학만 맞습니다. 조직 점탄성, 실제 F/T 잡음, 컨트롤러 지연은 재현하지 않습니다.
> mock 통과는 "로직이 돈다"는 뜻이지 "제어가 맞다"는 뜻이 아닙니다.

---

## 알려진 미완성

| 항목 | 영향 |
|---|---|
| 초음파 데이터셋 미도착 | 모델 학습 불가. 검증 계획은 선행 완료 |
| 프로브 마운트 CAD | 툴 변환 · wrench 기준점. **Mx/My 정렬의 전제** |
| 중력·페이로드 보상 | 프로브 자중 3–8 N은 `F_n*`와 같은 자릿수 |
| F/T 잡음 실측 | 데드밴드 값을 정할 수 없음 |
| US 프레임그래버 지연 | 50 ms 초과 시 5 Hz policy 대역 재설계 |
| `policy_node` | 모델 형태 미결 (world model / IBVS / behavior cloning) |
| 시간 동기 | 수집기가 "최신값 스냅샷" — 학습 데이터의 근본 품질 문제 |

`Q_raw`의 4개 지표는 **접촉력에 대한 단조성이 아직 검증되지 않았습니다.** 성립하지 않으면 Stage 1 전체를 재설계해야 하며, 이것이 Phase 3의 통과 기준입니다.

---

## 도식

`docs/figures/` — SVG(벡터) + PNG(2배율, 흰 배경). 흑백 인쇄에서 음영=힘 / 흰색=영상 / 회색=중립으로 구분됩니다.

| | |
|---|---|
| `01-architecture` | 전체 노드 그래프 |
| `02-axis-split` | 힘 축 vs 영상 축 |
| `03-dof-delta` | 5-DoF → 3-DoF 전환 |
| `04-force-search` | `Q̄(F)` 곡선과 최소 힘 선택 |
| `05-state-machine` | 상태 전이 |
| `06-timescale` | 루프 대역 분리 |
| `07-quality-definitions` | 두 품질함수 |
| `08-slim-unet` / `09-slim-vs-standard` | 모델 구조 |
