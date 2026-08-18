# 검증 계획 — 방광 내강 세그멘테이션과 두 품질함수를 얼마나 믿을 수 있는가

> 데이터가 도착하면 그대로 실행할 수 있도록 만든 계획. 지표와 그림은 이미 구현되어 있고,
> 여기에는 **무엇을 측정하는가**와 **어느 값이면 통과인가**만 적는다.
> 최종 갱신: 2026-08-17

---

## 0. 이 문서가 답하려는 질문

> 품질함수 1(`Q_raw`)·2(`Q_seg`)가 방광 내강을 얼마나 정확하게 잡아내고,
> **우리가 그것을 얼마나 믿을 수 있는가.**

이 질문은 두 개다. 그리고 두 번째가 더 중요하다.

| | 질문 | 지표 | 모듈 |
|---|---|---|---|
| **정확도** | 마스크가 맞았는가 | Dice, IoU, HD95 | `metrics/spatial.py` (기존) |
| **신뢰도** | **틀렸을 때 그것을 아는가** | rho, AUROC, trusted-bad, AURC | `metrics/trust.py` (신규) |

정확도만으로는 두 시스템을 구별할 수 없다:

- Dice가 보통이지만 품질 점수가 정직한 모델 → **안전하다.** 나쁜 프레임을 컨트롤러가 거른다.
- Dice 평균이 훌륭하지만 자기 실패를 점수로 드러내지 못하는 모델 → **위험하다.**
  컨트롤러가 틀린 마스크 위에서 아무 경고 없이 움직인다.

평균 Dice는 이 둘에 같은 점수를 준다. `metrics/trust.py`는 다르게 준다.
**그래서 이 계획의 중심은 정확도가 아니라 신뢰도다.**

---

## 1. 선행 조건 — 측정 전에 확정해야 하는 것

| # | 항목 | 왜 먼저인가 | 상태 |
|---|---|---|---|
| 1 | **US 영상 기하** (프로브 종류·깊이 스케일·부채꼴 ROI) | `control.roi`가 `full`인 채로 측정하면 `mask_area_ratio`의 분모가 프레임그래버 크롭이 되고, `segmentation_confidence`는 빔 바깥 검은 영역이 지배한다. 두 값 모두 게이트 입력이다 | ⏳ |
| 2 | **`accuracy_floor`** — "행동해도 되는" Dice 하한 | 아래 모든 판정이 이 값에 걸린다. **측정이 아니라 결정이다.** 마스크가 얼마나 틀려도 프로브가 엉뚱하게 움직이지 않는가에서 온다 | ❓ 기본 0.70은 자리표시 |
| 3 | **`target_bad_rate`** — 안전 예산 | "신뢰한 프레임 중 실제로 나빴던 비율"의 상한. 임상 위험 판단이지 튜닝 파라미터가 아니다 | ❓ 기본 2% |
| 4 | **환자 단위 데이터셋 + 라벨** | 프레임 단위 분할은 같은 환자의 해부를 학습셋에서 검증셋으로 흘려 모든 수치를 부풀린다. 리포가 `PatientLeakageError`로 막고 있다. **분할은 7:3으로 설정 완료** (§2) | ⏳ `data/` 비어 있음 |
| 5 | **mm/pixel** | 제어 경로에 물리 스케일이 없다. 정규화 좌표만으로는 프로브를 몇 mm 옮길지 정할 수 없다 | ⏳ probe-to-image 캘리브 산출물 |

> 1·2·3이 정해지기 전의 모든 수치는 **참고값**이다. 특히 2를 나중에 바꾸면
> §4의 모든 합격/불합격이 뒤집힌다.

---

## 2. 실행 절차 — 데이터가 도착하면 이 순서 그대로

분할은 **학습 : 검증 = 7 : 3**으로 이미 설정되어 있다 (`configs/_base.yaml`).
비율은 프레임이 아니라 **환자 수**에 걸린다. 모든 config가 `_base.yaml`을 상속하므로
바꿀 곳은 한 군데뿐이고, `tests/test_split_config.py`가 그 설정을 고정한다.

```bash
# 0. 프리플라이트 — 학습에 시간을 쓰기 전에 데이터가 성립하는지 확인
python scripts/check_dataset.py --manifest data/bladder/manifest.csv
#    manifest 구조 · 파일 존재 · 시퀀스 순서 · 환자 수 · 분할/누수 · 라벨 커버리지
#    종료 코드 0 이면 학습 가능. --write-split 로 분할을 manifest 에 고정할 수 있다.

# 1. 플로우 사전계산 (학습 때마다 dense flow 를 다시 뽑지 않기 위해)
python scripts/precompute_flow.py --manifest data/bladder/manifest.csv \
    --backend farneback --output-dir data/bladder/flow \
    --update-manifest --with-reliability

# 2. 세 조건을 같은 분할로 학습 (seed 가 같으므로 자동으로 같은 분할)
python scripts/train.py --config configs/standard_unet_baseline.yaml
python scripts/train.py --config configs/slim_unet_paper.yaml
python scripts/train.py --config configs/slim_unet_temporal.yaml

# 3. 평가 — control_states.jsonl + frame_metrics.csv 를 남긴다
#    evaluation.split: val (test 가 0 이므로)
python scripts/evaluate.py --config configs/slim_unet_temporal.yaml \
    --checkpoint checkpoints/slim_unet_temporal/best.pt

# 4. 보고서 — 그림 13종 + trust_report.json
python scripts/plot_report.py --run-dir runs/.../eval \
    --accuracy-floor 0.70 --target-bad-rate 0.02
```

`plot_report.py`는 `history.json`을 학습 출력 디렉토리에서 찾는다. 평가 디렉토리와
다르면 복사하거나 `--history`로 직접 지정한다.

### 분할에 대한 단서 조항 ⚠️

**시험셋이 없다.** `ratios: [0.70, 0.30, 0.00]`은 의도된 결정이지만 대가가 있다:
§4의 운용점(Q 임계), `accuracy_floor`, `target_bad_rate`는 **전부 검증셋에서 고르는 값**이고,
같은 셋에서 성능을 보고한다. 따라서 신뢰도 수치는 **낙관적으로 나온다.**

지금 단계(임계 탐색·설계 검증)에서는 합리적이다. 환자가 충분해지면 되돌릴 것:

```yaml
split:      { ratios: [0.70, 0.15, 0.15] }
evaluation: { split: test }
```

두 키는 **함께** 움직여야 한다. `test`가 0인데 `evaluation.split: test`면
`evaluate.py`가 "Split 'test' is empty"로 멈춘다 —
`tests/test_split_config.py`가 이 조합을 막는다.

**환자가 적으면 7:3은 거칠다.** 환자 10명 → train 7 / val 3. 검증 지표가 사실상
3명에서 나오므로 오차 막대가 매우 넓다. `check_dataset.py`가 8명 미만이면 경고한다.

`Q_raw`의 힘 응답만 로봇에서 온다. Stage 1 탐색이 힘 레벨마다 홀드 윈도우 전체 샘플을
`{"3.0": [q, ...], "3.5": [...]}` 형식으로 남기면 된다:

```bash
python scripts/plot_report.py --run-dir runs/... --force-log runs/.../force_log.json
```

---

## 3. 그림 13종 — 무엇을 보는 그림인가

### A. 학습 (`history.json`)

| 그림 | 형태 | 이 그림이 잡아내는 실패 |
|---|---|---|
| **A1** loss decomposition | 2단 패널 (공간/시간), 램프 음영 | 한 항만 줄고 나머지가 정체 — 가중치가 잘못 걸린 경우 |
| **A2** validation curves | 3선 + 선택 에포크 표시 | 모델 선택이 엉뚱한 에포크를 고르는 조용한 버그 |
| **A3** temporal diagnostics | 신뢰 화소 비율 + 폐기 쌍 수 | **플로우가 죽어 시간축 손실이 아무것도 감독하지 않게 된 상태.** 학습은 계속되고 곡선은 멀쩡해 보인다 |
| **A4** optimization health | grad norm·LR (로그, 2단) | 비유한 배치가 조용히 건너뛰어진 에포크 |

### B. 신뢰도 (`control_states.jsonl` + `frame_metrics.csv`)

| 그림 | 형태 | 읽는 법 |
|---|---|---|
| **B1** Q vs Dice ★ | 산점도, 4사분면 | **좌상단 붉은 영역의 점 개수가 이 검증의 존재 이유다.** 품질함수가 보증했는데 마스크가 틀린 프레임 |
| **B2** risk–coverage ★ | 곡선 + 운용점 | "프레임의 몇 %에서 행동할 수 있고, 그때 얼마나 틀리는가" |
| **B3** reliability | 구간별 Q vs 실제 Dice | 대각선과의 거리보다 **단조성**이 중요하다 |
| **B4** ROC + 게이트 점 | 곡선 + 마름모 | 곡선은 높은데 게이트 점이 아래면 **임계값 문제**(설정 변경). 곡선이 평평하면 **가중치 문제** |
| **B5** component attribution ★ | 발산 막대 + 점 | **0 왼쪽 막대 = 그 항을 빼면 Q가 좋아진다.** 가중치를 0으로 |
| **B6** reason codes | 덤벨 | 두 점이 겹치면 그 코드는 아무것도 구별하지 못한다 = 공짜로 일어나는 상태 전이 |
| **B7** per patient | 스트립 + 중앙값 | 4명 성공 1명 실패를 평균이 가린다 |
| **B8** Q_raw force response ★ | 오차막대 + F\*/argmax | **Stage 1이 성립하는지 아닌지를 정하는 그림** |
| **B9** latency | 히스토그램 + 예산선 | 평균이 아니라 **꼬리**가 예산을 넘는지 |

★ = 결론이 바뀌는 그림.

---

## 4. 합격 기준

> 아래 임계는 **제안값이다.** 특히 `accuracy_floor`와 `target_bad_rate`는 §1-2, §1-3에서
> 확정한 뒤 여기 숫자를 갱신해야 한다. 지금 값은 "이 정도면 다음 단계로 간다"는 공학적 출발점이지
> 검증된 기준이 아니다.

### 게이트 1 — 세그멘테이션이 쓸 만한가

| 지표 | 통과 | 경계 | 불합격 |
|---|---|---|---|
| 환자별 중앙 Dice | ≥ 0.85 | 0.75–0.85 | < 0.75 |
| 환자별 중앙 **IoU** | ≥ 0.739 | 0.600–0.739 | < 0.600 |
| **최악 환자** 중앙 Dice | ≥ 0.70 | 0.60–0.70 | < 0.60 |
| **최악 환자** 중앙 **IoU** | ≥ 0.538 | 0.429–0.538 | < 0.429 |
| `missed_bladder_rate` | ≤ 2% | 2–5% | > 5% |
| `false_positive_on_empty` | ≤ 2% | 2–5% | > 5% |
| HD95 | ⏳ mm 단위 기준은 `pixel_spacing` 확정 후 | | |

최악 환자 기준이 따로 있는 이유는 B7이 보여주는 그대로다. 평균은 한 명의 실패를 흡수한다.

### Dice와 IoU를 둘 다 적는 이유

**프레임 단위로는 완전히 동치다.** `IoU = Dice/(2−Dice)`는 순증가 함수이므로 두 지표는
프레임을 똑같은 순서로 세운다. 그래서 게이트 2의 순위 기반 지표(ρ, AUROC, AURC,
risk–coverage)는 **어느 쪽을 써도 소수점까지 같은 값이 나온다.** 실제로 확인했다.
`plot_report.py --accuracy-key`가 둘 다 받지만, 바꿔도 결론이 달라지지 않는다.

**바뀌는 것은 임계값 하나뿐이고, 그게 함정이다.** Dice 0.70 = IoU 0.538이다.
`--accuracy-key iou`로 바꾸면서 floor를 0.70으로 두면 훨씬 엄격한 질문이 되고
trusted-bad가 대략 두 배로 뛴다. `plot_report.py`가 매 실행마다 두 단위를 모두 찍고,
IoU에 Dice스러운 값이 들어오면 경고한다. 변환은 `metrics.trust.dice_to_iou` /
`iou_to_dice`.

**그러나 프레임을 평균 내는 순간 동치가 깨진다.** `d ↦ d/(2−d)`는 볼록이라
평균 Dice가 같아도 **분산이 큰 쪽이 평균 IoU가 높다.** 실제 예:

| 모델 | 프레임 | 평균 Dice | 평균 IoU |
|---|---|---|---|
| A | 7개 완벽(1.00) + 3개 완전 실패(0.00) | 0.700 | **0.700** |
| B | 10개 모두 0.72 | **0.720** | 0.563 |

**Dice는 B를, IoU는 A를 고른다.** A의 실패 양상(대부분 완벽 + 소수 완전 실패)이
바로 이 시스템이 내는 방광 소실 실패다. 그래서 둘 다 보고하고,
**정작 판정은 평균이 아니라 프레임 단위 floor + trusted-bad로 한다** (게이트 2).

### 게이트 2 — 품질함수를 믿을 수 있는가 ★ 핵심

| 지표 | 통과 | 경계 | 불합격 | 불합격이면 |
|---|---|---|---|---|
| Spearman rho (Q_seg, Dice) | ≥ 0.60 | 0.40–0.60 | < 0.40 | **가중치를 다시 뽑는다** (B5) |
| AUROC | ≥ 0.85 | 0.75–0.85 | < 0.75 | 임계값이 아니라 항 구성이 문제 |
| **trusted-bad rate @ 현행 게이트** | ≤ 2% | 2–5% | > 5% | 게이트 임계 재설정 (B2 운용점) |
| coverage @ trusted-bad ≤ 2% | ≥ 60% | 40–60% | < 40% | 쓸 수는 있으나 처리량이 절반 이하 |
| AURC | ≤ 0.10 | 0.10–0.20 | > 0.20 | |
| calibration `monotone_fraction` | = 1.0 | ≥ 0.8 | < 0.8 | **재보정으로 고쳐지지 않는다.** 항 구성 문제 |

> **`operating_point`가 `feasible=False`를 돌려주면 그것이 결론이다.**
> 어떤 임계값도 안전 예산을 만족하지 못한다는 뜻이고, 차선 임계를 대신 쓰면
> 아무도 하지 않은 안전 주장을 하는 것이 된다.

### 게이트 3 — 이유 코드가 의미가 있는가

| 지표 | 통과 | 불합격이면 |
|---|---|---|
| 각 코드의 `lift` | ≥ 1.3 | lift ≈ 1.0인 코드는 **비활성화한다.** 그 코드에 걸린 supervisor 복구 동작이 공짜로 일어나고 있다 |
| 활성 코드 중 `fire_rate` > 50%인 것 | 없어야 함 | 상시 발화하는 게이트는 게이트가 아니다 |

### 게이트 4 — `Q_raw`의 힘 응답 ★ Stage 1의 존폐

`Q_raw`에는 정답 라벨이 없다. "잘 결합되었다"는 주석이 존재하지 않는다.
따라서 검증은 비교가 아니라 **구조 검사**다.

| 지표 | 통과 | 불합격이면 |
|---|---|---|
| `is_unimodal` | True | **Stage 1 힘 탐색 전체를 재설계한다** |
| `sign_changes` | ≤ 1 | 위와 같음 |
| \|rho vs F\| (상승 구간) | ≥ 0.7 | 응답이 평평 = 최적화할 대상이 없음 |
| `f_star` 재현성 (동일 지점 3회) | 표준편차 ≤ 0.5 N | 탐색이 잡음을 오르고 있음 |
| 하위점수 4개 각각의 단조/단봉성 | 4개 중 ≥ 3개 | 실패한 항의 가중치를 0으로 |

> 이 게이트가 불합격이면 §6.2와 §7의 힘 탐색 설계가 무너진다.
> **가정이 아니라 실험이고, Phase 4 이전에 반드시 끝나야 한다.**

### 게이트 5 — 지연

| 지표 | 통과 | 비고 |
|---|---|---|
| end-to-end p95 | ≤ 33.3 ms | 30 Hz 프레임 예산 |
| U-Net 추론 p95 | ≤ 15 ms | |
| **옵티컬 플로우 계산 p95** | ≤ 10 ms | ⚠️ 현재 지연 예산에 미반영 (DESIGN_NOTES §14.3) |

평균이 아니라 **p95**로 판정한다. 제어 루프는 늦은 프레임을 놓치지 평균 프레임을 놓치지 않는다.

---

## 5. 결과별 다음 행동

| 결과 | 해석 | 행동 |
|---|---|---|
| 게이트 1 통과, 2 통과 | 정상 | Phase 4로 |
| 게이트 1 통과, 2 불합격 (rho 낮음) | 마스크는 맞는데 **자기가 언제 틀리는지 모른다** | B5로 가중치 재도출 → 재평가. 그 전에 폐루프 금지 |
| 게이트 1 불합격, 2 통과 | 마스크는 나쁘지만 **정직하다** | 데이터·학습 문제. 게이트가 보호하므로 안전하게 반복 가능 |
| 둘 다 불합격 | | 아키텍처 이전에 데이터·라벨 품질부터 |
| 게이트 4 불합격 | Stage 1 무효 | 힘 탐색 재설계. `Q_raw` 지표 교체 또는 다른 탐색 변수 |
| `monotone_fraction` < 0.8 | Q가 오도한다 | **재보정으로 안 고쳐진다.** 항 구성 재검토 |

---

## 6. 아직 이 계획에 없는 것

정직하게 적어둔다.

1. **확률 보정(temperature scaling)** — `segmentation_confidence`는 보정된 정확도가 아닌데
   게이트가 임계를 건다. B3가 그 어긋남을 *측정*하지만 *고치지는* 않는다.
2. **불확실성 추정** — 단일 결정론적 forward뿐. 인코더에 dropout이 있으므로 MC Dropout이
   거의 공짜지만, 30 Hz에서 N회는 비싸다. Stage 1처럼 느린 구간만 켜는 선택적 경로가 후보.
3. **관측자 간 변이** — 방광 경계는 본질적으로 모호하다. 라벨러 2명의 Dice 상한을 모르면
   0.85가 좋은 값인지 판단할 수 없다. **가능하면 일부 프레임을 이중 라벨링**할 것.
4. **폐루프 검증** — 위 전부는 개루프 지각 지표다. 실제 컨트롤러·실제 지연·실제 안전
   감시 하의 폐루프 시험을 대체하지 않는다.
5. **다기관 / 다장비** — 도메인 시프트는 US에서 가장 흔한 실패다. 한 장비 데이터로는
   배포 주장을 할 수 없다.

---

## 부록: 구현 위치

| 대상 | 위치 |
|---|---|
| 신뢰도 지표 | `Unet_seg/rus_perception/metrics/trust.py` |
| 지표 테스트 (39개) | `Unet_seg/tests/test_trust.py` |
| 그림 | `Unet_seg/rus_perception/reporting/plots.py` |
| CLI | `Unet_seg/scripts/plot_report.py` |
| ROI 정의 | `Unet_seg/rus_perception/control/roi.py` · 설정 `control.roi` |
| 분할 (7:3) | `Unet_seg/configs/_base.yaml` `split:` · 고정 테스트 `tests/test_split_config.py` |
| 데이터 프리플라이트 | `Unet_seg/scripts/check_dataset.py` |
| 정확도 지표 (기존) | `Unet_seg/rus_perception/metrics/spatial.py` · `temporal.py` |

그림 색은 `docs/architecture.html`과 같은 규약을 따른다 — 따뜻한 쪽이 힘(`Q_raw`),
차가운 쪽이 영상(`Q_seg`). 명도로도 갈라 두어 흑백 인쇄에서 구분된다.
축 라벨만 영문인 이유는 matplotlib이 한글 글리프에 폰트 파일을 요구하고, 학습 머신에
그 폰트가 있으리라 보장할 수 없기 때문이다 — 생성한 자리에서 네모로 깨지는 그림보다
영문 기술 용어가 낫다.
