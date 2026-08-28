# GT 라벨 판정 루프

판정 → 필터 → 학습 → 평가를 한 줄로 잇는다. `batch_A/` `batch_B/` 의 정적 HTML 을
대체한다 — 그쪽은 라디오 버튼이 저장되지 않아 `verdicts.csv` 를 손으로 채워야 했다.

## 1. 판정

```bash
cd ~/FR5-for-RUS/Unet_seg
.venv/bin/python experiments/label_review/review/server.py
# → http://127.0.0.1:8778
```

미판정 82명이 큐에 오른다. 키 한 번이 곧 디스크 기록이므로 아무 때나 Ctrl+C 로 끊고
다시 켜면 그 지점부터 이어진다.

| 키 | 동작 |
|---|---|
| `1` `2` `3` `4` | label_ok / partial / label_wrong / unjudgeable — 기록하고 다음 환자로 |
| `←` `→` | 이동 (판정 없이) |
| `u` | 판정 취소 |
| `n` | 메모 — 왜 그렇게 판정했는지. 재검토할 때 이게 근거가 된다 |
| `s` | 다음 미판정 환자로 건너뛰기 |
| `?` | 분류 기준 다시 보기 |

시트(PNG)는 `batch_A/` `batch_B/` 에 있는 48명 분을 재사용하고, 시트조차 없던 34명은
백그라운드로 미리 렌더링한다. 환자당 약 1.6초.

옵션: `--all` 기존 판정도 재확인 · `--patients P022 P025` 특정 환자만 · `--port`

## 2. 분류 기준

**판정 대상은 라벨이지 모델이 아니다.** 화면의 Dice 는 어느 프레임을 특히 볼지 고르는
참고값일 뿐이다. Dice 가 낮다고 라벨이 틀린 것도, 높다고 맞는 것도 아니다 — 모델이
잘못된 라벨을 그대로 재현하고 있던 사례가 이미 있다(P097: 원본 GT 0.850, 수정 GT 0.785).
train 환자는 Dice 를 아예 표시하지 않는다. 학습 데이터에 대한 점수는 암기를 재는 것이라
판정 근거가 될 수 없기 때문이다.

기준은 하나다: **초록 윤곽이 무에코(검은) 방광 내강 위에 놓여 있는가.**

| 판정 | 뜻 | 학습 |
|---|---|---|
| `label_ok` | 윤곽이 영상 구조를 따른다. 경계가 한두 픽셀 헐거운 건 괜찮다 | **씀** |
| `partial` | 대체로 맞지만 일부 구간에서 밝은 조직을 삼켰다 | **씀** (손보기 후보로 표시) |
| `label_wrong` | 윤곽이 영상 구조를 아예 따르지 않는다 | 제외 |
| `unjudgeable` | 사람이 봐도 판단 불가 | 제외 |

`label_wrong` 의 구체적 형태 — 하나만 해당해도 여기다:

* 방광 추정부와 비방광부가 한 라벨에 섞여 있다 (P043)
* 균일 speckle 위에 얹힌 평면 다각형이다 (P021)
* 프레임이 지나도 윤곽이 거의 안 움직인다 — 복사된 라벨
* 영상 자체에 무에코 방광이 안 보인다

`partial` 을 버리지 않는 이유: 경계가 헐거운 것과 라벨이 틀린 것은 다른 문제다.
전자는 손으로 고칠 수 있고(`mask_editor/`), 후자는 고칠 근거가 없다.

`unjudgeable` 은 "틀렸다"가 아니라 "근거 없이 쓸 수 없다"는 뜻이다 (P000: 용적이 작아
사람도 판단 불가).

확대 시트는 **경계**를, 전체 시트는 **해부 위치와 영상 품질**을 본다. 둘이 어긋나면
— 경계는 예쁜데 위치가 엉뚱하면 — `label_wrong` 이다.

망설여지면 `n` 으로 이유를 남기고 `partial` 로 두라. 나중에 다시 볼 수 있다.

### 왜 눈으로 하는가

자동 라벨품질 스크리닝은 이 데이터셋에서 판별력이 없었다 (**AUC 0.45~0.49**).
5가지 통계를 시도했고 전부 시각적으로 멀쩡한 환자를 P021 아래로 랭크했다:
boundary/background gradient ratio (P048 1.01 vs P021 0.93) · interior-to-ring contrast
(0.058 vs 0.046) · GT-to-anechoic-component overlap (P020 0.061, P012 0.030 vs 0.050) ·
label-to-image motion ratio (P020 1.384 vs 1.332) · interior-brighter-than-surround
(얇은 마스크를 침식하면 lumen 대신 방광벽을 샘플링해 P000·P019 를 오탐).
어느 것에 게이트를 걸어도 멀쩡한 환자가 먼저 떨어진다.

## 3. 필터 + 학습 + 평가

```bash
experiments/label_review/review/run_loop.sh
```

`verdicts.csv` 의 `label_wrong` + `unjudgeable` 을 제외하고, hydro 면적 필터(0.022888)를
**전 split 에 일관되게** 적용한 뒤 학습하고 test·val 을 평가한다. 마지막에 이전 런과의
비교표를 찍는다.

| 옵션 | 뜻 |
|---|---|
| (없음) | 전 필터, 전 split 일관 적용 — 배치 도메인(HoLEP 관류 확장 방광) 기준 |
| `--protect-test` | test 는 건드리지 않는다. 이전 런들과 헤드라인을 비교할 때 |
| `--no-hydro` | 면적 필터 없이. 같은 코호트에서 hydro 는 −0.009 였으므로 이쪽이 Dice 는 낫다 |
| `NAME=` `EPOCHS=` | 런 이름 / epoch 수 |

산출물: `manifest_<NAME>.csv` · `configs/pfus_bladder_<NAME>.yaml` ·
`checkpoints/pfus_bladder_<NAME>/` · `runs/<NAME>/{best,last}_{test,val}/`

**코호트가 달라지면 Dice 를 직접 비교하지 말 것.** 환자를 빼면 예측이 하나도 안 바뀌어도
헤드라인이 오른다. 비교표가 환자 수를 함께 찍는 이유다.

## 4. 알아둘 것

* 원본 마스크와 `manifest.csv` 는 어느 단계에서도 수정되지 않는다. 쓰는 것은
  `review/verdicts.csv` 와 새 매니페스트뿐이다.
* 기존 판정 28건은 `../VERDICTS.csv` 에서 자동으로 이월된다. 최초 실행 시 한 번만.
* 40 epoch 중 30여 epoch 이 과적합이다. `val_dice` 최고점은 epoch 1~9 에서 잡히고
  어느 체크포인트가 뽑히느냐가 테스트 점수를 좌우한다. 그래서 `best.pt` 와 `last.pt` 를
  둘 다 평가한다. 조기 종료(patience 10)와 다중 시드 평균이 들어가기 전에는
  ±0.03 미만의 데이터 실험 결과를 믿지 말 것.
