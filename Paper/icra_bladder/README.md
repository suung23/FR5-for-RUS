# ICRA 논문 작업 — 저장소만으로 끝까지

이 폴더는 **실험 PC 와의 연결이 끊겨도 논문을 처음부터 끝까지 쓸 수 있도록** 필요한
것을 전부 담는다. 원시 세션(`policy_learning/runs/`, 3.2 GB)은 gitignore 이고
넣을 수도 없다 — 이 저장소는 GitHub 100 MB 하드 리밋에 두 번 걸린 적이 있다
(루트 `.gitignore` 주석). 그래서 **필요한 것만 추려서** 여기로 내보낸다.

```
Paper/icra_bladder/
  main.tex            IEEEtran 두 단, 8 쪽
  refs.bib            참고문헌
  build_docx.py       LaTeX → docx (pandoc 없이)
  figures/            논문에 들어가는 그림
  data/               ← 실험 결과. export_results.py 가 만든다
    results.md          조건별·자세별 표 (그대로 옮겨 쓸 수 있는 형태)
    summary/*.csv       원본 요약표 사본
    verdicts.json       에피소드별 판정 전체 (힘·유지·찾기 지표 포함)
    frames/roi_mask.png 부채꼴 시야 마스크
    frames/<ep>/index.csv   프레임마다 t · Q_seg · 면적 · 중심 · 잘림 · 힘
    frames/<ep>/f####.png   B-mode 원본 (무손실 PNG, 256², 1 Hz)
    frames/<ep>/m####.png   같은 프레임의 U-Net 마스크 (이진)
    contact/<ep>.jpg    컨택트 시트 — 눈으로 훑어 고르는 용도
  export_results.py   실험 PC 에서 **한 번** 돌린다 (torch·runs/ 필요)
  make_figure.py      어디서나 돈다 (numpy·opencv 만 필요)
```

## 노트북에서 하는 일 (실험 PC 불필요)

### 1. 프레임 고르기

먼저 `data/contact/<에피소드>.jpg` 를 이미지 뷰어로 연다. 타일마다 **번호 · 시각 ·
Q_seg** 가 찍혀 있다. 쓸 만한 것의 번호를 적어 둔다.

숫자로 고르고 싶으면:

```bash
python3 Paper/icra_bladder/make_figure.py --list                        # 에피소드 목록
python3 Paper/icra_bladder/make_figure.py --list ep0010_pose04_rule --best 8
```

### 2. 그림 만들기

```bash
python3 Paper/icra_bladder/make_figure.py \
    --pick ep0010_pose04_rule:0 ep0010_pose04_rule:60 ep0012_pose04_hold:60 \
    --labels "rule t=0 s" "rule t=60 s" "hold t=60 s" \
    --overlay --roi --out Paper/icra_bladder/figures/fig_rule_vs_hold.png
```

`--overlay` 는 U-Net 마스크 윤곽(빨강), `--roi` 는 부채꼴 시야 경계(주황)를 그린다.
마스크가 PNG 로 함께 나가 있어서 **torch 없이** 겹쳐 그릴 수 있다.

### 3. 표 옮기기

`data/results.md` 를 그대로 쓴다. 숫자를 다시 확인하고 싶으면 `data/verdicts.json`
에 에피소드별 판정이 전부 들어 있다.

### 4. 원고 빌드

```bash
python3 Paper/icra_bladder/build_docx.py        # → ICRA_draft.docx
```

## 실험 PC 에서 하는 일

새 세션을 돌린 뒤 **한 번만**:

```bash
Unet_seg/.venv/bin/python3 Paper/icra_bladder/export_results.py \
    --run policy_learning/runs/<세션>
git add Paper/icra_bladder/data && git commit && git push
```

`--no-frames` 를 주면 표만 다시 만든다 (빠르다). `--hz` 로 프레임 밀도를 올릴 수
있는데, **1 Hz 가 43 MB · 2 Hz 가 86 MB** 라는 것을 알고 올릴 것.

## 이번 회차(main_rule)를 읽을 때 주의할 것

- **자세 5 × 조건 3 = 15 에피소드.** 사전 계획은 8 자세 24 에피소드였고 5 자세에서
  종료했다. 축소한 사실을 원고에 밝힐 것.
- **짝 5 개로는 p < 0.05 가 구조적으로 불가능하다** (전승이어도 최소 p = 0.031,
  실제 4승이라 p = 0.0625). 유의성이 아니라 효과 크기와 완전성으로 서술할 것.
- **힘 한계 초과가 있었다.** rule 5.07 N · search 5.15 N 으로 5 N 선을 넘었고 그
  위에 1.2 s · 0.8 s 머물렀다. 후퇴 로직이 되돌렸지만 그대로 보고할 것.
- **학습된 것은 U-Net 하나다.** 제어 계수는 로봇이 자기를 60 s 재서 얻었다
  (`rus_policy.axis_probe`). 이 5/5 는 플랫폼·지각·힘 포락선의 증명이지 자유손
  데이터 수집론의 증명이 아니다.
- **Q_seg 눈금이 이 회차 직전에 바뀌었다** (게이트 × 내용, `AREA_FULL` 0.12,
  합격선 0.977). 이전 회차의 Q_seg 숫자와 직접 비교하지 말 것 — 근거는
  `policy_learning/rus_policy/view_quality.py` 와 `episode.py` 주석에 있다.

## 다음 회차 (Sonologger 면내 학습 검증)

같은 프로토콜로 한 번 더 돌려 비교한다. **비교 가능성을 지키려면 고정해야 할 것**:

U-Net 체크포인트 (`phantom_dyn_r2`, id `9e8ef542d833`) · `AREA_FULL` 0.12 ·
`q_view_min` 0.977 · `q_good_enough` 0.85 · 판정 세 조건 (0.04 / 0.80 / 0.30) ·
팬텀 충전 상태 · 영상 깊이 설정 · ROI 부채꼴 마스크.

팬텀 충전을 바꾸면 `AREA_FULL` 이 무효가 된다 — 2026-09-14 에 실제로 밟은 함정이다.

조건 구성은 `sonologger` 가 `rule` 을 **대체하지 말고 나란히** 들어가야 한다.
`hold` 는 두 회차 공통 대조군이라 반드시 유지. 자세는 6 개 이상으로 잡으면 전승 시
p = 0.031 까지 내려간다.
