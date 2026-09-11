# 실전 런북 — 이 문서만 보고 실험한다

2026-09-11. 왜 그런지는 `PROTOCOL_EXPERIMENT.md`(실행) · `EVAL_PLAN_POLICY_RESCUE.md`(사전
등록) · `RUNBOOK_POLICY_INFERENCE.md`(시스템)에 있다. 여기는 **칠 것과 볼 것**만 담는다.

---

## A. 시작 전 한 번 (코드가 바뀐 날)

```bash
source ~/FR5-for-RUS/env.sh
cd ~/FR5-for-RUS
colcon build --packages-select fr5_control fr5_ik --symlink-install
source ~/FR5-for-RUS/env.sh
cd ~/FR5-for-RUS/teleop_gui && npm run build        # GUI 를 고쳤다면
```

`환경 OK — rclpy · torch …` 가 안 뜨면 그 터미널에서는 아무것도 하지 않는다.

## B. 힘 영점 — **매 실행 직전에 잡는다**

닿기 전에 ‖F‖ 가 1~2 N 씩 읽히는 것은 고장이 아니다. 이 장비의 보상 잔차를 자세 75 개에서
실측한 값(2026-09-11):

| | 크기 | 영점으로 |
|---|---|---|
| 상수 성분 | **0.83 N** (거의 전부 −z) | 지워진다 |
| 자세 따라 도는 성분 | **0.42 N** 중앙, 최대 1.2 N | 안 지워진다 |

그래서 **영점을 잡으면 대부분 사라진다.** 남는 0.4 N 남짓은 3 N 목표와 5 N 한계에
비해 작다. 다만 영점은 상수를 빼는 것이므로 **잰 자세에서 멀어지면 뺀 만큼이 오차로
돌아온다** — 영점 자세에서 ±30° 안에 머무는 동안은 0.24~0.38 N, 90° 넘게 눕히면 0.9 N.

**따라서 순서가 중요하다:**

1. 로봇을 **실제 시작 자세**(프로브가 팬텀을 향해 거의 수직 아래)로 옮긴다
2. 아무것도 **닿지 않은 상태**에서
3. GUI Calibration 의 **작업 자세 영점**을 누른다
4. 그 다음에 접근·접촉한다

미리 잡아 둔 영점을 재사용하지 마라. 시작 자세가 달라지면 그만큼이 그대로 오차다.

### 그래도 이상하면

```bash
~/FR5-for-RUS/scripts/start_session.sh --calib --no-us --no-ap     # 다른 창에서 (제어 스택 없이)
python3 ~/FR5-for-RUS/policy_learning/scripts/diag_wrench_poses.py   # 공중에서 자세 3~5 곳, Enter
```

잔차를 **영점이 지우는 몫**과 **남는 몫**으로 쪼개 인쇄한다. 남는 몫이

- **0.5 N 미만** → 영점 잡고 진행 (정상)
- **0.5~1.0 N** → 진행하되 회전 범위를 영점 자세 ±30° 로 유지
- **1.0 N 이상** → 영점 문제가 아니다. 중력 모델이 이 자세 영역을 설명하지 못한다:

```bash
python3 ~/FR5-for-RUS/phantom_stiffness/refit_calibration.py \
        --count 75 --down-cone-deg 60 --min-poses 12 --dry-run
```

⚠️ 재적합할 때 **자세를 많이 써라.** 자세가 적으면 적합이 그 자세들에만 맞고 rms 는
좋아 보이지만 다른 자세에서 두 배로 틀어진다 — 지금 쓰는 6 자세 프로파일이 그 예다
(자칭 rms 0.19 N, 교차검증 0.90 N). `--count` 를 키울 때는 스크립트가 규약이 섞였는지
검사해 거부하므로, 거부당하면 값을 줄여 한 세션 안에 머물러라.

## C. 세션

```bash
cd ~/FR5-for-RUS
~/FR5-for-RUS/scripts/start_session.sh
```

`--seg` 를 붙이지 않는다 (러너가 자기 U-Net 을 돌린다).

**GUI 확인** — 상단이 `NO TELEMETRY` 가 아니고, **Monitoring** 하단 오른쪽에 품질 두 눈금과
`Policy inference` 버튼이 보인다. 안 보이면 Electron 창을 새로 띄운다 (`npm run console`).

## D. 평가 — 한 명령

| 무엇을 | 명령 |
|---|---|
| 배선·축 확인 (지령 없음) | `./scripts/start_policy_eval.sh --dry` |
| 자유 루프 평가 | `./scripts/start_policy_eval.sh` |
| **파일럿** 6 자세 × 3 조건 = 18 | `./scripts/start_policy_eval.sh --pilot` |
| **본 시험** N 자세 × 4 조건 | `./scripts/start_policy_eval.sh --main N` |

셋 다 힘 탐색을 함께 띄운다. **Ctrl-C 하나로 둘 다 내린다.** 힘 탐색을 따로 띄우지
않는다 — 빠지면 힘 설정값이 출발값에 고정되고 그 에피소드는 분리 보고 대상이 된다.

파일럿·본시험은 `policy_learning/runs/pilot` · `runs/main` 으로 간다. **끊기면 같은 명령을 다시 친다**
(이어서 한다). 본 시험의 `N` 은 파일럿 분석이 알려준다:

```bash
python3 ~/FR5-for-RUS/policy_learning/scripts/analyze_experiment.py \
        ~/FR5-for-RUS/policy_learning/runs/pilot
```

처음 한 번은 `--dry` 로 확인한다:

| 확인 | 기대 |
|---|---|
| `B-mode 변환:` | `scan_convert+letterbox` |
| `지각 XX ms/장` | < 125 |
| `ω=(…)°/s` | 값이 나온다 |
| 힘 탐색 로그 | 2.5 s 마다 설정값 부호가 뒤집힌다 |

## E. 시도 하나 (반복)

```
1. teleop 으로 접근·접촉        Touch 데드맨을 쥔다
2. 손을 놓는다
3. GUI  Start inference         ← 여기서 teleop 이 사라진다. 접촉이 먼저다
4. 러너가 알린다                ○ 진단 가능 뷰 도달 — 시작 후 23 s
5. GUI  Stop                    ← 시도가 닫히고 기록된다
                                시도 3  ○ 성공  최장 연속 5.2 s  → attempt_003
                                  누적 2/3
6. 1 번으로
```

Ctrl-C 로 끝내면 시도별 표가 나온다. 기록은 `policy_learning/runs/eval_<시각>/`.

## F. 러너가 조용하면 — 5 s 마다 이유를 찍는다

| 메시지 | 할 일 |
|---|---|
| `렌치가 중력 보상되지 않았다` | B 로 |
| `접촉 부족 ‖F‖=…` | teleop 으로 닿는다 |
| `정책이 꺼져 있다` | GUI Start inference |
| `힘 탐색이 보이지 않는다` | `start_policy_eval.sh` 로 띄웠는지 확인 |
| `유효한 관측 프레임이 없다` | 영상이 끊겼다 — 프로브 전원·AP |

## G. 멈춰야 할 때

콘솔 `Stop` → Touch 데드맨 놓기(워치독 후퇴) → 터미널 Ctrl-C → `./scripts/stop_all.sh`

## H. 알아 둘 것 넷

1. **접촉이 먼저다.** `Start inference` 를 누르면 조작자 축이 0 이 되어 접촉을 만들 수단이 없다.
2. **`Stop` 을 눌러야** régime 이 풀리고 Touch 가 돌아온다. 다음 자세로 가려면 반드시.
3. **버튼 전 teleop 구간에는 접촉 시 자동 감속이 없다.** 5 N 후퇴가 유일한 보호다.
4. **힘은 3 N 고정이 아니다.** Q_raw 극값 탐색이 설정값을 옮긴다. 힘 탐색이 안 떠 있으면
   출발값에 고정되고, 그 시도는 나중에 **분리해서 보고**해야 한다.

## I. 오늘의 설정

| | |
|---|---|
| 체크포인트 | `policy_learning/runs/qres2_ep25.pt` (6 자유도, 잔차 Q̂, λ=100, ep25) |
| 정책 축 | 회전만 (`--axes rot`), 3 °/s 상한 |
| 시작 조건 | 기본 **끔** — 버튼이 곧 시작. 충족 여부는 기록만 |
| 성공 판정 | 면적비 ≥ 8 % · 연결성분 ≥ 80 % · \|중심−0.5\| ≤ 0.30, 연속 3 s |
