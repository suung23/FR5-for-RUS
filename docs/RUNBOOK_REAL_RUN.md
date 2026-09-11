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
cd teleop_gui && npm run build && cd ..        # GUI 를 고쳤다면
```

`환경 OK — rclpy · torch …` 가 안 뜨면 그 터미널에서는 아무것도 하지 않는다.

## B. 교정 (힘이 이상하면)

공중에서 아무것도 안 닿았는데 ‖F‖ 가 0.3 N 을 넘으면 먼저 고친다.

```bash
python3 policy_learning/scripts/diag_wrench_poses.py     # 자세 3~5 곳에서 Enter
```

판정이 인쇄된다. **"전기 영점 오차"** 면 GUI 작업 자세 영점으로 끝. **"질량·질량중심 오차"**
면 영점으로는 못 고치므로 중력 교정을 다시 한다:

```bash
./scripts/stop_all.sh && ./scripts/start_session.sh --calib
```

GUI Calibration 에서 **전자영점 → 다자세 중력 → 작업 자세 영점(아래 수직)**.

## C. 세션

```bash
cd ~/FR5-for-RUS
./scripts/start_session.sh
```

`--seg` 를 붙이지 않는다 (러너가 자기 U-Net 을 돌린다).

**GUI 확인** — 상단이 `NO TELEMETRY` 가 아니고, **Monitoring** 하단 오른쪽에 품질 두 눈금과
`Policy inference` 버튼이 보인다. 안 보이면 Electron 창을 새로 띄운다 (`npm run console`).

## D. 평가 — 한 명령

```bash
./scripts/start_policy_eval.sh
```

힘 탐색과 러너가 함께 뜬다. **Ctrl-C 하나로 둘 다 내린다.**

처음 한 번은 지령 없이 확인한다:

```bash
./scripts/start_policy_eval.sh --dry
```

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
